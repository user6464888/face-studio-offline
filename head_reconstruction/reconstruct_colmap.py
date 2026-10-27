"""Run and log the local masked COLMAP reconstruction."""
import argparse
from pathlib import Path
import subprocess
import time
import os
import json
import re
import numpy as np
import cv2
from colmap_data import read_text_model, intrinsics, canonical_frame

ROOT = Path(__file__).resolve().parent
EXE = Path(os.environ.get('COLMAP_EXE',ROOT / 'tools/colmap/bin/colmap.exe'))


def run(step, *args, exe=EXE):
    print(f'[{step}] started', flush=True)
    (ROOT/'logs').mkdir(exist_ok=True)
    log = ROOT / 'logs' / f'{step}.log'
    started = time.time()
    env = os.environ.copy()
    env['QT_PLUGIN_PATH'] = str(exe.parent.parent / 'plugins')
    env['PATH'] = str(exe.parent) + os.pathsep + env.get('PATH', '')
    with log.open('w', encoding='utf-8') as handle:
        result = subprocess.run([str(exe), *map(str, args)], stdout=handle, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        print(log.read_text(encoding='utf-8', errors='replace')[-5000:], flush=True)
        raise RuntimeError(f'{step}: {result.returncode}; see {log}')
    print(f'[{step}] finished in {time.time()-started:.1f}s', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('--dense', action='store_true')
    p.add_argument('--learned', action='store_true')
    a = p.parse_args()
    src = a.input.resolve()
    output = ROOT / 'outputs' / src.name / ('colmap_aliked' if a.learned else 'colmap')
    output.mkdir(parents=True, exist_ok=True)
    sparse = output / 'sparse'
    sparse.mkdir(exist_ok=True)
    db = output / 'database.db'
    if not (output / 'features.done').exists():
        extra = (['--FeatureExtraction.type', 'ALIKED_N16ROT', '--AlikedExtraction.n16rot_model_path',
                  ROOT / 'models/aliked-n16rot.onnx', '--AlikedExtraction.min_score', '.1',
                  '--AlikedExtraction.max_num_features', '4000'] if a.learned else [])
        run('01_features_aliked' if a.learned else '01_features', 'feature_extractor', '--database_path', db,
            '--image_path', src / 'images', '--ImageReader.mask_path', src / 'masks',
            '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'SIMPLE_RADIAL',
            '--ImageReader.camera_params', '1100,640,360,0',
            '--FeatureExtraction.use_gpu', '0', '--FeatureExtraction.num_threads', '2' if a.learned else '8',
            '--SiftExtraction.peak_threshold', '0.003', '--SiftExtraction.max_num_features', '8192', *extra)
        (output / 'features.done').touch()
    if not (output / 'matches.done').exists():
        extra = (['--FeatureMatching.type', 'ALIKED_LIGHTGLUE', '--AlikedMatching.lightglue_model_path',
                  ROOT / 'models/aliked-lightglue.onnx', '--SequentialMatching.overlap', '10',
                  '--SequentialMatching.quadratic_overlap', '1', '--TwoViewGeometry.compute_relative_pose', '1'] if a.learned else [])
        run('02_matches_aliked' if a.learned else '02_matches', 'sequential_matcher' if a.learned else 'exhaustive_matcher', '--database_path', db,
            '--FeatureMatching.use_gpu', '0', '--FeatureMatching.num_threads', '4' if a.learned else '8',
            '--FeatureMatching.guided_matching', '0' if a.learned else '1', *extra)
        (output / 'matches.done').touch()
    if not (sparse / '0/images.bin').exists():
        run('03_mapper', 'mapper', '--database_path', db, '--image_path', src / 'images',
            '--output_path', sparse, '--Mapper.num_threads', '8',
            '--Mapper.ba_refine_extra_params', '0', '--Mapper.init_min_num_inliers', '50',
            '--Mapper.init_min_tri_angle', '8', '--Mapper.min_model_size', '8')
    models = list(sparse.glob('*/images.bin'))
    if not models:
        raise RuntimeError('No registered reconstruction; inspect 03_mapper.log')
    # Largest binary image file generally has most observations; analyzer reports
    # the actual registered count before this model is used downstream.
    model = max(models, key=lambda p: p.stat().st_size).parent
    text = output / 'sparse_text'
    text.mkdir(exist_ok=True)
    run('04_analyzer', 'model_analyzer', '--path', model)
    run('05_text', 'model_converter', '--input_path', model, '--output_path', text, '--output_type', 'TXT')
    (output / 'selected_model.txt').write_text(str(model), encoding='utf-8')
    analysis=(ROOT / 'logs/04_analyzer.log').read_text(encoding='utf-8')
    error=re.search(r'Mean reprojection error: ([0-9.]+)px',analysis)
    (output/'reconstruction_stats.json').write_text(json.dumps({'mean_reprojection_error_pixels':float(error[1]) if error else None},indent=2),encoding='utf-8')
    print(analysis, flush=True)
    if a.dense:
        dense = output / 'dense'
        dense.mkdir(exist_ok=True)
        cameras, image_data, points, _, _ = read_text_model(text)
        basis, origin = canonical_frame(image_data, points)
        chosen = {}
        for name, im in image_data.items():
            direction = basis @ (im['center'] - origin)
            yaw = np.arctan2(direction[0], direction[2])
            bucket = int(np.rint(yaw / np.deg2rad(9))) % 40
            distance = abs(((yaw - bucket * np.deg2rad(9) + np.pi) % (2*np.pi)) - np.pi)
            if bucket not in chosen or distance < chosen[bucket][0]:
                chosen[bucket] = distance, name
        names = sorted(v[1] for v in chosen.values())
        (output / 'dense_images.txt').write_text('\n'.join(names), encoding='utf-8')
        print('Dense view count:', len(names), flush=True)
        run('06_undistort', 'image_undistorter', '--image_path', src / 'images', '--input_path', model,
            '--output_path', dense, '--output_type', 'COLMAP', '--max_image_size', '1280',
            '--image_list_path', output / 'dense_images.txt', '--roi_min_x', '.30', '--roi_max_x', '.72',
            '--roi_min_y', '.05', '--roi_max_y', '.93', '--num_patch_match_src_images', '10')
        dense_text = dense / 'sparse_text'
        dense_text.mkdir(exist_ok=True)
        run('06b_dense_text', 'model_converter', '--input_path', dense / 'sparse', '--output_path', dense_text, '--output_type', 'TXT')
        dense_cams, dense_imgs, _, _, _ = read_text_model(dense_text)
        mask_dir = dense / 'masks'
        mask_dir.mkdir(exist_ok=True)
        for name, im in dense_imgs.items():
            orig_k = intrinsics(cameras[image_data[name]['camera_id']])
            dest_cam = dense_cams[im['camera_id']]
            dest_k = intrinsics(dest_cam)
            yy, xx = np.indices((dest_cam['height'], dest_cam['width']), dtype=np.float32)
            mx = (xx - dest_k[0, 2]) / dest_k[0, 0] * orig_k[0, 0] + orig_k[0, 2]
            my = (yy - dest_k[1, 2]) / dest_k[1, 1] * orig_k[1, 1] + orig_k[1, 2]
            original_mask = cv2.imread(str(src / 'masks' / (name + '.png')), 0)
            cropped = cv2.remap(original_mask, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_NEAREST)
            cv2.imwrite(str(mask_dir / (name + '.png')), cropped)
        # The undistorter keeps the full sparse model when an image list is used.
        # Restrict every depth dependency to images actually exported to disk.
        config = []
        for name in names:
            source = image_data[name]
            ref_dir = source['center'] - origin
            ref_dir /= np.linalg.norm(ref_dir)
            ranking = []
            for other in names:
                if other == name:
                    continue
                direction = image_data[other]['center'] - origin
                direction /= np.linalg.norm(direction)
                angle = np.arccos(np.clip(ref_dir @ direction, -1, 1))
                ranking.append((angle, other))
            config += [name, ', '.join(other for _, other in sorted(ranking)[:10])]
        (dense / 'stereo/patch-match.cfg').write_text('\n'.join(config) + '\n', encoding='utf-8')
        (dense / 'stereo/fusion.cfg').write_text('\n'.join(names) + '\n', encoding='utf-8')
        run('07_depth', 'patch_match_stereo', '--workspace_path', dense,
            '--PatchMatchStereo.gpu_index', '0', '--PatchMatchStereo.max_image_size', '800',
            '--PatchMatchStereo.cache_size', '4', exe=Path(os.environ.get('COLMAP_DENSE_EXE',ROOT / 'tools/colmap-3.11.1/bin/colmap.exe')))
        run('08_fusion', 'stereo_fusion', '--workspace_path', dense, '--input_type', 'geometric',
            '--output_path', dense / 'fused.ply', '--StereoFusion.num_threads', '6',
            '--StereoFusion.cache_size', '4', '--StereoFusion.mask_path', mask_dir)
        run('09_mesh', 'poisson_mesher', '--input_path', dense / 'fused.ply',
            '--output_path', dense / 'mesh.ply', '--PoissonMeshing.depth', '10',
            '--PoissonMeshing.trim', '0', '--PoissonMeshing.point_weight', '4', '--PoissonMeshing.num_threads', '8')


if __name__ == '__main__':
    main()
