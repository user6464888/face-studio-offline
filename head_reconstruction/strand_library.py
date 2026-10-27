"""Retarget authored HAIR curves to a measured head, without a cap silhouette.

The imported curves are authored assets, not measurements of the subject.
All sources are local; see assets/hair_sources/NOTICE.md for attribution/terms.
"""
from pathlib import Path
import struct
import hashlib
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d

SOURCE = Path(__file__).resolve().parent / 'assets' / 'hair_sources'
PRESETS = {'straight': ('中分直发', 'wStraight.hair', 25),
           'wavy': ('中分微波浪', 'wWavy.hair', 40),
           'curly': ('后梳蓬松卷发', 'wCurly.hair', 52)}


def read_curves(path, count=8000, samples=40):
    data = path.read_bytes()
    if data[:4] != b'HAIR':
        raise ValueError(f'发丝素材损坏：{path.name}')
    n, total, flags, default = struct.unpack_from('<4I', data, 4)
    offset = 128
    sizes = np.full(n, default + 1, dtype=int)
    if flags & 1:
        sizes = np.frombuffer(data, '<u2', n, offset).astype(int) + 1
        offset += n * 2
    if not flags & 2 or sizes.sum() != total:
        raise ValueError(f'发丝点数不一致：{path.name}')
    points = np.frombuffer(data, '<f4', total * 3, offset).reshape(-1, 3)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    # A fixed random subset avoids the spatial stripes of every-nth sampling.
    ids = np.sort(np.random.default_rng(803).choice(n, min(count, n), replace=False))
    curves = np.empty((len(ids), samples, 3))
    for j, i in enumerate(ids):
        p = points[starts[i]:starts[i] + sizes[i]]
        length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        for axis in range(3):
            curves[j, :, axis] = np.interp(np.linspace(0, length[-1], samples), length, p[:, axis])
    return curves, n


def fitted_strands(atlas, key):
    label, filename, samples = PRESETS[key]
    path = SOURCE / filename
    if not path.exists():
        raise FileNotFoundError(f'缺少离线发丝素材 {path}，请恢复项目 assets/hair_sources 文件夹。')
    raw, original_count = read_curves(path, samples=samples)
    # The supplied woman reference has Z up, X forward and Y across the head.
    # Source scalp ellipsoid: centre (-10,0,15), radii (34,28,45).
    # Metric target scale and the final attachment offsets come from RGB-D.
    canonical = np.stack([-raw[:, :, 1] / 28, (raw[:, :, 2] - 15) / 45,
                          (raw[:, :, 0] + 10) / 34], axis=-1)
    curves = atlas.centre + canonical * atlas.radii
    root_dir = canonical[:, 0] / np.linalg.norm(canonical[:, 0], axis=1, keepdims=True)
    # The reference woman's frontal hairline is substantially higher than this
    # scan. Expand the frontal polar angle to the measured hairline, rather than
    # leaving a broad, invented forehead between the face and the groom.
    theta = np.arctan2(root_dir[:, 0], root_dir[:, 2])
    phi = np.arccos(np.clip(root_dir[:, 1], -1, 1))
    phi *= 1 + .52 * np.maximum(np.cos(theta), 0) ** 2
    root_dir = np.column_stack([np.sin(phi) * np.sin(theta), np.cos(phi),
                                np.sin(phi) * np.cos(theta)])
    radius, rgb = atlas.sample(root_dir)
    fitted_root = atlas.centre + root_dir * atlas.radii * (radius + .015)[:, None]
    arc = np.cumsum(np.linalg.norm(np.diff(curves, axis=1), axis=2), axis=1)
    arc = np.column_stack([np.zeros(len(curves)), arc])
    curves += (fitted_root - curves[:, 0])[:, None] * np.exp(-arc / .08)[:, :, None]
    # Root displacement alone can pull a guide back through the forehead on
    # its next samples. Project the whole proximal curve outside the measured
    # envelope, retaining the authored free-hanging ends below the head.
    q = (curves - atlas.centre) / atlas.radii
    r = np.linalg.norm(q, axis=2)
    direction = q / np.maximum(r[:, :, None], 1e-9)
    _, nearest = atlas.tree.query(direction.reshape(-1, 3), k=8, workers=4)
    envelope = np.median(atlas.radius[nearest], axis=1).reshape(r.shape)
    envelope = gaussian_filter1d(envelope, 1.2, axis=1) + .025
    near_head = (curves[:, :, 1] > atlas.c['sideLine'] - .020) & (r < 1.5)
    corrected = np.where(near_head, np.maximum(r, envelope), r)
    curves = atlas.centre + direction * corrected[:, :, None] * atlas.radii
    # Round, tapered micro-bundles are real 3D triangles and export to GLB.
    # No billboard, alpha texture, skirt sheet, or broad closed hair cap.
    tangent = np.gradient(curves, axis=1)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=2, keepdims=True), 1e-12)
    reference = np.zeros_like(tangent); reference[:, :, 2] = 1
    parallel = abs(tangent[:, :, 2]) > .9
    reference[parallel] = [1, 0, 0]
    axis_a = np.cross(tangent, reference)
    axis_a /= np.maximum(np.linalg.norm(axis_a, axis=2, keepdims=True), 1e-12)
    axis_b = np.cross(tangent, axis_a)
    t = np.linspace(0, 1, samples)
    rng = np.random.default_rng(24)
    width = rng.uniform(.00020, .00032, len(curves))[:, None]
    radius = width * (.85 + .15 * np.sin(np.pi * t)) * np.maximum(.035, (1 - t) ** .32)
    phase = np.arange(3) * (2 * np.pi / 3)
    vertices = curves[:, :, None] + radius[:, :, None, None] * (
        axis_a[:, :, None] * np.cos(phase)[None, None, :, None] +
        axis_b[:, :, None] * np.sin(phase)[None, None, :, None])
    ring = np.arange(len(curves) * samples * 3).reshape(len(curves), samples, 3)
    a = ring[:, :-1]; b = np.roll(a, -1, axis=2); d = ring[:, 1:]; e = np.roll(d, -1, axis=2)
    faces = np.concatenate([np.stack([a, b, d], axis=-1).reshape(-1, 3),
                            np.stack([b, e, d], axis=-1).reshape(-1, 3)])
    # Captured root RGB sets the colour/contrast; fibre-level variation belongs
    # to rendering the imported asset, and is explicitly not captured texture.
    variation = rng.uniform(.78, 1.26, len(curves))[:, None] * (.78 + .22 * np.sin(t * np.pi * .8))[None]
    colours = np.clip(rgb[:, None] * variation[:, :, None], 0, 255).astype(np.uint8)
    colours = np.repeat(colours[:, :, None], 3, axis=2).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices.reshape(-1, 3), faces, process=False)
    mesh.visual.vertex_colors = np.column_stack([colours, np.full(len(colours), 255, np.uint8)])
    return mesh, {'name': label, 'source_file': filename,
                  'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'author': 'Cem Yuksel', 'source_url': 'https://www.cemyuksel.com/research/hairmodels/',
                  'original_strands': original_count, 'display_strands': len(curves),
                  'samples_per_strand': samples, 'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
                  'proximal_strands_projected_outside_depth_envelope': True,
                  'dimensions_mm': (mesh.extents * 1000).tolist(),
                  'generated_hair_is_measured': False, 'representation': 'authored_curves_as_tapered_3D_microbundles'}
