"""Public-data geometry and export regression checks; no capture fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import open3d as o3d

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'head_reconstruction'))
from salon import heads as head_module,geometry,demo
from salon.catalog import STYLES,FACE_SHAPES
from salon.face_advice import recommendations
from salon.server import Handler


class ReleaseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='face-studio-test-')
        cls.root=Path(cls.temp.name)
        cls.patches=[patch.object(module,'ROOT',cls.root) for module in [head_module,geometry,demo]]
        for p in cls.patches:p.start()
        cls.heads=demo.prepare()

    @classmethod
    def tearDownClass(cls):
        geometry.SCENES.clear();geometry.EAR_SCENES.clear()
        for p in reversed(cls.patches):p.stop()
        cls.temp.cleanup()

    def test_public_heads_and_all_choices(self):
        self.assertEqual({h['id'] for h in self.heads},set(FACE_SHAPES))
        self.assertEqual(len({h['fingerprint'] for h in self.heads}),7)
        self.assertEqual([s['number'] for s in STYLES],list(range(1,17))+[19,22,27])
        for h in self.heads:
            self.assertTrue(h['is_standard'])
            self.assertEqual(h['all_style_ids'],[s['id'] for s in STYLES])
            self.assertTrue(np.isfinite(h['measurements']['ratios']).all())
            self.assertLess(h['radial_fallback_fraction'],.05)
            self.assertEqual(len(recommendations(h['id'])),3)

    def test_sampled_attachment_all_133_pairs(self):
        for h in self.heads:
            for style in STYLES:
                q,_=geometry.long_curves(style)
                if len(q):q=q[np.linspace(0,len(q)-1,min(len(q),48)).astype(int)]
                long=geometry.fitted_long_curves(q,h,style)
                medium,_=geometry.fitted_medium_curves(style,h,count=48)
                for curves in [long,medium]:
                    self.assertTrue(np.isfinite(curves).all())
                    if not len(curves):continue
                    scene=geometry.SCENES[h['fingerprint']]
                    dist=scene.compute_distance(o3d.core.Tensor(curves[:,0].copy())).numpy()
                    self.assertLess(float(np.quantile(dist,.99)),.0007,(h['id'],style['id']))

    def test_representative_full_exports(self):
        h=self.heads[0]
        for number in [9,10,8,12]:
            style=next(s for s in STYLES if s['number']==number)
            manifest=geometry.style_assets(h,style)
            expected={'short'} if number==9 else {'short','medium'} if number==10 else {'short','medium','long'}
            self.assertEqual(set(manifest['layers']),expected)
            for layer in manifest['layers'].values():
                path=self.root/layer['file']
                self.assertGreater(path.stat().st_size,0)
                mesh=geometry.trimesh.load(path,process=False)
                self.assertTrue(np.isfinite(mesh.vertices).all())
                self.assertEqual(len(mesh.faces),layer['triangles'])
            self.assertEqual(geometry.style_assets(h,style)['identity'],manifest['identity'])

    def test_server_does_not_serve_local_inputs(self):
        handler=object.__new__(Handler)
        for path in ['/runs/test/config.json','/salon/heads.json','/salon/assets/unknown.png','/salon/../server.py','/salon/cache/../../recording.bag','/viewer/','/salon/']:
            with patch('salon.server.ROOT',self.root):self.assertFalse(handler.allowed_static(path),path)
        self.assertTrue(handler.allowed_static('/salon/index.html'))

    def test_release_selection_omits_runtime_assets(self):
        spec=importlib.util.spec_from_file_location('packer',PROJECT/'scripts/build_release.py');packer=importlib.util.module_from_spec(spec);spec.loader.exec_module(packer)
        files=packer.selected_files();packer.audit(files)
        self.assertFalse(any(p.suffix in packer.FORBIDDEN for p in files))
        self.assertTrue(any(p.name=='THREE-LICENSE.txt' for p in files))


if __name__=='__main__':unittest.main()
