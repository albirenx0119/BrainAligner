"""Synthetic registration and real file I/O checks; no user images are accessed."""
import copy
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from image_io import ImageStore, inspect_source, load_sources, make_proxy, render
from project import Project, export_names, export_project
from registration import (Cancelled, Correlation, Proxy, RegistrationFailure, as_matrix,
                          compose, half, identity, inverse, register_rigid, register_series, rigid)


def texture(height=161,width=193):
    rng = np.random.default_rng(20261021)
    image = cv2.GaussianBlur(rng.random((height,width)).astype(np.float32),(0,0),1.2)
    y,x = np.mgrid[:height,:width]
    image = image*.65+.35*np.exp(-((x-width*.38)**2+(y-height*.45)**2)/250)
    return np.clip(image,0,1).astype(np.float32)


def pixel_transform(matrix):
    m = as_matrix(matrix).copy()
    m[:2,2] += m[:2,:2] @ np.array([.5,.5])-.5
    return m[:2]


class RegistrationTests(unittest.TestCase):
    def test_known_rotation_translation_odd_pyramid(self):
        fixed = texture()
        h,w = fixed.shape
        displacement = rigid(8.,-6.,7.,w/2,h/2)
        moving = cv2.warpAffine(fixed,pixel_transform(displacement),(w,h))
        support = np.ones_like(fixed,dtype=bool)
        matrix,score = register_rigid(Proxy(fixed,support),Proxy(moving,support))
        expected = inverse(displacement)
        self.assertGreater(score,.95)
        self.assertLess(np.max(np.abs(matrix[:4]-expected[:4])),.006)
        self.assertLess(np.max(np.abs(matrix[4:]-expected[4:])),.7)

    def test_padding_masks_and_partial_overlap(self):
        f = texture(45,61)
        mask = np.zeros_like(f,dtype=bool); mask[4:-4,7:-7] = True
        proxy = Proxy(f,mask)
        score = Correlation(proxy,proxy)(rigid(2.,1.,0.,0,0))
        self.assertTrue(np.isfinite(score))
        self.assertAlmostEqual(Correlation(proxy,proxy)(identity()),1.,places=7)
        self.assertEqual(Correlation(proxy,proxy)(rigid(1000,0,0,0,0)),-1)
        down = half(proxy)
        self.assertEqual(down.data.shape,(23,31))

    def test_textureless_and_cancel(self):
        p = Proxy(np.zeros((45,61),np.float32),np.ones((45,61),bool))
        with self.assertRaises(RegistrationFailure): register_rigid(p,p)
        event = threading.Event(); event.set()
        with self.assertRaises(Cancelled): register_rigid(p,p,event)

    def test_pair_order_scaled_accumulation_and_failure_descendants(self):
        values = [Proxy(np.ones((32,32),np.float32)*i,np.ones((32,32),bool)) for i in range(5)]
        calls = []
        def pair(f,m,cancel):
            child,parent = int(m.data[0,0]),int(f.data[0,0])
            calls.append((child,parent))
            if child==3: raise RegistrationFailure('synthetic failure',.1)
            return rigid(1,0,0,0,0),.9
        with patch('registration.register_rigid',side_effect=pair):
            matrices,records = register_series(5,2,lambda i:(values[i],.5))
        self.assertEqual(calls,[(1,2),(0,1),(3,2),(4,3)])
        self.assertEqual(matrices[0][4],4.)
        self.assertEqual(matrices[3][4],0.)
        self.assertEqual(matrices[4][4],2.)
        self.assertIsNotNone(records[4]['warning'])


class FileWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.test-files-',dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)
        self.rgb = np.zeros((51,67,3),np.uint8)
        y,x = np.mgrid[:51,:67]
        self.rgb[:,:,0] = (x*3+y)%256
        self.rgb[:,:,1] = (y*4+x)%256
        self.rgb[:,:,2] = (x+y*2)%256
        for name in ('image10.png','image2.png','image1.png'):
            Image.fromarray(self.rgb).save(self.root/name)
        sources,store = load_sources(self.root.glob('*.png'))
        self.project = Project(sources,store)

    def tearDown(self):
        self.temp.cleanup()

    def test_natural_order_and_single_tiff_16bit_mapping(self):
        self.assertEqual([s.name for s in self.project.sources],['image1.png','image2.png','image10.png'])
        data = np.tile(np.array([0,257,32768,65535],np.uint16),(18,1))
        path = self.root/'単一ページ.tif'
        tifffile.imwrite(path,data)
        source = inspect_source(path)
        pixels = ImageStore([source]).get(0)
        np.testing.assert_array_equal(pixels[0,:,0],[0,1,128,255])

    def test_multipage_ome_and_float_tiff_rejected(self):
        for name,kwargs,data in [
            ('multi.tif',{'photometric':'minisblack'},np.zeros((2,20,24),np.uint8)),
            ('ome.ome.tif',{'ome':True},np.zeros((20,24),np.uint8)),
            ('float.tif',{},np.zeros((20,24),np.float32))]:
            path = self.root/name
            tifffile.imwrite(path,data,**kwargs)
            with self.subTest(name=name),self.assertRaises(ValueError): inspect_source(path)
        with self.assertRaises(ValueError): load_sources([self.root/'image1.png',self.root/'multi.tif'])

    def test_identity_render_and_unequal_padding_proxy_mask(self):
        p = self.project
        result = render(p.store,0,p.canvas,identity(),p.viewport,'white')
        np.testing.assert_array_equal(result,self.rgb)
        small = self.root/'small.png'
        Image.fromarray(self.rgb[:42,:52]).save(small)
        sources,store = load_sources([self.root/'image1.png',small])
        p = Project(sources,store)
        rgb = render(store,1,p.canvas,identity(),p.viewport,'black')
        np.testing.assert_array_equal(rgb[4:46,7:59],self.rgb[:42,:52])
        proxy,scale = make_proxy(store,1,p.canvas,512,'Red','black')
        self.assertFalse(proxy.mask[4,7])
        self.assertTrue(proxy.mask[5,8])
        self.assertEqual(scale,1.)

    def test_manual_range_crop_undo_and_strict_json(self):
        p = self.project
        correction = rigid(3,2,4.,33.5,25.5)
        p.apply_manual(correction,2,1)
        np.testing.assert_allclose(p.final(0),identity())
        np.testing.assert_allclose(p.final(1),correction)
        p.crop((3,4,40,35))
        p.apply_manual(rigid(1,0,0,0,0),0,0)
        p.undo_crop()
        self.assertEqual(p.viewport,(0,0,67,51))
        self.assertEqual(p.manual[0][4],1.)
        saved = p.to_dict()
        q = Project(p.sources,p.store)
        q.load_dict(json.loads(json.dumps(saved)))
        self.assertEqual(q.to_dict(),saved)
        invalid = copy.deepcopy(saved)
        invalid['slices'][0]['automaticTransform'][0] = 1.2
        with self.assertRaises(ValueError): q.load_dict(invalid)
        self.assertEqual(q.to_dict(),saved)
        self.assertTrue(q.undo())

    def test_png_folder_zip_and_jpeg_export(self):
        p = self.project
        p.apply_manual(rigid(2,-1,2.,33.5,25.5),0,2)
        p.crop((2,3,44,36))
        output = export_project(p,self.root/'out')
        self.assertFalse((output/'INCOMPLETE.txt').exists())
        expected = render(p.store,0,p.canvas,p.final(0),p.viewport,'white')
        with Image.open(output/'image1.png') as image:
            np.testing.assert_array_equal(np.asarray(image),expected)
        payload = json.loads((output/'transforms.json').read_text(encoding='utf-8'))
        self.assertEqual(payload['export']['bitDepth'],8)
        archive = export_project(p,self.root/'out',as_zip=True)
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(len(z.namelist()),4)
            with Image.open(io.BytesIO(z.read('image1.png'))) as image:
                np.testing.assert_array_equal(np.asarray(image),expected)
        jpeg = export_project(p,self.root/'out','jpg')
        with Image.open(jpeg/'image1.jpg') as image: self.assertEqual(image.size,(44,36))

    def test_cancelled_zip_removed_and_export_collision_names(self):
        event = threading.Event(); event.set()
        dest = self.root/'cancelled'
        with self.assertRaises(Cancelled): export_project(self.project,dest,as_zip=True,cancel=event)
        self.assertEqual(list(dest.iterdir()),[])
        self.assertEqual(export_names([self.project.sources[0]]*2,'png'),['image1.png','image1__002.png'])

    def test_real_proxy_scale_registration_to_original_resolution(self):
        fixed = (texture(421,573)*255).astype(np.uint8)
        expected_motion = rigid(11,-7,5.,573/2,421/2)
        moving = cv2.warpAffine(fixed,pixel_transform(expected_motion),(573,421))
        fixed_path,moving_path = self.root/'large1.png',self.root/'large2.png'
        Image.fromarray(fixed).save(fixed_path)
        Image.fromarray(moving).save(moving_path)
        sources,store = load_sources([fixed_path,moving_path])
        project = Project(sources,store)
        matrices,records = register_series(2,0,lambda i:make_proxy(store,i,project.canvas,512,'Gray','black'))
        target = inverse(expected_motion)
        self.assertLess(np.max(np.abs(matrices[1][4:]-target[4:])),.9)
        self.assertGreater(records[1]['score'],.9)

    def test_white_is_zero_and_lzw_input_decode(self):
        values = np.arange(24*20,dtype=np.uint16).reshape(20,24)*100
        white_zero = self.root/'white-zero.tif'
        tifffile.imwrite(white_zero,values,photometric='miniswhite')
        source = inspect_source(white_zero)
        result = ImageStore([source]).get(0)
        expected = np.floor((65535-values).astype(np.float64)/257+.5).astype(np.uint8)
        np.testing.assert_array_equal(result[:,:,0],expected)
        lzw = self.root/'lzw.tif'
        Image.fromarray(values).save(lzw,compression='tiff_lzw')
        decoded = ImageStore([inspect_source(lzw)]).get(0)
        np.testing.assert_array_equal(decoded[:,:,0],np.floor(values/257.+.5).astype(np.uint8))


class GuiTests(unittest.TestCase):
    def test_worker_import_manual_crop_and_cancel_controls(self):
        from app import Application
        with tempfile.TemporaryDirectory(prefix='.test-gui-',dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory)/'slice.png'
            Image.fromarray((texture(81,95)*255).astype(np.uint8)).save(path)
            app = Application()
            errors = []
            app.report_callback_exception = lambda *args:errors.append(str(args[1]))
            try:
                app.update()
                app._load([path])
                deadline = time.monotonic()+15
                while app.busy and time.monotonic()<deadline:
                    app.update(); time.sleep(.02)
                self.assertFalse(app.busy)
                self.assertIsNotNone(app.project)
                app.start_recording()
                app.correction = [4.,-2.,1.]
                app.finish_recording()
                np.testing.assert_allclose(app.project.manual[0],identity())
                app.apply_recorded()
                self.assertNotEqual(app.project.manual[0][4],0)
                app._draw()
                app.undo()
                np.testing.assert_allclose(app.project.manual[0],identity())
                self.assertEqual(errors,[])
            finally:
                app.destroy()


if __name__=='__main__':
    unittest.main()
