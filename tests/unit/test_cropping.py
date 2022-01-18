import unittest
import cv2
import numpy as np

from extractor.cropping import clip_to_image_region, \
    crop_module, build_merged_index

quadrilaterals = {
    ('e3e70682-c209-4cac-a29f-6fbed82c07cd', 
     'frame_000000', 
     'mask_000000'): {
        'quadrilateral': [
            [424, 279],
            [499, 280],
            [499, 327],
            [421, 323]
        ], 
        'center': (
            460.95042812077514, 
            302.4197085774373
        )
    },
    ('f728b4fa-4248-4e3a-8a5d-2f346baa9455', 
     'frame_000000', 
     'mask_000001'): {
        'quadrilateral': [
            [425, 326],
            [499, 326],
            [499, 377],
            [425, 372]
        ], 
        'center': (
            462.13331381447324, 
            350.2644805543356
        )
    },
    ('eb1167b3-67a9-4378-bc65-c1e582e2e662', 
     'frame_000000', 
     'mask_000002'): {
        'quadrilateral': [
            [164, 358],
            [233, 363],
            [233, 412],
            [164, 408]
        ], 
        'center': (
            198.48300673606857, 
            385.4114104919371
        )
    },
    ('f7c1bd87-4da5-4709-9471-3d60c8a70639', 
     'frame_000000', 
     'mask_000003'): {
        'quadrilateral': [
            [425, 234],
            [497, 231],
            [501, 279],
            [421, 278]
        ], 
        'center': (
            461.41970207121716, 
            255.7820630547903
        )
    },
    ('e443df78-9558-467f-9ba9-1faf7a024204', 
     'frame_000000', 
     'mask_000004'): {
        'quadrilateral': [
            [425, 94],
            [498, 90],
            [502, 136],
            [425, 142]
        ], 
        'center': (
            462.19730041647847, 
            115.55311355311355
        )
    }
}


class TestCropping(unittest.TestCase):
    
    def test_clip_to_image_region_no_clip(self):        
        quad = np.array([
            [[424, 279]], 
            [[499, 280]], 
            [[499, 327]], 
            [[421, 323]]
        ])
        image_width = 640
        image_height = 512
        quad_clipped_gt = quad
        
        quad_clipped = clip_to_image_region(
            np.copy(quad), image_width, image_height)
        
        self.assertTrue(
            np.allclose(
                quad_clipped, 
                quad_clipped_gt
            )
        )
        
        
    def test_clip_to_image_region_clip_max(self):        
        quad = np.array([
            [[424, 279]], 
            [[499, 280]], 
            [[499, 327]], 
            [[421, 323]]
        ])
        image_width = 300
        image_height = 200
        quad_clipped_gt = np.array([
            [[299, 199]], 
            [[299, 199]], 
            [[299, 199]], 
            [[299, 199]]
        ])
        
        quad_clipped = clip_to_image_region(
            np.copy(quad), image_width, image_height)
        
        self.assertTrue(
            np.allclose(
                quad_clipped, 
                quad_clipped_gt
            )
        )
        
        
    def test_clip_to_image_region_clip_min(self):        
        quad = np.array([
            [[ -1,  -1]], 
            [[100,  -1]], 
            [[100, 100]], 
            [[ -1, 100]]
        ])
        image_width = 200
        image_height = 200
        quad_clipped_gt = np.array([
            [[  0,   0]], 
            [[100,   0]], 
            [[100, 100]], 
            [[  0, 100]]
        ])
        
        quad_clipped = clip_to_image_region(
            np.copy(quad), image_width, image_height)
        
        self.assertTrue(
            np.allclose(
                quad_clipped, 
                quad_clipped_gt
            )
        )
        
        
    def test_build_merged_index_merged_none(self):
        merged_modules = None
        merged_index_gt = {
            'e3e70682-c209-4cac-a29f-6fbed82c07cd': 'e3e70682-c209-4cac-a29f-6fbed82c07cd',
            'e443df78-9558-467f-9ba9-1faf7a024204': 'e443df78-9558-467f-9ba9-1faf7a024204',
            'f7c1bd87-4da5-4709-9471-3d60c8a70639': 'f7c1bd87-4da5-4709-9471-3d60c8a70639',
            'eb1167b3-67a9-4378-bc65-c1e582e2e662': 'eb1167b3-67a9-4378-bc65-c1e582e2e662',
            'f728b4fa-4248-4e3a-8a5d-2f346baa9455': 'f728b4fa-4248-4e3a-8a5d-2f346baa9455'
        }
        merged_index = build_merged_index(merged_modules, quadrilaterals)
        self.assertEqual(merged_index, merged_index_gt)
        
    
    def test_build_merged_index_merged_empty(self):
        merged_modules = []
        merged_index_gt = {
            'e3e70682-c209-4cac-a29f-6fbed82c07cd': 'e3e70682-c209-4cac-a29f-6fbed82c07cd',
            'e443df78-9558-467f-9ba9-1faf7a024204': 'e443df78-9558-467f-9ba9-1faf7a024204',
            'f7c1bd87-4da5-4709-9471-3d60c8a70639': 'f7c1bd87-4da5-4709-9471-3d60c8a70639',
            'eb1167b3-67a9-4378-bc65-c1e582e2e662': 'eb1167b3-67a9-4378-bc65-c1e582e2e662',
            'f728b4fa-4248-4e3a-8a5d-2f346baa9455': 'f728b4fa-4248-4e3a-8a5d-2f346baa9455'
        }
        merged_index = build_merged_index(merged_modules, quadrilaterals)
        self.assertEqual(merged_index, merged_index_gt)
        
        
    def test_build_merged_index_pair_merged(self):
        merged_modules = [[
            'f728b4fa-4248-4e3a-8a5d-2f346baa9455', 
            'f7c1bd87-4da5-4709-9471-3d60c8a70639'
        ]]
        merged_index_gt = {
            'e3e70682-c209-4cac-a29f-6fbed82c07cd': 'e3e70682-c209-4cac-a29f-6fbed82c07cd',
            'e443df78-9558-467f-9ba9-1faf7a024204': 'e443df78-9558-467f-9ba9-1faf7a024204',
            'f7c1bd87-4da5-4709-9471-3d60c8a70639': 'f728b4fa-4248-4e3a-8a5d-2f346baa9455',
            'eb1167b3-67a9-4378-bc65-c1e582e2e662': 'eb1167b3-67a9-4378-bc65-c1e582e2e662',
            'f728b4fa-4248-4e3a-8a5d-2f346baa9455': 'f728b4fa-4248-4e3a-8a5d-2f346baa9455'
        }
        merged_index = build_merged_index(merged_modules, quadrilaterals)
        self.assertEqual(merged_index, merged_index_gt)
    
    
    def test_build_merged_index_triplet_merged(self):
        merged_modules = [[
            'f728b4fa-4248-4e3a-8a5d-2f346baa9455', 
            'f7c1bd87-4da5-4709-9471-3d60c8a70639',
            'e3e70682-c209-4cac-a29f-6fbed82c07cd'
        ]]
        merged_index_gt = {
            'e3e70682-c209-4cac-a29f-6fbed82c07cd': 'f728b4fa-4248-4e3a-8a5d-2f346baa9455',
            'e443df78-9558-467f-9ba9-1faf7a024204': 'e443df78-9558-467f-9ba9-1faf7a024204',
            'f7c1bd87-4da5-4709-9471-3d60c8a70639': 'f728b4fa-4248-4e3a-8a5d-2f346baa9455',
            'eb1167b3-67a9-4378-bc65-c1e582e2e662': 'eb1167b3-67a9-4378-bc65-c1e582e2e662',
            'f728b4fa-4248-4e3a-8a5d-2f346baa9455': 'f728b4fa-4248-4e3a-8a5d-2f346baa9455'
        }
        merged_index = build_merged_index(merged_modules, quadrilaterals)
        self.assertEqual(merged_index, merged_index_gt)
        
        
    def test_build_merged_index_two_pairs_merged(self):
        merged_modules = [
            ['f728b4fa-4248-4e3a-8a5d-2f346baa9455', 
             'f7c1bd87-4da5-4709-9471-3d60c8a70639'],
            ['e3e70682-c209-4cac-a29f-6fbed82c07cd', 
             'e443df78-9558-467f-9ba9-1faf7a024204']
        ]
        merged_index_gt = {
            'e3e70682-c209-4cac-a29f-6fbed82c07cd': 'e3e70682-c209-4cac-a29f-6fbed82c07cd',
            'e443df78-9558-467f-9ba9-1faf7a024204': 'e3e70682-c209-4cac-a29f-6fbed82c07cd',
            'f7c1bd87-4da5-4709-9471-3d60c8a70639': 'f728b4fa-4248-4e3a-8a5d-2f346baa9455',
            'eb1167b3-67a9-4378-bc65-c1e582e2e662': 'eb1167b3-67a9-4378-bc65-c1e582e2e662',
            'f728b4fa-4248-4e3a-8a5d-2f346baa9455': 'f728b4fa-4248-4e3a-8a5d-2f346baa9455'
        }
        merged_index = build_merged_index(merged_modules, quadrilaterals)
        self.assertEqual(merged_index, merged_index_gt)
        
        
    def test_build_merged_index_all_merged(self):
        merged_modules = [[             
            'f7c1bd87-4da5-4709-9471-3d60c8a70639',
            'f728b4fa-4248-4e3a-8a5d-2f346baa9455',
            'e3e70682-c209-4cac-a29f-6fbed82c07cd',
            'e443df78-9558-467f-9ba9-1faf7a024204',
            'eb1167b3-67a9-4378-bc65-c1e582e2e662',    
        ]]
        merged_index_gt = {
            'e3e70682-c209-4cac-a29f-6fbed82c07cd': 'f7c1bd87-4da5-4709-9471-3d60c8a70639',
            'e443df78-9558-467f-9ba9-1faf7a024204': 'f7c1bd87-4da5-4709-9471-3d60c8a70639',
            'f7c1bd87-4da5-4709-9471-3d60c8a70639': 'f7c1bd87-4da5-4709-9471-3d60c8a70639',
            'eb1167b3-67a9-4378-bc65-c1e582e2e662': 'f7c1bd87-4da5-4709-9471-3d60c8a70639',
            'f728b4fa-4248-4e3a-8a5d-2f346baa9455': 'f7c1bd87-4da5-4709-9471-3d60c8a70639'
        }
        merged_index = build_merged_index(merged_modules, quadrilaterals)
        self.assertEqual(merged_index, merged_index_gt)
        
        
    def test_crop_modules_real_data(self):
        frame_file = "tests/unit/data/frame_000000.tiff"
        frame = cv2.imread(frame_file, cv2.IMREAD_ANYDEPTH)
        
        quad = np.array([
            [[424, 279]],
            [[499, 280]],
            [[499, 327]],
            [[421, 323]]
        ])
        
        patch_file = "tests/unit/data/frame_000000_mask_000000.tiff"
        patch_gt = cv2.imread(patch_file, cv2.IMREAD_ANYDEPTH)
        
        patch, _ = crop_module(
            frame, 
            quad, 
            crop_width=None, 
            crop_aspect=None,
            rotate_mode="portrait"
        )
        
        self.assertTrue(np.allclose(patch, patch_gt))
        
        
    def test_crop_modules_crop_full_frame(self):
        frame_file = "tests/unit/data/frame_000000.tiff"
        frame = cv2.imread(frame_file, cv2.IMREAD_ANYDEPTH)
        
        quad = np.array([
            [[0, 0]],
            [[640, 0]],
            [[640, 512]],
            [[0, 512]]
        ])
        
        patch, homography = crop_module(
            frame, 
            quad, 
            crop_width=None, 
            crop_aspect=None,
            rotate_mode="landscape"
        )
        
        self.assertTrue(np.allclose(patch, frame[0:-1, 0:-1]))
        self.assertTrue(np.allclose(homography, np.eye(3)))
        
        
    def test_crop_modules_portrait_vs_landscape(self):
        frame_file = "tests/unit/data/frame_000000.tiff"
        frame = cv2.imread(frame_file, cv2.IMREAD_ANYDEPTH)
        
        quad = np.array([
            [[424, 279]],
            [[499, 280]],
            [[499, 327]],
            [[421, 323]]
        ])
        
        patch, _ = crop_module(
            frame, 
            quad, 
            crop_width=None, 
            crop_aspect=None,
            rotate_mode="portrait"
        )
        
        self.assertEqual(patch.shape, (78, 47))
        
        patch, _ = crop_module(
            frame, 
            quad, 
            crop_width=None, 
            crop_aspect=None,
            rotate_mode="landscape"
        )
        
        self.assertEqual(patch.shape, (47, 78))
        
        patch, _ = crop_module(
            frame, 
            quad, 
            crop_width=None, 
            crop_aspect=None,
            rotate_mode=None
        )
        
        self.assertEqual(patch.shape, (47, 78))  # ?
        
        
    def test_crop_modules_crop_width_and_aspect(self):
        frame_file = "tests/unit/data/frame_000000.tiff"
        frame = cv2.imread(frame_file, cv2.IMREAD_ANYDEPTH)
        
        quad = np.array([
            [[424, 279]],
            [[499, 280]],
            [[499, 327]],
            [[421, 323]]
        ])
        
        patch, _ = crop_module(
            frame, 
            quad, 
            crop_width=50, 
            crop_aspect=0.625,  # 1/1.6
            rotate_mode="portrait"
        )
        
        self.assertEqual(patch.shape, (50, 31))
        
        patch, _ = crop_module(
            frame, 
            quad, 
            crop_width=50, 
            crop_aspect=1,
            rotate_mode="portrait"
        )
        
        self.assertEqual(patch.shape, (50, 50))
        
        patch, _ = crop_module(
            frame, 
            quad, 
            crop_width=50, 
            crop_aspect=0.625,  # 1/1.6
            rotate_mode="landscape"
        )
        
        self.assertEqual(patch.shape, (31, 50))
        
        patch, _ = crop_module(
            frame, 
            quad, 
            crop_width=300, 
            crop_aspect=0.625,  # 1/1.6
            rotate_mode="portrait"
        )
        
        self.assertEqual(patch.shape, (300, 187))