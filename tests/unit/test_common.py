import unittest
import numpy as np
import cv2

from extractor.common import contour_and_convex_hull, \
    compute_mask_center, sort_cw

convex_hull_gt = np.array([
    [[499, 324]], 
    [[498, 325]],
    [[488, 326]],
    [[476, 326]],
    [[442, 324]],
    [[423, 322]],
    [[422, 321]],
    [[421, 319]],
    [[421, 316]],
    [[424, 281]],
    [[425, 280]],
    [[427, 279]],
    [[436, 279]],
    [[491, 280]],
    [[495, 281]],
    [[496, 282]],
    [[497, 287]],
    [[499, 300]]
], dtype=np.int32)

contour_gt = np.array([
    [[427, 279]],
    [[426, 280]],
    [[425, 280]],
    [[424, 281]],
    [[424, 296]],
    [[423, 297]],
    [[423, 304]],
    [[422, 305]],
    [[422, 315]],
    [[421, 316]],
    [[421, 319]],
    [[422, 320]],
    [[422, 321]],
    [[423, 322]],
    [[439, 322]],
    [[440, 323]],
    [[441, 323]],
    [[442, 324]],
    [[459, 324]],
    [[460, 325]],
    [[475, 325]],
    [[476, 326]],
    [[488, 326]],
    [[489, 325]],
    [[498, 325]],
    [[499, 324]],
    [[499, 300]],
    [[498, 299]],
    [[498, 296]],
    [[497, 295]],
    [[497, 287]],
    [[496, 286]],
    [[496, 282]],
    [[495, 281]],
    [[492, 281]],
    [[491, 280]],
    [[437, 280]],
    [[436, 279]]
], dtype=np.int32)


class TestCommon(unittest.TestCase):
    
    def test_sort_cw_all_zeros(self):
        pts = np.array([
            [[0., 0.]],
            [[0., 0.]],
            [[0., 0.]],
            [[0., 0.]]
        ])
        self.assertTrue(np.allclose(sort_cw(pts), pts))
    
    
    def test_sort_cw_initial_order_random(self):
        pts = np.array([
            [[1., 0.]],
            [[1., 1.]],
            [[0., 0.]],
            [[0., 1.]]
        ])
        pts_sorted = np.array([
            [0., 0.],
            [1., 0.],
            [1., 1.],
            [0., 1.]
        ])
        self.assertTrue(np.allclose(sort_cw(pts), pts_sorted))
    

    def test_sort_cw_initial_order_ccw(self):
        pts = np.array([
            [[0., 1.]],
            [[1., 1.]],
            [[1., 0.]],
            [[0., 0.]]
        ])
        pts_sorted = np.array([
            [0., 0.],
            [1., 0.],
            [1., 1.],
            [0., 1.]
        ])
        self.assertTrue(np.allclose(sort_cw(pts), pts_sorted))
    

    def test_sort_cw_real_data(self):
        pts = np.array([
            [[550.94701019,  63.73629146]],
            [[588.31014706,  58.91243193]],
            [[203.31004274, 989.78343506]],
            [[277.23160467, 699.42132832]]
        ])
        pts_sorted = np.array([
            [277.23160467, 699.42132832],
            [588.31014706,  58.91243193],
            [550.94701019,  63.73629146],
            [203.31004274, 989.78343506]
        ])
        self.assertTrue(np.allclose(sort_cw(pts), pts_sorted))
    

    def test_sort_cw_all_same(self):
        pts = np.array([
            [[100.,  100.]],
            [[100.,  100.]],
            [[100.,  100.]],
            [[100.,  100.]]
        ])
        pts_sorted = np.array([
            [100.,  100.],
            [100.,  100.],
            [100.,  100.],
            [100.,  100.]
        ])
        self.assertTrue(np.allclose(sort_cw(pts), pts_sorted))
        

    def test_contour_and_convex_hull(self):
        mask_file = "tests/unit/data/mask_000000.png"
        mask = cv2.imread(mask_file, cv2.IMREAD_ANYDEPTH)
        convex_hull, contour = contour_and_convex_hull(mask)
        
        self.assertEqual(convex_hull.dtype, convex_hull_gt.dtype)
        self.assertEqual(contour.dtype, contour_gt.dtype)
        self.assertTrue(np.all(convex_hull == convex_hull_gt))
        self.assertTrue(np.all(contour == contour_gt))
    

    def test_compute_mask_center(self):
        center = compute_mask_center(convex_hull_gt, contour_gt, method=1)
        self.assertTrue(np.allclose(center, (460.95042812077514, 302.4197085774373)))
        
        center = compute_mask_center(convex_hull_gt, contour_gt, method=0)
        self.assertTrue(np.allclose(center, (461.18243408203125, 303.0538635253906)))