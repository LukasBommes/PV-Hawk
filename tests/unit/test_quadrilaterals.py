import unittest
import numpy as np

from extractor.quadrilaterals import line_intersection, \
    line, find_enclosing_polygon, compute_iou

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

class TestQuadrilaterals(unittest.TestCase):
    
    def test_line(self):
        self.assertEqual(line([0, 0], [1, 1]), (-1, 1, 0))
        self.assertEqual(line([0, 1], [1, 0]), (1, 1, 1))
        self.assertEqual(line([0, 0], [-1, -1]), (1, -1, 0))
    
    def test_line_intersection(self):
        # intersecting lines
        l1 = (-1, 1, 0) # line([0, 0], [1, 1])
        l2 = (1, 1, 1) # line([0, 1], [1, 0])
        has_intersect, intersect_pt = line_intersection(l1, l2)
        self.assertTrue(has_intersect)
        self.assertEqual(intersect_pt, (0.5, 0.5))
        
        # intersecting lines 2
        l1 = (1, -1, 0) # line([0, 0], [-1, -1])
        l2 = (1, 1, 1) # line([0, 1], [1, 0])
        has_intersect, intersect_pt = line_intersection(l1, l2)
        self.assertTrue(has_intersect)
        self.assertEqual(intersect_pt, (0.5, 0.5))
        
        # colinear lines
        l1 = (-1, 1, 0) # line([0, 0], [1, 1])
        l2 = (-1, 1, 0) # line([1, 1], [2, 2])
        has_intersect, intersect_pt = line_intersection(l1, l2)
        self.assertFalse(has_intersect)
        self.assertEqual(intersect_pt, (0.0, 0.0))
        
        # parallel lines
        l1 = (0, 1, 0) # line([0, 0], [1, 0])
        l2 = (0, 1, 1) # line([0, 1], [1, 1])
        has_intersect, intersect_pt = line_intersection(l1, l2)
        self.assertFalse(has_intersect)
        self.assertEqual(intersect_pt, (0.0, 0.0))
    
    def test_find_enclosing_polygon(self):
        # ensure method does not return enclosing polygon with more vertices than original
        enclosing_poly = find_enclosing_polygon(convex_hull_gt, num_vertices=20)
        self.assertTrue(
            np.allclose(
                enclosing_poly,
                convex_hull_gt
            )
        )
        
        enclosing_poly = find_enclosing_polygon(convex_hull_gt, num_vertices=4)
        enclosing_poly_gt = np.array([
            [[499, 327]],
            [[421, 323]],
            [[424, 279]],
            [[499, 280]]
        ], dtype=np.int32)
        self.assertTrue(
            np.allclose(
                enclosing_poly,
                enclosing_poly_gt
            )
        )
        
        enclosing_poly = find_enclosing_polygon(convex_hull_gt, num_vertices=5)
        enclosing_poly_gt = np.array([
            [[499, 326]],
            [[476, 326]],
            [[421, 323]],
            [[424, 279]],
            [[499, 280]]
        ], dtype=np.int32)
        self.assertTrue(
            np.allclose(
                enclosing_poly,
                enclosing_poly_gt
            )
        )  
        
    def test_compute_iou(self):
        # partial overlap
        quad = np.array([
            [[499, 327]],
            [[421, 323]],
            [[424, 279]],
            [[499, 280]]
        ], dtype=np.int32)
        iou = compute_iou(convex_hull_gt, quad)
        self.assertAlmostEqual(iou, 0.9794293394183168)
        
        # full overlap
        iou = compute_iou(quad, quad)
        self.assertAlmostEqual(iou, 1.0)
        
        iou = compute_iou(convex_hull_gt, convex_hull_gt)
        self.assertAlmostEqual(iou, 1.0)
        
        # no overlap
        quad = np.array([
            [[0, 0]],
            [[1, 0]],
            [[1, 1]],
            [[0, 1]]
        ], dtype=np.int32)
        quad2 = quad + 1
        iou = compute_iou(quad, quad2)
        self.assertAlmostEqual(iou, 0.0)