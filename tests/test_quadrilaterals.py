import os
import pickle
import unittest
from tempfile import TemporaryDirectory

from extractor import quadrilaterals


class TestQuadrilaterals(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("tests", "data")
        self.work_dir = TemporaryDirectory()
        self.settings = {
            "min_iou": 0.9
        }
        self.frames_root = os.path.join(self.data_dir, "splitted")
        self.inference_root = os.path.join(self.data_dir, "segmented")
        self.tracks_root = os.path.join(self.data_dir, "tracking")
        self.output_dir = self.work_dir.name

        # where to load files with desired output format from
        self.ground_truth_dir = os.path.join(self.data_dir, "quadrilaterals")

    def test_run(self):
        quadrilaterals.run(
            self.frames_root, 
            self.inference_root, 
            self.tracks_root,
            self.output_dir, 
            **self.settings)

        # check if outputs equal ground truth
        with open(os.path.join(self.output_dir, "quadrilaterals.pkl"), "rb") as file:
            quads = pickle.load(file)
        with open(os.path.join(self.ground_truth_dir, "quadrilaterals.pkl"), "rb") as file:
            quads_ground_truth = pickle.load(file)
        self.assertEqual(quads, quads_ground_truth)

    def tearDown(self):
        self.work_dir.cleanup()