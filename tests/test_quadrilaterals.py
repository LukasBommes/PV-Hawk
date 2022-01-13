import os
import pickle
import unittest
from tempfile import TemporaryDirectory
from deepdiff import DeepDiff 

from extractor import quadrilaterals
from tests.common import temp_dir_prefix


class TestQuadrilaterals(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("tests", "data", "large")
        self.work_dir = TemporaryDirectory(prefix=temp_dir_prefix)
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
        file_name = "quadrilaterals.pkl"
        with open(os.path.join(self.output_dir, file_name), "rb") as file:
            content = pickle.load(file)
        with open(os.path.join(self.ground_truth_dir, file_name), "rb") as file:
            content_ground_truth = pickle.load(file)
        self.assertEqual(
            DeepDiff(
                content_ground_truth, 
                content
            ), {},
            "{} differs from ground truth".format(file_name)
        )

    def tearDown(self):
        self.work_dir.cleanup()