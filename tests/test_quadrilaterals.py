import os
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
        self.output_dir = os.path.join(self.work_dir.name, "quadrilaterals")
        os.makedirs(self.output_dir, exist_ok=True)

    def test_run(self):
        quadrilaterals.run(
            self.frames_root, 
            self.inference_root, 
            self.tracks_root,
            self.output_dir, 
            **self.settings)

        print(os.listdir(self.work_dir.name))
        print(os.listdir(self.tracks_root))
        print(os.listdir(self.output_dir))
        
        # check if output files have desired form
        # 1) Load ground truth file from tests/data
        # 2) compare
        #assert

    def tearDown(self):
        self.work_dir.cleanup()