import os
import unittest
from tempfile import TemporaryDirectory

from extractor import cropping
from .common import temp_dir_prefix, dirs_equal


class TestCropping(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("tests", "data", "large")
        self.work_dir = TemporaryDirectory(prefix=temp_dir_prefix)
        self.settings = {
            "rotate_mode": "portrait"
        }
        self.frames_root = os.path.join(self.data_dir, "splitted")
        self.quads_root = os.path.join(self.data_dir, "quadrilaterals")
        self.mapping_root = os.path.join(self.data_dir, "mapping")
        self.output_dir = self.work_dir.name

        # where to load files with desired output format from
        self.ground_truth_dir = os.path.join(self.data_dir, "patches")

    def test_run(self):
        cropping.run(
            self.frames_root, 
            self.quads_root, 
            self.mapping_root,
            self.output_dir, 
            **self.settings)

        # check if outputs equal ground truth
        self.assertTrue(
            dirs_equal(
                self.output_dir, 
                self.ground_truth_dir
            )
        )

    def tearDown(self):
        self.work_dir.cleanup()