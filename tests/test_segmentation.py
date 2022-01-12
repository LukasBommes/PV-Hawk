import os
import unittest
from tempfile import TemporaryDirectory

from extractor.segmentation import inference
from tests.common import dirs_equal


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("tests", "data")
        self.work_dir = TemporaryDirectory()
        self.settings = {
            
        }
        self.frames_root = os.path.join(self.data_dir, "splitted")
        self.output_dir = self.work_dir.name

        # where to load files with desired output format from
        self.ground_truth_dir = os.path.join(self.data_dir, "segmented")

    def test_run(self):

            frames_root = os.path.join(
                config["work_dir"], group_name, "splitted", "radiometric")
            output_dir = os.path.join(config["work_dir"], group_name, "segmented")
        inference.run(
            self.frames_root, 
            self.output_dir,
            self.settings)

        # check if outputs equal ground truth
        #self.assertTrue(dirs_equal(self.output_dir, self.ground_truth_dir))

    def tearDown(self):
        self.work_dir.cleanup()