import os
import shutil
import unittest
from tempfile import TemporaryDirectory
from deepdiff import DeepDiff 

from extractor.preprocessing import interpolation
from .common import temp_dir_prefix, load_file


class TestInterpolateGps(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("tests", "integration", "data")
        self.work_dir = TemporaryDirectory(prefix=temp_dir_prefix)
        self.settings = {}
        frames_root = os.path.join(self.data_dir, "splitted")    
        
        # copy files into temporary directory as interpolation works in-place
        shutil.copytree(
            src=os.path.join(frames_root),
            dst=os.path.join(self.work_dir.name, "splitted")
        )

        # where to load files with desired output format from
        self.ground_truth_dir = frames_root

    def test_run(self):
        frames_root = os.path.join(self.work_dir.name, "splitted")
        interpolation.run(
            frames_root,
            **self.settings)

        # check if outputs equal ground truth 
        file_names = [
            "gps.json",
            "gps_orig.json"
        ]
        for file_name in file_names:
            content, content_ground_truth = load_file(
                os.path.join(frames_root, "gps"), 
                os.path.join(self.ground_truth_dir, "gps"),
                file_name)

            self.assertEqual(
                DeepDiff(
                    content_ground_truth, 
                    content
                ), {},
                "{} differs from ground truth".format(file_name)
            )        

    def tearDown(self):
        self.work_dir.cleanup()