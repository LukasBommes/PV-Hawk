import os
import json
import unittest
from tempfile import TemporaryDirectory
from deepdiff import DeepDiff 

from extractor.preprocessing import split_tiffs
from .common import temp_dir_prefix, dirs_equal, load_file


class TestSplitSequences(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("tests", "integration", "data")
        self.work_dir = TemporaryDirectory(prefix=temp_dir_prefix)
        self.settings = {
            "ir_file_extension": "TIFF",
            "rgb_file_extension": "mov",
            "extract_timestamps": True,
            "extract_gps": True,
            "extract_gps_altitude": False,
            "sync_rgb": False,
            "subsample": None,
            "rotate_rgb": None,
            "rotate_ir": None,
            "resize_rgb": {
                "width": None,
                "height": None
            },
            "resize_ir": {
                "width": None,
                "height": None
            }
        }
        self.video_dir = os.path.join(self.data_dir, "videos")
        self.output_dir = self.work_dir.name

        # where to load files with desired output format from
        self.ground_truth_dir = os.path.join(self.data_dir, "splitted")

    def test_run(self):
        split_tiffs.run(
            self.video_dir, 
            self.output_dir, 
            **self.settings)

        # check if outputs equal ground truth
        self.assertTrue(
            dirs_equal(
                os.path.join(self.output_dir, "radiometric"), 
                os.path.join(self.ground_truth_dir, "radiometric")
            )
        )

        file_name = "timestamps.csv"
        content, content_ground_truth = load_file(
            self.output_dir, 
            self.ground_truth_dir,
            file_name)

        self.assertEqual(
            DeepDiff(
                content_ground_truth, 
                content
            ), {},
            "{} differs from ground truth".format(file_name)
        ) 
        
        with open(os.path.join(self.output_dir, "gps", "gps.json"), "r") as file:
            content = json.load(file)
        with open(os.path.join(self.ground_truth_dir, "gps", "gps_orig.json"), "r") as file:
            content_ground_truth = json.load(file)

        self.assertEqual(
            DeepDiff(
                content_ground_truth, 
                content
            ), {},
            "{} differs from ground truth".format("gps.json")
        )        

    def tearDown(self):
        self.work_dir.cleanup()