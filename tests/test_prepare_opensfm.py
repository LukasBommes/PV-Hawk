import os
import unittest
from tempfile import TemporaryDirectory
import json
import yaml

from extractor.mapping import prepare_opensfm
from tests.common import temp_dir_prefix, dirs_equal


class TestPrepareOpensfm(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("tests", "data", "large")
        self.work_dir = TemporaryDirectory(prefix=temp_dir_prefix)
        self.settings = {
            "select_frames_mode": "gps_visual",
            "frame_selection_gps_distance": 0.75,
            "frame_selection_visual_distance": 0.15,
            "orb_nfeatures": 5000,
            "orb_fast_thres": 12,
            "orb_scale_factor": 1.2,
            "orb_nlevels": 8,
            "match_distance_thres": 20.0,
            "gps_dop": 0.1,
            "output_video_fps": 5.0            
        }
        self.opensfm_settings = {
            "matching_gps_distance": 15,
            "processes": 16,
            "use_altitude_tag": "no",
            "align_method": "orientation_prior",
            "align_orientation_prior": "vertical"
        }
        self.cluster = {
            "cluster_idx": 0,
            "frame_idx_start": 0,
            "frame_idx_end": 2541
        }
        self.frames_root = os.path.join(self.data_dir, "splitted")
        self.calibration_root = os.path.join(self.data_dir, "..", "calibration_params")
        self.output_dir = self.work_dir.name

        # where to load files with desired output format from
        self.ground_truth_dir = os.path.join(self.data_dir, "mapping")

    def test_run(self):
        prepare_opensfm.run(
            self.cluster, 
            self.frames_root, 
            self.calibration_root, 
            self.output_dir, 
            self.opensfm_settings, 
            **self.settings)

        #print(os.listdir(self.output_dir))
        #print(os.listdir(os.path.join(self.output_dir, "cluster_000000")))
        #print(os.listdir(os.path.join(self.output_dir, "cluster_000000", "images")))

        # check if outputs equal ground truth
        self.assertTrue(
            dirs_equal(
                os.path.join(self.output_dir, "cluster_000000", "images"), 
                os.path.join(self.ground_truth_dir, "cluster_000000", "images")
            )
        )

        self.assertEqual(
            json.load(open(os.path.join(
                self.output_dir, "cluster_000000", "exif_overrides.json"), "r")),
            json.load(open(os.path.join(
                self.ground_truth_dir, "cluster_000000", "exif_overrides.json"), "r")),
        )

        self.assertEqual(
            json.load(open(os.path.join(
                self.output_dir, "cluster_000000", "camera_models_overrides.json"), "r")),
            json.load(open(os.path.join(
                self.ground_truth_dir, "cluster_000000", "camera_models_overrides.json"), "r")),
        )

        self.assertEqual(
            yaml.safe_load(open(os.path.join(
                self.output_dir, "cluster_000000", "config.yaml"), "r")),
            yaml.safe_load(open(os.path.join(
                self.ground_truth_dir, "cluster_000000", "config.yaml"), "r")),
        )

        self.assertTrue(
            "selected_images.avi" in os.listdir(os.path.join(self.output_dir, "cluster_000000"))
        )

    def tearDown(self):
        self.work_dir.cleanup()