import os
import glob
import sys
import shutil
import subprocess
import unittest
from tempfile import TemporaryDirectory
from deepdiff import DeepDiff 

from tests.common import temp_dir_prefix, load_file


class TestOpenSfm(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("tests", "data", "large")
        self.work_dir = TemporaryDirectory(prefix=temp_dir_prefix)
        self.settings = {
            "matching_gps_distance": 15,
            "processes": 16,
            "use_altitude_tag": False,
            "align_method": "orientation_prior",
            "align_orientation_prior": "vertical"
        }
        self.opensfm_tasks = [
            "opensfm_extract_metadata", 
            "opensfm_detect_features",
            "opensfm_match_features",
            "opensfm_create_tracks",
            "opensfm_reconstruct"
        ]
        self.opensfm_bin = "/pvextractor/extractor/mapping/OpenSfM/bin/opensfm"

        mapping_root = os.path.join(self.data_dir, "mapping", "cluster_000000")
        
        # copy files into temporary directory as opensfm works in-place
        shutil.copytree(
            src=os.path.join(mapping_root), 
            dst=os.path.join(self.work_dir.name, "mapping", "cluster_000000")
        )

        # where to load files with desired output format from
        self.ground_truth_dir = mapping_root

    def test_run(self):
        mapping_root = os.path.join(self.work_dir.name, "mapping", "cluster_000000")
        
        opensfm_command = []
        for task in self.opensfm_tasks:
            opensfm_command.append(f'{self.opensfm_bin} {task[8:]} "{mapping_root}"')
        opensfm_command = " && ".join(opensfm_command)

        proc = subprocess.Popen(opensfm_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sys.stdout.write(line.decode("utf-8"))

        # check if outputs equal ground truth
        file_names = [
            "camera_models.json",
            "reference_lla.json",
        ]

        for file_name in file_names:
            content, content_ground_truth = load_file(
                mapping_root, 
                self.ground_truth_dir, 
                file_name)

            self.assertEqual(
                DeepDiff(
                    content_ground_truth, 
                    content,
                    math_epsilon=1e-5
                ), {},
                "{} differs from ground truth".format(file_name)
            )

        # due to randomness in OpenSfM the reconstruction differs between runs 
        # and we can test only some characteristics
        reconstruction, reconstruction_ground_truth = load_file(
                mapping_root, 
                self.ground_truth_dir, 
                "reconstruction.json")

        self.assertEqual(len(reconstruction), len(reconstruction_ground_truth))

        self.assertEqual(DeepDiff(
            reconstruction_ground_truth[0]["reference_lla"], 
            reconstruction[0]["reference_lla"],
            math_epsilon=1e-5
        ), {})

        self.assertEqual(DeepDiff(
            reconstruction_ground_truth[0]["biases"], 
            reconstruction[0]["biases"],
            math_epsilon=1e-5
        ), {})

        self.assertEqual(DeepDiff(
            reconstruction_ground_truth[0]["cameras"], 
            reconstruction[0]["cameras"],
            math_epsilon=1e-2 # maybe set to 1e-1 to tolerate larger deviations
        ), {})

        self.assertTrue(len(reconstruction[0]["points"]) > 1000)
        self.assertEqual(
            len(reconstruction[0]["shots"]),
            len(glob.glob(os.path.join(mapping_root, "images", "*.jpg")))
        )

    def tearDown(self):
        self.work_dir.cleanup()