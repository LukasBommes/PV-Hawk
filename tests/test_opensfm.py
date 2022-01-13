import os
import sys
import shutil
import json
import subprocess
import unittest
from tempfile import TemporaryDirectory
from deepdiff import DeepDiff 

from tests.common import temp_dir_prefix


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

        #print(opensfm_command)
        #print(os.listdir(mapping_root))
        #print(os.listdir(self.ground_truth_dir))

        proc = subprocess.Popen(opensfm_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sys.stdout.write(line.decode("utf-8"))

        # check if outputs equal ground truth
        file_names = [
            "camera_models.json",
            "reference_lla.json",
            "reconstruction.json",
        ]

        for file_name in file_names:
            with open(os.path.join(mapping_root, file_name)) as file:
                content = json.load(file)
            with open(os.path.join(self.ground_truth_dir, file_name)) as file:
                content_ground_truth = json.load(file)
            self.assertEqual(
                DeepDiff(
                    content_ground_truth, 
                    content
                ), {},
                "{} differs from ground truth".format(file_name)
            )

    def tearDown(self):
        self.work_dir.cleanup()