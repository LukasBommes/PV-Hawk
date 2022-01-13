import os
import shutil
import unittest
from tempfile import TemporaryDirectory
from deepdiff import DeepDiff

from extractor.mapping import triangulate_modules
from tests.common import temp_dir_prefix, load_file


class TestTriangulateModules(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("tests", "data", "large")
        self.work_dir = TemporaryDirectory(prefix=temp_dir_prefix)
        self.settings = {
            "min_track_len": 2,
            "merge_overlapping_modules": True,
            "merge_threshold": 20,
            "max_module_depth": -1,
            "max_num_modules": 300,
            "max_combinations": -1,
            "reproj_thres": 5.0,
            "min_ray_angle_degrees": 1.0 
        }
        self.tracks_root = os.path.join(self.data_dir, "tracking")        
        self.quads_root = os.path.join(self.data_dir, "quadrilaterals")
        mapping_root = os.path.join(self.data_dir, "mapping")
        
        # copy files into temporary directory as triangulation works in-place
        shutil.copytree(
            src=os.path.join(mapping_root, "cluster_000000"), 
            dst=os.path.join(self.work_dir.name, "mapping", "cluster_000000")
        )

        # where to load files with desired output format from
        self.ground_truth_dir = mapping_root

    def test_run(self):
        mapping_root = os.path.join(self.work_dir.name, "mapping")
        triangulate_modules.run(
            mapping_root, 
            self.tracks_root, 
            self.quads_root, 
            **self.settings)
        
        # check if outputs equal ground truth
        file_names = [
            "modules.pkl",
            "module_geolocations.geojson",
            "pose_graph.pkl",
            "module_corners.pkl",
            "merged_modules.pkl",
            "map_points.pkl",
            "reference_lla.pkl"
        ]

        # compare with deepdiff
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

    def tearDown(self):
        self.work_dir.cleanup()