import os
import glob
import unittest
from tempfile import TemporaryDirectory
import cv2

from extractor.segmentation import inference
from extractor.common import get_immediate_subdirectories
from .common import temp_dir_prefix, dirs_equal


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join("tests", "integration", "data")
        self.work_dir = TemporaryDirectory(prefix=temp_dir_prefix)
        self.settings = {
            "output_video_fps": 8.0
        }
        self.frames_root = os.path.join(self.data_dir, "splitted")
        self.output_dir = self.work_dir.name

        # where to load files with desired output format from
        self.ground_truth_dir = os.path.join(self.data_dir, "segmented")

    def test_run(self):
        inference.run(
            self.frames_root, 
            self.output_dir,
            **self.settings)

        # check if output files and directories equal ground truth
        # do not consider file contents since Mask R-CNN generates slightly
        # different results depending on the hardware used (e.g. CPU vs. GPU)
        self.assertTrue(
            dirs_equal(
                os.path.join(self.output_dir, "masks"), 
                os.path.join(self.ground_truth_dir, "masks"),
                ignore_file_contents=True
            )
        )

        # check shapes and dtypes of masks
        mask_dirs = get_immediate_subdirectories(
            os.path.join(self.output_dir, "masks"))
        mask_dirs_gt = get_immediate_subdirectories(
            os.path.join(self.ground_truth_dir, "masks"))

        for mask_dir, mask_dir_gt in zip(mask_dirs, mask_dirs_gt):
            mask_files = sorted(glob.glob(os.path.join(self.output_dir, "masks", mask_dir, "mask_*.png")))
            mask_files_gt = sorted(glob.glob(os.path.join(self.ground_truth_dir, "masks", mask_dir_gt, "mask_*.png")))
            
            for mask_file, mask_file_gt in zip(mask_files, mask_files_gt):
                mask = cv2.imread(mask_file)
                mask_gt = cv2.imread(mask_file_gt)
                
                self.assertEqual(mask.shape, mask_gt.shape)
                self.assertEqual(mask.dtype, mask_gt.dtype)

    def tearDown(self):
        self.work_dir.cleanup()