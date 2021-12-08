import os
import glob
import shutil
import csv
from copy import deepcopy
from collections.abc import Mapping
from collections import defaultdict
import logging
import numpy as np
import cv2


logger = logging.getLogger(__name__)


def delete_output(output_dir, cluster=None):
    """Deletes the specified directory.
    If a cluster is specified, the behaviour is different. Instead of deleting
    the entire directory, only files or subdirectories belonging to this cluster
    (found via the cluster_idx) are deleted."""
    if cluster is None:
        shutil.rmtree(output_dir, ignore_errors=True)
        logger.info("Deleted {}".format(output_dir))
    else:
        cluster_idx = cluster["cluster_idx"]
        path_objects = glob.glob(os.path.join(output_dir, "*{:06d}*".format(cluster_idx)))
        for path_object in path_objects:
            try:
                os.remove(path_object)
                logger.info("Deleted {}".format(path_object))
            except IsADirectoryError:
                shutil.rmtree(path_object, ignore_errors=True)
                logger.info("Deleted {}".format(path_object))
            except FileNotFoundError:
            	pass


def get_immediate_subdirectories(a_dir):
    """Returns the immediate subdirectories of the provided directory."""
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def get_group_name(group):
    """Returns the group name if available, empty string otherwise."""
    try:
        group_name = group["name"]
    except KeyError:
        group_name = ""
    return group_name


def merge_dicts(dict1, dict2):
    """Return a new dictionary by merging two dictionaries recursively."""
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, Mapping):
            result[key] = merge_dicts(result.get(key, {}), value)
        else:
            result[key] = deepcopy(dict2[key])
    return result


def remove_none(obj):
    """Removes all None items (either key or value) from nested data structures
    of dicts, lists, tuples and sets."""
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_none(x) for x in obj if x is not None)
    elif isinstance(obj, dict):
        return type(obj)((remove_none(k), remove_none(v))
            for k, v in obj.items() if k is not None and v is not None)
    else:
        return obj


def replace_empty_fields(dict1):
    """Takes a potentially nested dictionary and replaces all None values with
    an empty dictionary."""
    for key, value in dict1.items():
        if value is None:
            dict1[key] = {}


# def raw_image_to_celsius(image, gain, offset):
#     """Convert raw intensity values of radiometric image to Celsius scale."""
#     return image*gain + offset


# def preprocess_radiometric_frame(frame, equalize_hist=True):
#     """Preprocesses raw radiometric frame.

#     First, the raw 16-bit radiometric intensity values are converted to Celsius
#     scale. Then, the image values are normalized to range [0, 255] and converted
#     to 8-bit. Finally, histogram equalization is performed to normalize
#     brightness and enhance contrast.
#     """
#     frame = to_celsius(frame)
#     frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
#     frame = (frame*255.0).astype(np.uint8)
#     if equalize_hist:
#         frame = cv2.equalizeHist(frame)
#     return frame


def truncate_patch(patch, margin=0.1):
    """Truncates module edges by margin (percent of width) to remove module frame."""
    width = patch.shape[1]
    margin_px = int(margin*width)
    patch = patch[margin_px:-margin_px, margin_px:-margin_px]
    return patch


def get_max_mean_temp_patch(patch_files, patch_idxs_sun_reflections=[]):
    """Returns index of the patch in patch_files with largest mean temperature.
    Optionally ignores patches with sun reflections."""
    if len(patch_files) == 0:
        return None
    patch_idx = 0
    previous_mean_temp = -np.inf
    for idx, patch_file in enumerate(patch_files):
        if idx in patch_idxs_sun_reflections:  # ignore patches with sun reflections
            continue
        patch = cv2.imread(patch_file, cv2.IMREAD_ANYDEPTH)
        patch = truncate_patch(patch, margin=0.2)
        mean_temp = to_celsius(np.mean(patch))
        if mean_temp > previous_mean_temp:
            patch_idx = idx
            previous_mean_temp = mean_temp
    return patch_idx


def contour_and_convex_hull(mask):
    """Computes the contour and convex hull of a binary mask image.

    If the mask consists of several disconnected contours, only the largest one
    is considered.

    Args:
        mask (`numpy.ndarray`): Binary image of a segmentation mask with shape
            `(H, W)` and dtype uint8. The background should be represented by 0
            and the segmented object by 255.

    Returns:
        convex_hull (`numpy.ndarray`): Subset of the M contour points which
        represents the convex hull of the provided mask. Shape (M, 1, 2),
        dtype int32.

        contour (`numpy.ndarray`): The N contour points uniquely describing the
        boundary between segmented object and background in the provided mask.
        Shape (N, 1, 2), dtype int32.
    """
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # get largest contour
    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))
    cnt_idx = np.argmax(areas)
    convex_hull = cv2.convexHull(contours[cnt_idx], clockwise=False, returnPoints=True)
    return convex_hull, contours[cnt_idx]


def compute_mask_center(convex_hull, contour, method=1):
    """Computes the center point of a contour representing a segmentation mask.

    Can be used to compute the center point of a segmentation mask by first
    computing the mask's countour with the `contour_and_convex_hull` method.

    Args:
        convex_hull (`numpy.ndarray`): Shape (M, 1, 2), dtype int32. Convex hull
        of a segmented object as returned by `contour_and_convex_hull` method.

        contour (`numpy.ndarray`): Shape (N, 1, 2), dtype int32. Contour points
        of a segmented object as returned by `contour_and_convex_hull` method.

        method (`int`): If 0 compute the center point as the center of the
            minimum enclosing circle of the convex hull. If 1 compute the center
            by means of image moments of the contour.

    Returns:
        center (`tuple` of `float`): x and y position of the countour's center
        point.
    """
    if method == 0:
        center, _ = cv2.minEnclosingCircle(convex_hull)
    if method == 1:
        M = cv2.moments(contour)
        center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    return center


def line(p1, p2):
    """Converts line from a 2-point representation into a 3-parameter representation."""
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def line_intersection(L1, L2):
    """Computes intersection of two lines.

    Source: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    """
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return True, (x, y)
    else:
        return False, (0., 0.)


def parse_sun_filter_file(sun_filter_file):
    """Parse the results of the sun filter into a dictionary. Keys are plant_ids
    and values lists of patch indices containing sun reflections."""
    patch_idxs_sun_reflections = defaultdict(list)
    try:
        with open(sun_filter_file, newline='', encoding="utf-8-sig") as csvfile:  # specifying the encoding skips optional BOM
            # automatically infer CSV file format
            dialect = csv.Sniffer().sniff(csvfile.readline(), delimiters=",;")
            csvfile.seek(0)
            csvreader = csv.reader(csvfile, dialect)
            for row in csvreader:
                plant_id = row[0]
                patch_idx = int(row[1])
                if patch_idx != -1:
                    patch_idxs_sun_reflections[plant_id].append(patch_idx)
    except FileNotFoundError:
        logger.info("Sun filter file not found. Ignoring.")
    finally:
        return patch_idxs_sun_reflections


class Capture:
    def __init__(self, image_files, mask_files=None, camera_matrix=None,
            dist_coeffs=None, to_celsius=None):
        self.frame_counter = 0
        self.image_files = image_files
        self.mask_files = mask_files
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        if to_celsius is None:
            to_celsius = {"gain": 1.0, "offset": 0.0}
        self.gain = to_celsius["gain"]
        self.offset = to_celsius["offset"]
        self.num_images = len(self.image_files)
        # precompute undistortion maps
        probe_frame = cv2.imread(self.image_files[0], cv2.IMREAD_ANYDEPTH)
        self.img_w = probe_frame.shape[1]
        self.img_h = probe_frame.shape[0]
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            new_camera_matrix = self.camera_matrix
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None,
                new_camera_matrix, (self.img_w, self.img_h), cv2.CV_32FC1)
        if mask_files is not None:
            assert len(mask_files) == len(image_files), "Number of mask_files and image_files do not match"
            self.mask_files = mask_files

    def raw_to_celsius(self, image):
        """Convert raw intensity values of radiometric image to Celsius scale."""
        return image*self.gain + self.offset

    def preprocess_radiometric_frame(self, frame, equalize_hist):
        """Preprocesses raw radiometric frame.

        First, the raw 16-bit radiometric intensity values are converted to Celsius
        scale. Then, the image values are normalized to range [0, 255] and converted
        to 8-bit. Finally, histogram equalization is performed to normalize
        brightness and enhance contrast.
        """
        frame = self.raw_to_celsius(frame)
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        frame = (frame*255.0).astype(np.uint8)
        if equalize_hist:
            frame = cv2.equalizeHist(frame)
        return frame

    def get_next_frame(self, preprocess=True, undistort=False,
            equalize_hist=True):
        frame, masks, frame_name, mask_names = self.get_frame(
            self.frame_counter, preprocess, undistort,
            equalize_hist)
        self.frame_counter += 1
        return frame, masks, frame_name, mask_names

    def get_frame(self, index, preprocess=True, undistort=False,
            equalize_hist=True):
        frame = None
        masks = None
        frame_name = None
        mask_names = None
        if index < self.num_images:
            image_file = self.image_files[index]
            frame_name = str.split(os.path.basename(image_file), ".")[0]
            frame = cv2.imread(image_file, cv2.IMREAD_ANYDEPTH)
            if self.mask_files is not None:
                mask_file = self.mask_files[index]
                masks = [cv2.imread(m, cv2.IMREAD_ANYDEPTH) for m in mask_file]
                mask_names = [str.split(os.path.basename(m), ".")[0] for m in mask_file]
            if preprocess:
                frame = self.preprocess_radiometric_frame(frame, equalize_hist)
            if undistort and self.camera_matrix is not None and self.dist_coeffs is not None:
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_CUBIC)
                if self.mask_files is not None:
                    masks = [cv2.remap(mask, self.mapx, self.mapy, cv2.INTER_CUBIC) for mask in masks]
        return frame, masks, frame_name, mask_names
