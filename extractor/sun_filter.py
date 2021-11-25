"""Filters patches of a PV module containing sun reflections.

Depending on the camera angle some of the patches of a PV module may contain
severe sun relfections which disturb downstream tasks, such as fault
classification. Sun reflections differ from thermal anomalies in that they are
non-stationary, but instead change position over subsequent patches. This filter
exploits this fact to differentiate modules with reflections from thoss without.

It works as follows:
1) Compute the maximum temperature and the x, y coordinate of the maximum
   temperature in each patch (after slight blurring to reduce impact of noise)
2) Compute discrete element-wise difference between x, y coordinates of max
   temp point in subsequent patches
3) Threshold the norm of the two difference-singals (threshold_changepoint)
4) Find the segments of the signal from 3 which are zero and contain more than
   30 percent of the sanples in the sequence (if no segment fullfills this
   simply choose the longest segment)
5) Select the segment which has the lowest variance as a reference
6) Compute the median max temperature and median x, y coordinates of the max
   temp point in the reference sequence
7) Compute the element-wise difference between the median max temp and the max
   temp of each patch (repeat for x, y cooridnates of max tmep point)
8) Label all patches in which the differences from step 7 exceed predefined
   thresholds (threshold_temp, threshold_loc) as patches with sun reflection
"""

import os
import glob
import csv
import logging
import itertools
import operator
from tqdm import tqdm
import numpy as np
import cv2

from extractor.common import delete_output, to_celsius, \
    get_immediate_subdirectories


logger = logging.getLogger(__name__)


def get_zero_islands(signal):
    """Get start and stop indices of all zero islands
    in the binary signal sorted after length."""
    sig = np.copy(signal)
    idxs = []
    while len(sig[sig == 0]) > 0:
        indices = max((list(y)
            for (x, y)
            in itertools.groupby((enumerate(sig)), operator.itemgetter(1))
            if x == 0), key=len)
        start_idx = indices[0][0]
        stop_idx = indices[-1][0]
        sig[start_idx:stop_idx+1] = 1
        idxs.append((start_idx, stop_idx+1))
    return idxs


def min_temp_var_segment(max_locs_peaks, max_temps,
    segment_length_threshold=0.3):
    """Get all sequences which are longer than `segment_length_threshold`*100
    percent of the total sequence length, e.g. if `segment_length_threshold`
    is 0.3 up to three segments can be selected. Then selected the segment
    with lowest variance of max temperature.
    """
    idxs = get_zero_islands(max_locs_peaks)
    if len(idxs) == 0:
        return 0, len(max_temps)-1
    idxs_tmp = []
    for start_idx, stop_idx in idxs:
        if (stop_idx - start_idx) / len(max_locs_peaks) > segment_length_threshold:
            idxs_tmp.append((start_idx, stop_idx))
    # no sequence is longer than segment_length_threshold, just use the longest one
    if len(idxs_tmp) == 0:
        idxs_tmp = [idxs[0]]
    idxs = idxs_tmp
    # compute tmeperature variance (or interquartile range)
    # of the segments and choose the one with the lowest variance
    segment_vars = [np.var(max_temps[start_idx:stop_idx])
        for start_idx, stop_idx in idxs]
    nominal_segment_idxs = idxs[np.argmin(segment_vars)]
    start_idx, stop_idx = nominal_segment_idxs
    return start_idx, stop_idx


def predict_sun_reflections(patch_files, threshold_temp=5.0, threshold_loc=10.0,
    threshold_changepoint=10.0):
    if len(patch_files) < 2:
        return (
            np.array([], dtype=np.int32), np.array([], dtype=np.float64),
            np.array([], dtype=np.float64), None, None)

    max_locs = []
    max_temps = []
    for patch_file in patch_files:
        patch = cv2.imread(patch_file, cv2.IMREAD_ANYDEPTH)

        # average blur image to prevent noise from affecting
        # the maximum location
        patch = cv2.blur(patch, ksize=(3, 3))

        max_temp = np.max(to_celsius(patch))
        max_loc = np.unravel_index(np.argmax(patch, axis=None), patch.shape)

        max_locs.append(max_loc)
        max_temps.append(max_temp)
    max_locs = np.vstack(max_locs).astype(np.float64)
    max_temps = np.hstack(max_temps)

    # compute difference between subsequent max_loc points
    max_locs_diff = np.diff(max_locs, n=1, axis=0)
    max_locs_diff = np.linalg.norm(max_locs_diff, axis=1)

    # find peaks in the difference signal
    max_locs_peaks = np.zeros(max_locs_diff.shape, dtype=np.int32)
    max_locs_peaks[max_locs_diff >= threshold_changepoint] = 1
    max_locs_peaks[max_locs_diff < threshold_changepoint] = 0

    # find start and stop index of the segment of zeros which
    # is sufficiently long and has smallest temperature variance
    start_idx, stop_idx = min_temp_var_segment(max_locs_peaks, max_temps)

    # compute median location point and median temperature of this segment
    center_loc = np.median(max_locs[start_idx:stop_idx+1, :], axis=0)
    center_temp = np.median(max_temps[start_idx:stop_idx+1])

    # compute distance of each point to centroid
    distances_loc = np.linalg.norm(max_locs - center_loc, axis=1)
    distances_temp = np.abs(max_temps - center_temp)

    # determine which patches exceed thresholds
    outliers_temp = np.where(distances_temp > threshold_temp)[0]
    outliers_loc = np.where(distances_loc > threshold_loc)[0]

    # determine patches for which one of the two criteria is not met
    patch_idxs_sun_reflections = np.array(
        list(set(outliers_loc) | set(outliers_temp)), dtype=np.int32)

    return (patch_idxs_sun_reflections, distances_loc,
        distances_temp, start_idx, stop_idx)


def run(patches_root, output_dir, threshold_temp=5.0,
    threshold_loc=10.0, threshold_changepoint=10.0):
    delete_output(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    patch_dir = os.path.join(patches_root, "radiometric")
    if not os.path.isdir(patch_dir):
        logger.info("No patches found. Skipping.")
        return None

    with open(os.path.join(output_dir, "sun_filter.csv"), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        plant_ids = sorted(get_immediate_subdirectories(patch_dir))
        for i, plant_id in enumerate(tqdm(plant_ids)):
            patch_files = sorted(glob.glob(
                os.path.join(patch_dir, plant_id, "*")))
            patch_idxs_sun_reflections, _, _, _, _ = predict_sun_reflections(
                patch_files, threshold_temp, threshold_loc,
                threshold_changepoint)
            if len(patch_idxs_sun_reflections) > 0:
                for patch_idx in patch_idxs_sun_reflections:
                    csvwriter.writerow([plant_id, patch_idx])
            else:
                csvwriter.writerow([plant_id, -1])
