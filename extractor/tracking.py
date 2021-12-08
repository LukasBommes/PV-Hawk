"""Tracks PV modules over subsequent video frames.

This module performs multiple-object tracking of segmentation masks of PV
modules in subsequent IR video frames. The aim is to identify the same physical
PC module in subsequent frames and assign a unique tracking ID to it. This
allows to group image patches of the same PV module which were cropped from
subsequent frames.

Tracking is based on center points of segmentation masks of PV modules in a frame.
First, each center point in frame t-1 is projected into t using estimated camera
motion. Subsequently, projected centers and segmentation mask centers in frame t
are matched with each other. In case of a succesful match, the tracking ID of
the projected mask center point is assigned to the segmentation mask center in
frame t. This way, tracking IDs are propagated through subsequent frames.
Whenever a new PV module becomes visible, a new tracking ID is created. Once a
previously tracked PV module leaves the frame its tracking ID gets deleted.
Note, that throughout the code the terms "segmentation" and "detection" are used
synonymously.

Projection of mask center points works in detail as follows:
1) Extract FAST keypoints and ORB descriptors for two frames t-1 and t
   (optionally apply SSC method to keep only a subset of evenly distributed FAST
   feature points)
2) Match keypoints with opencv brute force matcher
3) Estimate a motion model (homography, affine, etc.) based on the two sets of
   points in frame t-1 and t
4) Project mask center from frame t-1 into t using the estimated motion model

Association of projected and segmented mask center points works in detail as
follows:
1) Build a KD-tree of center positions of segmentation masks in frame t
2) For each projected mask center query the nearest neighbour segmentation mask
   in frame t and name this a "match"
3) If two or more projected masks were matched with the same segmentation mask,
   keep only the match with the smallest distance and enlist the other projected
   masks as "unmatched tracks"
3) Enlist every segmentation mask which has not been matched to a projected mask
   as "unmatched segmentation"

The last step which is performed is track management. Here, for each unmatched
segmentation a new track with a new tracking ID (random UUID) is created. Each
track which corresponds to an unmatched track is deleted. And for each match the
tracking ID of the segmentation mask in frame t is set to the tracking ID of the
matched projected mask in frame t-1.

Note, that we also tried the Hungarian algorithm for matching of projected and
segmented masks. However, this lead to failure cases.

For reference see the paper "Simple Online and Realtime Tracking":
https://arxiv.org/abs/1602.00763
"""

import os
import glob
import csv
import uuid
import logging
from tqdm import tqdm
import numpy as np
import cv2
import scipy.spatial
import scipy.optimize

from extractor.common import Capture, delete_output, contour_and_convex_hull, \
    compute_mask_center, get_immediate_subdirectories
from extractor.keypoints import extract_keypoints, match_keypoints


logger = logging.getLogger(__name__)


def get_duplicates(list_):
    """Returns the duplicates in a list l."""
    seen = {}
    duplicates = []
    for x in list_:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                duplicates.append(x)
            seen[x] += 1
    return duplicates


def match_modules(t_points, d_points, max_distance):
    """Matches detected points with tracked points based on Euclidean distance.

    This function can be used to find matches between sets of points
    found with an object detector and tracked with a tracker. It yields three
    arrays with indices indicating which detected point corresponds to which
    tracked point. This information is needed to update the location of the
    tracked point to the one of the detection box. Matching is performed by
    finding the nearest neighbour tracked point of each detected point.

    Args:
        t_points (`numpy.ndarray`): Array of shape (T, 2) and dtype float of the
            T tracked points in the format [x_center, y_centers] each.

        d_points (`numpy.ndarray`): Array of shape (D, 2) and dtype float of the
            D detected points in the format [x_center, y_centers] each.

        max_distance (`float`): Maximum Euclidean distance (in pixels) between 
            a tracked and detected point to be considered a match.

    Returns:
        matches (`numpy.ndarray`): Array of shape (M, 2) containing the indices
            of all M matched pairs of detected and tracked points. Each row in
            this array has the form [d, t] indicating that the `d`th detected
            point has been matched with the `t`th tracked point (d and t being
            the row indices of d_points and t_points).

        unmatched_trackers (`numpy.ndarray`): Array of shape (L,) containing
            the L row indices of all tracked points which could not be matched
            to any detected point. This indicates an event, such as a previously
            tracked target leaving the scene.

        unmatched_detectors (`numpy.ndarray`): Array of shape (K,) containing
            the K row indices of all detected points which could not be matched
            to any tracked point. This indicates an event such as a new target
            entering the scene.
    """
    matches = []
    unmatched_detections = []
    unmatched_tracks = []
    # build KD tree and query nearest neighbours
    tree = scipy.spatial.KDTree(d_points)
    dists, detected_idx = tree.query(t_points)

    # find which indices of d_points in detected_idx are missing and
    # enlist them as unmatched detections
    for d in list(range(d_points.shape[0])):
        if not d in detected_idx:
            unmatched_detections.append(d)

    # if a detection is associated to multiple tracked points, keep only the
    # match with the smallest distance
    duplicates = get_duplicates(list(detected_idx))
    for duplicate in duplicates:
        duplicate_idx = np.argwhere(detected_idx == duplicate)
        duplicate_dists = []
        dist = dists[duplicate_idx]
        duplicate_dists.append(dist)
        keep_idx = np.argmin(duplicate_dists)
        duplicate_idx = np.delete(duplicate_idx, keep_idx)
        for t in duplicate_idx:
            unmatched_tracks.append(t)
        # disable all other indices
        detected_idx[duplicate_idx] = -1

    # enlist matches between tracked and detected points
    for t, (d, dist) in enumerate(zip(detected_idx, dists)):
        if d != -1:
            if dist <= max_distance:
                matches.append([d, t])
            else:
                unmatched_tracks.append(t)
                unmatched_detections.append(d)

    unmatched_tracks = np.array(sorted(unmatched_tracks, reverse=True))
    unmatched_detections = np.array(sorted(unmatched_detections, reverse=True))
    return matches, unmatched_tracks, unmatched_detections


class Tracker:
    def __init__(self, motion_model, orb_nfeatures, orb_fast_thres, orb_scale_factor, 
        orb_nlevels, match_distance_thres, max_distance):
        self.modules_tracked = np.zeros((0, 2), dtype=np.float32)
        self.module_ids = []
        self.detection_ids_tracked = []
        self.previous_detected_modules = None
        self.previous_frame = None
        self.frame_idx = 0
        self.tracked_modules = []

        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.orb = cv2.ORB_create(
            nfeatures=orb_nfeatures, fastThreshold=orb_fast_thres, 
            scaleFactor=orb_scale_factor, nlevels=orb_nlevels)

        self.motion_model = motion_model
        self.match_distance_thres = match_distance_thres
        self.max_distance = max_distance


    def predict(self, previous_frame, frame, previous_points):
        # extract and match keypoints
        kp_prev, des_prev = extract_keypoints(previous_frame, self.orb)
        kp, des = extract_keypoints(frame, self.orb)

        _, last_pts, current_pts = match_keypoints(
            self.bf_matcher, des_prev, des, kp_prev, kp, 
            distance_threshold=self.match_distance_thres)

        # if there are no matches no motion can be predicted
        if last_pts.shape[1] == 0 or current_pts.shape[1] == 0:
            return np.zeros(shape=(0, 2), dtype=np.float64)

        # estimate a transformation (homography or affine)
        # and predict module centers with it
        if self.motion_model == "homography":
            transform, _ = cv2.findHomography(
                last_pts, current_pts, method=cv2.RANSAC)
            if transform is None:
                return np.zeros(shape=(0, 2), dtype=np.float64)
            centers_pred = cv2.perspectiveTransform(
                previous_points.reshape(-1, 1, 2), transform)

        elif self.motion_model == "affine":
            transform, _ = cv2.estimateAffine2D(
                last_pts, current_pts, method=cv2.RANSAC)
            if transform is None:
                return np.zeros(shape=(0, 2), dtype=np.float64)
            centers_pred = cv2.transform(
                previous_points.reshape(-1, 1, 2), transform)

        elif self.motion_model == "affine_partial":
            transform, _ = cv2.estimateAffinePartial2D(
                last_pts, current_pts, method=cv2.RANSAC)
            if transform is None:
                return np.zeros(shape=(0, 2), dtype=np.float64)
            centers_pred = cv2.transform(
                previous_points.reshape(-1, 1, 2), transform)

        centers_pred = centers_pred.reshape(-1, 2)
        return centers_pred


    def step(self, frame, detected_modules, vis_frame):
        # visualize detections
        for d, detected_module in enumerate(detected_modules):
            text = str(d)
            pos = (int(detected_module[0]), int(detected_module[1]))
            vis_frame = cv2.putText(
                vis_frame, text, (pos[0]+5, pos[1]+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            vis_frame = cv2.circle(vis_frame, pos, 5, (0,255,0), -1)

        if self.frame_idx > 0:
            self.modules_tracked = self.predict(
                self.previous_frame, frame, self.modules_tracked)

            # visualize predicted modules
            for p, module_tracked in enumerate(self.modules_tracked):
                text = str(p)
                pos = (int(module_tracked[0]), int(module_tracked[1]))
                vis_frame = cv2.putText(
                    vis_frame, text, (pos[0], pos[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                vis_frame = cv2.circle(vis_frame, pos, 3, (0,0,255), -1)

        # re-initialize if could not predict motion between frames
        if len(self.modules_tracked) == 0:
            self.module_ids = []
            self.modules_tracked = np.zeros((0, 2), dtype=np.float32)
            self.detection_ids_tracked = []
            for d in range(len(detected_modules)):
                self.module_ids.append(uuid.uuid4())
                self.modules_tracked = np.vstack(
                    (self.modules_tracked, detected_modules[d]))
                self.detection_ids_tracked.append(d)

        # motion could be predicted between frames
        else:
            matches_idx, unmatched_tracked_modules_idx, \
                unmatched_detected_modules_idx = match_modules(
                    t_points=self.modules_tracked, d_points=detected_modules, 
                    max_distance=self.max_distance)

            # handle matches
            for d, t in matches_idx:
                self.modules_tracked[t] = detected_modules[d]
                self.detection_ids_tracked[t] = d

            # handle unmatched detected modules
            for d in unmatched_detected_modules_idx:
                self.module_ids.append(uuid.uuid4())
                self.modules_tracked = np.vstack(
                    (self.modules_tracked, detected_modules[d]))
                self.detection_ids_tracked.append(d)

            # handle undetected tracked modules
            for t in unmatched_tracked_modules_idx:
                self.modules_tracked = np.delete(self.modules_tracked, t, axis=0)
                self.module_ids.pop(t)
                self.detection_ids_tracked.pop(t)

        # visualize tracked modules
        for module_id, module_tracked in zip(
                self.module_ids, self.modules_tracked):
            text = str(module_id)[:4]
            pos = (int(module_tracked[0]), int(module_tracked[1]))
            vis_frame = cv2.putText(
                vis_frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        self.previous_frame = frame
        self.frame_idx += 1

        return vis_frame


    def get_tracks(self):
        """Returns current status of the tracker.

        Returns:
            module_ids (`list` of `UUID.uuid4`): Unique ID for each currently
                tracked module.

            detection_ids_tracked (`list` of `int`): The index of the
                corresponding module in the `detected_modules` array which was
                passed into the `step` method.

            modules_tracked (`numpy.ndarray`): The center coordinates of each
                tracked module of shape `(N, 2)` where `N` is the number of
                currently tracked modules. The ith row corresponds to the ith
                list entry in `module_ids` and `detection_ids_tracked`.
        """
        return self.module_ids, self.detection_ids_tracked, self.modules_tracked


def run(frames_root, inference_root, output_dir, to_celsius, motion_model, orb_nfeatures, 
        orb_fast_thres, orb_scale_factor, orb_nlevels, match_distance_thres, 
        max_distance, output_video_fps):
    delete_output(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # load frames & masks
    frame_files = sorted(glob.glob(os.path.join(frames_root, "*.tiff")))
    mask_dirs = sorted(get_immediate_subdirectories(os.path.join(
        inference_root, "masks")))
    mask_files = [sorted(glob.glob(os.path.join(
        inference_root, "masks", r, "*.png"))) for r in mask_dirs]

    cap = Capture(frame_files, mask_files, to_celsius=to_celsius)
    tracker = Tracker(
        motion_model, 
        orb_nfeatures, 
        orb_fast_thres, 
        orb_scale_factor, 
        orb_nlevels, 
        match_distance_thres, 
        max_distance)

    lost_track = False

    # video writer
    video_shape = (cap.img_w, cap.img_h)
    video_path = os.path.join(output_dir, "tracks_preview.avi")
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    videowriter = cv2.VideoWriter(video_path, fourcc, output_video_fps, video_shape)

    # tracking file output
    tracks_file = os.path.join(output_dir, 'tracks.csv')
    with open(tracks_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')

        pbar = tqdm(total=len(frame_files))
        while True:
            frame, masks, frame_name, mask_names = cap.get_next_frame(
                preprocess=True)
            if frame is None:
                break

            vis_frame = np.copy(frame)
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)

            mask_centers = []
            mask_contours = []
            for mask in masks:
                convex_hull, contour = contour_and_convex_hull(mask)
                center = compute_mask_center(convex_hull, contour, method=1)
                mask_centers.append(center)
                mask_contours.append(convex_hull)

            pbar.update(1)

            if len(mask_centers) == 0:
                lost_track = True
                continue

            if lost_track:
                tracker = Tracker(
                    motion_model, 
                    orb_nfeatures, 
                    orb_fast_thres, 
                    orb_scale_factor, 
                    orb_nlevels, 
                    match_distance_thres, 
                    max_distance)
                lost_track = False

            mask_centers = np.vstack(mask_centers).reshape(-1, 2)

            vis_frame = tracker.step(
                frame, detected_modules=mask_centers, vis_frame=vis_frame)
            module_ids, detection_ids_tracked, modules_tracked = \
                tracker.get_tracks()

            for module_id, detection_id_tracked, module_tracked in zip(
                    module_ids, detection_ids_tracked, modules_tracked):
                csvwriter.writerow([
                    frame_name,
                    mask_names[detection_id_tracked],
                    module_id,
                    module_tracked[0],
                    module_tracked[1]])

            videowriter.write(vis_frame)

        pbar.close()
        videowriter.release()
