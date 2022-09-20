import os
import glob
import json
import pickle
import yaml
import cv2
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

from extractor.common import Capture, delete_output
from extractor.gps import gps_to_ltp, gps_from_ltp
from extractor.keypoints import extract_keypoints, match_keypoints


def iuo(last_points, points):
    """Returns IoU of two polygons."""
    last_polygon = Polygon(last_points)
    polygon = Polygon(points)
    if not last_polygon.is_valid:
        return None
    if not polygon.is_valid:
        return None
    intersection = last_polygon.intersection(polygon).area
    union = polygon.union(last_polygon).area
    return intersection / union


def frame_overlap(last_pts, current_pts, image_size):    
    """Returns the Intersection over Union of two frames based on matched keypoints."""
    img_h, img_w = image_size
    last_points = np.array([[0, 0],
                            [img_w, 0],
                            [img_w, img_h],
                            [0, img_h]], dtype=np.float32)
    transform, _ = cv2.findHomography(last_pts, current_pts, method=cv2.RANSAC)
    points = cv2.perspectiveTransform(last_points.reshape(-1, 1, 2), transform).reshape(-1, 2)
    iou = iuo(last_points, points)
    return iou


def make_camera_models_file(output_dir, camera_matrix, dist_coeffs, image_width, image_height):
    """Convert camera matrix and distortion coefficients into OpenSfM compatible format.
    Stores the camera_models_overrides file in the OpenSfM dataset."""
    k1, k2, p1, p2, k3 = dist_coeffs[0]

    scale = np.maximum(image_width, image_height)
    c_x = (camera_matrix[0, 2] - 0.5*image_width) / scale
    c_y = (camera_matrix[1, 2] - 0.5*image_height) / scale

    camera_models_overrides = {
        "all": {
            "projection_type": "brown",
            "width": image_width,
            "height": image_height,
            "focal_x": camera_matrix[0][0] / scale,
            "focal_y": camera_matrix[1][1] / scale,
            "c_x": c_x,
            "c_y": c_y,
            "k1": k1,
            "k2": k2,
            "k3": k3, 
            "p1": p1,
            "p2": p2
        }
    }
    
    json.dump(camera_models_overrides, open(
        os.path.join(output_dir, "camera_models_overrides.json"), "w"))


def make_exif_file(output_dir, selected_frames, frame_names, gps, gps_dop, use_altitude):
    """Creates the exif_overrides file  for OpenSfM containing GPS positions of each selected frame."""
    exif_overrides = {}
    for selected_frame, frame_name in zip(selected_frames, frame_names):
        lon, lat, alt = gps[selected_frame]
        if not use_altitude:
            alt = 0.0
        exif_overrides["{}.jpg".format(frame_name)] = {
            "gps": {
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "dop": gps_dop                
            }
        }
    json.dump(exif_overrides, open(os.path.join(output_dir, "exif_overrides.json"), "w"))


def select_frames_gps_visual(cap, gps, orb_detector, bf_matcher, frame_selection_gps_distance, 
    frame_selection_visual_distance, match_distance_thres):
    """Select subset of frames based on travelled GPS distance and frame overlap (IoU).
    Returns indices of selected frames.
    
    Iterates over subsequent frames and computes the distance travelled along the GPS
    trajectory and a visual distance to the last keyframe. If either of the distances
    exceeds the specified threshold, the current frame is selected. The current frame
    is also used as the new keyframe for visual distance computation and the GPS 
    distance is reset to zero. Visual distance is computed as 1.0 - IoU (intersection
    over union). IoU is estimated by finding the homography which maps the last frame
    onto the current frame and then computing the IoU between the frame boundaries of
    the last frame and the transformed current frame.
    """

    # for GPS distance computation
    gps_select = False
    last_pos = np.zeros((2,))
    gps_distance = 0

    # for visual distance computation
    visual_select = False
    last_kps = None
    last_des = None

    frame_idx = 0
    selected_frames = []
    pbar = tqdm(total=cap.num_images)
    while True:
        frame, _, _, _ = cap.get_next_frame(
            preprocess=True, undistort=True, equalize_hist=True)  # Note: undistort should probably be set to False
        if frame is None:
            break

        # preprocessing for RGB frames
        if cap.ir_or_rgb == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.equalizeHist(frame)  # not sure if needed
            
        # test for GPS distance criterion
        pos = gps[frame_idx, :2]
        gps_distance += np.linalg.norm(pos - last_pos)
        last_pos = pos
        if gps_distance > frame_selection_gps_distance:
            gps_select = True
        
        # test for visual distance criterion
        kps, des = extract_keypoints(frame, orb_detector)
        visual_distance = np.inf
        if (last_kps is not None) and (len(kps) > 0):
            _, last_pts, current_pts = match_keypoints(
                bf_matcher, last_des, des, last_kps, kps, 
                distance_threshold=match_distance_thres)
            
            if last_pts.shape[1] > 0 and current_pts.shape[1] > 0:
                
                # estimate motion with homography
                transform, _ = cv2.findHomography(
                    last_pts, current_pts, method=cv2.RANSAC)
                
                if transform is not None:
                    # compute frame overlap from transformed bounding rectangle
                    iou = frame_overlap(last_pts, current_pts, frame.shape)
                    if iou is not None:
                        visual_distance = 1.0 - iou
        
        if visual_distance > frame_selection_visual_distance:
            visual_select = True
        
        if gps_select or visual_select:
            gps_distance = 0 
            last_kps = kps
            last_des = des        
            gps_select = False
            visual_select = False
            selected_frames.append(frame_idx)
            
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    
    return selected_frames


def select_frames_gps(gps, frame_selection_gps_distance):
    """Select subset of frames based on travelled GPS distance. Returns frame indices."""
    last_pos = np.zeros((2,))
    selected_frames = [0]
    travelled_distance = 0
    for i in range(len(gps)):
        pos = gps[i, :2]
        travelled_distance += np.linalg.norm(pos - last_pos)
        last_pos = pos
        if travelled_distance > frame_selection_gps_distance:
            travelled_distance = 0
            selected_frames.append(i)
    return selected_frames


def run(cluster, frames_root, calibration_root, output_dir, opensfm_settings, ir_or_rgb,
        select_frames_mode, frame_selection_gps_distance, frame_selection_visual_distance, 
        orb_nfeatures, orb_fast_thres, orb_scale_factor, orb_nlevels, match_distance_thres,
        gps_dop, output_video_fps):
    cluster_idx = cluster["cluster_idx"]
    frame_cluster = (cluster["frame_idx_start"], cluster["frame_idx_end"])

    output_dir = os.path.join(output_dir, "cluster_{:06d}".format(cluster_idx))
    delete_output(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    camera_matrix = pickle.load(open(os.path.join(calibration_root, ir_or_rgb, "camera_matrix.pkl"), "rb"))
    dist_coeffs = pickle.load(open(os.path.join(calibration_root, ir_or_rgb, "dist_coeffs.pkl"), "rb"))

    # get video frames
    if ir_or_rgb == "ir":
        frame_files = sorted(glob.glob(os.path.join(frames_root, "radiometric", "*.tiff")))
    else:
        frame_files = sorted(glob.glob(os.path.join(frames_root, "rgb", "*.jpg")))
    frame_files = frame_files[frame_cluster[0]:frame_cluster[1]]
    cap = Capture(frame_files, ir_or_rgb, None, camera_matrix, dist_coeffs)

    make_camera_models_file(output_dir, camera_matrix, dist_coeffs, cap.img_w, cap.img_h)

    # write config.yaml for OpenSfM
    yaml.dump(opensfm_settings, open(os.path.join(output_dir, "config.yaml"), 'w'))

    # select frames based on travelled GPS trajectory
    gps = json.load(open(os.path.join(frames_root, "gps", "gps.json"), "r"))
    gps = np.array(gps[frame_cluster[0]:frame_cluster[1]])
    gps[:, 2] -= np.min(gps[:, 2])  # offset so minimum altitude is zero
    gps, origin = gps_to_ltp(gps)

    if select_frames_mode == "gps_visual":
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        orb_detector = cv2.ORB_create(
            nfeatures=orb_nfeatures, fastThreshold=orb_fast_thres, 
            scaleFactor=orb_scale_factor, nlevels=orb_nlevels)
        selected_frames = select_frames_gps_visual(
            cap, gps, orb_detector, bf_matcher, 
            frame_selection_gps_distance, frame_selection_visual_distance,
            match_distance_thres)
    elif select_frames_mode == "gps":
        selected_frames = select_frames_gps(gps, frame_selection_gps_distance)

    gps = gps_from_ltp(gps, origin)

    # store selected frames in OpenSfM dataset and write output video from selected frames for debugging
    video_path = os.path.join(output_dir, "selected_images.avi")
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    videowriter = cv2.VideoWriter(video_path, fourcc, output_video_fps, (cap.img_w, cap.img_h))

    image_path = os.path.join(output_dir, "images")
    os.makedirs(image_path, exist_ok=True)

    frame_names = []
    for selected_frame in selected_frames:
        frame, _, frame_name, _ = cap.get_frame(selected_frame, preprocess=True, undistort=False, equalize_hist=True)  # Note: undistort should probably be set to False
        frame_names.append(frame_name)
        if ir_or_rgb == "ir":
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        videowriter.write(frame)
        cv2.imwrite(os.path.join(output_dir, "images", "{}.jpg".format(frame_name)), frame)

    videowriter.release()

    # Create exif overwrites file
    make_exif_file(output_dir, selected_frames, frame_names, gps, 
        gps_dop, opensfm_settings["use_altitude_tag"])

