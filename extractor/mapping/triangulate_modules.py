import os
import random
import pickle
import logging
from itertools import combinations
from collections import defaultdict
import numpy as np
import cv2
import networkx as nx
from tqdm import tqdm

from extractor.geotransforms import enu2geodetic, geodetic2enu
from extractor.mapping.geometry import from_twist, triangulate_points, get_visible_modules
from extractor.mapping.common import load_tracks, save_modules, save_modules_geocoords, \
    load_reconstructions, get_image_size


logger = logging.getLogger(__name__)


def convert_camera_matrix(reconstruction, camera_model="brown"):
    """Convert camera matrix from OpenSfM format to OpenCV format 
    (pixel units). Needed because OpenSfM modifies camera parameters 
    during reconstruction."""
    
    if camera_model == "brown":
        camera = list(reconstruction['cameras'].keys())[0]
        image_width = reconstruction['cameras'][camera]['width']
        image_height = reconstruction['cameras'][camera]['height']
        scale = np.maximum(image_width, image_height)
        fx = reconstruction['cameras'][camera]['focal_x'] * scale
        fy = reconstruction['cameras'][camera]['focal_y'] * scale
        cx = reconstruction['cameras'][camera]['c_x'] * scale + 0.5*image_width
        cy = reconstruction['cameras'][camera]['c_y'] * scale + 0.5*image_height

        camera_matrix = np.array([[fx, 0.0, cx],
                                  [0.0, fy, cy],
                                  [0.0, 0.0, 1.0]])

        k1 = reconstruction['cameras'][camera]['k1']
        k2 = reconstruction['cameras'][camera]['k2']
        p1 = reconstruction['cameras'][camera]['p1']
        p2 = reconstruction['cameras'][camera]['p2']
        k3 = reconstruction['cameras'][camera]['k3']
        dist_coeffs = np.array([[k1, k2, p1, p2, k3]])
        
    elif camera_model == "perspective":
        dist_coeffs = None
        camera = list(reconstruction['cameras'].keys())[0]
        image_width = reconstruction['cameras'][camera]['width']
        image_height = reconstruction['cameras'][camera]['height']
        scale = np.maximum(image_width, image_height)
        focal = reconstruction['cameras'][camera]['focal'] * scale
        cx = image_width / 2
        cy = image_height / 2
        camera_matrix = np.array([[focal, 0.0, cx],
                                  [0.0, focal, cy],
                                  [0.0, 0.0, 1.0]])
    
    return camera_matrix, dist_coeffs


def equalize_cordinate_references(reconstructions):
    """Equalize coordinate reference points of reconstructions.
    
    When combining multiple reconstructions each one uses its own ECEF 
    origin. To equalize them, we select the origin of the first reconstruction 
    and convert all other reconstructions to this origin.
    """
    overall_origin = tuple(reconstructions[0]["reference_lla"].values())

    for reconstruction in reconstructions:
        origin = tuple(reconstruction["reference_lla"].values())
        if origin == overall_origin:
            continue
        
        # transform map points
        for point_id in reconstruction["points"].keys():
            geo = enu2geodetic(*reconstruction["points"][point_id]["coordinates"], *origin)
            transformed_ecef = geodetic2enu(*geo, *overall_origin)
            reconstruction["points"][point_id]["coordinates"] = transformed_ecef

    for reconstruction in reconstructions:
        origin = tuple(reconstruction["reference_lla"].values())
        if origin == overall_origin:
            continue
        
        # transform camera poses
        for shot_id in reconstruction["shots"].keys():
            shot = reconstruction["shots"][shot_id]
            R, _ = cv2.Rodrigues(np.array(shot["rotation"]))
            t = np.array(shot["translation"])
            t = -R.T.dot(t)        
            geo = enu2geodetic(*t, *origin)
            transformed_ecef = geodetic2enu(*geo, *overall_origin)        
            transformed_ecef = -1.0 * (R @ np.array(transformed_ecef).reshape(3, 1)).reshape(3,)
            reconstruction["shots"][shot_id]["translation"] = transformed_ecef

    return reconstructions, overall_origin


def get_map_points(reconstructions):
    """Retrieve map points from reconstructions and return them
    as a single numpy.ndarray of shape (-1, 3)."""
    map_points = []
    for reconstruction in reconstructions:
        for point in reconstruction['points'].values():
            map_points.append(point['coordinates'])
    map_points = np.vstack(map_points)
    return map_points


def get_camera_poses(reconstructions):
    """Retrieve camera poses from the reconstructions and assemble
    a pose graph containing the camera pose as an attribute. The
    graph nodes are indexed by the frame name, e.g. `frame_000000`.
    Each camera pose is a vector of shape (6,) where the first
    three entries are a rotation vector (see cv2.Rodrigues) and
    the remaining three entries are the translation w.r.t. the 
    origin. The poses describe the translation and rotation of
    the camera in the world coordinate frame. To project points
    from world into camera coordinates use the inverse of this 
    pose.
    """
    pose_graphs = []
    for reconstruction in reconstructions:
        pose_graph = nx.Graph()
        for frame_name in sorted(reconstruction['shots']):
            shot = reconstruction['shots'][frame_name]
            R, _ = cv2.Rodrigues(np.array(shot["rotation"]))
            t = np.array(shot["translation"])
            t = -R.T.dot(t)
            R, _ = cv2.Rodrigues(R.T)
            pose = np.hstack((R.reshape(3,), t))
            frame_name = str.split(frame_name, ".")[0]
            pose_graph.add_node(frame_name, pose=pose)
        pose_graphs.append(pose_graph)        
    pose_graph = nx.compose_all(pose_graphs)
    return pose_graph


def get_camera_parameters(pose_graph, reconstructions):
    """Retrieve camera matrix and distortion coefficients from each of
    the reconstructions and inject them into the pose graph as an
    attribute for each frame."""
    for reconstruction in reconstructions:
        camera_matrix, dist_coeffs = convert_camera_matrix(reconstruction)
        for frame_name in sorted(reconstruction['shots']):
            frame_name = str.split(frame_name, ".")[0]
            pose_graph.nodes[frame_name]["camera_matrix"] = camera_matrix
            pose_graph.nodes[frame_name]["dist_coeffs"] = dist_coeffs
    return pose_graph


def get_module_points(metas, track_id, frame_name, mask_name, 
    camera_matrix=None, dist_coeffs=None):
    """Loads module corners and center points. Undistorts points if 
    dist_coeffs and camera_matrix are not None."""
    try:
        meta = metas[(track_id, frame_name, mask_name)]
    except KeyError:
        points = None
        logger.info("Meta data for module {} not found".format(track_id))
    else:
        center = np.array(meta["center"]).reshape(1, 2).astype(np.float64)
        quadrilateral = np.array(meta["quadrilateral"]).reshape(-1, 2).astype(np.float64)
        points = np.vstack((center, quadrilateral))
        
        if dist_coeffs is not None:
            points = cv2.undistortPoints(
                points, camera_matrix, dist_coeffs, None, camera_matrix)
            
    return points


def triangulate_observations(pose_graph, observations, max_combinations, 
    reproj_thres, min_ray_angle_degrees):
    """Triangulates PV module corners (and centers) by triangulating
    observations from all possible 2-pairs of frames and computing the
    median points in 3D space.
    
    Args:
        pose_graph (`networkx.Graph`): Pose graph containing camera poses
            in the world frame.
        
        observations (`dict` of `dict`): The outer dictionary is indexed by
            the track_id of the module. The inner dictionary is indexed
            by the frame_name (`str`) and contains a `numpy.ndarray` of
            shape (5, 1, 2) of undistorted pixel coordinates of the PV module 
            corners in the respective frame. The first row is the center point, 
            the other rows correspond to the top-left (tl), tr, br and bl points.
    
        max_combinations (`int`): If -1 triangulate points from all
            possible 2-pairs of frames. To limit this number specify the maximum
            number of combinations to try here.
    
    Returns:
        module_corners (`dict`): Triangulated 3D points of each module. Index is 
            the module track_id and value is a `numpy.ndarray` of shape (5, 3).
            The meaning of rows in this array corresponds to the meaning of
            rows in the observations.
    """
    module_corners = {}

    for track_id in tqdm(observations.keys()):
        num_observations = len(observations[track_id])
        frame_names = list(observations[track_id].keys())

        all_combinations = list(combinations(range(num_observations), 2))

        # limit number of combinations to try
        if max_combinations > 0:
            random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_combinations]

        pts_3d = []
        for i, j in all_combinations:

            R1, t1 = from_twist(pose_graph.nodes[frame_names[i]]["pose"])
            R2, t2 = from_twist(pose_graph.nodes[frame_names[j]]["pose"])

            pts1 = observations[track_id][frame_names[i]]
            pts2 = observations[track_id][frame_names[j]]
            
            camera_matrix1 = pose_graph.nodes[frame_names[i]]["camera_matrix"]
            camera_matrix2 = pose_graph.nodes[frame_names[j]]["camera_matrix"]

            valid, pts = triangulate_points(
                pts1, pts2, R1, t1, R2, t2, camera_matrix1, camera_matrix2,
                reproj_thres, min_ray_angle_degrees)
            if valid:
                pts_3d.append(pts)

        if len(pts_3d) > 0:
            pts_3d = np.median(pts_3d, axis=0)
            module_corners[track_id] = pts_3d
    return module_corners


def merge_list_of_dicts(dicts):
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


def triangulate(pose_graph, tracks_file, patches_meta_file, min_track_len, 
    max_combinations, reproj_thres, min_ray_angle_degrees):
    """Triangulate PV module corners and centers."""
    _, tracks_per_id = load_tracks(tracks_file)
    metas = pickle.load(open(patches_meta_file, "rb"))

    # keep only tracks of key frames
    tracks_per_id_filtered = defaultdict(list)
    for track_id, frame_mask_names in tracks_per_id.items():
        for frame_mask_name in frame_mask_names:
            frame_name, mask_name = frame_mask_name
            if frame_name in pose_graph.nodes:
                tracks_per_id_filtered[track_id].append((frame_name, mask_name))
                
    # filter out short tracks
    assert min_track_len >= 2, "min_track_len must be >= 2"
    tracks_per_id_filtered = {
        k: v for k, v in tracks_per_id_filtered.items() 
        if len(v) >= min_track_len}

    # get module observations
    observations = {}
    for track_id, frame_mask_names in tracks_per_id_filtered.items():
        observations[track_id] = {}
        for frame_name, mask_name in frame_mask_names:
            camera_matrix = pose_graph.nodes[frame_name]["camera_matrix"]
            dist_coeffs = pose_graph.nodes[frame_name]["dist_coeffs"]
            points = get_module_points(metas, track_id, 
                frame_name, mask_name, camera_matrix, dist_coeffs)
            if points is not None:
                observations[track_id][frame_name] = points

    module_corners = triangulate_observations(
        pose_graph, observations, max_combinations, reproj_thres, min_ray_angle_degrees)
    return module_corners, observations


def merge_modules(pose_graph, module_corners, observations, 
    image_width, image_height, merge_threshold, max_module_depth, max_num_modules, 
    max_combinations, reproj_thres, min_ray_angle_degrees):
    """Merge duplicate modules by projecting each module into each keyframe 
    and finding overlapping modules."""
    merge_candidates = nx.Graph()
    for frame_name in tqdm(pose_graph.nodes):
        
        visible_modules = get_visible_modules(
            pose_graph, frame_name, module_corners, image_width, image_height, max_module_depth)

        if len(visible_modules) > max_num_modules:
            logger.warning(("Number of modules visible in frame {} is {} and"
                " exceeds threshold of {}. Skipping module merging for this frame.")
                .format(frame_name, len(visible_modules), max_num_modules))
            continue
                
        # find overlapping modules in the image
        all_combinations = list(combinations(range(len(visible_modules)), 2))
        for i, j in all_combinations:
            mean_dist = np.mean(
                np.linalg.norm(visible_modules[i][1] - visible_modules[j][1], axis=1))

            if mean_dist < merge_threshold:
                merge_candidates.add_node(visible_modules[i][0])
                merge_candidates.add_node(visible_modules[j][0])
                merge_candidates.add_edge(visible_modules[i][0], visible_modules[j][0])

    merge_candidates = [
        list(track_ids) for track_ids in nx.connected_components(merge_candidates)]

    # fuse observations of merge candidates into a single track and store with first track ID
    observations_merge_candidates = {}
    for track_ids in merge_candidates:
        obs = [observations[track_id] for track_id in track_ids]
        obs = merge_list_of_dicts(obs)
        observations_merge_candidates[track_ids[0]] = obs
        
    # re-triangulate
    module_corners_updated = triangulate_observations(
        pose_graph, observations_merge_candidates, 
        max_combinations, reproj_thres, min_ray_angle_degrees)
    for track_id, pts_3d in module_corners_updated.items():
        module_corners[track_id] = pts_3d
        
    # since we stored the updated module corners under first track ID, we can delete all other track IDs
    for track_ids in merge_candidates:
        for track_id in track_ids[1:]:
            del module_corners[track_id]
    
    return module_corners, merge_candidates


def run(mapping_root, tracks_root, patches_root, min_track_len=3, 
    merge_overlapping_modules=True, merge_threshold=20, max_module_depth=-1, 
    max_num_modules=300, max_combinations=-1, reproj_thres=5.0, 
    min_ray_angle_degrees=1.0):

    tracks_file = os.path.join(tracks_root, "tracks.csv")
    patches_meta_file = os.path.join(patches_root, "meta.pkl")

    reconstructions = load_reconstructions(mapping_root)
    logger.info("Number of reconstructions: {}".format(len(reconstructions)))

    # get image width and height from first reconstruction (they are all the same)
    image_width, image_height = get_image_size(reconstructions)

    logger.info("Equalizing coordinate references")
    reconstructions, overall_origin = equalize_cordinate_references(reconstructions)
    pickle.dump(overall_origin, open(os.path.join(mapping_root, "reference_lla.pkl"), "wb"))

    # store map points for visualization
    logger.info("Getting map points")
    map_points = get_map_points(reconstructions)
    pickle.dump(map_points, open(os.path.join(mapping_root, "map_points.pkl"), "wb"))

    # store camera poses and camera parameters in pose graph
    logger.info("Getting camera poses and camera parameters")
    pose_graph = get_camera_poses(reconstructions)
    pose_graph = get_camera_parameters(pose_graph, reconstructions)

    logger.info("Triangulating modules")
    module_corners, observations = triangulate(
        pose_graph, tracks_file, patches_meta_file, min_track_len, 
        max_combinations, reproj_thres, min_ray_angle_degrees)

    if merge_overlapping_modules:
        logger.info("Merging overlapping modules")
        module_corners, merge_candidates = merge_modules(
            pose_graph, module_corners, observations, image_width, image_height, 
            merge_threshold, max_module_depth, max_num_modules, max_combinations, 
            reproj_thres, min_ray_angle_degrees)
        pickle.dump(merge_candidates, open(os.path.join(mapping_root, "merged_modules.pkl"), "wb"))

    logger.info("Saving georeferenced modules")
    pickle.dump(pose_graph, open(os.path.join(mapping_root, "pose_graph.pkl"), "wb"))
    pickle.dump(module_corners, open(os.path.join(mapping_root, "module_corners.pkl"), "wb"))

    save_modules(module_corners, open(
        os.path.join(mapping_root, "modules.pkl"), "wb"))
    save_modules_geocoords(module_corners, overall_origin, 
        open(os.path.join(mapping_root, "module_geolocations.geojson"), "w"))


    


