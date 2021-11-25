import os
import copy
import pickle
import logging
import numpy as np
import g2o
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from extractor.mapping.geometry import get_visible_modules
from extractor.mapping.common import save_modules, save_modules_geocoords, \
    load_reconstructions, get_image_size


logger = logging.getLogger(__name__)


def get_connected_modules(pose_graph, module_corners, image_width, 
    image_height, merge_threshold_image, merge_threshold_world, 
    max_module_depth, max_num_modules):
    """Determine which module corners are connected. Two corners are
    conected if their projected distance in the image is below 
    the `merge_threshold_image` and 3D world points are closer 
    than `merge_threshold_world`."""
    merged_corners = []
    for frame_name in tqdm(pose_graph.nodes):
        
        # get reprojected points of all visible modules
        visible_modules = get_visible_modules(
            pose_graph, frame_name, module_corners, image_width, image_height, max_module_depth)

        if len(visible_modules) > max_num_modules:
            logger.warning(("Number of modules visible in frame {} is {} and"
                " exceeds threshold of {}. Skipping refinement for this frame.")
                .format(frame_name, len(visible_modules), max_num_modules))
            continue
        
        track_ids = []
        all_reproj_pts = []
        all_pts_3d = []
        for track_id, reproj_pts, pts_3d in visible_modules:
            track_ids.extend([track_id for _ in range(5)])  # 4 corners + 1 center
            all_reproj_pts.append(reproj_pts)
            all_pts_3d.append(pts_3d)
        
        if len(all_reproj_pts) > 0:
            all_reproj_pts = np.vstack(all_reproj_pts)
            all_pts_3d = np.vstack(all_pts_3d)

            # determine which points are close to another in projected image space and 3D world space
            dist_reproj = squareform(pdist(all_reproj_pts))
            dist_reproj = np.triu(dist_reproj)
            dist_reproj[dist_reproj == 0] = np.inf

            dist_3d = squareform(pdist(all_pts_3d))
            dist_3d = np.triu(dist_3d)
            dist_3d[dist_3d == 0] = np.inf

            idxs = np.where((dist_reproj < merge_threshold_image) & (dist_3d < merge_threshold_world))

            for i, j in zip(idxs[0], idxs[1]):
                merged_corners.append((track_ids[i], track_ids[j], i%5, j%5))
    return merged_corners


def create_lookup_tables(merged_corners):
    """Built lookup table (track_id, pts_idx) -> unique vertex_id (int) and reverse."""
    track_to_vertex_id = {}
    vertex_id_to_track = {}
    for i, (track_id1, track_id2, pts_idx1, pts_idx2) in enumerate(merged_corners):
        if (track_id1, pts_idx1) not in track_to_vertex_id.keys():
            track_to_vertex_id[(track_id1, pts_idx1)] = 2*i   
            vertex_id_to_track[2*i] = (track_id1, pts_idx1)
            
        if (track_id2, pts_idx2) not in track_to_vertex_id.keys():
            track_to_vertex_id[(track_id2, pts_idx2)] = 2*i + 1    
            vertex_id_to_track[2*i+1] = (track_id2, pts_idx2)
    return track_to_vertex_id, vertex_id_to_track


def run(mapping_root, merge_threshold_image=20, merge_threshold_world=1, 
    max_module_depth=-1, max_num_modules=300, optimizer_steps=10):
    
    pose_graph = pickle.load(open(os.path.join(mapping_root, "pose_graph.pkl"), "rb"))
    module_corners = pickle.load(open(os.path.join(mapping_root, "module_corners.pkl"), "rb"))
    overall_origin = pickle.load(open(os.path.join(mapping_root, "reference_lla.pkl"), "rb"))

    reconstructions = load_reconstructions(mapping_root)
    image_width, image_height = get_image_size(reconstructions)

    logger.info("Obtaining connected module corners")
    merged_corners = get_connected_modules(
        pose_graph, module_corners, image_width, image_height, 
        merge_threshold_image, merge_threshold_world, 
        max_module_depth, max_num_modules)
                
    track_to_vertex_id, vertex_id_to_track = create_lookup_tables(merged_corners)

    # setup optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    # add 3D corner points and edges
    logger.info("Building optimization graph")
    vertex_ids_added = []
    for track_id1, track_id2, pts_idx1, pts_idx2 in tqdm(merged_corners):
        
        vertex_id1 = track_to_vertex_id[(track_id1, pts_idx1)]
        if vertex_id1 not in vertex_ids_added:
            v1 = g2o.VertexPointXYZ()        
            v1.set_id(vertex_id1)
            pts_3d1 = module_corners[track_id1][pts_idx1]
            v1.set_estimate(pts_3d1)
            v1.set_fixed(False)
            optimizer.add_vertex(v1)
            vertex_ids_added.append(vertex_id1)
        
        vertex_id2 = track_to_vertex_id[(track_id2, pts_idx2)]
        if vertex_id2 not in vertex_ids_added:
            v2 = g2o.VertexPointXYZ()
            v2.set_id(vertex_id2)
            pts_3d2 = module_corners[track_id2][pts_idx2]
            v2.set_estimate(pts_3d2)
            v2.set_fixed(False)
            optimizer.add_vertex(v2)
            vertex_ids_added.append(vertex_id2)
        
        # add edge
        edge = g2o.EdgePointXYZ()
        edge.set_vertex(0, optimizer.vertex(vertex_id1))
        edge.set_vertex(1, optimizer.vertex(vertex_id2))
        edge.set_measurement(np.array([0, 0, 0]))
        edge.set_information(np.identity(3)) # TODO: give lower confidence to camera z coordinate, not easy as world z != camer z
        edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.99)))
        optimizer.add_edge(edge)
        
    logger.info("Number of vertices: {}".format(len(optimizer.vertices())))
    logger.info("Number of edges: {}".format(len(optimizer.edges())))

    logger.info("Running optimization")
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(optimizer_steps)

    # write back result
    logger.info("Saving refined georeferenced modules")
    module_corners_estimated = copy.deepcopy(module_corners)
    for vertex_id, vertex in optimizer.vertices().items():
        pts_estimated = vertex.estimate()
        track_id, pts_idx = vertex_id_to_track[vertex_id]
        module_corners_estimated[track_id][pts_idx] = pts_estimated

    # move module center points into the center of the four corners
    for track_id in module_corners_estimated.keys():
        module_corners_estimated[track_id][0, :] = np.mean(module_corners_estimated[track_id][1:, :], axis=0)

    save_modules(module_corners_estimated, open(
        os.path.join(mapping_root, "modules_refined.pkl"), "wb"))
    save_modules_geocoords(module_corners_estimated, overall_origin, 
        open(os.path.join(mapping_root, "module_geolocations_refined.geojson"), "w"))