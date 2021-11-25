import math 
import cv2
import numpy as np


def to_twist(R, t):
    """Convert a 3x3 rotation matrix and translation vector (shape (3,))
    into a 6D twist coordinate (shape (6,))."""
    r, _ = cv2.Rodrigues(R)
    twist = np.zeros((6,))
    twist[:3] = r.reshape(3,)
    twist[3:] = t.reshape(3,)
    return twist


def from_twist(twist):
    """Convert a 6D twist coordinate (shape (6,)) into a 3x3 rotation matrix
    and translation vector (shape (3,))."""
    r = twist[:3].reshape(3, 1)
    t = twist[3:].reshape(3, 1)
    R, _ = cv2.Rodrigues(r)
    return R, t


def pixel_bearings(pts, camera_rotation, camera_matrix):
    """Returns the bearing vectors for given image points.
    
    Args:
        pts (`numpy.ndarray`): Image points (u, v) in pixel coordinates. 
            Shape should be either (-1, 2) or (-1, 1, 2).
        
        rvec (`numpy.ndarray`): Camera rotations matrix (see cv2.Rodrigues).
            It is the rotation which transforms points from image coordinates
            into world coordinates. Note: This is inverse of the OpenCV
            convention in which the transformation maps points from world
            to camera coordinates.
        
        camera_matrix (`numpy.ndarray`): OpenCv camera matrix in pixel units.
        
    Returns:
        bearings (`numpy.ndarray`): Bearing vector for each point. Array of 
            shape (-1, 3).
    """
    # compute bearing vectors    
    c = np.array([camera_matrix[0, 2], camera_matrix[1, 2]])
    f = np.array([camera_matrix[0, 0], camera_matrix[1, 1]])
    
    # compute bearing for each point
    pts = pts.reshape(-1, 2)
    b = np.ones((pts.shape[0], 3))
    b[:, :2] = (pts - c) / f
    b /= np.linalg.norm(b, axis=1).reshape(-1, 1)
    
    # rotate bearings by camera rotation
    b = np.matmul(camera_rotation, b.T).T
    return b


def angle_between_vectors(u, v):
    """Returns the angle between two vectors of vectors in radians.
    
    Args:
        u (`numpy.ndarray`): Shape (-1, 3). Each row is a separate
            3D-vector.
            
        v (`numpy.ndarray`): Shape (-1, 3). Each row is a separate
            3D-vector.
            
    Returns:
        angles (`numpy.ndarray`): Shape (-1,). The ith entry corresponds
            to the angle (in radians) between the ith row in u and v.
    """
    u = u.reshape(-1, 3)
    v = v.reshape(-1, 3)
    s1 = np.diagonal(np.matmul(u, u.T))
    s2 = np.diagonal(np.matmul(v, v.T))
    c = np.diagonal(np.matmul(u, v.T)) / np.sqrt(s1 * s2)    
    c[np.abs(c) >= 1.0] = 0.0
    c[np.abs(c) < 1.0] = np.arccos(c)
    return c


def triangulate_points(pts1, pts2, R1, t1, R2, t2, camera_matrix1, camera_matrix2,
        reproj_thres=5.0, min_ray_angle_degrees=1.0):
    """Triangulate 3D map points from corresponding points in two
    keyframes. R1, t1, R2, t2 are the rotation and translation of
    the two key frames w.r.t. to the map origin. Also check for
    the reprojection error in both frames.
    """
    # if any ray angle is smaller than threshold, mark triangulation as failed
    if min_ray_angle_degrees is not None:
        b1 = pixel_bearings(pts1, R1, camera_matrix1)
        b2 = pixel_bearings(pts2, R2, camera_matrix2)
        angles = angle_between_vectors(b1, b2) * 180.0/math.pi
        if np.sum(angles < min_ray_angle_degrees):
            return False, np.empty(shape=(0, 3), dtype=np.float64)
    
    # create projection matrices needed for triangulation of 3D points
    proj_matrix1 = np.hstack([R1.T, -R1.T.dot(t1)])
    proj_matrix2 = np.hstack([R2.T, -R2.T.dot(t2)])
    proj_matrix1 = camera_matrix1.dot(proj_matrix1)
    proj_matrix2 = camera_matrix2.dot(proj_matrix2)

    # triangulate new map points based on matches with previous key frame
    pts_3d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, pts1.reshape(-1, 2).T, pts2.reshape(-1, 2).T).T
    pts_3d = cv2.convertPointsFromHomogeneous(pts_3d).reshape(-1, 3)
    
    # if any reprojection error exceeds threshold, mark triangulation as failed
    if reproj_thres is not None:
        for R, t, pts, camera_matrix in [(R1, t1, pts1, camera_matrix1), 
                                         (R2, t2, pts2, camera_matrix2)]:     
            rvec, _ = cv2.Rodrigues(R.T)
            tvec = -R.T.dot(t)
            reproj_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, None)
            reproj_err = np.linalg.norm(reproj_pts - pts, axis=2)

            if np.sum(reproj_err > reproj_thres):
                return False, np.empty(shape=(0, 3), dtype=np.float64)
    
    return True, pts_3d


def get_visible_modules(pose_graph, frame_name, module_corners, 
    image_width, image_height, max_depth=-1):
    """Obtain track IDs, 3D corner points and reprojected 2D points of all modules 
    visible in the specified key frame. Project only points closer to the camera center
    than `max_depth` (in meters)."""
    visible_modules = []

    # project all modules into key frame
    R, t = from_twist(pose_graph.nodes[frame_name]["pose"])
    camera_matrix = pose_graph.nodes[frame_name]["camera_matrix"]
    for track_id, pts_3d in module_corners.items():

        # omit modules which are too far away from camera
        if max_depth > 0:
            depths = np.linalg.norm(t.reshape(3,) - pts_3d, axis=1)
            if any(depths > max_depth):
                continue

        rvec, _ = cv2.Rodrigues(R.T)
        tvec = -R.T.dot(t)
        reproj_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, None)
        reproj_pts = reproj_pts.reshape(-1, 2)
        
        # determine if module is fully visible
        if (all(reproj_pts[:, 0] >= 0.0) 
            and all(reproj_pts[:, 0] < image_width) 
            and all(reproj_pts[:, 1] >= 0.0) 
            and all(reproj_pts[:, 1] < image_height)):            
            visible_modules.append((track_id, reproj_pts, pts_3d.reshape(-1, 3)))

    return visible_modules
