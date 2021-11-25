import numpy as np


def extract_keypoints(frame, orb):
    """Extracts ORB keypoints and descriptors in the frame."""
    kp = orb.detect(frame, None)
    kp, des = orb.compute(frame, kp)
    return kp, des


def match_keypoints(bf_matcher, last_des, des, last_kp, kp, distance_threshold):
    """Matches ORB descriptors between two frames using brute force matcher."""
    matches = bf_matcher.match(last_des, des)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = [m for m in matches if m.distance < distance_threshold]
    last_pts = np.array(
        [last_kp[m.queryIdx].pt for m in matches]).reshape(1, -1, 2)
    current_pts = np.array(
        [kp[m.trainIdx].pt for m in matches]).reshape(1, -1, 2)
    return matches, last_pts, current_pts