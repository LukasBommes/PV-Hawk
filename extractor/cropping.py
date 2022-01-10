"""Crops and rectifies segmented PV modules.

This module is used to crop out and rectify an image patch for every segmented
PV module in an IR video frame. Patches are generated from 16-bit radiometric
TIFFs. The procedure for creating a single patch is as follows:

1) Load binary segmentation mask of the PV module
2) Compute the contour and convex hull of the mask
3) Find the approximately smallest enclosing quadrilateral of the convex hull
4) If the IoU between convex hull and quadrilaterl is lower than a specified
   threshold abort the procedure and do not crop out this PV module
5) Else, sort the corner points of the quadrilateral in CW order
6) Compute the maximum width and height of the quadrilateral
7) Compute a homography from the corners of the quadrilateral to a rectangular
   destination image with the previously computed width and height
8) Project the image region inside the quadrilateral onto the rectangular
   destination image using the computed homography
9) If the width is larger than the height, rotate the rectangular patch by 90
   degrees CCW
"""

import os
import glob
import csv
import pickle
import logging
from tqdm import tqdm
import numpy as np
import cv2

from extractor.common import Capture, delete_output, contour_and_convex_hull, \
    compute_mask_center, get_immediate_subdirectories, \
    line, line_intersection


logger = logging.getLogger(__name__)


def findEnclosingPolygon(convex_hull, num_vertices=4, visu=False):
    """Computes the enclosing k-polygon of a convex shape.

    The algorithm works as follows:

    While number of edges > N do:
        remove the shortest edge by replacing its endpoints with the
        intersection point of the adjacent edges

    Source: https://stackoverflow.com/questions/11602259/find-the-smallest-
            containing-convex-polygon-with-a-given-number-of-points

    Args:
        convex_hull (`numpy.ndarray`): Shape [-1, 0, 2] of dtype int32. The
            convex hull of the polygon as returned by `cv2.convexHull` with
            `clockwise = False` and `returnPoints = True`.

        num_vertices (`int`): Number of vertices of the returned enclosing
            polygon.

    Returns:
        Enclosing polygon (`numpy.ndarray`) of shape [num_vertices, 0, 2] and
        dtype int32.
    """
    hull = np.copy(convex_hull)
    while len(hull) > num_vertices:
        # get shortest edge
        edge_lenghts = []
        for i in range(0, hull.shape[0]-1):
            edge_lenght = np.linalg.norm(
                hull[i, 0, :] - hull[i+1, 0, :], axis=-1)
            edge_lenghts.append(edge_lenght)
        edge_lenghts.append(np.linalg.norm(
            hull[-1, 0, :] - hull[0, 0, :], axis=-1))
        # point in hull where shortest edge starts
        min_edge_idx = np.argmin(edge_lenghts)
        #print("Removing hull point {}".format(min_edge_idx))

        # get indices of hull points which form the two adjacent edges
        n = len(hull)
        previous_edge_idx = ((min_edge_idx-1)%n, min_edge_idx%n)
        subsequent_edge_idx = ((min_edge_idx+1)%n, (min_edge_idx+2)%n)

        # compute intersection between adjacent edges
        edge_previous = line(list(hull[previous_edge_idx[0], 0, :]), list(
            hull[previous_edge_idx[1], 0, :]))
        edge_subsequent = line(list(hull[subsequent_edge_idx[0], 0, :]), list(
            hull[subsequent_edge_idx[1], 0, :]))
        has_intersection, intersection = line_intersection(
            edge_previous, edge_subsequent)
        #print(intersection)

        # if lines do not intersect (they are parallel) use the middle point instead
        if not has_intersection:
            #print("No intersection, using mean")
            intersection = np.mean(np.vstack((hull[previous_edge_idx[1], 0, :],
                hull[subsequent_edge_idx[0], 0, :])), axis=0)
            #print(intersection)

        # replace endpoints of shortest edge with computed intersection
        hull[min_edge_idx, 0, :] = np.array(
            [round(intersection[0]), round(intersection[1])])
        hull = np.delete(hull, (min_edge_idx+1)%len(hull), axis=0)

        # ensure that new hull is convex
        hull_tmp = cv2.convexHull(hull, clockwise=False, returnPoints=True)
        if len(hull_tmp) > num_vertices:
            hull = hull_tmp
    return hull


def compute_iou(convex_hull, quadrilateral):
    """Computes the IoU of the convex hull and
    the estimated bounding quadrilateral."""
    intersect_area, _p12 = cv2.intersectConvexConvex(convex_hull, quadrilateral)
    iou = intersect_area / cv2.contourArea(quadrilateral)
    return iou


def sort_cw(pts):
    """Sort points clockwise by first splitting
    left/right points and then top/bottom."""
    pts = [list(p) for p in pts.reshape(-1, 2)]
    pts_sorted = sorted(pts , key=lambda k: k[0])
    pts_left = pts_sorted[:2]
    pts_right = pts_sorted[2:]
    pts_left_sorted = sorted(pts_left , key=lambda k: k[1])
    pts_right_sorted = sorted(pts_right , key=lambda k: k[1])
    tl = pts_left_sorted[0]
    bl = pts_left_sorted[1]
    tr = pts_right_sorted[0]
    br = pts_right_sorted[1]
    return [tl, tr, br, bl]


def clip_to_image_region(quadrilateral, image_width, image_height):
    """Clips quadrilateral to image region."""
    quadrilateral[:, 0, 0] = np.clip(
        quadrilateral[:, 0, 0], 0, image_width - 1)
    quadrilateral[:, 0, 1] = np.clip(
        quadrilateral[:, 0, 1], 0, image_height - 1)
    return quadrilateral


def crop_single_module(frame, quadrilateral, crop_width=None,crop_aspect=None,
    rotate_mode="portrait"):
    """Crops out and rectifies image patch of single PV module in a given frame.

    Args:
        frame (`numpy.ndarray`): 1- or 3-channel frame (visual or IR) from which
            the module patch will be cropped out.

        quadrilaterals (`numpy.ndarray`): Shape (4, 1, 2). The four corner
            points of the module in the frame which were obtained by
            `findEnclosingPolygon` with `num_vertices = 4`.

        crop_width (`int`): If specified the resulting image patch will have
            this width. Its height is computed based on the provided value of
            `crop_aspect` as `crop_height = crop_width * crop_aspect`. If either
            `crop_width` or `crop_aspect` is set to `None` the width and height
            of the resulting patch correspdond to the maximum width and height
            of the module in the original frame.

        crop_aspect (`float`): The cropping aspect ratio.

        rotate_mode (`str` or `None`): If "portrait" ensures that module height
            is larger than module width by rotating modules with a wrong
            orientation. If "landscape" ensure width is larger than height. If
            `None` do not rotate modules with a potentially wrong orientation.

    Returns:
        Module patch (`numpy.ndarray`): The cropped and rectified patch of the
        module.

        Homography (`numpy.ndarray`): The homography which maps the
        quadrilateral onto a rectangular region.
    """
    quadrilateral = clip_to_image_region(
        quadrilateral, frame.shape[1], frame.shape[0])
    tl, tr, br, bl = sort_cw(quadrilateral.reshape(-1, 2))

    if crop_width is not None and crop_aspect is not None:
        crop_width = int(crop_width)
        crop_height = int(crop_width*crop_aspect)
    else:
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        crop_width = int(max(width_a, width_b))
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        crop_height = int(max(height_a, height_b))

    quadrilateral_sorted = np.array([[tl],[tr],[br],[bl]])
    dst_pts = np.array([[0., 0.],
                        [1., 0.],
                        [1., 1.],
                        [0., 1.]])
    dst_pts[:, 0] *= float(crop_width)
    dst_pts[:, 1] *= float(crop_height)
    homography = cv2.getPerspectiveTransform(
        quadrilateral_sorted.astype(np.float32), dst_pts.astype(np.float32))
    # note: setting border to "replicate" prevents insertion
    # of black pixels which screw up the range of image values
    module_patch = cv2.warpPerspective(
        frame, homography, dsize=(crop_width, crop_height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if rotate_mode == "portrait":
        if module_patch.shape[0] < module_patch.shape[1]:
            module_patch = cv2.rotate(
                module_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate_mode == "landscape":
        if module_patch.shape[1] < module_patch.shape[0]:
            module_patch = cv2.rotate(
                module_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return module_patch, homography


def crop_modules(frame, quadrilaterals, crop_width=None, crop_aspect=None,
    rotate_mode="portrait"):
    """Crops out and rectifies image patch of each PV module in the given frame.

    Args:
        frame (`numpy.ndarray`): 1- or 3-channel frame (visual or IR) from which
            module patches will be cropped out.

        quadrilaterals (`list` of `numpy.ndarray`): Each list item contains the
            four corner points of a module in the frame. Each array should have
            shape (4, 1, 2) and should have been computed by
            `findEnclosingPolygon` with `num_vertices = 4`.

        crop_width (`int`): If specified the resulting image patches will have
            this width. Its height is computed based on the provided value of
            `crop_aspect` as `crop_height = crop_width * crop_aspect`. If either
            `crop_width` or `crop_aspect` is set to `None` the width and height
            of each resulting patch correspdond to the maximum width and height
            of the corresponding module in the original frame.

        crop_aspect (`float`): The cropping aspect ratio.

        rotate_mode (`str` or `None`): If "portrait" ensures that module height
            is larger than module width by rotating modules with a wrong
            orientation. If "landscape" ensure width is larger than height. If
            `None` do not rotate modules with a potentially wrong orientation.

    Returns:
        Module patches (`list` of `numpy.ndarray`): Each list item is a cropped
        and rectified patch of the frame corresponding to the module corner
        points provided in the `quadrilaterals` argument.

        Homographies (`numpy.ndarray`): Each list item is the homography which
        maps the quadrilateral of eachmodule mask onto a rectangular region.
    """
    module_patches = []
    homographies = []
    for quadrilateral in quadrilaterals:
        module_patch, homography = crop_single_module(
            frame, quadrilateral, crop_width, crop_aspect, rotate_mode)
        module_patches.append(module_patch)
        homographies.append(homography)
    return module_patches, homographies


def load_tracks(tracks_file):
    """Load Tracks CSV file."""
    tracks = {}
    with open(tracks_file, newline='', encoding="utf-8-sig") as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.readline(), delimiters=",;")
        csvfile.seek(0)
        csvreader = csv.reader(csvfile, dialect)
        for row in csvreader:
            frame_name, mask_name, track_id, _, _ = row
            tracks[(frame_name, mask_name)] = track_id
    return tracks


def run(frames_root, inference_root, tracks_root, output_dir, min_iou, rotate_mode):

    delete_output(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # load frames & masks
    frame_files_radiometric = sorted(
        glob.glob(os.path.join(frames_root, "radiometric", "*.tiff")))
    mask_dirs = sorted(get_immediate_subdirectories(
        os.path.join(inference_root, "masks")))
    mask_files = [sorted(glob.glob(os.path.join(inference_root, 
        "masks", r, "*.png"))) for r in mask_dirs]

    # load track file
    tracks_file = os.path.join(tracks_root, "tracks.csv")
    tracks = load_tracks(tracks_file)

    cap_radiometric = Capture(frame_files_radiometric, mask_files)

    meta = {}

    pbar = tqdm(total=len(frame_files_radiometric))
    while True:
        frame_radiometric, masks, frame_name, mask_names = \
            cap_radiometric.get_next_frame(preprocess=False)
        if frame_radiometric is None:
            break

        # get minimum enclosing quadrilateral for each mask
        # (and compute mean IoU for all masks in the frame)
        quadrilaterals = []
        centers = []
        mask_names_filtered = []
        for mask, mask_name in zip(masks, mask_names):
            convex_hull, contour = contour_and_convex_hull(mask)
            center = compute_mask_center(convex_hull, contour, method=1)
            quadrilateral = findEnclosingPolygon(convex_hull, num_vertices=4)
            iou = compute_iou(convex_hull, quadrilateral)
            if iou > min_iou:
                quadrilaterals.append(quadrilateral)
                centers.append(center)
                mask_names_filtered.append(mask_name)

        # crop and rectify patches
        module_patches, homographies = crop_modules(
            frame_radiometric, quadrilaterals, rotate_mode=rotate_mode)

        # save patches to disk
        for module_patch, mask_name in zip(
                module_patches, mask_names_filtered):
            # get tracking ID of module
            module_id = tracks[(frame_name, mask_name)]
            patch_dir = os.path.join(output_dir, "radiometric", module_id)
            os.makedirs(patch_dir, exist_ok=True)
            patch_file = os.path.join(
                patch_dir, "{}_{}.tiff".format(frame_name, mask_name))
            cv2.imwrite(patch_file, module_patch)

        # save meta info (homographies, quadrilaterals, module centers)
        for homography, quadrilateral, center, mask_name in zip(
                homographies, quadrilaterals, centers, mask_names_filtered):
            
            module_id = tracks[(frame_name, mask_name)]
            meta[(module_id, frame_name, mask_name)] = {
                "homography": homography,
                "quadrilateral": sort_cw(quadrilateral),
                "center": center,
            }

        pbar.update(1)
    pbar.close()

    # save meta file
    pickle.dump(meta, open(os.path.join(output_dir, "meta.pkl") , "wb"))
