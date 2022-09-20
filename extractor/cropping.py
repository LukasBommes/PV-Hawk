"""Crops and rectifies PV modules from IR frames.

This module is used to crop out and rectify an image patch for every segmented
PV module in an IR video frame. The outlines of each module are based on
the quadrilaterals, which are computed from the segmentation mask of each module
in the `quadrilaterals` module.

You can run the cropping either directely after having estimated the module 
quadrilaterals or later after completion of the mapping step. If you run cropping
after mapping, then only a single patch directory will be created for multiple
modules in case they were merged during the mapping. If instead you run cropping before 
mapping, a separate patch directory will be created for each of those merged modules.

The procedure for creating a single patch is as follows:

1) Load quadrilateral of each PV module
2) Compute the maximum width and height of the quadrilateral
3) Compute a homography from the corners of the quadrilateral to a rectangular
   destination image with the previously computed width and height
4) Project the image region inside the quadrilateral onto the rectangular
   destination image using the computed homography
5) If the width is larger than the height, rotate the rectangular patch by 90
   degrees counter-clockwise
"""

import os
import glob
import re
import pickle
from tqdm import tqdm
import numpy as np
import cv2

from extractor.common import Capture, delete_output, sort_cw


def clip_to_image_region(quadrilateral, image_width, image_height):
    """Clips quadrilateral to image region.
    Note: Quadrilateral is modified in-place.
    """
    assert image_width > 0 and image_height > 0
    assert quadrilateral.shape == (4, 1, 2)
    quadrilateral[:, 0, 0] = np.clip(
        quadrilateral[:, 0, 0], 0, image_width - 1)
    quadrilateral[:, 0, 1] = np.clip(
        quadrilateral[:, 0, 1], 0, image_height - 1)
    return quadrilateral


def crop_module(frame, quadrilateral, crop_width=None, crop_aspect=None,
    rotate_mode="portrait"):
    """Crops out and rectifies image patch of single PV module in a given frame.

    Args:
        frame (`numpy.ndarray`): 1- or 3-channel frame (visual or IR) from which
            the module patch will be cropped out.

        quadrilaterals (`numpy.ndarray`): Shape (4, 1, 2). The four corner
            points of the module in the frame which were obtained by
            `find_enclosing_polygon` with `num_vertices = 4`.

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
    assert frame.ndim == 2 or frame.ndim == 3
    assert quadrilateral.shape == (4, 1, 2)
    assert crop_width is None or crop_width > 0
    assert crop_aspect is None or crop_aspect > 0
    assert rotate_mode is None or rotate_mode in ["portrait", "landscape"]

    quadrilateral = clip_to_image_region(
        quadrilateral, frame.shape[1], frame.shape[0])
    tl, tr, br, bl = sort_cw(quadrilateral.reshape(-1, 2)).tolist()

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


def build_merged_index(merged_modules, quadrilaterals):
    """Creates a mapping (dict) of the form {module_id: merged_module_id}.
    Here, `module_id` is the original module ID as in the quadrilaterals dict. The
    `merged_module_id` corresponds to the module ID of the first module the module
    is merged with. If the module is not merged with another module,
    `merged_module_id` equals `module_id`."""
    module_ids = set([module_id for module_id, _, _ in quadrilaterals.keys()])
    if merged_modules is None:
        merged_index = {module_id: module_id for module_id in module_ids}
    else:
        merged_index = {}
        for module_id in module_ids:    
            merged_id = module_id
            for modules in merged_modules:
                if module_id in modules:
                    merged_id = modules[0]            
            merged_index[module_id] = merged_id
    return merged_index


def run(frames_root, quads_root, mapping_root, output_dir, ir_or_rgb, rotate_mode):
    delete_output(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # load frames & masks
    if ir_or_rgb == "ir":
        frame_files = sorted(glob.glob(os.path.join(frames_root, "radiometric", "*.tiff")))
    else:
        frame_files = sorted(glob.glob(os.path.join(frames_root, "rgb", "*.jpg")))

    # load module corners
    quadrilaterals = pickle.load(open(os.path.join(quads_root, "quadrilaterals.pkl"), "rb"))

    # handle modules merged during 3D reconstruction
    try:
        merged_modules = pickle.load(open(os.path.join(mapping_root, "merged_modules.pkl"), "rb"))
    except FileNotFoundError:
        merged_modules = None

    merged_index = build_merged_index(merged_modules, quadrilaterals)

    cap = Capture(frame_files, ir_or_rgb, mask_files=None)

    pbar = tqdm(total=len(quadrilaterals.keys()))
    for module_id, frame_name, mask_name in quadrilaterals.keys():

        # get frame
        frame_idx = int(re.findall(r'\d+', frame_name)[0])
        frame, _, frame_name_loaded, _ = cap.get_frame(frame_idx, preprocess=False)

        assert frame_name == frame_name_loaded, "something wrong with frame name"  # just for debugging

        # get quadrilaterals
        quad = quadrilaterals[(module_id, frame_name, mask_name)]["quadrilateral"]
        quad = np.array(quad).reshape(-1, 1, 2)

        # crop and rectify module patch from frame
        module_patch, _ = crop_module(
            frame, quad, rotate_mode=rotate_mode)

        # consider possible mergers with other modules
        save_module_id = merged_index[module_id]

        # save to disk
        patch_dir = os.path.join(output_dir, "radiometric", save_module_id)
        os.makedirs(patch_dir, exist_ok=True)
        if ir_or_rgb == "ir":
            patch_file = os.path.join(
                patch_dir, "{}_{}.tiff".format(frame_name, mask_name))
        else:
            patch_file = os.path.join(
                patch_dir, "{}_{}.jpg".format(frame_name, mask_name))
        cv2.imwrite(patch_file, module_patch)

        pbar.update(1)
    pbar.close()