"""Perform some preprocessing of the frames, such as subsampling every Nth
frame, resizing, or rotating."""

import os
import logging
import cv2

from extractor.common import delete_output


logger = logging.getLogger(__name__)


def rotate(frame, rotation):
    if rotation == "90_deg_cw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "180_deg_cw":
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == "270_deg_cw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Invalid value for rotation")
    return frame


def run(frames_root, output_dir, subsample=None, resize=None, rotate=None):

    # for no preprocessing create a symlink to save disk space
    if not subsample and not resize and not rotate:
        src = os.path.join(frames_root, "images")
        dst = os.path.join(output_dir, "images")

        try:
            os.unlink(dst)
        except OSError:
            pass

        os.symlink(src, dst)
        return

    delete_output(output_dir)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # load images...
    #image

    # subsample
    if subsample:
        # TODO: implement
        pass

    # resize
    if resize:
        # TODO: implement
        pass

    # rotate
    if rotate:
        image = rotate(image, rotate)
