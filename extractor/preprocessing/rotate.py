"""Batch rotates video frames

In case the input video is rotated use this module to rotate the frames.
Possible rotation values provided for the "video_rotation" attribute in the run
method are "90_deg_cw", "180_deg_cw" and "270_deg_cw". Files are rotated in
place, so apply only once.
"""

import os
import glob
import shutil
import cv2
from tqdm import tqdm


def rotate(frame_file, rotation):
    frame = cv2.imread(frame_file, cv2.IMREAD_ANYDEPTH)
    if rotation == "90_deg_cw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "180_deg_cw":
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == "270_deg_cw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(frame_file, frame)


def run(frames_root, rotation):
    # abort if rotation has been performed already
    if os.path.isfile(os.path.join(frames_root, ".rotated")):
        return

    # copy original frame folders
    shutil.copytree(os.path.join(frames_root, "radiometric"), os.path.join(
        frames_root, "radiometric_orig"))

    frame_files_radiometric = sorted(
        glob.glob(os.path.join(frames_root, "radiometric", "*")))

    total = len(frame_files_radiometric)
    pbar = tqdm(total=total)
    for frame_file_radiometric in frame_files_radiometric:
        rotate(frame_file_radiometric, rotation)
        pbar.update(1)
    pbar.close()

    # create a hidden file to indicate that rotation was performed sucessfully
    with open(os.path.join(frames_root, ".rotated"), "w") as file:
        file.write('')
