"""Splits a multipage TIFF file of a FLIR IR camera into individual frames."""

import glob
import os
import csv
import json
import logging
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm
import tifffile

from extractor.common import delete_output


logger = logging.getLogger(__name__)


def to_decimal_degrees(degrees_n, degrees_d, minutes_n, minutes_d,
    seconds_n, seconds_d):
    """Converts degrees, minutes and seconds into decimal degrees."""
    degrees = degrees_n / degrees_d
    minutes = minutes_n / minutes_d
    seconds = seconds_n / seconds_d
    deg_loc = degrees + (minutes/60) + (seconds/3600)
    return deg_loc


def exif_gps_to_degrees(gps_info):
    """Transforms EXIF GPS info into degrees north and east.

    Positive values correspond to north and east whereas negative
    values correspond to south and west.
    """
    latitude_deg = to_decimal_degrees(*gps_info['GPSLatitude'])
    longitude_deg = to_decimal_degrees(*gps_info['GPSLongitude'])
    if gps_info['GPSLatitudeRef'] == "S":
        latitude_deg *= -1.0
    if gps_info['GPSLongitudeRef'] == "W":
        longitude_deg *= -1.0
    return latitude_deg, longitude_deg


def get_gps_altitude(gps_info):
    """Obtain decimal representation of GPS altitude. Returns 0.0
    if EXIF GPS altitude tag is not available."""
    try:
        altitude = gps_info['GPSAltitude']
    except KeyError:
        altitude = 0.0
    else:
        altitude = altitude[0] / altitude[1]
    return altitude


def get_num_rgb_frames(rgb_files):
    """Return the number of frames in the provided videos.

    Args:
        rgb_files (`list` of `str`): List of video files names.

    Returns:
        Number of video frames (`int`).
    """
    n_rgb = 0
    for rgb_file in rgb_files:
        cap = cap = cv2.VideoCapture(rgb_file)
        n_rgb += cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return int(n_rgb)


def get_num_ir_frames(tiff_files):
    """Return the number of frames in the provided videos (TIFF stacks)."""
    n_ir = 0
    for tiff_file in tiff_files:
        with tifffile.TiffFile(tiff_file) as tif:
            for page in tif.pages:
                n_ir += 1
    return int(n_ir)


def get_ir_frame_number(rgb_idx, n_ir, n_rgb):
    """Returns index of IR frame corresponding to the RGB frame idx."""
    ir_idx = round(n_ir*float(rgb_idx)/n_rgb)
    return ir_idx


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


def run(video_dir, output_dir, ir_file_extension=None, rgb_file_extension=None,
extract_timestamps=True,
    extract_gps=True, extract_gps_altitude=False, sync_rgb=True,
    rotate_frames=None):

    delete_output(output_dir)
    for dirname in ["radiometric", "gps"]:
        os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)

    tiff_files = sorted(glob.glob(os.path.join(video_dir, "*.{}".format(ir_file_extension))))
    n_ir = get_num_ir_frames(tiff_files)
    logger.info("Found {} TIFF videos with {} frames for splitting".format(
        len(tiff_files), n_ir))

    if sync_rgb:
        os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
        rgb_files = sorted(glob.glob(os.path.join(video_dir, "*.{}".format(rgb_file_extension))))
        n_rgb = get_num_rgb_frames(rgb_files)
        assert get_ir_frame_number(n_rgb, n_ir, n_rgb) == n_ir
        logger.info("Found {} RGB videos with {} frames for splitting".format(
            len(rgb_files), n_rgb))

    frame_idx = 0
    gps = []
    timestamps = []
    for i, tiff_file in enumerate(tiff_files):
        logger.info("Splitting TIFF file {} of {}".format(i+1, len(tiff_files)))

        with tifffile.TiffFile(tiff_file) as tif:
            for page in tqdm(tif.pages, total=len(tif.pages)):
                image_radiometric = page.asarray().astype(np.uint16)  # BUG: for Optris camera conversion to int is detrimental to accuracy as raw values in TIFF are float
                radiometric_file = os.path.join(
                    output_dir, "radiometric", "frame_{:06d}.tiff".format(
                    frame_idx))
                if rotate_frames:
                    image_radiometric = rotate(image_radiometric, rotate_frames)
                cv2.imwrite(radiometric_file, image_radiometric)

                if extract_timestamps:
                    try:
                        timestamp = page.tags["ExifTag"].value['DateTimeOriginal']
                        timestamp_subsec = page.tags["ExifTag"].value['SubsecTimeOriginal']
                    except KeyError:
                        pass
                    else:
                        timestamp = "{}.{:06.0f}".format(timestamp, float(timestamp_subsec)*10000)
                        timestamp_iso = datetime.strptime(
                            timestamp, '%Y:%m:%d %H:%M:%S.%f').isoformat(timespec='microseconds')
                        timestamps.append(timestamp_iso)


                # extract GPS position
                if extract_gps:
                    try:
                        gps_info = page.tags["GPSTag"].value
                    except KeyError:
                        pass
                    else:
                        deg_lat, deg_long = exif_gps_to_degrees(gps_info)
                        
                        if extract_gps_altitude:
                            alt = get_gps_altitude(gps_info)
                            gps.append((deg_long, deg_lat, alt))
                        else:
                            gps.append((deg_long, deg_lat))

                frame_idx += 1

    # synchronize RGB videos
    if sync_rgb:
        rgb_frame_idx = 0
        last_frame_idx = None
        for i, rgb_file in enumerate(rgb_files):
            logger.info("Splitting RGB file {} of {}".format(i+1, len(rgb_files)))
            cap = cv2.VideoCapture(rgb_file)
            for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
                res, frame = cap.read()
                if res:
                    frame_idx = get_ir_frame_number(rgb_frame_idx, n_ir, n_rgb)
                    if last_frame_idx is None or frame_idx != last_frame_idx:
                        out_path = os.path.join(output_dir,
                            "rgb", "frame_{:06d}.jpg".format(frame_idx))
                        if rotate_frames:
                            frame = rotate(frame, rotate_frames)
                        cv2.imwrite(
                            out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    last_frame_idx = frame_idx

                    rgb_frame_idx += 1

    # store extracted timestamps to disk
    if extract_timestamps and len(timestamps) > 0:
        with open(os.path.join(
                output_dir, "timestamps.csv"), "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            for timestamp in timestamps:
                writer.writerow([timestamp])

    # store extracted GPS positions to disk
    if extract_gps and len(gps) > 0:

        # save GPS trajectory to JSON
        json.dump(gps, open(os.path.join(
            output_dir, "gps", "gps.json"), "w"))
