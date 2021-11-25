"""Splits a multipage TIFF file of a FLIR IR camera into individual frames.

The output directory contain the following directories:
 - `preview`: 8-bit grayscale preview frames (JPG) of the IR video
 - `radiometric`: 16-bit grayscale radiometric frames (TIFF) of the IR video
"""

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
import simplekml

from extractor.common import delete_output
from extractor.gps import gps_to_ltp, gps_from_ltp, interpolate_gps


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


def create_preview(image_radiometric):
    temp_range = image_radiometric.max() - image_radiometric.min()
    image_preview = (image_radiometric - image_radiometric.min()) / temp_range
    image_preview = (255*image_preview).astype(np.uint8)
    return image_preview


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


def run(input, output_dir, input_rgb=None, extract_timestamps=True,
    extract_gps=True, extract_gps_altitude=False, sync_rgb=True):

    delete_output(output_dir)
    for dirname in ["radiometric", "preview", "gps"]:
        os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)

    tiff_files = sorted(glob.glob(input))
    n_ir = get_num_ir_frames(tiff_files)
    logger.info("Found {} TIFF videos with {} frames for splitting".format(
        len(tiff_files), n_ir))

    if sync_rgb and (input_rgb is not None):
        os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
        rgb_files = sorted(glob.glob(input_rgb))
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
                image_preview = create_preview(image_radiometric)
                radiometric_file = os.path.join(
                    output_dir, "radiometric", "frame_{:06d}.tiff".format(
                    frame_idx))
                preview_file = os.path.join(
                    output_dir, "preview", "frame_{:06d}.jpg".format(
                    frame_idx))
                cv2.imwrite(radiometric_file, image_radiometric)
                cv2.imwrite(preview_file, image_preview)

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
    if sync_rgb and (input_rgb is not None):
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

        # interpolate GPS trajectory
        gps = np.array(gps)
        if gps.shape[-1] == 2:
            gps = np.insert(
                gps, 2, np.zeros(len(gps)), axis=1)
        gps, origin = gps_to_ltp(gps)
        gps = interpolate_gps(gps)
        gps = gps_from_ltp(gps, origin)
        gps = gps.tolist()

        # save GPS trajectory to CSV
        with open(os.path.join(
                output_dir, "gps", "gps.csv"), "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            for row in gps:
                writer.writerow(row)

        # save GPS trajectory to JSON
        json.dump(gps, open(os.path.join(
            output_dir, "gps", "gps.json"), "w"))

        # save GPS trajectory to KML file
        kml_file = simplekml.Kml()
        kml_file.newlinestring(name="trajectory", coords=gps)
        kml_file.save(os.path.join(output_dir, "gps", "gps.kml"))
