"""Splits a multipage TIFF file of a FLIR IR camera into individual frames.

The output directory contain the following directories:
 - `preview`: 8-bit grayscale preview frames (JPG) of the IR video
 - `radiometric`: 16-bit grayscale radiometric frames (TIFF) of the IR video
 - `rgb`: 8-bit color image (JPG) of the visual camera
 - `gps`: CSV, JSON and KML files containing the GPS position in WGS-84 
          coordinate of each frame
"""

import glob
import os
import csv
import json
import logging
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


def map_rgb_frame_idx(rgb_idx, n_ir, n_rgb):
    """Returns index of IR frame corresponding to the RGB frame idx.
    If there are more IR than RGB frames, this is simply a 1:1 mapping."""
    if n_rgb > n_ir:
        ir_idx = round(n_ir*float(rgb_idx)/n_rgb)
        return ir_idx
    else:
        return rgb_idx


def map_ir_frame_idx(ir_idx, n_ir, n_rgb):
    """Returns index of RGB frame corresponding to the IR frame idx.
    If there are more RGB than IR frames, this is simply a 1:1 mapping."""
    if n_ir > n_rgb:
        rgb_idx = round(n_rgb*float(ir_idx)/n_ir)
        return rgb_idx
    else:
        return ir_idx


def run(video_dir, output_dir, extract_gps=True, extract_gps_altitude=False, sync_rgb=True):

    delete_output(output_dir)
    for dirname in ["radiometric", "preview", "gps"]:
        os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)
    if sync_rgb:
        os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    
    tiff_files = sorted(glob.glob(os.path.join(
        video_dir, "[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].TIFF")))
    rgb_files = sorted(glob.glob(os.path.join(
        video_dir, "[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].mov")))
    
    # find groups of videos belonging to the same sequence
    groups = set([
        int(str.split(os.path.basename(file), "_")[0]) for file in tiff_files])
    groups_rgb = set([
        int(str.split(os.path.basename(file), "_")[0]) for file in rgb_files])
    assert groups == groups_rgb, "Your files are incomplete or the naming is wrong. Please adhere to the naming scheme."

    # split each group individually
    gps = {}
    for group in groups:
        print("Processing video sequence {} out of {}".format(group+1, len(groups)))
        tiff_files = sorted(glob.glob(os.path.join(
            video_dir, "{:04d}_[0-9][0-9][0-9][0-9].TIFF".format(group))))
        
        n_ir = get_num_ir_frames(tiff_files)
        print("Found {} TIFF videos with {} frames".format(len(tiff_files), n_ir))
        
        if sync_rgb:
            rgb_files = sorted(glob.glob(os.path.join(
                video_dir, "{:04d}_[0-9][0-9][0-9][0-9].mov".format(group))))
            n_rgb = get_num_rgb_frames(rgb_files)

            if n_ir > n_rgb:
                assert map_rgb_frame_idx(n_rgb, n_ir, n_rgb) == n_rgb
                assert map_ir_frame_idx(n_ir, n_ir, n_rgb) == n_rgb
            else:
                assert map_rgb_frame_idx(n_rgb, n_ir, n_rgb) == n_ir
                assert map_ir_frame_idx(n_ir, n_ir, n_rgb) == n_ir

            print("Found {} RGB videos with {} frames".format(
                len(rgb_files), n_rgb))
        
        # lookup name of the last file written to disk for previous sequence
        last_files = sorted(glob.glob(os.path.join(output_dir, "radiometric", "*.tiff")))
        last_frame_idx = -1
        if len(last_files) > 0:
            last_frame_name = str.split(os.path.basename(last_files[-1]), ".")[0]
            last_frame_idx = int(str.split(last_frame_name, "_")[1])

        # split IR sequence
        ir_frame_idx = 0
        for i, tiff_file in enumerate(tiff_files):
            print("Splitting TIFF file {} of {}".format(i+1, len(tiff_files)))

            with tifffile.TiffFile(tiff_file) as tif:
                for page in tqdm(tif.pages, total=len(tif.pages)):
                    image_radiometric = page.asarray().astype(np.uint16)  # BUG: for Optris camera conversion to int is detrimental to accuracy as raw values in TIFF are float
                    image_preview = create_preview(image_radiometric)

                    if sync_rgb:
                        frame_idx = map_ir_frame_idx(ir_frame_idx, n_ir, n_rgb)
                    else:
                        frame_idx = ir_frame_idx                    
                    frame_idx += (last_frame_idx + 1)

                    radiometric_file = os.path.join(
                        output_dir, "radiometric", "frame_{:06d}.tiff".format(
                        frame_idx))
                    preview_file = os.path.join(
                        output_dir, "preview", "frame_{:06d}.jpg".format(
                        frame_idx))

                    cv2.imwrite(radiometric_file, image_radiometric)
                    cv2.imwrite(preview_file, image_preview)

                    # extract GPS position
                    if extract_gps:
                        try:
                            gps_info = page.tags["GPSTag"].value
                        except KeyError:
                            continue
                        deg_lat, deg_long = exif_gps_to_degrees(gps_info)

                        if extract_gps_altitude:
                            alt = get_gps_altitude(gps_info)
                            gps[frame_idx] = (deg_long, deg_lat, alt)
                        else:
                            gps[frame_idx] = (deg_long, deg_lat)

                    ir_frame_idx += 1

        # synchronize RGB videos
        if sync_rgb:
            rgb_frame_idx = 0
            for i, rgb_file in enumerate(rgb_files):
                print("Splitting RGB file {} of {}".format(i+1, len(rgb_files)))
                cap = cv2.VideoCapture(rgb_file)
                for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
                    res, frame = cap.read()
                    if res:
                        frame_idx = map_rgb_frame_idx(rgb_frame_idx, n_ir, n_rgb)
                        frame_idx += (last_frame_idx + 1)

                        out_path = os.path.join(output_dir,
                            "rgb", "frame_{:06d}.jpg".format(frame_idx))
                        cv2.imwrite(
                            out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

                        rgb_frame_idx += 1

    # store extracted GPS positions to disk
    if extract_gps and len(gps) > 0:        

        # interpolate GPS trajectory
        gps = np.array(list(gps.values()))
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
