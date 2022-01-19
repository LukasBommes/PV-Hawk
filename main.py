"""CLI interface to the video processing backend.

Terminal sends commands to the video processing backend via REST API. To perform
video processing tasks, a configuration YAML file has to be provided.
"""

import os
import sys
import yaml
import json
import logging
import argparse
import subprocess

from extractor.common import get_group_name, merge_dicts, remove_none, \
    replace_empty_fields
from extractor.preprocessing import split_tiffs
from extractor import tracking, quadrilaterals, cropping
from extractor.mapping import prepare_opensfm, triangulate_modules, \
    refine_triangulation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(work_dir):

    # load config file
    config = yaml.safe_load(open(os.path.join(work_dir, "config.yml"), "r"))
    tasks = config["tasks"]
    logger.info("The following tasks will be run: {}".format(tasks))

    default_settings = yaml.safe_load(open("/pvextractor/defaults.yml", "r"))

    for videogroup in config["groups"]:
        group_name = get_group_name(videogroup)

        # load algorithm settings and merged with defaults
        try:
            settings = videogroup["settings"]
        except KeyError:
            settings = {}
            
        replace_empty_fields(default_settings)
        settings = merge_dicts(default_settings, remove_none(settings))        

        # write dataset version info into workdir
        os.makedirs(os.path.join(work_dir, group_name), exist_ok=True)
        version_info = {
            "dataset_version": "v2"
        }
        json.dump(version_info, open(os.path.join(work_dir, group_name, "version.json"), "w"))

        if tasks is None:
            continue
        
        # split video sequences into frames
        if "split_sequences" in tasks:
            logger.info("Splitting raw video files into individual frames")
            output_dir = os.path.join(work_dir, group_name, "splitted")
            input_ir = os.path.join(work_dir, group_name, "videos", "*.TIFF")  # caution: case sensitive on Linux
            input_rgb = os.path.join(work_dir, group_name, "videos", "*.mov")
            split_tiffs.run(input_ir, output_dir, input_rgb, **settings["split_tiffs"])

        # segment PV modules
        if "segment_pv_modules" in tasks:
            from extractor.segmentation import inference            
            logger.info("Segmenting PV modules")
            frames_root = os.path.join(work_dir, group_name, "splitted")
            output_dir = os.path.join(work_dir, group_name, "segmented")
            inference.run(frames_root, output_dir,
                **settings["segment_pv_modules"])

        # track PV modules in subsequent frames
        if "track_pv_modules" in tasks:
            logger.info("Tracking PV modules in subsequent frames")
            frames_root = os.path.join(work_dir, group_name, "splitted")
            inference_root = os.path.join(work_dir, group_name, "segmented")
            output_dir = os.path.join(work_dir, group_name, "tracking")
            tracking.run(frames_root, inference_root, output_dir,
                **settings["track_pv_modules"])

        # compute module corners
        if "compute_pv_module_quadrilaterals" in tasks:
            logger.info("Estimating bounding quadrilaterals for PV modules")
            frames_root = os.path.join(work_dir, group_name, "splitted")
            inference_root = os.path.join(work_dir, group_name, "segmented")
            tracks_root = os.path.join(work_dir, group_name, "tracking")
            output_dir = os.path.join(work_dir, group_name, "quadrilaterals")
            quadrilaterals.run(frames_root, inference_root, tracks_root,
                output_dir, **settings["compute_pv_module_quadrilaterals"])

        # prepare data for 3D reconstruction with OpenSfM
        if "prepare_opensfm" in tasks:
            for cluster in videogroup["clusters"]:
                logger.info("Preparing data for OpenSfM reconstruction")
                frames_root = os.path.join(work_dir, group_name, "splitted")
                calibration_root = os.path.join(videogroup["cam_params_dir"], "ir")
                output_dir = os.path.join(work_dir, group_name, "mapping")
                opensfm_settings = settings["opensfm"]
                prepare_opensfm.run(cluster, frames_root, calibration_root, 
                    output_dir, opensfm_settings, **settings["prepare_opensfm"])

        # run OpenSfM for 3D reconstruction
        opensfm_tasks = [
            "opensfm_extract_metadata", 
            "opensfm_detect_features",
            "opensfm_match_features",
            "opensfm_create_tracks",
            "opensfm_reconstruct"
        ]
        if any([t in opensfm_tasks for t in tasks]):
            opensfm_bin = "/pvextractor/extractor/mapping/OpenSfM/bin/opensfm"
            for cluster in videogroup["clusters"]:
                mapping_root = os.path.join(
                    work_dir, group_name, "mapping", 
                    "cluster_{:06d}".format(cluster["cluster_idx"]))

                # determine which OpenSfM commands to run
                opensfm_command = []
                for task in tasks:
                    if task[:7] == "opensfm":
                        opensfm_command.append(f'{opensfm_bin} {task[8:]} "{mapping_root}"')
                opensfm_command = " && ".join(opensfm_command)

                if len(opensfm_command) > 0:
                    logger.info("Running OpenSfM reconstruction")
                    proc = subprocess.Popen(opensfm_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    for line in proc.stdout:
                        sys.stdout.write(line.decode("utf-8"))

        if "triangulate_pv_modules" in tasks:
            mapping_root = os.path.join(work_dir, group_name, "mapping")
            tracks_root = os.path.join(work_dir, group_name, "tracking")
            quads_root = os.path.join(work_dir, group_name, "quadrilaterals")
            triangulate_modules.run(mapping_root, tracks_root, quads_root, 
                **settings["triangulate_pv_modules"])

        if "refine_triangulation" in tasks:
            mapping_root = os.path.join(work_dir, group_name, "mapping")
            refine_triangulation.run(mapping_root, **settings["refine_triangulation"])

        if "crop_pv_modules" in tasks:
            logger.info("Cropping PV module patches")
            frames_root = os.path.join(work_dir, group_name, "splitted")       
            quads_root = os.path.join(work_dir, group_name, "quadrilaterals")
            mapping_root = os.path.join(work_dir, group_name, "mapping")
            output_dir = os.path.join(work_dir, group_name, "patches")
            cropping.run(frames_root, quads_root, mapping_root, output_dir, 
                **settings["crop_pv_modules"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', type=str,
        help="Path of to the working directory.")
    args = parser.parse_args()

    main(args.workdir)
