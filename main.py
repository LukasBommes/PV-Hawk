"""CLI interface to the video processing backend.

Terminal sends commands to the video processing backend via REST API. To perform
video processing tasks, a configuration YAML file has to be provided.
"""

import os
import sys
import yaml
import logging
import argparse
import subprocess

from extractor.common import get_group_name, merge_dicts, remove_none, \
    replace_empty_fields
from extractor.preprocessing import split_tiffs, rotate
from extractor import cropping, tracking, reorganize_patches
from extractor.mapping import prepare_opensfm, triangulate_modules, \
    refine_triangulation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config_file, start_from_task):
    # load config file
    config = yaml.safe_load(open(config_file, "r"))
    tasks = config["tasks"]

    assert start_from_task is None or start_from_task in tasks, \
        "start_from_task must be one of the tasks specified in the config file"

    # run only the tasks specified by "start_from_tasks" and all subsequent tasks
    if start_from_task is not None:
        tasks = tasks[tasks.index(start_from_task):]
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
        
        # split video sequences into frames
        if "split_sequences" in tasks:
            logger.info("Splitting raw video files into individual frames")
            output_dir = os.path.join(config["work_dir"], group_name, "splitted")
            input_ir = os.path.join(config["video_dir"], group_name, "*.TIFF")  # caution: case sensitive on Linux
            input_rgb = os.path.join(config["video_dir"], group_name, "*.mov")
            split_tiffs.run(input_ir, output_dir, input_rgb, **settings["split_tiffs"])

            # rotate frames if rows are oriented vertically in the video
            logger.info("Rotating video frames")
            if videogroup["row_orientation"] == "vertical":
                frames_root = os.path.join(config["work_dir"], group_name, "splitted")
                rotate.run(frames_root, **settings["rotate_frames"])
            else:
                logger.info("Nothing to rotate. Skipping.")

        # segment PV modules
        if "segment_pv_modules" in tasks:
            from extractor.segmentation import inference            
            logger.info("Segmenting PV modules")
            frames_root = os.path.join(
                config["work_dir"], group_name, "splitted", "radiometric")
            output_dir = os.path.join(config["work_dir"], group_name, "segmented")
            inference.run(frames_root, output_dir,
                **settings["segment_pv_modules"])

        # track PV modules in subsequent frames
        if "track_pv_modules" in tasks:
            logger.info("Tracking PV modules in subsequent frames")
            frames_root = os.path.join(
                config["work_dir"], group_name, "splitted", "radiometric")
            inference_root = os.path.join(config["work_dir"], group_name, "segmented")
            output_dir = os.path.join(config["work_dir"], group_name, "tracking")
            tracking.run(frames_root, inference_root, output_dir,
                **settings["track_pv_modules"])

        # crop and rectify modules
        if "crop_and_rectify_modules" in tasks:
            logger.info("Cropping and rectifying image patches of PV modules")
            frames_root = os.path.join(config["work_dir"], group_name, "splitted")
            inference_root = os.path.join(config["work_dir"], group_name, "segmented")
            tracks_root = os.path.join(config["work_dir"], group_name, "tracking")
            output_dir = os.path.join(config["work_dir"], group_name, "patches")
            cropping.run(frames_root, inference_root, tracks_root,
                output_dir, **settings["crop_and_rectify_modules"])

        # prepare data for 3D reconstruction with OpenSfM
        if "prepare_opensfm" in tasks:
            for cluster in videogroup["clusters"]:
                logger.info("Preparing data for OpenSfM reconstruction")
                frames_root = os.path.join(config["work_dir"], group_name, "splitted")
                calibration_root = os.path.join(videogroup["cam_params_dir"], "ir")
                output_dir = os.path.join(config["work_dir"], group_name, "mapping")
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
                    config["work_dir"], group_name, "mapping", 
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
            mapping_root = os.path.join(config["work_dir"], group_name, "mapping")
            tracks_root = os.path.join(config["work_dir"], group_name, "tracking")
            patches_root = os.path.join(config["work_dir"], group_name, "patches")
            triangulate_modules.run(mapping_root, tracks_root, patches_root, 
                **settings["triangulate_pv_modules"])

        if "refine_triangulation" in tasks:
            mapping_root = os.path.join(config["work_dir"], group_name, "mapping")
            refine_triangulation.run(mapping_root, **settings["refine_triangulation"])

        if "reorganize_patches" in tasks:
            output_dir = os.path.join(config["work_dir"], group_name, "patches_final")
            mapping_root = os.path.join(config["work_dir"], group_name, "mapping")
            patches_root = os.path.join(config["work_dir"], group_name, "patches")
            reorganize_patches.run(mapping_root, patches_root, output_dir,
                **settings["reorganize_patches"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
        help="Path of to the config YAML file.")
    parser.add_argument('--start_from_task', type=str,
        help=("start the pipeline from this task (including the task). Must be "
              "one of the tasks specified in the config file."))
    args = parser.parse_args()

    main(args.config, args.start_from_task)
