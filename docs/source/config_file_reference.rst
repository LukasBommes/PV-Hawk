Config File Reference
=====================

The config file is a YAML file named `config.yml` located in the working directory of your dataset, which is the main interface to the PV Hawk pipeline. In the config file you determine the splitting of your dataset into clusters, you provide algorithm settings and determine which steps of the pipeline are run.

Using the following example config file we will now explain the available options in detail.

.. code-block:: text
    
	---
	plant_name: Example config
	groups:
	- name: double_rows
	  cam_params_dir: calibration/camera_8hz/parameters
	  ir_or_rgb: ir
	  clusters:
	  - cluster_idx: 0
	    frame_idx_start: 0
	    frame_idx_end: 2640
	  - cluster_idx: 1
	    frame_idx_start: 2790
	    frame_idx_end: 4680
	  settings:
	    prepare_opensfm:
	      select_frames_mode: gps_visual
	    opensfm:
	      matching_gps_distance: 15
	      align_method: orientation_prior
	      align_orientation_prior: vertical
	- name: single_rows
	  cam_params_dir: calibration/camera_8hz/parameters
	  ir_or_rgb: ir
	  clusters:
	  - cluster_idx: 0
	    frame_idx_start: 0
	    frame_idx_end: 5295
	  settings:
	    prepare_opensfm:
	      select_frames_mode: gps_visual
	    opensfm:
	      matching_gps_distance: 15
	      align_method: orientation_prior
	      align_orientation_prior: vertical

	tasks:
	  #- split_sequences
	  - interpolate_gps
	  - segment_pv_modules
	  - track_pv_modules
	  - compute_pv_module_quadrilaterals
	  - prepare_opensfm
	  - opensfm_extract_metadata
	  - opensfm_detect_features
	  - opensfm_match_features
	  - opensfm_create_tracks
	  - opensfm_reconstruct
	  - triangulate_pv_modules

General
-------

.. rubric:: plant_name (string)

A note for yourself about the origin of the dataset. The field is not used internally.

.. rubric:: groups (list of group objects)

A list of dataset groups corresponding to sectors in your PV plant. See :doc:`configure_multiple_sectors` for details on using multiple groups. Each group is processed indepently of each group can have its own algorithm settings.

.. rubric:: group::name (string)

Name of each group. Must be euqivalent to the name of the group's subdirectory in the working directory. E.g. if you have groups with names "sector_1" and "sector_2" the working dir must look as follows

.. code-block:: text

  /workdir
    |-- sector_1
    |    |-- splitted
    |    |    |-- ...
    |-- sector_2
    |    |-- splitted
    |    |    |-- ...
    |-- ...

If your dataset consists of a single group you can omit the name field and the subdirectory level in the working directory. I.e. the working directory would look like

.. code-block:: text

  /workdir
    |-- splitted

.. rubric:: group::cam_params_dir (string)

Path to the directory containing two subfolders with calibrated IR and RGB camera parameters (`ir/camera_matrix.pkl`, `ir/dist_coeffs.pkl`, ..., `rgb/camera_matrix.pkl`, `rgb/dist_coeffs.pkl`, ...). Parameters are created with the camera calibration script (see :ref:`camera_calibration`).

In the following example, the `cam_params_dir` should be set to `/path/to/cameraparams`.

.. code-block:: text

  /path/to/cameraparams
    |-- ir
	|    |-- camera_matrix.pkl
	|    |-- dist_coeffs.pkl
	|    |-- ...
	|-- rgb
	|    |-- camera_matrix.pkl
	|    |-- dist_coeffs.pkl
	|    |-- ...

.. rubric:: group::ir_or_rgb (string)

Must be either `ir` or `rgb`. Selects whether the pipeline should process IR or RGB images. When set to IR, the pipeline operates on IR video frames from `splitted/radiometric`. When set to RGB, the pipeline operates on RGB video frames from `splitted/rgb`. This setting also determines, whether to use the IR or RGB camera calibration parameters and Mask R-CNN instance segmentation model.

.. rubric:: group::clusters (list of cluster objects)

A cluster corresponds to a subset of the video frames in the group that is to be processed indepently of other clusters. Use clusters to exclude parts of the video, e.g., when you change batteries or start/land the drone. It is recommended to split long sequences to clusters of at most 5000 video frames to enhance processing speed and robustness of the pipeline.

.. rubric:: group::cluster::cluster_idx (integer)

Identifier of the cluster. The first cluster must have index 0, the second 1, and so on.

.. rubric:: group::cluster::frame_idx_start (integer)

Index of the first frame in the cluster.

.. rubric:: group::cluster::frame_idx_end (integer)

Index of the frame at which the cluster ends. This frame is not included in the cluster anymore. 

.. rubric:: group::settings (settings object)

Algorithm settings for each pipeline task that apply group-wide. If you do not provide a setting then the default specified in `defaults.yml` in the root directory is applied. If you want to overwrite a default value, provide the settings name (equiavlent to the task name), parameter name and value in the settings object. In the examplary config file above the `prepare_opensfm` task is configured to subsample video frames based on both GPS distance and visual distance. Similarly, some defaults for the `opensfm` task are overwritten. See below for a complete overview of the available settings.


Tasks
-----

List of tasks to perform when running the PV Hawk pipeline. (Un)comment to control which steps to run.

.. rubric:: split_sequences

Split multipage TIFF IR videos into individual IR frames and split mov videos into individual RGB frames. Also performs naive synchronization between IR and RGB streams. Specific to Flir Zenmuse XT2 camera.

.. rubric:: interpolate_gps

Perform linear interpolation of the GPS trajectory to match the GPS measurement rate with a potentially higher video frame rate.

.. rubric:: segment_pv_modules

Run Mask R-CNN inference to segment PV modules in each video frame.

.. rubric:: track_pv_modules

Track PV modules (segmentation masks) over subsequent video frames, assigning a unique tracking ID to each module.

.. rubric:: compute_pv_module_quadrilaterals

Estimate a bounding quadrilateral (polygon with 4 points) for each segmentation mask. Needed for later cropping of a rectangular image patches of the PV modules.

.. rubric:: prepare_opensfm

Create the OpenSfM input datasets for the reconstruction of the camera trajectory. For each cluster a separate OpenSfM dataset is created on which OpenSfM is run in the subsequent step. Preparation inclused selection of a subset of video frames, which are used for reconstructionby OpenSfM.

.. rubric:: opensfm_extract_metadata

Run the `extract_metadata` step of the OpenSfM pipeline for each cluster.

.. rubric:: opensfm_detect_features

Run the `detect_features` step of the OpenSfM pipeline for each cluster.

.. rubric:: opensfm_match_features

Run the `match_features` step of the OpenSfM pipeline for each cluster.

.. rubric:: opensfm_create_tracks

Run the `extract_metadata` step of the OpenSfM pipeline for each cluster.

.. rubric:: opensfm_reconstruct

Run the `reconstruct` step of the OpenSfM pipeline for each cluster. This reconstructs the 6-DOF camera pose (rotation and translation) for each video frame selected in `prepare_opensfm` step.

.. rubric:: triangulate_pv_modules

Use the reconstructed camera poses and known corner points of each PV module to triangulate PV modules into the OpenSfM reconstruction of the PV plant.

.. rubric:: refine_triangulation

Smoothen the triangulated PV modules. Nearby module corners are moved closer to each other by means of an iterative graph optimization algorithm.

.. rubric:: crop_pv_modules

Crops the image patches of each PV module based on the estimated quadrilaterals. Patches are transformed to a rectangular region with a homography.


Task Order
----------

Tasks are executed in the same order as they are enlisted in the config file. In general each task depends on the preceeding tasks. Thus, you have to run them in the same order as they are enlisted in the exemplary config file above. It can make sense to run individual tasks or only a few tasks at a time to validate intermediate results. In this case, make sure to uncomment the tasks you already ran, or otherwise they will be rerun.

An exception to the sequantial order is the `crop_pv_modules` tasks. Normally, you would want to run it as the last step in the pipeline. However, if you only want to extract IR image patches and do not need geocoordinates of the modules, you can omit all tasks from `prepare_opensfm` (included) to `refine_triangulation` (included) and run the `crop_pv_modules` task as last step in the pipeline directly after the `compute_pv_module_quadrilaterals` task.

.. _config_file_reference_settings:

Settings
--------

Note: Boolean values can be represented in the `config.yml` as True / False or as yes / no.

.. rubric:: split_sequences

* **ir_file_extension** (string): File extensions of input IR videos, e.g. `TIFF`. Case sensitive on Linux.
* **rgb_file_extension** (string): File extensions of visual input videos, e.g. `MOV`. Case sensitive on Linux.
* **extract_timestamps** (boolean): If True extract frame timestamps from input TIFF stack.
* **extract_gps** (boolean): If True extract GPS trajectory of the drone from input TIFF stack.
* **extract_gps_altitude** (boolean): If True extract the GPS altitude.
* **sync_rgb** (boolean): If True attempt a simplistic synchronization of visual and IR video stream. If False ignore visual stream.
* **subsample** (string or null): Subsample both IR and RGB frames. If set to a value N, only every Nth frame will be extracted. Set to `null` to extract all frames.
* **rotate_rgb** (string or null): Set to "90_deg_cw", "180_deg_cw", or "270_deg_cw" to rotate splitted RGB video frames. If set to `null` frames are not rotated.
* **rotate_ir** (string or null): Set to "90_deg_cw", "180_deg_cw", or "270_deg_cw" to rotate splitted IR video frames. If set to `null` frames are not rotated.
* **resize_rgb** (object with keys width and height): Resize RGB frames to the given height and width. If width and height are `null`, no resizing is performed.
* **resize_ir** (object with keys width and height): Resize IR frames to the given height and width. If width and height are `null`, no resizing is performed.

.. rubric:: interpolate_gps

No settings.

.. rubric:: segment_pv_modules

* **gpu_count** (integer): Number of GPUs to use.
* **images_per_gpu** (integer): Number of frames (per GPU) to feed into Mask R-CNN simultaneously.
* **detection_min_confidence** (float):  PV module instances with prediction confidence (0.0..1.0) below this value are ignored.
* **weights_file_rgb** (string): Absolute path to the Mask R-CNN weights file trained on RGB images.
* **weights_file_ir** (string): Absolute path to the Mask R-CNN weights file trained on IR images.
* **output_video_fps** (float): Frame rate of the generated preview video in 1/s.

The segmentation task has some further settings in `extractor/segmentation/configs.py`, for example the inference batch size.

.. rubric:: track_pv_modules

* **motion_model** (string): How to model the motion between two subsequent frames. Either "homography", "affine", or "affine_partial".
* **orb_nfeatures** (integer): Number of ORB features to extract in each frame. Needed for motion estimation.
* **orb_fast_thres** (integer): FAST threshold for ORB feature extraction.
* **orb_scale_factor** (float): Scale factor for ORB feature extraction.
* **orb_nlevels** (integer): Number of pyramid levels for ORB feature extraction.
* **match_distance_thres** (float): Maximum feature distance of two feature descriptors to be matched. Needed for motion estimation.
* **max_distance** (integer): Maximum Euclidean distance (in pixels) a module center point can travel in two subsequent frames to still be considered the same module.
* **output_video_fps** (float): Frame rate of the generated preview video in 1/s.
* **deterministic_track_ids** (boolean): If True make module UUIDs deterministic, i.e. produce same module UUIDs for multiple runs on same data. Otherwise, random UUID are used.

.. rubric:: compute_pv_module_quadrilaterals

* **min_iou** (float): Minimum IoU between segmentation mask and estimated quarilateral needed to consider quadrilateral as valid.

.. rubric:: prepare_opensfm

* **select_frames_mode** (string): Select frames for 3D reconstruction based on travelled GPS distance alone ("gps") or GPS and visual distance ("gps_visual").
* **frame_selection_gps_distance** (float): Select a frame as keyframe if drone travelled this many meters along the GPS track
* **frame_selection_visual_distance** (float): Select a frame as keyframe if the visual distance (1 - intersection over union) of the frame to the previous one is larger than this value. The value must be a fraction in range 0 to 1.
* **orb_nfeatures** (integer): Number of ORB features to extract in each frame. Needed to compute visual distance.
* **orb_fast_thres** (integer): FAST threshold for ORB feature extraction.
* **orb_scale_factor** (float): Scale factor for ORB feature extraction.
* **orb_nlevels** (integer): Number of pyramid levels for ORB feature extraction.
* **match_distance_thres** (float): Maximum feature distance of two feature descriptors to be matched. Needed to compute visual distance.
* **gps_dop** (float): If no measurement of the GPS dilution of precision (DOP) is available, this constant DOP is used instead.
* **output_video_fps** (float): Frame rate of the generated preview video in 1/s.

.. rubric:: opensfm

* **matching_gps_distance** (integer): Maximum GPS distance (in meters) between two images for matching.
* **use_altitude_tag** (boolean): If True use GPS altitude measurement during reconstruction. Set to False if you do not have a reliable GPS altitude measurement.
* **align_method** (string): Method for global alignment of the reconstruction. Either "orientation_prior" or "naive". Set to "orientation_prior" to assume a constant camera orientation.
* **align_orientation_prior** (string): If orientation prior is used, which orientation prior to use. Either "horizontal", "vertical" or "no_roll".
* **processes** (integer): Number of parallel threads to use.

For further OpenSfM settings see `extractor/mapping/OpenSfM/opensfm/config.py`. You can change any of these settings in your config file. But do not edit the OpenSfM config directly.

.. rubric:: triangulate_pv_modules

* **min_track_len** (integer): Triangulate only modules observed in at least this many keyframes.
* **merge_overlapping_modules** (boolean): Set to True to merge duplicate detections of the same PV module.
* **merge_threshold** (float): Merge multiple modules if the mean L2 norm of their corresponding corner points (projected into the image) is below this threshold value (in pixels).
* **max_module_depth** (float): Consider only modules for merging which are at most this many meters away from reconstructed camera center. Set to -1 to disable this filter.
* **max_num_modules** (integer): If number of PV modules per frame exceeds this value skip merging of overlapping modules in this frame.
* **max_combinations** (integer): Maximum number of pairs to consider when triangulating 3D points of PV module corners from all observing keyframes. Set to -1 to consider all pairs.
* **reproj_thres** (float): Maximum reprojection error (in pixels) for a triangulated point to be valid.
* **min_ray_angle_degrees** (float): Minimum ray angle (in degrees) for a triangulated point to be valid.

.. rubric:: refine_triangulation

* **merge_threshold_image** (float): Pull module corners together which are closer in projected image space than this threshold (in pixels).
* **merge_threshold_world** (float): Pull module corners together which are closer in 3D world space than this threshold (in meters).
* **max_module_depth** (float): Project only modules which are at most this many meters away from reconstructed camera center. Set to -1 to disable this filter.
* **max_num_modules** (integer): If number of PV modules per frame exceeds this value skip do not consider this frame for refinement.
* **optimizer_steps** (integer): Number of graph optimization steps to perform.

.. rubric:: crop_pv_modules

* **rotate_mode** (string or null): Rotate cropped module images into "portrait" or "landscape" orientation. Set to `null` to ignore patches with potentially wrong orientation.
