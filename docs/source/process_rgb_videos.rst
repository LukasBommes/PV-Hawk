Processing of visual (RGB) videos
=================================

PV Hawk can process either thermal IR videos or visual RGB videos. By default, PV Hawk processes IR videos, which requires IR video frames to be placed under `splitted/radiometric` in the working directory as explained in :ref:`here <dataset-creation-from-videos>`. To process RGB videos, you have to place RGB video frames under `splitted/rgb` in the working directory as explained in :ref:`here <dataset-creation-from-videos>` and set the `ir_or_rgb` setting to `rgb`. Depending on the setting of the `ir_or_rgb` variable, PV Hawk will process either IR or RGB frames, but not both. 

To process both IR and RGB, you have to duplicate the working directory and run PV Hawk in one working directory with `ir_or_rgb` set to `rgb` and in the other working directory with `ir_or_rgb` set to `ir`.

Processing of RGB videos requires adjustment of some :ref:`config file settings <config_file_reference_settings>` as the defaults in the `defaults.yml` file are tuned for IR videos. Below is an exemplary config file for the processing of RGB videos. Different from the default settings are the `detection_min_confidence` for the PV module segmentation task and the `match_distance_thres` for the tracking and OpenSfM preparation tasks. The config below also shows the settings for the optional `split_sequences` tasks for extracting and resizing RGB frames from a video file.

.. code-block:: text

	---
	plant_name: Example Plant
	groups:
	  - cam_params_dir: calibration/camera_8hz/parameters
	    ir_or_rgb: rgb
	    clusters:
	    - cluster_idx: 0
		  frame_idx_start: 0
		  frame_idx_end: 2541
    settings:
	# split_sequences:
	#   sync_rgb: True
	#   resize_rgb: 
	#     width: 1280
	#     height: 720
	  segment_pv_modules:
	    detection_min_confidence: 0.7
	  track_pv_modules:
	    match_distance_thres: 50.0
	  prepare_opensfm:
	    select_frames_mode: gps
	    match_distance_thres: 50.0
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
	- refine_triangulation
	- crop_pv_modules


If you want to give RGB processing a try, you can download an example dataset of RGB video frames from `here <https://github.com/LukasBommes/PV-Hawk/releases/tag/v1.0.0>`_ (`part 1 <https://github.com/LukasBommes/PV-Hawk/releases/tag/v1.0.0/example_data_double_row_rgb.z01>`_, `part 2 <https://github.com/LukasBommes/PV-Hawk/releases/tag/v1.0.0/example_data_double_row_rgb.z02>`_, `part 3 <https://github.com/LukasBommes/PV-Hawk/releases/tag/v1.0.0/example_data_double_row_rgb.zip>`_). Download all the parts into one directory and extract them into a single directory with the following commands

.. code-block:: console

  zip -F example_data_double_row_rgb.zip --out example_data_double_row_rgb_full.zip
  unzip example_data_double_row_rgb_full.zip

You can then follow the :doc:`tutorial` using the RGB data you just downloaded and the config shown above.