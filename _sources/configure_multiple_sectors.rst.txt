Configuring Multiple Sectors
============================

For large PV plants it is helpful to split the plant into smaller sectors and process sectors individually. To do so the structure of the working directory and the config file have to be altered slightly.

Lets consider an extended version of the dataset used in the :doc:`tutorial`. Download the second part of the dataset `from here <https://github.com/LukasBommes/PV-Hawk/releases/download/v1.0.0/example_data_single_row.zip>`_. This second part is just another scan of the same section of the PV plant (with rows scanned individually rather than in pairs). However, in practice, each dataset part will correspond to different sectors of your PV plant. But for the explanation below this difference does not matter.

To process both datasets in a single pipeline run, first create a new working directory with the following structure

.. code-block:: text

  /storage/pv-hawk-tutorial/workdir
    |-- double_rows
    |    |-- splitted
    |    |    |-- ...
    |-- single_rows
    |    |-- splitted
    |    |    |-- ...
    
As opposed to the single-sector dataset in the tutorial, the directory levels `double_rows` and `single_rows` are introduced. The `double_rows` directory contains the dataset from the tutorial and the `single_rows` directory the newly downloaded dataset.

Next, create a `config.yml` file with the text below    
    
.. code-block:: text
    
	---
	plant_name: Multiple sectors
	groups:
	- name: double_rows
	  cam_params_dir: calibration/camera_8hz/parameters
	  ir_or_rgb: ir
	  clusters:
	  - cluster_idx: 0
	    frame_idx_start: 0
	    frame_idx_end: 2541
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
	  - refine_triangulation
	  - crop_pv_modules
	 
The file now contains two items under `groups`, each configuring one of the two datasets. Note, that each item now has an additional `name` attribute which corresponds to the directory names of the sectors, i.e. `double_rows` and `single_rows`. 

Apart from that, there are no differences to the simple case described in the :doc:`tutorial`. Simply follow the remaining steps of the tutorial to process the extended dataset.


Difference between sectors and clusters
---------------------------------------

You could also use clusters to split a larger dataset into smaller ones. The difference to sector-based splitting is that all clusters share the same algorithm settings, whereas each sector can have its own unique settings. 

Furthermore, a dataset with multiple clusters will be processed in a single working directory, whereas each sector has its own subdirectory. Each subdirectory is treated like an independent working directory when viewing the data with the PV Hawk Viewer.
