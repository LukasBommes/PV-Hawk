Tutorial
========

This tutorial will get you started with PV Drone Inspect on an exemplary IR video dataset. It shows the general workflow of processing a PV plant. Please :ref:`install <installation>` PV Drone Inspect on your machine before continuing. After finishing this tutorial, head over to :ref:`using_own_data` to learn how to create your own dataset.

- download and place example dataset (already splitted images, link to data acquisition for further information on acquiring own dataset)
- run docker image with correct path mapping
- create config file and briefly explain the different settings (link to config file reference and explain how to use view_gps.py script to select clusters, explain important settings)
- run video processing pipeline (first four steps, then latter), explain how to validate intermediate outputs
- visualize final results
- open result in pv drone inspect viewer

Step 1: Prepare the working directory
-------------------------------------

PV Drone Inspect reads and writes your input data, results and intermediate data from and to the *working directory*, which is simply a directory on your hard drive.

Before anything else, you have to create such a working directory. Choose a suitable location on your machine and create an empty directory. In our example, we have mounted a hard drive at `/storage` and create a directory here with the following command

.. code-block:: console

    cd /storage
    mkdir -p pv-drone-inspect-tutorial/workdir
    
You can also choose any other location than `/storage`.


Step 2: Download the example dataset
------------------------------------

Download and extract the `example dataset <https://drive.google.com/file/d/1NlRpSqFIzaTuEGkFHur6nbu1bE9aPew0/view?usp=sharing>`_, which covers the first twelve rows of a large-scale PV plant. The download contains a folder named `splitted`, which follows the required format of an input dataset for PV Drone Inspect (see :ref:`dataset-creation-from-videos`).

Place the extracted `splitted` directory and its contents into the working directory. The resulting directory structure should look as follows

.. code-block:: text

  /storage/pv-drone-inspect-tutorial/workdir
    |-- splitted
    |    |-- ...
    

Step 3: Create a config file
----------------------------

Every working directory must contain a config file named `config.yml`. This file is your interface to the processing pipeline. Here, you configure settings of the individual pipeline steps and determine which steps to run.

Create an empty text file named `config.yml` in the working directory and paste the following text

.. code-block:: text

 	---
	plant_name: Example Plant
	groups:
	- cam_params_dir: calibration/camera_8hz/parameters
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

	# list of tasks to perform
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

Important for this tutorial is the `tasks` list, which specifies the pipeline steps to run. Steps that are commented out will not be run. In this case, the `split_sequences` step will be omitted because the example data is already given in the form of individual images instead of video files.

Another important field in the config is the `clusters` field. Each cluster corresponds to a subset of the video frames in the dataset, which is processed indepently of other clusters. This enables excluding parts of the video, e.g., when you change batteries or start/land the drone. We recommend to split long sequences to clusters of at most 5000 video frames to enhance processing speed and robustness of the pipeline. Each cluster must contain the index of its first and last frame (not inclusive) and a unique `cluster_idx`, which is an integer starting from 0 and incrementing by 1 for each cluster.

To aid specification of the clusters we provide a script in `scripts/view_gps.py`. This script plots the GPS trajectory and corresponding video frames as shown below. You can use this to obtain the frame indices for your clusters. Note, that you must run the script inside the Docker container as explained :ref:`below <run-the-docker-image>`.

.. image:: images/view_gps_script.png


In case of this tutorial there is only a single cluster starting at the first frame (`frame_idx_start: 0`) and ending at the last frame (`frame_idx_end: 2541`) of the dataset.

For an in-depth explanation of the other fields in the config file see the :doc:`config_file_reference`.

.. note::
  The dataset in this tutorial is relatively small. For larger datasets it is useful to split the data into multiple parts as described in :doc:`configure_multiple_sectors`.


.. _run-the-docker-image:

Step 4: Run the Docker image
----------------------------

You have to run PV Drone Inspect in an interactive terminal session inside the Docker image that you built in the previous steps. Before doing so, make sure access control of your machine's X server is disabled by running

.. code-block:: console

  xhost +

This enables graphical output from the Docker container to be forwarded to your machine. Note, that you have to repeat this step every time you rebooted your machine.

To run the interactive terminal session inside the Docker container run the following command from the project's root directory

.. code-block:: console

  sudo docker run -it \
    --ipc=host \
    --env="DISPLAY" \
    --gpus=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$(pwd)":/pvextractor \
    -v /storage:/storage \
    -p "8888:8888" \
    pvextractor-geo \
    bash
    
You can omit the `-p "8888:8888"` option if you do not plan to use jupyter lab inside the container. Jupyter lab is needed, for instance, for camera claibration or to fine-tune the Mask R-CNN model contained in this tool.

If you encounter a "permission denied" error make the entrypoint script executable by running the following in the project's root directory

.. code-block:: console

  chmod +x docker-entrypoint.sh


Step 5: Run the pipeline
------------------------

Once the config file is created, you can process the data by executing the following command inside the interactive session in the Docker container

.. code-block:: console

  python main.py testing/configs/config_plant_A.yml
  
To control which tasks are executed you can (un)comment tasks under `tasks` in the config file. Note, that you cannot skip any of the tasks, i.e. you will have to run each tasks at least once in the order specified in the config file.

We recommend to first uncomment the steps "split_sequences", "segment_pv_modules", "track_pv_modules", "crop_and_rectify_modules" and commenting all subsequent steps. These are preprocessing steps. You should ensure the correctness of the output of these steps before continuing with the remaining processing steps. To continue, comment out the first four steps and uncomment the remaining steps. Rerun :code:`python main.py testing/configs/config_plant_A.yml`.

Step 5: Visualize results
-------------------------

We provide a script `extractor/mapping/plot_reconstruction.py` for plotting the reconstructed camera poses, PV modules and map points. You can use this script to validate whether your PV plant was reconstructed and georeferenced correctly.

To this end, run the script from within the interactive Docker session and provide the `work_dir` of the plant

.. code-block:: console

  python plot_reconstruction.py "/storage-2/pvextractor-georeferencing/Plant_A/workdir"
  
You can view the help for additional optional arguments

.. code-block:: console

  python plot_reconstruction.py -h
  
  
[mention how to use PV Drone Inspect Viewer for actual inspection of the data]
