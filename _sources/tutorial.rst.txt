Tutorial
========

This tutorial will get you started with PV Drone Inspect on an exemplary IR video dataset. After finishing this tutorial, head over to :ref:`using_own_data` to learn how to create your own dataset.
 

- download and place example dataset (already splitted images, link to data acquisition for further information on acquiring own dataset)
- run docker image with correct path mapping
- create config file and briefly explain the different settings (link to config file reference and explain how to use view_gps.py script to select clusters, explain important settings)
- run video processing pipeline (first four steps, then latter), explain how to validate intermediate outputs
- visualize final results
- open result in pv drone inspect viewer

Step 1: Get the example dataset
-------------------------------

[...]

.. _run-the-docker-image:

Step 2: Run the Docker image
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
  
  
Step 3: Create a config file
----------------------------


Step 4: Run the pipeline
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
