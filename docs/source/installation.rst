.. _installation:

Installation
============

Follow the steps below to setup PV Hawk on your machine.

Step 1: Fullfill prerequisites
------------------------------

PV Hawk requires a machine running a 64-bit version of Ubuntu 21.10, 21.04, 20.04 LTS or 18.04 LTS with `Docker CE <https://docs.docker.com/engine/install/ubuntu/>`_ installed.

To make use of GPU acceleration (highly recommended) you need a Nvidia CUDA-compatible GPU with the latest Nvidia drivers and you must install the `Nvidia container toolkit <https://github.com/NVIDIA/nvidia-docker>`_.

Step 2: Download source code
----------------------------

Open a new terminal and navigate to the location where you want to install PV Hawk, e.g. `/software`. Run the command below to clone the Git repository to your machine

.. code-block:: console

  git clone https://github.com/LukasBommes/PV-Hawk


Step 3: Download Mask R-CNN weights
-----------------------------------

PV Hawk uses a pretrained `Mask R-CNN <https://github.com/matterport/Mask_RCNN>`_ for PV module detection. Download the pretrained Mask R-CNN model weights file from `here <https://github.com/LukasBommes/PV-Hawk/releases/download/v1.0.0/mask_rcnn_pv_modules_0120.h5>`_, and place it under `extractor/segmentation/Mask_RCNN`.


Step 4: Pull Docker image
-------------------------

We provide a prebuilt Docker image containing all runtime dependencies of PV Hawk, such as Python, CUDA and Tensorflow. All you have to do is pull the Docker image with

.. code-block:: console

	sudo docker pull lubo1994/pv-hawk:latest
	
If you run into problems with the prebuilt image, you can instead :doc:`built_docker_image` locally.


Step 5: Test the installation
-----------------------------

PV Hawk comes with some test cases, which you can run to test whether the installation was successful. To this end, start a terminal in the source code root directory and run the following two commands

.. code-block:: console

  xhost +

.. code-block:: console
    
  sudo docker run -it \
    --ipc=host \
    --env="DISPLAY" \
    --gpus=all \
    --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix:rw \
    --mount type=bind,src="$(pwd)",dst=/pvextractor \
    --mount type=volume,dst=/pvextractor/extractor/mapping/OpenSfM \
    lubo1994/pv-hawk:latest \
    bash 
    
This should start an interactive shell in the Docker container. Run the tests in that shell with

.. code-block:: console

  python -m unittest tests/**/test_*.py
  
If everything was installed correctly you should see an `OK` message after a few minutes. If you see any failures, confirm that you followed the steps above correctly. You can also open an issue in the GitHub repository.


