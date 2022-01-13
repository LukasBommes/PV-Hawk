Installation
============

Follow the steps below to setup PV Drone Inspect on your machine.

Step 1: Install prerequisites
-----------------------------

To run PV Drone Inspect you need a machine running Ubuntu 18.04 LTS / 20.04 LTS and a Nvidia CUDA-compatible GPU with the latest Nvidia drivers installed. Furthermore, you need to install `Docker CE <https://docs.docker.com/engine/install/ubuntu/>`_ and the `Nvidia container toolkit <https://github.com/NVIDIA/nvidia-docker>`_.

Step 2: Clone source code
-------------------------

Clone the Git repository to your machine

.. code-block:: console

  git clone https://github.com/LukasBommes/PV-Drone-Inspect


Step 3: Download Mask R-CNN model files
---------------------------------------

The tool uses a pretrained `Mask R-CNN <https://github.com/matterport/Mask_RCNN>`_ for PV module detection. Download the pretrained Mask R-CNN model weights from `here <https://drive.google.com/file/d/1F0GiR8QpKZEHV-4wtfbPeE5dvIiOeIG3/view>`_, extract the zip archive and place the folder "pv_modules20210521T1611" under `extractor/segmentation/Mask_RCNN/logs`. The resulting directory structure should look like follows:

.. code-block:: text

  |-- extractor/segmentation/Mask_RCNN/logs
  |    |-- pv_modules20210521T1611
  |    |     |-- events.out.tfevents.1621613480.27049cbf8e56
  |    |     |-- events.out.tfevents.1621782508.27049cbf8e56
  |    |     |-- mask_rcnn_pv_modules_0060.h5
  |    |     |-- mask_rcnn_pv_modules_0120.h5

Step 4: Build Docker image
--------------------------

We use Docker to provide a consistent environment for the execution of PV Drone Inspect. When building the provided Docker image all required dependencies, e.g., Python, CUDA, Tensorflow, and OpenSfM, are installed and configured automatically. There are two ways to use the Docker image: A) building the image from the provided Dockerfile, or B) load a prebuilt image.


Variant A: Build Docker image from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the Docker image from the provided Dockerfile run the following command from the root directory of PV Drone Inspect

.. code-block:: console

  sudo docker build . --tag=pvextractor-geo

Variant B: Load prebuild Docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can download a prebuilt Docker image from `here <https://drive.google.com/file/d/1ksjtbYPbpkMeZbChfqtSVkXBKHAq9Nbo/view>`_. Place the tar archive in the project's root directory an load the Docker image by executing

.. code-block:: console

  sudo docker load < pvextractor-geo.tar

Note, that the image was built on a machine with Ubuntu 20.04 LTS. Transferability to other operating systems is not guaranteed. If you run into issue with the prebuild image, please build the image from source as specified above.


Step 5: Test the installation
-----------------------------

Test whether PV Drone Inspect was correctly installed by running the Docker container with the following command from the project's root directory

.. code-block:: console

  sudo docker run -it \
    --ipc=host \
    --env="DISPLAY" \
    --gpus=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$(pwd)":/pvextractor \
    pvextractor-geo \
    bash
    
This should start an interactive terminal session in the Docker container you just built.

If you encounter a "permission denied" error make the entrypoint script executable by running
 
.. code-block:: console
 
  chmod +x docker-entrypoint.sh
   
from the project's root directory.


