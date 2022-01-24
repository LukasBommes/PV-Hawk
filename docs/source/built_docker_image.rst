Build the Docker Image
======================

Instead of pulling the prebuilt Docker image from DockerHub you can also build the Docker image locally based on the specification provided in the `Dockerfile`. To this end, run the following command from the root directory of the PV Hawk source code

.. code-block:: console

  sudo docker build . --tag=pv-hawk-custom-build
  
Now, you can use this image by stating `pv-hawk-custom-build` instead of `lubo1994/pv-hawk:latest` when running the Docker container.
