Build the Docker Image
======================

You can build the Docker image locally based on the specification provided in the `Dockerfile`. To this end, run the following command from the root directory of the PV Drone Inspect source code

.. code-block:: console

  sudo docker build . --tag=pv-drone-inspect-custom-build
  
Now, you can use this image by stating `pv-drone-inspect-custom-build` instead of `lubo1994/pv-drone-inspect:latest` when running the Docker container.
