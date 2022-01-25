How PV Hawk Works
=================

PV Hawk is a computer vision pipeline, which processes thermal IR video of large-scale PV plants fulfilling two main tasks:

#. Extracting of IR image patches of each PV module in each video frame
#. Obtaining WGS84 geocoordinates of the corner points of each PV module

An overview of the method is shown in :numref:`overview`. First, IR videos are split into individual frames and the GPS trajectory of the drone is extracted and interpolated. PV modules are then segmented by Mask R-CNN, tracked over subsequent frames, extracted and stored to disk. To georeference PV modules, a subset of keyframes is selected based on travelled GPS distance and visual overlap. Subsequently, a georeferenced 3D reconstruction of the PV plant is obtained by incremental SfM (using `OpenSfM <https://opensfm.org>`_) alongside the 6-DOF camera pose of each keyframe. This requires calibrated camera parameters, which are obtained beforehand. The known keyframe poses are then used to triangulate observed PV modules into the 3D reconstruction, yielding the desired module geocoordinates.

.. _overview:
.. figure:: images/overview.png

  Overview of the PV Hawk pipeline.

The :doc:`config_file_reference` contains more details on the different tasks of the pipeline and the resulting output dataset structure is described in more detail in :doc:`output_directory`.
