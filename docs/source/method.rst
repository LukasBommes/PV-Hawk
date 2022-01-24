How PV Hawk Works
=================

Inputs to PV Hawk are IR video frames with the corresponding coarse GPS position (latitude, longitude, and optionally altitude) of the drone. Outputs are multiple cropped and rectified IR image patches per PV module and geocoordinates of the four corners of each PV module. PV Hawk first segments all PV modules in each video frame with a Mask R-CNN instance segmentation model. Each module is then tracked over subsequent frames. The corresponding image regions are cropped, rectified, and stored to disk with a unique tracking ID. 

For the localization of PV modules in the PV plant a flexible, layout-independent, and fully automated solution based on structure from motion is provided. The `OpenSfM <https://opensfm.org>`_ structure from motion library is used to reconstruct the 6-DOF camera trajectory.


 Since the coarse GPS trajectory of the drone is known the camera trajectory is reconstructed in geocoordinates. Based on the known camera coordinates, PV module corners are triangulated into the 3D reconstruction and refined with an iterative graph optimization, yielding absolute geocoordinates for each PV module (see fig. 3).

.. image:: images/overview.png


- reference to output dataset strcuture and config file reference
