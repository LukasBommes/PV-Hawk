PV Drone Inspect
================

Note: We are currently working on the initial release of the code and the documentation. Please consider the current state as preliminary.

**Documentation**: https://lukasbommes.github.io/PV-Drone-Inspect/

**Code**: https://lukasbommes.github.io/PV-Drone-Inspect/

PV Drone Inspect is a computer vision pipeline for the **automated inspection** of large-scale **photovoltaic (PV) plants** by means of **thermal infrared (IR) videos** acquired by a drone.

After recording an IR video of a PV plant, individual video frames and the corresponding coarse GPS position (latitude, longitude, and optionally altitude) of the drone are fed into PV Drone Inspect. PV Drone Inspect crops each PV module from each video frame and stores the resulting IR image patches. Geocoordinates of the PV module corners are obtained faciliating visualization of analysis results on a map. You can use the `PV Drone Inspect Viewer <https://github.com/LukasBommes/PV-Drone-Inspect-Viewer>`_ to browse the resulting map of your PV plant and to perform analyses, such as PV module defect prediction, based on the extracted IR images.

.. raw:: html

    <embed>
      <object data="high_level_overview.png" type="image/png" style="width: 100%;">
        <img src="docs/source/images/high_level_overview.png">
      </object>
    </embed>

The method is described in more detail in `How PV Drone Inspect Works <https://lukasbommes.github.io/PV-Drone-Inspect/method.html>`_ and in our journal papers:

[1] L. Bommes, T. Pickel, C. Buerhop-Lutz, J. Hauch, C. Brabec, I. Peters, ”Computer vision tool for detection, mapping, and fault classification of photovoltaics modules in aerial IR videos,” Progress in Photovoltaics: Research and Applications, 2021. [`Wiley PIP <https://onlinelibrary.wiley.com/doi/10.1002/pip.3448>`_, `ArXiv <https://arxiv.org/abs/2106.07314>`_]

[2] L. Bommes, T. Pickel, C. Buerhop-Lutz, J. Hauch, C. Brabec, I. Peters, ”Georeferencing of photovoltaic modules from aerial infrared videos using structure-from-motion,” Progress in Photovoltaics: Research and Applications, 2022 (submitted, acceptance pending).

[3] L. Bommes, M. Hoffmann, C. Buerhop-Lutz, T. Pickel, J. Hauch, C. Brabec, A. Maier, I. Peters, ”Anomaly detection in IR images of PV modules using supervised contrastive learning,” Progress in Photovoltaics: Research and Applications, 2022 (accepted for publication). [`ArXiv <https://arxiv.org/abs/2112.02922>`_]

PV Drone Inspect is a command line tool written in Python. It is free of charge, open-source, and MIT licensed. Source code is available on `GitHub <https://github.com/LukasBommes/PV-Drone-Inspect>`_.

Why is PV Drone Inspect needed?
-------------------------------

PV plants contain typically around 10 percent anomalous PV modules, which are potential fire hazards and cause significant power and yield losses. Thus, to enable safe and profitable operation PV plants should be regularly inspected. A popular inspection technique is drone-based thermal IR imaging, which detects anomalous PV modules in a contectless way based on heat dissipated in defective regions of the PV module. Thermal IR imaging has been applied sucessfully to small PV systems. However, when applied to large-scale PV plants with many thousands to millions of PV modules so much video data is produced that manual sighting is economically infeasible. This is where PV Drone Inspect comes into play for the fully automated processing of the generated IR videos.

Project status
--------------

PV Drone Inspect is a research project built during my PhD. In its current state PV Drone Inspect should be seen more as a proof-of-concept instead of a production-grade system. Please do not expect the pipeline to work smoothly and produce best results on the first attempt. Especially the OpenSfM-based reconstruction stage can be instable and may require multiple trials with different settings until you get a good result. When using your own IR videos it is important that you carefully follow the instructions in `Using Your Own Data <https://lukasbommes.github.io/PV-Drone-Inspect/using_own_data.html#using-own-data>`_. Furthermore, breaking changes to the configuration file specification and structure of the input and output files are possible.

Examplary results
-----------------

[Show some example maps of reconstructed PV plants]

Who are the target audiences?
-----------------------------

- Researchers who want to assemble large-scale IR image datasets of PV modules, for instance, to develop machine learning algorithms for defect detection, or power prediction.

- Companies or individuals who want to inspect their own PV plants or want to offer PV plant inspection as a service. 

How to use PV Drone Inspect?
----------------------------

Please follow the `Installation <https://lukasbommes.github.io/PV-Drone-Inspect/installation.html>`_ instructions to setup PV Drone Inspect on your machine. Afterwards, follow the `Tutorial <https://lukasbommes.github.io/PV-Drone-Inspect/tutorial.html>`_ to get started with an exemplary IR video dataset recorded by us. After you learned how to use PV Drone Inspect, you can proceed to `Using Your Own Data <https://lukasbommes.github.io/PV-Drone-Inspect/using_own_data.html#using-own-data>`_ to learn how to record suitable IR videos of your own PV plant with your own IR camera and drone.

What do you need to run PV Drone Inspect?
-----------------------------------------

PV Drone Inspect should be installed on a sufficiently powerful workstation with Ubuntu 18.04 or newer and CUDA-compatible GPU. We developed and tested PV Drone Inspect on a machine with Ubuntu 20.04 LTS, Intel Core i9-9900K, 64GB of DDR4 RAM, an SSD and a GeForce RTX 2080 Ti. Furthermore, you need a drone and a thermal IR camera. Details on this are provided in `Hardware setup <https://lukasbommes.github.io/PV-Drone-Inspect/using_own_data.html#hardware-setup>`_.
