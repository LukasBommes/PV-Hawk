PV Hawk
=======

**Documentation**: https://lukasbommes.github.io/PV-Hawk/

**Code**: https://lukasbommes.github.io/PV-Hawk/

PV Hawk is a computer vision pipeline for the **automated inspection** of large-scale **photovoltaic (PV) plants** by means of **thermal infrared (IR) videos** acquired by a drone.

After recording an IR video of a PV plant, individual video frames and the corresponding coarse GPS position (latitude, longitude, and optionally altitude) of the drone are fed into PV Hawk. PV Hawk crops each PV module from each video frame and stores the resulting IR image patches. Geocoordinates of the PV module corners are obtained and faciltate visualization of analysis results on a map. You can use the `PV Hawk Viewer <https://github.com/LukasBommes/PV-Hawk-Viewer>`_ to browse the resulting map of your PV plant, annotate extracted IR images, and perform analyses, such as defect prediction.

.. raw:: html

    <embed>
      <object data="high_level_overview.png" type="image/png" style="width: 100%;">
        <img src="docs/source/images/high_level_overview.png">
      </object>
    </embed>

PV Hawk implements the method described briefly in `How PV Hawk Works <https://lukasbommes.github.io/PV-Hawk/method.html>`_. For more details see our journal papers:

[1] L. Bommes, T. Pickel, C. Buerhop-Lutz, J. Hauch, C. Brabec, I. Peters, ”Georeferencing of photovoltaic modules from aerial infrared videos using structure-from-motion,” Progress in Photovoltaics: Research and Applications, 2022 (submitted, acceptance pending).

[2] L. Bommes, T. Pickel, C. Buerhop-Lutz, J. Hauch, C. Brabec, I. Peters, ”Computer vision tool for detection, mapping, and fault classification of photovoltaics modules in aerial IR videos,” Progress in Photovoltaics: Research and Applications, 2021. [`Wiley PIP <https://onlinelibrary.wiley.com/doi/10.1002/pip.3448>`_, `ArXiv <https://arxiv.org/abs/2106.07314>`_]

You may also find our related work on PV module defect detection interesting, which uses a dataset created with PV Hawk:

[3] L. Bommes, M. Hoffmann, C. Buerhop-Lutz, T. Pickel, J. Hauch, C. Brabec, A. Maier, I. Peters, ”Anomaly detection in IR images of PV modules using supervised contrastive learning,” Progress in Photovoltaics: Research and Applications, 2022 (accepted for publication). [`ArXiv <https://arxiv.org/abs/2112.02922>`_]

PV Hawk is a command line tool written in Python. It is free of charge, open-source, and MIT licensed.

Examplary results
-----------------

Shown below is a map of a PV plant with 13640 modules created by PV Hawk from an IR video with 42272 frames. The top image shows the PV module outlines. The bottom image visualizes the maximum temperature of each module, which facilitates fast detection of anomalous PV modules.

Note that the scope of PV Hawk is the mapping of the plant and extraction of module images. The temperature map is obtained with the `PV Hawk Viewer <https://github.com/LukasBommes/PV-Hawk-Viewer>`_.

We used standard GPS (as opposed to accurate RTK-GPS) and ignored the altitude measurement. The drone trajectory of the recording has a length of 7612 meters.

.. raw:: html

    <embed>
      <object data="example_outputs/module_layout.png" type="image/png" style="width: 100%;">
        <img src="docs/source/images/example_outputs/module_layout.png">
      </object>
    </embed>

.. raw:: html

    <embed>
      <object data="example_outputs/mean_of_max_temps_corrected.png" type="image/png" style="width: 100%;">
        <img src="docs/source/images/example_outputs/mean_of_max_temps_corrected.png">
      </object>
    </embed>

The next image shows the 3D reconstruction of a PV plant created with PV Hawk. Red rectangles resemble PV modules and the black line corresponds to reconstructed camera positions. Coordinates of the reconstruction are `WGS84 geocoordinates <https://en.wikipedia.org/wiki/World_Geodetic_System>`_, i.e latitude, longitude, and altitude.

.. raw:: html

    <embed>
      <object data="example_outputs/reconstruction.png" type="image/png" style="width: 100%;">
        <img src="docs/source/images/example_outputs/reconstruction.png">
      </object>
    </embed>

How to use PV Hawk?
-------------------

Please follow the `Installation <https://lukasbommes.github.io/PV-Hawk/installation.html>`_ instructions to setup PV Hawk on your machine. Afterwards, follow the `Tutorial <https://lukasbommes.github.io/PV-Hawk/tutorial.html>`_ to get started with an exemplary IR video dataset. After you learned how to use PV Hawk, you can proceed to `Using Your Own Data <https://lukasbommes.github.io/PV-Hawk/using_own_data.html#using-own-data>`_ to learn how to record suitable IR videos of your own PV plant with your own IR camera and drone.

Why is PV Hawk needed?
----------------------

PV plants contain typically around 10 percent anomalous PV modules, which are potential fire hazards and cause significant power and yield losses. Thus, to enable safe and profitable operation PV plants should be regularly inspected. A popular inspection technique is drone-based thermal IR imaging, which detects anomalous PV modules in a contectless way based on heat dissipated in defective regions of the PV module. Thermal IR imaging has been applied sucessfully to small PV systems. However, when applied to large-scale PV plants with many thousands to millions of PV modules so much video data is produced that manual sighting is economically infeasible. This is where PV Hawk comes into play for the fully automated processing of the generated IR videos.

Project status
--------------

PV Hawk is a research project built during my PhD. In its current state PV Hawk should be seen more as a proof-of-concept instead of a production-grade system. Please do not expect the pipeline to work smoothly and produce best results on the first attempt. Especially the OpenSfM-based reconstruction stage can be instable and may require multiple trials with different settings until you get a good result. When using your own IR videos it is important that you carefully follow the instructions in `Using Your Own Data <https://lukasbommes.github.io/PV-Hawk/using_own_data.html#using-own-data>`_. Furthermore, breaking changes to the configuration file specification and structure of the input and output files are possible.

Who are the target audiences?
-----------------------------

- Researchers who want to assemble large-scale IR image datasets of PV modules, for instance, to develop machine learning algorithms for defect detection, or power prediction.

- Companies or individuals who want to inspect their own PV plants or want to offer PV plant inspection as a service. 

What do you need to run PV Hawk?
--------------------------------

PV Hawk should be installed on a sufficiently powerful workstation with Ubuntu 18.04 or newer and CUDA-compatible GPU. We developed and tested PV Hawk on a machine with Ubuntu 20.04 LTS, Intel Core i9-9900K, 64GB of DDR4 RAM, an SSD and a GeForce RTX 2080 Ti. Furthermore, you need a drone and a thermal IR camera. Details on this are provided in `Hardware setup <https://lukasbommes.github.io/PV-Hawk/using_own_data.html#hardware-setup>`_.
