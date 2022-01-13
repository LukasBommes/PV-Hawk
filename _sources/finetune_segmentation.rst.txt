Finetuning the Mask R-CNN Model
===============================

PV Drone Inspect uses Mask R-CNN for instance segmentation of PV modules in IR video frames. It is pretrained on a large PV module dataset. However, if you encounter issues with the accuracy of the Mask R-CNN model, you may wish to fine-tune the model on your own dataset. For this, we recommend annotating data using the `Grid Annotation Tool <https://github.com/LukasBommes/Grid-Annotation-Tool>`_. Data labelled with this tool can be directly used for training the Mask R-CNN model.

To train/fine-tune Mask R-CNN start jupyter lab in the interactive Docker session

.. code-block:: console

  jupyter lab --allow-root --ip=0.0.0.0 --port=8888
    
and open the displayed URL in the web browser on your machine. In jupyter lab navigate to `extractor/segmentation` and open the `train.ipynb` notebook.

Prior to training the model with this script, you may need to edit the training config in `extractor/segmentation/configs.py` in the same directory. Also make sure the training dataset is available at the location specified in `DATASET_PATH` in `extractor/segmentation/configs.py`.

The training dataset should contain the following two folders:

- `images_radiometric`: Your custom 16-bit IR video frames containing PV modules.
- `annotations`: JSON file for each image with annotated PV modules. The annotation can be created with the `Grid Annotation Tool <https://github.com/LukasBommes/Grid-Annotation-Tool>`_.

After training weights of the new model with be available in `extractor/segmentation/Mask_RCNN/ logs/pv_modules\<timestamp\>/mask_rcnn_pv_modules_\<epoch\>.h5` where `<timestamp>` is the timestamp at the beginning of training and `<epoch>` the number of epochs the model was trained. To use the newly trained model set the `WEIGHTS_FILE` parameter in `extractor/segmentation/configs.py` to the path of the `*.h5` weights file.
