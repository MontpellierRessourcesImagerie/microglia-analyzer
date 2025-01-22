=========================
Quick start: A user guide
=========================

0. Introduction
===============

- "Microglia Analyzer" is a Python module allowing to segment, classify and measure microglia from 2D fluo images.
- This module is also a Napari plugin in case you would need a graphical interface.
- The segmentation produced doesn't allow to recover the actual surface of microglia since the output of the segmentation is a shape optimized for skeletonization.
- Each cell is classified according 3 classes: "amoeboid", "intermediate" and "homeostatic".
- The measures (bundled in a CSV file by the end) include:
   - The total length of each cell
   - The number of junctions for each cell
   - The number of leaves for each cell
   - The average branch length for each cell
- From the GUI (Napari), a batch mode is available allowing you run the whole workflow over an entire folder.

.. image:: _images/global/overview.png
      |   :height: 720px
      |   :align: center

1. Install the plugin 
=====================

- We strongly recommand to use `conda <https://docs.conda.io/en/latest/miniconda.html>`_ or any other virtual environment manager instead of installing Napari and microglia-analyzer in your system's Python.
- Napari is only required if you want to use microglia-analyzer with a graphical interface.
- Napari is not part of microglia-analyzer's dependencies, so you will have to install it separately.
- Each of the commands below is supposed to be ran after you activated your virtual environment.
- If the install was successful, you should see the plugin in Napari in the top bar menu: Plugins > Microglia Analyzer > Microglia Analyzer.

A. Development version
-----------------------

+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Method                | Instructions                                                                                                                                                                             |
+=======================+==========================================================================================================================================================================================+
| pip                   | :code:`pip install git+https://github.com/MontpellierRessourcesImagerie/microglia-analyzer.git`                                                                                          |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| GitHub                | :code:`pip install git+https://github.com/MontpellierRessourcesImagerie/microglia-analyzer.git`.                                                                                         |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| From an archive       | - Download `the archive <https://github.com/MontpellierRessourcesImagerie/microglia-analyzer/archive/refs/heads/main.zip>`_  :code:`pyproject.toml` and launch :code:`pip install -e .`. |
|                       | - From the terminal containing your virtual env, move to the folder containing the file :code:`pyproject.toml`                                                                           |
|                       | - Run the command :code:`pip install -e .`                                                                                                                                               |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

B. Stable version
-----------------

+-----------------------+------------------------------------------------------------------------------------+
| Method                | Instructions                                                                       |
+=======================+====================================================================================+
| pip                   | Activate your conda environment, and type :code:`pip install microglia-analyzer`.  |
+-----------------------+------------------------------------------------------------------------------------+
| NapariHub             | Go in the plugins menu of Napari and search for "Microglia Analyzer"               |
+-----------------------+------------------------------------------------------------------------------------+


2. Get your data ready
======================

From this point, we will only consider that you are using the Napari plugin. If you are using the Python module, you can look at the "__main__" section of the "ma_worker.py" file.
This Napari plugin expects your data to respect a precise format:

* The images must be available as dissociated TIFF images (no jpg, no 3D stack, no video, ...).
* The files must have exactly 1 channel. We recommend to keep your images in 16-bits grayscale.
* All the images that you want to process at the same time must be in the same folder.
* Avoid using special characters in the folder's name. (Tips: `Doranum <https://doranum.fr/stockage-archivage/comment-nommer-fichiers_10_13143_wgqw-aa59/>`_)
* Both models used by this plugin were trained on images having a pixel size of 0.325 µm, you way have troubles if your images have a different pixel size.

.. code-block:: bash

   .
   ├── 📁 my-awesome-experiment
   │   ├── some-image.tif
   │   ├── another-image.tif
   │   ├── ...
   │   └── last-image.tif


3. Workflow 
===========

a. Import your experiment (Media control)
----------------------------------------------

- In this section, you will import your experiment so you can visualize and analyze it.
- The :code:`Clear state` button is useless unless you just analyzed something else and need to reset the plugin.
- Use the :code:`Sources folder` to provide the path to the folder containing your images. It is the path of "my-awesome-experiment" in the previous example.
- In the drop-dowm menu below, all the images of your folder should show up. Ckicking on any of them should display it in the main viewer.
- You can adjust the contrast and brightness in the upper-left corner of Napari.


b. Calibration (Calibration)
----------------------------

- Providing measures in physical units requires the plugin to know the size of a pixel.
- Just fill the number field with the size of a pixel and select the unit in the drop-down menu next to it before clicking the :code:`Apply calibration` button.
- If you loose your image or if it becomes too small, you can click on the little "Home" button in the lower-left corner of Napari to center the view.

c. Segment the microglia (Segmentation)
----------------------------------------

+-----------------------+-------------------------------------------------------------------------------------------------------------+
| Setting               | Description                                                                                                 |
+=======================+=============================================================================================================+
| Min area (µm²)        | Minimal area that an object must reach to avoid being discarded.                                            |
+-----------------------+-------------------------------------------------------------------------------------------------------------+
| Min probability       | The raw output of the segmentation process is a probability map that must be thresholded to create a mask.  |
+-----------------------+-------------------------------------------------------------------------------------------------------------+

- You can set the "Min area" and the "Min probability" before or after the segmentation.
- You can just click the :code:`Segment microglia` button to start the segmentation.
- The first time, the model will have to be downloaded from internet.

d. Classify the microglia (Classification)
------------------------------------------

- Click the classify button.
- The model has to be downloaded from the internet the first time you use it.
- By the end of the process, the possible classes will show up below the :code:`Classify` button, with their assoiated color.
- Some elements are classified as "garbages" (aggregated objects, out-of-focus, filaments from other slices, ...) so you can use the :code:`Show garbage` checkbox to hide them.
- At this point, you must have each microglia surounded by a colored box, representing its class.

e. Extract measures + batch (Measures)
--------------------------------------------

- If you click the :code:`Measure` button, the plugin will compute the skeleton of each non-garbage object and extract the measures mentioned in the introduction.
- A new folder named "controls" should appear in the folder containing your images. It should contain a CSV file named after your image as well as a control image.
- If everything looks fine, you can click the :code:`Run batch` button to run the whole workflow over the entire folder using the same settings you just used.
- The number of images left to process should be displayed on the button now named :code:`Stop batch`.
- By the end of the process, your "controls" folder should contain a control image for each input image, and a unique CSV file ("results.csv") aggregating the measures of all the images.

4. Examples of processed data
=============================

.. tabs::

   .. tab:: Input images

      +----------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+
      | .. image:: _images/seg-ex/01-in.png                | .. image:: _images/seg-ex/02-in.png                         | .. image:: _images/seg-ex/03-in.png                          |
      |   :height: 128px                                   |   :height: 128px                                            |   :height: 128px                                             |
      |   :width: 128px                                    |   :width: 128px                                             |   :width: 128px                                              |
      |   :align: center                                   |   :align: center                                            |   :align: center                                             |
      +----------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+
      | .. image:: _images/seg-ex/04-in.png                | .. image:: _images/seg-ex/05-in.png                         | .. image:: _images/seg-ex/06-in.png                          |
      |   :height: 128px                                   |   :height: 128px                                            |   :height: 128px                                             |
      |   :width: 128px                                    |   :width: 128px                                             |   :width: 128px                                              |
      |   :align: center                                   |   :align: center                                            |   :align: center                                             |
      +----------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+

   .. tab:: Segmented images

      +----------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+
      | .. image:: _images/seg-ex/01-out.png               | .. image:: _images/seg-ex/02-out.png                        | .. image:: _images/seg-ex/03-out.png                         |
      |   :height: 128px                                   |   :height: 128px                                            |   :height: 128px                                             |
      |   :width: 128px                                    |   :width: 128px                                             |   :width: 128px                                              |
      |   :align: center                                   |   :align: center                                            |   :align: center                                             |
      +----------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+
      | .. image:: _images/seg-ex/04-out.png               | .. image:: _images/seg-ex/05-out.png                        | .. image:: _images/seg-ex/06-out.png                         |
      |   :height: 128px                                   |   :height: 128px                                            |   :height: 128px                                             |
      |   :width: 128px                                    |   :width: 128px                                             |   :width: 128px                                              |
      |   :align: center                                   |   :align: center                                            |   :align: center                                             |
      +----------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+

   .. tab:: Classified images

      +----------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+
      | .. image:: _images/seg-ex/01-out.png               | .. image:: _images/seg-ex/02-out.png                        | .. image:: _images/seg-ex/03-out.png                         |
      |   :height: 128px                                   |   :height: 128px                                            |   :height: 128px                                             |
      |   :width: 128px                                    |   :width: 128px                                             |   :width: 128px                                              |
      |   :align: center                                   |   :align: center                                            |   :align: center                                             |
      +----------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+
      | .. image:: _images/seg-ex/04-out.png               | .. image:: _images/seg-ex/05-out.png                        | .. image:: _images/seg-ex/06-out.png                         |
      |   :height: 128px                                   |   :height: 128px                                            |   :height: 128px                                             |
      |   :width: 128px                                    |   :width: 128px                                             |   :width: 128px                                              |
      |   :align: center                                   |   :align: center                                            |   :align: center                                             |
      +----------------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------+


7. Notes 
========

- The plugin provides verbose output, so it's recommended to monitor the terminal if you want detailed information about its actions.
- If a crash occurs, please `create an issue <https://github.com/MontpellierRessourcesImagerie/proto-swelling-analyzer/issues>`_ and include the relevant image(s) for further investigation.
- Napari currently supports only open file formats, so make sure to convert your images to TIFF format before using them with Napari.
