=========================
Quick start: A user guide
=========================


1. Install the plugin 
=====================

+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Method                | Instructions                                                                                                                                                             |
+=======================+==========================================================================================================================================================================+
| pip                   | **Stable version:** Activate your conda environment, and type :code:`pip install microglia-analyzer`. The plugin should then appear in your plugins list in Napari.      |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| NapariHub             | **Stable version:** Go in the plugins menu of Napari and search for "Microglia Analyzer"                                                                                 |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| GitHub                | **Dev version:** Activate your conda environment and use the command :code:`pip install git+https://github.com/MontpellierRessourcesImagerie/microglia-analyzer.git`.    |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| From an archive       | Uncompress the archive, place your terminal in the folder containing :code:`pyproject.toml` and launch :code:`pip install -e .`.                                         |
+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

If the install was successful, you should see the plugin in Napari in the top bar menu: Plugins > Microglia Analyzer > Microglia Analyzer.

2. Quick demonstration 
======================

.. raw:: html

   <iframe width="672" height="378" src="https://www.youtube.com/embed/cy3vnBahIas" title="Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


3. Get your data ready
======================

This Napari plugin expects your data to respect a precise format:

* The frames must be dissociated in individual TIFF images (no video).
* The files must have exactly 3 channels (no grayscale, no alpha channel).
* The timestamp until the milliseconds must be encoded in the name of each file, with the form: :code:`hh-mm-ss.SSS.tif`.
* All the frames must be located in the same folder.
* Avoid using special characters in the folder's name. (Tips: `Doranum <https://doranum.fr/stockage-archivage/comment-nommer-fichiers_10_13143_wgqw-aa59/>`_)
* Be aware that the plugin creates data along its execution. This data is located next the folder containing the frames. This data is approximately 60% of the original's size. Be sure to have enough spare space on your disk.

.. code-block:: bash

   .
   ├── 📁 experiment-2024-06-26-v001
   │   ├── 14-25-23.336.tif
   │   ├── 14-25-23.379.tif
   │   ├── ...
   │   └── 14-25-36.949.tif


4. Settings
===========

+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| Name          | Description                                                                                                                        |
+===============+====================================================================================================================================+
| Frame         | Index of the frame that you are currently visualizing in the viewport. **Starts at 1**.                                            |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| Calibration   | Length, in physical unit, of the line that you drew on the active shape layer.                                                     |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| Median blur   | For protoplasts detection. Applies a median blur of this radius to the activity map to remove noise.                               |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| LoG radius    | For protoplasts detection. Applies a Laplacian of Gaussian filter to the activity map to find the center of protoplasts.           |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| Threshold     | For protoplasts detection. Minimum amount of activity to reach to consider that we have a protoplast.                              |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+


5. Workflow 
===========

a. Import your experiment (Experiment manager)
----------------------------------------------

- In this section, you will import your experiment so you can visualize and analyze it.
- The :code:`Clear state` button is useless unless you just analyzed something else and need to reset the plugin.
- Use the :code:`Experiment directory` to provide the path to the folder containing your frames. If you already performed an analysis or if you stopped in the process, you can select the `.data` folder associated to your experiment.
- The :code:`Backward` and :code:`Forward` buttons allow you to navigate by 15 frames hops.
- The slider allows you to visualize your frames freely.
- If you need to visualize a precise frame, you can provide its index in the :code:`Frame` slot.
- The three last lines in this group are: 
   1. The ellapsed time in seconds from the first frame to the one you are looking at. 
   2. The name of the experiment.
   3. The total duration of the experiment in seconds.


b. Calibration (Calibration)
----------------------------

- To create our patches, we need to know the physical size of a pixel. If you know the pixel size, you can simply fill the :code:`Calibration` slot and provide the correct unit.
- If you don't know the pixel size, you can:
   - Add a new shape layer in your project.
   - Make a line over a structure of which you know the length (ex: the width of the chip). The line is so thin that you won't see it in the viewport.
   - Fill the :code:`Calibration` with the length and unit of this structure.
   - Make sure to have your shape layer still active (its name must be on a blue background in the layers list), and click :code:`Apply calibration`.
   - The last line of text of the box should now indicate the size of a pixel.

c. Generate the patches (Create patches)
----------------------------------------

- For a better understanding of how and why the patches are created, check the ":doc:`make_patches`" chapter.
- Now the goal will be to adjust the three settings :code:`Median blur`, :code:`LoG radius` and :code:`Threshold`.
- To do so, you need a reference. So start by clicking the :code:`Reprocess patches` button to get your starting point.
- If your acquisition are noisy or if you caught small debris, increase the median blur radius.
- Adjust your LoG radius to be approximately half the diameter of a protoplast (in pixels).
- Decrease the threshold value until you removed every slow object and every protoplast that explodes along time.
- For every change you make, don't forget to hit :code:`Reprocess patches` again.

d. Segment the protoplasts and measure them (Segment & measure)
---------------------------------------------------------------

- Start by exporting the patches that you just created with the :code:`Export patches` button. It might take a few seconds to start when you click the button, don't mash it.
- It will write on the disk the patches as they are expected in input of our segmentation algorithm.
- Once its done, you can click the :code:`Segment protoplasts` button.
- This last step takes a bunch of time as it is deep-learning based (with `StarDist <https://arxiv.org/abs/1806.03535>`_).
- You can visualize the progress by clicking the "Activity" button in the bottom right corner of Napari's window.
- Once it's over, the segmentation will show up above your image, and a results table will show up.
- In this results table, the measures are areas in µm².
- Each line is a time-point. 
- The first column is the ellapsed time in seconds. 
- The next columns are for the segmentation: Pxxx indicates that this measure comes from the xxx patch, the L is the unique identifier of the protoplast on the chip.

e. Edit the segmentation (Segment & measure)
--------------------------------------------

- The segmentation may produce false positives or false negatives. You can manually edit the labels in the viewer.
- If you want to do so, the easiest way is to keep your left hand on the keyboard, and the right one on the mouse:
   - Left ring finger: :code:`E` key: Allows you to get one frame back.
   - Left middle finger: :code:`R` key: Allows you to fill and save the current labels.
   - Left index finger: :code:`T` key: Allows you to get one frame forward.
   - Left thumb: :code:`Space` key: Allows you to switch between the drawing and moving mode.

6. Examples of segmented patches
================================

.. tabs::

   .. tab:: Input patches

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

   .. tab:: Labeled patches

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
