# Microglia Analyzer

![GitHub License](https://img.shields.io/github/license/MontpellierRessourcesImagerie/microglia-analyzer)
![Python Version](https://img.shields.io/badge/Python-3.9|3.10|3.11-blue?logo=python)
![Unit tests](https://img.shields.io/github/actions/workflow/status/MontpellierRessourcesImagerie/microglia-analyzer/test_and_deploy.yml?logo=pytest&label=tests)

# What is it?

A Napari plugin that allows for the segmentation and detection of microglia on 2D fluorescent images.
Z-stacks are not handled.

It consists in:
- Segmenting the microglia with a UNet2D model.
- Create classified bounding-boxes with a YOLOv5.
- Using some morphology to extract metrics such as:
    - The number of branches.
    - The number of leaves (== end points).
    - The number of vertices (== internal crossings).
    - The mean branch length.
    - The total length.
    - The length of the longest path.
    
- We end-up with a ".csv" (separated with tabulations instead of commas) file containing all these metrics.

# 01. How to install/upgrade it?

## Install

```
pip install git+https://github.com/MontpellierRessourcesImagerie/microglia-analyzer.git napari[all]
```

## Upgrade

```
pip install --upgrade git+https://github.com/MontpellierRessourcesImagerie/microglia-analyzer.git
```

# 02. How to use it?

# A. Open the widget

- First of all, you need all your images to be converted to TIFF, and placed in the same folder.
- Once Napari is opened, you can visit your plugins list. It should contain a `Microglia Analyzer` entry.
    - `Tiles Creator` allows you to create patches to annotate if you ever want to retrain the UNet or the YOLOv5.
    - `Annotations Helper` allows you to define and create classified bounding-boxes exported under the YOLOv5 format as well as masks. This widget is meant to help you create some ground-truth data.
    - `Microglia Analyzer` contains the whole analysis workflow.

# B. Load your images

- Click the `📁 Sources folder` button and navigate to the folder containing your TIFF images before pressing OK.
- In the drop-down menu below, you can choose the image on which you want to run the analysis.
- In the "Calibration" field, you just have to provide the size of your pixels in physical unit and confirm. Your image may look small after this step, so you may want to press the button with a little picture of home in the lower-left corner of Napari's window.

# C. Segment your microglia

- Press the `🔍 Segment` button and wait for the labeled microglia to show up.
- The first time, this step may take a little longer as the plugin must download the deep-learning model from MRI's server.
- You can adjust the area of the smallest tolerated object either before or after the segmentation, both ways work.
- At this point, each individual microglia should be represented by its own color.
- We focused the segmentation on the "graph" produced by microglia, so the soma won't look thicker than any other part of the microglia.

# D. Classify your microglia

- Click the `Classify` button.
- Once again, a model has to be downloaded from MRI's server.
- By the end of this step, you should have a colored bounding-box around each tolerated microglia.
- The color indicates which class it belongs to (amoeboid, intermediate ou homeostatic).
- The color code should show up in the array below the classification button.
- If you wish, you can adjust the prediction threshold ("how sure the model is about what it sees")

# E. Measure your microglia

- If you click on the `📊 Measure` button, the skeleton of each microglia should appear.
- The measures are generated and stored in a `.tsv` file located in the `controls` sub-folder (auto-generated in your images' folder)

# F. Analyze the whole folder

- Once you got to that point, each parameter is correctly set (as you just used them for an image), so you click the `▶ Run batch` to apply them to the whole folder.
- The button should now indicate `■ Kill batch (i/N)`. You can click it to interrupt the execution. `i` is the rank of the current image, and `N` is the number of images detected in the folder.
- By the end of the run, the button should be normal again, and a `results.tsv` file should be located in the `controls` folder.

--------

[🐛 Found a bug?]: https://github.com/MontpellierRessourcesImagerie/microglia-analyzer/issues
[🔍 Need some help?]: mri-cia@mri.cnrs.fr
