=====================
Measures on microglia
=====================

- To provide some measurements, microglia are interpreted as graphs.
- The lengths are in physical unit if you didn't forget to calibrate your images.
- The final CSV contains the following data:

+-----------------------+------------------------------------------------------------------------------------+
| Column                | Description                                                                        |
+=======================+====================================================================================+
| Source                | Name of the image from which the following microglia were extracted.               |
+-----------------------+------------------------------------------------------------------------------------+
| # branches            | Total number of branches in the given cell.                                        |
+-----------------------+------------------------------------------------------------------------------------+
| # leaves              | Total number of leaves (== end point vertices) for this cell.                      |
+-----------------------+------------------------------------------------------------------------------------+
| # juctions            | Number of branches crossing in this cell (all vertices except leaves)              |
+-----------------------+------------------------------------------------------------------------------------+
| Average branch length | Average length of a branch for the given cell.                                     |
+-----------------------+------------------------------------------------------------------------------------+
| Total length          | Summed length of all the branches contained within this cell.                      |
+-----------------------+------------------------------------------------------------------------------------+
| Max branch length     | Length of the longest branch for this cell.                                        |
+-----------------------+------------------------------------------------------------------------------------+
| Label                 | Only useful for debuging. Index of the segmented microglia on the mask.            |
+-----------------------+------------------------------------------------------------------------------------+
| Class                 | Class (Amoeboid, Intermediate or Homeostatic) attributed to this cell              |
+-----------------------+------------------------------------------------------------------------------------+

- Here is an example file as it is produced by the plugin: you can download it from `this link <_images/results.csv>`_.
- Values are separated with tabulations.