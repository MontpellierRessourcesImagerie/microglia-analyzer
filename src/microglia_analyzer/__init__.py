__version__ = "1.0.3"

import re

ORIGINAL_PIXEL_SIZE = 0.325
ORIGINAL_UNIT = "µm"
TIFF_REGEX = re.compile(r"(.+)\.tiff?", re.IGNORECASE)

"""
The networks used in this package (A 2D UNet and a YOLOv5) rely on images that have a pixel size of 0.325 µm.
Images with a different pixel size will be resized to artificially have a pixel size of 0.325 µm.
It is from these resized images that we will extract the patches.
"""
