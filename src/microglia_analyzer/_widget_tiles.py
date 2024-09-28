from qtpy.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, 
                            QSpinBox, QHBoxLayout, QPushButton, 
                            QFileDialog, QComboBox, QLabel,
                            QCheckBox, QSpinBox, QSlider, QLineEdit)

from qtpy.QtCore import QThread, Qt

from PyQt5.QtGui import QFont, QDoubleValidator
from PyQt5.QtCore import pyqtSignal

import napari
from napari.utils.notifications import show_info
from napari.utils import progress

import tifffile
import numpy as np
import math
import os
import re
import random
import cv2
import shutil

from microglia_analyzer import TIFF_REGEX
from microglia_analyzer.tiles.tiler import ImageTiler2D
from microglia_analyzer.tiles.recalibrate import (recalibrate_shape, 
                                                  recalibrate_image, 
                                                  process_factor)

_DEFAULT_PATCH_SIZE = 512
_DEFAULT_OVERLAP    = 128

_PREVIEW_IMAGE_LAYER = "Whole-image"
_PREVIEW_SHAPE_LAYER = "Tile-box-"
_FOLDER_MOSAIC_LAYER = "Folder-mosaic"


class TilesCreatorWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()
        # Size of a pixel in the input images, in physical units
        self.pixel_size = None
        # Unit of the pixel size
        self.calib_unit = None
        # Size (in pixels) of the tiles to be exported
        self.patch_size = _DEFAULT_PATCH_SIZE
        # Overlap (in pixels) between the tiles
        self.overlap    = _DEFAULT_OVERLAP
        # Path of the folder containing the TIFF to convert in tiles.
        self.in_path    = None
        # Path of the folder where the tiles will be exported.
        self.out_path   = None
        self.font = None
        self.init_ui()

    # -------- UI: ----------------------------------

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.font = QFont()
        self.font.setFamily("Arial Unicode MS, Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji")
        self.media_control_panel()
        self.calibration_panel()
        self.normalization_panel()
        self.configure_tiles_panel()
        self.export_tiles_group()
        self.setLayout(self.layout)
    
    def media_control_panel(self):
        media_control_group = QGroupBox("Media Control")
        layout = QVBoxLayout()

        # Reset button
        self.clear_state_button = QPushButton("❌ Clear state")
        self.clear_state_button.setFont(self.font)
        self.clear_state_button.clicked.connect(self.clear_state)
        layout.addWidget(self.clear_state_button)

        # Some vertical spacing
        layout.addSpacing(20)

        # Load button
        self.load_button = QPushButton("📂 Load")
        self.load_button.setFont(self.font)
        self.load_button.clicked.connect(self.load_folder)
        layout.addWidget(self.load_button)

        # Show data sample button
        self.show_sample_button = QPushButton("🔍 Show sample")
        self.show_sample_button.setFont(self.font)
        self.show_sample_button.clicked.connect(self.build_folder_preview)
        layout.addWidget(self.show_sample_button)

        # Number of images label
        self.n_sources_label = QLabel("🖼️ sources: ---")
        self.n_sources_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.n_sources_label)

        # Detected shape label
        self.shape_label = QLabel("Shape: (XXX, XXX)")
        self.shape_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.shape_label)

        media_control_group.setLayout(layout)
        self.layout.addWidget(media_control_group)
    
    def calibration_panel(self):
        self.calibration_group = QGroupBox("Calibration")
        layout = QVBoxLayout()
        nav_layout = QHBoxLayout()

        # Use calibration checkbox
        self.use_calibration = QCheckBox("Use calibration")
        self.use_calibration.setChecked(True)
        self.use_calibration.stateChanged.connect(self.update_calibration)
        layout.addWidget(self.use_calibration)

        # Create QLineEdit for float input
        self.calibration_input = QLineEdit()
        float_validator = QDoubleValidator()
        float_validator.setNotation(QDoubleValidator.StandardNotation)
        self.calibration_input.setValidator(float_validator)
        nav_layout.addWidget(self.calibration_input)

        # Create QComboBox for unit selection
        self.unit_selector = QComboBox()
        units = ["nm", "μm", "mm", "cm", "dm", "m"]  # Define the units from nanometers to meters
        self.unit_selector.addItems(units)
        nav_layout.addWidget(self.unit_selector)

        # Add the nav_layout to the calibration layout
        layout.addLayout(nav_layout)

        # Apply calibration button
        self.calibrationButton = QPushButton("📏 Apply calibration")
        self.calibrationButton.setFont(self.font)
        self.calibrationButton.clicked.connect(self.apply_calibration)
        layout.addWidget(self.calibrationButton)

        # Display pixel size:
        self.pixel_size_label = QLabel("Pixel size: ---")
        self.pixel_size_label.setAlignment(Qt.AlignCenter)
        self.pixel_size_label.setMaximumHeight(30)
        layout.addWidget(self.pixel_size_label)

        # Scaling factor label
        self.scaling_factor_label = QLabel("Scaling factor: x---")
        self.scaling_factor_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.scaling_factor_label)

        self.calibration_group.setLayout(layout)
        self.layout.addWidget(self.calibration_group)

    def normalization_panel(self):
        self.normalization_group = QGroupBox("Normalization")
        layout = QVBoxLayout()

        # Use normalization checkbox
        self.use_normalization = QCheckBox("Use normalization")
        layout.addWidget(self.use_normalization)

        # Lower bound input float
        h_layout = QHBoxLayout()
        self.lower_bound_label = QLabel("Lower bound:")
        self.lower_bound_input = QLineEdit()
        float_validator = QDoubleValidator()
        float_validator.setNotation(QDoubleValidator.StandardNotation)
        self.lower_bound_input.setValidator(float_validator)
        self.lower_bound_input.setText("0.0")
        h_layout.addWidget(self.lower_bound_label)
        h_layout.addWidget(self.lower_bound_input)
        layout.addLayout(h_layout)

        # Upper bound input float
        h_layout = QHBoxLayout()
        self.upper_bound_label = QLabel("Upper bound:")
        self.upper_bound_input = QLineEdit()
        float_validator = QDoubleValidator()
        float_validator.setNotation(QDoubleValidator.StandardNotation)
        self.upper_bound_input.setValidator(float_validator)
        self.upper_bound_input.setText("1.0")
        h_layout.addWidget(self.upper_bound_label)
        h_layout.addWidget(self.upper_bound_input)
        layout.addLayout(h_layout)

        # Init checkbox state
        self.use_normalization.stateChanged.connect(self.update_normalization)
        self.use_normalization.setChecked(True)

        self.normalization_group.setLayout(layout)
        self.layout.addWidget(self.normalization_group)
    
    def configure_tiles_panel(self):
        self.tiles_group = QGroupBox("Tiles configuration")
        layout = QVBoxLayout()

        # Patch size
        h_layout = QHBoxLayout()
        patch_size_label = QLabel("Patch size:")
        self.patch_size_input = QSpinBox()
        self.patch_size_input.setRange(1, 10000)
        self.patch_size_input.setValue(_DEFAULT_PATCH_SIZE)
        self.patch_size_input.valueChanged.connect(self.update_patch_size)
        h_layout.addWidget(patch_size_label)
        h_layout.addWidget(self.patch_size_input)
        layout.addLayout(h_layout)

        # Overlap size
        h_layout = QHBoxLayout()
        overlap_label = QLabel("Overlap:")
        self.overlap_input = QSpinBox()
        self.overlap_input.setRange(1, 10000)
        self.overlap_input.setValue(_DEFAULT_OVERLAP)
        self.overlap_input.valueChanged.connect(self.update_overlap)
        h_layout.addWidget(overlap_label)
        h_layout.addWidget(self.overlap_input)
        layout.addLayout(h_layout)

        # Preview tiles button
        self.tilesButton = QPushButton("💡 Preview tiles")
        self.tilesButton.setFont(self.font)
        self.tilesButton.clicked.connect(self.preview_tiles)
        layout.addWidget(self.tilesButton)

        # Slider to change the preview layout/box
        h_layout = QHBoxLayout()
        self.preview_slider = QSlider(Qt.Horizontal)
        self.preview_slider.setMinimum(0)
        self.preview_slider.setMaximum(0)
        self.preview_slider.setValue(0)
        self.preview_slider.setTickInterval(1)
        self.preview_slider.setTickPosition(QSlider.TicksBelow)
        self.preview_slider.valueChanged.connect(self.update_patch_preview)
        h_layout.addWidget(self.preview_slider)
        self.slider_value_label = QLabel("0")
        h_layout.addWidget(self.slider_value_label)
        layout.addLayout(h_layout)

        # Show all checkbox
        self.show_all_tiles = QCheckBox("Show all tiles")
        self.show_all_tiles.stateChanged.connect(self.update_show_all_tiles)
        layout.addWidget(self.show_all_tiles)

        self.tiles_group.setLayout(layout)
        self.layout.addWidget(self.tiles_group)
    
    def export_tiles_group(self):
        self.export_group = QGroupBox("Export configuration")
        layout = QVBoxLayout()

        # Select export folder button
        self.exportFolderButton = QPushButton("📂 Select export folder")
        self.exportFolderButton.setFont(self.font)
        self.exportFolderButton.clicked.connect(self.choose_folder)
        layout.addWidget(self.exportFolderButton)

        # Empty export folder button
        self.emptyFolderButton = QPushButton("🗑️ Empty export folder")
        self.emptyFolderButton.setFont(self.font)
        self.emptyFolderButton.clicked.connect(self.empty_folder)
        layout.addWidget(self.emptyFolderButton)

        # Export tiles button
        self.exportButton = QPushButton("📦 Export tiles")
        self.exportButton.setFont(self.font)
        self.exportButton.clicked.connect(self.export_tiles)
        layout.addWidget(self.exportButton)

        self.export_group.setLayout(layout)
        self.layout.addWidget(self.export_group)
    
    # -------- CALLBACKS: ------------------------------

    def clear_state(self):
        self.pixel_size = None
        self.calib_unit = None
        self.patch_size = _DEFAULT_PATCH_SIZE
        self.overlap    = _DEFAULT_OVERLAP
        self.in_path    = None
        self.out_path   = None
        self.calibration_input.setText("")
        self.unit_selector.setCurrentText("nm")
        self.lower_bound_input.setText("0.0")
        self.upper_bound_input.setText("1.0")
        self.use_calibration.setChecked(True)
        self.use_normalization.setChecked(True)
        self.update_calibration()
        self.update_normalization()
        self.update_patch_size()
        self.update_overlap()
        self.reset_preview_boxes()
        if _FOLDER_MOSAIC_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_FOLDER_MOSAIC_LAYER)
        self.pixel_size_label.setText("Pixel size: ---")
        self.scaling_factor_label.setText("Scaling factor: x---")
        self.shape_label.setText("Shape: (XXX, XXX)")
        self.n_sources_label.setText("🖼️ sources: ---")

    def load_folder(self):
        """
        Prompts the use to select a folder containing TIFF images.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select sources folder")
        if (folder_path is None) or (folder_path == ""):
            return
        self.set_sources_folder(folder_path)
    
    def update_calibration(self):
        """
        The user has the possibility to export patches with a specific calibration to match the original images.
        They also have the possibility to not use calibration.
        In the case of using calibration, images will be rescaled to have a pixel size of 0.325 µm.
        Otherwise, they will be left untouched.
        """
        status = self.use_calibration.isChecked()
        self.calibration_input.setEnabled(status)
        self.unit_selector.setEnabled(status)
        self.calibrationButton.setEnabled(status)
        if status:
            # We restore the calibration to the last known one is the checkbox is checked again.
            self.set_calibration(self.pixel_size, self.calib_unit)
        else:
            self.pixel_size_label.setText("Pixel size: 1pxl")
            self.scaling_factor_label.setText("Scaling factor: x1.0")

    def apply_calibration(self):
        length = float(self.calibration_input.text())
        unit = self.unit_selector.currentText()
        pixelSize, unit = self.convert_to_optimal_unit(length, unit)
        self.set_calibration(pixelSize, unit)

    def update_normalization(self):
        active = self.use_normalization.isChecked()
        self.lower_bound_input.setEnabled(active)
        self.upper_bound_input.setEnabled(active)
        self.lower_bound_label.setEnabled(active)
        self.upper_bound_label.setEnabled(active)

    def update_patch_size(self):
        self.patch_size = int(self.patch_size_input.value())

    def update_overlap(self):
        self.overlap = int(self.overlap_input.value())
    
    def reset_preview_boxes(self):
        """
        Each bounding box is stored in its own layer for the preview.
        One of the images to which they correspond are is displayed below them.
        This function removes all the bounding boxes and the image.
        """
        names = [layer.name for layer in self.viewer.layers]
        # The bounding-box layers
        for name in names:
            if name.startswith(_PREVIEW_SHAPE_LAYER):
                self.viewer.layers.remove(name)
        # The whole image layer
        if _PREVIEW_IMAGE_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_PREVIEW_IMAGE_LAYER)
        # The data sample layer
        if _FOLDER_MOSAIC_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_FOLDER_MOSAIC_LAYER)

    def pick_random_image(self):
        """
        Returns the name (not the path) of a random TIFF file from the input folder.
        """
        tiff_list = [f for f in os.listdir(self.in_path) if re.match(TIFF_REGEX, f)]
        random.shuffle(tiff_list)
        return tiff_list[0]
    
    def is_setup_ready(self):
        # Is the input folder path correctly configured
        if (self.in_path is None) or (self.in_path == ""):
            show_info("Please select a folder with tiff images.")
            return False
        # If we want to use the calibration, did we provide the pixel size and unit?
        if self.use_calibration.isChecked():
            if (self.pixel_size is None) or (self.calib_unit is None):
                show_info("Please calibrate the images first.")
                return False
        # Did we configure the tiles (size and overlap)?
        if (self.patch_size is None) or (self.overlap is None):
            show_info("Please configure the tiles first.")
            return False
        return True

    def preview_tiles(self):
        if not self.is_setup_ready():
            show_info("Please configure the tiles first.")
            return
        self.reset_preview_boxes()
        # Open random image to show tiles over it.
        im_data = tifffile.imread(os.path.join(self.in_path, self.pick_random_image()))
        if self.use_calibration.isChecked():
            # Rescale the image to have a pixel size of 0.325 µm.
            im_data = recalibrate_image(im_data, self.pixel_size, self.calib_unit)
        l = self.viewer.add_image(im_data, name=_PREVIEW_IMAGE_LAYER, colormap='gray')
        if self.use_calibration.isChecked():
            l.scale = (0.325, 0.325)
        tiler = ImageTiler2D(self.patch_size, self.overlap, im_data.shape)
        for i, tile in enumerate(tiler.get_layout()):
            l = self.viewer.add_shapes(
                [tile.as_napari_rectangle()], 
                name=_PREVIEW_SHAPE_LAYER + str(i).zfill(3), 
                face_color='transparent', 
                edge_color='red', 
                edge_width=6
            )
            if self.use_calibration.isChecked():
                l.scale = (0.325, 0.325)
        # The slider allows to hide the boxes and show them one by one.
        show_info(f"Cut {len(tiler.get_layout())} tiles from the image.")
        self.preview_slider.setMinimum(0)
        self.preview_slider.setMaximum(len(tiler.get_layout())-1)
        self.update_patch_preview()
    
    def update_patch_preview(self):
        """
        Function allowing to show only one bounding box at a time.
        Bounding boxes are stored in layers with names starting with _PREVIEW_SHAPE_LAYER.
        There is the possibility to show them all at once with a checkbox.
        In this case, the active one shows up red and the others blue.
        """
        index = int(self.preview_slider.value())
        self.slider_value_label.setText(str(index))
        target = _PREVIEW_SHAPE_LAYER + str(index).zfill(3)
        if target not in self.viewer.layers:
            return
        for layer in self.viewer.layers:
            if layer.name.startswith(_PREVIEW_SHAPE_LAYER):
                layer.selected_data = set()
                layer.visible = False
                layer.opacity = 0.5
                layer.edge_color = "blue"
                layer.edge_width = 3
        l = self.viewer.layers[target]
        l.visible = True
        l.opacity = 1.0
        l.edge_color = "red"
        l.edge_width = 6
    
    def update_show_all_tiles(self):
        """
        Function allowing to show all the bounding boxes at once.
        """
        status = self.show_all_tiles.isChecked()
        if status:
            self.preview_slider.setEnabled(False)
            for layer in self.viewer.layers:
                if layer.name.startswith(_PREVIEW_SHAPE_LAYER):
                    layer.visible = True
        else:
            self.preview_slider.setEnabled(True)
            self.update_patch_preview()

    def choose_folder(self):
        """
        Function to choose the folder where the tiles will be exported.
        Prompts the user to select a folder.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select export folder")
        if (folder_path is None) or (folder_path == ""):
            return
        if self.in_path == folder_path:
            show_info("Please select a folder different from the input folder.")
            return
        self.out_path = folder_path

    def empty_folder(self):
        """
        Removes every file located in the previously selected export folder.
        """
        if self.out_path is None or self.out_path == "":
            return
        content = os.listdir(self.out_path)
        for item in content:
            if os.path.isfile(os.path.join(self.out_path, item)):
                os.remove(os.path.join(self.out_path, item))

    def export_tiles(self):
        """
        Converts all the content of the input folder to tiles.
        The tiles are exported to the previously selected output folder.
        """
        tifffiles = [f for f in os.listdir(self.in_path) if re.match(TIFF_REGEX, f)]
        if len(tifffiles) == 0:
            show_info("No tiff files found in the selected folder.")
            return
        if not self.is_setup_ready():
            show_info("Please configure the tiles first.")
            return
        if (self.out_path is None) or (self.out_path == ""):
            show_info("Please select an export folder.")
            return
        lower_bound = None # For global normalization
        upper_bound = None
        use_norm = False
        if self.use_normalization.isChecked():
            use_norm = True
            lower_bound = float(self.lower_bound_input.text())
            upper_bound = float(self.upper_bound_input.text())
        for img in progress(tifffiles, "Exporting tiles..."):
            im_data = tifffile.imread(os.path.join(self.in_path, img))
            if self.use_calibration.isChecked():
                im_data = recalibrate_image(im_data, self.pixel_size, self.calib_unit)
            tiler = ImageTiler2D(self.patch_size, self.overlap, im_data.shape)
            tiles = tiler.image_to_tiles(im_data, use_norm, lower_bound, upper_bound)
            for i, tile in enumerate(tiles):
                tile_name = TIFF_REGEX.match(img).group(1) + f"_{str(i).zfill(3)}.tif"
                tifffile.imwrite(os.path.join(self.out_path, tile_name), tile)
        

    # -------- METHODS: ----------------------------------

    def probe_folder_shape(self, tifffiles):
        """
        The goal here is to determine if a folder contains images of the same shape.
        If it doesn't, a warning is raised.
        No error is signaled as a new ImageTiler2D object will be created for each image.
        """
        shape = None
        same = True
        for tiff in tifffiles:
            data = tifffile.imread(os.path.join(self.in_path, tiff))
            if shape is None:
                shape = data.shape
            elif shape != data.shape:
                same = False
                break
        if same:
            self.shape_label.setText(f"Shape: {shape}")
        else:
            self.shape_label.setText("Shape: ⚠️ Different shapes")

    def set_sources_folder(self, folder_path):
        """
        Expects the path to a folder containing tiff images.
        """
        tiff_list = sorted([f for f in os.listdir(folder_path) if re.match(TIFF_REGEX, f)])
        if len(tiff_list) == 0:
            show_info("No tiff files found in the selected folder.")
            return
        self.in_path = folder_path
        self.n_sources_label.setText(f"🖼️ sources: {len(tiff_list)}")
        self.probe_folder_shape(tiff_list)
    
    def set_calibration(self, size, unit):
        if (size is None) or (unit is None):
            self.pixel_size_label.setText(f"Pixel size: ---")
            self.scaling_factor_label.setText(f"Scaling factor: x---")
            return
        self.calibration_input.setText(f"{size:.3f}")
        self.unit_selector.setCurrentText(unit)
        self.viewer.scale_bar.unit = unit
        for layer in self.viewer.layers:
            layer.scale = (size, size)
        self.pixel_size_label.setText(f"Pixel size: {size:.3f} {unit}")
        self.viewer.scale_bar.visible = True
        self.pixel_size = size
        self.calib_unit = unit
        factor = process_factor(self.pixel_size, self.calib_unit)
        self.scaling_factor_label.setText(f"Scaling factor: x{factor:.2f}")
    
    def convert_to_optimal_unit(self, size, unit):
        unit_factors = {
            'nm': 1e9,
            'μm': 1e6, 
            'mm': 1e3, 
            'cm': 1e2, 
            'dm': 1e1,
            'm': 1 
        }
        meters = size / unit_factors[unit]
        for unit_name, factor in sorted(unit_factors.items(), key=lambda x: x[1]):
            converted_size = meters * factor
            if converted_size*10 >= 1:
                return converted_size, unit_name
        return size, unit
    
    def mosaic_shape(self, items):
        """
        Given a list of items, we want to determine the shape of the mosaic.
        We want a mosaic as close as possible to a square.
        The content of the 'items' list doesn't matter, it is not taken into account.
        """
        sqrt_N = math.sqrt(len(items))
        rows = math.floor(sqrt_N)
        cols = math.ceil(sqrt_N)
        if rows * cols < len(items):
            rows += 1
        return rows, cols

    def lower_resolution(self, img_height, img_width, max_size=512):
        """
        When we build a mosaic of images to show what's in the input folder, we want to keep the size of the images reasonable.
        So we downscale everything we find to have a size (per image) of 512 pixels on the largest side.
        """
        while (img_height > max_size) or (img_width > max_size):
            img_height //= 2
            img_width //= 2
        return img_height, img_width

    def build_folder_preview(self, padding=10, background_value=0):
        """
        Takes the first 25 images in the input folder and builds a mosaic with them.
        The name of each selected image is written over it.
        """
        items = [f for f in os.listdir(self.in_path) if re.match(TIFF_REGEX, f)]
        random.shuffle(items)
        items = items[:min(25, len(items))]
        max_len = max(items, key=len)
        rows, cols = self.mosaic_shape(items)
        img_0 = tifffile.imread(os.path.join(self.in_path, items[0]))
        img_height, img_width = img_0.shape
        img_height, img_width = self.lower_resolution(img_height, img_width)
        mosaic_height = rows * img_height + (rows + 1) * padding
        mosaic_width = cols * img_width + (cols + 1) * padding
        mosaic = np.full((mosaic_height, mosaic_width), fill_value=background_value, dtype=img_0.dtype)
        font_scale = 1.0
        text_size = cv2.getTextSize(max_len, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        while text_size[0] + 2 > img_width:
            font_scale -= 0.05
            text_size = cv2.getTextSize(max_len, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]

        for idx, image in enumerate(items):
            image_data = tifffile.imread(os.path.join(self.in_path, image))
            image_data = cv2.resize(image_data, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            cv2.putText(
                image_data, 
                image, 
                (5, img_height-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                np.iinfo(image_data.dtype).max, 
                2
            )
            row = idx // cols
            col = idx % cols
            if row >= rows:
                break 
            top_left_y = row * img_height + (row + 1) * padding
            top_left_x = col * img_width + (col + 1) * padding
            mosaic[top_left_y:top_left_y + img_height, top_left_x:top_left_x + img_width] = image_data
        
        if _FOLDER_MOSAIC_LAYER in self.viewer.layers:
            self.viewer.layers[_FOLDER_MOSAIC_LAYER].data = mosaic
        else:
            self.viewer.add_image(mosaic, name=_FOLDER_MOSAIC_LAYER, colormap='gray')
