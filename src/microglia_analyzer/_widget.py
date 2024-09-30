from qtpy.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, 
                            QSpinBox, QHBoxLayout, QPushButton, 
                            QFileDialog, QComboBox, QLabel, 
                            QSlider, QSpinBox, QFrame, QLineEdit)

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

from microglia_analyzer import TIFF_REGEX
from microglia_analyzer.microglia_analyzer import MicrogliaAnalyzer


_IMAGE_LAYER_NAME = "µ-Image"

class MicrogliaAnalyzerWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()
        self.mam = MicrogliaAnalyzer()
        self.font = None
        self.init_ui()

        self.sources_folder = None

    # -------- UI: ----------------------------------

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.font = QFont()
        self.font.setFamily("Arial Unicode MS, Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji")
        self.media_control_panel()
        self.calibration_panel()
        self.segment_microglia_panel()
        self.classify_microglia_panel()
        self.measures_panel()
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

        # Select sources folder button
        self.select_sources_button = QPushButton("📂 Sources folder")
        self.select_sources_button.setFont(self.font)
        self.select_sources_button.clicked.connect(self.select_sources_folder)
        layout.addWidget(self.select_sources_button)

        # Images drop down menu
        self.images_combo = QComboBox()
        self.images_combo.addItem("---")
        self.images_combo.currentIndexChanged.connect(self.select_image)
        layout.addWidget(self.images_combo)

        media_control_group.setLayout(layout)
        self.layout.addWidget(media_control_group)

    def calibration_panel(self):
        self.calibration_group = QGroupBox("Calibration")
        layout = QVBoxLayout()

        nav_layout = QHBoxLayout()

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

        self.calibration_group.setLayout(layout)
        self.layout.addWidget(self.calibration_group)

    def segment_microglia_panel(self):
        self.segment_microglia_group = QGroupBox("Segmentation")
        layout = QVBoxLayout()

        # Dropdown menu to choose a model
        self.model_selector = QComboBox()
        self.model_selector.addItem("---")
        layout.addWidget(self.model_selector)

        # Minimal area of a microglia
        h_layout = QHBoxLayout()
        self.minimal_area_label = QLabel("Min area (µm²):")
        h_layout.addWidget(self.minimal_area_label)
        self.minimal_area_input = QSpinBox()
        self.minimal_area_input.setRange(0, 1000000)
        h_layout.addWidget(self.minimal_area_input)
        layout.addLayout(h_layout)

        # Probality threshold slider
        h_layout = QHBoxLayout()
        self.probability_threshold_label = QLabel("Min probability (%)")
        h_layout.addWidget(self.probability_threshold_label)
        self.probability_threshold_slider = QSlider(Qt.Horizontal)
        self.probability_threshold_slider.setRange(0, 100)
        self.probability_threshold_slider.setValue(50)
        self.probability_threshold_slider.setTickInterval(1)
        self.probability_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.probability_threshold_slider.valueChanged.connect(self.proba_threshold_update)
        h_layout.addWidget(self.probability_threshold_slider)
        self.proba_value_label = QLabel("50%")
        h_layout.addWidget(self.proba_value_label)
        layout.addLayout(h_layout)

        # Segmentation button
        self.segment_microglia_button = QPushButton("🔍 Segment")
        self.segment_microglia_button.setFont(self.font)
        self.segment_microglia_button.clicked.connect(self.segment_microglia)
        layout.addWidget(self.segment_microglia_button)

        self.segment_microglia_group.setLayout(layout)
        self.layout.addWidget(self.segment_microglia_group)

    def classify_microglia_panel(self):
        self.classify_microglia_group = QGroupBox("Classification")
        layout = QVBoxLayout()

        # Dropdown menu to choose a model
        self.model_selector = QComboBox()
        self.model_selector.addItem("---")
        layout.addWidget(self.model_selector)

        # Classification button
        self.classify_microglia_button = QPushButton("🧠 Classify")
        self.classify_microglia_button.setFont(self.font)
        self.classify_microglia_button.clicked.connect(self.classify_microglia)
        layout.addWidget(self.classify_microglia_button)

        self.classify_microglia_group.setLayout(layout)
        self.layout.addWidget(self.classify_microglia_group)

    def measures_panel(self):
        self.microglia_group = QGroupBox("Measures")
        layout = QVBoxLayout()

        self.skeletonize_microglia_button = QPushButton("🦴 Skeletonize")
        self.skeletonize_microglia_button.setFont(self.font)
        self.skeletonize_microglia_button.clicked.connect(self.skeletonize_microglia)
        layout.addWidget(self.skeletonize_microglia_button)

        self.export_control_images_button = QPushButton("📸 Export control image")
        self.export_control_images_button.setFont(self.font)
        self.export_control_images_button.clicked.connect(self.export_control_images)
        layout.addWidget(self.export_control_images_button)

        self.export_measures_button = QPushButton("📊 Export measures")
        self.export_measures_button.setFont(self.font)
        self.export_measures_button.clicked.connect(self.export_measures)
        layout.addWidget(self.export_measures_button)

        self.microglia_group.setLayout(layout)
        self.layout.addWidget(self.microglia_group)

    # -------- Callbacks: ----------------------------------

    def clear_state(self):
        self.mam = MicrogliaAnalyzer()
        self.clear_viewer()
        self.clear_gui_elements()
        self.clear_attributes()

    def select_sources_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select sources folder")
        self.set_sources_folder(folder_path)

    def select_image(self):
        current_image = self.images_combo.currentText()
        if (self.sources_folder is None) or (current_image is None) or (current_image == "---") or (current_image == ""):
            return
        full_path = os.path.join(self.sources_folder, current_image)
        if not os.path.isfile(full_path):
            return
        self.open_image(full_path)

    def apply_calibration(self):
        pass

    def proba_threshold_update(self):
        self.proba_value_label.setText(f"{self.probability_threshold_slider.value()}%")

    def produce_patches(self):
        pass

    def segment_microglia(self):
        pass

    def classify_microglia(self):
        pass

    def skeletonize_microglia(self):
        pass

    def export_control_images(self):
        pass

    def export_measures(self):
        pass

    # -------- Methods: ----------------------------------

    def clear_attributes(self):
        self.sources_folder = None

    def clear_viewer(self):
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()

    def clear_gui_elements(self):
        self.images_combo.clear()
        self.images_combo.addItem("---")
        self.calibration_input.clear()
        self.pixel_size_label.setText("Pixel size: ---")

    def get_all_tiff_files(self, folder_path, no_ext=False):
        """
        Probes a folder and filters its content with a regex.
        All the TIFF are returned, whatever the number of 'f' or the case.
        If the `no_ext` attribute is True, the name is returned without the extension.
        """
        tiff_files = []
        for file_name in os.listdir(folder_path):
            match = TIFF_REGEX.match(file_name)
            if match:
                if no_ext:
                    tiff_files.append(match.group(1))
                else:
                    tiff_files.append(match.group(0))
        return sorted(tiff_files)

    def set_sources_folder(self, folder_path):
        if (folder_path is None) or (folder_path == ""):
            show_info("No folder selected")
            return
        self.sources_folder = folder_path
        self.images_combo.clear()
        self.images_combo.addItems(self.get_all_tiff_files(folder_path))
    
    def open_image(self, image_path):
        data = tifffile.imread(image_path)
        if _IMAGE_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[_IMAGE_LAYER_NAME].data = data
        else:
            self.viewer.add_image(data, name=_IMAGE_LAYER_NAME, colormap='green')
        # self.mam.load_image(image_path)
        
    
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
    
    def apply_calibration(self):
        length = float(self.calibration_input.text())
        unit = self.unit_selector.currentText()
        pixelSize, unit = self.convert_to_optimal_unit(length, unit)
        self.set_calibration(pixelSize, unit)
    
    def set_calibration(self, size, unit):
        self.calibration_input.setText(f"{size:.2f}")
        self.unit_selector.setCurrentText(unit)
        self.viewer.scale_bar.unit = unit
        for layer in self.viewer.layers:
            layer.scale = (size, size)
        self.pixel_size_label.setText(f"Pixel size: {size:.2f} {unit}")
        self.viewer.scale_bar.visible = True
        # self.mam.set_calibration(size, unit)