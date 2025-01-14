from qtpy.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QTableWidget, 
                            QSpinBox, QHBoxLayout, QPushButton, QHeaderView,
                            QFileDialog, QComboBox, QLabel, QTableWidgetItem,
                            QSlider, QSpinBox, QFrame, QLineEdit)

from qtpy.QtCore import QThread, Qt

from PyQt5.QtGui import QFont, QDoubleValidator, QColor
from PyQt5.QtCore import pyqtSignal, Qt, QLocale

import napari
from napari.utils.notifications import show_info
from napari.utils import progress

import tifffile
import numpy as np
import os
import re

from microglia_analyzer import TIFF_REGEX
from microglia_analyzer.utils import boxes_as_napari_shapes, BBOX_COLORS
from microglia_analyzer.ma_worker import MicrogliaAnalyzer
from microglia_analyzer.qt_workers import (QtSegmentMicroglia, QtClassifyMicroglia,
                                          QtMeasureMicroglia, QtBatchRunners)

_IMAGE_LAYER_NAME          = "µ-Image"
_SEGMENTATION_LAYER_NAME   = "µ-Segmentation"
_CLASSIFICATION_LAYER_NAME = "µ-Classification"
_YOLO_LAYER_NAME           = "µ-YOLO"
_SKELETON_LAYER_NAME       = "µ-Skeleton"

class MicrogliaAnalyzerWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()
        self.mam = MicrogliaAnalyzer(lambda x: print(x))
        self.font = None
        self.init_ui()

        self.sources_folder = None
        self.active_worker = False
        self.n_images = 0

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
        float_validator.setLocale(QLocale(QLocale.English))
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

        # Segmentation button
        self.segment_microglia_button = QPushButton("🔍 Segment")
        self.segment_microglia_button.setFont(self.font)
        self.segment_microglia_button.clicked.connect(self.segment_microglia)
        layout.addWidget(self.segment_microglia_button)

        # Minimal area of a microglia
        h_layout = QHBoxLayout()
        self.minimal_area_label = QLabel("Min area (µm²):")
        h_layout.addWidget(self.minimal_area_label)
        self.minimal_area_input = QSpinBox()
        self.minimal_area_input.setRange(0, 1000000)
        self.minimal_area_input.setValue(40)
        self.minimal_area_input.valueChanged.connect(self.min_area_update)
        h_layout.addWidget(self.minimal_area_input)
        layout.addLayout(h_layout)

        # Probality threshold slider
        h_layout = QHBoxLayout()
        self.probability_threshold_label = QLabel("Min probability:")
        h_layout.addWidget(self.probability_threshold_label)
        self.probability_threshold_slider = QSlider(Qt.Horizontal)
        self.probability_threshold_slider.setRange(0, 100)
        self.probability_threshold_slider.setValue(40)
        self.probability_threshold_slider.setTickInterval(1)
        self.probability_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.probability_threshold_slider.valueChanged.connect(self.proba_threshold_update)
        h_layout.addWidget(self.probability_threshold_slider)
        self.proba_value_label = QLabel("40%")
        h_layout.addWidget(self.proba_value_label)
        layout.addLayout(h_layout)

        self.segment_microglia_group.setLayout(layout)
        self.layout.addWidget(self.segment_microglia_group)

    def reset_table(self):
        classes = self.mam.classes
        self.table.setRowCount(0)

        items = []
        if classes is not None:
            items = [(QColor(BBOX_COLORS[i]), c) for i, c in classes.items()]
            self.table.setRowCount(len(classes))

        for row, (color, word) in enumerate(items):
            color_item = QTableWidgetItem()
            color_item.setBackground(color)
            color_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 0, color_item)

            word_item = QTableWidgetItem(word)
            word_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 1, word_item)

    def classes_table_ui(self):
        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Color", "Class"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 60)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        return self.table

    def classify_microglia_panel(self):
        self.classify_microglia_group = QGroupBox("Classification")
        layout = QVBoxLayout()

        # Classification button
        self.classify_microglia_button = QPushButton("🧠 Classify")
        self.classify_microglia_button.setFont(self.font)
        self.classify_microglia_button.clicked.connect(self.classify_microglia)
        layout.addWidget(self.classify_microglia_button)

        # List of classes
        self.table = self.classes_table_ui()
        self.reset_table()
        layout.addWidget(self.table)

        # Minimum score for classification
        h_layout = QHBoxLayout()
        self.minimal_score_label = QLabel("Min score:")
        h_layout.addWidget(self.minimal_score_label)
        self.minimal_score_slider = QSlider(Qt.Horizontal)
        self.minimal_score_slider.setRange(0, 100)
        self.minimal_score_slider.setValue(15)
        h_layout.addWidget(self.minimal_score_slider)
        self.min_score_label = QLabel("15%")
        h_layout.addWidget(self.min_score_label)
        self.minimal_score_slider.valueChanged.connect(self.min_score_update)
        layout.addLayout(h_layout)

        self.classify_microglia_group.setLayout(layout)
        self.layout.addWidget(self.classify_microglia_group)

    def measures_panel(self):
        self.microglia_group = QGroupBox("Measures")
        layout = QVBoxLayout()

        self.export_measures_button = QPushButton("📊 Measure")
        self.export_measures_button.setFont(self.font)
        self.export_measures_button.clicked.connect(self.export_measures)
        layout.addWidget(self.export_measures_button)

        self.run_batch_button = QPushButton("▶ Run batch")
        # self.run_batch_button.setFont(self.font)
        self.run_batch_button.clicked.connect(self.run_batch)
        layout.addWidget(self.run_batch_button)

        self.microglia_group.setLayout(layout)
        self.layout.addWidget(self.microglia_group)

    # -------- Callbacks: ----------------------------------

    def clear_state(self):
        self.mam = MicrogliaAnalyzer()
        self.clear_viewer()
        self.clear_gui_elements()
        self.clear_attributes()

    def min_score_update(self):
        self.min_score_label.setText(f"{self.minimal_score_slider.value()}%")

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

    def proba_threshold_update(self):
        self.proba_value_label.setText(f"{self.probability_threshold_slider.value()}%")
        self.update_seg_pp()
    
    def min_area_update(self):
        self.update_seg_pp()

    def segment_microglia(self):
        self.pbr = progress()
        self.pbr.set_description("Segmenting microglia...")
        self.set_active_ui(False)
        self.thread = QThread()

        self.mam.set_cc_min_size(self.minimal_area_input.value())
        self.mam.set_proba_threshold(self.probability_threshold_slider.value() / 100)

        self.worker = QtSegmentMicroglia(self.pbr, self.mam)
        self.total = 0
        self.worker.update.connect(self.update_pbr)
        self.active_worker = True

        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.show_microglia)
        self.thread.started.connect(self.worker.run)
        
        self.thread.start()

    def update_seg_pp(self):
        self.mam.set_cc_min_size(self.minimal_area_input.value())
        self.mam.set_proba_threshold(self.probability_threshold_slider.value() / 100)
        self.mam.segmentation_postprocessing()
        self.show_microglia()

    def classify_microglia(self):
        self.pbr = progress()
        self.pbr.set_description("Classifying microglia...")
        self.set_active_ui(False)
        self.thread = QThread()

        self.mam.set_min_score(self.minimal_score_slider.value() / 100)

        self.worker = QtClassifyMicroglia(self.pbr, self.mam)
        self.total = 0
        self.worker.update.connect(self.update_pbr)
        self.active_worker = True

        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.show_classification)
        self.thread.started.connect(self.worker.run)
        
        self.thread.start()

    def export_measures(self):
        self.pbr = progress()
        self.pbr.set_description("Measuring microglia...")
        self.set_active_ui(False)
        self.thread = QThread()

        self.worker = QtMeasureMicroglia(self.pbr, self.mam)
        self.total = 0
        self.worker.update.connect(self.update_pbr)
        self.active_worker = True

        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.write_measures)
        self.thread.started.connect(self.worker.run)
        
        self.thread.start()
    
    def run_batch(self):
        self.n_images = len(self.get_all_tiff_files(self.sources_folder))
        self.pbr = progress()
        self.pbr.set_description("Running on folder...")
        self.run_batch_button.setText(f"▶ Run batch ({str(1).zfill(2)}/{str(self.n_images).zfill(2)})")
        self.set_active_ui(False)
        self.thread = QThread()

        settings = {
            'calibration': self.mam.calibration,
            'cc_min_size': self.minimal_area_input.value(),
            'proba_threshold': self.probability_threshold_slider.value() / 100,
            'unet_path': os.path.dirname(self.mam.segmentation_model_path),
            'yolo_path': os.path.dirname(os.path.dirname(self.mam.classification_model_path)),
            'min_score': self.minimal_score_slider.value() / 100
        }

        self.worker = QtBatchRunners(self.pbr, self.sources_folder, settings)
        self.worker.update.connect(self.update_pbr)
        self.active_worker = True

        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.end_batch)
        self.thread.started.connect(self.worker.run)
        
        self.thread.start()

    # -------- Methods: ----------------------------------

    def end_batch(self):
        self.pbr.close()
        self.set_active_ui(True)
        self.n_images = 0
        self.run_batch_button.setText("▶ Run batch")
        show_info("Batch completed.")

    def write_measures(self):
        self.end_worker()
        measures = self.mam.as_csv(self.images_combo.currentText())
        skeleton = self.mam.skeleton
        if _SKELETON_LAYER_NAME not in self.viewer.layers:
            layer = self.viewer.add_image(skeleton, name=_SKELETON_LAYER_NAME, colormap='red', blending='additive')
        else:
            layer = self.viewer.layers[_SKELETON_LAYER_NAME]
            layer.data = skeleton
        if self.mam.calibration is not None:
            self.set_calibration(*self.mam.calibration)
        root_folder = os.path.join(self.sources_folder, "controls")
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)
        measures_path = os.path.join(root_folder, os.path.splitext(self.images_combo.currentText())[0] + "_measures.csv")
        control_path  = os.path.join(root_folder, os.path.splitext(self.images_combo.currentText())[0] + "_control.tif")
        tifffile.imwrite(control_path, np.stack([self.mam.skeleton, self.mam.mask], axis=0))
        with open(measures_path, 'w') as f:
            f.write("\n".join(measures))
        show_info(f"Microglia measured.")

    def show_classification(self):
        self.end_worker()
        bindings = self.mam.bindings
        self.reset_table()
        # Showing bound classification
        boxes, colors = boxes_as_napari_shapes(bindings.values())
        layer = None
        if _CLASSIFICATION_LAYER_NAME not in self.viewer.layers:
            layer = self.viewer.add_shapes(boxes, name=_CLASSIFICATION_LAYER_NAME, edge_color=colors, face_color='#00000000', edge_width=4)
        else:
            layer = self.viewer.layers[_CLASSIFICATION_LAYER_NAME]
            layer.data = boxes
            layer.edge_colors = colors
        # Showing raw classification
        classification = self.mam.classifications
        tps = [(c, None, b) for c, b in zip(classification['classes'], classification['boxes'])]
        boxes, colors = boxes_as_napari_shapes(tps, True)
        if _YOLO_LAYER_NAME not in self.viewer.layers:
            layer = self.viewer.add_shapes(boxes, name=_YOLO_LAYER_NAME, edge_color=colors, face_color='#00000000', edge_width=4, visible=False)
        else:
            layer = self.viewer.layers[_YOLO_LAYER_NAME]
            layer.data = boxes
            layer.edge_colors = colors
            layer.edge_width = 4
        # Update calibration
        if self.mam.calibration is not None:
            self.set_calibration(*self.mam.calibration)
        show_info(f"Microglia classified.")

    def show_microglia(self):
        self.end_worker()
        labeled = self.mam.mask
        show_info(f"Microglia segmented.")
        if _SEGMENTATION_LAYER_NAME in self.viewer.layers:
            layer = self.viewer.layers[_SEGMENTATION_LAYER_NAME]
            layer.data = labeled
        else:
            layer = self.viewer.add_labels(labeled, name=_SEGMENTATION_LAYER_NAME)
        if self.mam.calibration is not None:
            self.set_calibration(*self.mam.calibration)
        self.set_active_ui(True)

    def set_active_ui(self, state):
        self.clear_state_button.setEnabled(state)
        self.select_sources_button.setEnabled(state)
        self.images_combo.setEnabled(state)
        self.calibration_input.setEnabled(state)
        self.unit_selector.setEnabled(state)
        self.calibrationButton.setEnabled(state)
        self.segment_microglia_button.setEnabled(state)
        self.minimal_area_input.setEnabled(state)
        self.probability_threshold_slider.setEnabled(state)
        self.classify_microglia_button.setEnabled(state)
        self.run_batch_button.setEnabled(state)
        self.run_batch_button.setEnabled(state)
        self.export_measures_button.setEnabled(state)

    def end_worker(self):
        if self.active_worker:
            self.active_worker = False
            self.pbr.close()
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
            self.set_active_ui(True)
            self.total = -1

    def update_pbr(self, text, current, total):
        self.pbr.set_description(text)
        # if (total != self.total):
        #     self.pbr.reset(total=total)
        #     self.total = total
        if (self.n_images > 0):
            self.run_batch_button.setText(f"▶ Run batch ({str(current+1).zfill(2)}/{str(self.n_images).zfill(2)})")
        # self.pbr.update(current)

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
    
    def reset_layers(self):
        if _SEGMENTATION_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[_SEGMENTATION_LAYER_NAME].data = np.zeros_like(self.viewer.layers[_SEGMENTATION_LAYER_NAME].data)
        if _CLASSIFICATION_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[_CLASSIFICATION_LAYER_NAME].data = []
        if _YOLO_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[_YOLO_LAYER_NAME].data = []
        if _SKELETON_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[_SKELETON_LAYER_NAME].data = np.zeros_like(self.viewer.layers[_SKELETON_LAYER_NAME].data)

    def open_image(self, image_path):
        data = tifffile.imread(image_path)
        layer = None
        if _IMAGE_LAYER_NAME in self.viewer.layers:
            layer = self.viewer.layers[_IMAGE_LAYER_NAME]
            layer.data = data
        else:
            layer = self.viewer.add_image(data, name=_IMAGE_LAYER_NAME, colormap='gray')
        self.reset_layers()
        self.mam.set_input_image(data.copy())
        if self.mam.calibration is not None:
            self.set_calibration(*self.mam.calibration)
        
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
        self.calibration_input.setText(f"{size:.3f}")
        self.unit_selector.setCurrentText(unit)
        self.viewer.scale_bar.unit = unit
        for layer in self.viewer.layers:
            layer.scale = (size, size)
        self.pixel_size_label.setText(f"Pixel size: {size:.3f} {unit}")
        self.viewer.scale_bar.visible = True
        self.mam.set_calibration(size, unit)