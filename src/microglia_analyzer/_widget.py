from qtpy.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QTableWidget, 
                            QSpinBox, QHBoxLayout, QPushButton, QHeaderView,
                            QFileDialog, QComboBox, QLabel, QTableWidgetItem,
                            QSlider, QSpinBox, QFrame, QLineEdit, QCheckBox, QApplication)

from qtpy.QtCore import QThread, Qt


from PyQt5.QtGui import QFont, QDoubleValidator, QColor
from PyQt5.QtCore import Qt, QLocale

import napari
from napari.utils.notifications import show_info, show_warning
from napari.utils import progress

import tifffile
import numpy as np
import os
import json
import warnings
import shutil

from microglia_analyzer import __release__, __version__
from microglia_analyzer.utils import (get_all_tiff_files, 
                                      bindings_as_napari_shapes, 
                                      BBOX_COLORS)
from microglia_analyzer.ma_worker import MicrogliaAnalyzer
from microglia_analyzer.qt_workers import (QtSegmentMicroglia, QtClassifyMicroglia,
                                          QtMeasureMicroglia, QtBatchRunners)

_IMAGE_LAYER_NAME          = "µ-Image"
_SEGMENTATION_LAYER_NAME   = "µ-Segmentation"
_CLASSIFICATION_LAYER_NAME = "µ-Classification"
_SKELETON_LAYER_NAME       = "µ-Skeleton"

class MicrogliaAnalyzerWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()
        self.mam    = MicrogliaAnalyzer(lambda x: print(x))
        self.font   = QFont()
        self.font.setFamily("Arial Unicode MS, Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji")
        warnings.simplefilter("ignore", FutureWarning)
        self.sources_folder = None
        self.active_worker  = False
        self.n_images       = 0
        self.init_ui()

    # -------- UI: ----------------------------------

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.media_control_panel()
        self.calibration_panel()
        self.segment_microglia_panel()
        self.classify_microglia_panel()
        self.measures_panel()
        self.setLayout(self.layout)
        v = QLabel(f"Version: {__version__}/{__release__}")
        v.setAlignment(Qt.AlignRight)
        self.layout.addWidget(v)
    
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

        # Checkbox to discard previous runs
        self.discard_runs_checkbox = QCheckBox("Discard previous run at opening")
        layout.addWidget(self.discard_runs_checkbox)

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
        # self.segment_microglia_button.setFont(self.font)
        self.segment_microglia_button.clicked.connect(self.segment_microglia)
        layout.addWidget(self.segment_microglia_button)

        # Minimal area of a microglia
        h_layout = QHBoxLayout()
        self.minimal_area_label = QLabel("Min area (µm²):")
        h_layout.addWidget(self.minimal_area_label)
        self.minimal_area_input = QSpinBox()
        self.minimal_area_input.setRange(0, 1000000)
        self.minimal_area_input.setValue(5)
        self.minimal_area_input.valueChanged.connect(self.min_area_update)
        h_layout.addWidget(self.minimal_area_input)
        layout.addLayout(h_layout)

        # Probality threshold slider
        h_layout = QHBoxLayout()
        self.probability_threshold_label = QLabel("Min probability:")
        h_layout.addWidget(self.probability_threshold_label)
        self.probability_threshold_slider = QSlider(Qt.Horizontal)
        self.probability_threshold_slider.setRange(0, 100)
        self.probability_threshold_slider.setValue(20)
        self.probability_threshold_slider.setTickInterval(1)
        self.probability_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.probability_threshold_slider.valueChanged.connect(self.proba_threshold_update)
        h_layout.addWidget(self.probability_threshold_slider)
        self.proba_value_label = QLabel("20%")
        h_layout.addWidget(self.proba_value_label)
        layout.addLayout(h_layout)

        self.segment_microglia_group.setLayout(layout)
        self.layout.addWidget(self.segment_microglia_group)

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
        # self.classify_microglia_button.setFont(self.font)
        self.classify_microglia_button.clicked.connect(self.classify_microglia)
        layout.addWidget(self.classify_microglia_button)

        # List of classes
        self.table = self.classes_table_ui()
        self.reset_table_ui()
        layout.addWidget(self.table)

        # "Show garbage" checkbox
        self.show_garbage_box = QCheckBox("Show garbage")
        self.show_garbage_box.setChecked(True)
        self.show_garbage_box.stateChanged.connect(self.update_garbage_display)
        layout.addWidget(self.show_garbage_box)

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
        self.run_batch_button.clicked.connect(self.batch_callback)
        layout.addWidget(self.run_batch_button)

        self.microglia_group.setLayout(layout)
        self.layout.addWidget(self.microglia_group)
    
    def set_active_ui(self, state, affect_batch=False):
        self.clear_state_button.setEnabled(state)
        self.select_sources_button.setEnabled(state)
        self.images_combo.setEnabled(state)
        self.calibration_input.setEnabled(state)
        self.unit_selector.setEnabled(state)
        self.discard_runs_checkbox.setEnabled(state)
        self.calibrationButton.setEnabled(state)
        self.segment_microglia_button.setEnabled(state)
        self.minimal_area_input.setEnabled(state)
        self.probability_threshold_slider.setEnabled(state)
        self.classify_microglia_button.setEnabled(state)
        self.export_measures_button.setEnabled(state)
        self.show_garbage_box.setEnabled(state)
        if not affect_batch:
            self.run_batch_button.setEnabled(state)

    # -------- Callbacks: ----------------------------------

    def clear_state(self):
        self.mam = MicrogliaAnalyzer(lambda x: print(x))
        self.clear_viewer()
        self.clear_attributes()
        self.clear_gui_elements()

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
        if self.calibration_input.text() == "":
            return
        length = float(self.calibration_input.text())
        unit = self.unit_selector.currentText()
        pixelSize, unit = self.convert_to_optimal_unit(length, unit)
        self.set_calibration(pixelSize, unit)

    def ask_for_model_folder_popup(self, category):
        if __release__:
            return
        model_path = QFileDialog.getExistingDirectory(self, f"Select {category} model folder")
        if not os.path.isdir(model_path):
            print("No model selected (invalid path).")
            return
        if category == "segmentation":
            self.mam.set_segmentation_model(model_path)
            print("Local segmentation model used")
        elif category == "classification":
            self.mam.set_classification_model(model_path)
            print("Local classification model used")
        else:
            print("No model selected (invalid category).")

    def segment_microglia(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            self.ask_for_model_folder_popup("segmentation")

        self.pbr = progress()
        self.pbr.set_description("Segmenting microglia...")
        self.set_active_ui(False)
        self.thread = QThread()

        self.mam.set_min_surface(self.minimal_area_input.value())
        self.mam.set_proba_threshold(self.probability_threshold_slider.value() / 100)

        self.worker = QtSegmentMicroglia(self.pbr, self.mam)
        self.total = 0
        self.worker.update.connect(self.update_pbr)
        self.active_worker = True

        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.show_microglia)
        self.thread.started.connect(self.worker.run)
        
        self.thread.start()

    def min_area_update(self):
        self.update_seg_pp()

    def proba_threshold_update(self):
        self.proba_value_label.setText(f"{self.probability_threshold_slider.value()}%")
        self.update_seg_pp()

    def reset_table_ui(self):
        classes = self.mam.class_names
        if classes is None:
            return

        self.table.setRowCount(0)
        items = [(QColor(BBOX_COLORS[i][:7]), c) for i, c in enumerate(classes)]
        self.table.setRowCount(len(classes))

        for row, (color, word) in enumerate(items):
            color_item = QTableWidgetItem()
            color_item.setBackground(color)
            color_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 0, color_item)

            word_item = QTableWidgetItem(word)
            word_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 1, word_item)

    def classify_microglia(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            self.ask_for_model_folder_popup("classification")

        self.pbr = progress()
        self.pbr.set_description("Classifying microglia...")
        self.set_active_ui(False)
        self.thread = QThread()

        self.worker = QtClassifyMicroglia(self.pbr, self.mam)
        self.total = 0
        self.worker.update.connect(self.update_pbr)
        self.active_worker = True

        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.show_classification)
        self.thread.started.connect(self.worker.run)
        
        self.thread.start()

    def update_garbage_display(self):
        self.show_microglia()
        self.show_classification()

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
    
    def batch_callback(self):
        if self.thread is None:
            self.run_batch()
        else:
            self.interupt_batch()

    def interupt_batch(self):
        print("!!! Interupting batch...")
        show_info("Interupting batch...")
        self.worker.interupt()
        self.end_worker()
        self.end_batch()

    # -------- Methods: ----------------------------------

    def clear_attributes(self):
        self.sources_folder = None
        self.active_worker  = False
        self.n_images       = 0

    def clear_viewer(self):
        self.end_worker()
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()

    def clear_gui_elements(self):
        self.images_combo.clear()
        self.images_combo.addItem("---")
        self.calibration_input.clear()
        self.unit_selector.setCurrentIndex(0)
        self.pixel_size_label.setText("Pixel size: ---")
        self.run_batch_button.setText("▶ Run batch")
    
    def reset_layers(self):
        if _SEGMENTATION_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[_SEGMENTATION_LAYER_NAME].data = np.zeros_like(self.viewer.layers[_SEGMENTATION_LAYER_NAME].data)
        if _CLASSIFICATION_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[_CLASSIFICATION_LAYER_NAME].data = []
        if _SKELETON_LAYER_NAME in self.viewer.layers:
            self.viewer.layers[_SKELETON_LAYER_NAME].data = np.zeros_like(self.viewer.layers[_SKELETON_LAYER_NAME].data)

    def set_sources_folder(self, folder_path):
        if (folder_path is None) or (folder_path == ""):
            show_info("No folder selected")
            return
        self.sources_folder = folder_path
        c_path = os.path.join(self.sources_folder, "controls")
        if os.path.isdir(c_path) and self.discard_runs_checkbox.isChecked():
            shutil.rmtree(c_path)
        self.images_combo.clear()
        self.images_combo.addItems(get_all_tiff_files(folder_path))

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
        self.attempt_restore()

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
    
    def set_calibration(self, size, unit):
        self.calibration_input.setText(f"{size:.3f}")
        self.unit_selector.setCurrentText(unit)
        self.viewer.scale_bar.unit = unit
        for layer in self.viewer.layers:
            layer.scale = (size, size)
        self.pixel_size_label.setText(f"Pixel size: {size:.3f} {unit}")
        self.viewer.scale_bar.visible = True
        self.mam.set_calibration(size, unit)

    def update_seg_pp(self):
        self.mam.set_min_surface(self.minimal_area_input.value())
        self.mam.set_proba_threshold(self.probability_threshold_slider.value() / 100)
        self.mam._segmentation_postprocessing()
        self.mam._classification_postprocessing()
        self.show_microglia()
        self.show_classification()

    def show_microglia(self):
        self.end_worker()
        labeled = self.mam.get_mask(self.show_garbage_box.isChecked())
        if labeled is None:
            return
        if _SEGMENTATION_LAYER_NAME in self.viewer.layers:
            layer = self.viewer.layers[_SEGMENTATION_LAYER_NAME]
            layer.data = (labeled > 0).astype(np.uint8) * 255
        else:
            layer = self.viewer.add_labels((labeled > 0).astype(np.uint8) * 255, name=_SEGMENTATION_LAYER_NAME)
            show_info(f"Microglia segmented.")
        if self.mam.calibration is not None:
            self.set_calibration(*self.mam.calibration)
        v = self.mam.get_segmentation_version()
        if v is not None:
            self.segment_microglia_button.setText(f"🔍 Segment · (V{v})")
        self.set_active_ui(True)

    def show_classification(self):
        self.end_worker()
        bindings = self.mam.bindings
        if bindings is None:
            return
        boxes, colors = bindings_as_napari_shapes(bindings, -1 if self.show_garbage_box.isChecked() else 0)
        self.reset_table_ui()
        layer = None
        if _CLASSIFICATION_LAYER_NAME not in self.viewer.layers:
            layer = self.viewer.add_shapes(boxes, name=_CLASSIFICATION_LAYER_NAME, edge_color=colors, face_color='#00000000', edge_width=4)
        else:
            layer = self.viewer.layers[_CLASSIFICATION_LAYER_NAME]
            layer.data = boxes
            layer.edge_color = colors
            layer.edge_width = 4
        # Update calibration
        if self.mam.calibration is not None:
            self.set_calibration(*self.mam.calibration)
        self.show_microglia()
        v = self.mam.get_classification_version()
        if v is not None:
            self.classify_microglia_button.setText(f"🧠 Classify · (V{v})")
        show_info(f"Microglia classified.")
    
    def show_skeleton(self):
        skeleton = (self.mam.skeleton > 0).astype(np.uint8) * 2
        if _SKELETON_LAYER_NAME not in self.viewer.layers:
            layer = self.viewer.add_labels(skeleton, name=_SKELETON_LAYER_NAME)
        else:
            layer = self.viewer.layers[_SKELETON_LAYER_NAME]
            layer.data = skeleton
        if self.mam.calibration is not None:
            self.set_calibration(*self.mam.calibration)

    def write_csv(self, controls_folder):
        measures_path = os.path.join(controls_folder, "results")
        measure_path  = os.path.join(measures_path, os.path.splitext(self.images_combo.currentText())[0]+".csv")
        os.makedirs(measures_path, exist_ok=True)
        measures = self.mam.as_tsv(self.images_combo.currentText())
        with open(measure_path, 'w') as f:
            f.write("\n".join(measures))

    def write_mask(self, controls_folder):
        masks_path = os.path.join(controls_folder, "masks")
        mask_path  = os.path.join(masks_path, self.images_combo.currentText())
        os.makedirs(masks_path, exist_ok=True)
        tifffile.imwrite(mask_path, self.mam.mask)
        # save_as_fake_colors(self.mam.mask, mask_path.replace(".tif", ".png"))

    def write_skeleton(self, controls_folder):
        skeletons_path = os.path.join(controls_folder, "skeletons")
        skeleton_path  = os.path.join(skeletons_path, self.images_combo.currentText())
        os.makedirs(skeletons_path, exist_ok=True)
        tifffile.imwrite(skeleton_path, self.mam.skeleton)

    def write_classification(self, controls_folder):
        classifications_path = os.path.join(controls_folder, "classifications")
        classification_path  = os.path.join(classifications_path, os.path.splitext(self.images_combo.currentText())[0]+".txt")
        os.makedirs(classifications_path, exist_ok=True)
        with open(classification_path, 'w') as f:
            f.write("\n".join([str(b) for b in self.mam.bindings_to_yolo()]))

    def write_settings(self, controls_folder):
        settings_path = os.path.join(controls_folder, "settings.txt")
        with open(settings_path, 'w') as f:
            f.write(str(self.mam))

    def write_measures(self):
        self.end_worker()
        self.show_skeleton()
        controls_folder = os.path.join(self.sources_folder, "controls")
        os.makedirs(controls_folder, exist_ok=True)
        self.write_csv(controls_folder)
        self.write_mask(controls_folder)
        self.write_skeleton(controls_folder)
        self.write_classification(controls_folder)
        self.write_settings(controls_folder)
        show_info(f"Microglia measured.")

    def run_batch(self):
        sources = get_all_tiff_files(self.sources_folder)
        print("Found sources: ", sources)
        self.n_images = len(sources)
        self.pbr = progress()
        self.pbr.set_description("Running on folder...")
        self.run_batch_button.setText(f"■ Kill ({str(1).zfill(2)}/{str(self.n_images).zfill(2)})")
        self.set_active_ui(False, True)
        self.thread = QThread()

        settings = {
            'calibration': self.mam.calibration,
            'cc_min_size': self.minimal_area_input.value(),
            'proba_threshold': self.probability_threshold_slider.value() / 100,
            'unet_path': os.path.dirname(self.mam.segmentation_model_path),
            'yolo_path': os.path.dirname(os.path.dirname(self.mam.classification_model_path))
        }

        self.worker = QtBatchRunners(self.pbr, self.sources_folder, settings)
        self.worker.update.connect(self.update_pbr)
        self.active_worker = True

        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.end_batch)
        self.thread.started.connect(self.worker.run)
        
        self.thread.start()
    
    def end_batch(self):
        self.end_worker()
        self.n_images = 0
        self.run_batch_button.setText("▶ Run batch")
        show_info("=== 🎉 Batch completed. ===")
    
    def end_worker(self):
        if self.active_worker:
            self.active_worker = False
            self.pbr.close()
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
            self.set_active_ui(True, False)
            self.total = -1
            self.thread = None

    def update_pbr(self, text, current, total):
        self.pbr.set_description(text)
        if (self.n_images > 0):
            self.run_batch_button.setText(f"■ Kill ({str(current+1).zfill(2)}/{str(self.n_images).zfill(2)})")
    
    def import_settings(self):
        # 1. The control folder has to exist for any image.
        controls_folder = os.path.join(self.sources_folder, "controls")
        if not os.path.isdir(controls_folder):
            print("Couldn't find the 'controls' folder.")
            return False
        # 2. The target image has to be fully processed.
        img_name = self.images_combo.currentText()
        classif_name = os.path.splitext(img_name)[0]+".txt"
        result_name = os.path.splitext(img_name)[0]+".csv"
        mask_path = os.path.join(controls_folder, "masks", img_name)
        skel_path = os.path.join(controls_folder, "skeletons", img_name)
        rslt_path = os.path.join(controls_folder, "results", result_name)
        clsf_path = os.path.join(controls_folder, "classifications", classif_name)
        if not os.path.isfile(mask_path):
            print("The mask for this image is not available.")
            return False
        if not os.path.isfile(skel_path):
            print("The skeleton for this image is not available.")
            return False
        if not os.path.isfile(rslt_path):
            print("The results for this image are not available.")
            return False
        if not os.path.isfile(clsf_path):
            print("The classification for this image is not available.")
            return False
        show_info("Restoring state...")
        # 3. The settings file has to exist.
        settings_path = os.path.join(controls_folder, "settings.txt")
        if not os.path.isfile(settings_path):
            print("Failed to locate the settings file.")
            return False
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        # 4. We raise a warning if we are restoring from an earlier version
        parts_x = list(map(int, settings['Software version'].split('.')))
        parts_y = list(map(int, __version__.split('.')))
        if parts_x < parts_y:
            show_warning("These data come from an anterior version of MGA, be careful!!!")
        # 5. The image shape has to be the same
        if self.mam.image.shape != tuple(settings['Image shape']):
            print("The image shape is different from the expected one.")
            return False
        # 6. We restore the calibration
        self.set_calibration(float(settings['Calibration'][0]), settings['Calibration'][1])
        # 7. We try to restore models
        if not self.set_unet_by_version(int(settings['Segmentation model'])):
            print("Failed to locate the local UNet.")
            return False
        if not self.set_yolo_by_version(int(settings['Classification model'])):
            print("Failed to locate the local YOLO.")
            return False
        # 8. We check that classes are compatibles
        known_classes = set(settings['Class names'])
        target_classes = set(self.mam.class_names)
        if len(known_classes.intersection(target_classes)) != len(known_classes):
            print("The list of classes doesn't match.")
            return False
        self.reset_table_ui()
        # 9. Restoring parameters
        self.probability_threshold_slider.setValue(int(100.0 * float(settings['Probability threshold'])))
        self.minimal_area_input.setValue(int(settings['Minimal surface']))
        # 10. Restore the mask
        self.mam.mask = tifffile.imread(mask_path)
        self.mam.skeleton = tifffile.imread(skel_path)
        with open(clsf_path, 'r') as f:
            self.mam.bindings_from_yolo(f.read())
        self.show_classification()
        self.show_skeleton()
        return True
    
    def attempt_restore(self):
        self.set_active_ui(False)
        status = self.import_settings()
        if status:
            show_info("State restored.")
        else:
            show_info("Failed to restore state.")
        self.set_active_ui(True)

    def set_unet_by_version(self, version):
        version = int(version)
        # The loaded version is the good one.
        loaded = self.mam.get_segmentation_version()
        loaded = int(loaded) if (loaded is not None) else None
        if (loaded is not None) and (loaded == version):
            return True
        # Check if the model is available locally.
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "µnet")
        if not os.path.isdir(model_path):
            print("The desired UNet model doesn't exist locally.")
            return False
        version_path = os.path.join(model_path, f"version.txt")
        if not os.path.isfile(version_path):
            print("The desired UNet model doesn't have a version file.")
            return False
        index = -1
        with open(version_path, 'r') as f:
            index = int(f.read().strip())
        if index != version:
            print(f"UNet version {version} not found.")
            return False
        # Load the model
        self.mam.set_segmentation_model(model_path)
        return True
    
    def set_yolo_by_version(self, version):
        version = int(version)
        # The loaded version is the good one.
        loaded = self.mam.get_classification_version()
        loaded = int(loaded) if (loaded is not None) else None
        if (loaded is not None) and (loaded == version):
            return True
        # Check if the model is available locally.
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "µyolo")
        if not os.path.isdir(model_path):
            print("The desired YOLO model doesn't exist locally.")
            return False
        version_path = os.path.join(model_path, f"version.txt")
        if not os.path.isfile(version_path):
            print("The desired UNet model doesn't have a version file.")
            return False
        index = -1
        with open(version_path, 'r') as f:
            index = int(f.read().strip())
        if index != version:
            print(f"YOLO version {version} not found.")
            return False
        # Load the model
        self.mam.set_classification_model(model_path)
        return True
        