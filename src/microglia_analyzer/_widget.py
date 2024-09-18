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


from microglia_analyzer.microglia_analyzer import MicrogliaAnalyzer


class MicrogliaAnalyzerWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()
        self.mam = None
        self.font = None
        self.init_ui()

    def init_ui(self):
        self.font = QFont()
        self.font.setFamily("Arial Unicode MS, Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji")
    
    def media_control_panel(self):
        pass

    def detect_microglia_panel(self):
        pass

    def segment_microglia_panel(self):
        pass

    def measure_microglia_panel(self):
        pass