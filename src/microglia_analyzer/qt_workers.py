from qtpy.QtCore import QObject
from PyQt5.QtCore import pyqtSignal
import requests
import os
from microglia_analyzer.utils import download_from_web

_MODELS = "https://raw.githubusercontent.com/MontpellierRessourcesImagerie/microglia-analyzer/refs/heads/main/src/microglia_analyzer/models.json"

class QtSegmentMicroglia(QObject):
    
    finished = pyqtSignal()
    update   = pyqtSignal(str, int, int)

    def _fetch_descriptor(self):
        """ Reads the JSON file from the URL and stores it in self.versions """
        try:
            self.versions = requests.get(_MODELS).json()
        except requests.exceptions.RequestException as e:
            print("Failed to fetch the models descriptor.")
            self.versions = None

    def _check_updates(self):
        if not self.versions:
            return
        if not os.path.isdir(self.model_path):
            download_from_web(self.versions['µnet']['url'], self.model_path)
            print("Model downloaded.")
            return
        v_path = os.path.join(self.model_path, "version.txt")
        local_version = 0
        with open(v_path, 'r') as f:
            local_version = int(f.read().strip())
        if local_version < self.versions['µnet']['version']:
            download_from_web(self.versions['µnet']['url'], self.model_path)
            print("Model updated.")
            return
        print("Model is up to date.")

    def __init__(self, pbr, mga):
        super().__init__()
        self.pbr = pbr
        self.mga = mga
        self.versions = None
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "µnet")

    def run(self):
        self._fetch_descriptor()
        self._check_updates()
        self.mga.set_segmentation_model(self.model_path)
        self.mga.segmentation_inference()
        self.mga.segmentation_postprocessing()
        self.finished.emit()


class QtClassifyMicroglia(QObject):
    
    finished = pyqtSignal()
    update   = pyqtSignal(str, int, int)

    def _fetch_descriptor(self):
        """ Reads the JSON file from the URL and stores it in self.versions """
        try:
            self.versions = requests.get(_MODELS).json()
        except requests.exceptions.RequestException as e:
            print("Failed to fetch the models descriptor.")
            self.versions = None

    def _check_updates(self):
        if not self.versions:
            return
        if not os.path.isdir(self.model_path):
            download_from_web(self.versions['µyolo']['url'], self.model_path)
            print("Model downloaded.")
            return
        v_path = os.path.join(self.model_path, "version.txt")
        local_version = 0
        with open(v_path, 'r') as f:
            local_version = int(f.read().strip())
        if local_version < self.versions['µyolo']['version']:
            download_from_web(self.versions['µyolo']['url'], self.model_path)
            print("Model updated.")
            return
        print("Model is up to date.")

    def __init__(self, pbr, mga):
        super().__init__()
        self.pbr = pbr
        self.mga = mga
        self.versions = None
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "µyolo")

    def run(self):
        self._fetch_descriptor()
        self._check_updates()
        self.mga.set_classification_model(self.model_path)
        self.mga.classification_inference()
        self.mga.classification_postprocessing()
        self.finished.emit()