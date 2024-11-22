from qtpy.QtCore import QObject
from PyQt5.QtCore import pyqtSignal
import requests
import os
import numpy as np
from microglia_analyzer.utils import download_from_web
from microglia_analyzer.ma_worker import MicrogliaAnalyzer
import tifffile

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
        if local_version < int(self.versions['µnet']['version']):
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
        if local_version < int(self.versions['µyolo']['version']):
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
        self.mga.bind_classifications()
        self.finished.emit()

class QtMeasureMicroglia(QObject):
    
    finished = pyqtSignal()
    update   = pyqtSignal(str, int, int)

    def __init__(self, pbr, mga):
        super().__init__()
        self.pbr = pbr
        self.mga = mga

    def run(self):
        self.mga.analyze_as_graph()
        self.finished.emit()

class QtBatchRunners(QObject):
    
    finished = pyqtSignal()
    update   = pyqtSignal(str, int, int)

    def __init__(self, pbr, source_dir, settings):
        super().__init__()
        self.pbr = pbr
        self.source_dir = source_dir
        self.settings = settings
        self.images_pool = [f for f in os.listdir(source_dir) if f.endswith(".tif")]
        self.csv_lines = []

    def workflow(self, index):
        img_path = os.path.join(self.source_dir, self.images_pool[index])
        img_data = tifffile.imread(img_path)
        s = self.settings
        ma = MicrogliaAnalyzer(lambda x: print(x))
        ma.set_input_image(img_data)
        ma.set_calibration(*s['calibration'])
        ma.set_segmentation_model(s['unet_path'])
        ma.set_classification_model(s['yolo_path'])
        ma.set_cc_min_size(s['cc_min_size'])
        ma.set_proba_threshold(s['proba_threshold'])
        ma.segmentation_inference()
        ma.segmentation_postprocessing()
        ma.set_min_score(s['min_score'])
        ma.classification_inference()
        ma.classification_postprocessing()
        ma.bind_classifications()
        ma.analyze_as_graph()
        csv = ma.as_csv(self.images_pool[index])
        if index == 0:
            self.csv_lines += csv
        else:
            self.csv_lines += csv[1:]
        control_path = os.path.join(self.source_dir, "controls", self.images_pool[index])
        tifffile.imwrite(control_path, np.stack([ma.skeleton, ma.mask], axis=0))
    
    def write_csv(self):
        with open(os.path.join(self.source_dir, "controls", "results.csv"), 'w') as f:
            f.write("\n".join(self.csv_lines))

    def run(self):
        for i in range(len(self.images_pool)):
            self.workflow(i)
            self.write_csv()
            self.update.emit(self.images_pool[i], i+1, len(self.images_pool))
        self.finished.emit()