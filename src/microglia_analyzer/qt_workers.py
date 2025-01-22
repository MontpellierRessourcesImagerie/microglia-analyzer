from qtpy.QtCore import QObject
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import requests
import os
import numpy as np
from microglia_analyzer.utils import download_from_web, get_all_tiff_files
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
        if local_version != int(self.versions['µnet']['version']):
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
        if self.mga.segmentation_model is None:
            self._fetch_descriptor()
            self._check_updates()
            self.mga._log(f"Segmenting microglia using the version {self.versions['µnet']['version']}")
            self.mga.set_segmentation_model(self.model_path)
        self.mga.segment_microglia()
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
        if local_version != int(self.versions['µyolo']['version']):
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
        if self.mga.classification_model is None:
            self._fetch_descriptor()
            self._check_updates()
            self.mga._log(f"Classifying microglia using the version {self.versions['µyolo']['version']}")
            self.mga.set_classification_model(self.model_path)
        self.mga.classify_microglia()
        self.finished.emit()

class QtMeasureMicroglia(QObject):
    
    finished = pyqtSignal()
    update   = pyqtSignal(str, int, int)

    def __init__(self, pbr, mga):
        super().__init__()
        self.pbr = pbr
        self.mga = mga

    def run(self):
        self.mga.analyze_graph()
        self.finished.emit()

class QtBatchRunners(QObject):
    
    finished = pyqtSignal()
    update   = pyqtSignal(str, int, int)
    to_kill  = pyqtSignal()

    def __init__(self, pbr, source_dir, settings):
        super().__init__()
        self.to_kill.connect(self.interupt)
        self.pbr = pbr
        self.source_dir = source_dir
        self.settings = settings
        self.images_pool = get_all_tiff_files(source_dir)
        self.tsv_lines = []
        self.is_condamned = False

    @pyqtSlot()
    def interupt(self):
        self.is_condamned = True

    def workflow(self, index):
        img_path = os.path.join(self.source_dir, self.images_pool[index])
        img_data = tifffile.imread(img_path)
        s = self.settings
        ma = MicrogliaAnalyzer(lambda x: print(x))
        ma.set_input_image(img_data)
        ma.set_calibration(*s['calibration'])
        ma.set_segmentation_model(s['unet_path'])
        ma.set_classification_model(s['yolo_path'])
        ma.set_min_surface(s['cc_min_size'])
        ma.set_proba_threshold(s['proba_threshold'])
        ma.segment_microglia()
        ma.classify_microglia()
        ma.analyze_graph()
        tsv = ma.as_tsv(self.images_pool[index])
        if index == 0:
            self.tsv_lines += tsv
        else:
            self.tsv_lines += tsv[1:]
        classified = np.zeros(ma.mask.shape, dtype=np.uint8)
        for (cls, seg_bbox) in ma.bindings:
            classified[seg_bbox[0]:seg_bbox[2], seg_bbox[1]:seg_bbox[3]] = int(cls)
        mask = (ma.mask > 0).astype(np.uint8) * classified
        control_path = os.path.join(self.source_dir, "controls", self.images_pool[index])
        tifffile.imwrite(control_path, np.stack([ma.skeleton, mask], axis=0))
    
    def write_tsv(self):
        with open(os.path.join(self.source_dir, "controls", "results.csv"), 'w') as f:
            f.write("\n".join(self.tsv_lines))

    def run(self):
        for i in range(len(self.images_pool)):
            if self.is_condamned:
                return
            print(f"=== [{str(i+1).zfill(2)}/{str(len(self.images_pool)).zfill(2)}] Processing {self.images_pool[i]}. ===")
            self.workflow(i)
            self.write_tsv()
            self.update.emit(self.images_pool[i], i+1, len(self.images_pool))
        self.finished.emit()
