from qtpy.QtCore import QObject
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import requests
import os
import numpy as np

from microglia_analyzer.utils import (download_from_web, get_all_tiff_files,
                                      save_as_fake_colors)
from microglia_analyzer.ma_worker import MicrogliaAnalyzer
import tifffile

_MODELS = "https://raw.githubusercontent.com/MontpellierRessourcesImagerie/microglia-analyzer/refs/heads/main/src/microglia_analyzer/models.json"

class QtVersionableDL(QObject):

    finished = pyqtSignal()
    update   = pyqtSignal(str, int, int)

    def __init__(self, pbr, mga, idf=None, tgt_path=None):
        super().__init__()
        self.pbr = pbr # Progress bar
        self.mga = mga # Microglia analyzer
        self.target_path = tgt_path # Theoritical path to the model
        self.latest = None # The latest version of the model (from the online JSON).
        self.model_path = None # The actual path to the model
        self.identifier = idf # The identifier of the model ('µnet', 'µyolo', etc.)
        self.local_version = None # The version of the model locally stored.
    
    def run(self):
        if not self._early_abort():
            self._seek_local_model()
            self._fetch_versions()
            self._download_updates()
            self._run_model()
        self.finished.emit()
    
    def _early_abort(self):
        model = None
        try:
            model = self.mga.get_model_path(self.identifier)
        except ValueError as _:
            print(f"Invalid identifier: {self.identifier}.")
            return True
        return model is not None

    def _seek_local_model(self):
        """ Checks if a model is present locally and sets the model_path attribute. """
        if not os.path.isdir(self.target_path):
            return False
        v_path = os.path.join(self.target_path, "version.txt")
        if not os.path.isfile(v_path):
            return False
        self.model_path = self.target_path
        with open(v_path, 'r') as f:
            self.local_version = int(f.read().strip())
        return True
    
    def _fetch_versions(self):
        """ Reads the JSON file from the URL and stores it in self.versions """
        try:
            versions = requests.get(_MODELS).json()
            if versions is None:
                raise requests.exceptions.RequestException
            self.latest = versions.get(self.identifier, None)
        except requests.exceptions.RequestException as _:
            print("Failed to fetch the latest models version.")
            self.latest = None
        return self.latest is not None
    
    def _download_updates(self):
        if self.latest is None:
            return
        if self.local_version == int(self.latest['version']):
            print(f"Model '{self.identifier}' is up to date.")
            return
        try:
            download_from_web(self.latest['url'], self.target_path)
            self.model_path = self.target_path
        except requests.exceptions.RequestException as _:
            print("Failed to download the latest model.")
    
    def _run_model(self):
        return
        

class QtSegmentMicroglia(QtVersionableDL):

    def __init__(self, pbr, mga):
        super().__init__(
            pbr, 
            mga,
            "µnet",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "µnet")
        )

    def _run_model(self):
        if (self.mga.segmentation_model is None) and (self.model_path is not None):
            self.mga._log(f"Segmenting microglia using the version {self.latest['version']}")
            self.mga.set_segmentation_model(self.model_path)
        self.mga.segment_microglia()


class QtClassifyMicroglia(QtVersionableDL):

    def __init__(self, pbr, mga):
        super().__init__(
            pbr,
            mga,
            "µyolo",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "µyolo")
        )

    def _run_model(self):
        if (self.mga.classification_model is None) and (self.model_path is not None):
            self.mga._log(f"Classifying microglia using the version {self.latest['version']}")
            self.mga.set_classification_model(self.model_path)
        self.mga.classify_microglia()

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

        mga = MicrogliaAnalyzer(lambda x: print(x))
        mga.set_input_image(img_data.copy())
        mga.set_calibration(*s['calibration'])
        mga.set_segmentation_model(s['unet_path'])
        mga.set_classification_model(s['yolo_path'])
        mga.set_min_surface(s['cc_min_size'])
        mga.set_proba_threshold(s['proba_threshold'])
        mga.segment_microglia()
        mga.classify_microglia()
        mga.analyze_graph()

        controls_folder = os.path.join(self.source_dir, "controls")
        self.write_csv(mga, controls_folder, self.images_pool[index])
        self.write_mask(mga, controls_folder, self.images_pool[index])
        self.write_skeleton(mga, controls_folder, self.images_pool[index])
        self.write_classification(mga, controls_folder, self.images_pool[index])
        self.write_visual_check(mga, controls_folder, self.images_pool[index])

        tsv = mga.as_tsv(self.images_pool[index])
        self.tsv_lines += tsv if (index == 0) else tsv[1:]
    
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

    def write_csv(self, mga, controls_folder, img_name):
        measures_path = os.path.join(controls_folder, "results")
        measure_path  = os.path.join(measures_path, os.path.splitext(img_name)[0]+".csv")
        os.makedirs(measures_path, exist_ok=True)
        measures = mga.as_tsv(img_name)
        with open(measure_path, 'w') as f:
            f.write("\n".join(measures))

    def write_mask(self, mga, controls_folder, img_name):
        masks_path = os.path.join(controls_folder, "masks")
        mask_path  = os.path.join(masks_path, img_name)
        os.makedirs(masks_path, exist_ok=True)
        tifffile.imwrite(mask_path, mga.mask)
    
    def write_visual_check(self, mga, controls_folder, img_name):
        checks_path = os.path.join(controls_folder, "checks")
        check_path  = os.path.join(checks_path, os.path.splitext(img_name)[0]+".png")
        os.makedirs(checks_path, exist_ok=True)
        save_as_fake_colors(
            [mga.image, (mga.skeleton > 0).astype(np.uint8)*255], 
            mga.bindings,
            mga.class_names,
            check_path
        )

    def write_skeleton(self, mga, controls_folder, img_name):
        skeletons_path = os.path.join(controls_folder, "skeletons")
        skeleton_path  = os.path.join(skeletons_path, img_name)
        os.makedirs(skeletons_path, exist_ok=True)
        tifffile.imwrite(skeleton_path, mga.skeleton)

    def write_classification(self, mga, controls_folder, img_name):
        classifications_path = os.path.join(controls_folder, "classifications")
        classification_path  = os.path.join(classifications_path, os.path.splitext(img_name)[0]+".txt")
        os.makedirs(classifications_path, exist_ok=True)
        with open(classification_path, 'w') as f:
            f.write("\n".join([str(b) for b in mga.bindings_to_yolo()]))