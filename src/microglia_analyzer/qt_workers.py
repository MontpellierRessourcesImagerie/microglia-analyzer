from qtpy.QtCore import QObject
from PyQt5.QtCore import pyqtSignal

class QtSegmentProtoplasts(QObject):
    
    finished = pyqtSignal()
    update   = pyqtSignal(str, int, int)

    def _probe_model(self, model):
        pass

    def __init__(self, pbr, mga):
        super().__init__()
        self.pbr = pbr
        self.mga = mga

    def run(self):
        self.mga.find_traps()
        self.mga.process_patches()
        self.finished.emit()