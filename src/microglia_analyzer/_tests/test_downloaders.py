import pytest
from unittest.mock import patch
from unittest.mock import Mock

from PyQt5.QtCore import QObject, pyqtSignal, QCoreApplication

import os
from tempfile import gettempdir
import shutil
import requests

from microglia_analyzer.qt_workers import QtVersionableDL, QtSegmentMicroglia, QtClassifyMicroglia
from microglia_analyzer.ma_worker import MicrogliaAnalyzer
from microglia_analyzer.utils import download_from_web

MODEL_PATH = os.path.join(gettempdir(), "test")

# A tester:
# - Identifier invalide.

# 1. Nothing local, no internet.
# 2. Nothing local, version ok, no download.
# 3. Nothing local, version not ok, try download.
# 4. Local available, no internet.
# 5. Local available, version ok, up to date.
# 6. Local available, version ok, need update.
# 7. Model already set.

# function that takes any number of arguments.
def make_model_path(*args):
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    v_path = os.path.join(MODEL_PATH, "version.txt")
    with open(v_path, "w") as f:
        f.write("-1")


def failed_download(*args):
    raise requests.exceptions.RequestException

@pytest.fixture
def worker_001():
    if os.path.isdir(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    mock_mga = Mock()
    mock_mga.get_model_path.return_value = None
    with patch("requests.get") as mock_get, patch("microglia_analyzer.utils.download_from_web") as mock_download:
        mock_get.return_value.json.return_value = None
        mock_download.return_value = None
        mock_download.side_effect = make_model_path
        qt_app = QCoreApplication([])
        worker = QtVersionableDL(
            None, 
            mock_mga,
            "test",
            MODEL_PATH
        )
        yield worker

def test_worker_v001(worker_001):
    signal_received = []
    def on_finished():
        signal_received.append(True)
    worker_001.finished.connect(on_finished)
    worker_001.run()
    assert signal_received == [True]
    assert worker_001.target_path == MODEL_PATH
    assert worker_001.model_path == None
    assert worker_001.latest == None
    assert worker_001.local_version == None

@pytest.fixture
def worker_002():
    if os.path.isdir(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    mock_mga = Mock()
    mock_mga.get_model_path.return_value = None
    mock_request = Mock()
    mock_request.json.return_value = {"test": {"version": "1", "url": "http://test.com"}}
    with patch("requests.get") as mock_get, patch("microglia_analyzer.qt_workers.download_from_web") as mock_download:
        mock_get.return_value = mock_request
        mock_download.return_value = None
        mock_download.side_effect = failed_download
        qt_app = QCoreApplication([])
        worker = QtVersionableDL(
            None, 
            mock_mga,
            "test",
            MODEL_PATH
        )
        yield worker

def test_worker_v002(worker_002):
    signal_received = []
    def on_finished():
        signal_received.append(True)
    worker_002.finished.connect(on_finished)
    worker_002.run()
    assert signal_received == [True]
    assert worker_002.target_path == MODEL_PATH
    assert worker_002.identifier == "test"
    assert worker_002.model_path == None
    assert worker_002.latest == {"version": "1", "url": "http://test.com"}
    assert worker_002.local_version == None

@pytest.fixture
def worker_004():
    if not os.path.isdir(MODEL_PATH):
        make_model_path()
    mock_mga = Mock()
    mock_mga.get_model_path.return_value = None
    mock_request = Mock()
    mock_request.json.return_value = None
    with patch("requests.get") as mock_get, patch("microglia_analyzer.qt_workers.download_from_web") as mock_download:
        mock_get.return_value = mock_request
        mock_download.return_value = None
        mock_download.side_effect = failed_download
        qt_app = QCoreApplication([])
        worker = QtVersionableDL(
            None, 
            mock_mga,
            "test",
            MODEL_PATH
        )
        yield worker

def test_worker_v004(worker_004):
    signal_received = []
    def on_finished():
        signal_received.append(True)
    worker_004.finished.connect(on_finished)
    worker_004.run()
    assert signal_received == [True]
    assert worker_004.target_path == MODEL_PATH
    assert worker_004.identifier == "test"
    assert worker_004.model_path == MODEL_PATH
    assert worker_004.latest == None
    assert worker_004.local_version == -1

@pytest.fixture
def worker_005():
    if not os.path.isdir(MODEL_PATH):
        make_model_path()
    mock_mga = Mock()
    mock_mga.get_model_path.return_value = None
    mock_request = Mock()
    mock_request.json.return_value = {"test": {"version": "-1", "url": "http://test.com"}}
    with patch("requests.get") as mock_get, patch("microglia_analyzer.qt_workers.download_from_web") as mock_download:
        mock_get.return_value = mock_request
        mock_download.return_value = None
        mock_download.side_effect = failed_download
        qt_app = QCoreApplication([])
        worker = QtVersionableDL(
            None, 
            mock_mga,
            "test",
            MODEL_PATH
        )
        yield worker

def test_worker_v005(worker_005):
    signal_received = []
    def on_finished():
        signal_received.append(True)
    worker_005.finished.connect(on_finished)
    worker_005.run()
    assert signal_received == [True]
    assert worker_005.target_path == MODEL_PATH
    assert worker_005.identifier == "test"
    assert worker_005.model_path == MODEL_PATH
    assert worker_005.latest == {"version": "-1", "url": "http://test.com"}
    assert worker_005.local_version == -1