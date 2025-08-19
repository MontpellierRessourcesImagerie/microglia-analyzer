from microglia_analyzer.tiles.tiler import ImageTiler2D
from microglia_analyzer.tiles import recalibrate as rc
from microglia_analyzer._tests import utils
from microglia_analyzer import ORIGINAL_PIXEL_SIZE

import random
import numpy as np
import math

def test_process_factor():
    assert math.isclose(rc.process_factor(0.1, "µm", 0.1, "µm"), 1.0)
    assert math.isclose(rc.process_factor(500, "nm", 500, "nm"), 1.0)

    assert math.isclose(rc.process_factor(0.1, "µm", 0.5, "µm"), 0.2)
    assert math.isclose(rc.process_factor(0.5, "µm", 0.1, "µm"), 5.0)

    assert math.isclose(rc.process_factor(0.5, "µm", 500, "nm"), 1.0)
    assert math.isclose(rc.process_factor(500, "nm", 0.5, "µm"), 1.0)

    assert math.isclose(rc.process_factor(0.1, "µm", 500, "nm"), 0.2)
    assert math.isclose(rc.process_factor(100, "nm", 0.5, "µm"), 0.2)

    f = 0.1725 / 0.325
    assert math.isclose(rc.process_factor(172.5, "nm", 0.325, "um"), f)
    assert math.isclose(rc.process_factor(172.5, "nm", 325, "nm"), f)
    assert math.isclose(rc.process_factor(0.1725, "um", 0.325, "um"), f)
    assert math.isclose(rc.process_factor(0.1725, "um", 325, "nm"), f)

def test_prebuilt_factors():
    assert rc.get_ori2net_factor(172.5, "nm") == rc.process_factor(172.5, "nm", 0.325, "um")
    assert rc.get_net2ori_factor(172.5, "nm") == rc.process_factor(0.325, "um", 172.5, "nm")

def test_recalibrate_shape():
    shapes = [(755, 1233), (1, 1233, 755), (3, 555, 1234)] # 2D: (C, Y, X) or (Y, X)
    for shape in shapes:
        s1 = rc.recalibrate_shape(shape, 172.5, "nm",  0.325, "um")
        s2 = rc.recalibrate_shape(s1   ,   325, "nm", 0.1725, "um")
        # Due to the use of float+rounding, we cannot expect exact equality.
        # We can only expect that the difference is less than 1 pixel.
        for i, (a, b) in enumerate(zip(shape, s2)):
            if i < len(shape) - 2:
                assert abs(a - b) == 0
            else:
                assert abs(a - b) <= 1

def test_scaling_from_calibration_ori2net():
    shape = (3, 1234, 755)
    data  = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    d2    = rc.scaling.from_calibration.ori2net(data, 0.325, "µm")
    assert d2.shape == shape
    d3    = rc.scaling.from_calibration.ori2net(data, 172.5, "nm")
    assert d3.shape[0] == shape[0]
    assert abs(int(d3.shape[1] * 0.325 / 0.1725) - shape[1]) <= 1
    assert abs(int(d3.shape[2] * 0.325 / 0.1725) - shape[2]) <= 1

def test_scaling_from_calibration_net2ori():
    shape = (3, 1234, 755)
    data  = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    d2    = rc.scaling.from_calibration.net2ori(data, 0.325, "µm")
    assert d2.shape == shape
    d3    = rc.scaling.from_calibration.net2ori(data, 172.5, "nm")
    assert d3.shape[0] == shape[0]
    assert abs(int(d3.shape[1] * 0.1725 / 0.325) - shape[1]) <= 1
    assert abs(int(d3.shape[2] * 0.1725 / 0.325) - shape[2]) <= 1

def test_scaling_from_shape():
    shape = (3, 1234, 755)
    data  = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    d2    = rc.scaling.from_calibration.ori2net(data, 172.5, "nm")
    d3    = rc.scaling.from_shape(d2, shape)
    assert d3.shape == shape

def test_demo_unet():
    # Generate a random 2D shapes in ([10, 99999], [10, 99999]).
    shape = (random.randint(512, 9999), random.randint(512, 9999))

    # Input image.
    input_img = utils.make_checkboard(shape, 128, 'grayscale')

    # The input's original pixel size and unit.
    pxl_unit = "µm"
    o = ORIGINAL_PIXEL_SIZE
    pxl_size = random.uniform(o - 0.150, o + 0.150)
    
    # Calibrate to 0.325 µm before feeding the image to the tiler.
    calibrated = rc.scaling.from_calibration.ori2net(input_img, pxl_size, pxl_unit)

    # Tiling
    tiles_manager = ImageTiler2D(128, 10, calibrated.shape)
    tiles = np.array(tiles_manager.image_to_tiles(calibrated, False))

    # Inference result
    predictions = np.copy(tiles)
    probability_map = tiles_manager.tiles_to_image(predictions)

    # Retrieve the original image size
    output_mask = rc.scaling.from_shape(probability_map, shape)
    assert output_mask.shape == input_img.shape

    difference = np.abs(input_img.astype(np.float32) - output_mask.astype(np.float32)).astype(np.uint16)
    assert np.percentile(difference, 75) == 0

def mock_yolo_inference(tile_size, max_boxes=10):
    # YOLO boxes are XYXY tuples (UL -> LR) -> we simulate the ones generated for each tile.
    num_boxes = random.randint(2, max_boxes)
    boxes = []
    for _ in range(num_boxes):
        x1 = random.randint(0     , tile_size - 5)
        y1 = random.randint(0     , tile_size - 5)
        x2 = random.randint(x1 + 1, tile_size)
        y2 = random.randint(y1 + 1, tile_size)
        boxes.append((x1, y1, x2, y2))

    return np.array(boxes)

def test_demo_yolo():
    # Generate a random 2D shapes in ([512, 9999], [512, 9999]).
    shape = (random.randint(512, 9999), random.randint(512, 9999))

    # Input image.
    input_img = utils.make_checkboard(shape, 128, 'grayscale')

    # The input's original pixel size and unit.
    pxl_unit = "µm"
    o = ORIGINAL_PIXEL_SIZE
    pxl_size = random.uniform(o - 0.150, o + 0.150)
    
    # Calibrate to 0.325 µm before feeding the image to the tiler.
    calibrated = rc.scaling.from_calibration.ori2net(input_img, pxl_size, pxl_unit)
    f = rc.get_net2ori_factor(pxl_size, pxl_unit)

    # Tiling
    tile_size = 256
    tiles_manager = ImageTiler2D(tile_size, 10, calibrated.shape)
    tiles = np.array(tiles_manager.image_to_tiles(calibrated, False))

    all_boxes = []
    for i in range(len(tiles)):
        boxes  = mock_yolo_inference(tile_size)
        print(boxes)
        print("-----")
        boxes  = (boxes.astype(np.float32) * f).astype(int)
        print(boxes)
        y, x   = tiles_manager.layout[i].ul_corner
        y, x   = int(y * f), int(x * f)
        boxes  = [(x1 + x, y1 + y, x2 + x, y2 + y) for (x1, y1, x2, y2) in boxes]
        all_boxes += boxes
        print("\n")

    # Check that the box coordinates are bound in the original shape.
    for box in all_boxes:
        assert box[0] >= 0
        assert box[1] >= 0
        assert box[2] <= shape[1]
        assert box[3] <= shape[0]
        assert box[0] <= box[2]
        assert box[1] <= box[3]

if __name__ == "__main__":
    test_demo_yolo()