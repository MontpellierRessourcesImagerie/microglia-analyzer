import cv2
from microglia_analyzer import ORIGINAL_PIXEL_SIZE, ORIGINAL_UNIT
import pint

"""
Both the UNet and the YOLOv5 trained for this project were trained with images having a pixel size of 0.325 µm.
So, to avoid having to retrain the models, we will recalibrate the images to have the same pixel size.
They will be either upscaled or downscaled to artificially reach a pixel size of 0.325 µm.
"""

def process_factor(input_calib, input_unit):
    """
    We process the factor to pass from the input calibration to the original calibration.
    We will multiply the input image's height and width by this factor to get the new size.
    """
    ureg = pint.UnitRegistry()
    l1 = ORIGINAL_PIXEL_SIZE * ureg(ORIGINAL_UNIT)
    l2 = input_calib * ureg(input_unit)
    l2_c = l2.to(ORIGINAL_UNIT)
    return float(l1 / l2_c)


def recalibrate_shape(input_shape, input_calib, input_unit):
    """
    The shape of the image has to be recalibrated to have a pixel size of 0.325 µm.

    Args:
        input_shape (tuple): The shape of the input image.
        input_calib (float): The calibration of the input image.
        input_unit (str)   : The unit of the input calibration.
    
    Returns:
        tuple: The recalibrated shape.
    """
    factor = process_factor(input_calib, input_unit)
    return tuple([int(i * factor) for i in input_shape])


def recalibrate_image(input_image, input_calib, input_unit, inter=True):
    """
    The image has to be recalibrated to have a pixel size of 0.325 µm.
    Its shape is either (Y, X) or (Y, X, C).
    If pixels are wider, it takes less pixels to represent a similar object.
    So, passing to a larger size should give a factor smaller than 1.0.
    The factor is calculated with the input length in the denominator.

    Args:
        input_image (np.array): The image to recalibrate.
        input_calib (float)   : The calibration of the input image.
        input_unit (str)      : The unit of the input calibration.
    
    Returns:
        np.array: The recalibrated (interpolated) image.
    """
    new_shape = recalibrate_shape(input_image.shape, input_calib, input_unit)
    interpolation = cv2.INTER_CUBIC if inter else cv2.INTER_NEAREST
    return cv2.resize(input_image, new_shape, interpolation=interpolation)