from skimage.transform import resize
from microglia_analyzer import ORIGINAL_PIXEL_SIZE, ORIGINAL_UNIT
import pint

"""
Both the UNet and the YOLOv5 from this project were trained with images having a pixel size of 0.325 µm.
So, to avoid having to retrain the models, we will recalibrate the images to have the same pixel size.
They will be either upscaled or downscaled to artificially reach a pixel size of 0.325 µm.
"""

def process_factor(input_px_size, input_unit, output_px_size, output_unit):
    """
    Process the scaling factor to convert the input pixel size to the output pixel size.

    Args:
        input_px_size (float): The pixel size of the input image.
        input_unit (str)     : The unit of the input pixel size.
        output_px_size (float): The pixel size of the output image.
        output_unit (str)    : The unit of the output pixel size.

    Returns:
        float: The scaling factor to convert the input pixel size to the output pixel size.
    """
    ureg = pint.UnitRegistry()
    input_px_basis = input_px_size * ureg(input_unit)
    output_px_basis = output_px_size * ureg(output_unit)
    return float(input_px_basis / output_px_basis)

def get_ori2net_factor(input_px_size, input_unit):
    """
    Get the scaling factor to convert the input pixel size to the UNet pixel size (0.325 µm).

    Args:
        input_px_size (float): The pixel size of the input image.
        input_unit (str)     : The unit of the input pixel size.

    Returns:
        float: The scaling factor to convert the input pixel size to the UNet pixel size.
    """
    return process_factor(input_px_size, input_unit, ORIGINAL_PIXEL_SIZE, ORIGINAL_UNIT)

def get_net2ori_factor(output_px_size, output_unit):
    """
    Get the scaling factor to convert the UNet pixel size (0.325 µm) to the output pixel size.

    Args:
        output_px_size (float): The pixel size of the output image.
        output_unit (str)     : The unit of the output pixel size.

    Returns:
        float: The scaling factor to convert the UNet pixel size to the output pixel size.
    """
    return process_factor(ORIGINAL_PIXEL_SIZE, ORIGINAL_UNIT, output_px_size, output_unit)

def recalibrate_shape(input_shape, input_px_size, input_unit, output_px_size, output_unit):
    """
    Takes the shape of an image and its calibration to process its shape with another calibration.
    The number of channels (C) is not modified, only the Y and X dimensions are recalibrated.

    Args:
        input_shape (tuple): The shape of the input image.
        input_px_size (float): The pixel size of the input image.
        input_unit (str): The unit of the input pixel size.
        output_px_size (float): The pixel size of the output image.
        output_unit (str): The unit of the output pixel size.
    
    Returns:
        tuple: The recalibrated shape.
    """
    # Only Y, and X have to be scaled, we don't touch the number of channels (C, Y, X).
    factor = process_factor(input_px_size, input_unit, output_px_size, output_unit)
    return input_shape[:-2] + tuple([int(round(factor * i)) for i in input_shape[-2:]])


class scaling: # namespace

    class from_calibration:

        @staticmethod
        def ori2net(data, pixel_size, unit, inter=True):
            """
            Scales an image to simulate a pixel size of 0.325 µm.
            Made to convert an input image ("original") to something that the UNet ("network") can process.

            Args:
                data (np.array): The image to scale, with a pixel size that is not 0.325µm.
                pixel_size (float): The pixel size of the 'data' image.
                unit (str): The unit of the pixel size.
                inter (bool): Whether to use interpolation or not. Defaults to True.

            Returns:
                np.array: The scaled image with a pixel size of 0.325µm.
            """
            new_shape = recalibrate_shape(data.shape, pixel_size, unit, ORIGINAL_PIXEL_SIZE, ORIGINAL_UNIT)
            return resize(
                data, 
                new_shape, 
                order=3 if inter else 0,
                preserve_range=True,
                anti_aliasing=False
            )
        
        @staticmethod
        def net2ori(data, pixel_size, unit, inter=True):
            """
            Scales an image having a pixel size of 0.325µm to the original pixel size.
            Made to convert the UNet's output ("network") to something having the original pixel size ("original").

            Args:
                data (np.array): The image to scale, with a pixel size of 0.325µm.
                pixel_size (float): The target pixel size to reach with the scaling.
                unit (str): The unit of the pixel size.
                inter (bool): Whether to use interpolation or not. Defaults to True.
            
            Returns:
                np.array: The scaled image with the target pixel size.
            """
            new_shape = recalibrate_shape(data.shape, ORIGINAL_PIXEL_SIZE, ORIGINAL_UNIT, pixel_size, unit)
            return resize(
                data, 
                new_shape, 
                order=3 if inter else 0,
                preserve_range=True,
                anti_aliasing=False
            )

    @staticmethod
    def from_shape(data, shape, inter=True):
        """
        Scales an image to mimic a pixel size of 0.325 µm.
        Made to convert an input image ("original") to something that the UNet ("network") can process.

        Args:
            data (np.array): The image to scale, with a pixel size that is not 0.325µm.
            shape (tuple): The target shape to reach with the scaling.
            inter (bool): Whether to use interpolation or not. Defaults to True.

        Returns:
            np.array: The scaled image with a pixel size of 0.325µm.
        """
        return resize(
            data, 
            shape, 
            order=3 if inter else 0,
            preserve_range=True,
            anti_aliasing=False
        )


if __name__ == "__main__":
    import tifffile
    input = "/home/benedetti/Documents/projects/2060-microglia/data/2025-04-11-calibration-debug/Snap-1272 contro.tiff"
    output = "/home/benedetti/Documents/projects/2060-microglia/data/2025-04-11-calibration-debug/scaled-python.tif"
    imin = tifffile.imread(input)
    imscaled = scaling.from_calibration.ori2net(imin, 172.5, "nm")
    imscaled = scaling.from_shape(imscaled, imin.shape)
    tifffile.imwrite(output, imscaled)