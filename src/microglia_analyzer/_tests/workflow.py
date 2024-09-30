import tifffile
import os
from microglia_analyzer.microglia_analyzer import MicrogliaAnalyzer
from microglia_analyzer.tiles import tiler


def microglia_to_tiles():
    input_folder  = "/home/benedetti/Documents/projects/2060-microglia/data/02-tiff-data/"
    output_folder = "/home/benedetti/Documents/projects/2060-microglia/data/04-unet-patches/"
    tiles_manager = None
    for im_name in os.listdir(input_folder):
        full_path = os.path.join(input_folder, im_name)
        im_data = tifffile.imread(full_path)
        if tiles_manager is None:
            patch_size = 512
            overlap = 128
            shape = im_data.shape
            tiles_manager = tiler.ImageTiler2D(patch_size, overlap, shape)
        tiles = tiles_manager.image_to_tiles(im_data)
        for i, tile in enumerate(tiles):
            tile_name = im_name.replace(".tif", f"_{str(i).zfill(2)}.tif")
            tifffile.imwrite(os.path.join(output_folder, tile_name), tile)

def classic_workflow_test():
    def logging(message):
        print("[MA] " + message)
    
    img_target = "/home/benedetti/Documents/projects/2060-microglia/data/_testing_images/adulte 4.tif"
    
    mam = MicrogliaAnalyzer(logging)
    mam.load_image(img_target)
    mam.set_calibration(0.325, "µm")
    mam.export_patches()
    return
    mam.set_segmentation_model("µnet-v002")
    mam.set_classification_model("µyolo-V005")
    mam.segment_microglia()
    mam.classify_microglia()
    mam.make_skeletons()
    mam.extract_metrics()


if __name__ == "__main__":
    classic_workflow_test()