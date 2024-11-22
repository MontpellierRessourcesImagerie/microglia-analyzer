import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
import networkx as nx
import os
import tifffile
from skan import Skeleton, summarize

def analyze_skeleton(mask, pixel_size=1.0):
    skeleton = skeletonize(mask)
    skel = Skeleton(skeleton)
    branch_data = summarize(skel, separator='_')

    num_branches      = len(branch_data)
    num_leaves        = np.sum(branch_data['branch_type'] == 1)
    num_junctions     = np.sum(branch_data['branch_type'] == 2)
    avg_branch_length = np.mean(branch_data['branch_distance']) * pixel_size
    total_length      = branch_data['branch_distance'].sum()    * pixel_size
    max_branch_length = branch_data['branch_distance'].max()    * pixel_size

    results = {
        "number_of_branches"   : num_branches,
        "number_of_leaves"     : num_leaves,
        "number_of_junctions"  : num_junctions,
        "average_branch_length": round(avg_branch_length, 2),
        "total_length"         : round(total_length, 2),
        "max_branch_length"    : round(max_branch_length, 2)
    }

    return results, skeleton

# Exemple d'utilisation
if __name__ == "__main__":
    source_dir = "/tmp/unet_working/predictions/epoch_261/"
    img_name   = "prediction_00004.tif"
    img_path   = os.path.join(source_dir, img_name)

    raw_mask   = (tifffile.imread(img_path) > 0.1).astype(np.uint8)
    labeled    = label(raw_mask, connectivity=1)
    mask       = labeled == 70
    
    tifffile.imwrite("/tmp/labeled.tif", labeled.astype(np.uint16))
    results, skeleton = analyze_skeleton(mask, 0.325)
    tifffile.imwrite("/tmp/skeleton.tif", skeleton.astype(np.uint8) * 255)

    print("Résultats de l'analyse du squelette :")
    max_key_len = max([len(key) for key in results.keys()])
    for key, value in results.items():
        print(f" | {key.replace('_', ' ').capitalize()}{' ' * (max_key_len - len(key) + 1)}: {value}")
