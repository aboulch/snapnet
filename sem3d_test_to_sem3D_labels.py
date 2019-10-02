import numpy as np
import os
import scipy.misc
from tqdm import *
import json

import semantic3D_utils.lib.python.semantic3D as Sem3D

# load the configuration file and define variables
print("Loading configuration file")
import argparse
parser = argparse.ArgumentParser(description='Semantic3D')
parser.add_argument('--config', type=str, default="config.json", metavar='N',
help='config file')
args = parser.parse_args()
json_data=open(args.config).read()
config = json.loads(json_data)

input_dir = config["test_input_dir"]
voxel_size = config["voxel_size"]
output_dir = config["output_directory"]

filenames = [
        "birdfountain_station1_xyz_intensity_rgb",
        "castleblatten_station1_intensity_rgb",
        "castleblatten_station5_xyz_intensity_rgb",
        "marketplacefeldkirch_station1_intensity_rgb",
        "marketplacefeldkirch_station4_intensity_rgb",
        "marketplacefeldkirch_station7_intensity_rgb",
        "sg27_station10_intensity_rgb",
        "sg27_station3_intensity_rgb",
        "sg27_station6_intensity_rgb",
        "sg27_station8_intensity_rgb",
        "sg28_station2_intensity_rgb",
        "sg28_station5_xyz_intensity_rgb",
        "stgallencathedral_station1_intensity_rgb",
        "stgallencathedral_station3_intensity_rgb",
        "stgallencathedral_station6_intensity_rgb"
    ]

# create outpu directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in filenames:
    print(filename)


    mesh_filename = os.path.join(output_dir, filename+".ply")
    sem3d_cloud_txt = os.path.join(input_dir,filename+".txt")
    output_results = os.path.join(output_dir, filename+".txt")

    semantizer.mesh_to_label_file_no_labels(mesh_filename,sem3d_cloud_txt,output_results)
    Sem3D.project_labels_to_pc()