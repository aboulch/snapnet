import numpy as np
import os
# from plyfile import PlyData, PlyElement
import scipy.misc
import pickle
from tqdm import *
import shutil
import json


print("Loading configuration file")
# Training settings
import argparse
parser = argparse.ArgumentParser(description='Semantic3D')
parser.add_argument('--config', type=str, default="config.json", metavar='N',
help='config file')
args = parser.parse_args()
json_data=open(args.config).read()
config = json.loads(json_data)

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


imsize = config["imsize"]
directory = config["test_results_root_dir"]
voxels_directory = os.path.join(directory,"voxels")
dir_images = os.path.join(directory,config["images_dir"])
saver_directory_rgb = config["saver_directory_rgb"]
saver_directory_composite = config["saver_directory_composite"]
saver_directory_fusion = config["saver_directory_fusion"]

label_nbr = config["label_nbr"]
batch_size = config["batch_size"]
input_ch = config["input_ch"]

save_dir = config["output_directory"]

backend = config["backend"]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if backend == "tensorflow":

    import tensorflow as tf
    from python.tf.tensorflow_tester_backprojeter import BackProjeter
    import python.tf.models.tensorflow_unet as model
    import python.tf.models.tensorflow_residual_fusion as model_fusion

    backproj = BackProjeter(model.model, model.model, model_fusion.model)

    for filename in filenames:

        backproj.backProj(
            filename=filename,
            label_nbr=label_nbr,
            dir_data=voxels_directory,
            dir_images=dir_images,
            imsize=imsize,
            input_ch=input_ch,
            batch_size=batch_size,
            saver_directory1=saver_directory_rgb,
            saver_directory2=saver_directory_composite,
            saver_directoryFusion=saver_directory_fusion,
            images_root1="rgb",
            images_root2="composite",
            variable_scope1="rgb",
            variable_scope2="composite",
            variable_scope_fusion="fusion")

        backproj.saveScores(os.path.join(save_dir, filename+"_scores"))
        backproj.createLabelPLY(filename, dir_data=voxels_directory, save_dir=save_dir)

if backend == "pytorch":

    from python.pytorch.pytorch_tester_backprojeter import BackProjeter
    from python.pytorch.models.net_unet import unet as model
    from python.pytorch.models.net_fusion import fusion_net as model_fusion

    backproj = BackProjeter(model, model, model_fusion)

    for filename in filenames:

        backproj.backProj(
            filename=filename,
            label_nbr=label_nbr,
            dir_mesh=voxels_directory,
            dir_images=dir_images,
            imsize=imsize,
            input_ch=input_ch,
            batch_size=batch_size,
            saver_directory1=saver_directory_rgb,
            saver_directory2=saver_directory_composite,
            saver_directoryFusion=saver_directory_fusion,
            images_root1="rgb",
            images_root2="composite",
            variable_scope1="rgb",
            variable_scope2="composite",
            variable_scope_fusion="fusion")

        backproj.saveScores(os.path.join(save_dir, filename+"_scores.txt"))
        backproj.saveSemantizedCloud(filename, voxels_directory, save_dir)
