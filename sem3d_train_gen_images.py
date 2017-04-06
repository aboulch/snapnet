import numpy as np
import os
import scipy.misc
from tqdm import *

from python.Semantic3D import Sem3D
from python.viewGenerator import ViewGenerator
from python.imageGenerator import ImageGenerator


# load the configuration file and define variables
print("Loading configuration file")
import argparse
import json
parser = argparse.ArgumentParser(description='Semantic3D')
parser.add_argument('--config', type=str, default="config.json", metavar='N',
help='config file')
args = parser.parse_args()
json_data=open(args.config).read()
config = json.loads(json_data)
input_dir = config["train_input_dir"]
directory = config["train_results_root_dir"]
voxels_directory = os.path.join(directory,"voxels")
image_directory = os.path.join(directory,config["images_dir"])

voxel_size = config["voxel_size"]
imsize = config["imsize"]
cam_number = config["train_cam_number"]

create_mesh = config["train_create_mesh"]
create_views = config["train_create_views"]
create_images = config["train_create_mesh"]

# create directories if not already existing
if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(voxels_directory):
    os.makedirs(voxels_directory)
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

# training filenames
filenames = [
    "bildstein_station1_xyz_intensity_rgb",
    "bildstein_station3_xyz_intensity_rgb",
    "bildstein_station5_xyz_intensity_rgb",
    "domfountain_station1_xyz_intensity_rgb",
    "domfountain_station2_xyz_intensity_rgb",
    "domfountain_station3_xyz_intensity_rgb",
    "neugasse_station1_xyz_intensity_rgb",
    "sg27_station1_intensity_rgb",
    "sg27_station2_intensity_rgb",
    "sg27_station4_intensity_rgb",
    "sg27_station5_intensity_rgb",
    "sg27_station9_intensity_rgb",
    "sg28_station4_intensity_rgb",
    "untermaederbrunnen_station1_xyz_intensity_rgb",
    "untermaederbrunnen_station3_xyz_intensity_rgb"
    ]

# create the view generator
view_gen = ViewGenerator()
view_gen.set_camera_generator(ViewGenerator.cam_generator_random_vertical_cone)
view_gen.opts["imsize"]= imsize

for filename in filenames:
    print(filename)

    if create_mesh:

        # create the mesher
        semantizer = Sem3D()
        semantizer.set_voxel_size(voxel_size)

        #loading data and voxelization
        print("  -- loading data")
        semantizer.load_Sem3D_labels(os.path.join(input_dir,filename+".txt"),
                os.path.join(input_dir,filename+".labels"))

        # estimate normals
        print("  -- estimating normals")
        semantizer.estimate_normals_regression(100)

        print("  -- estimating noise")
        semantizer.estimate_noise_radius(1.)

        print("  -- estimating Z orient")
        semantizer.estimate_z_orient()

        #save points and labels
        print("  -- saving plys")
        semantizer.savePLYFile(os.path.join(voxels_directory,filename+"_points.ply"))
        semantizer.savePLYFile_composite(os.path.join(voxels_directory,filename+"_composite.ply"))
        semantizer.savePLYFile_labels(os.path.join(voxels_directory,filename+"_labels.ply"))

        print("  -- building mesh")
        semantizer.build_mesh(False)
        semantizer.save_mesh(os.path.join(voxels_directory,filename+"_mesh.ply"))
        semantizer.save_mesh_composite(os.path.join(voxels_directory,filename+"_mesh_composite.ply"))
        semantizer.save_mesh_labels(os.path.join(voxels_directory,filename+"_mesh_labels.ply"))

        print("  -- extracting vertices")
        vertices = semantizer.get_vertices_numpy()
        np.savez(os.path.join(voxels_directory,filename+"_vertices"), vertices)
        print("  -- extracting normals")
        normals = semantizer.get_normals_numpy()
        np.savez(os.path.join(voxels_directory,filename+"_normals"), normals)
        print("  -- extracting faces")
        faces = semantizer.get_faces_numpy()
        np.savez(os.path.join(voxels_directory,filename+"_faces"), faces)
        print("  -- extracting colors")
        colors = semantizer.get_colors_numpy()
        np.savez(os.path.join(voxels_directory,filename+"_colors"), colors)
        print("  -- extracting composite")
        composite = semantizer.get_composite_numpy()
        np.savez(os.path.join(voxels_directory,filename+"_composite"), composite)
        print("  -- extracting labels")
        labels = semantizer.get_labels_numpy()
        np.savez(os.path.join(voxels_directory,filename+"_labels"), labels)
        print("  -- extracting labels colors")
        labelsColors = semantizer.get_labelsColors_numpy()
        np.savez(os.path.join(voxels_directory,filename+"_labelsColors"), labelsColors)

    if create_views:
        # generate_views
        print("  -- generating views")
        view_gen.initialize_acquisition(
                directory,
                image_directory,
                filename
            )
        view_gen.generate_cameras_scales(cam_number, distances=[5,10,20])
        view_gen.generate_views()

    if create_images:
        # generate images
        print("  -- generating images")
        im_gen = ImageGenerator()
        im_gen.set_isTraining(True)
        im_gen.initialize_acquisition(
                voxels_directory,
                image_directory,
                filename
            )
        im_gen.generate_images()
