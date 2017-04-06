import numpy as np
import os
import scipy.misc
from tqdm import *
import json

from python.Semantic3D import Sem3D # meshcreation from Semantic3D data files
from python.viewGenerator import ViewGenerator
from python.imageGenerator import ImageGenerator

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

input_dir = config["test_input_dir"]
directory = config["test_results_root_dir"]
voxels_directory = os.path.join(directory,"voxels")
image_directory = os.path.join(directory,config["images_dir"])

if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(voxels_directory):
    os.makedirs(voxels_directory)
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

voxel_size = config["voxel_size"]
imsize = config["imsize"]
cam_number = config["test_cam_number"]
create_mesh = config["test_create_mesh"]
create_views = config["test_create_views"]
create_images = config["test_create_mesh"]

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
        semantizer.load_Sem3D(os.path.join(input_dir,filename+".txt"))

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

        print("  -- building mesh")
        semantizer.build_mesh(False)
        semantizer.save_mesh(os.path.join(voxels_directory,filename+"_mesh.ply"))
        semantizer.save_mesh_composite(os.path.join(voxels_directory,filename+"_mesh_composite.ply"))

        # if needed a ply file can be loaded as follow
        # semantizer.loadPLYFile(os.path.join(voxels_directory,filename+"_points.ply"))

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

    if create_views:
        # generate_views
        print("  -- generating views")
        view_gen.initialize_acquisition(
                directory,
                image_directory,
                filename
            )
        view_gen.generate_cameras(cam_number, distance=20)
        view_gen.generate_views()

    if create_images:
        # generate images
        print("  -- generating images")
        im_gen = ImageGenerator()
        im_gen.set_isTraining(False)
        im_gen.initialize_acquisition(
                voxels_directory,
                image_directory,
                filename
            )
        im_gen.generate_images()
