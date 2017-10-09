import numpy as np
import pickle
import os
import scipy.misc
from tqdm import *

class ImageGenerator:

    def __init__(self):
        self.is_training = False

    def set_isTraining(self, is_training):
        self.is_training = is_training

    def initialize_acquisition(self,
            voxels_directory,
            dir_images,
            filename,
        ):
        #
        self.voxels_directory = voxels_directory # root directory of acquisition
        self.dir_images = dir_images # directory to contain the images
        self.filename = filename # acquisition name

        # loading data
        self.vertices = np.load(os.path.join(voxels_directory, self.filename+"_vertices.npz"))["arr_0"]
        self.faces = np.load(os.path.join(voxels_directory, self.filename+"_faces.npz"))["arr_0"].astype(int)
        self.colors = np.load(os.path.join(voxels_directory, self.filename+"_colors.npz"))["arr_0"]
        self.composite = np.load(os.path.join(voxels_directory, self.filename+"_composite.npz"))["arr_0"]
        if self.is_training:
            self.labels = np.load(os.path.join(voxels_directory, self.filename+"_labels.npz"))["arr_0"]
            self.labels_colors = np.load(os.path.join(voxels_directory, self.filename+"_labelsColors.npz"))["arr_0"]

        # loading cameras
        self.cameras = pickle.load( open( os.path.join(self.dir_images,filename+"_cameras.p"), "rb" ) )

        # create corresponding directory
        dir_images_views = os.path.join(self.dir_images,"rgb")
        if not os.path.exists(dir_images_views):
            os.makedirs(dir_images_views)
        dir_images_views = os.path.join(self.dir_images,"composite")
        if not os.path.exists(dir_images_views):
            os.makedirs(dir_images_views)

        if self.is_training:
            dir_images_views = os.path.join(self.dir_images,"labels")
            if not os.path.exists(dir_images_views):
                os.makedirs(dir_images_views)
            dir_images_views = os.path.join(self.dir_images,"labels_colors")
            if not os.path.exists(dir_images_views):
                os.makedirs(dir_images_views)


    def generate_images(self):

        cam_number = len(self.cameras)

        for i in tqdm(range(cam_number)):

            # load the face indices of current view
            indices = np.load(os.path.join(self.dir_images,"views", self.filename+("_%04d" % i)+".npz"))["arr_0"]
            mask = np.logical_and(indices>=0, indices<self.faces.shape[0])
            face_ids = indices[mask]#-1
            vertex_ids = self.faces[face_ids][:,0]

            if self.is_training:
                # label matrix
                im = np.zeros(indices.shape)
                im[mask] = self.labels[vertex_ids]
                np.savez(os.path.join(self.dir_images, "labels", self.filename+("_%04d" % i)), im)

                # label colors
                im = np.zeros(indices.shape+(3,))
                im[mask] = self.labels_colors[vertex_ids]
                scipy.misc.imsave(os.path.join(self.dir_images, "labels_colors", self.filename+("_%04d" % i))+".png", im)

            # rgb
            im = np.zeros(indices.shape+(3,))
            im[mask] = self.colors[vertex_ids]
            scipy.misc.imsave(os.path.join(self.dir_images, "rgb", self.filename+("_%04d" % i))+".png", im)

            # composite
            im = np.zeros(indices.shape+(3,))
            im[mask] = self.composite[vertex_ids]
            cam = self.cameras[i]
            center = np.array([cam["eyeX"],cam["eyeY"],cam["eyeZ"]])
            distances = np.sqrt(((self.vertices[vertex_ids]-center[None,:])**2).sum(axis=1))
            min_d = 30.
            max_d = 100.
            distances[distances < min_d] = min_d
            distances[distances > max_d] = max_d
            distances = (1-(distances - min_d)/(max_d-min_d))*255
            distances = distances.reshape(distances.shape+(1,)).repeat(3,axis=1)
            distances[:,:2] = im[mask][:,:2]
            im[mask] = distances
            scipy.misc.imsave(os.path.join(self.dir_images, "composite", self.filename+("_%04d" % i))+".png", im)
