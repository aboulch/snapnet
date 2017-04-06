###
# IMPORTS
###

import numpy as np
import os
import argparse
import math

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

from OpenGL.GL import *
import scipy.misc
import pickle
from tqdm import *
import sys
import numbers
import copy

class ViewGenerator:

    # opts
    # imsize : image size

    def __init__(self):
        self.app=pg.QtGui.QApplication([])#NEED TO call this once
        self.opts = {}
        self.opts["imsize"] = 128
        self.opts["fov"] = 60

    def initialize_acquisition(self,
            dir_mesh,
            dir_images,
            filename,
        ):

        # save the paths
        self.dir_mesh = dir_mesh # root directory of acquisition
        self.dir_images = dir_images # directory to contain the images
        self.filename = filename # acquisition name
        self.dir_images_views = os.path.join(self.dir_images,"views")
        voxels_directory = os.path.join(self.dir_mesh, "voxels")

        # create the directory to save the views
        if not os.path.exists(self.dir_images_views):
            os.makedirs(self.dir_images_views)

        # load the point cloud
        self.faces = np.load(os.path.join(voxels_directory, filename+"_faces.npz"))["arr_0"].astype(int)
        self.vertices = np.load(os.path.join(voxels_directory,self.filename+"_vertices.npz"))["arr_0"]

        # give a unique color to each face
        self.face_colors = np.zeros((self.faces.shape[0],4), dtype=float)
        self.face_colors[:,3] = 255 # opaque face
        for i in range(self.faces.shape[0]):
            # black is for back ground ==> i+1
            self.face_colors[i,2] = (i+1) % 256
            self.face_colors[i,1] = ((i+1)//256)%256
            self.face_colors[i,0] = (((i+1)//256)//256)%256
        self.face_colors /= 255. # pyqtgraph takes float colors

    def set_camera_generator(self, camera_generator_function):
        self.cam_gen_function = camera_generator_function


    def cam_generator_random_position_sphere(self):

        # pick a random vertex in the faces
        face_id = np.random.randint(0,self.faces.shape[0])
        vertex_id    = np.random.randint(0,3)
        v = self.vertices[self.faces[face_id,vertex_id]]

        # create the camera
        cam = {}
        cam["index"] = face_id
        cam["x"] = v[0]
        cam["y"] = v[1]
        cam["z"] = v[2]
        cam['azimuth'] = 360*np.random.rand()
        cam['elevation'] = -90 + 180*np.random.rand()
        # self.cameras.append(cam)
        return cam

    def cam_generator_random_vertical_cone(self):

        # pick a random vertex in the faces
        face_id = np.random.randint(0,self.faces.shape[0])
        vertex_id    = np.random.randint(0,3)
        v = self.vertices[self.faces[face_id,vertex_id]]

        # create camera
        cam = {}
        cam["index"] = face_id
        cam["x"] = v[0]
        cam["y"] = v[1]
        cam["z"] = v[2]
        cam['azimuth'] = 360*np.random.rand()
        cam['elevation'] = 90 - 20*np.random.rand()
        # self.cameras.append(cam)
        return cam

    def cam_generator_random_position_upper_half_sphere(self):
        # pick a random vertex in the faces
        face_id = np.random.randint(0,self.faces.shape[0])
        vertex_id    = np.random.randint(0,3)
        v = self.vertices[self.faces[face_id,vertex_id]]

        # create the camera
        cam = {}
        cam["index"] = face_id
        cam["x"] = v[0]
        cam["y"] = v[1]
        cam["z"] = v[2]
        cam['azimuth'] = 360*np.random.rand()
        cam['elevation'] = 0 + 90*np.random.rand()
        # self.cameras.append(cam)
        return cam

    def cam_generator_random_position_lower_half_sphere(self):

        # pick a random vertex in the faces
        face_id = np.random.randint(0,self.faces.shape[0])
        vertex_id    = np.random.randint(0,3)
        v = self.vertices[self.faces[face_id,vertex_id]]

        # create the camera
        cam = {}
        cam["index"] = face_id
        cam["x"] = v[0]
        cam["y"] = v[1]
        cam["z"] = v[2]
        cam['azimuth'] = 360*np.random.rand()
        cam['elevation'] = -90 + 90*np.random.rand()
        # self.cameras.append(cam)
        return cam

    # generate the cameras
    # distance float or list (len 2), either fix distance for the camera or random distance
    def generate_cameras(self, cam_number, distance=5):
        self.cameras = []
        for i in range(cam_number):
            cam = self.cam_gen_function(self)
            assert isinstance(distance, (list, numbers.Number))
            if isinstance(distance, list):
                dist = distance[0] + np.random.rand()* (distance[1]-distance[0])
            else:
                dist = distance
            cam["distance"] = dist
            self.cameras.append(cam)
        pickle.dump( self.cameras, open( os.path.join(self.dir_images, self.filename+"_cameras.p"), "wb" ) )

    # generate cameras at different distances
    def generate_cameras_scales(self, cam_number, distances=[5,10,20]):
        self.cameras=[]
        for i in range(cam_number):
            cam = self.cam_gen_function(self)
            for dist in distances:
                cam0 = copy.deepcopy(cam)
                cam0["distance"] = dist
                self.cameras.append(cam0)
        pickle.dump( self.cameras, open( os.path.join(self.dir_images, self.filename+"_cameras.p"), "wb" ) )

    def convertQImageToMat(self,incomingImage):
        ''' Converts a QImage into an opencv MAT format '''
        incomingImage = incomingImage.convertToFormat(4)
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)
        return arr

    def update(self):
        if(self.w.count >= len(self.cameras)):
            self.w.close()
            self.app.exit()
            return

        self.w.update()

        # get the camera
        cam = self.cameras[self.w.count]

        # set camera parameters
        self.w.opts["center"] = pg.Vector(cam["x"],cam["y"],cam["z"])
        self.w.opts["distance"] = cam["distance"]
        self.w.opts["elevation"] = cam["elevation"]
        self.w.opts["azimuth"] = cam["azimuth"]
        self.w.update()

        # get image from window
        im = self.convertQImageToMat(self.w.readQImage()).astype(np.uint64)
        im = im[:,:,:3] # get only the first 3 channels
        im[:,:,2] *= 256*256
        im[:,:,1] *= 256
        im = im.sum(axis=2)
        im -= 1 #we added 1 to handle the background

        # save the matrix
        np.savez(os.path.join(self.dir_images,"views", self.filename+("_%04d" % self.w.count)), im)
        self.w.count+=1

    def generate_views(self):

        # create the widget
        self.w= None
        self.w = gl.GLViewWidget()
        self.w.resize(self.opts["imsize"],self.opts["imsize"])
        self.w.opts["fov"] = self.opts["fov"]
        self.w.setWindowTitle('SnapNet image generator')
        self.w.show()

        # initialize the parameters
        self.w.count = 0

        # create and add the mesh to the GL widget
        self.w.m1 = gl.GLMeshItem(
            vertexes=self.vertices,
            faces=self.faces,
            faceColors=self.face_colors,
            smooth=False, drawEdges=False)
        self.w.m1.setGLOptions('opaque')
        self.w.addItem(self.w.m1)

        # launch the application
        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.update)
        self.t.start(50)

        # finish
        self.app.exit(self.app.exec_())
        self.w.close()
        self.w.m1 = None
