import numpy as np
import os
# from plyfile import PlyData, PlyElement
import scipy.misc
import pickle
from tqdm import *
import tensorflow as tf
import shutil

import pointcloud_tools.lib.python.PcTools as PcTls

class BackProjeter:

    def __init__(self, model_function1, model_function2, model_fusion):
            self.model_function1 = model_function1
            self.model_function2 = model_function2
            self.model_function_fusion = model_fusion


    def backProj(self,
        filename,
        label_nbr,
        dir_data,
        dir_images,
        imsize,
        input_ch,
        batch_size,
        saver_directory1,
        saver_directory2,
        saver_directoryFusion,
        images_root1,
        images_root2,
        variable_scope1,
        variable_scope2,
        variable_scope_fusion):

        # load mesh
        vertices = np.load(os.path.join(dir_data,filename+"_vertices.npz"))["arr_0"]
        faces = np.load(os.path.join(dir_data, filename+"_faces.npz"))["arr_0"].astype(int)

        # create score matrix
        scores = np.zeros((vertices.shape[0],label_nbr))
        counts = np.zeros(vertices.shape[0])

        dir_views = os.path.join(dir_images, "views")
        ### LOAD THE MODEL
        with tf.Graph().as_default() as g:

            images2 = tf.placeholder(tf.float32, [None, imsize, imsize, input_ch], name="images")
            images1 = tf.placeholder(tf.float32, [None, imsize, imsize, input_ch], name="images")
            is_training = tf.placeholder(bool)

            with tf.variable_scope(variable_scope1) as scope:
                deconv_net1, net1 = self.model_function1(images1, label_nbr, is_training)

            with tf.variable_scope(variable_scope2) as scope:
                deconv_net2, net2 = self.model_function2(images2, label_nbr, is_training)

            # create corresponding saver

            with tf.variable_scope(variable_scope_fusion) as scope:
                net_fusion, net = self.model_function_fusion(deconv_net1, deconv_net2, label_nbr)
                predictions = net_fusion[-1]

            # create saver
            saver1 = tf.train.Saver([v for v in tf.global_variables() if variable_scope1 in v.name])
            saver2 = tf.train.Saver([v for v in tf.global_variables() if variable_scope2 in v.name])
            saverFusion = tf.train.Saver([v for v in tf.global_variables() if variable_scope_fusion in v.name])

            sess = tf.Session()
            #init = tf.global_variables_initializer()
            #sess.run(init)

            ckpt = tf.train.get_checkpoint_state(saver_directory1)
            if ckpt and ckpt.model_checkpoint_path:
                saver1.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Error ...no checkpoint found...")
            ckpt = tf.train.get_checkpoint_state(saver_directory2)
            if ckpt and ckpt.model_checkpoint_path:
                saver2.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Error ...no checkpoint found...")

            ckpt = tf.train.get_checkpoint_state(saver_directoryFusion)
            #########
            # TODO Don't know why, needed to change the syntax
            # compared to previous weight loads
            #########
            model_checkpoint_path = saver_directoryFusion
            print(model_checkpoint_path)
            saverFusion.restore(sess, os.path.join(model_checkpoint_path, "model.ckpt"))


            # create the list of images in the folder
            directory1 = os.path.join(dir_images, images_root1)
            directory2 = os.path.join(dir_images, images_root2)
            files = []
            for file in os.listdir(directory1):
                if file.endswith(".png") and filename+"_" in file:
                    file = file.split(".")[:-1]
                    file = ".".join(file)
                    files.append(file)


            # load to get the size
            imsize = scipy.misc.imread(os.path.join(directory1,files[0]+".png")).shape

            # create batches
            batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

            # iterate over the batches
            for batch_files in tqdm(batches):

                # create the batch container
                batch_1= np.zeros((len(batch_files),imsize[0], imsize[1], imsize[2]), dtype=float)
                batch_2= np.zeros((len(batch_files),imsize[0], imsize[1], imsize[2]), dtype=float)
                for im_id in range(len(batch_files)):
                    batch_1[im_id] = scipy.misc.imread(os.path.join(directory1, batch_files[im_id]+".png"))
                    batch_2[im_id] = scipy.misc.imread(os.path.join(directory2, batch_files[im_id]+".png"))
                batch_1 /= 255
                batch_2 /= 255

                fd = {images1:batch_1, images2:batch_2, is_training:True}
                preds = sess.run(predictions, fd)

                # save the results
                for im_id,file in enumerate(batch_files):

                    indices = np.load(os.path.join(dir_views, file+".npz"))["arr_0"]
                    preds_ = preds[im_id]
                    indices = indices.reshape((-1))
                    indices[indices>faces.shape[0]] = 0
                    preds_ = preds_.reshape((-1, preds_.shape[2]))
                    scores[faces[indices-1][:,0]] += preds_
                    scores[faces[indices-1][:,1]] += preds_
                    scores[faces[indices-1][:,2]] += preds_

                    counts[faces[indices-1][:,0]] += 1
                    counts[faces[indices-1][:,1]] += 1
                    counts[faces[indices-1][:,2]] += 1

            counts[counts ==0] = 1
            scores /= counts[:,None]

            # force 0 label to unseen vertices
            scores[scores.sum(axis=1)==0][0] = 1

            scores = scores.argmax(axis=1)

            self.scores = scores

            # close session
            del sess



    def saveScores(self,filename):
        np.savez(filename, self.scores)

    def createLabelPLY(self, filename,
        dir_data,
        save_dir):

        # create the semantizer
        semantizer = PcTls.Semantic3D()
        semantizer.set_voxel_size(0.1)

        # loading data
        semantizer.set_vertices_numpy(os.path.join(dir_data,filename+"_vertices.npz"))
        semantizer.set_labels_numpy(os.path.join(save_dir, filename+"_scores.npz"))

        # removing unlabeled points
        semantizer.remove_unlabeled_points()

        # saving the labels
        semantizer.savePLYFile_labels(os.path.join(save_dir, filename+".ply"))
