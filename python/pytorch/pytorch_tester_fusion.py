import numpy as np
import tensorflow as tf
import os
from random import shuffle
import scipy.misc
import shutil

from tqdm import *

class TesterFusion:

    def __init__(self, model_function1, model_function2, model_fusion):
            self.model_function1 = model_function1
            self.model_function2 = model_function2
            self.model_function_fusion = model_fusion

    def Test(self,
        imsize,
        input_ch,
        label_nbr,
        batch_size,
        dir_images,
        saver_directory1,
        saver_directory2,
        saver_directoryFusion,
        images_root1,
        images_root2,
        result_directory,
        variable_scope1,
        variable_scope2,
        variable_scope_fusion):

        with tf.Graph().as_default() as g:

            images2 = tf.placeholder(tf.float32, [None, imsize, imsize, input_ch], name="images")
            images1 = tf.placeholder(tf.float32, [None, imsize, imsize, input_ch], name="images")

            with tf.variable_scope(variable_scope1) as scope:
                deconv_net1, net1 = self.model_function1(images1, label_nbr)

            with tf.variable_scope(variable_scope2) as scope:
                deconv_net2, net2 = self.model_function2(images2, label_nbr)

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
                if file.endswith(".png"):
                    file = file.split(".")[:-1]
                    file = ".".join(file)
                    files.append(file)

            # create directory
            if os.path.exists(result_directory):
                shutil.rmtree(result_directory)
            os.makedirs(result_directory)


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

                fd = {images1:batch_1, images2:batch_2}
                preds = sess.run(predictions, fd)

                # save the results
                for im_id in range(len(batch_files)):
                    np.savez(os.path.join(result_directory, batch_files[im_id]), preds[im_id])

            # close session
            del sess
