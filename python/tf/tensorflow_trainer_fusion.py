import numpy as np
import tensorflow as tf
import os
from random import shuffle
import scipy.misc
import shutil

from tqdm import *

class TrainerFusion:

    def __init__(self, model_function1, model_function2, model_function_fusion):
            self.model_function1 = model_function1
            self.model_function2 = model_function2
            self.model_function_fusion = model_function_fusion

    def train(self,
        imsize,
        batch_size,
        input_ch,
        epoch_nbr,
        net_weights_init,
        dir_images,
        saver_directory1,
        saver_directory2,
        saver_directory,
        images_root1,
        images_root2,
        label_nbr,
        learning_rate,
        variable_scope1,
        variable_scope2,
        variable_scope_fusion):

        with tf.Graph().as_default() as g:

            images2 = tf.placeholder(tf.float32, [None, imsize, imsize, input_ch], name="images")
            images1 = tf.placeholder(tf.float32, [None, imsize, imsize, input_ch], name="images")
            labels = tf.placeholder(tf.int32, [None, imsize, imsize], name="labels")

            with tf.variable_scope(variable_scope1) as scope:
                deconv_net1, net1 = self.model_function1(images1, label_nbr)

            with tf.variable_scope(variable_scope2) as scope:
                deconv_net2, net2 = self.model_function2(images2, label_nbr)

            # create corresponding saver
            saver1 = tf.train.Saver([v for v in tf.global_variables() if variable_scope1 in v.name])
            saver2 = tf.train.Saver([v for v in tf.global_variables() if variable_scope2 in v.name])

            with tf.variable_scope(variable_scope_fusion) as scope:
                net_fusion, net = self.model_function_fusion(deconv_net1, deconv_net2, label_nbr)
                predictions = net_fusion[-1]

            # create saver
            saver = tf.train.Saver([v for v in tf.global_variables() if variable_scope_fusion in v.name])

            # error
            reshaped_labels = tf.reshape(labels, [-1])
            reshaped_predictions = tf.reshape(predictions,[-1,label_nbr])
            loss = tf.contrib.losses.sparse_softmax_cross_entropy(reshaped_predictions, reshaped_labels)

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_step = optimizer.minimize(loss)

            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)

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




            # create the list of images in the folder
            directory1 = os.path.join(dir_images, images_root1)
            directory2 = os.path.join(dir_images, images_root2)
            directory_labels = os.path.join(dir_images, "labels/")
            files = []
            for file in os.listdir(directory_labels):
                if file.endswith(".npz"):
                    file = file.split(".")[:-1]
                    file = ".".join(file)
                    files.append(file)


            # create directory
            if os.path.exists(saver_directory):
                shutil.rmtree(saver_directory)
            os.makedirs(saver_directory)

            # open file for loss
            f = open(os.path.join(saver_directory,"loss.txt"),'w')

            # iterate
            for epoch in range(epoch_nbr):
                print("epoch "+str(epoch))

                total_loss = 0

                # create batches
                shuffle(files)
                batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
                batches = batches[:-1] # remove last batch (potentially not the same size)

                batch_1= np.zeros((batch_size,imsize, imsize, input_ch), dtype=float)
                batch_2= np.zeros((batch_size,imsize, imsize, input_ch), dtype=float)
                labels_ = np.zeros((batch_size,imsize, imsize), dtype=int)
                for batch_files in tqdm(batches):
                    for im_id in range(len(batch_files)):
                        batch_1[im_id] = scipy.misc.imread(os.path.join(directory1, batch_files[im_id]+".png"))
                        batch_2[im_id] = scipy.misc.imread(os.path.join(directory2, batch_files[im_id]+".png"))
                        labels_[im_id] = np.load(os.path.join(directory_labels, batch_files[im_id]+".npz"))["arr_0"]
                    batch_1 /= 255
                    batch_2 /= 255

                    fd = {images1:batch_1, images2:batch_2, labels:labels_}

                    [l,tr_] = sess.run([loss, train_step], fd)
                    total_loss += l

                print(total_loss/(len(batches)*batch_size))

                f.write(str(total_loss/(len(batches)*batch_size))+" \n")
                f.flush()

                if((epoch+1)%10==0):
                    # save the model
                    saver.save(sess, os.path.join(saver_directory,"model.ckpt"))

            # save the model
            saver.save(sess, os.path.join(saver_directory,"model.ckpt"))

            # close file
            f.close()

            # close session
            del sess
