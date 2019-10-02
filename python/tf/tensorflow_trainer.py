import numpy as np
import tensorflow as tf
import os
from random import shuffle
import scipy.misc
import shutil

from tqdm import *

class Trainer:

    def __init__(self, model_function):
            self.model_function = model_function

    def train(self,
        imsize,
        batch_size,
        input_ch,
        epoch_nbr,
        net_weights_init,
        dir_images,
        saver_directory,
        images_root,
        label_nbr,
        learning_rate,
        variable_scope="s"):

        with tf.Graph().as_default() as g:

            # create placeholders
            images = tf.placeholder(tf.float32, [None, imsize, imsize, input_ch], name="images")
            labels = tf.placeholder(tf.int32, [None, imsize, imsize], name="labels")

            with tf.variable_scope(variable_scope) as scope:

                # create model
                deconv_net, net = self.model_function(images, label_nbr)
                predictions = deconv_net[-1]

            # create saver
            saver = tf.train.Saver([v for v in tf.global_variables() if variable_scope in v.name])

            # error
            reshaped_labels = tf.reshape(labels, [-1])
            reshaped_predictions = tf.reshape(predictions,[-1,label_nbr])
            loss = tf.contrib.losses.sparse_softmax_cross_entropy(reshaped_predictions, reshaped_labels)

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_step = optimizer.minimize(loss)


            # create session
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)

            # load net weights if needed
            if net is not None:
                net.load(net_weights_init, variable_scope = variable_scope, session=sess)
                #net.load(net_weights_init, sess)

            # create the list of images in the folder
            directory = os.path.join(dir_images, images_root)
            directory_labels = os.path.join(dir_images, "labels/")
            files = []
            for file in os.listdir(directory_labels):
                if file.endswith(".npz"):
                    file = file.split(".")[:-1]
                    file = ".".join(file)
                    files.append(file)


            # load to get the size
            imsize = scipy.misc.imread(os.path.join(directory,files[0]+".png")).shape

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

                batch_= np.zeros((batch_size,imsize[0], imsize[1], imsize[2]), dtype=float)
                labels_ = np.zeros((batch_size,imsize[0], imsize[1]), dtype=int)
                for batch_files in tqdm(batches):
                    for im_id in range(len(batch_files)):
                        batch_[im_id] = scipy.misc.imread(os.path.join(directory, batch_files[im_id]+".png"))
                        labels_[im_id] = np.load(os.path.join(directory_labels, batch_files[im_id]+".npz"))["arr_0"]
                    batch_ /= 255

                    fd = {images:batch_, labels:labels_}

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
