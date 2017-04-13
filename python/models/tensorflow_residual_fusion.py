import numpy as np
import tensorflow as tf



convolution2d = lambda *args, **kwargs: tf.contrib.layers.convolution2d(*args,
    kernel_size=[3,3],
    stride=1,
    padding="SAME",
    activation_fn=tf.nn.relu,
    #weights_initializer=tf.truncated_normal_initializer(0, 0.1)
    weights_initializer=tf.contrib.layers.xavier_initializer()
    )

def model(net1, net2, label_nbr):


    net = [tf.concat([tf.stop_gradient(net1[-3]),tf.stop_gradient(net2[-3])],3)]
    net.append(convolution2d(net[-1], 64))
    net.append(convolution2d(net[-1], 64))
    net.append(convolution2d(net[-1], label_nbr))
    net.append(tf.stop_gradient(net1[-1])+tf.stop_gradient(net2[-1])+net[-1])

    return net, None
