import numpy as np
import tensorflow as tf
from .VGG_ILSVRC_16_layers import VGG_ILSVRC_16_layers as VGG16_net


###########################################
## NEEDED TO USE MAXPOOL WITh ARGMAX
###########################################
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                          grad,
                          op.outputs[1],
                          op.get_attr("ksize"),
                          op.get_attr("strides"),
                          padding=op.get_attr("padding"))

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

batch_norm = tf.contrib.layers.batch_norm

convolution2d = lambda *args, **kwargs: tf.contrib.layers.convolution2d(*args,
    kernel_size=[3,3],
    padding="SAME",
    activation_fn=tf.nn.relu,
    #weights_initializer=tf.truncated_normal_initializer(0, 0.1)
    weights_initializer=tf.contrib.layers.xavier_initializer()
    )
convolution2d_transpose = lambda *args, **kwargs : tf.contrib.layers.convolution2d_transpose(*args,
    kernel_size=[3,3],
    padding="SAME",
    activation_fn=tf.nn.relu,
    #weights_initializer=tf.truncated_normal_initializer(0, 0.1)
    weights_initializer=tf.contrib.layers.xavier_initializer(), **kwargs
    )

def model(images, label_nbr, is_training=None):
    net = VGG16_net({'data': images})

    vgg_output = net.layers["conv5_3"]

    deconv_net = [vgg_output]
    print_activations(deconv_net[-1])
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 512, stride=2))
    deconv_net.append(tf.concat([deconv_net[-1], net.layers["conv4_3"]],3))
    deconv_net.append(batch_norm(deconv_net[-1]))
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 512, stride=1))
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 512, stride=1))
    print_activations(deconv_net[-1])
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 256, stride=2))
    deconv_net.append(tf.concat([deconv_net[-1], net.layers["conv3_3"]],3))
    deconv_net.append(batch_norm(deconv_net[-1]))
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 256, stride=1))
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 256, stride=1))
    print_activations(deconv_net[-1])
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 128, stride=2))
    deconv_net.append(tf.concat([deconv_net[-1], net.layers["conv2_2"]],3))
    deconv_net.append(batch_norm(deconv_net[-1]))
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 128, stride=1))
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 128, stride=1))
    print_activations(deconv_net[-1])
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 128, stride=2))
    deconv_net.append(tf.concat([deconv_net[-1], net.layers["conv1_2"]],3))
    deconv_net.append(batch_norm(deconv_net[-1]))
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 64, stride=1))
    deconv_net.append(convolution2d_transpose(deconv_net[-1], 64, stride=1))
    deconv_net.append(convolution2d_transpose(deconv_net[-1], label_nbr, stride=1))
    print_activations(deconv_net[-1])



    return deconv_net, net
