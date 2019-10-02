import numpy as np
import tensorflow as tf


# ###########################################
# ## NEEDED TO USE MAXPOOL WITh ARGMAX
# ###########################################
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import gen_nn_ops
# @ops.RegisterGradient("MaxPoolWithArgmax")
# def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
#     return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
#                           grad,
#                           op.outputs[1],
#                           op.get_attr("ksize"),
#                           op.get_attr("strides"),
#                           padding=op.get_attr("padding"))

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

batch_norm = tf.contrib.layers.batch_norm

convolution2d = lambda *args, **kwargs: tf.contrib.layers.convolution2d(*args,
    kernel_size=[3,3],
    padding="SAME",
    activation_fn=tf.nn.relu,
    #weights_initializer=tf.truncated_normal_initializer(0, 0.1)
    weights_initializer=tf.contrib.layers.xavier_initializer(), **kwargs
    )
convolution2d_transpose = lambda *args, **kwargs : tf.contrib.layers.convolution2d_transpose(*args,
    kernel_size=[3,3],
    padding="SAME",
    activation_fn=tf.nn.relu,
    #weights_initializer=tf.truncated_normal_initializer(0, 0.1)
    weights_initializer=tf.contrib.layers.xavier_initializer(), **kwargs
    )


class VGG16_net:

    def __init__(self, images):
        self.layers = {}
        self.layers["conv1_1"] = convolution2d(images, 64, stride=1, scope="conv1_1")
        self.layers["conv1_2"] = convolution2d(self.layers["conv1_1"], 64, stride=1, scope="conv1_2")

        self.layers["pool1"] = tf.nn.max_pool(self.layers["conv1_2"], [1, 2, 2, 1], [1,2,2,1], padding="SAME")
        
        self.layers["conv2_1"] = convolution2d(self.layers["pool1"], 128, stride=1, scope="conv2_1")
        self.layers["conv2_2"] = convolution2d(self.layers["conv2_1"], 128, stride=1, scope="conv2_2")

        self.layers["pool2"] = tf.nn.max_pool(self.layers["conv2_2"], [1, 2, 2, 1], [1,2,2,1], padding="SAME")

        self.layers["conv3_1"] = convolution2d(self.layers["pool2"], 256, stride=1, scope="conv3_1")
        self.layers["conv3_2"] = convolution2d(self.layers["conv3_1"], 256, stride=1, scope="conv3_2")
        self.layers["conv3_3"] = convolution2d(self.layers["conv3_2"], 256, stride=1, scope="conv3_3")

        self.layers["pool3"] = tf.nn.max_pool(self.layers["conv3_3"], [1, 2, 2, 1], [1,2,2,1], padding="SAME")

        self.layers["conv4_1"] = convolution2d(self.layers["pool3"], 512, stride=1, scope="conv4_1")
        self.layers["conv4_2"] = convolution2d(self.layers["conv4_1"], 512, stride=1, scope="conv4_2")
        self.layers["conv4_3"] = convolution2d(self.layers["conv4_2"], 512, stride=1, scope="conv4_3")

        self.layers["pool4"] = tf.nn.max_pool(self.layers["conv4_3"], [1, 2, 2, 1], [1,2,2,1], padding="SAME")
        
        self.layers["conv5_1"] = convolution2d(self.layers["pool4"], 512, stride=1, scope="conv5_1")
        self.layers["conv5_2"] = convolution2d(self.layers["conv5_1"], 512, stride=1, scope="conv5_2")
        self.layers["conv5_3"] = convolution2d(self.layers["conv5_2"], 512, stride=1, scope="conv5_3")

    def load(self, filename, variable_scope, session):
        with tf.variable_scope(variable_scope) as scope:
            data_dict = np.load(filename, encoding="bytes").tolist()
            for op_name in data_dict:
                with tf.variable_scope(op_name, reuse=True):
                    for param_name, data in data_dict[op_name].items():
                        param_name = param_name.decode('utf-8')
                        try:
                            var = tf.get_variable(param_name)
                            session.run(var.assign(data))
                        except ValueError:
                            print("Error", op_name, param_name)
                            #if not ignore_missing:
                            #    raise
        


def model(images, label_nbr, is_training=None):
    net = VGG16_net(images)

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
