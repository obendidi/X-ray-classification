import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]

def vgg(rgb, train_mode=True):

    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
    assert red.get_shape().as_list()[1:] == [512, 512, 1]
    assert green.get_shape().as_list()[1:] == [512, 512, 1]
    assert blue.get_shape().as_list()[1:] == [512, 512, 1]
    bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [512, 512, 3]

    conv1_1 = conv_layer(bgr, 3, 64, "conv1_1")
    conv1_2 = conv_layer(conv1_1, 64, 64, "conv1_2")
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, 64, 128, "conv2_1")
    conv2_2 = conv_layer(conv2_1, 128, 128, "conv2_2")
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, 128, 256, "conv3_1")
    conv3_2 = conv_layer(conv3_1, 256, 256, "conv3_2")
    conv3_3 = conv_layer(conv3_2, 256, 256, "conv3_3")
    conv3_4 = conv_layer(conv3_3, 256, 256, "conv3_4")
    pool3 = max_pool(conv3_4, 'pool3')

    conv4_1 = conv_layer(pool3, 256, 512, "conv4_1")
    conv4_2 = conv_layer(conv4_1, 512, 512, "conv4_2")
    conv4_3 = conv_layer(conv4_2, 512, 512, "conv4_3")
    conv4_4 = conv_layer(conv4_3, 512, 512, "conv4_4")
    pool4 = max_pool(conv4_4, 'pool4')

    conv5_1 = conv_layer(pool4, 512, 512, "conv5_1")
    conv5_2 = conv_layer(conv5_1, 512, 512, "conv5_2")
    conv5_3 = conv_layer(conv5_2, 512, 512, "conv5_3")
    conv5_4 = conv_layer(conv5_3, 512, 512, "conv5_4")
    pool5 = max_pool(conv5_4, 'pool5')

    fc6 = fc_layer(pool5, 131072, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
    relu6 = tf.nn.relu(fc6)

    relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(relu6, dropout), lambda: relu6)

    fc7 = fc_layer(relu6, 4096, 4096, "fc7")
    relu7 = tf.nn.relu(fc7)

    relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(relu7, dropout), lambda: relu7)

    fc8 = fc_layer(relu7, 4096, 2, "fc8")

    prob = tf.nn.softmax(fc8, name="prob")

    return prob

def avg_pool(self, bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(self, bottom, in_channels, out_channels, name):
    with tf.variable_scope(name):
        filt, conv_biases = get_conv_var(3, in_channels, out_channels, name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)

        return relu

def fc_layer(self, bottom, in_size, out_size, name):
    with tf.variable_scope(name):
        weights, biases = get_fc_var(in_size, out_size, name)

        x = tf.reshape(bottom, [-1, in_size])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc

def get_conv_var(self, filter_size, in_channels, out_channels, name):
    initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
    filters = get_var(initial_value, name, 0, name + "_filters")

    initial_value = tf.truncated_normal([out_channels], .0, .001)
    biases = get_var(initial_value, name, 1, name + "_biases")

    return filters, biases

def get_fc_var(self, in_size, out_size, name):
    initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
    weights = get_var(initial_value, name, 0, name + "_weights")

    initial_value = tf.truncated_normal([out_size], .0, .001)
    biases = get_var(initial_value, name, 1, name + "_biases")

    return weights, biases

def get_var(self, initial_value, name, idx, var_name):
    if data_dict is not None and name in data_dict:
        value = data_dict[name][idx]
    else:
        value = initial_value

    if trainable:
        var = tf.Variable(value, name=var_name)
    else:
        var = tf.constant(value, dtype=tf.float32, name=var_name)

    var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
    assert var.get_shape() == initial_value.get_shape()

    return var
