#! /usr/bin/env python
# coding=utf-8


import tensorflow as tf
import tensorflow.contrib.layers as ly
from util_filters import *


def extract_parameters(net, cfg, trainable):
    output_dim = cfg.num_filter_parameters
    # net = net - 0.5
    min_feature_map_size = 4
    print('extract_parameters CNN:')
    channels = cfg.base_channels
    print('    ', str(net.get_shape()))
    net = convolutional(net, filters_shape=(3, 3, 3, channels), trainable=trainable, name='ex_conv0',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, channels, 2*channels), trainable=trainable, name='ex_conv1',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv2',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv3',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv4',
                        downsample=True, activate=True, bn=False)
    net = tf.reshape(net, [-1, 4096])
    features = ly.fully_connected(
        net,
        cfg.fc1_size,
        scope='fc1',
        activation_fn=lrelu,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    filter_features = ly.fully_connected(
        features,
        output_dim,
        scope='fc2',
        activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    return filter_features

# wxl：这里提取的参数，就是各个滤波器需要的参数

def extract_parameters_2(net, cfg, trainable):

    # wxl：输出的参数的维度，是滤波器参数的数量。
    output_dim = cfg.num_filter_parameters
    # net = net - 0.5
    min_feature_map_size = 4

    print('extract_parameters_2 CNN:')
    channels = 16

    # wxl：输出网络的形状。
    print('    ', str(net.get_shape()))


    net = convolutional(net, filters_shape=(3, 3, 3, channels), trainable=trainable, name='ex_conv0',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, channels, 2*channels), trainable=trainable, name='ex_conv1',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv2',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv3',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2*channels, 2*channels), trainable=trainable, name='ex_conv4',
                        downsample=True, activate=True, bn=False)
    net = tf.reshape(net, [-1, 2048])
    features = ly.fully_connected(
        net,
        64,
        scope='fc1',
        activation_fn=lrelu,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    filter_features = ly.fully_connected(
        features,
        output_dim,
        scope='fc2',
        activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    return filter_features

# wxl：自定义卷积结构，模块化卷积几乎是网络构架中最基本的操作。
# wxl：
def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    # wxl：为当前卷积层命名，此方法在tf中常用，在torch中被淘汰。
    with tf.variable_scope(name):
        
        # wxl：降采样，默认是false。
        if downsample:
            
            # wxl：可以看出来滤波器是3x3的卷积核，按照卷积核的尺度进行拓延。
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            
            # wxl：降采样的时候，在batch & chanel尺度上不进行降采样，因为步长是1
            # wxl：在width & height尺度上进行降采样，步长取2
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:

            # wxl：不降采样的情况下，步长都是1
            strides = (1, 1, 1, 1)
            padding = "SAME"

        # wxl：随机初始化权重参数
        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        
        # wxl：初始化卷积操作
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            # wxl：卷积迭代进行归一化
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            # wxl：偏移值
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            # wxl：卷积迭代偏移
            conv = tf.nn.bias_add(conv, bias)

        # wxl：卷积迭代进行激活函数
        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output



