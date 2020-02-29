import config 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def weight_variable(shape):
    init= tf.truncated_normal_initializer( stddev= 0.01)
    return tf.get_variable('W', dtype= tf.float32, shape= shape, initializer=init)

def bias_variable(shape):
    init= tf.constant(0, shape= shape, dtype= tf.float32)
    return tf.get_variable('b', initializer= init)

def conv_layer(x, filter_size, num_filters, stride, name):

    with tf.variable_scope(name):
        input_channels= x.get_shape().as_list()[-1]
        shape= [filter_size, filter_size, input_channels, num_filters]

        W= weight_variable(shape= shape)
        tf.summary.histogram('weight_summary', W)
        b= bias_variable(shape= [num_filters])
        tf.summary.histogram('bias_summary', b)

        layer= tf.nn.conv2d(x, W, strides= [1, stride, stride, 1], padding= 'SAME')
        layer+= b

        return tf.nn.relu(layer)

def max_pooling(x, k_size, stride, name):
    return tf.nn.max_pool(x, [1, k_size, k_size, 1], strides= [1, stride, stride, 1], padding= 'SAME', name= name)


def flatten_layer(layer):

    with tf.variable_scope('flatten_layer'):
        layer_shape= layer.get_shape()
        num_features= layer_shape[1:4].num_elements()
        layer_flat= tf.reshape(layer, [-1, num_features])
    return layer_flat

def fc_layer(x, hidden_dim, name, use_relu= True):

    with tf.variable_scope( name):
        input_dim= x.get_shape()[1]
        W= weight_variable(shape= [input_dim, hidden_dim])
        tf.summary.histogram('fc_weight_summary', W)
        b= bias_variable(shape= [hidden_dim])
        tf.summary.histogram('fc_bias_summary', b)
        layer= tf.matmul(x, W) + b
        if use_relu:
            layer= tf.nn.relu(layer)
        
        return layer

def model(X):
    conv1= conv_layer(X, config.filter_size1, config.num_filters1, config.stride1, name= 'conv1')
    pool1= max_pooling(conv1, k_size= 2, stride= 2, name= 'pool1')
    conv2= conv_layer(pool1, config.filter_size2, config.num_filters2, config.stride2, name= 'conv2')
    pool2= max_pooling(conv2, k_size= 2, stride= 2, name= 'pool2')
    flat_layer= flatten_layer(pool2)
    fc1= fc_layer(flat_layer, config.h1, name= 'fc1', use_relu= True)
    output_logits= fc_layer(fc1, config.num_classes, name= 'OUTPUT', use_relu= False)

    return output_logits

        