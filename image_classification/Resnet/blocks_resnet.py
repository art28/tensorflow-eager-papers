import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np


class IdentitiyBlock_3(tf.keras.Model):
    """ 3-depth Identity block which input and output both have same sizes.
    Args:
        filters : list of 3 integers, last size(filters[2]) should be same as input's kernel size
        kernel_sizes : list of 3 tuples, which are kernel size of convolution filters
    """
    def __init__(self,
                 filters,
                 kernel_sizes,
                 ):
        super(IdentitiyBlock_3, self).__init__()

        self.conv1 = tf.layers.Conv2D(filters[0], kernel_sizes[0], padding="same", activation=tf.nn.relu)
        self.bn1 = tf.layers.BatchNormalization()

        self.conv2 = tf.layers.Conv2D(filters[1], kernel_sizes[1], padding="same", activation=tf.nn.relu)
        self.bn2 = tf.layers.BatchNormalization()

        self.conv3 = tf.layers.Conv2D(filters[2], kernel_sizes[2], padding="same", activation=tf.nn.relu)
        self.bn3 = tf.layers.BatchNormalization()

    def call(self, X, training=False):
        x = self.conv1(X)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x += X
        return tf.nn.relu(x)


class ConvolutionBlock_3(tf.keras.Model):
    """ 3-depth Convolution block to reduce dimension residually.
    Args:
        filters : list of 3 integers, last size(filters[2]) will be used in shortcut convolution also
        kernel_sizes : list of 4 tuples. last size is for shortcut
    """
    def __init__(self,
                 filters,
                 kernel_sizes,
                 ):
        super(ConvolutionBlock_3, self).__init__()

        self.conv1 = tf.layers.Conv2D(filters[0], kernel_sizes[0], strides=(2, 2), padding="same",
                                      activation=tf.nn.relu)
        self.bn1 = tf.layers.BatchNormalization()

        self.conv2 = tf.layers.Conv2D(filters[1], kernel_sizes[1], padding="same", activation=tf.nn.relu)
        self.bn2 = tf.layers.BatchNormalization()

        self.conv3 = tf.layers.Conv2D(filters[2], kernel_sizes[2], padding="same", activation=tf.nn.relu)
        self.bn3 = tf.layers.BatchNormalization()

        self.shortcut = tf.layers.Conv2D(filters[2], kernel_sizes[3], strides=(2, 2), padding="same",
                                         activation=tf.nn.relu)
        self.shortcut_bn = tf.layers.BatchNormalization()

    def call(self, X, training=False):
        x = self.conv1(X)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        shorcut = self.shortcut(X)
        shorcut = self.shortcut_bn(shorcut, training=training)

        x += shorcut
        return tf.nn.relu(x)
