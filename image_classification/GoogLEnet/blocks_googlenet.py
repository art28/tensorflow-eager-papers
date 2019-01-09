import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np


class InceptionBlock(tf.keras.Model):

    def __init__(self,
                 conv11,
                 reduce_conv33, conv33,
                 reduce_conv55, conv55,
                 convpool,
                 ):
        super(InceptionBlock, self).__init__(name='')

        self.conv11 = tf.layers.Conv2D(filters=conv11, kernel_size=(1,1), strides=(1,1), padding="same", activation=tf.nn.relu)

        self.reduce_conv33 = tf.layers.Conv2D(reduce_conv33, (1,1), (1,1), padding="same", activation=tf.nn.relu)
        self.conv33 = tf.layers.Conv2D(conv33, (3,3), (1,1), padding="same", activation=tf.nn.relu)

        self.reduce_conv55 = tf.layers.Conv2D(reduce_conv55, (1,1), (1,1), padding="same", activation=tf.nn.relu)
        self.conv55 = tf.layers.Conv2D(conv55, (5,5), (1,1), padding="same", activation=tf.nn.relu)

        self.maxpool = tf.layers.MaxPooling2D((3,3), (1,1), padding="same")
        self.convpool = tf.layers.Conv2D(convpool, (1,1), (1,1), padding="same" , activation=tf.nn.relu)

    def __call__(self, X, *args, **kwargs):
        x11 = self.conv11(X)

        x33 = self.reduce_conv33(X)
        x33 = self.conv33(x33)

        x55 = self.reduce_conv55(X)
        x55 = self.conv55(x55)

        xcp = self.maxpool(X)
        xcp = self.convpool(xcp)

        return tf.concat([x11, x33, x55, xcp], axis=3)
