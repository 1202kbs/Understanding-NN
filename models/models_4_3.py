from tensorflow.python.ops import nn_ops, gen_nn_ops
import tensorflow as tf


class MNIST_CNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            with tf.variable_scope('layer0'):
                X_img = tf.reshape(X, [-1, 40, 40, 1])

            # Convolutional Layer #1 and Pooling Layer #1
            with tf.variable_scope('layer1'):
                conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, use_bias=False)
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.variable_scope('layer2'):
                conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, use_bias=False)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)

            # Convolutional Layer #3 and Pooling Layer #3
            with tf.variable_scope('layer3'):
                conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, use_bias=False)

            # Dense Layer with Relu
            with tf.variable_scope('layer4'):
                flat = tf.reshape(conv3, [-1, 128 * 10 * 10])
                dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu, use_bias=False)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            with tf.variable_scope('layer5'):
                logits = tf.layers.dense(inputs=dense4, units=10, activation=None, use_bias=False)
                prediction = tf.nn.relu(logits)

        return [X_img, conv1, pool1, conv2, pool2, conv3, flat, dense4, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
