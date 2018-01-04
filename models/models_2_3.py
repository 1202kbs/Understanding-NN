import tensorflow as tf
import numpy as np


class MNIST_NN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            dense1 = tf.layers.dense(inputs=X, units=512, activation=tf.nn.relu, use_bias=True, name='layer1')
            dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu, use_bias=True, name='layer2')
            logits = tf.layers.dense(inputs=dense2, units=10, activation=None, use_bias=True, name='layer3')
            prediction = tf.nn.softmax(logits)


        return [dense1, dense2, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class MNIST_DNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            dense1 = tf.layers.dense(inputs=X, units=512, activation=tf.nn.relu, use_bias=True, name='layer1')
            dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu, use_bias=True, name='layer2')
            dense3 = tf.layers.dense(inputs=dense2, units=512, activation=tf.nn.relu, use_bias=True, name='layer3')
            dense4 = tf.layers.dense(inputs=dense3, units=512, activation=tf.nn.relu, use_bias=True, name='layer4')
            logits = tf.layers.dense(inputs=dense4, units=10, activation=None, use_bias=True, name='layer5')
            prediction = tf.nn.softmax(logits)

        return [dense1, dense2, dense3, dense4, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


def LRP(activations, weights, biases, logit, alpha):
    """
    Calculates Relevance scores through Layer-wise Relevance Propagation.

    (Note: this function works only for fully-connected neural networks)

    :param activations: List of activations, including the input image, in the order of lowest to highest layer. (e.g. [Image, X1, X2, ...])
    :param weights: List of DNN weights/kernels in the order of lowest to highest layer. (e.g. [W1, W2, W3, ...])
    :param biases: List of DNN biases in the order of lowest to highest layer. (e.g. [b1, b2, b3, ...])
    :param logit: The index of logit to calculate Relevance scores for.
    :param alpha: Hyperparameter that controls the amount of positive emphasis. Beta is automatically calculated by 1 - alpha.
    :returns: Relevance scores for the image with respect to the indicated logit.

    """

    def pos_neg(x):
        return np.maximum(x, 0), np.minimum(x, 0)

    idx = len(activations) - 2
    Rs = []

    for i in range(idx, -1, -1):

        if i is idx:
            z_t = np.transpose(activations[i]) * weights[i][:,logit,None]
            b_p, b_n = pos_neg(biases[i][logit,None])
            Rs.append(activations[-1][:,logit])
        else:
            z_t = np.transpose(activations[i]) * weights[i]
            b_p, b_n = pos_neg(biases[i])

        z_t_p, z_t_n = pos_neg(z_t)

        z_p = np.sum(z_t_p, axis=0) + b_p
        z_n = np.sum(z_t_n, axis=0) + b_n

        R_t = Rs[-1] * (alpha * z_t_p / z_p + (1 - alpha) * z_t_n / z_n)
        R = np.sum(R_t, axis=1)
        Rs.append(R)

    return Rs[-1]
