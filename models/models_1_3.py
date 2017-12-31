import tensorflow.contrib.layers as tcl
import tensorflow as tf


class MNIST_DNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, training, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            dense1 = tf.layers.dense(inputs=X, units=512, activation=tf.nn.relu)
            dense2 = tf.layers.dense(inputs=dense1, units=100, activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=dense2, units=10)

        return dense2, logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class MNIST_G(object):

    def __init__(self, z_dim, name):
        self.z_dim = z_dim
        self.name = name

    def __call__(self, z, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            G_W1 = tf.get_variable('G_W1', [self.z_dim, 128], initializer=tcl.xavier_initializer())
            G_b1 = tf.get_variable('G_b1', [128], initializer=tf.constant_initializer())
            G_W2 = tf.get_variable('G_W2', [128, 784], initializer=tcl.xavier_initializer())
            G_b2 = tf.get_variable('G_b2', [784], initializer=tf.constant_initializer())

            layer1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            layer2 = tf.nn.sigmoid(tf.matmul(layer1, G_W2) + G_b2)

        return layer2

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class MNIST_D(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [784, 128], initializer=tcl.xavier_initializer())
            D_b1 = tf.get_variable('D_b1', [128], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [128, 1], initializer=tcl.xavier_initializer())
            D_b2 = tf.get_variable('D_b2', [1], initializer=tf.constant_initializer())

            layer1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
            layer2 = tf.matmul(layer1, D_W2) + D_b2
            prediction = tf.nn.sigmoid(layer2)

        return prediction

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
