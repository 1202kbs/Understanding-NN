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

        return logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
