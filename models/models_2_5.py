from tensorflow.python.ops import nn_ops, gen_nn_ops
import tensorflow as tf

class MNIST_CNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, training, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            with tf.variable_scope('layer0'):
                X_img = tf.reshape(X, [-1, 28, 28, 1])

            # Convolutional Layer #1 and Pooling Layer #1
            with tf.variable_scope('layer1'):
                conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[4, 4], strides=[2, 2], padding="SAME", activation=None, use_bias=True)
                relu1 = tf.nn.relu(conv1)

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.variable_scope('layer2'):
                conv2 = tf.layers.conv2d(inputs=relu1, filters=64, kernel_size=[4, 4], strides=[2, 2], padding="SAME", activation=None, use_bias=True)
                relu2 = tf.nn.relu(conv2)
                drop2 = tf.layers.dropout(inputs=relu2, rate=0.25, training=training)
                flat = tf.reshape(drop2, [-1, 7 * 7 * 64])

            # Convolutional Layer #3 and Pooling Layer #3
            with tf.variable_scope('layer3'):
                dense3 = tf.layers.dense(inputs=flat, units=128, activation=None, use_bias=True)
                relu3 = tf.nn.relu(dense3)
                drop3 = tf.layers.dropout(inputs=relu3, rate=0.5, training=training)

            # Dense Layer with Relu
            with tf.variable_scope('layer4'):
                logits = tf.layers.dense(inputs=drop3, units=10, activation=None, use_bias=True)

        return [X_img, conv1, relu1, conv2, relu2, dense3, relu3, logits]

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
