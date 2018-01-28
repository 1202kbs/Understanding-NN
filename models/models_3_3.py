import tensorflow as tf


class MNIST_CNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            with tf.variable_scope('layer0'):
                X_img = tf.reshape(X, [-1, 28, 28, 1])

            # Convolutional Layer #1 and Pooling Layer #1
            with tf.variable_scope('layer1'):
                conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                pool1 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[2, 2], strides=2, padding="SAME", activation=tf.nn.relu, use_bias=True)

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.variable_scope('layer2'):
                conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                pool2 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[2, 2], strides=2, padding="SAME", activation=tf.nn.relu, use_bias=True)

            # Convolutional Layer #3 and Pooling Layer #3
            with tf.variable_scope('layer3'):
                conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)

            # Logits (no activation) Layer
            with tf.variable_scope('layer4'):
                dense = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[1, 1], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                pool4 = tf.layers.conv2d(inputs=dense, filters=10, kernel_size=[1, 1], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                global_avg = tf.layers.average_pooling2d(inputs=pool4, pool_size=[7, 7], strides=1, padding="VALID")
                logits = tf.reshape(global_avg, [-1, 10])
                prediction = tf.nn.softmax(logits)

        return [X_img, conv1, pool1, conv2, pool2, conv3, dense, pool4, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
