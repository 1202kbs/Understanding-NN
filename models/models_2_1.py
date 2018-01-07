import tensorflow as tf


class MNIST_DNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, training, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            with tf.variable_scope('layer1'):
                dense1 = tf.layers.dense(inputs=X, units=512, activation=tf.nn.relu)
                dropout1 = tf.layers.dropout(inputs=dense1, rate=0.7, training=training)

            with tf.variable_scope('layer2'):
                dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)
                dropout2 = tf.layers.dropout(inputs=dense2, rate=0.7, training=training)

            with tf.variable_scope('layer3'):
                dense3 = tf.layers.dense(inputs=dropout2, units=512, activation=tf.nn.relu)
                dropout3 = tf.layers.dropout(inputs=dense3, rate=0.7, training=training)

            with tf.variable_scope('layer4'):
                dense4 = tf.layers.dense(inputs=dropout3, units=512, activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(inputs=dense4, rate=0.7, training=training)

            with tf.variable_scope('layer5'):
                logits = tf.layers.dense(inputs=dropout4, units=10)

        return logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class MNIST_CNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, training, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            X_img = tf.reshape(X, [-1, 28, 28, 1])

            # Convolutional Layer #1 and Pooling Layer #1
            with tf.variable_scope('layer1'):
                conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
                dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=training)

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.variable_scope('layer2'):
                conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
                dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=training)

            # Convolutional Layer #3 and Pooling Layer #3
            with tf.variable_scope('layer3'):
                conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)
                dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=training)

            # Dense Layer with Relu
            with tf.variable_scope('layer4'):
                flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
                dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            with tf.variable_scope('layer5'):
                logits = tf.layers.dense(inputs=dropout4, units=10)

        return logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
