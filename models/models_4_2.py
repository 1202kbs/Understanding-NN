from tensorflow.python.ops import nn_ops, gen_nn_ops
import tensorflow as tf


class MNIST_DNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            dense1 = tf.layers.dense(inputs=X, units=512, activation=tf.nn.relu, use_bias=False)
            dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu, use_bias=False)
            dense3 = tf.layers.dense(inputs=dense2, units=512, activation=tf.nn.relu, use_bias=False)
            dense4 = tf.layers.dense(inputs=dense3, units=512, activation=tf.nn.relu, use_bias=False)
            logits = tf.layers.dense(inputs=dense4, units=10, activation=None, use_bias=False)
            prediction = tf.nn.softmax(logits)

        return [dense1, dense2, dense3, dense4, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


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
                conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, use_bias=False)
                pool1 = tf.layers.average_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.variable_scope('layer2'):
                conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, use_bias=False)
                pool2 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)

            # Convolutional Layer #3 and Pooling Layer #3
            with tf.variable_scope('layer3'):
                conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, use_bias=False)
                pool3 = tf.layers.average_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)

            # Dense Layer with Relu
            with tf.variable_scope('layer4'):
                flat = tf.reshape(pool3, [-1, 128 * 4 * 4])
                dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu, use_bias=False)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            with tf.variable_scope('layer5'):
                logits = tf.layers.dense(inputs=dense4, units=10, activation=None, use_bias=False)
                prediction = tf.nn.relu(logits)

        return [X_img, conv1, pool1, conv2, pool2, conv3, pool3, flat, dense4, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Taylor:

    def __init__(self, activations, weights, conv_ksize, pool_ksize, conv_strides, pool_strides, name):

        self.last_ind = len(activations)
        for op in activations:
            self.last_ind -= 1
            if any([word in op.name for word in ['conv', 'pooling', 'dense']]):
                break

        self.activations = activations
        self.activations.reverse()

        self.weights = weights
        self.weights.reverse()

        self.conv_ksize = conv_ksize
        self.pool_ksize = pool_ksize
        self.conv_strides = conv_strides
        self.pool_strides = pool_strides
        self.name = name

    def __call__(self, logit):

        with tf.name_scope(self.name):
            Rs = []
            j = 0

            for i in range(len(self.activations) - 1):

                if i is self.last_ind:

                    if 'conv' in self.activations[i].name.lower():
                        Rs.append(self.backprop_conv_input(self.activations[i + 1], self.weights[j], Rs[-1], self.conv_strides))
                    else:
                        Rs.append(self.backprop_dense_input(self.activations[i + 1], self.weights[j], Rs[-1]))

                    continue

                if i is 0:
                    Rs.append(self.activations[i][:,logit,None])
                    Rs.append(self.backprop_dense(self.activations[i + 1], self.weights[j][:,logit,None], Rs[-1]))
                    j += 1

                    continue

                if 'dense' in self.activations[i].name.lower():
                    Rs.append(self.backprop_dense(self.activations[i + 1], self.weights[j], Rs[-1]))
                    j += 1
                elif 'reshape' in self.activations[i].name.lower():
                    shape = self.activations[i + 1].get_shape().as_list()
                    shape[0] = -1
                    Rs.append(tf.reshape(Rs[-1], shape))
                elif 'conv' in self.activations[i].name.lower():
                    Rs.append(self.backprop_conv(self.activations[i + 1], self.weights[j], Rs[-1], self.conv_strides))
                    j += 1
                elif 'pooling' in self.activations[i].name.lower():

                    # Apply average pooling backprop regardless of type of pooling layer used, following recommendations by Montavon et al.
                    # Uncomment code below if you want to apply the winner-take-all redistribution policy suggested by Bach et al.
                    #
                    # if 'max' in self.activations[i].name.lower():
                    #     pooling_type = 'max'
                    # else:
                    #     pooling_type = 'avg'
                    # Rs.append(self.backprop_pool(self.activations[i + 1], Rs[-1], self.pool_ksize, self.pool_strides, pooling_type))

                    Rs.append(self.backprop_pool(self.activations[i + 1], Rs[-1], self.pool_ksize, self.pool_strides, 'avg'))
                else:
                    raise Exception('Unknown operation.')

            return Rs[-1]

    def backprop_conv(self, activation, kernel, relevance, strides, padding='SAME'):
        W_p = tf.maximum(0., kernel)
        z = nn_ops.conv2d(activation, W_p, strides, padding) + 1e-10
        s = relevance / z
        c = nn_ops.conv2d_backprop_input(tf.shape(activation), W_p, s, strides, padding)
        return activation * c

    def backprop_pool(self, activation, relevance, ksize, strides, pooling_type, padding='SAME'):

        if pooling_type.lower() in 'avg':
            z = nn_ops.avg_pool(activation, ksize, strides, padding) + 1e-10
            s = relevance / z
            c = gen_nn_ops._avg_pool_grad(tf.shape(activation), s, ksize, strides, padding)
            return activation * c
        else:
            z = nn_ops.max_pool(activation, ksize, strides, padding) + 1e-10
            s = relevance / z
            c = gen_nn_ops._max_pool_grad(activation, z, s, ksize, strides, padding)
            return activation * c

    def backprop_dense(self, activation, kernel, relevance):
        W_p = tf.maximum(0., kernel)
        z = tf.matmul(activation, W_p) + 1e-10
        s = relevance / z
        c = tf.matmul(s, tf.transpose(W_p))
        return activation * c

    def backprop_conv_input(self, X, kernel, relevance, strides, padding='SAME', lowest=0., highest=1.):
        W_p = tf.maximum(0., kernel)
        W_n = tf.minimum(0., kernel)

        L = tf.ones_like(X, tf.float32) * lowest
        H = tf.ones_like(X, tf.float32) * highest

        z_o = nn_ops.conv2d(X, kernel, strides, padding)
        z_p = nn_ops.conv2d(L, W_p, strides, padding)
        z_n = nn_ops.conv2d(H, W_n, strides, padding)

        z = z_o - z_p - z_n + 1e-10
        s = relevance / z

        c_o = nn_ops.conv2d_backprop_input(tf.shape(X), kernel, s, strides, padding)
        c_p = nn_ops.conv2d_backprop_input(tf.shape(X), W_p, s, strides, padding)
        c_n = nn_ops.conv2d_backprop_input(tf.shape(X), W_n, s, strides, padding)

        return X * c_o - L * c_p - H * c_n

    def backprop_dense_input(self, X, kernel, relevance, lowest=0., highest=1.):
        W_p = tf.maximum(0., kernel)
        W_n = tf.minimum(0., kernel)

        L = tf.ones_like(X, tf.float32) * lowest
        H = tf.ones_like(X, tf.float32) * highest

        z_o = tf.matmul(X, kernel)
        z_p = tf.matmul(L, W_p)
        z_n = tf.matmul(H, W_n)

        z = z_o - z_p - z_n + 1e-10
        s = relevance / z

        c_o = tf.matmul(s, tf.transpose(kernel))
        c_p = tf.matmul(s, tf.transpose(W_p))
        c_n = tf.matmul(s, tf.transpose(W_n))

        return X * c_o - L * c_p - H * c_n
