from tensorflow.python.ops import nn_ops, gen_nn_ops
import tensorflow as tf


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

            dense1 = tf.layers.dense(inputs=X, units=512, activation=tf.nn.relu, use_bias=True)
            dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu, use_bias=True)
            dense3 = tf.layers.dense(inputs=dense2, units=512, activation=tf.nn.relu, use_bias=True)
            dense4 = tf.layers.dense(inputs=dense3, units=512, activation=tf.nn.relu, use_bias=True)
            logits = tf.layers.dense(inputs=dense4, units=10, activation=None, use_bias=True)
            prediction = tf.nn.softmax(logits)

        return [dense1, dense2, dense3, dense4, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class LRP:

    def __init__(self, alpha, activations, weights, biases, conv_ksize, pool_ksize, conv_strides, pool_strides, name):
        self.alpha = alpha
        self.activations = activations
        self.weights = weights
        self.biases = biases
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

                if i is 0:
                    Rs.append(self.activations[i][:,logit,None])
                    Rs.append(self.backprop_dense(self.activations[i + 1], self.weights[j][:,logit,None], self.biases[j][logit,None], Rs[-1]))
                    j += 1

                    continue

                elif 'dense' in self.activations[i].name.lower():
                    Rs.append(self.backprop_dense(self.activations[i + 1], self.weights[j], self.biases[j], Rs[-1]))
                    j += 1
                elif 'reshape' in self.activations[i].name.lower():
                    shape = self.activations[i + 1].get_shape().as_list()
                    shape[0] = -1
                    Rs.append(tf.reshape(Rs[-1], shape))
                elif 'conv' in self.activations[i].name.lower():
                    Rs.append(self.backprop_conv(self.activations[i + 1], self.weights[j], self.biases[j], Rs[-1], self.conv_strides))
                    j += 1
                elif 'pooling' in self.activations[i].name.lower():
                    if 'max' in self.activations[i].name.lower():
                        pooling_type = 'max'
                    else:
                        pooling_type = 'avg'
                    Rs.append(self.backprop_pool(self.activations[i + 1], Rs[-1], self.pool_ksize, self.pool_strides, pooling_type))
                else:
                    raise Error('Unknown operation.')

            return Rs[-1]

    def backprop_conv(self, activation, kernel, bias, relevance, strides, padding='SAME'):
        W_p = tf.maximum(0., kernel)
        b_p = tf.maximum(0., bias)
        z_p = nn_ops.conv2d(activation, W_p, strides, padding) + b_p
        s_p = relevance / z_p
        c_p = nn_ops.conv2d_backprop_input(tf.shape(activation), W_p, s_p, strides, padding)

        W_n = tf.minimum(0., kernel)
        b_n = tf.minimum(0., bias)
        z_n = nn_ops.conv2d(activation, W_n, strides, padding) + b_n
        s_n = relevance / z_n
        c_n = nn_ops.conv2d_backprop_input(tf.shape(activation), W_n, s_n, strides, padding)

        return activation * (self.alpha * c_p + (1 - self.alpha) * c_n)

    def backprop_pool(self, activation, relevance, ksize, strides, pooling_type, padding='SAME'):

        if pooling_type.lower() is 'avg':
            z = nn_ops.avg_pool(activation, ksize, strides, padding) + 1e-10
            s = relevance / z
            c = gen_nn_ops._avg_pool_grad(tf.shape(activation), s, ksize, strides, padding)
            return activation * c
        else:
            z = nn_ops.max_pool(activation, ksize, strides, padding) + 1e-10
            s = relevance / z
            c = gen_nn_ops._max_pool_grad(activation, z, s, ksize, strides, padding)
            return activation * c

    def backprop_dense(self, activation, kernel, bias, relevance):
        W_p = tf.maximum(0., kernel)
        b_p = tf.maximum(0., bias)
        z_p = tf.matmul(activation, W_p) + b_p
        s_p = relevance / z_p
        c_p = tf.matmul(s_p, tf.transpose(W_p))

        W_n = tf.minimum(0., kernel)
        b_n = tf.minimum(0., bias)
        z_n = tf.matmul(activation, W_n) + b_n
        s_n = relevance / z_n
        c_n = tf.matmul(s_n, tf.transpose(W_n))

        return activation * (self.alpha * c_p + (1 - self.alpha) * c_n)
