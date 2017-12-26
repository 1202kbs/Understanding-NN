import tensorflow.contrib.layers as tcl
import tensorflow as tf
import numpy as np

# ========================================================================================================================================================
# ================================================================================ UTILS =================================================================
# ========================================================================================================================================================

# class batch_norm(object):

#     def __init__(self, scope_name="batch_norm"):
#         self.scope_name = scope_name


#     def __call__(self, X, train=True):
#         normed = tcl.batch_norm(X, center=True, scale=True, is_training=train, scope=self.scope_name)

#         return normed

# https://r2rt.com/implementing-batch-normalization-in-tensorflow.html

class batch_norm(object):

    def __init__(self, type, dim, scope_name='batch_norm'):

        if type == 'fc':
            self.axes = [0]
        elif type == 'conv':
            self.axes = [0, 1, 2]
        else:
            raise Exception('Unimplemented type of Batch Normalization')

        self.dim = dim
        self.scope_name = scope_name

    def __call__(self, X, is_training=True, decay=0.999):

        with tf.variable_scope(self.scope_name) as scope:

            epsilon = 1e-3
            scale = tf.get_variable('scale', [self.dim], initializer=tf.constant_initializer(1.))
            beta = tf.get_variable('beta', [self.dim], initializer=tf.constant_initializer(0.))
            pop_mean = tf.get_variable('pop_mean', [self.dim], initializer=tf.constant_initializer(1.), trainable=False)
            pop_var = tf.get_variable('pop_var', [self.dim], initializer=tf.constant_initializer(0.), trainable=False)

            if is_training:
                batch_mean, batch_var = tf.nn.moments(X, self.axes)
                train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(X, batch_mean, batch_var, beta, scale, epsilon)

            else:
                return tf.nn.batch_normalization(X, pop_mean, pop_var, beta, scale, epsilon)


# ========================================================================================================================================================
# ============================================================================= MNIST FC =================================================================
# ========================================================================================================================================================

class MNIST_FC2_G(object):

    def __init__(self, X_dim, z_dim, scope_name):
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.scope_name = scope_name

    def __call__(self, z, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            G_W1 = tf.get_variable('G_W1', [self.z_dim, 128], initializer=tcl.xavier_initializer())
            G_b1 = tf.get_variable('G_b1', [128], initializer=tf.constant_initializer())
            G_W2 = tf.get_variable('G_W2', [128, self.X_dim], initializer=tcl.xavier_initializer())
            G_b2 = tf.get_variable('G_b2', [self.X_dim], initializer=tf.constant_initializer())

            layer1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            layer2 = tf.nn.sigmoid(tf.matmul(layer1, G_W2) + G_b2)

        return layer2

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)


class MNIST_FC2_CLS(object):

    def __init__(self, X_dim, n_cls, scope_name):
        self.X_dim = X_dim
        self.n_cls = n_cls
        self.scope_name = scope_name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.X_dim, 128], initializer=tcl.xavier_initializer())
            D_b1 = tf.get_variable('D_b1', [128], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [128, self.n_cls], initializer=tcl.xavier_initializer())
            D_b2 = tf.get_variable('D_b2', [self.n_cls], initializer=tf.constant_initializer())

            layer1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
            layer2 = tf.matmul(layer1, D_W2) + D_b2

            if self.n_cls == 2:
                prediction = tf.nn.sigmoid(layer2)
            else:
                prediction = tf.nn.softmax(layer2)

        return prediction

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)


class MNIST_FC2_D_INFO(object):

    def __init__(self, X_dim, c_dim, n_cls, scope_name):
        self.X_dim = X_dim
        self.c_dim = c_dim
        self.n_cls = n_cls
        self.scope_name = scope_name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.X_dim, 128], initializer=tcl.xavier_initializer())
            D_b1 = tf.get_variable('D_b1', [128], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [128, 1], initializer=tcl.xavier_initializer())
            D_b2 = tf.get_variable('D_b2', [1], initializer=tf.constant_initializer())

        with tf.variable_scope(self.scope_name + '_info') as scope:

            if reuse:
                 scope.reuse_variables()

            Q_W2 = tf.get_variable('Q_W2', [128, self.c_dim], initializer=tcl.xavier_initializer())
            Q_b2 = tf.get_variable('Q_b2', [self.c_dim], initializer=tf.constant_initializer())

        layer1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
        layer2 = tf.matmul(layer1, D_W2) + D_b2
        info = tf.nn.tanh(tf.matmul(layer1, Q_W2) + Q_b2)

        if self.n_cls == 2:
            prediction = tf.nn.sigmoid(layer2)
        else:
            prediction = tf.nn.softmax(layer2)

        return prediction, info

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name), \
               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name + '_info')


class MNIST_FC2_CLS_ENC(object):

    def __init__(self, X_dim, c_dim, n_cls, scope_name):
        self.X_dim = X_dim
        self.c_dim = c_dim
        self.n_cls = n_cls
        self.scope_name = scope_name

    def __call__(self, X, eps, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.X_dim, 128], initializer=tcl.xavier_initializer())
            D_b1 = tf.get_variable('D_b1', [128], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [128, self.n_cls], initializer=tcl.xavier_initializer())
            D_b2 = tf.get_variable('D_b2', [self.n_cls], initializer=tf.constant_initializer())

        with tf.variable_scope(self.scope_name + '_enc') as scope:

            if reuse:
                scope.reuse_variables()

            Q1_W2 = tf.get_variable('Q1_W2', [128, self.c_dim], initializer=tcl.xavier_initializer())
            Q1_b2 = tf.get_variable('Q1_b2', [self.c_dim], initializer=tf.constant_initializer())
            Q2_W2 = tf.get_variable('Q2_W2', [128, self.c_dim], initializer=tcl.xavier_initializer())
            Q2_b2 = tf.get_variable('Q2_b2', [self.c_dim], initializer=tf.constant_initializer())

        layer1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
        layer2 = tf.matmul(layer1, D_W2) + D_b2

        if self.n_cls == 2:
            prediction = tf.nn.sigmoid(layer2)
        else:
            prediction = tf.nn.softmax(layer2)

        z_mu = tf.matmul(layer1, Q1_W2) + Q1_b2
        z_log_sigma_sq = tf.matmul(layer1, Q2_W2) + Q2_b2
        z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

        return prediction, z, z_mu, z_log_sigma_sq

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name), \
               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name + '_enc')

# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================

class MNIST_FC3_G(object):

    def __init__(self, X_dim, z_dim, scope_name):
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.scope_name = scope_name

    def __call__(self, z, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            G_W1 = tf.get_variable('G_W1', [self.z_dim, 128], initializer=tcl.xavier_initializer())
            G_b1 = tf.get_variable('G_b1', [128], initializer=tf.constant_initializer())
            G_W2 = tf.get_variable('G_W2', [128, 512], initializer=tcl.xavier_initializer())
            G_b2 = tf.get_variable('G_b2', [512], initializer=tf.constant_initializer())
            G_W3 = tf.get_variable('G_W3', [512, self.X_dim], initializer=tcl.xavier_initializer())
            G_b3 = tf.get_variable('G_b3', [self.X_dim], initializer=tf.constant_initializer())

            layer1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, G_W2) + G_b2)
            layer3 = tf.nn.sigmoid(tf.matmul(layer2, G_W3) + G_b3)

            return layer3

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)


class MNIST_FC3_CLS(object):

    def __init__(self, X_dim, n_cls, scope_name):
        self.X_dim = X_dim
        self.n_cls = n_cls
        self.scope_name = scope_name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.X_dim, 512], initializer=tcl.xavier_initializer())
            D_b1 = tf.get_variable('D_b1', [512], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [512, 128], initializer=tcl.xavier_initializer())
            D_b2 = tf.get_variable('D_b2', [128], initializer=tf.constant_initializer())
            D_W3 = tf.get_variable('D_W3', [128, self.n_cls], initializer=tcl.xavier_initializer())
            D_b3 = tf.get_variable('D_b3', [self.n_cls], initializer=tf.constant_initializer())

            layer1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, D_W2) + D_b2)
            layer3 = tf.matmul(layer2, D_W3) + D_b3

            if self.n_cls == 2:
                prediction = tf.nn.sigmoid(layer3)
            else:
                prediction = tf.nn.softmax(layer3)

        return prediction

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)


class MNIST_FC3_D_INFO(object):

    def __init__(self, X_dim, c_dim, n_cls, scope_name):
        self.X_dim = X_dim
        self.c_dim = c_dim
        self.n_cls = n_cls
        self.scope_name = scope_name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.X_dim, 512], initializer=tcl.xavier_initializer())
            D_b1 = tf.get_variable('D_b1', [512], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [512, 128], initializer=tcl.xavier_initializer())
            D_b2 = tf.get_variable('D_b2', [128], initializer=tf.constant_initializer())
            D_W3 = tf.get_variable('D_W3', [128, 1], initializer=tcl.xavier_initializer())
            D_b3 = tf.get_variable('D_b3', [1], initializer=tf.constant_initializer())

        with tf.variable_scope(self.scope_name + '_info') as scope:

            if reuse:
                 scope.reuse_variables()

            Q_W3 = tf.get_variable('Q_W2', [128, self.c_dim], initializer=tcl.xavier_initializer())
            Q_b3 = tf.get_variable('Q_b2', [self.c_dim], initializer=tf.constant_initializer())

        layer1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, D_W2) + D_b2)
        layer3 = tf.matmul(layer2, D_W3) + D_b3
        info = tf.nn.tanh(tf.matmul(layer2, Q_W3) + Q_b3)

        if self.n_cls == 2:
            prediction = tf.nn.sigmoid(layer3)
        else:
            prediction = tf.nn.softmax(layer3)

        return prediction, info

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name), \
               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name + '_info')


class MNIST_FC3_CLS_ENC(object):

    def __init__(self, X_dim, c_dim, n_cls, scope_name):
        self.X_dim = X_dim
        self.c_dim = c_dim
        self.n_cls = n_cls
        self.scope_name = scope_name

    def __call__(self, X, eps, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.X_dim, 512], initializer=tcl.xavier_initializer())
            D_b1 = tf.get_variable('D_b1', [512], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [512, 128], initializer=tcl.xavier_initializer())
            D_b2 = tf.get_variable('D_b2', [128], initializer=tf.constant_initializer())
            D_W3 = tf.get_variable('D_W3', [128, self.n_cls], initializer=tcl.xavier_initializer())
            D_b3 = tf.get_variable('D_b3', [self.n_cls], initializer=tf.constant_initializer())

        with tf.variable_scope(self.scope_name + '_enc') as scope:

            if reuse:
                scope.reuse_variables()

            Q1_W3 = tf.get_variable('Q1_W3', [128, self.c_dim], initializer=tcl.xavier_initializer())
            Q1_b3 = tf.get_variable('Q1_b3', [self.c_dim], initializer=tf.constant_initializer())
            Q2_W3 = tf.get_variable('Q2_W3', [128, self.c_dim], initializer=tcl.xavier_initializer())
            Q2_b3 = tf.get_variable('Q2_b3', [self.c_dim], initializer=tf.constant_initializer())


        layer1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, D_W2) + D_b2)
        layer3 = tf.matmul(layer2, D_W3) + D_b3

        if self.n_cls == 2:
            prediction = tf.nn.sigmoid(layer3)
        else:
            prediction = tf.nn.softmax(layer3)

        z_mu = tf.matmul(layer2, Q1_W3) + Q1_b3
        z_log_sigma_sq = tf.matmul(layer2, Q2_W3) + Q2_b3
        z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

        return prediction, z, z_mu, z_log_sigma_sq

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name), \
               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name + '_enc')

# ========================================================================================================================================================
# ======================================================================== CIFAR FC ======================================================================
# ========================================================================================================================================================

class CIFAR_FC2_G(object):

    def __init__(self, X_dim, h1_dim, h2_dim, z_dim, scope_name):
        self.X_dim = X_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.z_dim = z_dim
        self.scope_name = scope_name


    def __call__(self, z):

        with tf.variable_scope(self.scope_name) as scope:
            G_W1 = tf.get_variable('G_W1', [self.z_dim, self.h1_dim], initializer=tcl.xavier_initializer())
            G_b1 = tf.get_variable('G_b1', [self.h1_dim], initializer=tf.constant_initializer())
            G_W2 = tf.get_variable('G_W2', [self.h1_dim, self.h2_dim], initializer=tcl.xavier_initializer())
            G_b2 = tf.get_variable('G_b2', [self.h2_dim], initializer=tf.constant_initializer())
            G_W3 = tf.get_variable('G_W3', [self.h2_dim, self.X_dim], initializer=tcl.xavier_initializer())
            G_b3 = tf.get_variable('G_b3', [self.X_dim], initializer=tf.constant_initializer())

            layer1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, G_W2) + G_b2)
            layer3 = tf.nn.sigmoid(tf.matmul(layer2, G_W3) + G_b3)

        return layer3

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)


class CIFAR_FC2_D(object):

    def __init__(self, X_dim, h1_dim, h2_dim, scope_name):
        self.X_dim = X_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.scope_name = scope_name


    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.X_dim, self.h2_dim], initializer=tcl.xavier_initializer())
            D_b1 = tf.get_variable('D_b1', [self.h2_dim], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [self.h2_dim, self.h1_dim], initializer=tcl.xavier_initializer())
            D_b2 = tf.get_variable('D_b2', [self.h1_dim], initializer=tf.constant_initializer())
            D_W3 = tf.get_variable('D_W3', [self.h1_dim, 1], initializer=tcl.xavier_initializer())
            D_b3 = tf.get_variable('D_b3', [1], initializer=tf.constant_initializer())

            layer1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, D_W2) + D_b2)
            layer3 = tf.nn.sigmoid(tf.matmul(layer2, D_W3) + D_b3)

        return layer3


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)

# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================

class CIFAR_FC3_G(object):

    def __init__(self, X_dim, h1_dim, h2_dim, h3_dim, z_dim, scope_name):
        self.X_dim = X_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.h3_dim = h3_dim
        self.z_dim = z_dim
        self.scope_name = scope_name


    def __call__(self, z):

        with tf.variable_scope(self.scope_name) as scope:
            G_W1 = tf.get_variable('G_W1', [self.z_dim, self.h1_dim], initializer=tcl.xavier_initializer())
            G_b1 = tf.get_variable('G_b1', [self.h1_dim], initializer=tf.constant_initializer())
            G_W2 = tf.get_variable('G_W2', [self.h1_dim, self.h2_dim], initializer=tcl.xavier_initializer())
            G_b2 = tf.get_variable('G_b2', [self.h2_dim], initializer=tf.constant_initializer())
            G_W3 = tf.get_variable('G_W3', [self.h2_dim, self.h3_dim], initializer=tcl.xavier_initializer())
            G_b3 = tf.get_variable('G_b3', [self.h3_dim], initializer=tf.constant_initializer())
            G_W4 = tf.get_variable('G_W4', [self.h3_dim, self.X_dim], initializer=tcl.xavier_initializer())
            G_b4 = tf.get_variable('G_b4', [self.X_dim], initializer=tf.constant_initializer())

            layer1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, G_W2) + G_b2)
            layer3 = tf.nn.relu(tf.matmul(layer2, G_W3) + G_b3)
            layer4 = tf.nn.sigmoid(tf.matmul(layer3, G_W4) + G_b4)

        return layer4

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)


class CIFAR_FC3_D(object):

    def __init__(self, X_dim, h1_dim, h2_dim, h3_dim, scope_name):
        self.X_dim = X_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.h3_dim = h3_dim
        self.scope_name = scope_name

    def leaky_relu(self, z, name=None):
        return tf.maximum(0.2 * z, z, name=name)

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.X_dim, self.h3_dim], initializer=tcl.xavier_initializer())
            D_b1 = tf.get_variable('D_b1', [self.h3_dim], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [self.h3_dim, self.h2_dim], initializer=tcl.xavier_initializer())
            D_b2 = tf.get_variable('D_b2', [self.h2_dim], initializer=tf.constant_initializer())
            D_W3 = tf.get_variable('D_W3', [self.h2_dim, self.h1_dim], initializer=tcl.xavier_initializer())
            D_b3 = tf.get_variable('D_b3', [self.h1_dim], initializer=tf.constant_initializer())
            D_W4 = tf.get_variable('D_W4', [self.h1_dim, 1], initializer=tcl.xavier_initializer())
            D_b4 = tf.get_variable('D_b4', [1], initializer=tf.constant_initializer())

            layer1 = self.leaky_relu(tf.matmul(X, D_W1) + D_b1)
            layer2 = self.leaky_relu(tf.matmul(layer1, D_W2) + D_b2)
            layer3 = self.leaky_relu(tf.matmul(layer2, D_W3) + D_b3)
            layer4 = tf.nn.sigmoid(tf.matmul(layer3, D_W4) + D_b4)

        return layer4

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)

# ========================================================================================================================================================
# =========================================================================== CONV =======================================================================
# ========================================================================================================================================================

class CONV2_G(object):

    def __init__(self, X_dim, z_dim, channel, kernel_size, depth, scope_name):
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.channel = channel
        self.kernel_size = kernel_size

        self.dim1 = int(np.sqrt(self.X_dim))
        self.dim2 = int(self.dim1 / 2)
        self.dim3 = int(self.dim1 / 4)

        self.depth1 = depth
        self.depth2 = int(depth / 2)

        self.scope_name = scope_name
        self.g_bn1 = batch_norm(type='fc', dim=self.dim3 * self.dim3 * self.depth1, scope_name='g_bn1')
        self.g_bn2 = batch_norm(type='conv', dim=self.depth2, scope_name='g_bn2')

    def __call__(self, z, is_training=True, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            dim1 = int(np.sqrt(self.X_dim))
            dim2 = int(dim1 / 2)
            dim3 = int(dim1 / 4)

            G_W1 = tf.get_variable('G_W1', [self.z_dim, self.dim3 * self.dim3 * self.depth1], initializer=tcl.xavier_initializer())
            G_b1 = tf.get_variable('G_b1', [self.dim3 * self.dim3 * self.depth1], initializer=tf.constant_initializer())
            G_W2 = tf.get_variable('G_W2', [self.kernel_size, self.kernel_size, self.depth2, self.depth1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            G_W3 = tf.get_variable('G_W3', [self.kernel_size, self.kernel_size, self.channel, self.depth2], initializer=tf.truncated_normal_initializer(stddev=0.02))

            layer1 = tf.nn.relu(self.g_bn1(tf.matmul(z, G_W1) + G_b1, is_training=is_training))
            layer1 = tf.reshape(layer1, [-1, self.dim3, self.dim3, self.depth1])
            layer2 = tf.nn.relu(self.g_bn2(tf.nn.conv2d_transpose(layer1, G_W2, [tf.shape(layer1)[0], self.dim2, self.dim2, self.depth2], [1, 2, 2, 1], 'SAME'), is_training=is_training))
            layer3 = tf.nn.tanh(tf.nn.conv2d_transpose(layer2, G_W3, [tf.shape(layer2)[0], self.dim1, self.dim1, self.channel], [1, 2, 2, 1], 'SAME'))
            layer3 = tf.reshape(layer3, [-1, self.dim1 * self.dim1 * self.channel])

        return layer3

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name), \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)


class CONV2_CLS(object):

    def __init__(self, X_dim, n_cls, channel, kernel_size, depth, scope_name):
        self.X_dim = X_dim
        self.n_cls = n_cls
        self.channel = channel
        self.kernel_size = kernel_size

        self.dim1 = int(np.sqrt(self.X_dim))
        self.dim2 = int(self.dim1 / 2)
        self.dim3 = int(self.dim1 / 4)

        self.depth1 = depth
        self.depth2 = int(depth / 2)

        self.scope_name = scope_name
        self.d_bn1 = batch_norm(type='conv', dim=self.depth1, scope_name='d_bn1')

    def leaky_relu(self, z, name=None):
        return tf.maximum(0.2 * z, z, name=name)

    def __call__(self, X, reuse=False, is_training=True):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.kernel_size, self.kernel_size, self.channel, self.depth2], initializer=tf.truncated_normal_initializer(stddev=0.02))
            D_b1 = tf.get_variable('D_b1', [self.depth2], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [self.kernel_size, self.kernel_size, self.depth2, self.depth1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            D_b2 = tf.get_variable('D_b2', [self.depth1], initializer=tf.constant_initializer())
            D_W3 = tf.get_variable('D_W3', [self.dim3 * self.dim3 * self.depth1, self.n_cls], initializer=tcl.xavier_initializer())
            D_b3 = tf.get_variable('D_b3', [self.n_cls], initializer=tf.constant_initializer())

            X = tf.reshape(X, [-1, self.dim1, self.dim1, self.channel])
            layer1 = self.leaky_relu(tf.nn.conv2d(X, D_W1, [1, 2, 2, 1], 'SAME') + D_b1)
            layer2 = self.leaky_relu(self.d_bn1(tf.nn.conv2d(layer1, D_W2, [1, 2, 2, 1], 'SAME') + D_b2, is_training=is_training))
            layer2 = tf.reshape(layer2, [-1, self.dim3 * self.dim3 * self.depth1])
            layer3 = tf.matmul(layer2, D_W3) + D_b3

            if self.n_cls == 2:
                prediction = tf.nn.sigmoid(layer3)
            else:
                prediction = tf.nn.softmax(layer3)

        return prediction

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name), \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)

# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================

class CONV3_G(object):

    def __init__(self, X_dim, z_dim, channel, kernel_size, depth, scope_name):
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.channel = channel
        self.kernel_size = kernel_size

        self.dim1 = int(np.sqrt(self.X_dim))
        self.dim2 = int(self.dim1 / 2)
        self.dim3 = int(self.dim1 / 4)
        self.dim4 = int(self.dim1 / 8)

        self.depth1 = depth
        self.depth2 = int(depth / 2)
        self.depth3 = int(depth / 4)

        self.scope_name = scope_name

        self.g_bn1 = batch_norm(type='conv', dim=self.depth1, scope_name='g_bn1')
        self.g_bn2 = batch_norm(type='conv', dim=self.depth2, scope_name='g_bn2')
        self.g_bn3 = batch_norm(type='conv', dim=self.depth3, scope_name='g_bn3')

    def __call__(self, z, reuse=False, is_training=True):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            G_W1 = tf.get_variable('G_W1', [self.z_dim, 2 * 2 * 1024], initializer=tcl.xavier_initializer())
            G_b1 = tf.get_variable('G_b1', [2 * 2 * 1024], initializer=tf.constant_initializer())
            G_W2 = tf.get_variable('G_W2', [self.kernel_size, self.kernel_size, self.depth1, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            G_W3 = tf.get_variable('G_W3', [self.kernel_size, self.kernel_size, self.depth2, self.depth1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            G_W4 = tf.get_variable('G_W4', [self.kernel_size, self.kernel_size, self.depth3, self.depth2], initializer=tf.truncated_normal_initializer(stddev=0.02))
            G_W5 = tf.get_variable('G_W5', [self.kernel_size, self.kernel_size, self.channel, self.depth3], initializer=tf.truncated_normal_initializer(stddev=0.02))

            layer1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            layer1 = tf.reshape(layer1, [-1, 2, 2, 1024])

            layer2 = tf.nn.relu(self.g_bn1(tf.nn.conv2d_transpose(layer1, G_W2, [tf.shape(layer1)[0], self.dim4, self.dim4, self.depth1], [1, 2, 2, 1], 'SAME'), is_training=is_training))
            layer3 = tf.nn.relu(self.g_bn2(tf.nn.conv2d_transpose(layer2, G_W3, [tf.shape(layer2)[0], self.dim3, self.dim3, self.depth2], [1, 2, 2, 1], 'SAME'), is_training=is_training))
            layer4 = tf.nn.relu(self.g_bn3(tf.nn.conv2d_transpose(layer3, G_W4, [tf.shape(layer3)[0], self.dim2, self.dim2, self.depth3], [1, 2, 2, 1], 'SAME'), is_training=is_training))
            layer5 = tf.nn.tanh(tf.nn.conv2d_transpose(layer4, G_W5, [tf.shape(layer4)[0], self.dim1, self.dim1, self.channel], [1, 2, 2, 1], 'SAME'))
            layer5 = tf.reshape(layer5, [-1, self.dim1 * self.dim1 * self.channel])

        return layer5

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name), \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)


class CONV3_CLS(object):

    def __init__(self, X_dim, n_cls, channel, kernel_size, depth, scope_name):
        self.X_dim = X_dim
        self.n_cls = n_cls
        self.channel = channel
        self.kernel_size = kernel_size

        self.dim1 = int(np.sqrt(self.X_dim))
        self.dim2 = int(self.dim1 / 2)
        self.dim3 = int(self.dim1 / 4)
        self.dim4 = round(self.dim1 / 8)

        self.depth1 = depth
        self.depth2 = int(depth / 2)
        self.depth3 = int(depth / 4)

        self.d_bn1 = batch_norm(type='conv', dim=self.depth2, scope_name='d_bn1')
        self.d_bn2 = batch_norm(type='conv', dim=self.depth1, scope_name='d_bn2')

        self.scope_name = scope_name

    def leaky_relu(self, z, name=None):
        return tf.maximum(0.2 * z, z, name=name)

    def __call__(self, X, reuse=False, is_training=True):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.kernel_size, self.kernel_size, self.channel, self.depth3], initializer=tf.truncated_normal_initializer(stddev=0.02))
            D_b1 = tf.get_variable('D_b1', [self.depth3], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [self.kernel_size, self.kernel_size, self.depth3, self.depth2], initializer=tf.truncated_normal_initializer(stddev=0.02))
            D_b2 = tf.get_variable('D_b2', [self.depth2], initializer=tf.constant_initializer())
            D_W3 = tf.get_variable('D_W3', [self.kernel_size, self.kernel_size, self.depth2, self.depth1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            D_b3 = tf.get_variable('D_b3', [self.depth1], initializer=tf.constant_initializer())
            D_W4 = tf.get_variable('D_W4', [self.dim4 * self.dim4 * self.depth1, 1], initializer=tcl.xavier_initializer())
            D_b4 = tf.get_variable('D_b4', [1], initializer=tf.constant_initializer())

            X = tf.reshape(X, [-1, self.dim1, self.dim1, self.channel])
            layer1 = self.leaky_relu(tf.nn.conv2d(X, D_W1, [1, 2, 2, 1], 'SAME') + D_b1)
            layer2 = self.leaky_relu(self.d_bn1(tf.nn.conv2d(layer1, D_W2, [1, 2, 2, 1], 'SAME') + D_b2, is_training=is_training))
            layer3 = self.leaky_relu(self.d_bn2(tf.nn.conv2d(layer2, D_W3, [1, 2, 2, 1], 'SAME') + D_b3, is_training=is_training))
            layer3 = tf.reshape(layer3, [-1, self.dim4 * self.dim4 * self.depth1])
            layer4 = tf.matmul(layer3, D_W4) + D_b4

            if self.n_cls == 2:
                prediction = tf.nn.sigmoid(layer4)
            else:
                prediction = tf.nn.softmax(layer4)

        return prediction

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name), \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)

# ========================================================================================================================================================
# ========================================================================================================================================================
# ========================================================================================================================================================

class CONV4_G(object):

    def __init__(self, X_dim, z_dim, channel, kernel_size, depth, scope_name):
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.channel = channel
        self.kernel_size = kernel_size

        self.dim1 = int(np.sqrt(self.X_dim))
        self.dim2 = int(self.dim1 / 2)
        self.dim3 = int(self.dim1 / 4)
        self.dim4 = int(self.dim1 / 8)
        self.dim5 = int(self.dim1 / 16)

        self.depth1 = depth
        self.depth2 = int(depth / 2)
        self.depth3 = int(depth / 4)
        self.depth4 = int(depth / 8)

        self.g_bn1 = batch_norm(type='conv', dim=self.depth1, scope_name='g_bn1')
        self.g_bn2 = batch_norm(type='conv', dim=self.depth2, scope_name='g_bn2')
        self.g_bn3 = batch_norm(type='conv', dim=self.depth3, scope_name='g_bn3')
        self.g_bn4 = batch_norm(type='conv', dim=self.depth4, scope_name='g_bn4')

        self.scope_name = scope_name

    def __call__(self, z, reuse=False, is_training=True):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            G_W1 = tf.get_variable('G_W1', [self.z_dim, 2 * 2 * 1024], initializer=tcl.xavier_initializer())
            G_b1 = tf.get_variable('G_b1', [2 * 2 * 1024], initializer=tf.constant_initializer())
            G_W2 = tf.get_variable('G_W2', [self.kernel_size, self.kernel_size, self.depth1, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            G_W3 = tf.get_variable('G_W3', [self.kernel_size, self.kernel_size, self.depth2, self.depth1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            G_W4 = tf.get_variable('G_W4', [self.kernel_size, self.kernel_size, self.depth3, self.depth2], initializer=tf.truncated_normal_initializer(stddev=0.02))
            G_W5 = tf.get_variable('G_W5', [self.kernel_size, self.kernel_size, self.depth4, self.depth3], initializer=tf.truncated_normal_initializer(stddev=0.02))
            G_W6 = tf.get_variable('G_W6', [self.kernel_size, self.kernel_size, self.channel, self.depth4], initializer=tf.truncated_normal_initializer(stddev=0.02))

            layer1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            layer1 = tf.reshape(layer1, [-1, 2, 2, 1024])

            layer2 = tf.nn.relu(self.g_bn1(tf.nn.conv2d_transpose(layer1, G_W2, [tf.shape(layer1)[0], self.dim5, self.dim5, self.depth1], [1, 2, 2, 1], 'SAME'), is_training=is_training))
            layer3 = tf.nn.relu(self.g_bn2(tf.nn.conv2d_transpose(layer2, G_W3, [tf.shape(layer2)[0], self.dim4, self.dim4, self.depth2], [1, 2, 2, 1], 'SAME'), is_training=is_training))
            layer4 = tf.nn.relu(self.g_bn3(tf.nn.conv2d_transpose(layer3, G_W4, [tf.shape(layer3)[0], self.dim3, self.dim3, self.depth3], [1, 2, 2, 1], 'SAME'), is_training=is_training))
            layer5 = tf.nn.relu(self.g_bn4(tf.nn.conv2d_transpose(layer4, G_W5, [tf.shape(layer4)[0], self.dim2, self.dim2, self.depth4], [1, 2, 2, 1], 'SAME'), is_training=is_training))
            layer6 = tf.nn.tanh(tf.nn.conv2d_transpose(layer5, G_W6, [tf.shape(layer5)[0], self.dim1, self.dim1, self.channel], [1, 2, 2, 1], 'SAME'))
            layer6 = tf.reshape(layer6, [-1, self.dim1 * self.dim1 * self.channel])

        return layer6

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name), \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)


class CONV4_CLS(object):

    def __init__(self, X_dim, n_cls, channel, kernel_size, depth, scope_name):
        self.X_dim = X_dim
        self.n_cls = n_cls
        self.channel = channel
        self.kernel_size = kernel_size

        self.dim1 = int(np.sqrt(self.X_dim))
        self.dim2 = int(self.dim1 / 2)
        self.dim3 = int(self.dim1 / 4)
        self.dim4 = int(self.dim1 / 8)
        self.dim5 = int(self.dim1 / 16)

        self.depth1 = depth
        self.depth2 = int(depth / 2)
        self.depth3 = int(depth / 4)
        self.depth4 = int(depth / 8)

        self.scope_name = scope_name
        self.d_bn1 = batch_norm(type='conv', dim=self.depth3, scope_name='d_bn1')
        self.d_bn2 = batch_norm(type='conv', dim=self.depth2, scope_name='d_bn2')
        self.d_bn3 = batch_norm(type='conv', dim=self.depth1, scope_name='d_bn3')

    def leaky_relu(self, z, name=None):
        return tf.maximum(0.2 * z, z, name=name)

    def __call__(self, X, reuse=False, is_training=True):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.kernel_size, self.kernel_size, self.channel, self.depth4], initializer=tf.truncated_normal_initializer(stddev=0.02))
            D_b1 = tf.get_variable('D_b1', [self.depth4], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [self.kernel_size, self.kernel_size, self.depth4, self.depth3], initializer=tf.truncated_normal_initializer(stddev=0.02))
            D_b2 = tf.get_variable('D_b2', [self.depth3], initializer=tf.constant_initializer())
            D_W3 = tf.get_variable('D_W3', [self.kernel_size, self.kernel_size, self.depth3, self.depth2], initializer=tf.truncated_normal_initializer(stddev=0.02))
            D_b3 = tf.get_variable('D_b3', [self.depth2], initializer=tf.constant_initializer())
            D_W4 = tf.get_variable('D_W4', [self.kernel_size, self.kernel_size, self.depth2, self.depth1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            D_b4 = tf.get_variable('D_b4', [self.depth1], initializer=tf.constant_initializer())
            D_W4 = tf.get_variable('D_W4', [self.dim5 * self.dim5 * self.depth1, self.n_cls], initializer=tcl.xavier_initializer())
            D_b4 = tf.get_variable('D_b4', [self.n_cls], initializer=tf.constant_initializer())

            X = tf.reshape(X, [-1, self.dim1, self.dim1, self.channel])
            layer1 = self.leaky_relu(tf.nn.conv2d(X, D_W1, [1, 2, 2, 1], 'SAME') + D_b1)
            layer2 = self.leaky_relu(self.d_bn1(tf.nn.conv2d(layer1, D_W2, [1, 2, 2, 1], 'SAME') + D_b2, is_training=is_training))
            layer3 = self.leaky_relu(self.d_bn2(tf.nn.conv2d(layer2, D_W3, [1, 2, 2, 1], 'SAME') + D_b3, is_training=is_training))
            layer4 = self.leaky_relu(self.d_bn3(tf.nn.conv2d(layer3, D_W4, [1, 2, 2, 1], 'SAME') + D_b4, is_training=is_training))
            layer4 = tf.reshape(layer4, [-1, self.dim5 * self.dim5 * self.depth1])
            layer5 = tf.matmul(layer4, D_W5) + D_b5

            if self.n_cls == 2:
                prediction = tf.nn.sigmoid(layer5)
            else:
                prediction = tf.nn.softmax(layer5)

        return prediction

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name), \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
