"""
UGATIT Neural Network Operations
Ported from taki0112/UGATIT (TF1.14) to tf.compat.v1 for TF2 compatibility.
All variable names and scopes are preserved exactly to match pretrained checkpoint keys.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# ─────────────────────────────────────────────
# Weight initializers (must match original)
# ─────────────────────────────────────────────
weight_init = tf.compat.v1.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf.keras.regularizers.l2(0.0001)


# ─────────────────────────────────────────────
# Layer operations
# ─────────────────────────────────────────────

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero',
         use_bias=True, sn=False, scope='conv_0'):
    with tf.compat.v1.variable_scope(scope):
        if pad > 0:
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad
            else:
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom],
                                [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom],
                                [pad_left, pad_right], [0, 0]],
                           mode='REFLECT')

        if sn:
            w = tf.compat.v1.get_variable(
                "kernel",
                shape=[kernel, kernel, x.get_shape()[-1], channels],
                initializer=weight_init,
                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filters=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.compat.v1.get_variable(
                    "bias", [channels],
                    initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else:
            x = tf.compat.v1.layers.conv2d(
                inputs=x, filters=channels,
                kernel_size=kernel,
                kernel_initializer=weight_init,
                kernel_regularizer=weight_regularizer,
                strides=stride, use_bias=use_bias)

        return x


def fully_connected_with_w(x, use_bias=True, sn=False, reuse=False,
                            scope='linear'):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        x = flatten(x)
        bias = 0.0
        shape = x.get_shape().as_list()
        channels = shape[-1]

        w = tf.compat.v1.get_variable(
            "kernel", [channels, 1], tf.float32,
            initializer=weight_init, regularizer=weight_regularizer)

        if sn:
            w = spectral_norm(w)

        if use_bias:
            bias = tf.compat.v1.get_variable(
                "bias", [1],
                initializer=tf.constant_initializer(0.0))
            x = tf.matmul(x, w) + bias
        else:
            x = tf.matmul(x, w)

        if use_bias:
            weights = tf.gather(tf.transpose(tf.nn.bias_add(w, bias)), 0)
        else:
            weights = tf.gather(tf.transpose(w), 0)

        return x, weights


def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.compat.v1.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.compat.v1.get_variable(
                "kernel", [channels, units], tf.float32,
                initializer=weight_init, regularizer=weight_regularizer)
            if use_bias:
                bias = tf.compat.v1.get_variable(
                    "bias", [units],
                    initializer=tf.constant_initializer(0.0))
                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))
        else:
            x = tf.compat.v1.layers.dense(
                x, units=units,
                kernel_initializer=weight_init,
                kernel_regularizer=weight_regularizer,
                use_bias=use_bias)

        return x


def flatten(x):
    return tf.compat.v1.layers.flatten(x)


# ─────────────────────────────────────────────
# Residual blocks
# ─────────────────────────────────────────────

def resblock(x_init, channels, use_bias=True, scope='resblock_0'):
    with tf.compat.v1.variable_scope(scope):
        with tf.compat.v1.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1,
                     pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.compat.v1.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1,
                     pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init


def adaptive_ins_layer_resblock(x_init, channels, gamma, beta,
                                 use_bias=True, smoothing=True,
                                 scope='adaptive_resblock'):
    with tf.compat.v1.variable_scope(scope):
        with tf.compat.v1.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1,
                     pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_layer_norm(x, gamma, beta, smoothing)
            x = relu(x)

        with tf.compat.v1.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1,
                     pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_layer_norm(x, gamma, beta, smoothing)

        return x + x_init


# ─────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.compat.v1.image.resize_nearest_neighbor(x, size=new_size)


def global_avg_pooling(x):
    return tf.reduce_mean(x, axis=[1, 2])


def global_max_pooling(x):
    return tf.reduce_max(x, axis=[1, 2])


# ─────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────

def lrelu(x, alpha=0.01):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


def sigmoid(x):
    return tf.sigmoid(x)


# ─────────────────────────────────────────────
# Normalization functions
# ─────────────────────────────────────────────

def adaptive_instance_layer_norm(x, gamma, beta, smoothing=True,
                                  scope='instance_layer_norm'):
    with tf.compat.v1.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + eps))

        rho = tf.compat.v1.get_variable(
            "rho", [ch],
            initializer=tf.constant_initializer(1.0),
            constraint=lambda x: tf.clip_by_value(
                x, clip_value_min=0.0, clip_value_max=1.0))

        if smoothing:
            rho = tf.clip_by_value(rho - tf.constant(0.1), 0.0, 1.0)

        x_hat = rho * x_ins + (1 - rho) * x_ln
        x_hat = x_hat * gamma + beta

        return x_hat


def instance_norm(x, scope='instance_norm'):
    """
    Manual instance normalization replacing tf.contrib.layers.instance_norm.
    Variable names 'gamma' and 'beta' match the original TF1 checkpoint keys.
    """
    with tf.compat.v1.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        mu, sigma_sq = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_normed = (x - mu) / tf.sqrt(sigma_sq + eps)

        gamma = tf.compat.v1.get_variable(
            "gamma", [ch],
            initializer=tf.constant_initializer(1.0))
        beta = tf.compat.v1.get_variable(
            "beta", [ch],
            initializer=tf.constant_initializer(0.0))

        return gamma * x_normed + beta


def layer_instance_norm(x, scope='layer_instance_norm'):
    with tf.compat.v1.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + eps))

        rho = tf.compat.v1.get_variable(
            "rho", [ch],
            initializer=tf.constant_initializer(0.0),
            constraint=lambda x: tf.clip_by_value(
                x, clip_value_min=0.0, clip_value_max=1.0))

        gamma = tf.compat.v1.get_variable(
            "gamma", [ch],
            initializer=tf.constant_initializer(1.0))
        beta = tf.compat.v1.get_variable(
            "beta", [ch],
            initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_ins + (1 - rho) * x_ln
        x_hat = x_hat * gamma + beta

        return x_hat


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.compat.v1.get_variable(
        "u", [1, w_shape[-1]],
        initializer=tf.compat.v1.random_normal_initializer(),
        trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

def L1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def regularization_loss(scope_name):
    collection_regularization = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    loss = []
    for item in collection_regularization:
        if scope_name in item.name:
            loss.append(item)
    return tf.reduce_sum(loss)
