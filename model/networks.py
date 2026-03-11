"""
UGATIT Generator Network (Light Version)
Ported from taki0112/UGATIT for inference only.
Only the A→B generator (selfie→anime) is needed for test mode.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from model.ops import (
    conv, instance_norm, relu, tanh, resblock,
    adaptive_ins_layer_resblock, layer_instance_norm,
    up_sample, global_avg_pooling, global_max_pooling,
    fully_connected_with_w, fully_connected
)


class UGATITGenerator:
    """UGATIT Light Generator for selfie→anime translation."""

    def __init__(self, ch=64, n_res=4, img_size=256, img_ch=3,
                 smoothing=True):
        self.ch = ch
        self.n_res = n_res
        self.img_size = img_size
        self.img_ch = img_ch
        self.smoothing = smoothing
        self.light = True  # Kaggle pretrained model uses light mode
        self.batch_size = 1

    def generator(self, x_init, reuse=False, scope="generator"):
        """Build the generator graph."""
        channel = self.ch
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x = conv(x_init, channel, kernel=7, stride=1, pad=3,
                     pad_type='reflect', scope='conv')
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)

            # Down-Sampling
            for i in range(2):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1,
                         pad_type='reflect', scope='conv_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = relu(x)
                channel = channel * 2

            # Down-Sampling Bottleneck
            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_' + str(i))

            # Class Activation Map
            cam_x = global_avg_pooling(x)
            cam_gap_logit, cam_x_weight = fully_connected_with_w(
                cam_x, scope='CAM_logit')
            x_gap = tf.multiply(x, cam_x_weight)

            cam_x = global_max_pooling(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(
                cam_x, reuse=True, scope='CAM_logit')
            x_gmp = tf.multiply(x, cam_x_weight)

            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            x = tf.concat([x_gap, x_gmp], axis=-1)

            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = relu(x)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

            if self.light:
                x_ = global_avg_pooling(x)
                x_ = fully_connected(x_, channel, scope='FC')
                x_ = relu(x_)
                gamma, beta = self.MLP(x_, reuse=reuse)
            else:
                gamma, beta = self.MLP(x, reuse=reuse)

            # Up-Sampling Bottleneck
            for i in range(self.n_res):
                x = adaptive_ins_layer_resblock(
                    x, channel, gamma, beta,
                    smoothing=self.smoothing,
                    scope='adaptive_resblock' + str(i))

            # Up-Sampling
            for i in range(2):
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel // 2, kernel=3, stride=1, pad=1,
                         pad_type='reflect', scope='up_conv_' + str(i))
                x = layer_instance_norm(
                    x, scope='layer_ins_norm_' + str(i))
                x = relu(x)
                channel = channel // 2

            x = conv(x, channels=3, kernel=7, stride=1, pad=3,
                     pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x, cam_logit, heatmap

    def MLP(self, x, use_bias=True, reuse=False, scope='MLP'):
        """Multi-Layer Perceptron for AdaLIN parameters."""
        channel = self.ch * self.n_res

        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            for i in range(2):
                x = fully_connected(x, channel, use_bias,
                                    scope='linear_' + str(i))
                x = relu(x)

            gamma = fully_connected(x, channel, use_bias, scope='gamma')
            beta = fully_connected(x, channel, use_bias, scope='beta')

            gamma = tf.reshape(gamma,
                               shape=[self.batch_size, 1, 1, channel])
            beta = tf.reshape(beta,
                              shape=[self.batch_size, 1, 1, channel])

            return gamma, beta

    def generate_a2b(self, x_A, reuse=False):
        """Selfie → Anime translation (generator_B)."""
        out, cam, _ = self.generator(x_A, reuse=reuse, scope="generator_B")
        return out, cam


def build_test_graph(img_size=256, img_ch=3):
    """
    Build the test-mode computation graph.
    Returns (input_placeholder, output_tensor, generator_instance).
    """
    gen = UGATITGenerator(img_size=img_size, img_ch=img_ch)

    test_input = tf.compat.v1.placeholder(
        tf.float32,
        [1, img_size, img_size, img_ch],
        name='test_domain_A')

    test_output, _ = gen.generate_a2b(test_input)

    return test_input, test_output, gen
