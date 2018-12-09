import tensorflow as tf
import tensorlayer as tl
import numpy as np
from utils import *

#PARAMETERS
BASE_CHANNELS = 64
G_CHANNEL_TIMES = [2, 4, 8, 8]
D_CHANNEL_TIMES = [1, 2, 4, 8]

class Model:
    def __init__(self, input_size, batch_size, mean_error_scaling, learning_rate, adam_beta1):
        self.input_size = input_size
        self.batch_size = batch_size
        self.mean_error_scaling = mean_error_scaling
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        self.rgb_imgs = tf.placeholder(tf.float32, [batch_size, input_size, input_size, 3], name='rgb_imgs')
        self.skch_imgs = tf.placeholder(tf.float32, [batch_size, input_size, input_size, 1], name='skch_imgs')
        self.hint_imgs = tf.placeholder(tf.float32, [batch_size, input_size, input_size, 3], name='hint_imgs')

        g_input = tf.concat([self.skch_imgs, self.hint_imgs], axis = 3)

        self.g_net, g_logits = self.generator(g_input, isTrain = True, reuse = False)

        d_input_fake = tf.concat([g_input, self.g_net.outputs], 3)
        d_input_real = tf.concat([g_input, self.rgb_imgs], 3)

        self.d_fake_net, d_logits_fake = self.discriminator(d_input_fake, isTrain = True, reuse = False)
        d_real_net, d_logits_real = self.discriminator(d_input_real, isTrain = True, reuse = True)

        self.g_loss, self.d_loss = self.dcgan_loss(d_logits_fake, d_logits_real)
        self.g_loss += mean_error_scaling * self.mean_error(self.rgb_imgs, self.g_net.outputs)

        g_vars = tl.layers.get_variables_with_name('generator', True, False)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, False)

        self.g_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1 = adam_beta1).minimize(self.g_loss, var_list=g_vars)
        self.d_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1 = adam_beta1).minimize(self.d_loss, var_list=d_vars)

        self.g_net_eval, g_logits_eval = self.generator(g_input, isTrain = False, reuse = True)

    def mean_error(self, real, fake, mode = 'L1'):
        error = tf.abs(real - fake)
        if mode == 'L2':
            error = tf.square(error)
        return tf.reduce_mean(error)

    def dcgan_loss(self, fake_logits, real_logits):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
        d_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
        return g_loss, d_loss

    def generator(self, inputs, isTrain = True, reuse = True):
        with tf.variable_scope("generator", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            input = tl.layers.InputLayer(inputs, name='input')
            c0 = tl.layers.Conv2d(input, BASE_CHANNELS, (5, 5), (2, 2), act = lrelu, name = 'c0')
            c1 = tl.layers.Conv2d(c0, BASE_CHANNELS * G_CHANNEL_TIMES[0], (5, 5), (2, 2), name = 'c1')
            bnc1 = tl.layers.BatchNormLayer(c1, is_train = isTrain, act = lrelu, name = 'bnc1')
            c2 = tl.layers.Conv2d(bnc1, BASE_CHANNELS * G_CHANNEL_TIMES[1], (5, 5), (2, 2), name = 'c2')
            bnc2 = tl.layers.BatchNormLayer(c2, is_train = isTrain, act = lrelu, name = 'bnc2')
            c3 = tl.layers.Conv2d(bnc2, BASE_CHANNELS * G_CHANNEL_TIMES[2], (5, 5), (2, 2), name = 'c3')
            bnc3 = tl.layers.BatchNormLayer(c3, is_train = isTrain, act = lrelu, name = 'bnc3')
            c4 = tl.layers.Conv2d(bnc3, BASE_CHANNELS * G_CHANNEL_TIMES[3], (5, 5), (2, 2), name = 'c4')
            bnc4 = tl.layers.BatchNormLayer(c4, is_train = isTrain, act = tf.nn.relu, name = 'bnc4')

            dc4 = tl.layers.DeConv2d(bnc4, BASE_CHANNELS * G_CHANNEL_TIMES[-1], (5, 5), (2, 2), name = 'dc4')
            bnd4 = tl.layers.BatchNormLayer(dc4, is_train= isTrain, name='bnd4')
            concat1 = tl.layers.ConcatLayer([bnd4, bnc3], concat_dim=3, name='concat_layer1')
            concat1.outputs = tf.nn.relu(concat1.outputs)
            dc3 = tl.layers.DeConv2d(concat1, BASE_CHANNELS * G_CHANNEL_TIMES[-3], (5, 5), (2, 2), name = 'dc3')
            bnd3 = tl.layers.BatchNormLayer(dc3, is_train= isTrain, name='bnd3')
            concat2 = tl.layers.ConcatLayer([bnd3, bnc2], concat_dim=3, name='concat_layer2')
            concat2.outputs = tf.nn.relu(concat2.outputs)
            dc2 = tl.layers.DeConv2d(concat2, BASE_CHANNELS * G_CHANNEL_TIMES[-4], (5, 5), (2, 2), name = 'dc2')
            bnd2 = tl.layers.BatchNormLayer(dc2, is_train= isTrain, name='bnd2')
            concat3 = tl.layers.ConcatLayer([bnd2, bnc1], concat_dim=3, name='concat_layer3')
            concat3.outputs = tf.nn.relu(concat3.outputs)
            dc1 = tl.layers.DeConv2d(concat3, BASE_CHANNELS, (5, 5), (2, 2), name = 'dc1')
            bnd1 = tl.layers.BatchNormLayer(dc1, is_train= isTrain, name='bnd1')
            concat4 = tl.layers.ConcatLayer([bnd1, c0], concat_dim=3, name='concat_layer4')
            concat4.outputs = tf.nn.relu(concat4.outputs)
            dc0 = tl.layers.DeConv2d(concat4, 3, (5, 5), (2, 2), name = 'dc0')

            logits = dc0.outputs
            dc0.outputs = tf.nn.tanh(logits)
        return dc0, logits

    def discriminator(self, inputs, isTrain = True, reuse = True):
        with tf.variable_scope("discriminator", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            network = tl.layers.InputLayer(inputs, name='dis_input')
            network = tl.layers.Conv2d(network, BASE_CHANNELS * D_CHANNEL_TIMES[0], (5, 5), (2, 2), act=lrelu, name = 'dis_c' + str(0))
            for i in range(1, len(D_CHANNEL_TIMES) - 1):
                network = tl.layers.Conv2d(network, BASE_CHANNELS * D_CHANNEL_TIMES[i], (5, 5), (2, 2), name = 'dis_c' + str(i))
                network = tl.layers.BatchNormLayer(network, is_train=isTrain, act=lrelu, name='dis_bnc' + str(i))

            network = tl.layers.Conv2d(network, BASE_CHANNELS * D_CHANNEL_TIMES[-1], (5, 5), (1, 1), name = 'dis_c')
            network = tl.layers.BatchNormLayer(network, is_train=isTrain,act=lrelu, name='dis_bnc')

            logits = linear(tf.reshape(network.outputs, [self.batch_size, -1]), 1, 'linear')

            network.outputs = tf.nn.sigmoid(logits)
        return network, logits

    def load_model(self, model_paths, sess):
        params = tl.files.load_npz(model_paths[0], '')
        tl.files.assign_params(sess, params, self.g_net)
        params = tl.files.load_npz(model_paths[1], '')
        tl.files.assign_params(sess, params, self.d_fake_net)

    def save_model(self, model_paths, sess):
        tl.files.save_npz(self.g_net.all_params, model_paths[0], sess)
        tl.files.save_npz(self.d_fake_net.all_params, model_paths[1], sess)
