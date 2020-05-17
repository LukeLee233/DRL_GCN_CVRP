import tensorflow as tf
import numpy as np
from version import *

class Module(object):
    def __init__(self,name,scope='',logging = False):
        self.name = name
        self.scope = scope

        self.logging = logging

    def __repr__(self):
        return self.scope + '/' + self.name


class GraphConvolutionLayer(object):
    '''
    one single Convulution layer
    '''

    def __init__(self, input_dim, output_dim, support, name,
                 scope='',keep_prob=1, act=tf.nn.relu, bias=True,
                 featureless=False, logging=True):
        '''
        :param featureless: If this is true, means don't do feature extraction in this layer
        :param logging: Whether to log the traing
        '''
        self.name = name
        self.scope = scope

        self.logging = logging

        self.keep_prob = keep_prob

        # one is for theta_0,the other is for theta_1
        self.support = support

        self.act = act
        self.featureless = featureless
        self.bias = bias

        self.vars = {}
        with tf.variable_scope(self.name + '/vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _log_vars(self):
        with tf.variable_scope(self.name):
            for var in self.vars:
                tf.summary.histogram('vars/' + var, self.vars[var])

    def __call__(self, inputs):
        if self.logging:
            tf.summary.histogram(self.name + '/inputs', inputs)

        with tf.variable_scope(self.name + '/Forward'):
            outputs = self._call(inputs)

        if self.logging:
            tf.summary.histogram(self.name + '/outputs', outputs)

        return outputs

    def _call(self, inputs):
        x = inputs

        # dropout
        x = tf.nn.dropout(x, keep_prob=self.keep_prob)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                if cur_version == VERSION['PC']:
                    pre_sup = tf.matmul(x, self.vars['weights_' + str(i)])
                elif cur_version == VERSION['Server']:
                    pre_sup = future_matmul(x, self.vars['weights_' + str(i)])
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = tf.matmul(self.support[i], pre_sup)
            supports.append(support)
        # just add different degree support matrix element-wise
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def future_matmul(A, B):
    B_ = tf.tile(B, [tf.shape(A)[0], 1])
    B = tf.reshape(B_, [tf.shape(A)[0], tf.shape(B)[0], tf.shape(B)[1]])
    return tf.matmul(A, B)
