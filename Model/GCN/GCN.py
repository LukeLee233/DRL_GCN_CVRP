

import tensorflow as tf

class GCN(object):
    '''
    This class is the implementation of the Graph-convolution network
    '''
    def __init__(self,args,logging=True):
        self.name = 'Graph conv net'
        self.logging = logging
        self.args = args
        self.n_customers = self.args['n_customers']
        self.n_nodes = self.n_customers + 1

        #input is the distance matrix of the graph
        self.inputs = tf.placeholder(tf.float32,shape=[None,self.n_nodes,self.n_nodes])

        #latent layer config
        self.latent_layer_dim = self.args['GCN_latent_layer_dim']
        self.layer_num = self.args['GCN_layer_num']


    def get_layer_uid(self,layer_name=''):
        '''
        Helper function, assigns unique layer IDs.
        :param layer_name: layer name
        :return: layer unique id
        '''

        if layer_name not in self._LAYER_UIDS:
            self._LAYER_UIDS[layer_name] = 1
            return 1
        else:
            self._LAYER_UIDS[layer_name] += 1
            return self._LAYER_UIDS[layer_name]

    def build(self):
        self.layers = []

        # global unique layer ID dictionary for layer name assignment
        self._LAYER_UIDS = {}

        with tf.variable_scope(self.name):
            self._build()



    def _build(self):



