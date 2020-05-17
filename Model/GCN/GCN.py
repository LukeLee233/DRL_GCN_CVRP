from Model.GCN.Layer import *
from Environment import *
from Data import *
import numpy as np


class GCN(Module):
    '''
    This class is the implementation of the Graph-convolution network
    '''

    def __init__(self, args, logging=False,scope='',inputs = None):
        super(GCN,self).__init__(name='GCN',scope=scope,logging=logging)

        self.args = args

        self.n_customers = self.args['n_customers']
        self.n_nodes = self.n_customers + 1
        self.num_supports = self.args['GCN_max_degree'] + 1
        self.batch_size = self.args['batch_size']

        self.vertex_dim = self.args['GCN_vertex_dim']  # 32 may be OK

        # static input needed by GCN
        with tf.variable_scope(self.name + '/Data_Input'):
            if inputs:
                self.adj_inputs = inputs['input_distance_matrix']
            else:
                self.adj_inputs = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.n_nodes,self.n_nodes],name='adj_inputs')
            # self.adj_inputs = inputs['input_distance_matrix']
            self.initial_vertex_state = self.intial_features()
            self.supports = self.data_preprocess()

        # latent layer config
        self.latent_layer_dim = self.args['GCN_latent_layer_dim']
        self.layer_num = self.args['GCN_layer_num']

        # diver_num: how many graph network
        self.diver_num = self.args['GCN_diver_num']

        self.build()

    def intial_features(self):
        # process feature
        features = tf.ones([self.batch_size, self.n_nodes, self.vertex_dim])
        row_sum = tf.reduce_sum(features, axis=2)
        features = tf.divide(features, tf.expand_dims(row_sum, 2), name='Initial_vertex_feature')

        return features

    # support matrix
    def data_preprocess(self):
        return simple_polynomials(self.adj_inputs, self.args['GCN_max_degree'], batch_size=self.batch_size)

    def get_layer_uid(self, layer_name=''):
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
        self.activations = []

        # global unique layer ID dictionary for layer name assignment
        self._LAYER_UIDS = {}

        with tf.variable_scope(self.name):
            self._build()

            # start build compute data flow between layers
            layer_id = 0
            self.activations.append(self.initial_vertex_state)
            for layer in self.layers:
                if layer_id < len(self.layers) - 1:
                    hidden = tf.nn.relu(layer(self.activations[-1]))
                    self.activations.append(hidden)
                    layer_id = layer_id + 1
                else:
                    hidden = layer(self.activations[-1])
                    self.activations.append(hidden)
                    layer_id = layer_id + 1

            # activations[-1]:[batch x n_nodes x diver_number*vertex_dim]
            # outputs:[batch x n_nodes x vertex_dim]
            # Note: These graph convs share variables
            with tf.variable_scope('raw_output'):
                self.raw_outputs = self.activations[-1][:, :, :self.vertex_dim]

                for out_id in range(1, self.diver_num):
                    self.raw_outputs = self.raw_outputs + self.activations[-1][:, :,
                                                          self.vertex_dim * out_id: self.vertex_dim * (out_id + 1)]

                # finally output is the average of different draph
                self.raw_outputs = tf.divide(self.raw_outputs, tf.cast(self.diver_num, tf.float32), name='average')

            self.outputs = tf.identity(self.raw_outputs, name='output')

    def _build(self):
        '''
        build the basic frame of GCN,
        but don't specify compute data flow between different layer
        '''
        self.layers.append(GraphConvolutionLayer(input_dim=self.vertex_dim,
                                                 output_dim=self.latent_layer_dim,
                                                 support=self.supports,
                                                 act=tf.nn.relu,
                                                 keep_prob=self.args['keep_prob'],
                                                 name=('graphconvulation_' + str(
                                                     self.get_layer_uid('graphconvulation'))),
                                                 scope= self.scope+'/'+self.name,
                                                 logging=self.logging))

        for i in range(self.layer_num - 2):
            self.layers.append(GraphConvolutionLayer(input_dim=self.latent_layer_dim,
                                                     output_dim=self.latent_layer_dim,
                                                     support=self.supports,
                                                     act=tf.nn.relu,
                                                     keep_prob=self.args['keep_prob'],
                                                     name=('graphconvulation_' + str(
                                                         self.get_layer_uid('graphconvulation'))),
                                                     scope=self.scope + '/'+self.name,
                                                     logging=self.logging))

        self.layers.append(GraphConvolutionLayer(input_dim=self.latent_layer_dim,
                                                 output_dim=self.vertex_dim * self.diver_num,
                                                 support=self.supports,
                                                 act=lambda x: x,
                                                 keep_prob=self.args['keep_prob'],
                                                 name=('graphconvulation_' + str(
                                                     self.get_layer_uid('graphconvulation'))),
                                                 scope= self.scope+'/'+self.name,
                                                 logging=self.logging))

def simple_polynomials(adj, k, batch_size):
    """Calculate polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)

    # laplacian = None
    # eye = None
    # first_flag = True
    #
    # for i in range(batch_size):
    #     if first_flag:
    #         laplacian = tf.expand_dims(tf.eye(tf.cast(adj.shape[1],tf.int32)) - adj_normalized[i,:,:],axis=0)
    #         eye = tf.expand_dims(tf.eye(tf.cast(adj.shape[1],tf.int32)),axis=0)
    #         first_flag = False
    #     else:
    #         laplacian = tf.concat((laplacian,tf.expand_dims(tf.eye(tf.cast(adj.shape[1],tf.int32)) - adj_normalized[i,:,:],axis=0)),axis=0)
    #         eye = tf.concat((eye,tf.expand_dims(tf.eye(tf.cast(adj.shape[1],tf.int32)),axis=0)),axis=0)

    laplacian = None
    first_flag = True

    # no self-connect loop
    laplacian = adj_normalized

    t_k = list()
    # t_k.append(tf.convert_to_tensor(eye,tf.float32))
    # t_k.append(tf.convert_to_tensor(laplacian,tf.float32))

    # t_k.append(eye)
    t_k.append(laplacian)

    for i in range(1, k):
        # Hadamard product
        t_new = tf.multiply(t_k[-1], laplacian)
        # t_k.append(tf.convert_to_tensor(t_new),tf.float32)
        t_k.append(t_new)

    return t_k


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     inverse_adj = np.where(adj == 0, 0, 1 / adj)

    inverse_adj = tf.divide(1, adj)
    inverse_adj = tf.where(tf.equal(inverse_adj, tf.multiply(tf.ones_like(inverse_adj), np.inf)),
                           tf.zeros_like(inverse_adj), inverse_adj)

    row_sum = tf.reduce_sum(inverse_adj, axis=2)
    norm_inv_adj = inverse_adj / tf.expand_dims(row_sum, axis=2)
    return norm_inv_adj


if __name__ == '__main__':
    args = {}
    args['n_customers'] = 10
    args['GCN_max_degree'] = 2
    args['GCN_vertex_dim'] = 32
    args['GCN_latent_layer_dim'] = 128
    args['GCN_layer_num'] = 5
    args['GCN_diver_num'] = 15
    args['batch_size'] = 1
    args['data_dir'] = '../data/'
    args['random_seed'] = 1
    args['instance_num'] = 1000
    args['capacity'] = 20
    args['keep_prob'] = 0.1

    with tf.variable_scope('Input'):
        input_data = {
            'input_pnt': tf.placeholder(tf.float32, shape=[args['batch_size'], 11, 2], name='coordinates'),
            'input_distance_matrix': tf.placeholder(tf.float32, shape=[args['batch_size'], 11, 11],
                                                    name='distance_matrix'),
            'demand': tf.placeholder(tf.float32, shape=[args['batch_size'], 11], name='demand')
        }

    prototype = GCN(args, input_data, logging=True)

    datamanager = DataManager(args, 'train')
    datamanager.create_data()

    init = tf.initialize_all_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    writer = tf.summary.FileWriter('./graph/', tf.get_default_graph())
    summaries = tf.summary.merge_all()

    for i in range(1):
        with tf.Session(config=config) as sess:
            sess.run(init)


            _,train_data = datamanager.load_task()

            print('Initial vertex state:\n', sess.run(prototype.initial_vertex_state, feed_dict={
                prototype.adj_inputs: train_data['input_distance_matrix']}))

            print('Support matrix Degree 1:\n',
                  sess.run(prototype.supports[0], feed_dict={prototype.adj_inputs: train_data['input_distance_matrix']}))

            print('Support matrix Degree 2:\n',
                  sess.run(prototype.supports[1], feed_dict={prototype.adj_inputs: train_data['input_distance_matrix']}))


            summ = sess.run(summaries,feed_dict={prototype.adj_inputs: train_data['input_distance_matrix']})
            writer.add_summary(summ, global_step=i)
