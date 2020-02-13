from Model.GCN.GCN import *
from Model.A3C.Critic import *
from Model.A3C.Actor import *


class Model(object):
    def __init__(self,args,inputs,placeholders,env):
        self.name = 'Model'
        self.inputs = inputs

        self.gcn = GCN(args,placeholders)
        self.actor = Actor(args,input=self.gcn.outputs,env=env,placeholder=placeholders)
        self.critic = Critic(args,self.gcn.outputs,env)

        self.outputs = {
            'sequence_id' :self.actor.summary[2],
            'sequence_coordinates':self.actor.summary[1]
            }

    def inference(self):
        pass



if __name__ == '__main__':

    args = {}
    args['capacity'] = 10
    args['n_customers'] = 10


    args['GCN_max_degree'] = 1
    args['GCN_vertex_dim'] = 32
    args['GCN_latent_layer_dim'] = 128
    args['GCN_layer_num'] = 5
    args['GCN_diver_num'] = 15
    placeholders = {
        'support': [tf.compat.v1.placeholder(tf.float32,name='support_matrix_'+str(i)) for i in range(args['GCN_max_degree']+1)],#related to adjacent matrix
        'dropout': tf.compat.v1.placeholder(tf.float32, shape=(),name='drop_prob'),
        'adjacent_inputs' : tf.compat.v1.placeholder(tf.float32,shape=[None,args['n_customers']+1,args['n_customers']+1],name='adjacent_matrix'),
        'initial_vertex_state' : tf.compat.v1.placeholder(tf.float32,shape=[None,args['n_customers']+1,args['GCN_vertex_dim']],name='initial_vertex_state')
    }

    args['actor_hidden_dim'] = 128
    args['actor_use_tanh'] = False
    args['actor_tanh_exploration'] = 10
    args['actor_n_glimpses'] = 2
    args['actor_mask_glimpses'] = False
    args['actor_mask_pointer'] = False
    args['actor_forget_bias'] = True
    args['actor_rnn_layer_num'] = 3
    args['actor_decode_len'] = 5

    args['critic_rnn_layers'] = 3
    args['critic_hidden_dim'] = 128
    args['critic_rnn_layers'] = 4
    args['critic_n_process_blocks'] = 3
    args['critic_hidden_dim'] = 128

    placeholders = {
        'support': [tf.compat.v1.placeholder(tf.float32,name='support_matrix_'+str(i)) for i in range(args['GCN_max_degree']+1)],#related to adjacent matrix
        'dropout': tf.compat.v1.placeholder(tf.float32, shape=(),name='drop_prob'),
        'adjacent_inputs' : tf.compat.v1.placeholder(tf.float32,shape=[None,args['n_customers']+1,args['n_customers']+1],name='adjacent_matrix'),
        'initial_vertex_state' : tf.compat.v1.placeholder(tf.float32,shape=[None,args['n_customers']+1,args['GCN_vertex_dim']],name='initial_vertex_state')
        }

    with tf.variable_scope('Input'):
        input_data = {
            'input_pnt': tf.placeholder(tf.float32, shape=[None, 11, 2], name='coordinates'),
            'input_distance_matrix': tf.placeholder(tf.float32, shape=[None, 11, 11], name='distance_matrix'),
            'demand': tf.placeholder(tf.float32, shape=[None, 11], name='demand')
        }

    model = model()