'''
Author: Luke
Date: 2019-9-4
'''


from Environment import *
from Model.GCN.Layer import Module
from Data import *

class Critic(Module):
    def __init__(self,args,input,env,scope='',logging = False):
        super(Critic,self).__init__(name='Critic',scope=scope,logging=logging)

        self.embedding_input = input
        self.batch_size = args['batch_size']

        with tf.variable_scope(self.name):
            with tf.variable_scope("Encoder/Initial_state"):
                # init states
                initial_state = tf.zeros([args['critic_rnn_layers'], 2, self.batch_size, args['critic_hidden_dim']],
                                         name='stacked_intial_state')
                l = tf.unstack(initial_state, axis=0,name='unstacked_state')
                rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[i][0], l[i][1])
                                         for i in range(args['critic_rnn_layers'])])

                hy = tf.identity(rnn_tuple_state[0][1],name='hidden_state')

            with tf.variable_scope("Encoder/Process"):
                for i in range(args['critic_n_process_blocks']):
                    process = CriticAttentionLayer(args['critic_hidden_dim'], _name = "encoding_step_"+str(i),scope=self.scope + '/'+ self.name + '/Encoder/Process')
                    e, logit = process(hy, self.embedding_input, env)

                    prob = tf.nn.softmax(logit)
                    # hy : [batch_size x 1 x sourceL] * [batch_size  x sourceL x hidden_dim]  ->
                    # [batch_size x h_dim ]
                    hy = tf.squeeze(tf.matmul(tf.expand_dims(prob, 1), e), 1)

            with tf.variable_scope("Decoder"):
                self.reward_predict = tf.layers.dense( tf.layers.dense(hy, args['critic_hidden_dim'], tf.nn.relu, name='Full_Connect_L1'),
                                                            1, name='Full_Connect_L2')

            self.reward_predict = tf.squeeze(self.reward_predict,1, name= 'Reward_Predict')

        if self.logging == True:
            self._log()


    def _log(self):
        if self.scope == '':
            scope = self.name
        else:
            scope = self.scope + '/' + self.name
        for var in tf.trainable_variables(scope=scope):
            if self.scope == '':
                tf.summary.histogram(var.name, var)
            else:
                tf.summary.histogram(var.name[len(self.scope) + 1:], var)



class CriticAttentionLayer(object):
    """A generic attention module for the attention in vrp model"""

    def __init__(self, dim, _name,use_tanh=False, C=10,scope=''):
        self.scope = scope
        self.call_time = 0
        self.use_tanh = use_tanh
        self.name = _name

        with tf.variable_scope(self.name):
            # self.v: is a variable with shape [1 x dim]
            self.v = tf.get_variable('v_vector', [1, dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.v = tf.expand_dims(self.v, 2, name='expand_v_vector')

            self.emb_d = tf.keras.layers.Conv1D(dim, 1, name= self.name + '/emb_d')
            self.project_d = tf.keras.layers.Conv1D(dim, 1, name= self.name + '/proj_d')

            self.project_query = tf.keras.layers.Dense(dim, name= self.name + '/proj_q')
            self.project_ref = tf.keras.layers.Conv1D(dim, 1, name= self.name + '/proj_ref')

        self.C = C  # tanh exploration parameter
        self.tanh = tf.nn.tanh

    def __call__(self, query, ref, env):
        """
        This function gets a query tensor and ref rensor and returns the logit op.
        Args:
            query: is the hidden state of the decoder at the current
                time step. [batch_size x dim]
            ref: the set of hidden states from the encoder.
                [batch_size x max_time x dim]

            env: keeps demand ond load values and help decoding. Also it includes mask.
                env.mask: a matrix used for masking the logits and glimpses. It is with shape
                         [batch_size x max_time]. Zeros in this matrix means not-masked nodes. Any
                         positive number in this mask means that the node cannot be selected as next
                         decision point.
                env.demands: a list of demands which changes over time.

        Returns:
            e: convolved ref with shape [batch_size x max_time x dim]
            logits: [batch_size x max_time]
        """
        # we need the first demand value for the critic
        self.call_time += 1
        demand = tf.identity(env.demand_trace[-1],name= self.name + '/demand')
        max_time = tf.identity(tf.shape(demand)[1],name= self.name + '/maxtime')

        # embed demand and project it
        # emb_d:[batch_size x max_time x dim ]
        emb_d = tf.identity(self.emb_d(tf.expand_dims(demand, 2)),name=self.name + '/new_emb_d')
        # d:[batch_size x max_time x dim ]
        d = tf.identity(self.project_d(emb_d),name=self.name+'/new_proj_d')

        # expanded_q,e: [batch_size x max_time x dim]
        e = tf.identity(self.project_ref(ref),name=self.name+'/new_proj_ref')
        q = tf.identity(self.project_query(query),name=self.name+'/new_proj_q')  # [batch_size x dim]
        expanded_q = tf.tile(tf.expand_dims(q, 1), [1, max_time, 1],name=self.name+'/expended_proj_q')

        # v_view:[batch_size x dim x 1]
        v_view = tf.tile(self.v, [tf.shape(e)[0], 1, 1])

        # u : [batch_size x max_time x dim] * [batch_size x dim x 1] =
        #       [batch_size x max_time]
        u = tf.squeeze(tf.matmul(self.tanh(expanded_q + e + d), v_view), 2)

        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u

        return e, logits


if __name__ == '__main__':
    args = {}

    args['trainer_save_interval'] = 10000
    args['trainer_inspect_interval'] = 10000
    args['trainer_model_dir'] = 'model_trained/'
    args['trainer_total_epoch'] = 100000

    args['n_customers'] = 10
    args['data_dir'] = 'data/'
    args['random_seed'] = 1
    args['instance_num'] = 10000
    args['capacity'] = 20
    args['batch_size'] = 128

    args['actor_net_lr'] = 0.01
    args['critic_net_lr'] = 0.01
    args['gcn_net_lr'] = 0.01
    args['max_grad_norm'] = 10
    args['keep_prob'] = 0.1

    args['critic_rnn_layers'] = 3
    args['critic_hidden_dim'] = 128
    args['critic_rnn_layers'] = 4
    args['critic_n_process_blocks'] = 3


    with tf.variable_scope('Input'):
        input_data = {
            'input_pnt': tf.placeholder(tf.float32, shape=[args['batch_size'], args['n_customers'] + 1, 2],
                                        name='coordinates'),
            'input_distance_matrix': tf.placeholder(tf.float32, shape=[args['batch_size'], args['n_customers'] + 1,
                                                                       args['n_customers'] + 1],
                                                    name='distance_matrix'),
            'demand': tf.placeholder(tf.float32, shape=[args['batch_size'], args['n_customers'] + 1], name='demand')
        }


    datamanager = DataManager(args, 'train')
    datamanager.create_data()

    environment = Environment(args)
    model = Critic(args, input_data['input_distance_matrix'], env=environment, logging=True)

    init = tf.initialize_all_variables()

    writer = tf.summary.FileWriter('./graph/', tf.get_default_graph())
    summaries = tf.summary.merge_all()


    for i in range(1):
        with tf.Session() as sess:
            sess.run(init)

            input_data = datamanager.load_task()

            summ = sess.run(summaries, feed_dict={
                model.embedding_input : input_data['input_distance_matrix'],
                environment.input_pnt: input_data['input_pnt'],
                environment.input_distance_matrix: input_data['input_distance_matrix'],
                environment.demand_trace[0]: input_data['demand']})

            writer.add_summary(summ, global_step=i)