from Model.A3C.Actor import *
from Model.A3C.Critic import *
from Model.GCN.GCN import *
from Data import *
import json


class LinearEmbedding(Module):
    def __init__(self, args, scope='', logging=False):
        super(LinearEmbedding, self).__init__(name='LinearEmbediding', scope=scope, logging=logging)

        self.units = args['linear_embedding_num_units']
        if args['linear_embedding_use_tanh'] :
            self.layer = tf.layers.Dense(self.units, use_bias=args['linear_embedding_use_bias'])
        else :
            self.layer = tf.layers.Dense(self.units, activation=tf.nn.tanh, use_bias=args['linear_embedding_use_bias'])


        self.keep_prob = args['keep_prob']

    def __call__(self, inputs, *args, **kwargs):
        with tf.variable_scope(self.name):
            input = tf.cast(inputs['input_pnt'], dtype=tf.float32)

            input = tf.nn.dropout(input, keep_prob=self.keep_prob)
            with tf.variable_scope('Forward'):
                self.output = self.layer(input)

        # layer's variables will be created after invoked
        if self.logging:
            self._log_vars()

        return self.output

    def _log_vars(self):
        # tf.trainable_variables() use absolute path to locate variable, and yet tf.summary.histogram() use relative path
        # so I need to rename the scope to make sure the right relationship
        for var in tf.trainable_variables(scope=self.scope + '/' + self.name):
            tf.summary.histogram(var.name[len(self.scope) + 1:], var)


class Model(object):
    def __init__(self, args, Inputs, env, embedding_type='gcn',operation='train'):
        self.name = 'Model'

        #input include coordinates,demand,adjacent matrix
        self.inputs = Inputs

        with tf.variable_scope(self.name):
            if embedding_type == 'gcn':
                self.gcn = GCN(args, scope=self.name, logging=True, inputs=self.inputs)
                self.embedding = self.gcn.outputs
            elif embedding_type == 'linear_embedding':
                self.linear_layer = LinearEmbedding(args, scope=self.name, logging=True)
                self.embedding = self.linear_layer(self.inputs)

            if operation == 'train':
                print('Use greedy policy to train...')
                self.actor = Actor(args, input=self.embedding, env=env, scope=self.name,logging=True,mode='greedy')
            elif operation == 'infer':
                self.actor = Actor(args, input=self.embedding, env=env, scope=self.name,logging=True,mode='greedy')

            self.critic = Critic(args, self.embedding, env, scope=self.name,logging=True)

            self.outputs = self.actor.outputs

    def inference(self):
        pass


def record_args(args, file_name='args.json'):
    with open(file_name, 'w') as f:
        json.dump(args, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    args = {}
    args['capacity'] = 10
    args['n_customers'] = 10
    args['data_dir'] = '../data/'
    args['random_seed'] = 1
    args['instance_num'] = 1000
    args['capacity'] = 20
    args['batch_size'] = 128
    args['keep_prob'] = 0.2

    args['GCN_max_degree'] = 1
    args['GCN_vertex_dim'] = 32
    args['GCN_latent_layer_dim'] = 128
    args['GCN_layer_num'] = 5
    args['GCN_diver_num'] = 15

    # {'gcn','linear_embedding'}
    args['embedding_type'] = 'linear_embedding'
    args['linear_embedding_num_units'] = 128

    args['actor_hidden_dim'] = 128
    args['actor_use_tanh'] = False
    args['actor_tanh_exploration'] = 10
    args['actor_n_glimpses'] = 2
    args['actor_mask_glimpses'] = False
    args['actor_mask_pointer'] = True
    args['actor_forget_bias'] = True
    args['actor_rnn_layer_num'] = 3
    args['actor_decode_len'] = 5

    args['critic_rnn_layers'] = 3
    args['critic_hidden_dim'] = 128
    args['critic_rnn_layers'] = 4
    args['critic_n_process_blocks'] = 3
    args['critic_hidden_dim'] = 128

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
    model = Model(args, Inputs=input_data, env=environment,
                  embedding_type=args['embedding_type'])

    init = tf.initialize_all_variables()

    writer = tf.summary.FileWriter('./graph/', tf.get_default_graph())
    summaries = tf.summary.merge_all()

    for i in range(1):
        with tf.Session() as sess:
            sess.run(init)

            input_data = datamanager.load_task()

            summ = sess.run(summaries, feed_dict={
                model.inputs['input_pnt']: input_data['input_pnt'],
                model.inputs['input_distance_matrix']: input_data['input_distance_matrix'],
                model.inputs['demand']: input_data['demand'],
                environment.input_pnt: input_data['input_pnt'],
                environment.input_distance_matrix: input_data['input_distance_matrix'],
                environment.demand_trace[0]: input_data['demand']})

            writer.add_summary(summ, global_step=i)
