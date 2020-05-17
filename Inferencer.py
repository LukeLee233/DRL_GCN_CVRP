from Model.model import *


class Inferencer(object):
    def __init__(self, model_path, model, env):
        self.model = model
        self.environment = env

        self.output = model.outputs

        self.model_path = model_path
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt:
            print('Loading model...\npath: ' + ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('Fatal Error,can\'t find trained model!')

    def watch_variables(self):
        # Variables are independent with inputs
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print('Model variables watch:')
        for item in var_list:
            print(item.name, ' shape:', item.shape)
            print(self.sess.run(item))

    def __call__(self, input_data):
        # self.watch_variables()
        self.environment.info_inspect(input_data, self.model, self.sess)
        self.environment.get_routes(input_data, self.model, self.sess)


if __name__ == '__main__':
    args = {}

    # Trainer
    args['trainer_save_interval'] = 10000
    args['trainer_inspect_interval'] = 10000
    args['trainer_model_dir'] = 'model_trained/'
    args['trainer_epoch'] = 260000
    args['batch_size'] = 1
    args['keep_prob'] = 1

    # Environment
    args['n_customers'] = 10
    args['data_dir'] = 'data/'
    args['random_seed'] = 1
    args['instance_num'] = 5000
    args['capacity'] = 20

    # Network
    # embedding type
    # {'gcn','linear_embedding'}
    args['embedding_type'] = 'linear_embedding'

    #linear embedding
    args['linear_embedding_num_units'] = 128
    args['linear_embedding_lr'] = 0.01

    # GCN
    args['gcn_lr'] = 0.01
    args['max_grad_norm'] = 10
    args['GCN_max_degree'] = 1
    args['GCN_vertex_dim'] = 32
    args['GCN_latent_layer_dim'] = 128
    args['GCN_layer_num'] = 5
    args['GCN_diver_num'] = 15

    # Actor
    args['actor_lr'] = 0.01
    args['actor_hidden_dim'] = 128
    args['actor_use_tanh'] = False
    args['actor_tanh_exploration'] = 10
    args['actor_n_glimpses'] = 1
    args['actor_mask_glimpses'] = False
    args['actor_mask_pointer'] = True
    args['actor_forget_bias'] = True
    args['actor_rnn_layer_num'] = 1
    args['actor_decode_len'] = 16

    # Critic
    args['critic_lr'] = 0.01
    args['critic_rnn_layers'] = 1
    args['critic_hidden_dim'] = 128
    args['critic_n_process_blocks'] = 3



    try:
        if args['actor_decode_len'] < args['n_customers']:
            raise Exception('Error, Decision step less than customer number!')
    except Exception as expection:
        print(expection)
        exit()

    tf.reset_default_graph()

    with tf.variable_scope('Input'):
        input_data = {
            'input_pnt': tf.placeholder(tf.float32, shape=[args['batch_size'], args['n_customers'] + 1, 2],
                                        name='coordinates'),
            'input_distance_matrix': tf.placeholder(tf.float32, shape=[args['batch_size'], args['n_customers'] + 1,
                                                                       args['n_customers'] + 1],
                                                    name='distance_matrix'),
            'demand': tf.placeholder(tf.float32, shape=[args['batch_size'], args['n_customers'] + 1], name='demand')
        }


    environment = Environment(args)
    my_model = Model(args, input_data, environment, embedding_type=args['embedding_type'],operation='infer')
    datamanager = DataManager(args, 'test')
    datamanager.create_data()

    agent = Inferencer('model_trained', my_model, environment)
    task_name,test_data = datamanager.load_task()
    print(task_name)

    agent(test_data)
