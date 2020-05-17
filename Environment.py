'''
Author: Luke
Date: 2019-9-2
'''

import tensorflow as tf
import collections
import time
import numpy as np
import contextlib
import scipy.io as sio


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class Environment(object):
    def __init__(self, args, beam_width=1):
        print('Create environment...')
        start_time = time.time()
        self.args = args

        self.n_customers = args['n_customers']
        self.depot_id = self.n_customers
        self.n_nodes = self.n_customers + 1
        self.capacity = args['capacity']
        self.batch_size = args['batch_size']

        self.beam_width = beam_width
        self.batch_beam = self.batch_size * self.beam_width

        self.demand_trace = []
        self.load_trace = []
        self.mask_trace = []

        self.reward_trace = []
        self.prob_trace = []

        with tf.variable_scope('Environment'):
            with tf.variable_scope('Costumer_Info'):
                self.input_pnt = tf.placeholder(tf.float32, shape=[None, self.n_nodes, 2], name='Input_pnt')

                self.input_distance_matrix = tf.placeholder(tf.float32, shape=[None, self.n_nodes, self.n_nodes],
                                                            name='Input_distance_matrix')

                init_demand = tf.placeholder(tf.float32, shape=[None, self.n_nodes], name='Initial_Input_demand')

            with tf.variable_scope('Vehicle_Info'):
                init_load = tf.multiply(tf.ones([self.batch_beam]), self.capacity, name='Initial_Load')

            self.demand_trace.append(init_demand)

            with tf.variable_scope('Mask'):
                mask = tf.concat([tf.cast(tf.equal(self.demand_trace[-1], 0), tf.float32)[:, :-1],
                                  tf.ones([self.batch_beam, 1])], 1, name='Initial_mask')

            self.load_trace.append(init_load)
            self.mask_trace.append(mask)

        model_time = time.time() - start_time
        print(f'It took {model_time:.2f}s to build the environment.')


    def info_inspect(self, input_data, model, sess):
        '''
        Resets the environment. The environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.

        :param beam_width: width of the beam search, default = 1 is greedy search
        :return: a snapshot of the env
        '''
        # dimensions
        with tf.variable_scope('Environment/Info_inspect'):
            print("Info Statistic!")
            for i in range(self.args['actor_decode_len'] + 1):
                prob = self.prob_trace[i].eval(session=sess, feed_dict={
                    self.demand_trace[0]: input_data['demand'],
                    model.inputs['input_pnt']:input_data['input_pnt'],
                    model.inputs['input_distance_matrix']: input_data['input_distance_matrix']})

                with printoptions(precision=2, suppress=True):
                    print('Porbability snap shot step', i, ':\n', prob)

                idxs = self.idxs_trace[i].eval(session=sess, feed_dict={
                    self.demand_trace[0]: input_data['demand'],
                    model.inputs['input_distance_matrix']: input_data['input_distance_matrix'],
                    model.inputs['input_pnt']:input_data['input_pnt'],
                    self.input_pnt: input_data['input_pnt']})

                # actions = self.actions_trace[i].eval(session=sess, feed_dict={
                #     self.demand_trace[0]: input_data['demand'],
                #     model.inputs['input_distance_matrix']: input_data['input_distance_matrix'],
                #     model.inputs['input_pnt']: input_data['input_pnt'],
                #     self.input_pnt: input_data['input_pnt']})

                # print('Actor', i, 'step decision:\n ', idxs, '\n', actions)

                print('Actor', i, 'step decision:\n ', idxs)

                demand = (self.demand_trace[i].eval(session=sess, feed_dict={
                    self.demand_trace[0]: input_data['demand'],
                    model.inputs['input_pnt']:input_data['input_pnt'],
                    model.inputs['input_distance_matrix']: input_data['input_distance_matrix']})).astype(int)
                print(f'Demand snap shot step {i} :\n{demand}')

                load = (self.load_trace[i].eval(session=sess, feed_dict={
                    self.demand_trace[0]: input_data['demand'],
                    model.inputs['input_pnt']: input_data['input_pnt'],
                    model.inputs['input_distance_matrix']: input_data['input_distance_matrix']})).astype(int)
                print('Load snap shot step ', i, ':\n', load)


                reward = self.reward_trace[i].eval(session=sess, feed_dict={self.demand_trace[0]: input_data['demand'],
                                                                            model.inputs['input_distance_matrix']:
                                                                                input_data['input_distance_matrix'],
                                                                            model.inputs['input_pnt']: input_data[
                                                                                'input_pnt'],
                                                                            self.input_pnt: input_data['input_pnt']})
                print('Current Reward step ', i, ':\n', reward)

                mask = ~(0 == self.mask_trace[i].eval(session=sess, feed_dict={
                    self.demand_trace[0]: input_data['demand'],
                    model.inputs['input_pnt']:input_data['input_pnt'],
                    model.inputs['input_pnt']:input_data['input_pnt'],
                    model.inputs['input_distance_matrix']: input_data['input_distance_matrix']}))
                print('Next step Mask snap shot ', i, ':\n', mask)

    def response(self,
                 idx,
                 beam_parent=None):
        '''
        reponse the choice made by actor and updates environment(demands,loads and masks)
        :param idx: [batch_size x 1],indicate the choice of the actor
        :param beam_parent: (default=None),whether to use beam search decoder
        :return: a snapshot of the updated env
    '''
        with tf.variable_scope('Environment/Response', reuse=False):
            # if the environment is used in beam search decoder
            demand = self.demand_trace[-1]
            load = self.load_trace[-1]

            if beam_parent is not None:
                # BatchBeamSeq: [batch_size*beam_width x 1]
                # [0,1,2,3,...,127,0,1,...127,0,1,...127]
                batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(self.batch_size), tf.int64),
                                                      [self.beam_width]), 1)
                # batchedBeamIdx:[batch_size*beam_width]
                batchedBeamIdx = batchBeamSeq + tf.cast(self.batch_size, tf.int64) * beam_parent
                # demand:[batch_size*beam_width x sourceL]
                demand = tf.gather_nd(demand, batchedBeamIdx)
                # load:[batch_size*beam_width]
                load = tf.gather_nd(load, batchedBeamIdx, name='Updated_load')
                # MASK:[batch_size*beam_width x sourceL]
                # mask = tf.gather_nd(mask,batchedBeamIdx)

            # batched_idx[ [0,a] [1,b] ... [127,z] ]
            BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_beam), tf.int64), 1)
            batched_idx = tf.concat([BatchSequence, idx], 1)

            # how much the demand is satisfied,here use split delivery mechanism
            d_sat = tf.minimum(tf.gather_nd(demand, batched_idx), load)

            # update the demand
            d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(demand), tf.int64))
            demand = tf.subtract(demand, d_scatter, name='Updeted_demand')

            # update load
            load -= d_sat

            # refill the truck -- idx: [10;9;10] -> load_flag: [1 0 1]
            load_flag = tf.squeeze(tf.cast(tf.equal(idx, self.depot_id), tf.float32), 1)
            load = tf.add(tf.multiply(load, 1 - load_flag), tf.multiply(load_flag, self.capacity), name='Updated_Load')

            # mask for customers with zero demand
            mask = tf.concat([tf.cast(tf.equal(demand, 0), tf.float32)[:, :-1],
                              tf.zeros([self.batch_beam, 1])], 1)

            # mask if load= 0
            # mask if in depot and there is still a demand
            mask = tf.add(mask, tf.concat([tf.tile(tf.expand_dims(tf.cast(tf.equal(load, 0),
                                                                          tf.float32), 1), [1, self.n_customers]),
                                           tf.expand_dims(
                                               tf.multiply(tf.cast(tf.greater(tf.reduce_sum(demand, 1), 0), tf.float32),
                                                           tf.squeeze(
                                                               tf.cast(tf.equal(idx, self.depot_id), tf.float32))), 1)],
                                          1), name='Updated_mask')

            self.demand_trace.append(demand)
            self.load_trace.append(load)
            self.mask_trace.append(mask)

            return tf.reduce_sum(demand, name='Left_demand', axis=1)

    def get_routes(self, input_data, model, sess):
        solutions = [[]] * self.batch_size
        routes = [[]] * self.batch_size
        route_num = [0] * self.batch_size
        #first times empty of demand should be handle
        first_empty = [False]*self.batch_size

        for i in range(self.args['actor_decode_len'] + 1):
            actions = self.actions_trace[i].eval(session=sess, feed_dict={
                self.demand_trace[0]: input_data['demand'],
                model.inputs['input_distance_matrix']: input_data['input_distance_matrix'],
                model.inputs['input_pnt']: input_data['input_pnt'],
                self.input_pnt: input_data['input_pnt']})

            if i == 0:
                for j in range(self.batch_size):
                    routes[j] = [tuple(actions[j])]
            else:
                idxs = self.idxs_trace[i].eval(session=sess, feed_dict={
                    self.demand_trace[0]: input_data['demand'],
                    model.inputs['input_distance_matrix']: input_data['input_distance_matrix'],
                    model.inputs['input_pnt']: input_data['input_pnt'],
                    self.input_pnt: input_data['input_pnt']})

                ending_signal = (idxs == self.depot_id).squeeze(axis=1)

                demand = (self.demand_trace[i].eval(session=sess, feed_dict={
                    self.demand_trace[0]: input_data['demand'],
                    model.inputs['input_pnt']: input_data['input_pnt'],
                    model.inputs['input_distance_matrix']: input_data['input_distance_matrix']})).astype(int)

                for j, signal in enumerate(ending_signal):
                    if signal and (not first_empty[j]):
                        if all(demand[j] == 0):
                            first_empty[j] = True
                        routes[j].append(tuple(actions[j]))
                        if route_num[j] == 0:
                            solutions[j] = [routes[j]]
                        else:
                            solutions[j].append(routes[j])
                        route_num[j] += 1
                        # routes[j].clear()

                        routes[j] = [tuple(actions[j])]
                    elif not signal:
                        routes[j].append(tuple(actions[j]))


        # handle the left sequence which cannot go back to depot
        for j in range(self.batch_size):
            if len(routes[j]) > 1 and routes[j][0] != routes[j][1]:
                solutions[j].append(routes[j])

        fname = 'result.mat'
        sio.savemat(fname, {'route_num': route_num, 'solutions': solutions})

        return (solutions, route_num)

    # def get_reward(self, sample_solution):
    #
    #     """The reward for the VRP task is defined as the
    #     value of the route length
    #     note: any sub-sol can be calculated
    #     Args:
    #         sample_solution : a list tensor of size decode_len(sourceL) of shape [batch_size x input_dim]
    #     Returns:
    #         rewards: tensor of size [batch_size]
    #
    #     Example:
    #         sample_solution = [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
    #         two instances,two solutions:
    #         [1,1]->[3,3]->[5,5]
    #         [2,2]->[4,4]->[6,6]
    #         sourceL(#customer in a single solution) = 3
    #         batch_size(#solution) = 2
    #         input_dim(coordinates) = 2
    #         sample_solution_tilted[ [[5,5]
    #                                  [6,6]]
    #                                 [[1,1]
    #                                  [2,2]]
    #                                 [[3,3]
    #                                  [4,4]] ]
    #         result: [[11.3173],[11.3173]]
    #     """
    #     with tf.variable_scope('Environment/Reward'):
    #         # make sample_solution of shape [sourceL x batch_size x input_dim]
    #         sample_solution = tf.stack(sample_solution, 0, name='Solution')
    #
    #         # I don't think a loop isn't a right definition
    #         # sample_solution_rotated = tf.concat((tf.expand_dims(sample_solution[-1], 0),
    #         #                                     sample_solution[:-1]), 0)
    #         # get the reward based on the route lengths
    #         # route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow(
    #         #     (sample_solution_rotated - sample_solution), 2), 2), .5), 0, name='Route_length')
    #
    #         sample_solution_shifted = tf.concat((tf.expand_dims(sample_solution[0], 0),
    #                                              sample_solution[:-1]), 0)
    #
    #         route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow(
    #             (sample_solution_shifted - sample_solution), 2), 2), .5), 0, name='Route_length')
    #
    #         return route_lens_decoded

    def get_reward(self,sample_solution):
        """The reward for the VRP task is defined as the
        negative value of the route length

        Args:
            sample_solution : a list tensor of size decode_len of shape [batch_size x input_dim]
            demands satisfied: a list tensor of size decode_len of shape [batch_size]

        Returns:
            rewards: tensor of size [batch_size]

        Example:
            sample_solution = [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
            sourceL = 3
            batch_size = 2
            input_dim = 2
            sample_solution_tilted[ [[5,5]
                                                        #  [6,6]]
                                                        # [[1,1]
                                                        #  [2,2]]
                                                        # [[3,3]
                                                        #  [4,4]] ]
        """
        # make init_solution of shape [sourceL x batch_size x input_dim]

        # make sample_solution of shape [sourceL x batch_size x input_dim]
        if not sample_solution:
            return tf.zeros(shape=self.batch_size,dtype=tf.float32)

        sample_solution = tf.stack(sample_solution, 0)

        sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1], 0),
                                            sample_solution[:-1]), 0)
        # get the reward based on the route lengths

        route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow( \
            (sample_solution_tilted - sample_solution), 2), 2), .5), 0)
        return route_lens_decoded


    def trace_actor_act(self, idxs, actions):
        self.actions_trace = actions
        self.idxs_trace = idxs

    def record_reward(self, rewards):
        self.reward_trace.append(rewards)

    def record_prob(self, prob):
        self.prob_trace.append(prob)
