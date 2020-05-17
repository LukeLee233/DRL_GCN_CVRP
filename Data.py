import scipy.io as sio
from scipy.spatial import distance_matrix
import os
import numpy as np

def create_VRP_dataset(
        n_problems,
        n_cust,
        data_dir,
        capacity,
        seed=None,
        data_type='train'):
    '''
    This function creates VRP instances and saves them on disk. If a file is already available,
    nothing will be done.
    Input:
        n_problems: number of problems to generate.
        n_cust: number of customers in the problem.
        data_dir: the directory to save or load the file.
        seed: random seed for generating the data.
        data_type: the purpose for generating the data. It can be 'train', 'val', or any string.
    output:
        dir of the created data
     '''

    # set random number generator
    n_nodes = n_cust + 1
    if seed == None:
        rnd = np.random
    else:
        rnd = np.random.RandomState(seed)

    # build task name and datafiles
    task_dir = 'vrp{}'.format(n_cust)
    fname = os.path.join(data_dir, task_dir)

    instance_type=''

    # cteate data
    if os.path.exists(fname):
        print('Data {} already exists!'.format(task_dir))
    else:
        if not os.path.isdir(fname):
            os.makedirs(fname)
        for i in range(n_problems):
            prop = np.random.random()
            if prop <= .5:
                coordinates = rnd.uniform(0, 1, size=(n_nodes, 2))
                instance_type = 'uniform'
            else:
                # coordinates = rnd.triangular(0,mode=0.5,right=1,size=(n_nodes,2))
                coordinates = np.random.normal(.5,.15,size=(n_nodes,2))
                coordinates[coordinates > 1] = 1
                coordinates[coordinates < 0] = 0
                instance_type = 'guassian'


            demand = rnd.randint(1, 10, [n_nodes, 1])
            demand[-1, :] = 0

            shortest_path_matrix = distance_matrix(coordinates, coordinates)

            task_name = 'vrp-size-{}-id-{}-{}.mat'.format(n_cust, i + 1, data_type)

            path = os.path.join(fname, task_name)

            sio.savemat(path, {'shortest_path_matrix': shortest_path_matrix, 'demand': demand,
                               'coordinates': coordinates, 'capacity': capacity, 'instance_type': instance_type})

    return fname


file_id = 0


class DataManager(object):
    def __init__(self, args, data_type):
        self.args = args
        self.path = args['data_dir'] + '/' + str(data_type)
        self.batch_size = args['batch_size']
        self.rnd = np.random.RandomState(seed=args['random_seed'])

        self.n_problems = args['instance_num']
        self.data_type = data_type
        self.current_batch_id = 0
        self.total_batch = self.n_problems // self.batch_size

    def create_data(self):
        path = self.path
        self.path = create_VRP_dataset(self.n_problems, self.args['n_customers'], path,
                                       seed=self.args['random_seed'] + 1, data_type=self.data_type,
                                       capacity=self.args['capacity'])

        self._get_task_name()

    def _get_task_name(self):
        self.files_name = os.listdir(self.path)

    def load_task(self,fixed_name=''):
        if not fixed_name:
            if (self.current_batch_id + 1 == self.total_batch):
                print('Reuse data')
                self.current_batch_id = 0

            self.current_file_set = self.files_name[self.current_batch_id * self.batch_size:
                                                    (self.current_batch_id + 1) * self.batch_size]
            self.current_batch_id += 1

            # self.current_file_set = random.sample(self.files_name,self.batch_size)

            path_matrix = None
            demand = None
            coordinates = None
            intial_flag = True
            type_count = {'guassian':0,'uniform':0}
            for task_name in self.current_file_set:
                file = sio.loadmat(self.path + '/' + task_name)

                if file['instance_type'] == 'guassian':
                    type_count['guassian'] += 1
                elif file['instance_type'] == 'uniform':
                    type_count['uniform'] += 1

                if intial_flag:
                    path_matrix = np.expand_dims(file['shortest_path_matrix'], axis=0)
                    demand = np.expand_dims(file['demand'], axis=0)
                    coordinates = np.expand_dims(file['coordinates'], axis=0)
                    intial_flag = False
                else:
                    path_matrix = np.concatenate((path_matrix, np.expand_dims(file['shortest_path_matrix'], axis=0)),
                                                 axis=0)
                    demand = np.concatenate((demand, np.expand_dims(file['demand'], axis=0)), axis=0)
                    coordinates = np.concatenate((coordinates, np.expand_dims(file['coordinates'], axis=0)), axis=0)

            self.data = {}
            self.data['input_distance_matrix'] = path_matrix
            self.data['demand'] = np.squeeze(demand, axis=2)
            self.data['input_pnt'] = coordinates

            print(f"uniform count: {type_count['uniform']}, guassian count: {type_count['guassian']}\n")


            if len(self.current_file_set) == 1:
                return self.current_file_set[0], self.data
            else:
                return self.data

        else:
            file = sio.loadmat(self.path + '/' + fixed_name)
            path_matrix = np.expand_dims(file['shortest_path_matrix'], axis=0)
            demand = np.expand_dims(file['demand'], axis=0)
            coordinates = np.expand_dims(file['coordinates'], axis=0)

            self.data = {}
            self.data['input_distance_matrix'] = path_matrix
            self.data['demand'] = np.squeeze(demand, axis=2)
            self.data['input_pnt'] = coordinates


            return self.data

if __name__ == '__main__':
    args = {}
    args['data_dir'] = './data/'
    args['random_seed'] = 1
    args['instance_num'] = 1000
    args['n_cust'] = 10
    args['capacity'] = 20
    args['batch_size'] = 128

    datamanager = DataManager(args, data_type='train')
    datamanager.create_data()

    while True:
        data = datamanager.load_task()
        data
