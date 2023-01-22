import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from util import *
from model import *


class AgentV:
    def __init__(self, kwargs):
        self.name = 'agent_v'
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']
        self.graph_size = kwargs['graph_size']
        self.all_pairs_distance = kwargs['graph_all_pairs_distance']
        graph_name = kwargs['graph_name']
        # used to store model
        self.model = None
        # used to determine model accuracy
        self.model_mse = 0
        if not self.read(graph_name):
            # train model and get the final parameters
            self.get_model_params(graph_name)
            self.write(graph_name)
        # read in U*
        # as dictionary with form of {state: utility*}
        self.utility_star_dict = {}
        self.__read_u_star(graph_name)

    def write(self, graph_name):
        filename = f'v_model_params/{graph_name}.v_model_params'
        if os.path.isfile(filename):
            return False
        with open(filename, 'w') as f:
            # write out model parameters dictionary resulted from trained model
            # for later model reference
            f.write(','.join(map(str, list(self.model.params['w'].flatten()))) + '\n')
            f.write(','.join(map(str, list(self.model.params['b'].flatten()))) + '\n')
        return True

    def read(self, graph_name):
        filename = f'v_model_params/{graph_name}.v_model_params'
        if not os.path.isfile(filename):
            return False
        with open(filename, 'r') as f:
            # read in model parameters resulted from trained model
            temp_params = {}
            w_list = list(map(float, f.readline().rstrip().split(',')))
            temp_params['w'] = np.array(w_list).reshape((len(w_list), 1))
            b_list = list(map(float, f.readline().rstrip().split(',')))
            temp_params['b'] = np.array(b_list).reshape((len(b_list), 1))
            kwargs = {
                'train_mode': False,
                'params': temp_params
            }
            # set up model with read in parameters resulted from trained model
            self.model = LinearRegression(kwargs)
        return True

    def refresh(self, kwargs):
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']

    def is_infinite_state(self, init_state, graph):
        # check if a starting state is infinite (i.e., having infinite utility)
        return self.utility_star_dict[init_state] == float('Inf')

    def get_model_params(self, graph_name):
        raw_data_filename = f'v_dataset/one_graph_raw_data_v.csv'
        # construct raw data (include features and true label)
        self.__write_unigraph_raw_dataset(graph_name, raw_data_filename)
        # train model
        self.__train_model(raw_data_filename)
        os.remove(raw_data_filename)

    def update_node_name(self, next_node_name):
        self.pre_node_name = self.cur_node_name
        self.cur_node_name = next_node_name

    def next_node_name(self, graph):
        min_utility = float('Inf')
        utility_action_choice_list = []
        next_node_name_list = [neighbor.name for neighbor in graph.adj_list[self.cur_node_name].neighbors]
        next_node_name_list.append(self.cur_node_name)
        # get agent current standing node and its neighbor's utility
        for node_name in next_node_name_list:
            utility = reward = 1 if node_name != graph.predator.cur_node_name else float('Inf')
            predator_shortest_distance_neighbor_count = 0
            shortest_distance = float('Inf')
            for predator_neighbor in graph.adj_list[graph.predator.cur_node_name].neighbors:
                dist_node_to_predator_neighbor = self.all_pairs_distance[node_name][predator_neighbor.name]
                if dist_node_to_predator_neighbor < shortest_distance:
                    shortest_distance = dist_node_to_predator_neighbor
            for predator_neighbor in graph.adj_list[graph.predator.cur_node_name].neighbors:
                dist_node_to_predator_neighbor = self.all_pairs_distance[node_name][predator_neighbor.name]
                if dist_node_to_predator_neighbor == shortest_distance:
                    predator_shortest_distance_neighbor_count += 1
            # record next state
            prey_and_prey_neighbor_names = [prey_neighbor.name for prey_neighbor
                                            in graph.adj_list[graph.prey.cur_node_name].neighbors]
            prey_and_prey_neighbor_names.append(graph.prey.cur_node_name)
            predator_neighbor_names = [predator_neighbor.name for predator_neighbor
                                       in graph.adj_list[graph.predator.cur_node_name].neighbors]
            for prey_or_prey_neighbor_name in prey_and_prey_neighbor_names:
                for predator_neighbor_name in predator_neighbor_names:
                    data_list = self.turn_state_into_model_input(node_name,
                                                                 prey_or_prey_neighbor_name,
                                                                 predator_neighbor_name)
                    predicted_utility = self.model.predict(data_list).flatten()[0]
                    # compute and record transition probability
                    prey_transition_probability = 1 / (1 + len(graph.adj_list[graph.prey.cur_node_name].neighbors))
                    predator_transition_probability = 0.4 * \
                                                      (1 / len(graph.adj_list[graph.predator.cur_node_name].neighbors))
                    if self.all_pairs_distance[node_name][predator_neighbor_name] == shortest_distance:
                        predator_transition_probability += 0.6 * (
                                1 / predator_shortest_distance_neighbor_count)
                    joint_transition_probability = prey_transition_probability * \
                                                   predator_transition_probability
                    utility += joint_transition_probability * predicted_utility
            if utility < min_utility:
                min_utility = utility
            utility_action_choice_list.append((utility, node_name))
        # get optimal action(s) for current state
        optimal_action_choice_list = []
        for u, a in utility_action_choice_list:
            if u == min_utility:
                optimal_action_choice_list.append(a)
        return random.choice(optimal_action_choice_list)

    def turn_state_into_model_input(self, agent_node_name, prey_node_name, predator_node_name):
        dist_py, dist_pd = self.all_pairs_distance[agent_node_name][prey_node_name], \
                           self.all_pairs_distance[agent_node_name][predator_node_name]
        # normalize distance data with min max scalar
        dist_py, dist_pd = dist_py / (self.graph_size - 1), dist_pd / (self.graph_size - 1)
        data_list = [1 if _ == agent_node_name else 0
                     for _ in range(1, self.graph_size + 1)]
        data_list.extend([1 if _ == prey_node_name else 0
                          for _ in range(1, self.graph_size + 1)])
        data_list.extend([1 if _ == predator_node_name else 0
                          for _ in range(1, self.graph_size + 1)])
        data_list.extend([dist_py, dist_pd])
        return data_list

    def __write_unigraph_raw_dataset(self, graph_name, filename):
        with open(filename, 'w') as fout:
            # write out input data as raw data for later model training use
            headers = ['Id']
            headers.extend([f'Ag{i}' for i in range(1, self.graph_size + 1)])
            headers.extend([f'Pd{i}' for i in range(1, self.graph_size + 1)])
            headers.extend([f'Py{i}' for i in range(1, self.graph_size + 1)])
            headers.extend(['DistPy', 'DistPd', 'U*'])
            fout.write(','.join(headers) + '\n')
            with open(f'rsc/{graph_name}.agent_u_star_utility_policy_dict', 'r') as fin:
                fin.readline()
                num_state = self.graph_size ** 3
                for i in range(1, num_state + 1):
                    agent_node_name, prey_node_name, predator_node_name = \
                        map(int, fin.readline().rstrip().split(','))
                    utility_star, = map(float, fin.readline().rstrip().split(','))
                    fin.readline()
                    data_list = [i]
                    data_list.extend(self.turn_state_into_model_input(agent_node_name,
                                                                      prey_node_name,
                                                                      predator_node_name))
                    data_list.append(utility_star)
                    fout.write(','.join(map(str, data_list)) + '\n')

    def __train_model(self, raw_data_filename):
        # implement supervised learning with linear regression model to get U*
        df = pd.read_csv(raw_data_filename, sep=',')
        # exclude data with infinite U* from being fed into model
        u_df = df[df['U*'] != float('Inf')]
        all_x = u_df.drop(columns=['U*'])
        # get all features
        all_x = all_x.drop(columns=['Id'])
        # get all true labels
        all_y = u_df['U*']
        num_features = all_x.shape[1]
        # split out test set
        x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.1, random_state=1)
        # split out train and validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
        # set up hyperparameter and then initialize model
        kwargs = {
            'train_mode': True,
            'learning_rate': 1e-2,
            'mse_threshold': 5,
        }
        self.model = LinearRegression(kwargs)
        # train model with part of all data
        # self.model.fit(train_input_x=x_train.T, train_output_y=y_train.T, val_input_x=x_val.T, val_output_y=y_val.T)
        # train model with all input
        # (without splitting since after trying to use part of all data to train,
        # since there's an underfitting situation)
        self.model.fit(train_input_x=all_x.T, train_output_y=all_y.T, val_input_x=None, val_output_y=None)
        self.model.plot_loss()
        predicted_optimal_utilities = self.model.predict(input_x=all_x.T)
        # get mse of all data for usage in knowing accuracy of model
        self.model_mse = self.model.loss_func(predicted_optimal_utilities.flatten(), all_y.T, all_y.shape[0])

    def __read_u_star(self, graph_name):
        filename = f'rsc/{graph_name}.agent_u_star_utility_policy_dict'
        if not os.path.isfile(filename):
            return False
        with open(filename, 'r') as f:
            self.graph_size, = map(int, f.readline().rstrip().split(','))
            num_state = self.graph_size ** 3
            for _ in range(num_state):
                agent_node_name, prey_node_name, predator_node_name =\
                    map(int, f.readline().rstrip().split(','))
                kwargs = {
                    'agent_node_name': agent_node_name,
                    'prey_node_name': prey_node_name,
                    'predator_node_name': predator_node_name,
                    'graph_size': self.graph_size
                }
                state = State(kwargs)
                utility, = map(float, f.readline().rstrip().split(','))
                self.utility_star_dict[state] = utility
                f.readline()
        return True
