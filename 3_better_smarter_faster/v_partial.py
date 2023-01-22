import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from util import *
from model import *


class AgentVPartial:
    def __init__(self, kwargs):
        self.name = 'agent_v_partial'
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']
        self.graph_size = kwargs['graph_size']
        self.all_pairs_distance = kwargs['graph_all_pairs_distance']
        graph_name = kwargs['graph_name']
        self.prey_transition_matrix = None
        self.prey_belief_vector = None
        # used to store model
        self.model = None
        # used to determine model accuracy
        self.model_mse = 0
        self.model_num_prediction = 0
        self.utility_star_dict = {}
        self.__read_u_star(graph_name)
        if not self.read(graph_name):
            # train model and get the final parameters
            self.get_model_params(graph_name)
            self.write(graph_name)

    def write(self, graph_name):
        filename = f'v_partial_model_params/{graph_name}.test_v_partial_model_params'
        if os.path.isfile(filename):
            return False
        with open(filename, 'w') as f:
            # write out model hyperparameters and parameters dictionary resulted from trained model
            # for later model reference
            f.write(','.join(map(str, self.model.ith_layer_nodes)) + '\n')
            for i in range(1, len(self.model.ith_layer_nodes)):
                w = self.model.params[f'w{i}']
                for r in range(self.model.ith_layer_nodes[i - 1]):
                    f.write(','.join(map(str, list(w[r].flatten()))) + '\n')
                f.write(','.join(map(str, list(self.model.params[f'b{i}'].flatten()))) + '\n')
        return True

    def read(self, graph_name):
        filename = f'v_partial_model_params/{graph_name}.test_v_partial_model_params'
        if not os.path.isfile(filename):
            return False
        with open(filename, 'r') as f:
            # read in model hyperparameters and parameters resulted from trained model
            ith_layer_nodes = list(map(int, f.readline().rstrip().split(',')))
            temp_params = {}
            for i in range(1, len(ith_layer_nodes)):
                w = []
                for r in range(ith_layer_nodes[i - 1]):
                    w.append(list(map(float, f.readline().rstrip().split(','))))
                temp_params[f'w{i}'] = np.array(w).reshape((ith_layer_nodes[i - 1], ith_layer_nodes[i]))
                b = list(map(float, f.readline().rstrip().split(',')))
                temp_params[f'b{i}'] = np.array(b).reshape((ith_layer_nodes[i], 1))
            kwargs = {
                'train_mode': False,
                'params': temp_params,
                'ith_layer_nodes': ith_layer_nodes
            }
            # set up model with read in hyperparameters and parameters resulted from trained model
            self.model = NeuralNetwork(kwargs)
        return True

    def refresh(self, kwargs):
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']
        self.prey_transition_matrix = None
        self.prey_belief_vector = None

    def is_infinite_state(self, init_partial_state, graph):
        # check if a starting state is infinite (i.e., having infinite utility)
        min_partial_utility = float('Inf')
        agent_node_name = init_partial_state.agent_node_name
        next_node_name_list = [neighbor.name for neighbor in graph.adj_list[agent_node_name].neighbors]
        next_node_name_list.append(agent_node_name)
        # get agent current standing node and its neighbor's partial utility
        for node_name in next_node_name_list:
            kwargs = {
                'agent_node_name': node_name,
                'prey_belief_vector': self.prey_belief_vector,
                'predator_node_name': graph.predator.cur_node_name
            }
            partial_state = PartialState(kwargs)
            partial_utility = self.__get_partial_utility(graph.adj_list, partial_state)
            if partial_utility < min_partial_utility:
                min_partial_utility = partial_utility
        self.prey_transition_matrix = None
        self.prey_belief_vector = None
        self.model_mse = 0
        self.model_num_prediction = 0
        return min_partial_utility == float('Inf')

    def get_model_params(self, graph_name):
        raw_data_filename = f'v_partial_dataset/one_graph_raw_data_v_partial.csv'
        # construct raw data (include features and true label)
        self.__write_unigraph_raw_dataset(graph_name, raw_data_filename)
        # train model
        self.__train_model(raw_data_filename)
        # os.remove(raw_data_filename)

    def update_node_name(self, next_node_name):
        self.pre_node_name = self.cur_node_name
        self.cur_node_name = next_node_name

    def next_node_name(self, graph):
        min_great_utility = float('Inf')
        utility_action_choice_list = []
        next_node_name_list = [neighbor.name for neighbor in graph.adj_list[self.cur_node_name].neighbors]
        next_node_name_list.append(self.cur_node_name)
        # get agent current standing node and its neighbor's partial utility
        for node_name in next_node_name_list:
            kwargs = {
                'agent_node_name': node_name,
                'prey_belief_vector': self.prey_belief_vector,
                'predator_node_name': graph.predator.cur_node_name
            }
            partial_state = PartialState(kwargs)
            utility = self.__get_partial_utility(graph.adj_list, partial_state)
            if utility < min_great_utility:
                min_great_utility = utility
            utility_action_choice_list.append((utility, node_name))
        # get great action(s) for current state
        optimal_action_choice_list = []
        for u, a in utility_action_choice_list:
            if u == min_great_utility:
                optimal_action_choice_list.append(a)
        return random.choice(optimal_action_choice_list)

    def turn_state_into_model_input(self, agent_node_name, predator_node_name, prey_belief_vector):
        dist_pd = self.all_pairs_distance[agent_node_name][predator_node_name]
        dist_py_list = np.array([self.all_pairs_distance[agent_node_name][i]
                                for i in range(1, self.graph_size + 1)])
        expected_dist_py = np.dot(dist_py_list, prey_belief_vector[1:])

        dist_py_from_pd_list = np.array([self.all_pairs_distance[predator_node_name][i]
                                        for i in range(1, self.graph_size + 1)])
        expected_dist_py_from_pd = np.dot(dist_py_from_pd_list, prey_belief_vector[1:])

        # normalize distance data with min max scalar
        dist_pd, expected_dist_py = dist_pd / (self.graph_size - 1), expected_dist_py / (self.graph_size - 1)

        expected_dist_py_from_pd /= (self.graph_size - 1)

        data_list = []

        data_list.extend([1 if _ == agent_node_name else 0
                          for _ in range(1, self.graph_size + 1)])
        data_list.extend([1 if _ == predator_node_name else 0
                          for _ in range(1, self.graph_size + 1)])
        data_list.extend(prey_belief_vector[1:])

        data_list.extend([dist_pd, expected_dist_py, expected_dist_py_from_pd])
        return data_list

    def __write_unigraph_raw_dataset(self, graph_name, filename):
        if os.path.isfile(filename):
            return False
        with open(filename, 'w') as fout:
            # write out input data as raw data for later model training use
            headers = ['Id']
            headers.extend([f'Ag{i}' for i in range(1, self.graph_size + 1)])
            headers.extend([f'Pd{i}' for i in range(1, self.graph_size + 1)])
            headers.extend([f'Py{i}' for i in range(1, self.graph_size + 1)])
            headers.extend(['DistAgPd', 'EDistAgPy', 'EDistPyPd', 'U_partial'])
            # headers.append('U_partial')
            fout.write(','.join(headers) + '\n')
            with open(f'rsc/{graph_name}.agent_u_partial_utility_list', 'r') as fin:
                num_u_partial, = map(int, fin.readline().rstrip().split(','))
                for i in range(1, num_u_partial + 1):
                    agent_node_name, = map(int, fin.readline().rstrip().split(','))
                    predator_node_name, = map(int, fin.readline().rstrip().split(','))
                    prey_belief_vector = list(map(float, fin.readline().rstrip().split(',')))
                    partial_utility, = map(float, fin.readline().rstrip().split(','))
                    data_list = [i]
                    data_list.extend(self.turn_state_into_model_input(agent_node_name,
                                                                      predator_node_name,
                                                                      prey_belief_vector))
                    data_list.append(partial_utility)
                    fout.write(','.join(map(str, data_list)) + '\n')

    def __train_model(self, raw_data_filename):
        # implement supervised learning with neural network model to get optimal utility
        df = pd.read_csv(raw_data_filename, sep=',')
        # exclude data with infinite utility from being fed into model
        u_df = df[df['U_partial'] != float('Inf')]
        all_x = u_df.drop(columns=['U_partial'])
        # get all features
        all_x = all_x.drop(columns=['Id'])
        # get all true labels
        all_y = u_df['U_partial']
        num_features = all_x.shape[1]
        # split out test set
        x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.1, random_state=2)
        # split out train and validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2)
        # set up hyperparameter and then initialize model
        kwargs = {
            'train_mode': True,
            'learning_rate': 1e-2,
            'mse_threshold': 1,
            'ith_layer_nodes': [num_features, 1]
        }
        self.model = NeuralNetwork(kwargs)
        # train model with part of all data
        # self.model.fit(train_input_x=x_train.T, train_output_y=y_train.T, val_input_x=x_val.T, val_output_y=y_val.T)
        # train model with all data
        self.model.fit(train_input_x=all_x.T, train_output_y=all_y.T, val_input_x=None, val_output_y=None)
        self.model.plot_loss()

    def __get_partial_utility(self, graph_adj_list, partial_state):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            # where every probability is the same
            # except for the node's probability where agent initially stand on
            self.prey_belief_vector = np.array([1.0 / float(self.graph_size - 1) for _ in range(self.graph_size + 1)])
            self.prey_belief_vector[self.cur_node_name] = 0.0
            self.prey_belief_vector[0] = 0.0
        else:
            if graph_adj_list[self.cur_node_name].is_prey_existed():
                # adjust belief vector after knowing where agent current standing node has prey
                self.prey_belief_vector[self.cur_node_name] = 1.0
                for i in range(1, self.graph_size + 1):
                    if i != self.cur_node_name:
                        self.prey_belief_vector[i] = 0.0
            else:
                # adjust belief vector after knowing where agent current standing node has no prey
                self.prey_belief_vector[self.cur_node_name] = 0.0
                total_beliefs = 0.0
                for belief in self.prey_belief_vector:
                    total_beliefs += belief
                for i in range(len(self.prey_belief_vector)):
                    self.prey_belief_vector[i] = float(self.prey_belief_vector[i] / total_beliefs)
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix (from one round to another round)
            # since how prey move from one node to next node doesn't depend on where agent or predator is
            self.prey_transition_matrix = np.array([[0.0 for _ in range(self.graph_size + 1)]
                                                    for _ in range(self.graph_size + 1)])
            # assign probability in transition matrix by checking the quantity of a node's neighbors
            # since the prey chooses a node to move to uniformly at random
            # among its neighbors or its current standing node every time the prey moves
            for node in graph_adj_list[1:]:
                for neighbor in node.neighbors:
                    self.prey_transition_matrix[neighbor.name][node.name] = 1.0 / (1.0 + float(len(node.neighbors)))
                self.prey_transition_matrix[node.name][node.name] = 1.0 / (1.0 + float(len(node.neighbors)))
        else:
            # update belief of a prey being on a node for each node for this round
            # by using the previous round's belief vector and unchanged transition matrix (from any round to any round)
            self.prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # check if there's a node where prey certainly is
        if largest_belief == 1:
            # survey no node since it is certain where prey is
            return random.choice(largest_belief_node_names)
        # pick node with the largest belief to survey after knowing there's no node where prey certainly is
        node_name_to_survey = random.choice(largest_belief_node_names)
        node_to_survey = graph_adj_list[node_name_to_survey]
        is_prey_on_node = node_to_survey.is_prey_existed()
        if is_prey_on_node:
            # adjust belief vector after knowing the surveyed node has prey
            self.prey_belief_vector[node_name_to_survey] = 1.0
            for i in range(1, self.graph_size + 1):
                if i != node_name_to_survey:
                    self.prey_belief_vector[i] = 0.0
        else:
            # adjust belief vector after knowing the surveyed node has no prey
            self.prey_belief_vector[node_name_to_survey] = 0.0
            total_beliefs = 0.0
            for belief in self.prey_belief_vector:
                total_beliefs += belief
            for i in range(len(self.prey_belief_vector)):
                self.prey_belief_vector[i] = float(self.prey_belief_vector[i] / total_beliefs)

        # compute reward
        reward = float('Inf') if partial_state.agent_node_name == partial_state.predator_node_name else 1
        # compute the number of predator's neighbors closet to agent
        predator_shortest_distance_neighbor_count = 0
        shortest_distance = float('Inf')
        for predator_neighbor in graph_adj_list[partial_state.predator_node_name].neighbors:
            dist_agent_to_predator_neighbor = \
                self.all_pairs_distance[partial_state.agent_node_name][predator_neighbor.name]
            if dist_agent_to_predator_neighbor < shortest_distance:
                shortest_distance = dist_agent_to_predator_neighbor
        for predator_neighbor in graph_adj_list[partial_state.predator_node_name].neighbors:
            dist_agent_to_predator_neighbor = \
                self.all_pairs_distance[partial_state.agent_node_name][predator_neighbor.name]
            if dist_agent_to_predator_neighbor == shortest_distance:
                predator_shortest_distance_neighbor_count += 1
        # compute expected next state u partial
        expected_next_state_u_partial = 0
        future_prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
        predator_neighbor_names = [predator_neighbor.name for predator_neighbor
                                   in graph_adj_list[partial_state.predator_node_name].neighbors]
        for predator_neighbor_name in predator_neighbor_names:
            # compute transition probability
            predator_transition_probability = 0.4 * \
                                              (1 / len(graph_adj_list[partial_state.predator_node_name].neighbors))
            if self.all_pairs_distance[partial_state.agent_node_name][predator_neighbor_name] \
                    == shortest_distance:
                predator_transition_probability += 0.6 * (
                        1 / predator_shortest_distance_neighbor_count)
            # compute u partial
            data_list = self.turn_state_into_model_input(partial_state.agent_node_name,
                                                         predator_neighbor_name,
                                                         future_prey_belief_vector)
            predicted_partial_utility = self.model.predict(data_list).flatten()[0]
            expected_next_state_u_partial += predator_transition_probability * predicted_partial_utility

            # compute error between predicted partial utility and
            # true partial utility for later accuracy analysis
            # (note that state with infinite utility is excluded)
            true_partial_utility = self.__get_true_label(partial_state,
                                                         predator_neighbor_name,
                                                         future_prey_belief_vector)
            if true_partial_utility != float('Inf'):
                self.model_mse += (predicted_partial_utility - true_partial_utility) ** 2
                self.model_num_prediction += 1

        return reward + expected_next_state_u_partial

    def __get_true_label(self, partial_state, predator_neighbor_name, future_prey_belief_vector):
        # compute true partial utility with the formula on the spec
        partial_utility = 0.0
        for i in range(1, self.graph_size + 1):
            kwargs = {
                'agent_node_name': partial_state.agent_node_name,
                'prey_node_name': i,
                'predator_node_name': predator_neighbor_name,
                'graph_size': self.graph_size
            }
            state = State(kwargs)
            if future_prey_belief_vector[i] != 0:
                if self.utility_star_dict[state] != float('Inf'):
                    partial_utility += future_prey_belief_vector[i] * self.utility_star_dict[state]
                else:
                    partial_utility = float('Inf')
                    break
        return partial_utility

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
