import numpy as np
import os

from util import *
from model import *


class AgentUPartial:
    def __init__(self, kwargs):
        self.name = 'agent_u_partial'
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']
        self.graph_size = kwargs['graph_size']
        self.all_pairs_distance = kwargs['graph_all_pairs_distance']
        graph_name = kwargs['graph_name']
        self.prey_transition_matrix = None
        self.prey_belief_vector = None
        # read in U*
        # as dictionary with form of {state: utility*}
        self.utility_star_dict = {}
        self.__read_u_star(graph_name)
        # store partial utility and action corresponding to each state
        # in a list with form of (state, partial utility, action)
        self.partial_utility_list = []

    def write(self, graph_name):
        filename = f'rsc/{graph_name}.agent_u_partial_utility_list'
        if os.path.isfile(filename):
            return False
        with open(filename, 'w') as f:
            # write out dictionary with form of {state s : (utility, [action1, action2, ...])}
            f.write(f'{len(self.partial_utility_list)}\n')
            for partial_state, partial_utility in self.partial_utility_list:
                partial_state.write(f)
                f.write(f'{partial_utility}\n')
        return True

    def refresh(self, kwargs):
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']
        self.prey_transition_matrix = None
        self.prey_belief_vector = None

    def is_infinite_state(self, init_partial_state, graph):
        # check if a starting state is infinite (i.e., having infinite utility)
        min_great_utility = float('Inf')
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
            utility = self.__get_partial_utility(graph.adj_list, partial_state)
            if utility < min_great_utility:
                min_great_utility = utility
        self.prey_transition_matrix = None
        self.prey_belief_vector = None
        return min_great_utility == float('Inf')

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
        great_action_choice_list = []
        for u, a in utility_action_choice_list:
            if u == min_great_utility:
                great_action_choice_list.append(a)
        return random.choice(great_action_choice_list)

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
                                              (1 /
                                               len(graph_adj_list[partial_state.predator_node_name].neighbors))
            if self.all_pairs_distance[partial_state.agent_node_name][predator_neighbor_name] \
                    == shortest_distance:
                predator_transition_probability += 0.6 * (
                        1 / predator_shortest_distance_neighbor_count)
            # compute u partial
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
            kwargs = {
                'agent_node_name': partial_state.agent_node_name,
                'predator_node_name': predator_neighbor_name,
                'prey_belief_vector': future_prey_belief_vector
            }
            next_partial_state = PartialState(kwargs)
            self.partial_utility_list.append((next_partial_state, partial_utility))
            expected_next_state_u_partial += predator_transition_probability * partial_utility
        return reward + expected_next_state_u_partial


class AgentAnotherUPartial:
    def __init__(self, kwargs):
        self.name = 'agent_another_u_partial'
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']
        self.graph_size = kwargs['graph_size']
        self.all_pairs_distance = kwargs['graph_all_pairs_distance']
        graph_name = kwargs['graph_name']
        self.prey_transition_matrix = None
        self.prey_belief_vector = None
        # used to store model
        self.model_v = None
        # used to determine another U partial accuracy
        # (here calling self.model is for convenience)
        self.model_mse = 0
        self.model_num_prediction = 0
        self.utility_star_dict = {}
        self.__read_u_star(graph_name)
        self.__read_v_model_params(graph_name)
        # store partial utility and action corresponding to each state
        # in a list with form of (state, partial utility, action)
        self.partial_utility_list = []

    # to be completed in the future
    def write(self, graph_name):
        filename = f'rsc/{graph_name}.agent_another_u_partial_utility_list'
        if os.path.isfile(filename):
            return False
        with open(filename, 'w') as f:
            # write out dictionary with form of {state s : (utility, [action1, action2, ...])}
            f.write(f'{len(self.partial_utility_list)}\n')
            for partial_state, partial_utility, _ in self.partial_utility_list:
                partial_state.write(f)
                f.write(f'{partial_utility}\n')
        return True

    def refresh(self, kwargs):
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']
        self.prey_transition_matrix = None
        self.prey_belief_vector = None

    def is_infinite_state(self, init_partial_state, graph):
        # check if a starting state is infinite (i.e., having infinite utility)
        min_great_utility = float('Inf')
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
            utility = self.__get_partial_utility(graph.adj_list, partial_state)
            if utility < min_great_utility:
                min_great_utility = utility
        self.prey_transition_matrix = None
        self.prey_belief_vector = None
        return min_great_utility == float('Inf')

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
        great_action_choice_list = []
        for u, a in utility_action_choice_list:
            if u == min_great_utility:
                great_action_choice_list.append(a)
        return random.choice(great_action_choice_list)

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

    def __read_v_model_params(self, graph_name):
        filename = f'v_model_params/{graph_name}.v_model_params'
        if not os.path.isfile(filename):
            return False
        with open(filename, 'r') as f:
            # read in model parameters resulted from trained model V
            temp_params = {}
            w_list = list(map(float, f.readline().rstrip().split(',')))
            temp_params['w'] = np.array(w_list).reshape((len(w_list), 1))
            b_list = list(map(float, f.readline().rstrip().split(',')))
            temp_params['b'] = np.array(b_list).reshape((len(b_list), 1))
            kwargs = {
                'train_mode': False,
                'params': temp_params
            }
            # set up model with read in parameters resulted from trained model V
            self.model_v = LinearRegression(kwargs)
        return True

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
                                              (1 /
                                               len(graph_adj_list[partial_state.predator_node_name].neighbors))
            if self.all_pairs_distance[partial_state.agent_node_name][predator_neighbor_name] \
                    == shortest_distance:
                predator_transition_probability += 0.6 * (
                        1 / predator_shortest_distance_neighbor_count)
            # compute u partial
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
                    data_list = self.turn_state_into_model_input(state.agent_node_name,
                                                                 i,
                                                                 predator_neighbor_name)
                    predicted_utility = self.model_v.predict(data_list).flatten()[0]
                    partial_utility += future_prey_belief_vector[i] * predicted_utility
            kwargs = {
                'agent_node_name': partial_state.agent_node_name,
                'predator_node_name': predator_neighbor_name,
                'prey_belief_vector': future_prey_belief_vector
            }
            next_partial_state = PartialState(kwargs)
            self.partial_utility_list.append((next_partial_state, partial_utility))
            expected_next_state_u_partial += predator_transition_probability * partial_utility

            # compute error between predicted partial utility and
            # true partial utility for later accuracy analysis
            # (note that state with infinite utility is excluded)
            true_partial_utility = self.__get_true_label(partial_state,
                                                         predator_neighbor_name,
                                                         future_prey_belief_vector)
            if true_partial_utility != float('Inf'):
                self.model_mse += (partial_utility - true_partial_utility) ** 2
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
                agent_node_name, prey_node_name, predator_node_name = \
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
