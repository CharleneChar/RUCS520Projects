import numpy as np
import os

from graph import *
from util import *


class AgentOdd:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.name = None
            self.pre_node_name = None
            self.cur_node_name = kwargs['init_node_name']
            self.all_pairs_distance = kwargs['graph_all_pairs_distance']

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.name = self.name
        d_copy.pre_node_name = deepcopy(self.pre_node_name)
        d_copy.cur_node_name = deepcopy(self.cur_node_name)
        d_copy.all_pairs_distance = deepcopy(self.all_pairs_distance)
        return d_copy

    def refresh(self, kwargs):
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']

    def update_node_name(self, next_node_name):
        self.pre_node_name = self.cur_node_name
        self.cur_node_name = next_node_name

    def get_next_node_name_by_rules(self, graph, prey_node_name):
        cur_node = graph.adj_list[self.cur_node_name]

        # debug by printing out corresponding current distance from agent to prey, and to predator
        if DEBUG > 0:
            for neighbor in graph.adj_list[graph.agent.cur_node_name].neighbors:
                print(f'{neighbor.name} to prey distance:'
                      f' {self.all_pairs_distance[prey_node_name][neighbor.name]}')
                print(f'{neighbor.name} to predator distance: '
                      f'{self.all_pairs_distance[graph.predator.cur_node_name][neighbor.name]}')

        # record neighbors' info about their distances to prey, their distances to predator,
        # a random number which is for breaking ties, and their names
        # (as tuples in a list, called neighbors' information list in lab report)
        neighbors_to_prey_and_predator_info = []
        # record min and max of all shortest distances to predator (for all neighbors of the agent)
        # record min and max of all shortest distances to prey (for all neighbors of the agent)
        min_prey_distance = min_predator_distance = float('Inf')
        max_prey_distance = max_predator_distance = -1
        random_num_list = list(range(1, len(cur_node.neighbors) + 1))
        random.shuffle(random_num_list)
        for i, neighbor in enumerate(cur_node.neighbors):
            dist_prey_to_agent_neighbor = self.all_pairs_distance[prey_node_name][neighbor.name]
            if dist_prey_to_agent_neighbor < min_prey_distance:
                min_prey_distance = dist_prey_to_agent_neighbor
            if dist_prey_to_agent_neighbor > max_prey_distance:
                max_prey_distance = dist_prey_to_agent_neighbor
            dist_predator_to_agent_neighbor = self.all_pairs_distance[graph.predator.cur_node_name][neighbor.name]
            if dist_predator_to_agent_neighbor < min_predator_distance:
                min_predator_distance = dist_predator_to_agent_neighbor
            if dist_predator_to_agent_neighbor > max_predator_distance:
                max_predator_distance = dist_predator_to_agent_neighbor
            neighbors_to_prey_and_predator_info.append(
                (dist_prey_to_agent_neighbor, dist_predator_to_agent_neighbor,
                 random_num_list[i], neighbor.name))

        # get used for classifying up to three classes of neighbors:
        # those closer to the prey,
        # those not closer and not farther from the prey,
        # and those farther from the prey
        min_end_index = not_min_max_end_index = len(neighbors_to_prey_and_predator_info)
        neighbors_to_prey_and_predator_info.sort()
        for i, info in enumerate(neighbors_to_prey_and_predator_info):
            if info[0] != min_prey_distance:
                min_end_index = i
                break
        for i, info in enumerate(neighbors_to_prey_and_predator_info):
            if info[0] == max_prey_distance:
                not_min_max_end_index = i
                break

        # examine neighbors based on rules (for choosing which node to move to)
        for info in neighbors_to_prey_and_predator_info[:min_end_index]:
            # choose neighbor closer to prey and farther from predator
            neighbor_to_prey_distance, neighbor_to_predator_distance, _, neighbor_name = info
            if neighbor_to_predator_distance == max_predator_distance:
                return neighbor_name
        # choose neighbor closer to prey and not closer to predator
        for info in neighbors_to_prey_and_predator_info[:min_end_index]:
            neighbor_to_prey_distance, neighbor_to_predator_distance, _, neighbor_name = info
            if neighbor_to_predator_distance != min_predator_distance:
                return neighbor_name
        # choose neighbor not farther from prey and farther from predator
        for info in neighbors_to_prey_and_predator_info[min_end_index:not_min_max_end_index]:
            neighbor_to_prey_distance, neighbor_to_predator_distance, _, neighbor_name = info
            if neighbor_to_predator_distance == max_predator_distance:
                return neighbor_name
        # choose neighbor not farther from prey and not closer to predator
        for info in neighbors_to_prey_and_predator_info[min_end_index:not_min_max_end_index]:
            neighbor_to_prey_distance, neighbor_to_predator_distance, _, neighbor_name = info
            if neighbor_to_predator_distance != min_predator_distance:
                return neighbor_name
        # choose neighbor farther from predator
        for info in neighbors_to_prey_and_predator_info[not_min_max_end_index:]:
            neighbor_to_prey_distance, neighbor_to_predator_distance, _, neighbor_name = info
            if neighbor_to_predator_distance == max_predator_distance:
                return neighbor_name
        # choose neighbor not closer to predator
        for info in neighbors_to_prey_and_predator_info[not_min_max_end_index:]:
            neighbor_to_prey_distance, neighbor_to_predator_distance, _, neighbor_name = info
            if neighbor_to_predator_distance != min_predator_distance:
                return neighbor_name
        # choose to sit still (at current node) and pray
        return self.cur_node_name


class AgentEven:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.name = None
            self.pre_node_name = None
            self.cur_node_name = kwargs['init_node_name']
            self.all_pairs_distance = kwargs['graph_all_pairs_distance']

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.name = self.name
        d_copy.pre_node_name = deepcopy(self.pre_node_name)
        d_copy.cur_node_name = deepcopy(self.cur_node_name)
        d_copy.all_pairs_distance = deepcopy(self.all_pairs_distance)
        return d_copy

    def refresh(self, kwargs):
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']

    def update_node_name(self, next_node_name):
        self.pre_node_name = self.cur_node_name
        self.cur_node_name = next_node_name

    def get_next_node_name(self, graph, prey_node_name, predator_node_name):
        cur_node = graph.adj_list[self.cur_node_name]
        predator_and_its_neighbor_node_names = {predator_node_name}
        for predator_neighbor in graph.adj_list[predator_node_name].neighbors:
            predator_and_its_neighbor_node_names.add(predator_neighbor.name)
        source_names = {cur_node.name}
        for node in cur_node.neighbors:
            source_names.add(node.name)
        # exclude the neighbors (of the agent) or the agent's current standing node
        # who was one neighbor of the predator or was just the predator
        candidate_source_names = source_names.difference(predator_and_its_neighbor_node_names)
        if len(candidate_source_names) == 0:
            candidate_source_names = source_names.difference({predator_node_name})
        min_total_distance, min_node_name = float('Inf'), None
        prey_and_its_neighbor_node_names = {prey_node_name}
        for prey_neighbor in graph.adj_list[prey_node_name].neighbors:
            prey_and_its_neighbor_node_names.add(prey_neighbor.name)
        for candidate_source_name in candidate_source_names:
            total_distance = 0
            # computing the total distance to the prey's neighborhood
            # (for each of the agent's neighbor and for the agent's current standing node)
            for node_name in prey_and_its_neighbor_node_names:
                total_distance += self.all_pairs_distance[candidate_source_name][node_name]
            # choose the node (from the agent current standing node and the agent's neighbors)
            # that wasn't excluded for not having the shortest distance to the prey's neighborhood
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                min_node_name = candidate_source_name
        return min_node_name


class Agent1(AgentOdd):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent1'
            self.graph_size = kwargs['graph_size']
            graph_name = kwargs['graph_name']
            if not self.read(graph_name):
                # store action corresponding to each state
                # in a dictionary with form of {state: [action1, action2, ...]}
                self.action_dict = {}
                # get action for each state
                self.__get_action_dict(kwargs)
                self.write(graph_name)
            # read in U*
            # as dictionary with form of {state: utility*}
            self.utility_star_dict = {}
            self.__read_u_star(graph_name)

    def write(self, graph_name):
        filename = f'rsc/{graph_name}.agent1_policy_dict'
        if os.path.isfile(filename):
            return False
        with open(filename, 'w') as f:
            # write out dictionary with form of {state s : [action1, action2, ...]}
            for state, action in self.action_dict.items():
                f.write(f'{state}\n')
                f.write(','.join(map(str, action)) + '\n')
        return True

    def read(self, graph_name):
        filename = f'rsc/{graph_name}.agent1_policy_dict'
        if not os.path.isfile(filename):
            return False
        with open(filename, 'r') as f:
            # read in database as dictionary
            # with the form of {state s: [action1, action2, ...]}
            self.action_dict = {}
            for i in range(self.graph_size ** 3):
                agent_node_name, prey_node_name, predator_node_name = \
                    map(int, f.readline().rstrip().split(','))
                kwargs = {
                    'agent_node_name': agent_node_name,
                    'prey_node_name': prey_node_name,
                    'predator_node_name': predator_node_name,
                    'graph_size': self.graph_size
                }
                state = State(kwargs)
                line = f.readline().rstrip()
                if line != '':
                    action = list(map(int, line.split(',')))
                else:
                    action = []
                self.action_dict[state] = action
        return True

    def is_infinite_state(self, init_state, graph):
        # check if a starting state is infinite (i.e., having infinite utility)
        return self.utility_star_dict[init_state] == float('Inf')

    def refresh(self, kwargs):
        super().refresh(kwargs)

    def next_node_name(self, graph):
        kwargs = {
            'agent_node_name': self.cur_node_name,
            'prey_node_name': graph.prey.cur_node_name,
            'predator_node_name': graph.predator.cur_node_name,
            'graph_size': self.graph_size
        }
        state = State(kwargs)
        return random.choice(self.action_dict[state])

    def __get_action_dict(self, kwargs):
        graph_size = kwargs['graph_size']
        graph = kwargs['graph']
        for agent_node_name in range(1, graph_size + 1):
            for prey_node_name in range(1, graph_size + 1):
                for predator_node_name in range(1, graph_size + 1):
                    temp_kwargs = {
                        'agent_node_name': agent_node_name,
                        'prey_node_name': prey_node_name,
                        'predator_node_name': predator_node_name,
                        'graph_size': graph_size
                    }
                    state = State(temp_kwargs)
                    if agent_node_name == prey_node_name:
                        self.action_dict[state] = []
                    else:
                        self.action_dict[state] = self.__get_action_list(graph, state)

    def __get_action_list(self, graph, state):
        action_list = []
        cur_node = graph.adj_list[state.agent_node_name]

        # record neighbors' info about their distances to prey, their distances to predator,
        # a random number which is for breaking ties, and their names
        # (as tuples in a list, called neighbors' information list in lab report)
        neighbors_to_prey_and_predator_info = []
        # record min and max of all shortest distances to predator (for all neighbors of the agent)
        # record min and max of all shortest distances to prey (for all neighbors of the agent)
        min_prey_distance = min_predator_distance = float('Inf')
        max_prey_distance = max_predator_distance = -1
        for i, neighbor in enumerate(cur_node.neighbors):
            dist_prey_to_agent_neighbor = self.all_pairs_distance[state.prey_node_name][neighbor.name]
            if dist_prey_to_agent_neighbor < min_prey_distance:
                min_prey_distance = dist_prey_to_agent_neighbor
            if dist_prey_to_agent_neighbor > max_prey_distance:
                max_prey_distance = dist_prey_to_agent_neighbor
            dist_predator_to_agent_neighbor = self.all_pairs_distance[state.predator_node_name][neighbor.name]
            if dist_predator_to_agent_neighbor < min_predator_distance:
                min_predator_distance = dist_predator_to_agent_neighbor
            if dist_predator_to_agent_neighbor > max_predator_distance:
                max_predator_distance = dist_predator_to_agent_neighbor
            neighbors_to_prey_and_predator_info.append(
                (dist_prey_to_agent_neighbor, dist_predator_to_agent_neighbor,
                 neighbor.name))

        # get used for classifying up to three classes of neighbors:
        # those closer to the prey,
        # those not closer and not farther from the prey,
        # and those farther from the prey
        min_end_index = not_min_max_end_index = len(neighbors_to_prey_and_predator_info)
        neighbors_to_prey_and_predator_info.sort()
        for i, info in enumerate(neighbors_to_prey_and_predator_info):
            if info[0] != min_prey_distance:
                min_end_index = i
                break
        for i, info in enumerate(neighbors_to_prey_and_predator_info):
            if info[0] == max_prey_distance:
                not_min_max_end_index = i
                break

        # examine neighbors based on rules (for choosing which node to move to)
        for info in neighbors_to_prey_and_predator_info[:min_end_index]:
            # choose neighbor closer to prey and farther from predator
            neighbor_to_prey_distance, neighbor_to_predator_distance, neighbor_name = info
            if neighbor_to_predator_distance == max_predator_distance:
                action_list.append(neighbor_name)
        if len(action_list):
            return action_list
        # choose neighbor closer to prey and not closer to predator
        for info in neighbors_to_prey_and_predator_info[:min_end_index]:
            neighbor_to_prey_distance, neighbor_to_predator_distance, neighbor_name = info
            if neighbor_to_predator_distance != min_predator_distance:
                action_list.append(neighbor_name)
        if len(action_list):
            return action_list
        # choose neighbor not farther from prey and farther from predator
        for info in neighbors_to_prey_and_predator_info[min_end_index:not_min_max_end_index]:
            neighbor_to_prey_distance, neighbor_to_predator_distance, neighbor_name = info
            if neighbor_to_predator_distance == max_predator_distance:
                action_list.append(neighbor_name)
        if len(action_list):
            return action_list
        # choose neighbor not farther from prey and not closer to predator
        for info in neighbors_to_prey_and_predator_info[min_end_index:not_min_max_end_index]:
            neighbor_to_prey_distance, neighbor_to_predator_distance, neighbor_name = info
            if neighbor_to_predator_distance != min_predator_distance:
                action_list.append(neighbor_name)
        if len(action_list):
            return action_list
        # choose neighbor farther from predator
        for info in neighbors_to_prey_and_predator_info[not_min_max_end_index:]:
            neighbor_to_prey_distance, neighbor_to_predator_distance, neighbor_name = info
            if neighbor_to_predator_distance == max_predator_distance:
                action_list.append(neighbor_name)
        if len(action_list):
            return action_list
        # choose neighbor not closer to predator
        for info in neighbors_to_prey_and_predator_info[not_min_max_end_index:]:
            neighbor_to_prey_distance, neighbor_to_predator_distance, neighbor_name = info
            if neighbor_to_predator_distance != min_predator_distance:
                action_list.append(neighbor_name)
        if len(action_list):
            return action_list
        # choose to sit still (at current node) and pray
        return [cur_node.name]

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


class Agent2(AgentEven):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent2'
            self.graph_size = kwargs['graph_size']
            graph_name = kwargs['graph_name']
            if not self.read(graph_name):
                # store action corresponding to each state
                # in a dictionary with form of {state: [action1, action2, ...]}
                self.action_dict = {}
                # get action for each state
                self.__get_action_dict(kwargs)
                self.write(graph_name)
            # read in U*
            # as dictionary with form of {state: utility*}
            self.utility_star_dict = {}
            self.__read_u_star(graph_name)

    def write(self, graph_name):
        filename = f'rsc/{graph_name}.agent2_policy_dict'
        if os.path.isfile(filename):
            return False
        with open(filename, 'w') as f:
            # write out dictionary with form of {state s : [action1, action2, ...]}
            for state, action in self.action_dict.items():
                f.write(f'{state}\n')
                f.write(','.join(map(str, action)) + '\n')
        return True

    def read(self, graph_name):
        filename = f'rsc/{graph_name}.agent2_policy_dict'
        if not os.path.isfile(filename):
            return False
        with open(filename, 'r') as f:
            # read in database as dictionary
            # with form of {state s: [action1, action2, ...]}
            self.action_dict = {}
            for i in range(self.graph_size ** 3):
                agent_node_name, prey_node_name, predator_node_name = \
                    map(int, f.readline().rstrip().split(','))
                kwargs = {
                    'agent_node_name': agent_node_name,
                    'prey_node_name': prey_node_name,
                    'predator_node_name': predator_node_name,
                    'graph_size': self.graph_size
                }
                state = State(kwargs)
                line = f.readline().rstrip()
                if line != '':
                    action = list(map(int, line.split(',')))
                else:
                    action = []
                self.action_dict[state] = action
        return True

    def is_infinite_state(self, init_state, graph):
        # check if a starting state is infinite (i.e., having infinite utility)
        return self.utility_star_dict[init_state] == float('Inf')

    def refresh(self, kwargs):
        super().refresh(kwargs)

    def next_node_name(self, graph):
        kwargs = {
            'agent_node_name': self.cur_node_name,
            'prey_node_name': graph.prey.cur_node_name,
            'predator_node_name': graph.predator.cur_node_name,
            'graph_size': self.graph_size
        }
        state = State(kwargs)
        return random.choice(self.action_dict[state])

    def __get_action_dict(self, kwargs):
        graph_size = kwargs['graph_size']
        graph = kwargs['graph']
        for agent_node_name in range(1, graph_size + 1):
            for prey_node_name in range(1, graph_size + 1):
                for predator_node_name in range(1, graph_size + 1):
                    temp_kwargs = {
                        'agent_node_name': agent_node_name,
                        'prey_node_name': prey_node_name,
                        'predator_node_name': predator_node_name,
                        'graph_size': graph_size
                    }
                    state = State(temp_kwargs)
                    if agent_node_name == prey_node_name:
                        self.action_dict[state] = []
                    else:
                        self.action_dict[state] = self.__get_action_list(graph, state)

    def __get_action_list(self, graph, state):
        cur_node = graph.adj_list[state.agent_node_name]

        predator_and_its_neighbor_node_names = {state.predator_node_name}
        for predator_neighbor in graph.adj_list[state.predator_node_name].neighbors:
            predator_and_its_neighbor_node_names.add(predator_neighbor.name)
        source_names = {cur_node.name}
        for node in cur_node.neighbors:
            source_names.add(node.name)
        # exclude the neighbors (of the agent) or the agent's current standing node
        # who was one neighbor of the predator or was just the predator
        candidate_source_names = source_names.difference(predator_and_its_neighbor_node_names)
        if len(candidate_source_names) == 0:
            candidate_source_names = source_names.difference({state.predator_node_name})
        min_distance = float('Inf')
        min_node_dis_and_names = []
        distances_from_neighbor_to_prey = []
        # compute the distance to the prey
        # (for each of the agent's neighbor and for the agent's current standing node)
        for candidate_source_name in candidate_source_names:
            if self.all_pairs_distance[candidate_source_name][state.prey_node_name] < min_distance:
                min_distance = self.all_pairs_distance[candidate_source_name][state.prey_node_name]
            distances_from_neighbor_to_prey.append(self.all_pairs_distance[candidate_source_name][state.prey_node_name])
        for i, candidate_source_name in enumerate(candidate_source_names):
            if distances_from_neighbor_to_prey[i] == min_distance:
                min_node_dis_and_names.append(candidate_source_name)
        # choose the node (from the agent current standing node and the agent's neighbors)
        # that wasn't excluded for not having the shortest distance to the prey
        return min_node_dis_and_names

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


class Agent3(AgentOdd):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent3'
            self.prey_transition_matrix = None
            self.prey_belief_vector = None
            self.graph_size = kwargs['graph_size']
            graph_name = kwargs['graph_name']
            # read in U*
            # as dictionary with form of {state: utility*}
            self.utility_star_dict = {}
            self.__read_u_star(graph_name)

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.prey_transition_matrix = deepcopy(self.prey_transition_matrix)
        d_copy.prey_belief_vector = deepcopy(self.prey_belief_vector)
        return d_copy

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

    def refresh(self, kwargs):
        super().refresh(kwargs)
        self.prey_transition_matrix = None
        self.prey_belief_vector = None

    def next_node_name(self, graph):
        # get the likely location of the prey
        likely_prey_node_name = self.get_likely_prey_node_name(graph)
        # get next node for agent to move to after following some rule
        return self.get_next_node_name_by_rules(graph, likely_prey_node_name)

    def get_likely_prey_node_name(self, graph):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            # where every probability is the same
            # except for the node's probability where agent initially stand on
            self.prey_belief_vector = np.array([1.0 / float(graph.size - 1) for _ in range(graph.size + 1)])
            self.prey_belief_vector[self.cur_node_name] = 0.0
            self.prey_belief_vector[0] = 0.0
        else:
            if graph.adj_list[self.cur_node_name].is_prey_existed():
                # adjust belief vector after knowing where agent current standing node has prey
                self.prey_belief_vector[self.cur_node_name] = 1.0
                for i in range(1, graph.size + 1):
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
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            # assign probability in transition matrix by checking the quantity of a node's neighbors
            # since the prey chooses a node to move to uniformly at random
            # among its neighbors or its current standing node every time the prey moves
            for node in graph.adj_list[1:]:
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
        node_to_survey = graph.adj_list[node_name_to_survey]
        is_prey_on_node = node_to_survey.is_prey_existed()
        if is_prey_on_node:
            # adjust belief vector after knowing the surveyed node has prey
            self.prey_belief_vector[node_name_to_survey] = 1.0
            for i in range(1, graph.size + 1):
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

        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if
                                     belief == largest_belief]
        # pick node with the largest belief for treating it as where prey is
        # (i.e., the likely location of the prey) for this round
        return random.choice(largest_belief_node_names)

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
            expected_next_state_u_partial += predator_transition_probability * partial_utility
        return reward + expected_next_state_u_partial


class Agent4(AgentEven):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent4'
            self.prey_transition_matrix = None
            self.prey_belief_vector = None
            self.graph_size = kwargs['graph_size']
            graph_name = kwargs['graph_name']
            # read in U*
            # as dictionary with form of {state: utility*}
            self.utility_star_dict = {}
            self.__read_u_star(graph_name)

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.prey_transition_matrix = deepcopy(self.prey_transition_matrix)
        d_copy.prey_belief_vector = deepcopy(self.prey_belief_vector)
        return d_copy

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

    def refresh(self, kwargs):
        super().refresh(kwargs)
        self.prey_transition_matrix = None
        self.prey_belief_vector = None

    def next_node_name(self, graph):
        # get the likely location of the prey
        likely_prey_node_name = self.get_likely_prey_node_name(graph)
        # get next node for agent to move to after following some rule
        return self.get_next_node_name(graph=graph, prey_node_name=likely_prey_node_name,
                                       predator_node_name=graph.predator.cur_node_name)

    def get_likely_prey_node_name(self, graph):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            # where every probability is the same
            # except for the node's probability where agent initially stand on
            self.prey_belief_vector = np.array([1.0 / float(graph.size - 1) for _ in range(graph.size + 1)])
            self.prey_belief_vector[self.cur_node_name] = 0.0
            self.prey_belief_vector[0] = 0.0
        else:
            if graph.adj_list[self.cur_node_name].is_prey_existed():
                # adjust belief vector after knowing where agent current standing node has prey
                self.prey_belief_vector[self.cur_node_name] = 1.0
                for i in range(1, graph.size + 1):
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
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            # assign probability in transition matrix by checking the quantity of a node's neighbors
            # since the prey chooses a node to move to uniformly at random
            # among its neighbors or its current standing node every time the prey moves
            for node in graph.adj_list[1:]:
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
        node_to_survey = graph.adj_list[node_name_to_survey]
        is_prey_on_node = node_to_survey.is_prey_existed()
        if is_prey_on_node:
            # update belief vector after knowing the surveyed node has prey
            self.prey_belief_vector[node_name_to_survey] = 1.0
            for i in range(1, graph.size + 1):
                if i != node_name_to_survey:
                    self.prey_belief_vector[i] = 0.0
        else:
            # update belief vector after knowing the surveyed node has no prey
            self.prey_belief_vector[node_name_to_survey] = 0.0
            total_beliefs = 0.0
            for belief in self.prey_belief_vector:
                total_beliefs += belief
            for i in range(len(self.prey_belief_vector)):
                self.prey_belief_vector[i] = float(self.prey_belief_vector[i] / total_beliefs)

        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in
                                     enumerate(self.prey_belief_vector) if belief == largest_belief]

        # pick node with the largest belief for treating it as where prey is
        # (i.e., the likely location of the prey) for this round
        return random.choice(largest_belief_node_names)

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
            expected_next_state_u_partial += predator_transition_probability * partial_utility
        return reward + expected_next_state_u_partial


class Agent4WithFuturePrediction(AgentEven):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent4_with_future_prediction'
            self.prey_transition_matrix = None
            self.prey_belief_vector = None
            self.future_prey_belief_vector = None
            self.graph_size = kwargs['graph_size']
            graph_name = kwargs['graph_name']
            # read in U*
            # as dictionary with form of {state: utility*}
            self.utility_star_dict = {}
            self.__read_u_star(graph_name)

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.prey_transition_matrix = deepcopy(self.prey_transition_matrix)
        d_copy.prey_belief_vector = deepcopy(self.prey_belief_vector)
        d_copy.future_prey_belief_vector = deepcopy(self.future_prey_belief_vector)
        return d_copy

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

    def refresh(self, kwargs):
        super().refresh(kwargs)
        self.prey_transition_matrix = None
        self.prey_belief_vector = None
        self.future_prey_belief_vector = None

    def next_node_name(self, graph):
        # get the likely location of the prey
        likely_prey_node_name = self.get_likely_prey_node_name(graph)
        # get next node for agent to move to after following some rule
        return self.get_next_node_name(graph=graph, prey_node_name=likely_prey_node_name,
                                       predator_node_name=graph.predator.cur_node_name)

    def get_likely_prey_node_name(self, graph):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            # where every probability is the same
            # except for the node's probability where agent initially stand on
            self.prey_belief_vector = np.array([1.0 / float(graph.size - 1) for _ in range(graph.size + 1)])
            self.prey_belief_vector[self.cur_node_name] = 0.0
            self.prey_belief_vector[0] = 0.0
        else:
            if graph.adj_list[self.cur_node_name].is_prey_existed():
                # adjust belief vector after knowing where agent current standing node has prey
                self.prey_belief_vector[self.cur_node_name] = 1.0
                for i in range(1, graph.size + 1):
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
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            # assign probability in transition matrix by checking the quantity of a node's neighbors
            # since the prey chooses a node to move to uniformly at random
            # among its neighbors or its current standing node every time the prey moves
            for node in graph.adj_list[1:]:
                for neighbor in node.neighbors:
                    self.prey_transition_matrix[neighbor.name][node.name] = 1.0 / (1.0 + float(len(node.neighbors)))
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
        node_to_survey = graph.adj_list[node_name_to_survey]
        is_prey_on_node = node_to_survey.is_prey_existed()
        if is_prey_on_node:
            # update belief vector after knowing the surveyed node has prey
            self.prey_belief_vector[node_name_to_survey] = 1.0
            for i in range(1, graph.size + 1):
                if i != node_name_to_survey:
                    self.prey_belief_vector[i] = 0.0
        else:
            # update belief vector after knowing the surveyed node has no prey
            self.prey_belief_vector[node_name_to_survey] = 0.0
            total_beliefs = 0.0
            for belief in self.prey_belief_vector:
                total_beliefs += belief
            for i in range(len(self.prey_belief_vector)):
                self.prey_belief_vector[i] = float(self.prey_belief_vector[i] / total_beliefs)

        # compute where the likely prey will be for this round after the agent finished its move
        # and treat it as the likely location of the prey for determining the node for the agent to move to
        self.future_prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
        largest_belief = self.future_prey_belief_vector[np.argmax(self.future_prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in
                                     enumerate(self.future_prey_belief_vector) if belief == largest_belief]

        # pick node with the largest belief for treating it as where prey is
        # (i.e., the likely location of the prey) for this round
        return random.choice(largest_belief_node_names)

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
            expected_next_state_u_partial += predator_transition_probability * partial_utility
        return reward + expected_next_state_u_partial
