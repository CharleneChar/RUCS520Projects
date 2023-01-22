import os

from util import *


class AgentUStar:
    def __init__(self, kwargs):
        self.name = 'agent_u_star'
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']
        self.graph_size = kwargs['graph_size']
        self.all_pairs_distance = kwargs['graph_all_pairs_distance']
        graph_adj_list = kwargs['graph_adj_list']
        graph_name = kwargs['graph_name']
        # store utility and action corresponding to each state
        self.utility_action_dict = {}
        # store state space, action space, and transition probability
        self.info_dict = {}
        # store reward
        self.reward_dict = {}
        if not self.read(graph_name):
            self.__initialize_mdp(graph_adj_list)
            # compute U* and corresponding action
            self.__get_optimal_utilities()
            self.write(graph_name)

    def write(self, graph_name):
        filename = f'rsc/{graph_name}.agent_u_star_utility_policy_dict'
        if os.path.isfile(filename):
            return False
        with open(filename, 'w') as f:
            # write out (as database) dictionary
            # with form of {state s : (utility*, [action1, action2, ...])}
            f.write(f'{self.graph_size}\n')
            for state, (utility, action) in self.utility_action_dict.items():
                f.write(f'{state}\n')
                f.write(f'{utility}\n')
                f.write(','.join(map(str, action)) + '\n')
        return True

    def read(self, graph_name):
        filename = f'rsc/{graph_name}.agent_u_star_utility_policy_dict'
        if not os.path.isfile(filename):
            return False
        with open(filename, 'r') as f:
            # read in database
            # as dictionary with form of {state s : (utility*, [action1, action2, ...])}
            self.graph_size, = map(int, f.readline().rstrip().split(','))
            num_states = self.graph_size ** 3
            for _ in range(num_states):
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
                action_line = f.readline().rstrip()
                if len(action_line) == 0:
                    action = []
                else:
                    action = list(map(int, action_line.split(',')))
                self.utility_action_dict[state] = (utility, action)
        return True

    def is_infinite_state(self, init_state, graph):
        # check if a starting state is infinite (i.e., having infinite utility)
        return self.utility_action_dict[init_state][0] == float('Inf')

    def refresh(self, kwargs):
        self.pre_node_name = None
        self.cur_node_name = kwargs['init_node_name']

    def update_node_name(self, next_node_name):
        self.pre_node_name = self.cur_node_name
        self.cur_node_name = next_node_name

    def next_node_name(self, graph):
        kwargs = {
            'agent_node_name': self.cur_node_name,
            'prey_node_name': graph.prey.cur_node_name,
            'predator_node_name': graph.predator.cur_node_name,
            'graph_size': self.graph_size
        }
        state = State(kwargs)
        return random.choice(self.utility_action_dict[state][1])

    def get_state_with_largest_finite_u_star(self):
        # get state(s) with largest finite U*
        if len(self.utility_action_dict) == 0:
            return None
        else:
            # # used for computing seeing infinite state
            # inf_state_not_on_same_node = set()
            largest_finite_u_star = float('-Inf')
            state_list = []
            for state, (utility, _) in self.utility_action_dict.items():
                if utility > largest_finite_u_star and utility != float('Inf'):
                    largest_finite_u_star = utility
                # # used for computing seeing infinite state
                # if utility == float('Inf') and state.agent_node_name != state.predator_node_name:
                #     inf_state_not_on_same_node.add(state.agent_node_name)
            for state, (utility, _) in self.utility_action_dict.items():
                if utility == largest_finite_u_star:
                    state_list.append(state)
            # # used for computing seeing infinite state
            # print(f'infinite states where predator and agent not on same node: {inf_state_not_on_same_node}')
            return state_list

    def __initialize_mdp(self, graph_adj_list):
        # construct one dictionary with form of
        # {state s : {action a : {next state s': transition probability}}}
        # and another dictionary with form of
        # {state s : {action a : reward}
        # and the other dictionary with form of
        # {state s : (utility*, [action1, action2, ...])}

        # record state
        for agent_node_name in range(1, self.graph_size + 1):
            for prey_node_name in range(1, self.graph_size + 1):
                for predator_node_name in range(1, self.graph_size + 1):
                    kwargs = {
                        'agent_node_name': agent_node_name,
                        'prey_node_name': prey_node_name,
                        'predator_node_name': predator_node_name,
                        'graph_size': self.graph_size
                    }
                    state = State(kwargs)

                    # record utility* and action
                    self.utility_action_dict[state] = (0, [])
                    if agent_node_name == prey_node_name:
                        self.utility_action_dict[state] = (0, [])
                    elif agent_node_name == predator_node_name:
                        self.utility_action_dict[state] = (float('Inf'), [agent_node_name])

                    self.info_dict[state] = {}

                    self.reward_dict[state] = {}

                    # record base case (including absorbing state)
                    if agent_node_name == prey_node_name or agent_node_name == predator_node_name:
                        continue

                    # record action
                    action_space = [agent_neighbor.name for agent_neighbor in
                                    graph_adj_list[agent_node_name].neighbors]
                    action_space.append(agent_node_name)
                    for action in action_space:
                        # record reward
                        self.reward_dict[state][action] = 1
                        if action == predator_node_name \
                                and action != prey_node_name:
                            self.reward_dict[state][action] = float('Inf')

                        self.info_dict[state][action] = {}
                        # compute the number of predator's neighbors closet to agent
                        predator_shortest_distance_neighbor_count = 0
                        shortest_distance = float('Inf')
                        for predator_neighbor in graph_adj_list[predator_node_name].neighbors:
                            dist_action_to_predator_neighbor = self.all_pairs_distance[action][predator_neighbor.name]
                            if dist_action_to_predator_neighbor < shortest_distance:
                                shortest_distance = dist_action_to_predator_neighbor
                        for predator_neighbor in graph_adj_list[predator_node_name].neighbors:
                            dist_action_to_predator_neighbor = self.all_pairs_distance[action][predator_neighbor.name]
                            if dist_action_to_predator_neighbor == shortest_distance:
                                predator_shortest_distance_neighbor_count += 1
                        # record next state
                        prey_and_prey_neighbor_names = [prey_neighbor.name for prey_neighbor
                                                        in graph_adj_list[prey_node_name].neighbors]
                        prey_and_prey_neighbor_names.append(prey_node_name)
                        predator_neighbor_names = [predator_neighbor.name for predator_neighbor
                                                               in graph_adj_list[predator_node_name].neighbors]
                        for prey_or_prey_neighbor_name in prey_and_prey_neighbor_names:
                            for predator_neighbor_name in predator_neighbor_names:
                                kwargs = {
                                    'agent_node_name': action,
                                    'prey_node_name': prey_or_prey_neighbor_name,
                                    'predator_node_name': predator_neighbor_name,
                                    'graph_size': self.graph_size
                                }
                                next_state = State(kwargs)
                                # compute and record transition probability
                                prey_transition_probability = 1 / (1 + len(graph_adj_list[prey_node_name].neighbors))
                                predator_transition_probability = 0.4 * \
                                                                  (1 /
                                                                   len(graph_adj_list[predator_node_name].neighbors))
                                if self.all_pairs_distance[action][predator_neighbor_name] == shortest_distance:
                                    predator_transition_probability += 0.6 * (
                                            1 / predator_shortest_distance_neighbor_count)
                                joint_transition_probability = prey_transition_probability * \
                                                               predator_transition_probability
                                self.info_dict[state][action][next_state] = joint_transition_probability

    def __get_optimal_utilities(self):
        # implement Bellman's Equations with value iteration to get U*
        # (note here min utility is target rather than max)
        iter_times = 0
        while True:
            iter_times += 1
            uk_pk_dict = deepcopy(self.utility_action_dict)
            sum_u_diff = 0
            for state in self.info_dict.keys():
                if state.is_prey_captured() or state.is_agent_captured():
                    continue
                min_utility = float('Inf')
                utility_action_choice_list = []
                # compute expected future utility for each next state
                for action in self.info_dict[state]:
                    utility = reward = self.reward_dict[state][action]
                    for next_state, transition_probability in self.info_dict[state][action].items():
                        if uk_pk_dict[next_state][0] != float('Inf'):
                            utility += transition_probability * uk_pk_dict[next_state][0]
                        else:
                            utility = float('Inf')
                            break
                    utility_action_choice_list.append((utility, action))
                    if utility < min_utility:
                        min_utility = utility
                # get optimal action(s) for one state
                optimal_action_choice_list = []
                for u, a in utility_action_choice_list:
                    if u == min_utility:
                        optimal_action_choice_list.append(a)
                if min_utility != float('Inf'):
                    diff = abs(min_utility - self.utility_action_dict[state][0])
                    sum_u_diff += diff
                else:
                    # used for absorbing state
                    optimal_action_choice_list = [state.agent_node_name]
                # record and update one state's up-to-date utility and action
                self.utility_action_dict[state] = (min_utility, optimal_action_choice_list)

            # check if all states reach the optimal utility (with some degree of error)
            print(f'{iter_times}th sum diff of u: {sum_u_diff}')
            if sum_u_diff < 1:
                break
