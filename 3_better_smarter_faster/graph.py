import random
from random import randint as ri
from collections import deque
from sortedcontainers import SortedDict
from copy import *
import os

from node import *
from prey import *
from predator import *
from agent import *
from util import *
from u_star import *
from v import *
from u_partial import *
from v_partial import *
# from bonus import *


class Graph:
    ONGOING = 0
    CAPTURING = 1
    CAPTURED = 2
    TIMEOUT = 3

    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.name = kwargs['graph_name']
            self.size = kwargs['graph_size']
            self.adj_list = None
            self.prey = None
            self.predator = None
            self.agent = None
            self.initial_state = None
            self.possible_init_node_names = None
            self.status = self.ONGOING
            self.all_pairs_distance = None
            self.__init(kwargs)

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.size = deepcopy(self.size)
        d_copy.adj_list = deepcopy(self.adj_list)
        d_copy.prey = deepcopy(self.prey)
        d_copy.predator = deepcopy(self.predator)
        d_copy.agent = deepcopy(self.agent)
        d_copy.initial_state = deepcopy(self.initial_state)
        d_copy.possible_init_node_names = deepcopy(self.possible_init_node_names)
        d_copy.status = self.status
        d_copy.all_pairs_distance = deepcopy(self.all_pairs_distance)
        return d_copy

    def __repr__(self):
        attr_width = 4
        if self.agent.cur_node_name == self.prey.cur_node_name:
            if self.agent.cur_node_name == self.predator.cur_node_name:
                attr_width += 8
            else:
                attr_width += 4
        elif self.prey.cur_node_name == self.predator.cur_node_name:
            attr_width += 4
        name_width = len(str(self.size + 1)) + attr_width
        s = ''
        for i in range(1, self.size + 1):
            node = self.adj_list[i]
            t = f'{i}'
            if i == self.agent.cur_node_name:
                t += f'(ag)'
            if i == self.prey.cur_node_name:
                t += f'(py)'
            if i == self.predator.cur_node_name:
                t += f'(pd)'
            s += t.ljust(name_width) + f': '
            for neighbor in node.neighbors:
                s += f'{neighbor.name} -> '
            s += '\n'
        return s

    def write(self):
        filename = f'rsc/{self.name}.graph'
        if os.path.isfile(filename):
            return False
        with open(filename, 'w') as f:
            f.write(f'{self.size}\n')
            for node in self.adj_list[1:]:
                neighbors_str = ','.join(map(str, [neighbor.name for neighbor in node.neighbors])) + '\n'
                f.write(neighbors_str)
            f.write(f'{self.initial_state}\n')
        return True

    def read(self):
        filename = f'rsc/{self.name}.graph'
        if not os.path.isfile(filename):
            return False
        with open(filename, 'r') as f:
            self.size, = map(int, f.readline().rstrip().split(','))
            self.adj_list = [None]
            self.adj_list.extend(Node(i) for i in range(1, self.size + 1))
            for node in self.adj_list[1:]:
                for neighbor_name in map(int, f.readline().rstrip().split(',')):
                    node.neighbors.append(self.adj_list[neighbor_name])
            agent_node_name, prey_node_name, predator_node_name = map(int, f.readline().rstrip().split(','))
            kwargs = {
                'agent_node_name': agent_node_name,
                'prey_node_name': prey_node_name,
                'predator_node_name': predator_node_name,
                'graph_size': self.size
            }
            self.initial_state = State(kwargs)
        return True

    def refresh(self, kwargs):
        self.clear_graph_info()
        self.__generate_initial_state()
        self.__init_agent(kwargs)
        self.__init_prey()
        self.__init_predator()
        again = self.agent.is_infinite_state(self.initial_state, self)
        while again:
            self.__generate_initial_state()
            self.__init_agent(kwargs)
            self.__init_prey()
            self.__init_predator()
            again = self.agent.is_infinite_state(self.initial_state, self)
        self.update_status(0)

    def clear_graph_info(self):
        self.adj_list[self.agent.cur_node_name].remove_agent()
        self.adj_list[self.prey.cur_node_name].remove_prey()
        self.adj_list[self.predator.cur_node_name].remove_predator()
        self.status = self.ONGOING

    def move_prey(self):
        cur_node = self.adj_list[self.prey.cur_node_name]
        # randomly choose next node (from neighbors and current node) to move to
        next_node_name = random.choice([next_node_name.name for next_node_name in cur_node.neighbors] + [cur_node.name])
        self.prey.update_node_name(next_node_name)
        self.__update_node(prey=self.prey)

    def move_predator(self):
        cur_node = self.adj_list[self.predator.cur_node_name]
        agent_node = self.adj_list[self.agent.cur_node_name]
        # get the shortest distance among all distances from each predator's neighbor to agent
        shortest_distance = float('Inf')
        for neighbor in cur_node.neighbors:
            dist_from_agent_to_predator_neighbor = self.all_pairs_distance[agent_node.name][neighbor.name]
            if INVALID < dist_from_agent_to_predator_neighbor < shortest_distance:
                shortest_distance = dist_from_agent_to_predator_neighbor
        # determine if predator is distracted for this move
        is_distracted = random.randint(0, 9) < 4
        if is_distracted:
            # randomly choose next node (from all neighbors) to move to
            next_node_name = random.choice([neighbor.name for neighbor in cur_node.neighbors])
        else:
            # choose the neighbors (of predator) with the shortest distance to agent
            next_node_name_candidates = [neighbor.name for neighbor in cur_node.neighbors
                                         if self.all_pairs_distance[agent_node.name][neighbor.name]
                                         == shortest_distance]
            # randomly choose next node (from neighbors with same shortest distance) to move to
            next_node_name = random.choice(next_node_name_candidates)
        self.predator.update_node_name(next_node_name)
        self.__update_node(predator=self.predator)

    def move_agent(self):
        next_node_name = None
        if self.agent.name == 'agent1':
            # decide what's the name of next node to move to for agent 1
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent2':
            # decide what's the name of next node to move to for agent 2
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent3':
            # decide what's the name of next node to move to for agent 3
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent4':
            # decide what's the name of next node to move to for agent 4
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent4_with_future_prediction':
            # decide what's the name of next node to move to for agent 4 with future prediction
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent_u_star':
            # decide what's the name of next node to move to for agent U*
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent_v':
            # decide what's the name of next node to move to for agent V
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent_u_partial':
            # decide what's the name of next node to move to for agent U partial
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent_v_partial':
            # decide what's the name of next node to move to for agent V partial
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent_another_u_partial':
            # decide what's the name of next node to move to for agent another U partial
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent_bonus_1':
            # decide what's the name of next node to move to for agent bonus 1
            next_node_name = self.agent.next_node_name(self)
        else:
            # to be completed in the future
            pass
        self.agent.update_node_name(next_node_name)
        self.__update_node(agent=self.agent)

    def get_status(self):
        # tell whether the game is still ongoing or is terminated due to prey being captured
        if self.status == self.ONGOING:
            return 'ONGOING'
        if self.status == self.CAPTURING:
            return 'CAPTURING'
        if self.status == self.CAPTURED:
            return 'CAPTURED'
        if self.status == self.TIMEOUT:
            return 'TIMEOUT'
        return 'UNKNOWN'

    def update_status(self, step):
        cur_node = self.adj_list[self.agent.cur_node_name]
        # specify upper time limit for each strategy to solve circle of life game
        # (pending below: try different time limit for hung)
        if step > 100 * self.size:
            # confirm that agent, prey, and predator are all alive in the graph
            # but over time limit
            self.status = self.TIMEOUT
        elif cur_node.is_capturing_prey():
            # confirm that agent captures prey within time limit
            self.status = self.CAPTURING
        elif cur_node.is_captured_by_predator():
            # confirm that agent is captured by predator within time limit
            self.status = self.CAPTURED

    def get_state_with_different_choice_between_agents(self):
        # find and write out states where agent 1 and agent u star have different actions,
        # and states where agent 2 and agent u star have different actions
        filename_agent1 = f'rsc/{self.name}.state_agent_1_and_u_star_action_dict'
        filename_agent2 = f'rsc/{self.name}.state_agent_2_and_u_star_action_dict'
        if os.path.isfile(filename_agent1) and os.path.isfile(filename_agent2):
            return False
        state_agent_action_dict = {}
        with open(f'rsc/{self.name}.agent_u_star_utility_policy_dict', 'r') as f:
            f.readline()
            for _ in range(self.size ** 3):
                agent_node_name, prey_node_name, predator_node_name = \
                    map(int, f.readline().rstrip().split(','))
                kwargs = {
                    'agent_node_name': agent_node_name,
                    'prey_node_name': prey_node_name,
                    'predator_node_name': predator_node_name,
                    'graph_size': self.size
                }
                state = State(kwargs)
                f.readline()
                line = f.readline().rstrip()
                if len(line) > 0:
                    action = sorted(list(map(int, line.split(','))))
                else:
                    action = []
                state_agent_action_dict[state] = {'agent_u_star': action}
        state_diff_action_dict_for_1 = {}
        with open(f'rsc/{self.name}.agent1_policy_dict', 'r') as f:
            for _ in range(self.size ** 3):
                agent_node_name, prey_node_name, predator_node_name = \
                    map(int, f.readline().rstrip().split(','))
                kwargs = {
                    'agent_node_name': agent_node_name,
                    'prey_node_name': prey_node_name,
                    'predator_node_name': predator_node_name,
                    'graph_size': self.size
                }
                state = State(kwargs)
                line = f.readline().rstrip()
                if len(line) > 0:
                    action = sorted(list(map(int, line.split(','))))
                else:
                    action = []
                # if state_agent_action_dict[state]['agent_u_star'] not in action:
                #     agent_u_star_action = state_agent_action_dict[state]['agent_u_star']
                #     state_diff_action_dict_for_1[state] = (action, agent_u_star_action)
                if state_agent_action_dict[state]['agent_u_star'] != action:
                    agent_u_star_action = state_agent_action_dict[state]['agent_u_star']
                    state_diff_action_dict_for_1[state] = (action, agent_u_star_action)
        with open(filename_agent1, 'w') as f:
            f.write(f'{len(state_diff_action_dict_for_1)}\n')
            for state, (agent1_action, agent_u_star_action) in state_diff_action_dict_for_1.items():
                f.write(f'{state}\n')
                f.write(','.join(map(str, agent1_action)) + '\n')
                f.write(','.join(map(str, agent_u_star_action)) + '\n')

        state_diff_action_dict_for_2 = {}
        with open(f'rsc/{self.name}.agent2_policy_dict', 'r') as f:
            for _ in range(self.size ** 3):
                agent_node_name, prey_node_name, predator_node_name = \
                    map(int, f.readline().rstrip().split(','))
                kwargs = {
                    'agent_node_name': agent_node_name,
                    'prey_node_name': prey_node_name,
                    'predator_node_name': predator_node_name,
                    'graph_size': self.size
                }
                state = State(kwargs)
                line = f.readline().rstrip()
                if len(line) > 0:
                    action = sorted(list(map(int, line.split(','))))
                else:
                    action = []
                # if state_agent_action_dict[state]['agent_u_star'] not in action:
                #     agent_u_star_action = state_agent_action_dict[state]['agent_u_star']
                #     state_diff_action_dict_for_2[state] = (action, agent_u_star_action)
                if state_agent_action_dict[state]['agent_u_star'] != action:
                    agent_u_star_action = state_agent_action_dict[state]['agent_u_star']
                    state_diff_action_dict_for_2[state] = (action, agent_u_star_action)
        with open(filename_agent2, 'w') as f:
            f.write(f'{len(state_diff_action_dict_for_2)}\n')
            for state, (agent2_action, agent_u_star_action) in state_diff_action_dict_for_2.items():
                f.write(f'{state}\n')
                f.write(','.join(map(str, agent2_action)) + '\n')
                f.write(','.join(map(str, agent_u_star_action)) + '\n')
        print(len(state_diff_action_dict_for_1), len(state_diff_action_dict_for_2))
        # to be completed in the future
        # custom_plot(state_diff_action_dict_for_1, self.adj_list,
        #      f'{self.name}.state_with_diff_choice')
        # custom_plot(state_diff_action_dict_for_2, self.adj_list,
        #      f'{self.name}.state_with_diff_choice')
        return True

    def __init(self, kwargs):
        if not self.adj_list:
            if not self.read():
                self.__generate_graph()
                self.__generate_initial_state()
                self.write()
        self.__generate_all_pairs_distance()
        self.__init_agent(kwargs)
        self.__init_prey()
        self.__init_predator()
        while self.agent.is_infinite_state(self.initial_state, self):
            self.__generate_initial_state()
            self.__init_agent(kwargs)
            self.__init_prey()
            self.__init_predator()
        self.update_status(0)

    def __init_prey(self):
        # initialize prey
        self.prey = Prey(init_node_name=self.initial_state.prey_node_name)
        self.__update_node(prey=self.prey)

    def __init_predator(self):
        # initialize predator
        self.predator = Predator(init_node_name=self.initial_state.predator_node_name)
        self.__update_node(predator=self.predator)

    def __init_agent(self, kwargs):
        agent_name = kwargs['agent_name']
        if agent_name == 'agent1':
            # initialize agent 1
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph': self,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = Agent1(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent2':
            # initialize agent 2
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph': self,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = Agent2(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent3':
            # initialize agent 3
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph': self,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = Agent3(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent4':
            # initialize agent 4
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph': self,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = Agent4(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent4_with_future_prediction':
            # initialize agent 4 with future prediction
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph': self,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = Agent4WithFuturePrediction(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent_u_star':
            # initialize agent U*
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph_adj_list': self.adj_list,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = AgentUStar(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent_v':
            # initialize agent V
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph_adj_list': self.adj_list,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = AgentV(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent_u_partial':
            # initialize agent U partial
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph_adj_list': self.adj_list,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = AgentUPartial(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent_v_partial':
            # initialize agent V partial
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph_adj_list': self.adj_list,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = AgentVPartial(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent_another_u_partial':
            # initialize agent another U partial
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph_adj_list': self.adj_list,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = AgentAnotherUPartial(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent_bonus_1':
            # initialize agent bonus 1
            is_object_kept = kwargs['is_object_kept']
            kwargs = {
                'init_node_name': self.initial_state.agent_node_name,
                'graph_size': self.size,
                'graph_adj_list': self.adj_list,
                'graph_name': self.name,
                'graph_all_pairs_distance': self.all_pairs_distance
            }
            if is_object_kept and self.agent is not None:
                self.agent.refresh(kwargs)
            else:
                self.agent = AgentBonus1(kwargs)
            self.__update_node(agent=self.agent)
        else:
            # to be completed in the future
            pass

    def __generate_initial_state(self):
        # create candidates of nodes for agent, prey, and predator to choose to stand on
        # during initialization of the circle of life game
        self.possible_init_node_names = ListDict()
        for i in range(1, self.size + 1):
            self.possible_init_node_names.add_item(i)
        # set up initial states by assigning initial node to stand on for agent, prey and predator
        # before any entity start making move
        kwargs = {
            'agent_node_name': self.__get_random_init_node_name(agent=True),
            'prey_node_name': self.__get_random_init_node_name(),
            'predator_node_name': self.__get_random_init_node_name(),
            'graph_size': self.size
        }
        self.initial_state = State(kwargs)

    def __generate_graph(self):
        # create a graph with primary loop from node 1 to node 50
        # (to make things compatible, use 1-indexed,
        # i.e., self.adj_list[0] has no meaning ,and thus set it to None)
        self.adj_list = [None]
        self.adj_list.extend(Node(i) for i in range(1, self.size + 1))
        for i in range(1, self.size + 1):
            for di in [-1, 1]:
                self.adj_list[i].add_neighbor(self.adj_list[((i + di + self.size - 1) % self.size) + 1])
        # add extra edges between nodes to graph
        self.__add_edges()

    def __add_edges(self):
        node_set = ListDict()
        for i in range(1, self.size + 1):
            node_set.add_item(i)
        # set edges between the nodes with degree less than 3
        while len(node_set):
            # pick a node randomly
            random_node = node_set.choose_random_item()
            node_set.remove_item(random_node)
            # choose a random node within 5 steps forward or backward along the primary loop to add to neighbor list
            # (and ignore the node within 1 step since they are already connected to the picked node)
            temp_node_neighbors = []
            for i in [-5, -4, -3, -2, 2, 3, 4, 5]:
                temp_node_neighbor = ((random_node + i + self.size - 1) % self.size) + 1
                if temp_node_neighbor in node_set:
                    temp_node_neighbors.append(temp_node_neighbor)
            if len(temp_node_neighbors):
                random_node_neighbor = random.choice(temp_node_neighbors)
                # remove chosen node from the set of nodes with degree less than 3
                node_set.remove_item(random_node_neighbor)
                # adding edge between picked node and chosen node with some step away from the picked node
                # by making them neighbor with each other
                self.adj_list[random_node].add_neighbor(self.adj_list[random_node_neighbor])
                self.adj_list[random_node_neighbor].add_neighbor(self.adj_list[random_node])

    def __generate_all_pairs_distance(self):
        # compute all pairs' shortest distances in a graph
        self.all_pairs_distance = [[INVALID for _ in range(self.size + 1)]]
        for source_name in range(1, self.size + 1):
            distance = [INVALID for _ in range(self.size + 1)]
            source_node = self.adj_list[source_name]
            queue = deque([source_node])
            distance[source_node.name] = 0
            while len(queue):
                cur_node = queue.popleft()
                cur_dist = distance[cur_node.name]
                for neighbor_node in cur_node.neighbors:
                    if distance[neighbor_node.name] == INVALID:
                        distance[neighbor_node.name] = cur_dist + 1
                        queue.append(neighbor_node)
            self.all_pairs_distance.append(distance)
        # if DEBUG > -1:
        #     print_distance(self.all_pairs_distance, sep=' ')

    # def __get_distance_for_predator_neighbors(self, agent_node):
    #     # compute distance from agent to each node for each node
    #     distance_from_agent = [INVALID for _ in range(self.size + 1)]
    #     queue = deque([agent_node])
    #     distance_from_agent[agent_node.name] = 0
    #     while len(queue):
    #         cur_node = queue.popleft()
    #         cur_dist = distance_from_agent[cur_node.name]
    #         for neighbor_node in cur_node.neighbors:
    #             if distance_from_agent[neighbor_node.name] == INVALID:
    #                 distance_from_agent[neighbor_node.name] = cur_dist + 1
    #                 queue.append(neighbor_node)
    #     return distance_from_agent

    def __get_random_init_node_name(self, agent=False):
        # let agent pick its initial node to stand on first
        if agent:
            agent_init_node_name = self.possible_init_node_names.choose_random_item()
            # exclude initial node of agent from the choice of possible initial nodes for prey or predator to stand on
            self.possible_init_node_names.remove_item(agent_init_node_name)
            return agent_init_node_name
        other_init_node_name = self.possible_init_node_names.choose_random_item()
        return other_init_node_name

    def __update_node(self, prey=None, predator=None, agent=None):
        if prey:
            # update record of prey's current standing node
            if prey.pre_node_name:
                pre_node = self.adj_list[prey.pre_node_name]
                pre_node.remove_prey()
            cur_node = self.adj_list[prey.cur_node_name]
            cur_node.insert_prey(prey)
        if predator:
            # update record of predator's current standing node
            if predator.pre_node_name:
                pre_node = self.adj_list[predator.pre_node_name]
                pre_node.remove_predator()
            cur_node = self.adj_list[predator.cur_node_name]
            cur_node.insert_predator(predator)
        if agent:
            # update record of agent's current standing node
            if agent.pre_node_name:
                pre_node = self.adj_list[agent.pre_node_name]
                pre_node.remove_agent()
            cur_node = self.adj_list[agent.cur_node_name]
            cur_node.insert_agent(agent)
