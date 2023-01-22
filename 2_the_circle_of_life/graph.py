import random
from random import randint as ri
from collections import deque
from sortedcontainers import SortedDict
from copy import *

from node import *
from prey import *
from predator import *
from agent import *
from util import *


class Graph:
    ONGOING = 0
    CAPTURING = 1
    CAPTURED = 2
    TIMEOUT = 3

    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.size = kwargs['graph_size']
            self.graph = None
            self.distance_from_agent = None
            self.prey = None
            self.predator = None
            self.agent = None
            self.possible_init_node_names = None
            self.status = self.ONGOING
            self.correct_prey_count = None
            self.correct_predator_count = None
            self.__init(kwargs)

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.size = deepcopy(self.size)
        d_copy.graph = deepcopy(self.graph)
        d_copy.distance = deepcopy(self.distance_from_agent)
        d_copy.prey = deepcopy(self.prey)
        d_copy.predator = deepcopy(self.predator)
        d_copy.agent = deepcopy(self.agent)
        d_copy.init_node_names = deepcopy(self.possible_init_node_names)
        d_copy.status = self.status
        d_copy.correct_prey_count = self.correct_prey_count
        d_copy.correct_predator_count = self.correct_predator_count
        return d_copy

    # to be updated in the future
    def write(self):
        pass

    # to be updated in the future
    def read(self):
        pass

    def move_prey(self):
        cur_node = self.graph[self.prey.cur_node_name]
        # randomly choose next node (from neighbors and current node) to move to
        next_node_name = random.choice([next_node_name.name for next_node_name in cur_node.neighbors] + [cur_node.name])
        self.prey.update_node_name(next_node_name)
        self.__update_node(prey=self.prey)

    def move_predator(self):
        cur_node = self.graph[self.predator.cur_node_name]
        agent_node = self.graph[self.agent.cur_node_name]
        # get distances from predator's neighbors to agent
        self.__get_distance_for_predator_neighbors(cur_node, agent_node)
        # get the shortest distance among all distances from each predator's neighbor to agent
        shortest_distance = float('Inf')
        for neighbor in cur_node.neighbors:
            if INVALID < self.distance_from_agent[neighbor.name] < shortest_distance:
                shortest_distance = self.distance_from_agent[neighbor.name]
        # determine if predator is distracted for this move
        is_distracted = random.randint(0, 9) < 4
        if is_distracted:
            # randomly choose next node (from all neighbors) to move to
            next_node_name = random.choice([neighbor.name for neighbor in cur_node.neighbors])
        else:
            # choose the neighbors (of predator) with the shortest distance to agent
            next_node_name_candidates = [neighbor.name for neighbor in cur_node.neighbors
                                            if self.distance_from_agent[neighbor.name] == shortest_distance]
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
        elif self.agent.name == 'agent5':
            # decide what's the name of next node to move to for agent 5
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent6':
            # decide what's the name of next node to move to for agent 6
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent6_with_future_prediction':
            # decide what's the name of next node to move to for agent 6 with future prediction
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent7':
            # decide what's the name of next node to move to for agent 7
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent8':
            # decide what's the name of next node to move to for agent 8
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent8_with_future_prediction':
            # decide what's the name of next node to move to for agent 8 with future prediction
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'extended_agent7':
            # decide what's the name of next node to move to for extended agent 7
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'extended_agent8':
            # decide what's the name of next node to move to for extended agent 8
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'revised_extended_agent7':
            # decide what's the name of next node to move to for revised extended agent 7
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'revised_extended_agent8':
            # decide what's the name of next node to move to for revised extended agent 8
            next_node_name = self.agent.next_node_name(self)
        elif self.agent.name == 'agent9':
            # decide what's the name of next node to move to for agent 9
            next_node_name = self.agent.next_node_name(self)
        else:
            # to be completed in the future
            pass
        self.agent.update_node_name(next_node_name)
        self.__update_node(agent=self.agent)

    def get_status(self):
        # tell whether the game is still ongoing or is terminated due to prey being captured,
        # or agent being captured, or timeout (i.e., the time usage for the circle of life game is
        # passing the time limit)
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
        cur_node = self.graph[self.agent.cur_node_name]
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

    def update_correct_estimation_count(self):
        # increment the total number of times when agent knows exactly where prey is
        self.correct_prey_count += 1 if self.agent.is_correct_prey_estimation(self) else 0
        # increment the total number of times when the agent knows exactly where predator is
        self.correct_predator_count += 1 if self.agent.is_correct_predator_estimation(self) else 0

    def __init(self, kwargs):
        if not self.graph:
            self.__generate_graph()
            self.__get_possible_init_node_names()
            self.__init_agent(kwargs)
            self.__init_prey()
            self.__init_predator()
            self.update_status(0)

    def __init_prey(self):
        # get initial node to stand on at the start of the circle of life game
        init_node_name = self.__get_random_init_node_name()
        # initialize prey
        self.prey = Prey(init_node_name=init_node_name)
        self.__update_node(prey=self.prey)

    def __init_predator(self):
        # get initial node to stand on at the start of the circle of life game
        init_node_name = self.__get_random_init_node_name()
        # initialize predator
        self.predator = Predator(init_node_name=init_node_name)
        self.__update_node(predator=self.predator)

    def __init_agent(self, kwargs):
        # get used for determining how accurate agent predict or estimate where prey is
        # and where predator is
        self.correct_prey_count = self.correct_predator_count = 0
        # get initial node to stand on at the start of the circle of life game
        init_node_name = self.__get_random_init_node_name(agent=True)
        agent_name, is_bonus = kwargs['agent_name'], kwargs['is_bonus']
        if agent_name == 'agent1':
            # initialize agent 1
            kwargs = {
                'init_node_name': init_node_name
            }
            self.agent = Agent1(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent2':
            # initialize agent 2
            kwargs = {
                'init_node_name': init_node_name
            }
            self.agent = Agent2(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent3':
            # initialize agent 3
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = Agent3(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent4':
            # initialize agent 4
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = Agent4(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent4_with_future_prediction':
            # initialize agent 4 with future prediction
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = Agent4WithFuturePrediction(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent5':
            # initialize agent 5
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = Agent5(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent6':
            # initialize agent 6
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = Agent6(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent6_with_future_prediction':
            # initialize agent 6 with future prediction
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = Agent6WithFuturePrediction(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent7':
            # initialize agent 7
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = Agent7(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent8':
            # initialize agent 8
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = Agent8(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent8_with_future_prediction':
            # initialize agent 8 with future prediction
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = Agent8WithFuturePrediction(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'extended_agent7':
            # initialize extended agent 7
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = ExtendedAgent7(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'extended_agent8':
            # initialize extended agent 8
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = ExtendedAgent8(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'revised_extended_agent7':
            # initialize revised extended agent 7
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = RevisedExtendedAgent7(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'revised_extended_agent8':
            # initialize revised extended agent 8
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = RevisedExtendedAgent8(kwargs)
            self.__update_node(agent=self.agent)
        elif agent_name == 'agent9':
            # initialize agent 9
            kwargs = {
                'init_node_name': init_node_name,
                'is_bonus': is_bonus
            }
            self.agent = Agent9(kwargs)
            self.__update_node(agent=self.agent)
        else:
            # to be completed in the future
            pass

    def __generate_graph(self):
        # create a graph with primary loop from node 1 to node 50
        # (to make things compatible, use 1-indexed,
        # i.e., self.graph[0] has no meaning ,and thus set it to None)
        self.graph = [None]
        self.graph.extend(Node(i) for i in range(1, self.size + 1))
        for i in range(1, self.size + 1):
            for di in [-1, 1]:
                self.graph[i].add_neighbor(self.graph[((i + di + self.size - 1) % self.size) + 1])
        # add extra edges between nodes to graph
        self.__add_edges()

    def __add_edges(self):
        node_set = ListDict()
        for i in range(1, self.size + 1):
            node_set.add_item(i)
        # not ensure that every node is having degree equal to 3
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
                self.graph[random_node].add_neighbor(self.graph[random_node_neighbor])
                self.graph[random_node_neighbor].add_neighbor(self.graph[random_node])

    def __get_distance_for_predator_neighbors(self, predator_node, agent_node):
        # compute distance from agent to each node for each node
        self.distance_from_agent = [INVALID for _ in range(self.size + 1)]
        queue = deque([agent_node])
        self.distance_from_agent[agent_node.name] = 0
        is_predator_reached = False
        while len(queue) and not is_predator_reached:
            cur_node = queue.popleft()
            # early break when reach predator
            if cur_node.name == predator_node.name:
                is_predator_reached = True
                break
            cur_dist = self.distance_from_agent[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.distance_from_agent[neighbor_node.name] == INVALID:
                    self.distance_from_agent[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)

    def __get_possible_init_node_names(self):
        # create candidates of nodes for agent, prey, and predator to choose to stand on
        # during initialization of the circle of life game
        self.possible_init_node_names = ListDict()
        for i in range(1, self.size + 1):
            self.possible_init_node_names.add_item(i)

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
                pre_node = self.graph[prey.pre_node_name]
                pre_node.remove_prey()
            cur_node = self.graph[prey.cur_node_name]
            cur_node.insert_prey(prey)
        if predator:
            # update record of predator's current standing node
            if predator.pre_node_name:
                pre_node = self.graph[predator.pre_node_name]
                pre_node.remove_predator()
            cur_node = self.graph[predator.cur_node_name]
            cur_node.insert_predator(predator)
        if agent:
            # update record of agent's current standing node
            if agent.pre_node_name:
                pre_node = self.graph[agent.pre_node_name]
                pre_node.remove_agent()
            cur_node = self.graph[agent.cur_node_name]
            cur_node.insert_agent(agent)
