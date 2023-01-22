import copy
import random
from collections import deque, defaultdict
from copy import *
import numpy as np

from graph import *
from util import *


class AgentOdd:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.name = None
            self.pre_node_name = None
            self.cur_node_name = kwargs['init_node_name']
            self.to_prey_distance = None
            self.to_predator_distance = None
            self.is_bonus = None
            self.is_surveyed = None
            self.survey_count = 0
            # get used for counting the number of times when agent knows exactly where prey is
            self.estimated_prey_name = None
            # get used for counting the number of times when agent knows exactly where predator is
            self.estimated_predator_name = None

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.name = self.name
        d_copy.pre_node_name = deepcopy(self.pre_node_name)
        d_copy.cur_node_name = deepcopy(self.cur_node_name)
        d_copy.to_prey_distance = deepcopy(self.to_prey_distance)
        d_copy.to_predator_distance = deepcopy(self.to_predator_distance)
        d_copy.is_bonus = deepcopy(self.is_bonus)
        d_copy.is_surveyed = deepcopy(self.is_surveyed)
        d_copy.survey_count = deepcopy(self.survey_count)
        d_copy.estimated_prey_name = deepcopy(self.estimated_prey_name)
        d_copy.estimated_predator_name = deepcopy(self.estimated_predator_name)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def get_distance(self, graph, prey_node_name, predator_node_name):
        # compute distance from prey node to every node for every node in graph
        self.to_prey_distance = [INVALID for _ in range(graph.size + 1)]
        prey_node = graph.graph[prey_node_name]
        queue = deque([prey_node])
        self.to_prey_distance[prey_node.name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = self.to_prey_distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.to_prey_distance[neighbor_node.name] == INVALID:
                    self.to_prey_distance[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)
        # compute distance from predator node to every node for every node in graph
        self.to_predator_distance = [INVALID for _ in range(graph.size + 1)]
        predator_node = graph.graph[predator_node_name]
        queue = deque([predator_node])
        self.to_predator_distance[predator_node.name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = self.to_predator_distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.to_predator_distance[neighbor_node.name] == INVALID:
                    self.to_predator_distance[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)

    def update_node_name(self, next_node_name):
        self.pre_node_name = self.cur_node_name
        self.cur_node_name = next_node_name

    def get_next_node_name_by_rules(self, graph):
        cur_node = graph.graph[self.cur_node_name]

        # debug by printing out corresponding current distance from agent to prey, and to predator
        if DEBUG > 0:
            for neighbor in graph.graph[graph.agent.cur_node_name].neighbors:
                print(f'{neighbor.name} to prey distance: {self.to_prey_distance[neighbor.name]}')
                print(f'{neighbor.name} to predator distance: {self.to_predator_distance[neighbor.name]}')

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
            if self.to_prey_distance[neighbor.name] < min_prey_distance:
                min_prey_distance = self.to_prey_distance[neighbor.name]
            if self.to_prey_distance[neighbor.name] > max_prey_distance:
                max_prey_distance = self.to_prey_distance[neighbor.name]
            if self.to_predator_distance[neighbor.name] < min_predator_distance:
                min_predator_distance = self.to_predator_distance[neighbor.name]
            if self.to_predator_distance[neighbor.name] > max_predator_distance:
                max_predator_distance = self.to_predator_distance[neighbor.name]
            neighbors_to_prey_and_predator_info.append(
                (self.to_prey_distance[neighbor.name], self.to_predator_distance[neighbor.name],
                 random_num_list[i], neighbor.name))
        # get used for classifying up to three classes of neighbors:
        # those closer to the prey,
        # those not closer and not farther from the prey,
        # and those farther from the prey
        min_end_index = not_min_max_end_index = len(neighbors_to_prey_and_predator_info)
        neighbors_to_prey_and_predator_info.sort()
        for i, info in enumerate(neighbors_to_prey_and_predator_info):
            if min_end_index is None and info[0] != min_prey_distance:
                min_end_index = i
            if not_min_max_end_index is None and info[0] == max_prey_distance:
                not_min_max_end_index = i

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

    def is_survey_required(self, graph, predator_node_name,
                           prey_belief_vector=None, predator_belief_vector=None,
                           belief_threshold=0.7):
        # determine whether to survey (or not) based mainly on a threshold for belief
        for predator_neighbor in graph.graph[predator_node_name].neighbors:
            if predator_neighbor.name == self.cur_node_name:
                return False
            for predator_neighbor_neighbor in graph.graph[predator_neighbor.name].neighbors:
                if predator_neighbor_neighbor == self.cur_node_name:
                    return False
        if predator_belief_vector is not None:
            return np.max(predator_belief_vector) < belief_threshold
        if prey_belief_vector is not None:
            return np.max(prey_belief_vector) < belief_threshold
        return True

    @staticmethod
    def is_survey_required_based_on_distance(graph, source_name, target_name, distance_threshold):
        # determine whether to survey (or not) based mainly on
        # a threshold for distance between the source and the target
        # (i.e., distance between agent and predator)
        distance = [INVALID for _ in range(graph.size + 1)]
        source_node = graph.graph[source_name]
        queue = deque([source_node])
        distance[source_node.name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if distance[neighbor_node.name] == INVALID:
                    distance[neighbor_node.name] = cur_dist + 1
                    if neighbor_node.name == target_name:
                        return distance[neighbor_node.name] > distance_threshold
                    queue.append(neighbor_node)

    def is_correct_prey_estimation(self, graph):
        # check if the agent knows exactly where the prey is
        return graph.graph[self.estimated_prey_name].is_prey_existed()

    def is_correct_predator_estimation(self, graph):
        # check if the agent knows exactly where the predator is
        return graph.graph[self.estimated_predator_name].is_predator_existed()


class AgentEven:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.name = None
            self.pre_node_name = None
            self.cur_node_name = kwargs['init_node_name']
            self.distance = None
            self.is_bonus = None
            self.is_surveyed = None
            self.survey_count = 0
            # get used for counting the number of times when agent knows exactly where prey is
            self.estimated_prey_name = None
            # get used for counting the number of times when agent knows exactly where predator is
            self.estimated_predator_name = None

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.name = self.name
        d_copy.pre_node_name = deepcopy(self.pre_node_name)
        d_copy.cur_node_name = deepcopy(self.cur_node_name)
        d_copy.distance = deepcopy(self.distance)
        d_copy.is_bonus = deepcopy(self.is_bonus)
        d_copy.is_surveyed = deepcopy(self.is_surveyed)
        d_copy.survey_count = deepcopy(self.survey_count)
        d_copy.estimated_prey_name = deepcopy(self.estimated_prey_name)
        d_copy.estimated_predator_name = deepcopy(self.estimated_predator_name)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def get_distance(self, graph, source_name):
        # compute distance from source node to every node for every node in graph
        self.distance = [INVALID for _ in range(graph.size + 1)]
        source_node = graph.graph[source_name]
        queue = deque([source_node])
        self.distance[source_node.name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = self.distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.distance[neighbor_node.name] == INVALID:
                    self.distance[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)

    def update_node_name(self, next_node_name):
        self.pre_node_name = self.cur_node_name
        self.cur_node_name = next_node_name

    def get_next_node_name(self, graph, prey_node_name, predator_node_name):
        cur_node = graph.graph[self.cur_node_name]
        predator_and_its_neighbor_node_names = {predator_node_name}
        for predator_neighbor in graph.graph[predator_node_name].neighbors:
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
        for prey_neighbor in graph.graph[prey_node_name].neighbors:
            prey_and_its_neighbor_node_names.add(prey_neighbor.name)
        for candidate_source_name in candidate_source_names:
            self.get_distance(graph, candidate_source_name)
            total_distance = 0
            # computing the total distance to the prey's neighborhood
            # (for each of the agent's neighbor and for the agent's current standing node)
            for node_name in prey_and_its_neighbor_node_names:
                total_distance += self.distance[node_name]
            # choose the node (from the agent current standing node and the agent's neighbors)
            # that wasn't excluded for not having the shortest distance to the prey's neighborhood
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                min_node_name = candidate_source_name
        return min_node_name

    def is_survey_required(self, graph, predator_node_name,
                           prey_belief_vector=None, predator_belief_vector=None,
                           belief_threshold=0.7):
        # determine whether to survey (or not) based mainly on a threshold for belief
        for predator_neighbor in graph.graph[predator_node_name].neighbors:
            if predator_neighbor.name == self.cur_node_name:
                return False
            for predator_neighbor_neighbor in graph.graph[predator_neighbor.name].neighbors:
                if predator_neighbor_neighbor == self.cur_node_name:
                    return False
        if predator_belief_vector is not None:
            return np.max(predator_belief_vector) < belief_threshold
        if prey_belief_vector is not None:
            return np.max(prey_belief_vector) < belief_threshold
        return True

    @staticmethod
    def is_survey_required_based_on_distance(graph, source_name, target_name, distance_threshold):
        # determine whether to survey (or not) based mainly on
        # a threshold for distance between the source and the target
        # (i.e., distance between agent and predator)
        distance = [INVALID for _ in range(graph.size + 1)]
        source_node = graph.graph[source_name]
        queue = deque([source_node])
        distance[source_node.name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if distance[neighbor_node.name] == INVALID:
                    distance[neighbor_node.name] = cur_dist + 1
                    if neighbor_node.name == target_name:
                        return distance[neighbor_node.name] > distance_threshold
                    queue.append(neighbor_node)

    def is_correct_prey_estimation(self, graph):
        # check if the agent knows exactly where the prey is
        return graph.graph[self.estimated_prey_name].is_prey_existed()

    def is_correct_predator_estimation(self, graph):
        # check if the agent knows exactly where the predator is
        return graph.graph[self.estimated_predator_name].is_predator_existed()


class Agent1(AgentOdd):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent1'

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get shortest distance from prey and shortest distance from predator to every node with BFS
        self.get_distance(graph=graph, prey_node_name=graph.prey.cur_node_name,
                          predator_node_name=graph.predator.cur_node_name)
        self.estimated_prey_name = graph.prey.cur_node_name
        self.estimated_predator_name = graph.predator.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name_by_rules(graph)


class Agent2(AgentEven):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent2'

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        self.estimated_prey_name = graph.prey.cur_node_name
        self.estimated_predator_name = graph.predator.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name(graph=graph, prey_node_name=graph.prey.cur_node_name,
                                       predator_node_name=graph.predator.cur_node_name)

    def get_next_node_name(self, graph, prey_node_name, predator_node_name):
        cur_node = graph.graph[self.cur_node_name]
        predator_and_its_neighbor_node_names = {predator_node_name}
        for predator_neighbor in graph.graph[predator_node_name].neighbors:
            predator_and_its_neighbor_node_names.add(predator_neighbor.name)
        source_names = {cur_node.name}
        for node in cur_node.neighbors:
            source_names.add(node.name)
        # exclude the neighbors (of the agent) or the agent's current standing node
        # who was one neighbor of the predator or was just the predator
        candidate_source_names = source_names.difference(predator_and_its_neighbor_node_names)
        if len(candidate_source_names) == 0:
            candidate_source_names = source_names.difference({predator_node_name})
        min_distance = float('Inf')
        min_node_dis_and_names = []
        distances_from_neighbor_to_prey = []
        # compute the distance to the prey
        # (for each of the agent's neighbor and for the agent's current standing node)
        for candidate_source_name in candidate_source_names:
            self.get_distance(graph, candidate_source_name)
            if self.distance[prey_node_name] < min_distance:
                min_distance = self.distance[prey_node_name]
            distances_from_neighbor_to_prey.append(self.distance[prey_node_name])
        for i, candidate_source_name in enumerate(candidate_source_names):
            if distances_from_neighbor_to_prey[i] == min_distance:
                min_node_dis_and_names.append(candidate_source_name)
        # choose the node (from the agent current standing node and the agent's neighbors)
        # that wasn't excluded for not having the shortest distance to the prey
        return random.choice(min_node_dis_and_names)


class Agent3(AgentOdd):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent3'
            self.prey_transition_matrix = None
            self.prey_belief_vector = None
            self.is_bonus = kwargs['is_bonus']

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.prey_transition_matrix = deepcopy(self.prey_transition_matrix)
        d_copy.prey_belief_vector = deepcopy(self.prey_belief_vector)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get the likely location of the prey
        likely_prey_node_name = self.get_likely_prey_node_name(graph)
        # get shortest distance from prey and shortest distance from predator to every node with BFS
        self.get_distance(graph=graph, prey_node_name=likely_prey_node_name,
                          predator_node_name=graph.predator.cur_node_name)
        self.estimated_prey_name = likely_prey_node_name
        self.estimated_predator_name = graph.predator.cur_node_name
        if self.is_bonus and self.is_surveyed:
            return self.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name_by_rules(graph)

    def get_likely_prey_node_name(self, graph):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            # where every probability is set to zero
            self.prey_belief_vector = np.array([1.0 / float(graph.size) for _ in range(graph.size + 1)])
            self.prey_belief_vector[0] = 0.0
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix (from one round to another round)
            # since how prey move from one node to next node doesn't depend on where agent or predator is
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            # assign probability in transition matrix by checking the quantity of a node's neighbors
            # since the prey chooses a node to move to uniformly at random
            # among its neighbors or its current standing node every time the prey moves
            for node in graph.graph[1:]:
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
        node_name_to_survey = random.choice(largest_belief_node_names)
        is_survey_needed = False
        if not self.is_bonus or is_survey_needed:
            self.is_surveyed = True
            # pick node with the largest belief to survey after knowing there's no node where prey certainly is
            node_to_survey = graph.graph[node_name_to_survey]
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
        return node_name_to_survey


class Agent4(AgentEven):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent4'
            self.prey_transition_matrix = None
            self.prey_belief_vector = None
            self.is_bonus = kwargs['is_bonus']

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.prey_transition_matrix = deepcopy(self.prey_transition_matrix)
        d_copy.prey_belief_vector = deepcopy(self.prey_belief_vector)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get the likely location of the prey
        likely_prey_node_name = self.get_likely_prey_node_name(graph)
        self.estimated_prey_name = likely_prey_node_name
        self.estimated_predator_name = graph.predator.cur_node_name
        if self.is_bonus and self.is_surveyed:
            return self.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name(graph=graph, prey_node_name=likely_prey_node_name,
                                       predator_node_name=graph.predator.cur_node_name)

    def get_likely_prey_node_name(self, graph):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            # where every probability is set to zero
            self.prey_belief_vector = np.array([1.0 / float(graph.size) for _ in range(graph.size + 1)])
            self.prey_belief_vector[0] = 0.0
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix (from one round to another round)
            # since how prey move from one node to next node doesn't depend on where agent or predator is
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            # assign probability in transition matrix by checking the quantity of a node's neighbors
            # since the prey chooses a node to move to uniformly at random
            # among its neighbors or its current standing node every time the prey moves
            for node in graph.graph[1:]:
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
        node_name_to_survey = random.choice(largest_belief_node_names)
        is_survey_needed = False
        if not self.is_bonus or is_survey_needed:
            self.is_surveyed = True
            # pick node with the largest belief to survey after knowing there's no node where prey certainly is
            node_to_survey = graph.graph[node_name_to_survey]
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
        return node_name_to_survey


class Agent4WithFuturePrediction(AgentEven):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent4_with_future_prediction'
            self.prey_transition_matrix = None
            self.prey_belief_vector = None
            self.is_bonus = kwargs['is_bonus']
            self.future_prey_belief_vector = None

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.prey_transition_matrix = deepcopy(self.prey_transition_matrix)
        d_copy.prey_belief_vector = deepcopy(self.prey_belief_vector)
        d_copy.future_prey_belief_vector = deepcopy(self.future_prey_belief_vector)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get the likely location of the prey
        likely_prey_node_name = self.get_likely_prey_node_name(graph)
        self.estimated_prey_name = likely_prey_node_name
        self.estimated_predator_name = graph.predator.cur_node_name
        if self.is_bonus and self.is_surveyed:
            return self.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name(graph=graph, prey_node_name=likely_prey_node_name,
                                       predator_node_name=graph.predator.cur_node_name)

    def get_likely_prey_node_name(self, graph):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            # where every probability is set to zero
            self.prey_belief_vector = np.array([1.0 / float(graph.size) for _ in range(graph.size + 1)])
            self.prey_belief_vector[0] = 0.0
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix (from one round to another round)
            # since how prey move from one node to next node doesn't depend on where agent or predator is
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            # assign probability in transition matrix by checking the quantity of a node's neighbors
            # since the prey chooses a node to move to uniformly at random
            # among its neighbors or its current standing node every time the prey moves
            for node in graph.graph[1:]:
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
        node_name_to_survey = random.choice(largest_belief_node_names)
        is_survey_needed = False
        if not self.is_bonus or is_survey_needed:
            self.is_surveyed = True
            # pick node with the largest belief to survey after knowing there's no node where prey certainly is
            node_to_survey = graph.graph[node_name_to_survey]
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
        return node_name_to_survey


class Agent5(AgentOdd):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent5'
            self.predator_transition_matrix = None
            self.predator_belief_vector = None
            self.agent_to_every_node_distance = None
            self.is_bonus = kwargs['is_bonus']

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.predator_transition_matrix = deepcopy(self.predator_transition_matrix)
        d_copy.predator_belief_vector = deepcopy(self.predator_belief_vector)
        d_copy.agent_to_every_node_distance = deepcopy(self.agent_to_every_node_distance)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get distance from agent to every node
        self.get_proximity_to_agent(graph)
        # get the likely location of the predator
        likely_predator_node_name = self.get_likely_predator_node_name(graph)
        # get shortest distance from prey and shortest distance from predator to every node with BFS
        self.get_distance(graph=graph, prey_node_name=graph.prey.cur_node_name,
                          predator_node_name=likely_predator_node_name)
        self.estimated_prey_name = graph.prey.cur_node_name
        self.estimated_predator_name = likely_predator_node_name
        if self.is_bonus and self.is_surveyed:
            return self.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name_by_rules(graph)

    def get_likely_predator_node_name(self, graph):
        if self.predator_belief_vector is None:
            # initialize belief for each node
            # under the condition that agent start by knowing where predator is
            self.predator_belief_vector = np.array([0.0 for _ in range(graph.size + 1)])
            self.predator_belief_vector[graph.predator.cur_node_name] = 1.0
        if self.predator_transition_matrix is None:
            # initialize transition matrix for each node
            self.predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                        for _ in range(graph.size + 1)])
        else:
            # compute transition matrix (from one round to another round)
            # which isn't necessarily the same for every round
            # since the agent won't stay at same place all the time
            for node in graph.graph[1:]:
                likely_predator_neighbor_node_names = self.get_predator_neighbor_name_candidates \
                    (graph=graph, predator_node_name=node.name)
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                    else:
                        self.predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            self.predator_belief_vector = np.matmul(self.predator_transition_matrix, self.predator_belief_vector)
        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # check if there's a node where predator certainly is
        if is_certain:
            # survey no node since it is certain where predator is
            return largest_belief_node_name
        node_name_to_survey = largest_belief_node_name
        # is_survey_needed = self.is_survey_required(graph=graph,
        #                                            predator_node_name=node_name_to_survey,
        #                                            predator_belief_vector=self.predator_belief_vector,
        #                                            belief_threshold=0)
        is_survey_needed = self.is_survey_required_based_on_distance(graph,
                                                                     self.cur_node_name, node_name_to_survey,
                                                                     13)
        if not self.is_bonus or is_survey_needed:
            self.is_surveyed = True
            self.survey_count += 1
            # pick node with the largest belief to survey after knowing there's no node where predator certainly is
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            node_to_survey = graph.graph[node_name_to_survey]
            is_predator_on_node = node_to_survey.is_predator_existed()
            if is_predator_on_node:
                # update belief vector after knowing the surveyed node has predator
                self.predator_belief_vector[node_name_to_survey] = 1.0
                for i in range(1, graph.size + 1):
                    if i != node_name_to_survey:
                        self.predator_belief_vector[i] = 0.0
            else:
                # update belief vector after knowing the surveyed node has no predator
                self.predator_belief_vector[node_name_to_survey] = 0.0
                total_beliefs = 0.0
                for belief in self.predator_belief_vector:
                    total_beliefs += belief
                for i in range(len(self.predator_belief_vector)):
                    self.predator_belief_vector[i] = float(self.predator_belief_vector[i] / total_beliefs)
            largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
            # pick node with the largest belief for treating it as where predator is for this round
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            return largest_belief_node_name
        return node_name_to_survey

    def get_predator_neighbor_name_candidates(self, graph, predator_node_name):
        neighbors = graph.graph[predator_node_name].neighbors
        shortest_distance = self.agent_to_every_node_distance[neighbors[0].name]
        for neighbor in neighbors[1:]:
            if self.agent_to_every_node_distance[neighbor.name] < shortest_distance:
                shortest_distance = self.agent_to_every_node_distance[neighbor.name]
        # choose the neighbors with the shortest distance to prey node
        next_node_name_candidates = set([neighbor.name for neighbor in neighbors
                                         if self.agent_to_every_node_distance[neighbor.name] == shortest_distance])
        return next_node_name_candidates

    def get_proximity_to_agent(self, graph):
        # get distance from agent to each node
        self.agent_to_every_node_distance = [INVALID for _ in range(graph.size + 1)]
        queue = deque([graph.graph[self.cur_node_name]])
        self.agent_to_every_node_distance[self.cur_node_name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = self.agent_to_every_node_distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.agent_to_every_node_distance[neighbor_node.name] == INVALID:
                    self.agent_to_every_node_distance[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)

    def get_largest_belief_node_name(self):
        largest_belief = self.predator_belief_vector[np.argmax(self.predator_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.predator_belief_vector) if
                                     belief == largest_belief]
        # check if there's a node where predator certainly is
        if largest_belief == 1:
            # survey no node since it is certain where predator is
            return random.choice(largest_belief_node_names), True

        # pick node with the largest belief to survey or to move
        # after knowing there's no node where predator certainly is
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        to_agent_distances = []
        for node_name in largest_belief_node_names:
            to_agent_distances.append(self.agent_to_every_node_distance[node_name])
        max_to_agent_distances = max(to_agent_distances)
        largest_belief_and_shortest_distance_node_names = [largest_belief_node_names[name]
                                                           for name, distance in enumerate(to_agent_distances)
                                                           if distance == max_to_agent_distances]
        return random.choice(largest_belief_and_shortest_distance_node_names), False


class Agent6(AgentEven):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent6'
            self.predator_transition_matrix = None
            self.predator_belief_vector = None
            self.agent_to_every_node_distance = None
            self.is_bonus = kwargs['is_bonus']

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.predator_transition_matrix = deepcopy(self.predator_transition_matrix)
        d_copy.predator_belief_vector = deepcopy(self.predator_belief_vector)
        d_copy.agent_to_every_node_distance = deepcopy(self.agent_to_every_node_distance)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get distance from agent to every node
        self.get_proximity_to_agent(graph)
        # get the likely location of the predator
        likely_predator_node_name = self.get_likely_predator_node_name(graph)
        self.estimated_prey_name = graph.prey.cur_node_name
        self.estimated_predator_name = likely_predator_node_name
        if self.is_bonus and self.is_surveyed:
            return self.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name(graph=graph, prey_node_name=graph.prey.cur_node_name,
                                       predator_node_name=likely_predator_node_name)

    def get_likely_predator_node_name(self, graph):
        if self.predator_belief_vector is None:
            # initialize belief for each node
            # under the condition that agent start by knowing where predator is
            self.predator_belief_vector = np.array([0.0 for _ in range(graph.size + 1)])
            self.predator_belief_vector[graph.predator.cur_node_name] = 1.0
        if self.predator_transition_matrix is None:
            # initialize transition matrix for each node
            self.predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                        for _ in range(graph.size + 1)])
        else:
            # compute transition matrix from previous round to this round
            # which isn't necessarily the same for every round
            # since the agent won't stay at same place all the time
            for node in graph.graph[1:]:
                likely_predator_neighbor_node_names = self.get_predator_neighbor_name_candidates \
                    (graph=graph, predator_node_name=node.name)
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                    else:
                        self.predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            self.predator_belief_vector = np.matmul(self.predator_transition_matrix, self.predator_belief_vector)
        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # check if there's a node where predator certainly is
        if is_certain:
            # survey no node since it is certain where predator is
            return largest_belief_node_name
        node_name_to_survey = largest_belief_node_name
        is_survey_needed = self.is_survey_required(graph=graph,
                                                   predator_node_name=node_name_to_survey,
                                                   predator_belief_vector=self.predator_belief_vector,
                                                   belief_threshold=0)
        if not self.is_bonus or is_survey_needed:
            self.is_surveyed = True
            # pick node with the largest belief to survey after knowing there's no node where predator certainly is
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            node_to_survey = graph.graph[node_name_to_survey]
            is_predator_on_node = node_to_survey.is_predator_existed()
            if is_predator_on_node:
                # update belief vector after knowing the surveyed node has predator
                self.predator_belief_vector[node_name_to_survey] = 1.0
                for i in range(1, graph.size + 1):
                    if i != node_name_to_survey:
                        self.predator_belief_vector[i] = 0.0
            else:
                # update belief vector after knowing the surveyed node has no predator
                self.predator_belief_vector[node_name_to_survey] = 0.0
                total_beliefs = 0.0
                for belief in self.predator_belief_vector:
                    total_beliefs += belief
                for i in range(len(self.predator_belief_vector)):
                    self.predator_belief_vector[i] = float(self.predator_belief_vector[i] / total_beliefs)

            largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
            # pick node with the largest belief for treating it as where predator is for this round
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            return largest_belief_node_name
        return node_name_to_survey

    def get_predator_neighbor_name_candidates(self, graph, predator_node_name):
        neighbors = graph.graph[predator_node_name].neighbors
        shortest_distance = self.agent_to_every_node_distance[neighbors[0].name]
        for neighbor in neighbors[1:]:
            if self.agent_to_every_node_distance[neighbor.name] < shortest_distance:
                shortest_distance = self.agent_to_every_node_distance[neighbor.name]
        # choose the neighbors with the shortest distance to prey node
        next_node_name_candidates = set([neighbor.name for neighbor in neighbors
                                         if self.agent_to_every_node_distance[neighbor.name] == shortest_distance])
        return next_node_name_candidates

    def get_proximity_to_agent(self, graph):
        # get distance from agent to each node
        self.agent_to_every_node_distance = [INVALID for _ in range(graph.size + 1)]
        queue = deque([graph.graph[self.cur_node_name]])
        self.agent_to_every_node_distance[self.cur_node_name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = self.agent_to_every_node_distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.agent_to_every_node_distance[neighbor_node.name] == INVALID:
                    self.agent_to_every_node_distance[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)

    def get_largest_belief_node_name(self):
        largest_belief = self.predator_belief_vector[np.argmax(self.predator_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.predator_belief_vector) if
                                     belief == largest_belief]
        # check if there's a node where predator certainly is
        if largest_belief == 1:
            # survey no node since it is certain where predator is
            return random.choice(largest_belief_node_names), True

        # pick node with the largest belief to survey or to move
        # after knowing there's no node where predator certainly is
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        to_agent_distances = []
        for node_name in largest_belief_node_names:
            to_agent_distances.append(self.agent_to_every_node_distance[node_name])
        max_to_agent_distances = max(to_agent_distances)
        largest_belief_and_shortest_distance_node_names = [largest_belief_node_names[name]
                                                           for name, distance in enumerate(to_agent_distances)
                                                           if distance == max_to_agent_distances]
        return random.choice(largest_belief_and_shortest_distance_node_names), False

    def get_next_node_name(self, graph, prey_node_name, predator_node_name):
        cur_node = graph.graph[self.cur_node_name]
        predator_and_its_neighbor_node_names = {predator_node_name}
        for predator_neighbor in graph.graph[predator_node_name].neighbors:
            predator_and_its_neighbor_node_names.add(predator_neighbor.name)
        source_names = {cur_node.name}
        for node in cur_node.neighbors:
            source_names.add(node.name)
        # exclude the neighbors (of the agent) or the agent's current standing node
        # who was one neighbor of the predator or was just the predator
        candidate_source_names = source_names.difference(predator_and_its_neighbor_node_names)
        if len(candidate_source_names) == 0:
            candidate_source_names = source_names.difference({predator_node_name})
        min_distance = float('Inf')
        min_node_dis_and_names = []
        distances_from_neighbor_to_prey = []
        # compute the distance to the prey
        # (for each of the agent's neighbor and for the agent's current standing node)
        for candidate_source_name in candidate_source_names:
            self.get_distance(graph, candidate_source_name)
            if self.distance[prey_node_name] < min_distance:
                min_distance = self.distance[prey_node_name]
            distances_from_neighbor_to_prey.append(self.distance[prey_node_name])
        for i, candidate_source_name in enumerate(candidate_source_names):
            if distances_from_neighbor_to_prey[i] == min_distance:
                min_node_dis_and_names.append(candidate_source_name)
        # choose the node (from the agent current standing node and the agent's neighbors)
        # that wasn't excluded for not having the shortest distance to the prey
        return random.choice(min_node_dis_and_names)


class Agent6WithFuturePrediction(AgentEven):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent6_with_future_prediction'
            self.predator_transition_matrix = None
            self.predator_belief_vector = None
            self.agent_to_every_node_distance = None
            self.is_bonus = kwargs['is_bonus']
            self.future_predator_belief_vector = None

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.predator_transition_matrix = deepcopy(self.predator_transition_matrix)
        d_copy.predator_belief_vector = deepcopy(self.predator_belief_vector)
        d_copy.agent_to_every_node_distance = deepcopy(self.agent_to_every_node_distance)
        d_copy.future_predator_belief_vector = deepcopy(self.future_predator_belief_vector)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get distance from agent to every node
        self.get_proximity_to_agent(graph)
        # get the likely location of the predator
        likely_predator_node_name = self.get_likely_predator_node_name(graph)
        self.estimated_prey_name = graph.prey.cur_node_name
        self.estimated_predator_name = likely_predator_node_name
        if self.is_bonus and self.is_surveyed:
            return self.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name(graph=graph, prey_node_name=graph.prey.cur_node_name,
                                       predator_node_name=likely_predator_node_name)
        # return self.get_next_node_name_nonaggressive(graph=graph, prey_node_name=graph.prey.cur_node_name,
        #                                                 predator_node_name=likely_predator_node_name)

    def get_likely_predator_node_name(self, graph):
        if self.predator_belief_vector is None:
            # initialize belief for each node
            # under the condition that agent start by knowing where predator is
            self.predator_belief_vector = np.array([0.0 for _ in range(graph.size + 1)])
            self.predator_belief_vector[graph.predator.cur_node_name] = 1.0
        if self.predator_transition_matrix is None:
            # initialize transition matrix for each node
            self.predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                        for _ in range(graph.size + 1)])
        else:
            # compute transition matrix from previous round to this round
            # which isn't necessarily the same for every round
            # since the agent won't stay at same place all the time
            for node in graph.graph[1:]:
                likely_predator_neighbor_node_names = self.get_predator_neighbor_name_candidates \
                    (graph=graph, predator_node_name=node.name)
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                    else:
                        self.predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            self.predator_belief_vector = np.matmul(self.predator_transition_matrix, self.predator_belief_vector)
        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # check if there's a node where predator certainly is
        if is_certain:
            self.future_predator_belief_vector = self.predator_belief_vector[:]
            # survey no node since it is certain where predator is
            return largest_belief_node_name
        node_name_to_survey = largest_belief_node_name
        is_survey_needed = self.is_survey_required(graph=graph,
                                                   predator_node_name=node_name_to_survey,
                                                   predator_belief_vector=self.predator_belief_vector,
                                                   belief_threshold=0)
        if not self.is_bonus or is_survey_needed:
            self.is_surveyed = True
            # pick node with the largest belief to survey after knowing there's no node where predator certainly is
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            node_to_survey = graph.graph[node_name_to_survey]
            is_predator_on_node = node_to_survey.is_predator_existed()
            if is_predator_on_node:
                # update belief vector after knowing the surveyed node has predator
                self.predator_belief_vector[node_name_to_survey] = 1.0
                for i in range(1, graph.size + 1):
                    if i != node_name_to_survey:
                        self.predator_belief_vector[i] = 0.0
            else:
                # update belief vector after knowing the surveyed node has no predator
                self.predator_belief_vector[node_name_to_survey] = 0.0
                total_beliefs = 0.0
                for belief in self.predator_belief_vector:
                    total_beliefs += belief
                for i in range(len(self.predator_belief_vector)):
                    self.predator_belief_vector[i] = float(self.predator_belief_vector[i] / total_beliefs)

            # use where the likely predator will be for this round after the agent and the prey all finished their move
            # and treat it as the likely location of the predator for determining the node for the agent to move to
            largest_belief_node_name, is_certain \
                = self.get_future_largest_belief_predator_node_name(self.predator_belief_vector[:], graph)
            # pick node with the largest belief for treating it as where predator is for this round
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            return largest_belief_node_name
        return node_name_to_survey

    def get_predator_neighbor_name_candidates(self, graph, predator_node_name):
        neighbors = graph.graph[predator_node_name].neighbors
        shortest_distance = self.agent_to_every_node_distance[neighbors[0].name]
        for neighbor in neighbors[1:]:
            if self.agent_to_every_node_distance[neighbor.name] < shortest_distance:
                shortest_distance = self.agent_to_every_node_distance[neighbor.name]
        # choose the neighbors with the shortest distance to prey node
        next_node_name_candidates = set([neighbor.name for neighbor in neighbors
                                         if self.agent_to_every_node_distance[neighbor.name] == shortest_distance])
        return next_node_name_candidates

    def get_proximity_to_agent(self, graph):
        # get distance from agent to each node
        self.agent_to_every_node_distance = [INVALID for _ in range(graph.size + 1)]
        queue = deque([graph.graph[self.cur_node_name]])
        self.agent_to_every_node_distance[self.cur_node_name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = self.agent_to_every_node_distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.agent_to_every_node_distance[neighbor_node.name] == INVALID:
                    self.agent_to_every_node_distance[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)

    def get_largest_belief_node_name(self):
        largest_belief = self.predator_belief_vector[np.argmax(self.predator_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.predator_belief_vector) if
                                     belief == largest_belief]
        # check if there's a node where predator certainly is
        if largest_belief == 1:
            # survey no node since it is certain where predator is
            return random.choice(largest_belief_node_names), True

        # pick node with the largest belief to survey or to move
        # after knowing there's no node where predator certainly is
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        to_agent_distances = []
        for node_name in largest_belief_node_names:
            to_agent_distances.append(self.agent_to_every_node_distance[node_name])
        max_to_agent_distances = max(to_agent_distances)
        largest_belief_and_shortest_distance_node_names = [largest_belief_node_names[name]
                                                           for name, distance in enumerate(to_agent_distances)
                                                           if distance == max_to_agent_distances]
        return random.choice(largest_belief_and_shortest_distance_node_names), False

    def get_future_largest_belief_predator_node_name(self, predator_belief_vector, graph):
        agent_and_its_neighbors = {self.cur_node_name}
        for node in graph.graph[self.cur_node_name].neighbors:
            agent_and_its_neighbors.add(node.name)
        all_largest_belief_and_shortest_distance_node_names = []
        corresponding_future_agent_names = {node_name: [] for node_name in agent_and_its_neighbors}
        for node_name in agent_and_its_neighbors:
            predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                   for _ in range(graph.size + 1)])
            # compute transition matrix from previous round to this round
            self.get_distance(graph, node_name)
            for node in graph.graph[1:]:
                neighbors = graph.graph[node.name].neighbors
                shortest_distance = self.distance[neighbors[0].name]
                for neighbor in neighbors[1:]:
                    if self.distance[neighbor.name] < shortest_distance:
                        shortest_distance = self.distance[neighbor.name]
                likely_predator_neighbor_node_names = set([neighbor.name for neighbor in neighbors
                                                           if self.distance[neighbor.name] == shortest_distance])
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                    else:
                        predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            predator_belief_vector = np.matmul(predator_transition_matrix, predator_belief_vector)

            largest_belief = predator_belief_vector[np.argmax(predator_belief_vector)]
            largest_belief_node_names = [i for i, belief in
                                         enumerate(predator_belief_vector) if belief == largest_belief]
            # check if there's a node where predator certainly is
            if largest_belief == 1:
                # survey no node since it is certain where predator is
                return random.choice(largest_belief_node_names), True

            # pick node with the largest belief to move to
            # after knowing there's no node where predator certainly is
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            to_agent_distances = []
            for largest_belief_node_name in largest_belief_node_names:
                to_agent_distances.append(self.distance[largest_belief_node_name])
            max_to_agent_distances = max(to_agent_distances)
            largest_belief_and_shortest_distance_node_names = [largest_belief_node_names[name]
                                                               for name, distance
                                                               in enumerate(to_agent_distances)
                                                               if distance == max_to_agent_distances]
            all_largest_belief_and_shortest_distance_node_names.extend(largest_belief_and_shortest_distance_node_names)
            corresponding_future_agent_names[node_name].extend(largest_belief_and_shortest_distance_node_names)
        all_largest_belief_and_shortest_distance_node_names_dict = defaultdict(int)
        max_count = 0
        for node_name in all_largest_belief_and_shortest_distance_node_names:
            all_largest_belief_and_shortest_distance_node_names_dict[node_name] += 1
            if all_largest_belief_and_shortest_distance_node_names_dict[node_name] > max_count:
                max_count = all_largest_belief_and_shortest_distance_node_names_dict[node_name]
        all_largest_belief_and_shortest_distance_node_names = []
        for k, v in all_largest_belief_and_shortest_distance_node_names_dict.items():
            if v == max_count:
                all_largest_belief_and_shortest_distance_node_names.append(k)
        likely_predator_node_name = random.choice(all_largest_belief_and_shortest_distance_node_names)
        # update future belief vector
        future_agent_candidates = []
        for k, v in corresponding_future_agent_names.items():
            if likely_predator_node_name in v:
                future_agent_candidates.append(k)
        future_agent_name = random.choice(future_agent_candidates)
        predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                               for _ in range(graph.size + 1)])
        self.get_distance_from_source(graph, future_agent_name)
        for node in graph.graph[1:]:
            neighbors = graph.graph[node.name].neighbors
            shortest_distance = self.distance[neighbors[0].name]
            for neighbor in neighbors[1:]:
                if self.distance[neighbor.name] < shortest_distance:
                    shortest_distance = self.distance[neighbor.name]
            likely_predator_neighbor_node_names = set([neighbor.name for neighbor in neighbors
                                                       if self.distance[neighbor.name] == shortest_distance])
            for neighbor in node.neighbors:
                if neighbor.name in likely_predator_neighbor_node_names:
                    self.predator_transition_matrix[neighbor.name][node.name] = \
                        (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                else:
                    self.predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
        self.future_predator_belief_vector = np.matmul(predator_transition_matrix, self.predator_belief_vector)
        return likely_predator_node_name, False

    def get_next_node_name(self, graph, prey_node_name, predator_node_name):
        cur_node = graph.graph[self.cur_node_name]
        predator_and_its_neighbor_node_names = {predator_node_name}
        for predator_neighbor in graph.graph[predator_node_name].neighbors:
            predator_and_its_neighbor_node_names.add(predator_neighbor.name)
        source_names = {cur_node.name}
        for node in cur_node.neighbors:
            source_names.add(node.name)
        # exclude the neighbors (of the agent) or the agent's current standing node
        # who was one neighbor of the predator or was just the predator
        candidate_source_names = source_names.difference(predator_and_its_neighbor_node_names)
        if len(candidate_source_names) == 0:
            candidate_source_names = source_names.difference({predator_node_name})
        min_distance = float('Inf')
        min_node_dis_and_names = []
        distances_from_neighbor_to_prey = []
        # compute the distance to the prey
        # (for each of the agent's neighbor and for the agent's current standing node)
        for candidate_source_name in candidate_source_names:
            self.get_distance(graph, candidate_source_name)
            if self.distance[prey_node_name] < min_distance:
                min_distance = self.distance[prey_node_name]
            distances_from_neighbor_to_prey.append(self.distance[prey_node_name])
        for i, candidate_source_name in enumerate(candidate_source_names):
            if distances_from_neighbor_to_prey[i] == min_distance:
                min_node_dis_and_names.append(candidate_source_name)
        # choose the node (from the agent current standing node and the agent's neighbors)
        # that wasn't excluded for not having the shortest distance to the prey
        return random.choice(min_node_dis_and_names)

    def get_next_node_name_nonaggressive(self, graph, prey_node_name, predator_node_name):
        cur_node = graph.graph[self.cur_node_name]

        # list neighbors' info about their names and distances to prey and distances to predator
        neighbors_to_prey_and_predator_info = []
        # record min and max of all shortest distances to predator (for all neighbors of the agent)
        # record min and max of all shortest distances to prey (for all neighbors of the agent)
        min_prey_distance = min_predator_distance = float('Inf')
        max_prey_distance = max_predator_distance = -1
        random_num_list = list(range(1, len(cur_node.neighbors) + 1))
        random.shuffle(random_num_list)
        for i, neighbor in enumerate(cur_node.neighbors):
            neighbor_name = neighbor.name
            self.get_distance_from_source(graph, neighbor_name)
            to_prey_distance = self.distance[neighbor_name]
            expected_to_predator_distance = sum([(self.future_predator_belief_vector[i]
                                                  * self.distance[i]) for i in range(1, 51)])
            if to_prey_distance < min_prey_distance:
                min_prey_distance = to_prey_distance
            if to_prey_distance > max_prey_distance:
                max_prey_distance = to_prey_distance
            if expected_to_predator_distance < min_predator_distance:
                min_predator_distance = expected_to_predator_distance
            if expected_to_predator_distance > max_predator_distance:
                max_predator_distance = expected_to_predator_distance
            neighbors_to_prey_and_predator_info.append(
                (to_prey_distance, expected_to_predator_distance,
                 random_num_list[i], neighbor_name))
        # get used for classifying up to three classes of neighbors:
        # those closer to the prey,
        # those not closer and not farther from the prey,
        # and those farther from the prey
        min_end_index = not_min_max_end_index = len(neighbors_to_prey_and_predator_info)
        neighbors_to_prey_and_predator_info.sort()
        for i, info in enumerate(neighbors_to_prey_and_predator_info):
            if min_end_index is None and info[0] != min_prey_distance:
                min_end_index = i
            if not_min_max_end_index is None and info[0] == max_prey_distance:
                not_min_max_end_index = i

        # examine neighbors based on rules
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

    def get_distance_from_source(self, graph, source_name):
        # compute distance from source node to every node in graph
        self.distance = [INVALID for _ in range(graph.size + 1)]
        source_node = graph.graph[source_name]
        queue = deque([source_node])
        self.distance[source_node.name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = self.distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.distance[neighbor_node.name] == INVALID:
                    self.distance[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)


class Agent7(AgentOdd):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent7'
            self.prey_transition_matrix = None
            self.prey_belief_vector = None
            self.predator_transition_matrix = None
            self.predator_belief_vector = None
            self.surveyed_node_name = None
            self.agent_to_every_node_distance = None
            self.is_bonus = kwargs['is_bonus']

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.prey_transition_matrix = deepcopy(self.prey_transition_matrix)
        d_copy.prey_belief_vector = deepcopy(self.prey_belief_vector)
        d_copy.predator_transition_matrix = deepcopy(self.predator_transition_matrix)
        d_copy.predator_belief_vector = deepcopy(self.predator_belief_vector)
        d_copy.surveyed_node_name = deepcopy(self.surveyed_node_name)
        d_copy.agent_to_every_node_distance = deepcopy(self.agent_to_every_node_distance)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get distance from agent to every node
        self.get_proximity_to_agent(graph)
        # get the likely location of the predator and
        # info about whether the location of the predator is known certainly
        likely_predator_node_name, is_certain_predator_location = self.get_likely_predator_node_name(graph)
        # get the likely location of the prey
        likely_prey_node_name = self.get_likely_prey_node_name(graph, is_certain_predator_location,
                                                               self.surveyed_node_name)
        # get shortest distance from prey and shortest distance from predator to every node with BFS
        self.get_distance(graph=graph, prey_node_name=likely_prey_node_name,
                          predator_node_name=likely_predator_node_name)
        self.estimated_prey_name = likely_prey_node_name
        self.estimated_predator_name = likely_predator_node_name
        if self.is_bonus and self.is_surveyed:
            return self.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name_by_rules(graph)

    def get_likely_predator_node_name(self, graph):
        if self.predator_belief_vector is None:
            # initialize belief for each node
            # under the condition that agent start by knowing where predator is
            self.predator_belief_vector = np.array([0.0 for _ in range(graph.size + 1)])
            self.predator_belief_vector[graph.predator.cur_node_name] = 1.0
        if self.predator_transition_matrix is None:
            # initialize transition matrix for each node
            self.predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                        for _ in range(graph.size + 1)])
        else:
            # compute transition matrix from previous round to this round
            for node in graph.graph[1:]:
                likely_predator_neighbor_node_names = self.get_predator_neighbor_name_candidates \
                    (graph=graph, predator_node_name=node.name)
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                    else:
                        self.predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            self.predator_belief_vector = np.matmul(self.predator_transition_matrix, self.predator_belief_vector)
        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # check if there's a node where predator certainly is
        if is_certain:
            # survey no node since it is certain where predator is
            return largest_belief_node_name, True
        node_name_to_survey = largest_belief_node_name
        is_survey_needed = self.is_survey_required(graph=graph,
                                                   predator_node_name=node_name_to_survey,
                                                   predator_belief_vector=self.predator_belief_vector,
                                                   belief_threshold=0)
        self.is_surveyed = is_survey_needed
        if not self.is_bonus or is_survey_needed:
            # pick node with the largest belief to survey
            # after knowing there's no node where predator certainly is
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            self.surveyed_node_name = node_name_to_survey
            node_to_survey = graph.graph[node_name_to_survey]
            is_predator_on_node = node_to_survey.is_predator_existed()
            if is_predator_on_node:
                # update belief vector after knowing the surveyed node has predator
                self.predator_belief_vector[node_name_to_survey] = 1.0
                for i in range(1, graph.size + 1):
                    if i != node_name_to_survey:
                        self.predator_belief_vector[i] = 0.0
            else:
                # update belief vector after knowing the surveyed node has no predator
                self.predator_belief_vector[node_name_to_survey] = 0.0
                total_beliefs = 0.0
                for belief in self.predator_belief_vector:
                    total_beliefs += belief
                for i in range(len(self.predator_belief_vector)):
                    self.predator_belief_vector[i] = float(self.predator_belief_vector[i] / total_beliefs)

            largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
            # pick node with the largest belief for treating it as where predator is for this round
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            return largest_belief_node_name, False
        return node_name_to_survey, False

    def get_predator_neighbor_name_candidates(self, graph, predator_node_name):
        neighbors = graph.graph[predator_node_name].neighbors
        shortest_distance = self.agent_to_every_node_distance[neighbors[0].name]
        for neighbor in neighbors[1:]:
            if self.agent_to_every_node_distance[neighbor.name] < shortest_distance:
                shortest_distance = self.agent_to_every_node_distance[neighbor.name]
        # choose the neighbors with the shortest distance to prey node
        next_node_name_candidates = set([neighbor.name for neighbor in neighbors
                                         if self.agent_to_every_node_distance[neighbor.name] == shortest_distance])
        return next_node_name_candidates

    def get_proximity_to_agent(self, graph):
        # get distance from agent to each node
        self.agent_to_every_node_distance = [INVALID for _ in range(graph.size + 1)]
        queue = deque([graph.graph[self.cur_node_name]])
        self.agent_to_every_node_distance[self.cur_node_name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = self.agent_to_every_node_distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.agent_to_every_node_distance[neighbor_node.name] == INVALID:
                    self.agent_to_every_node_distance[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)

    def get_largest_belief_node_name(self):
        largest_belief = self.predator_belief_vector[np.argmax(self.predator_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.predator_belief_vector)
                                     if belief == largest_belief]
        # check if there's a node where predator certainly is
        if largest_belief == 1:
            # survey no node since it is certain where predator is
            return random.choice(largest_belief_node_names), True

        # pick node with the largest belief to survey or to move to
        # after knowing there's no node where predator certainly is
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        to_agent_distances = []
        for node_name in largest_belief_node_names:
            to_agent_distances.append(self.agent_to_every_node_distance[node_name])
        max_to_agent_distances = max(to_agent_distances)
        largest_belief_and_shortest_distance_node_names = [largest_belief_node_names[name]
                                                           for name, distance in enumerate(to_agent_distances)
                                                           if distance == max_to_agent_distances]
        return random.choice(largest_belief_and_shortest_distance_node_names), False

    def get_likely_prey_node_name(self, graph, is_certain_predator_location, surveyed_node_name):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            self.prey_belief_vector = np.array([1.0 / float(graph.size + 1) for _ in range(graph.size + 1)])
            self.prey_belief_vector[0] = 0.0
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix from any round to any round
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            for node in graph.graph[1:]:
                for neighbor in node.neighbors:
                    self.prey_transition_matrix[neighbor.name][node.name] = 1.0 / (1.0 + float(len(node.neighbors)))
        else:
            # predict belief of where a prey is for each node for this round
            # by using the previous round's belief vector and unchanged transition matrix from any round to any round
            self.prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # check if there's a node where prey certainly is
        if largest_belief == 1:
            # survey no node since it is certain where prey is
            return random.choice(largest_belief_node_names)

        # pick node with the largest belief to survey after knowing there's no node where prey certainly is
        if is_certain_predator_location:
            node_name_to_survey = random.choice(largest_belief_node_names)
            is_survey_needed = False
        else:
            node_name_to_survey = surveyed_node_name
            is_survey_needed = self.is_surveyed
        self.is_surveyed = is_survey_needed
        if not self.is_bonus or is_survey_needed:
            node_to_survey = graph.graph[node_name_to_survey]
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
            largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if
                                         belief == largest_belief]
            # pick node with the largest belief for treating it as where prey is for this round
            return random.choice(largest_belief_node_names)
        return random.choice(largest_belief_node_names)


class Agent8(Agent7):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent8'
            self.prey_transition_matrix = None
            self.prey_belief_vector = None
            self.predator_transition_matrix = None
            self.predator_belief_vector = None
            self.surveyed_node_name = None
            self.agent_to_every_node_distance = None
            self.distance = None
            self.is_bonus = kwargs['is_bonus']

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.prey_transition_matrix = deepcopy(self.prey_transition_matrix)
        d_copy.prey_belief_vector = deepcopy(self.prey_belief_vector)
        d_copy.predator_transition_matrix = deepcopy(self.predator_transition_matrix)
        d_copy.predator_belief_vector = deepcopy(self.predator_belief_vector)
        d_copy.surveyed_node_name = deepcopy(self.surveyed_node_name)
        d_copy.agent_to_every_node_distance = deepcopy(self.agent_to_every_node_distance)
        d_copy.distance = deepcopy(self.distance)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get distance from agent to every node
        self.get_proximity_to_agent(graph)
        # get the likely location of the predator and
        # info about whether the location of the predator is known certainly
        likely_predator_node_name, is_certain_predator_location = self.get_likely_predator_node_name(graph)
        # get the likely location of the prey
        likely_prey_node_name = self.get_likely_prey_node_name(graph, is_certain_predator_location,
                                                               self.surveyed_node_name)
        # get shortest distance from prey and shortest distance from predator to every node with BFS
        self.get_distance(graph=graph, prey_node_name=likely_prey_node_name,
                          predator_node_name=likely_predator_node_name)
        self.estimated_prey_name = likely_prey_node_name
        self.estimated_predator_name = likely_predator_node_name
        if self.is_bonus and self.is_surveyed:
            return self.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name(graph=graph, prey_node_name=likely_prey_node_name,
                                       predator_node_name=likely_predator_node_name)

    def get_likely_predator_node_name(self, graph):
        if self.predator_belief_vector is None:
            # initialize belief for each node
            # under the condition that agent start by knowing where predator is
            self.predator_belief_vector = np.array([0.0 for _ in range(graph.size + 1)])
            self.predator_belief_vector[graph.predator.cur_node_name] = 1.0
        if self.predator_transition_matrix is None:
            # initialize transition matrix for each node
            self.predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                        for _ in range(graph.size + 1)])
        else:
            # compute transition matrix from previous round to this round
            for node in graph.graph[1:]:
                likely_predator_neighbor_node_names = self.get_predator_neighbor_name_candidates \
                    (graph=graph, predator_node_name=node.name)
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                    else:
                        self.predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            self.predator_belief_vector = np.matmul(self.predator_transition_matrix, self.predator_belief_vector)
        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # check if there's a node where predator certainly is
        if is_certain:
            # survey no node since it is certain where predator is
            return largest_belief_node_name, True
        node_name_to_survey = largest_belief_node_name
        is_survey_needed = self.is_survey_required(graph=graph,
                                                   predator_node_name=node_name_to_survey,
                                                   predator_belief_vector=self.predator_belief_vector,
                                                   belief_threshold=0)
        self.is_surveyed = is_survey_needed
        if not self.is_bonus or is_survey_needed:
            # pick node with the largest belief to survey
            # after knowing there's no node where predator certainly is
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            self.surveyed_node_name = node_name_to_survey
            node_to_survey = graph.graph[node_name_to_survey]
            is_predator_on_node = node_to_survey.is_predator_existed()
            if is_predator_on_node:
                # update belief vector after knowing the surveyed node has predator
                self.predator_belief_vector[node_name_to_survey] = 1.0
                for i in range(1, graph.size + 1):
                    if i != node_name_to_survey:
                        self.predator_belief_vector[i] = 0.0
            else:
                # update belief vector after knowing the surveyed node has no predator
                self.predator_belief_vector[node_name_to_survey] = 0.0
                total_beliefs = 0.0
                for belief in self.predator_belief_vector:
                    total_beliefs += belief
                for i in range(len(self.predator_belief_vector)):
                    self.predator_belief_vector[i] = float(self.predator_belief_vector[i] / total_beliefs)

            largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
            # pick node with the largest belief for treating it as where predator is for this round
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            return largest_belief_node_name, False
        return node_name_to_survey, False

    def get_likely_prey_node_name(self, graph, is_certain_predator_location, surveyed_node_name):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            self.prey_belief_vector = np.array([1.0 / float(graph.size + 1) for _ in range(graph.size + 1)])
            self.prey_belief_vector[0] = 0.0
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix from any round to any round
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            for node in graph.graph[1:]:
                for neighbor in node.neighbors:
                    self.prey_transition_matrix[neighbor.name][node.name] = 1.0 / (1.0 + float(len(node.neighbors)))
        else:
            # predict belief of where a prey is for each node for this round
            # by using the previous round's belief vector and unchanged transition matrix from any round to any round
            self.prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # check if there's a node where prey certainly is
        if largest_belief == 1:
            # survey no node since it is certain where prey is
            return random.choice(largest_belief_node_names)

        # pick node with the largest belief to survey after knowing there's no node where prey certainly is
        if is_certain_predator_location:
            node_name_to_survey = random.choice(largest_belief_node_names)
            is_survey_needed = False
        else:
            node_name_to_survey = surveyed_node_name
            is_survey_needed = self.is_surveyed
        self.is_surveyed = is_survey_needed
        if not self.is_bonus or is_survey_needed:
            node_to_survey = graph.graph[node_name_to_survey]
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
            # pick node with the largest belief for treating it as where prey is for this round
            return random.choice(largest_belief_node_names)
        return random.choice(largest_belief_node_names)

    def get_distance_from_source(self, graph, source_name):
        # compute distance from source node to every node in graph
        self.distance = [INVALID for _ in range(graph.size + 1)]
        source_node = graph.graph[source_name]
        queue = deque([source_node])
        self.distance[source_node.name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = self.distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.distance[neighbor_node.name] == INVALID:
                    self.distance[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)

    def get_next_node_name(self, graph, prey_node_name, predator_node_name):
        cur_node = graph.graph[self.cur_node_name]

        # record neighbors' info about their distances to prey, their distances to predator,
        # a random number which is for breaking ties, and their names (as tuples in a list)
        source_names = set()
        for node in cur_node.neighbors:
            source_names.add(node.name)
        prey_and_its_neighbor_node_names = {prey_node_name}
        for prey_neighbor in graph.graph[prey_node_name].neighbors:
            prey_and_its_neighbor_node_names.add(prey_neighbor.name)
        predator_and_its_neighbor_node_names = {predator_node_name}
        for predator_neighbor in graph.graph[predator_node_name].neighbors:
            predator_and_its_neighbor_node_names.add(predator_neighbor.name)
        neighbors_to_prey_and_predator_info = []
        # record min and max of all shortest distances to predator (for all neighbors of the agent)
        # record min and max of all shortest distances to prey (for all neighbors of the agent)
        min_prey_distance = min_predator_distance = float('Inf')
        max_prey_distance = max_predator_distance = -1
        random_num_list = list(range(1, len(cur_node.neighbors) + 1))
        random.shuffle(random_num_list)
        for i, node_name in enumerate(source_names):
            self.get_distance_from_source(graph, node_name)
            total_distance_for_prey = 0
            total_distance_for_predator = 0
            for prey_or_neighbor_node_name in prey_and_its_neighbor_node_names:
                total_distance_for_prey += self.distance[prey_or_neighbor_node_name]
                if total_distance_for_prey < min_prey_distance:
                    min_prey_distance = total_distance_for_prey
                if total_distance_for_prey > max_prey_distance:
                    max_prey_distance = total_distance_for_prey
            for predator_or_neighbor_node_name in predator_and_its_neighbor_node_names:
                total_distance_for_predator += self.distance[predator_or_neighbor_node_name]
                if total_distance_for_predator < min_predator_distance:
                    min_predator_distance = total_distance_for_predator
                if total_distance_for_predator > max_predator_distance:
                    max_predator_distance = total_distance_for_predator
            neighbors_to_prey_and_predator_info.append(
                (total_distance_for_prey, total_distance_for_predator,
                 random_num_list[i], node_name))
        # get used for classifying up to three classes of neighbors:
        # those closer to the prey,
        # those not closer and not farther from the prey,
        # and those farther from the prey
        min_end_index = not_min_max_end_index = len(neighbors_to_prey_and_predator_info)
        neighbors_to_prey_and_predator_info.sort()
        for i, info in enumerate(neighbors_to_prey_and_predator_info):
            if min_end_index is None and info[0] != min_prey_distance:
                min_end_index = i
            if not_min_max_end_index is None and info[0] == max_prey_distance:
                not_min_max_end_index = i

        # examine neighbors based on rules
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


class Agent8WithFuturePrediction(Agent7):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent8_with_future_prediction'
            self.prey_transition_matrix = None
            self.prey_belief_vector = None
            self.predator_transition_matrix = None
            self.predator_belief_vector = None
            self.surveyed_node_name = None
            self.agent_to_every_node_distance = None
            self.distance = None
            self.is_bonus = kwargs['is_bonus']
            self.future_prey_belief_vector = None
            self.future_predator_belief_vector = None

    def __deepcopy__(self, memodict=None):
        d_copy = super().__deepcopy__(memodict)
        d_copy.prey_transition_matrix = deepcopy(self.prey_transition_matrix)
        d_copy.prey_belief_vector = deepcopy(self.prey_belief_vector)
        d_copy.predator_transition_matrix = deepcopy(self.predator_transition_matrix)
        d_copy.predator_belief_vector = deepcopy(self.predator_belief_vector)
        d_copy.surveyed_node_name = deepcopy(self.surveyed_node_name)
        d_copy.agent_to_every_node_distance = deepcopy(self.agent_to_every_node_distance)
        d_copy.distance = deepcopy(self.distance)
        d_copy.future_prey_belief_vector = deepcopy(self.future_prey_belief_vector)
        d_copy.future_predator_belief_vector = deepcopy(self.future_predator_belief_vector)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get distance from agent to every node
        self.get_proximity_to_agent(graph)
        # get the likely location of the predator and
        # info about whether the location of the predator is known certainly
        likely_predator_node_name, is_certain_predator_location = self.get_likely_predator_node_name(graph)
        # get the likely location of the prey
        likely_prey_node_name = self.get_likely_prey_node_name(graph, is_certain_predator_location,
                                                               self.surveyed_node_name)
        # get shortest distance from prey and shortest distance from predator to every node with BFS
        self.get_distance(graph=graph, prey_node_name=likely_prey_node_name,
                          predator_node_name=likely_predator_node_name)
        self.estimated_prey_name = likely_prey_node_name
        self.estimated_predator_name = likely_predator_node_name
        if self.is_bonus and self.is_surveyed:
            return self.cur_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name(graph=graph, prey_node_name=likely_prey_node_name,
                                       predator_node_name=likely_predator_node_name)

    def get_likely_predator_node_name(self, graph):
        if self.predator_belief_vector is None:
            # initialize belief for each node
            # under the condition that agent start by knowing where predator is
            self.predator_belief_vector = np.array([0.0 for _ in range(graph.size + 1)])
            self.predator_belief_vector[graph.predator.cur_node_name] = 1.0
        if self.predator_transition_matrix is None:
            # initialize transition matrix for each node
            self.predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                        for _ in range(graph.size + 1)])
        else:
            # compute transition matrix from previous round to this round
            for node in graph.graph[1:]:
                likely_predator_neighbor_node_names = self.get_predator_neighbor_name_candidates \
                    (graph=graph, predator_node_name=node.name)
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                    else:
                        self.predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            self.predator_belief_vector = np.matmul(self.predator_transition_matrix, self.predator_belief_vector)
        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # check if there's a node where predator certainly is
        if is_certain:
            self.future_predator_belief_vector = self.predator_belief_vector[:]
            # survey no node since it is certain where predator is
            return largest_belief_node_name, True
        node_name_to_survey = largest_belief_node_name
        is_survey_needed = self.is_survey_required(graph=graph,
                                                   predator_node_name=node_name_to_survey,
                                                   predator_belief_vector=self.predator_belief_vector,
                                                   belief_threshold=0)
        self.is_surveyed = is_survey_needed
        if not self.is_bonus or is_survey_needed:
            # pick node with the largest belief to survey
            # after knowing there's no node where predator certainly is
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            self.surveyed_node_name = node_name_to_survey
            node_to_survey = graph.graph[node_name_to_survey]
            is_predator_on_node = node_to_survey.is_predator_existed()
            if is_predator_on_node:
                # update belief vector after knowing the surveyed node has predator
                self.predator_belief_vector[node_name_to_survey] = 1.0
                for i in range(1, graph.size + 1):
                    if i != node_name_to_survey:
                        self.predator_belief_vector[i] = 0.0
            else:
                # update belief vector after knowing the surveyed node has no predator
                self.predator_belief_vector[node_name_to_survey] = 0.0
                total_beliefs = 0.0
                for belief in self.predator_belief_vector:
                    total_beliefs += belief
                for i in range(len(self.predator_belief_vector)):
                    self.predator_belief_vector[i] = float(self.predator_belief_vector[i] / total_beliefs)

            # use where the likely predator will be for this round after the agent and the prey all finished their move
            # and treat it as the likely location of the predator for determining the node for the agent to move to
            largest_belief_node_name, is_certain \
                = self.get_future_largest_belief_predator_node_name(self.predator_belief_vector[:], graph)

            # pick node with the largest belief for treating it as where predator is for this round
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            return largest_belief_node_name, False
        return node_name_to_survey, False

    def get_future_largest_belief_predator_node_name(self, predator_belief_vector, graph):
        agent_and_its_neighbors = {self.cur_node_name}
        for node in graph.graph[self.cur_node_name].neighbors:
            agent_and_its_neighbors.add(node.name)
        all_largest_belief_and_shortest_distance_node_names = []
        corresponding_future_agent_names = {node_name: [] for node_name in agent_and_its_neighbors}
        for node_name in agent_and_its_neighbors:
            predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                   for _ in range(graph.size + 1)])
            # compute transition matrix from previous round to this round
            self.get_distance_from_source(graph, node_name)
            for node in graph.graph[1:]:
                neighbors = graph.graph[node.name].neighbors
                shortest_distance = self.distance[neighbors[0].name]
                for neighbor in neighbors[1:]:
                    if self.distance[neighbor.name] < shortest_distance:
                        shortest_distance = self.distance[neighbor.name]
                likely_predator_neighbor_node_names = set([neighbor.name for neighbor in neighbors
                                                           if self.distance[neighbor.name] == shortest_distance])
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                    else:
                        predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            predator_belief_vector = np.matmul(predator_transition_matrix, predator_belief_vector)
            largest_belief = predator_belief_vector[np.argmax(predator_belief_vector)]
            largest_belief_node_names = [i for i, belief in
                                         enumerate(predator_belief_vector) if belief == largest_belief]
            # pick node with the largest belief to move to
            # after knowing there's no node where predator certainly is
            # break ties when there are two or more belief with same highest value
            #  (based on proximity first, then random choice)
            to_agent_distances = []
            for largest_belief_node_name in largest_belief_node_names:
                to_agent_distances.append(self.distance[largest_belief_node_name])
            max_to_agent_distances = max(to_agent_distances)
            largest_belief_and_shortest_distance_node_names = [largest_belief_node_names[name]
                                                               for name, distance
                                                               in enumerate(to_agent_distances)
                                                               if distance == max_to_agent_distances]
            all_largest_belief_and_shortest_distance_node_names.extend(largest_belief_and_shortest_distance_node_names)
            corresponding_future_agent_names[node_name].extend(largest_belief_and_shortest_distance_node_names)
        all_largest_belief_and_shortest_distance_node_names_dict = defaultdict(int)
        max_count = 0
        for node_name in all_largest_belief_and_shortest_distance_node_names:
            all_largest_belief_and_shortest_distance_node_names_dict[node_name] += 1
            if all_largest_belief_and_shortest_distance_node_names_dict[node_name] > max_count:
                max_count = all_largest_belief_and_shortest_distance_node_names_dict[node_name]
        all_largest_belief_and_shortest_distance_node_names = []
        for k, v in all_largest_belief_and_shortest_distance_node_names_dict.items():
            if v == max_count:
                all_largest_belief_and_shortest_distance_node_names.append(k)
        likely_predator_node_name = random.choice(all_largest_belief_and_shortest_distance_node_names)
        # update future belief vector
        future_agent_candidates = []
        for k, v in corresponding_future_agent_names.items():
            if likely_predator_node_name in v:
                future_agent_candidates.append(k)
        future_agent_name = random.choice(future_agent_candidates)
        predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                               for _ in range(graph.size + 1)])
        self.get_distance_from_source(graph, future_agent_name)
        for node in graph.graph[1:]:
            neighbors = graph.graph[node.name].neighbors
            shortest_distance = self.distance[neighbors[0].name]
            for neighbor in neighbors[1:]:
                if self.distance[neighbor.name] < shortest_distance:
                    shortest_distance = self.distance[neighbor.name]
            likely_predator_neighbor_node_names = set([neighbor.name for neighbor in neighbors
                                                       if self.distance[neighbor.name] == shortest_distance])
            for neighbor in node.neighbors:
                if neighbor.name in likely_predator_neighbor_node_names:
                    self.predator_transition_matrix[neighbor.name][node.name] = \
                        (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                else:
                    self.predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
        self.future_predator_belief_vector = np.matmul(predator_transition_matrix, self.predator_belief_vector)
        return likely_predator_node_name, False

    def get_likely_prey_node_name(self, graph, is_certain_predator_location, surveyed_node_name):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            self.prey_belief_vector = np.array([1.0 / float(graph.size + 1) for _ in range(graph.size + 1)])
            self.prey_belief_vector[0] = 0.0
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix from any round to any round
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            for node in graph.graph[1:]:
                for neighbor in node.neighbors:
                    self.prey_transition_matrix[neighbor.name][node.name] = 1.0 / (1.0 + float(len(node.neighbors)))
        else:
            # predict belief of where a prey is for each node for this round
            # by using the previous round's belief vector and unchanged transition matrix from any round to any round
            self.prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # check if there's a node where prey certainly is
        if largest_belief == 1:
            # survey no node since it is certain where prey is
            return random.choice(largest_belief_node_names)

        # pick node with the largest belief to survey after knowing there's no node where prey certainly is
        if is_certain_predator_location:
            node_name_to_survey = random.choice(largest_belief_node_names)
            is_survey_needed = False
        else:
            node_name_to_survey = surveyed_node_name
            is_survey_needed = self.is_surveyed
        self.is_surveyed = is_survey_needed
        if not self.is_bonus or is_survey_needed:
            node_to_survey = graph.graph[node_name_to_survey]
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

            # use where the likely prey will be for this round after the agent finished its move
            # and treat it as the likely location of the prey for determining the node for the agent to move to
            self.future_prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
            largest_belief = self.future_prey_belief_vector[np.argmax(self.future_prey_belief_vector)]
            largest_belief_node_names = [i for i, belief in
                                         enumerate(self.future_prey_belief_vector) if belief == largest_belief]

            # pick node with the largest belief for treating it as where prey is for this round
            return random.choice(largest_belief_node_names)
        return random.choice(largest_belief_node_names)

    def get_distance_from_source(self, graph, source_name):
        # compute distance from source node to every node in graph
        self.distance = [INVALID for _ in range(graph.size + 1)]
        source_node = graph.graph[source_name]
        queue = deque([source_node])
        self.distance[source_node.name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = self.distance[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if self.distance[neighbor_node.name] == INVALID:
                    self.distance[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)

    def get_next_node_name(self, graph, prey_node_name, predator_node_name):
        cur_node = graph.graph[self.cur_node_name]

        # record neighbors' info about their distances to prey, their distances to predator,
        # a random number which is for breaking ties, and their names (as tuples in a list)
        neighbors_to_prey_and_predator_info = []
        # record min and max of all shortest distances to predator (for all neighbors of the agent)
        # record min and max of all shortest distances to prey (for all neighbors of the agent)
        min_prey_distance = min_predator_distance = float('Inf')
        max_prey_distance = max_predator_distance = -1
        random_num_list = list(range(1, len(cur_node.neighbors) + 1))
        random.shuffle(random_num_list)
        for i, neighbor in enumerate(cur_node.neighbors):
            neighbor_name = neighbor.name
            self.get_distance_from_source(graph, neighbor_name)
            expected_to_prey_distance = sum([(self.future_prey_belief_vector[i]
                                              * self.distance[i]) for i in range(1, 51)])
            expected_to_predator_distance = sum([(self.future_predator_belief_vector[i]
                                                  * self.distance[i]) for i in range(1, 51)])
            if expected_to_prey_distance < min_prey_distance:
                min_prey_distance = expected_to_prey_distance
            if expected_to_prey_distance > max_prey_distance:
                max_prey_distance = expected_to_prey_distance
            if expected_to_predator_distance < min_predator_distance:
                min_predator_distance = expected_to_predator_distance
            if expected_to_predator_distance > max_predator_distance:
                max_predator_distance = expected_to_predator_distance
            neighbors_to_prey_and_predator_info.append(
                (expected_to_prey_distance, expected_to_predator_distance,
                 random_num_list[i], neighbor_name))
        # get used for classifying up to three classes of neighbors:
        # those closer to the prey,
        # those not closer and not farther from the prey,
        # and those farther from the prey
        min_end_index = not_min_max_end_index = len(neighbors_to_prey_and_predator_info)
        neighbors_to_prey_and_predator_info.sort()
        for i, info in enumerate(neighbors_to_prey_and_predator_info):
            if min_end_index is None and info[0] != min_prey_distance:
                min_end_index = i
            if not_min_max_end_index is None and info[0] == max_prey_distance:
                not_min_max_end_index = i

        # examine neighbors based on rules
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


class ExtendedAgent7(Agent7):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'extended_agent7'

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get distance from agent to every node
        self.get_proximity_to_agent(graph)
        # determine if drone fail for this round
        is_failed_drone = 0 == random.randint(0, 9)
        likely_predator_node_name, is_certain_predator_location = self.get_likely_predator_node_name_with_broken_drone \
            (graph, is_failed_drone)
        likely_prey_node_name = self.get_likely_prey_node_name_with_broken_drone(graph, is_certain_predator_location,
                                                                                 self.surveyed_node_name,
                                                                                 is_failed_drone)
        self.estimated_prey_name = likely_prey_node_name
        self.estimated_predator_name = likely_predator_node_name
        # get shortest distance from prey and shortest distance from predator to every node with BFS
        self.get_distance(graph=graph, prey_node_name=likely_prey_node_name,
                          predator_node_name=likely_predator_node_name)
        # get next node for agent to move to after following some rule
        return self.get_next_node_name_by_rules(graph)

    def get_likely_predator_node_name_with_broken_drone(self, graph, is_failed_drone):
        if self.predator_belief_vector is None:
            # initialize belief for each node
            # under the condition that agent start by knowing where predator is
            self.predator_belief_vector = np.array([0.0 for _ in range(graph.size + 1)])
            self.predator_belief_vector[graph.predator.cur_node_name] = 1.0
        if self.predator_transition_matrix is None:
            # initialize transition matrix for each node
            self.predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                        for _ in range(graph.size + 1)])
        else:
            # compute transition matrix from previous round to this round
            for node in graph.graph[1:]:
                likely_predator_neighbor_node_names = self.get_predator_neighbor_name_candidates \
                    (graph=graph, predator_node_name=node.name)
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                    else:
                        self.predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            self.predator_belief_vector = np.matmul(self.predator_transition_matrix, self.predator_belief_vector)
        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # check if there's a node where predator certainly is
        if is_certain:
            # survey no node since it is certain where predator is
            return largest_belief_node_name, True

        # pick node with the largest belief to survey after knowing there's no node where predator certainly is
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        node_name_to_survey = largest_belief_node_name
        self.surveyed_node_name = node_name_to_survey
        node_to_survey = graph.graph[node_name_to_survey]
        is_predator_on_node = node_to_survey.is_predator_existed()
        if is_predator_on_node and is_failed_drone:
            is_predator_on_node = False
        if is_predator_on_node:
            # update belief vector after knowing the surveyed node has predator
            self.predator_belief_vector[node_name_to_survey] = 1.0
            for i in range(1, graph.size + 1):
                if i != node_name_to_survey:
                    self.predator_belief_vector[i] = 0.0
        else:
            # update belief vector after knowing the surveyed node has no predator
            self.predator_belief_vector[node_name_to_survey] = 0.0
            total_beliefs = 0.0
            for belief in self.predator_belief_vector:
                total_beliefs += belief
            for i in range(len(self.predator_belief_vector)):
                self.predator_belief_vector[i] = float(self.predator_belief_vector[i] / total_beliefs)

        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # pick node with the largest belief for treating it as where predator is for this round
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        return largest_belief_node_name, False

    def get_likely_prey_node_name_with_broken_drone(self, graph, is_certain_predator_location,
                                                    surveyed_node_name, is_failed_drone):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            self.prey_belief_vector = np.array([1.0 / float(graph.size + 1) for _ in range(graph.size + 1)])
            self.prey_belief_vector[0] = 0.0
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix from any round to any round
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            for node in graph.graph[1:]:
                for neighbor in node.neighbors:
                    self.prey_transition_matrix[neighbor.name][node.name] = 1.0 / (1.0 + float(len(node.neighbors)))
        else:
            # predict belief of where a prey is for each node for this round
            # by using the previous round's belief vector and unchanged transition matrix from any round to any round
            self.prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # check if there's a node where prey certainly is
        if largest_belief == 1:
            # survey no node since it is certain where prey is
            return random.choice(largest_belief_node_names)

        # pick node with the largest belief to survey after knowing there's no node where prey certainly is
        if is_certain_predator_location:
            node_name_to_survey = random.choice(largest_belief_node_names)
        else:
            node_name_to_survey = surveyed_node_name
        node_to_survey = graph.graph[node_name_to_survey]
        is_prey_on_node = node_to_survey.is_prey_existed()
        if is_prey_on_node and is_failed_drone:
            is_prey_on_node = False
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
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # pick node with the largest belief for treating it as where prey is for this round
        return random.choice(largest_belief_node_names)


class ExtendedAgent8(Agent8):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'extended_agent8'

    def write(self, f):
        pass

    def read(self, f):
        pass

    def next_node_name(self, graph):
        # get distance from agent to every node
        self.get_proximity_to_agent(graph)
        # determine if drone fail for this round
        is_failed_drone = 0 == random.randint(0, 9)
        likely_predator_node_name, is_certain_predator_location = self.get_likely_predator_node_name_with_broken_drone \
            (graph, is_failed_drone)
        likely_prey_node_name = self.get_likely_prey_node_name_with_broken_drone(graph, is_certain_predator_location,
                                                                                 self.surveyed_node_name,
                                                                                 is_failed_drone)
        self.estimated_prey_name = likely_prey_node_name
        self.estimated_predator_name = likely_predator_node_name
        # get next node for agent to move to after following some rule
        return self.get_next_node_name(graph=graph, prey_node_name=likely_prey_node_name,
                                       predator_node_name=likely_predator_node_name)

    def get_likely_predator_node_name_with_broken_drone(self, graph, is_failed_drone):
        if self.predator_belief_vector is None:
            # initialize belief for each node
            # under the condition that agent start by knowing where predator is
            self.predator_belief_vector = np.array([0.0 for _ in range(graph.size + 1)])
            self.predator_belief_vector[graph.predator.cur_node_name] = 1.0
        if self.predator_transition_matrix is None:
            # initialize transition matrix for each node
            self.predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                        for _ in range(graph.size + 1)])
        else:
            # compute transition matrix from previous round to this round
            for node in graph.graph[1:]:
                likely_predator_neighbor_node_names = self.get_predator_neighbor_name_candidates \
                    (graph=graph, predator_node_name=node.name)
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 \
                            + (1 / len(node.neighbors)) * 0.4
                    else:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(node.neighbors)) * 0.4

            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            self.predator_belief_vector = np.matmul(self.predator_transition_matrix, self.predator_belief_vector)
        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # check if there's a node where predator certainly is
        if is_certain:
            # survey no node since it is certain where predator is
            return largest_belief_node_name, True

        # pick node with the largest belief to survey after knowing there's no node where predator certainly is
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        largest_belief = self.predator_belief_vector[np.argmax(self.predator_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.predator_belief_vector)
                                     if belief == largest_belief]
        to_agent_distances = []
        for node_name in largest_belief_node_names:
            to_agent_distances.append(self.agent_to_every_node_distance[node_name])
        max_to_agent_distances = max(to_agent_distances)
        largest_belief_and_shortest_distance_node_names = [largest_belief_node_names[name]
                                                           for name, distance in enumerate(to_agent_distances)
                                                           if distance == max_to_agent_distances]
        node_name_to_survey = random.choice(largest_belief_and_shortest_distance_node_names)
        self.surveyed_node_name = node_name_to_survey
        node_to_survey = graph.graph[node_name_to_survey]
        is_predator_on_node = node_to_survey.is_predator_existed()
        if is_predator_on_node and is_failed_drone:
            is_predator_on_node = False
        if is_predator_on_node:
            # update belief vector after knowing the surveyed node has predator
            self.predator_belief_vector[node_name_to_survey] = 1.0
            for i in range(1, graph.size + 1):
                if i != node_name_to_survey:
                    self.predator_belief_vector[i] = 0.0
        else:
            # update belief vector after knowing the surveyed node has no predator
            self.predator_belief_vector[node_name_to_survey] = 0.0
            total_beliefs = 0.0
            for belief in self.predator_belief_vector:
                total_beliefs += belief
            for i in range(len(self.predator_belief_vector)):
                self.predator_belief_vector[i] = float(self.predator_belief_vector[i] / total_beliefs)

        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # pick node with the largest belief for treating it as where predator is for this round
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        return largest_belief_node_name, False

    def get_likely_prey_node_name_with_broken_drone(self, graph, is_certain_predator_location,
                                                    surveyed_node_name, is_failed_drone):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            self.prey_belief_vector = np.array([1.0 / float(graph.size + 1) for _ in range(graph.size + 1)])
            self.prey_belief_vector[0] = 0.0
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix from any round to any round
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            for node in graph.graph[1:]:
                for neighbor in node.neighbors:
                    self.prey_transition_matrix[neighbor.name][node.name] = 1.0 / (1.0 + float(len(node.neighbors)))
        else:
            # predict belief of where a prey is for each node for this round
            # by using the previous round's belief vector and unchanged transition matrix from any round to any round
            self.prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # check if there's a node where prey certainly is
        if largest_belief == 1:
            # survey no node since it is certain where prey is
            return random.choice(largest_belief_node_names)

        # pick node with the largest belief to survey after knowing there's no node where prey certainly is
        if is_certain_predator_location:
            node_name_to_survey = random.choice(largest_belief_node_names)
        else:
            node_name_to_survey = surveyed_node_name
        node_to_survey = graph.graph[node_name_to_survey]
        is_prey_on_node = node_to_survey.is_prey_existed()
        if is_prey_on_node and is_failed_drone:
            is_prey_on_node = False
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
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # pick node with the largest belief for treating it as where prey is for this round
        return random.choice(largest_belief_node_names)


class RevisedExtendedAgent7(ExtendedAgent7):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'revised_extended_agent7'

    def write(self, f):
        pass

    def read(self, f):
        pass

    def get_likely_predator_node_name_with_broken_drone(self, graph, is_failed_drone):
        if self.predator_belief_vector is None:
            # initialize belief for each node
            # under the condition that agent start by knowing where predator is
            self.predator_belief_vector = np.array([0.0 for _ in range(graph.size + 1)])
            self.predator_belief_vector[graph.predator.cur_node_name] = 1.0
        if self.predator_transition_matrix is None:
            # initialize transition matrix for each node
            self.predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                        for _ in range(graph.size + 1)])
        else:
            # compute transition matrix from previous round to this round
            for node in graph.graph[1:]:
                likely_predator_neighbor_node_names = self.get_predator_neighbor_name_candidates \
                    (graph=graph, predator_node_name=node.name)
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 + (1 / len(node.neighbors)) * 0.4
                    else:
                        self.predator_transition_matrix[neighbor.name][node.name] = (1 / len(node.neighbors)) * 0.4
            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            self.predator_belief_vector = np.matmul(self.predator_transition_matrix, self.predator_belief_vector)
        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # check if there's a node where predator certainly is
        if is_certain:
            # survey no node since it is certain where predator is
            return largest_belief_node_name, True

        # pick node with the largest belief to survey after knowing there's no node where predator certainly is
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        node_name_to_survey = largest_belief_node_name
        self.surveyed_node_name = node_name_to_survey
        node_to_survey = graph.graph[node_name_to_survey]
        is_predator_on_node = node_to_survey.is_predator_existed()
        if is_predator_on_node and is_failed_drone:
            is_predator_on_node = False
        if is_predator_on_node:
            # update belief vector after knowing the surveyed node has predator
            self.predator_belief_vector[node_name_to_survey] = 1.0
            for i in range(1, graph.size + 1):
                if i != node_name_to_survey:
                    self.predator_belief_vector[i] = 0.0
        else:
            # update belief vector after knowing the surveyed node has no predator
            # under the condition that drone may fail to tell the truth
            drone_failing_probability = self.predator_belief_vector[self.surveyed_node_name] * 0.1 + \
                                        (1 - self.predator_belief_vector[self.surveyed_node_name])
            for i, belief in enumerate(self.predator_belief_vector[1:]):
                self.predator_belief_vector[i + 1] = belief * 1.0 if (i + 1) != self.surveyed_node_name \
                    else belief * 0.1
            self.predator_belief_vector /= drone_failing_probability

        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # pick node with the largest belief for treating it as where predator is for this round
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        return largest_belief_node_name, False

    def get_likely_prey_node_name_with_broken_drone(self, graph, is_certain_predator_location,
                                                    surveyed_node_name, is_failed_drone):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            self.prey_belief_vector = np.array([1.0 / float(graph.size + 1) for _ in range(graph.size + 1)])
            self.prey_belief_vector[0] = 0.0
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix from any round to any round
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            for node in graph.graph[1:]:
                for neighbor in node.neighbors:
                    self.prey_transition_matrix[neighbor.name][node.name] = 1.0 / (1.0 + float(len(node.neighbors)))
        else:
            # predict belief of where a prey is for each node for this round
            # by using the previous round's belief vector and unchanged transition matrix from any round to any round
            self.prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # check if there's a node where prey certainly is
        if largest_belief == 1:
            # survey no node since it is certain where prey is
            return random.choice(largest_belief_node_names)

        # pick node with the largest belief to survey after knowing there's no node where prey certainly is
        if is_certain_predator_location:
            node_name_to_survey = random.choice(largest_belief_node_names)
        else:
            node_name_to_survey = surveyed_node_name
        self.surveyed_node_name = node_name_to_survey
        node_to_survey = graph.graph[node_name_to_survey]
        is_prey_on_node = node_to_survey.is_prey_existed()
        if is_prey_on_node and is_failed_drone:
            is_prey_on_node = False
        if is_prey_on_node:
            # update belief vector after knowing the surveyed node has prey
            self.prey_belief_vector[node_name_to_survey] = 1.0
            for i in range(1, graph.size + 1):
                if i != node_name_to_survey:
                    self.prey_belief_vector[i] = 0.0
        else:
            # update belief vector after knowing the surveyed node has no prey
            # under the condition that drone may fail to tell the truth
            drone_failing_probability = self.prey_belief_vector[self.surveyed_node_name] * 0.1 + \
                                        (1 - self.prey_belief_vector[self.surveyed_node_name])
            for i, belief in enumerate(self.prey_belief_vector[1:]):
                self.prey_belief_vector[i + 1] = belief * 1.0 if (i + 1) != self.surveyed_node_name \
                    else belief * 0.1
            self.prey_belief_vector /= drone_failing_probability if drone_failing_probability != 0 else 1.0

        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # pick node with the largest belief for treating it as where prey is for this round
        return random.choice(largest_belief_node_names)


class RevisedExtendedAgent8(ExtendedAgent8):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'revised_extended_agent8'

    def write(self, f):
        pass

    def read(self, f):
        pass

    def get_likely_predator_node_name_with_broken_drone(self, graph, is_failed_drone):
        if self.predator_belief_vector is None:
            # initialize belief for each node
            # under the condition that agent start by knowing where predator is
            self.predator_belief_vector = np.array([0.0 for _ in range(graph.size + 1)])
            self.predator_belief_vector[graph.predator.cur_node_name] = 1.0
        if self.predator_transition_matrix is None:
            # initialize transition matrix for each node
            self.predator_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                        for _ in range(graph.size + 1)])
        else:
            # compute transition matrix from previous round to this round
            for node in graph.graph[1:]:
                likely_predator_neighbor_node_names = self.get_predator_neighbor_name_candidates \
                    (graph=graph, predator_node_name=node.name)
                for neighbor in node.neighbors:
                    if neighbor.name in likely_predator_neighbor_node_names:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(likely_predator_neighbor_node_names)) * 0.6 \
                            + (1 / len(node.neighbors)) * 0.4
                    else:
                        self.predator_transition_matrix[neighbor.name][node.name] = \
                            (1 / len(node.neighbors)) * 0.4

            # predict belief of where a predator is for each node for this round
            # by using the previous round's belief vector and the transition matrix from previous round to this round
            self.predator_belief_vector = np.matmul(self.predator_transition_matrix, self.predator_belief_vector)
        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # check if there's a node where predator certainly is
        if is_certain:
            # survey no node since it is certain where predator is
            return largest_belief_node_name, True

        # pick node with the largest belief to survey after knowing there's no node where predator certainly is
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        node_name_to_survey = largest_belief_node_name
        self.surveyed_node_name = node_name_to_survey
        node_to_survey = graph.graph[node_name_to_survey]
        is_predator_on_node = node_to_survey.is_predator_existed()
        if is_predator_on_node and is_failed_drone:
            is_predator_on_node = False
        if is_predator_on_node:
            # update belief vector after knowing the surveyed node has predator
            self.predator_belief_vector[node_name_to_survey] = 1.0
            for i in range(1, graph.size + 1):
                if i != node_name_to_survey:
                    self.predator_belief_vector[i] = 0.0
        else:
            # update belief vector after knowing the surveyed node has no predator
            # under the condition that drone may fail to tell the truth
            drone_failing_probability = self.predator_belief_vector[self.surveyed_node_name] * 0.1 + \
                                        (1 - self.predator_belief_vector[self.surveyed_node_name])
            for i, belief in enumerate(self.predator_belief_vector[1:]):
                self.predator_belief_vector[i + 1] = belief * 1.0 if (i + 1) != self.surveyed_node_name \
                    else belief * 0.1
            self.predator_belief_vector /= drone_failing_probability

        largest_belief_node_name, is_certain = self.get_largest_belief_node_name()
        # pick node with the largest belief for treating it as where predator is for this round
        # break ties when there are two or more belief with same highest value
        #  (based on proximity first, then random choice)
        return largest_belief_node_name, False

    def get_likely_prey_node_name_with_broken_drone(self, graph, is_certain_predator_location,
                                                    surveyed_node_name, is_failed_drone):
        if self.prey_belief_vector is None:
            # initialize belief for each node
            self.prey_belief_vector = np.array([1.0 / float(graph.size + 1) for _ in range(graph.size + 1)])
            self.prey_belief_vector[0] = 0.0
        if self.prey_transition_matrix is None:
            # initialize unchanged transition matrix from any round to any round
            self.prey_transition_matrix = np.array([[0.0 for _ in range(graph.size + 1)]
                                                    for _ in range(graph.size + 1)])
            for node in graph.graph[1:]:
                for neighbor in node.neighbors:
                    self.prey_transition_matrix[neighbor.name][node.name] = 1.0 / (1.0 + float(len(node.neighbors)))
        else:
            # predict belief of where a prey is for each node for this round
            # by using the previous round's belief vector and unchanged transition matrix from any round to any round
            self.prey_belief_vector = np.matmul(self.prey_transition_matrix, self.prey_belief_vector)
        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # check if there's a node where prey certainly is
        if largest_belief == 1:
            # survey no node since it is certain where prey is
            return random.choice(largest_belief_node_names)

        # pick node with the largest belief to survey after knowing there's no node where prey certainly is
        if is_certain_predator_location:
            node_name_to_survey = random.choice(largest_belief_node_names)
        else:
            node_name_to_survey = surveyed_node_name
        self.surveyed_node_name = node_name_to_survey
        node_to_survey = graph.graph[node_name_to_survey]
        is_prey_on_node = node_to_survey.is_prey_existed()
        if is_prey_on_node and is_failed_drone:
            is_prey_on_node = False
        if is_prey_on_node:
            # update belief vector after knowing the surveyed node has prey
            self.prey_belief_vector[node_name_to_survey] = 1.0
            for i in range(1, graph.size + 1):
                if i != node_name_to_survey:
                    self.prey_belief_vector[i] = 0.0
        else:
            # update belief vector after knowing the surveyed node has no prey
            # under the condition that drone may fail to tell the truth
            drone_failing_probability = self.prey_belief_vector[self.surveyed_node_name] * 0.1 + \
                                        (1 - self.prey_belief_vector[self.surveyed_node_name])
            for i, belief in enumerate(self.prey_belief_vector[1:]):
                self.prey_belief_vector[i + 1] = belief * 1.0 if (i + 1) != self.surveyed_node_name \
                    else belief * 0.1
            self.prey_belief_vector /= drone_failing_probability if drone_failing_probability != 0 else 1.0

        largest_belief = self.prey_belief_vector[np.argmax(self.prey_belief_vector)]
        largest_belief_node_names = [i for i, belief in enumerate(self.prey_belief_vector) if belief == largest_belief]
        # pick node with the largest belief for treating it as where prey is for this round
        return random.choice(largest_belief_node_names)


class Agent9(RevisedExtendedAgent8):
    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is not None:
            self.name = 'agent9'

    def write(self, f):
        pass

    def read(self, f):
        pass

    def get_next_node_name(self, graph, prey_node_name, predator_node_name):
        cur_node = graph.graph[self.cur_node_name]

        # record neighbors' info about their distances to prey, their distances to predator,
        # a random number which is for breaking ties, and their names (as tuples in a list)
        source_names = set()
        for node in cur_node.neighbors:
            source_names.add(node.name)
        prey_and_its_neighbor_node_names = []
        prey_level_dict = self.get_level(graph, prey_node_name)
        level = 0
        for k, v in prey_level_dict.items():
            if k <= level:
                prey_and_its_neighbor_node_names.extend(v)
        predator_and_its_neighbor_node_names = []
        predator_level_dict = self.get_level(graph, predator_node_name)
        level = 3
        for k, v in predator_level_dict.items():
            if k <= level:
                predator_and_its_neighbor_node_names.extend(v)
        neighbors_to_prey_and_predator_info = []
        # record min and max of all shortest distances to predator (for all neighbors of the agent)
        # record min and max of all shortest distances to prey (for all neighbors of the agent)
        min_prey_distance = min_predator_distance = float('Inf')
        max_prey_distance = max_predator_distance = -1
        random_num_list = list(range(1, len(cur_node.neighbors) + 1))
        random.shuffle(random_num_list)
        for i, node_name in enumerate(source_names):
            self.get_distance_from_source(graph, node_name)
            weighted_distance_for_prey = 0
            weighted_distance_for_predator = 0
            for prey_or_neighbor_node_name in prey_and_its_neighbor_node_names:
                weighted_distance_for_prey += self.distance[prey_or_neighbor_node_name] * \
                                              self.prey_belief_vector[prey_or_neighbor_node_name]
                if weighted_distance_for_prey < min_prey_distance:
                    min_prey_distance = weighted_distance_for_prey
                if weighted_distance_for_prey > max_prey_distance:
                    max_prey_distance = weighted_distance_for_prey
            for predator_or_neighbor_node_name in predator_and_its_neighbor_node_names:
                weighted_distance_for_predator += self.distance[predator_or_neighbor_node_name] * \
                                                  self.predator_belief_vector[predator_or_neighbor_node_name]
                if weighted_distance_for_predator < min_predator_distance:
                    min_predator_distance = weighted_distance_for_predator
                if weighted_distance_for_predator > max_predator_distance:
                    max_predator_distance = weighted_distance_for_predator
            neighbors_to_prey_and_predator_info.append(
                (weighted_distance_for_prey, weighted_distance_for_predator,
                 random_num_list[i], node_name))
        # get used for classifying up to three classes of neighbors:
        # those closer to the prey,
        # those not closer and not farther from the prey,
        # and those farther from the prey
        min_end_index = not_min_max_end_index = len(neighbors_to_prey_and_predator_info)
        neighbors_to_prey_and_predator_info.sort()
        for i, info in enumerate(neighbors_to_prey_and_predator_info):
            if min_end_index is None and info[0] != min_prey_distance:
                min_end_index = i
            if not_min_max_end_index is None and info[0] == max_prey_distance:
                not_min_max_end_index = i

        # examine neighbors based on rules
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

    @staticmethod
    def get_level(graph, source_name):
        # compute distance from source node to every node in graph
        levels = [INVALID for _ in range(graph.size + 1)]
        source_node = graph.graph[source_name]
        queue = deque([source_node])
        levels[source_node.name] = 0
        while len(queue):
            cur_node = queue.popleft()
            cur_dist = levels[cur_node.name]
            for neighbor_node in cur_node.neighbors:
                if levels[neighbor_node.name] == INVALID:
                    levels[neighbor_node.name] = cur_dist + 1
                    queue.append(neighbor_node)
        level_dict = defaultdict(list)
        for node_name, level in enumerate(levels):
            if node_name != 0:
                level_dict[level].append(node_name)
        return level_dict
