from sortedcontainers import SortedDict
from copy import *


class Node:
    AGENT = 'A'
    PREY = 'Y'
    PREDATOR = 'D'
    UNKNOWN = '?'

    def __init__(self, name=None):
        self.prey = None
        self.predator = None
        self.agent = None
        self.name = name
        self.neighbors = []
        self.belief_prey_loc = 0
        self.belief_predator_loc = 0

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.prey = deepcopy(self.prey)
        d_copy.predator = deepcopy(self.predator)
        d_copy.agent = deepcopy(self.agent)
        d_copy.name = deepcopy(self.name)
        d_copy.neighbors = deepcopy(self.neighbors)
        d_copy.belief_prey_loc = deepcopy(self.belief_prey_loc)
        d_copy.belief_predator_loc = deepcopy(self.belief_predator_loc)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def set_belief_prey_loc(self, probability):
        self.belief_prey_loc = probability

    def set_belief_predator_loc(self, probability):
        self.belief_predator_loc = probability

    def is_prey_existed(self):
        return self.prey is not None

    def is_predator_existed(self):
        return self.predator is not None

    def is_agent_existed(self):
        return self.agent is not None

    def insert_prey(self, prey):
        if not self.prey:
            self.prey = prey

    def insert_predator(self, predator):
        if not self.predator:
            self.predator = predator

    def insert_agent(self, agent):
        if not self.agent:
            self.agent = agent

    def remove_prey(self):
        self.prey = None

    def remove_predator(self):
        self.predator = None

    def remove_agent(self):
        self.agent = None

    def is_capturing_prey(self):
        return self.is_agent_existed() and self.is_prey_existed()

    def is_captured_by_predator(self):
        return self.is_agent_existed() and self.is_predator_existed()

    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)



