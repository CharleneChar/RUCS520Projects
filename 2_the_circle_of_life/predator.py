from copy import *


class Predator:
    def __init__(self, init_node_name=None):
        self.name = 1
        self.pre_node_name = None
        self.cur_node_name = init_node_name

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.name = self.name
        d_copy.pre_node_name = deepcopy(self.pre_node_name)
        d_copy.cur_node_name = deepcopy(self.cur_node_name)
        return d_copy

    def write(self, f):
        pass

    def read(self, f):
        pass

    def update_node_name(self, next_node_name):
        self.pre_node_name = self.cur_node_name
        self.cur_node_name = next_node_name
