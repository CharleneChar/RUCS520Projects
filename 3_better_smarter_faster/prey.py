from copy import *


class Prey:
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
        f.write(f'{self.name}\n')
        if self.pre_node_name is not None:
            f.write(f'{self.pre_node_name}')
        f.write(f'\n')
        f.write(f'{self.cur_node_name}\n')

    def read(self, f):
        self.name, = map(int, f.readline().rstrip().split(','))
        self.pre_node_name, = f.readline().rstrip().split(',')[0]
        if len(self.pre_node_name) == 0:
            self.pre_node_name = None
        else:
            self.pre_node_name = int(self.pre_node_name)
        self.cur_node_name, = map(int, f.readline().rstrip().split(','))

    def update_node_name(self, next_node_name):
        self.pre_node_name = self.cur_node_name
        self.cur_node_name = next_node_name
