from copy import *


class Ghost:
    def __init__(self, name=None, init_point=None):
        self.name = name
        self.pre_point = None
        self.cur_point = init_point

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.name = self.name
        d_copy.pre_point = deepcopy(self.name)
        d_copy.cur_point = deepcopy(self.cur_point)
        return d_copy

    def write(self, f):
        f.wrtie(f'{self.name}\n')
        self.cur_point.wrtie(f)

    def read(self, f):
        self.name = f.readline().rstrip()
        self.cur_point.read(f)

    def update_point(self, next_point):
        self.pre_point = self.cur_point
        self.cur_point = next_point
