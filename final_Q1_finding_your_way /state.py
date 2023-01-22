from copy import *
import numpy as np
from collections import deque

from util import *


class State:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.drone_belief_vector = kwargs['drone_belief_vector']
            self.num_non_zero_points = None
            self.hash_value = None
            self.cmd_seq = None
            self.parent = kwargs['parent']
            self.__init(kwargs)

    def __repr__(self):
        return ','.join(map(str, [1 if p > 0 else 0
                                  for p in self.drone_belief_vector]))

    def __eq__(self, other):
        return other is not None and self.hash_value == other.hash_value

    def __hash__(self):
        return self.hash_value

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.drone_belief_vector = deepcopy(self.drone_belief_vector)
        d_copy.num_non_zero_points = deepcopy(self.num_non_zero_points)
        d_copy.hash_value = deepcopy(self.hash_value)
        d_copy.cmd_seq = deepcopy(self.cmd_seq)
        d_copy.parent = deepcopy(self.parent)
        return d_copy

    def print(self, col_size, is_cmd_only=True):
        print(len(self.cmd_seq), ','.join(map(str, self.cmd_seq)), sep=': ')
        print(self.num_non_zero_points)
        if not is_cmd_only:
            for i, p in enumerate(self.drone_belief_vector):
                if p > 0:
                    print('O', end='')
                else:
                    print('X', end='')
                if (i + 1) % col_size == 0:
                    print()

    def is_final_state(self):
        return self.num_non_zero_points == 1

    def is_final_state_2(self, num_unblocked_cells):
        return self.num_non_zero_points == num_unblocked_cells

    def reverse_cmd_seq(self):
        rev_cmd_seq = []
        for cmd in self.cmd_seq[::-1]:
            rev_cmd_seq.append(REV_ACT[cmd])
        return rev_cmd_seq

    def __init(self, kwargs):
        self.hash_value = self.num_non_zero_points = 0
        for i, p in enumerate(self.drone_belief_vector):
            self.hash_value <<= 1
            if p > 0:
                self.hash_value |= 1
                self.num_non_zero_points += 1
        if self.parent is not None:
            self.cmd_seq = deepcopy(self.parent.cmd_seq)
        else:
            self.cmd_seq = []
        if kwargs['act'] is not None:
            self.cmd_seq.append(kwargs['act'])
