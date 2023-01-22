from copy import *


class Cell:
    UNBLOCKED = 'O'
    BLOCKED = 'X'

    def __init__(self, r=None, c=None, blocked=False):
        self.blocked = blocked
        self.r = r
        self.c = c

    def __repr__(self):
        return f'{self.r},{self.c}'

    def __eq__(self, another_cell):
        return another_cell is not None \
               and self.r == another_cell.r \
               and self.c == another_cell.c

    def __hash__(self):
        return hash((self.r, self.c))

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.blocked = self.blocked
        d_copy.r = self.r
        d_copy.c = self.c
        return d_copy

    def write(self, f):
        f.write(f'{int(self.blocked)}\n')
        f.write(f'{self.r}\n')
        f.write(f'{self.c}\n')

    def read(self, f):
        self.blocked = bool(f.readline().rstrip())
        self.r = int(f.readline().rstrip())
        self.c = int(f.readline().rstrip())

    def set_blocked(self, blocked):
        self.blocked = blocked

    def is_blocked(self):
        return self.blocked
