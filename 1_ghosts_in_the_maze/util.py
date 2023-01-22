from copy import *

INVALID = -1
DEBUG = -1


class Point:
    def __init__(self, r=-1, c=-1):
        self.r = r
        self.c = c

    def __repr__(self):
        return f'({self.r}, {self.c})'

    def __eq__(self, another_point):
        if another_point is not None:
            return self.r == another_point.r and self.c == another_point.c
        return False

    def __hash__(self):
        return self.r + self.c

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.r = self.r
        d_copy.c = self.c
        return d_copy

    def write(self, f):
        f.wrtie(f'{self.r},{self.c}\n')

    def read(self, f):
        self.r, self.c = map(int, f.readline().rstrip().split(','))


def and_result(bool_expr):
    for expr in bool_expr:
        if not expr:
            return False
    return True


def or_result(bool_expr):
    for expr in bool_expr:
        if expr:
            return True
    return False


# is for debugging
def print_distance(distance, sep=' '*4):
    s = ''
    n_r = len(distance)
    if n_r:
        n_c = len(distance[0])
        if n_c:
            col_width = [0 for _ in range(n_c)]
            for c in range(n_c):
                if col_width[c] < len(str(c)): col_width[c] = len(str(c))
                for r in range(n_r):
                    width = len(str(distance[r][c]))
                    if col_width[c] < width: col_width[c] = width
            r_width = len(str(n_r))
            s += ''.rjust(r_width)
            for c in range(n_c):
                s += f'{sep}' + str(c).rjust(col_width[c])
            s += '\n'
            for r, row in enumerate(distance):
                s += str(r).rjust(r_width)
                for c, entry in enumerate(row):
                    s += f'{sep}' + str(entry).rjust(col_width[c])
                s += '\n'
    print(s)


# is for debugging
def print_path(path, n_c=5, sep=' '*1):
    path = path[::-1]
    s = ''
    if len(path):
        last_r = len(path) // n_c
        r_width = len(f'{last_r * n_c} ~ {min((last_r + 1) * n_c - 1, len(path) - 1)}')
        col_width = [0 for _ in range(n_c)]
        for c in range(n_c):
            for r in range(last_r + 1):
                if r * n_c + c < len(path):
                    width = len(str(path[r * n_c + c]))
                    if col_width[c] < width: col_width[c] = width
        for r in range(last_r + 1):
            s += f'{r * n_c} ~ {min((r + 1) * n_c - 1, len(path) - 1)}'.ljust(r_width)
            for c in range(n_c):
                i = r * n_c + c
                if i < len(path):
                    s += f'{sep}->{sep}' + str(path[i]).ljust(col_width[c])
            s += '\n'
    print(s)


# is for debugging
def print_path_history(path_history, n_c=5, sep=' '*1):
    path = path_history[::-1]
    s = ''
    if len(path):
        last_r = len(path) // n_c
        r_width = len(f'{last_r * n_c} ~ {min((last_r + 1) * n_c - 1, len(path) - 1)}')
        col_width = [0 for _ in range(n_c)]
        for c in range(n_c):
            for r in range(last_r + 1):
                if r * n_c + c < len(path):
                    width = len(str(path[r * n_c + c]))
                    if col_width[c] < width: col_width[c] = width
        for r in range(last_r + 1):
            s += f'-{r * n_c} ~ -{min((r + 1) * n_c - 1, len(path) - 1)}'.ljust(r_width)
            for c in range(n_c):
                i = r * n_c + c
                if i < len(path):
                    s += f'{sep}->{sep}' + str(path[i]).ljust(col_width[c])
            s += '\n'
    print(s)

