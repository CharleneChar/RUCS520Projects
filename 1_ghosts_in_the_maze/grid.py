from random import randint as ri
from collections import deque
from sortedcontainers import SortedDict
from copy import *

from cell import *
from ghost import *
from agent import *
from util import *


class Grid:
    ONGOING = 0
    SUCCESS = 1
    COLLISION = 2
    TIMEOUT = 3

    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.size = kwargs['grid_size']
            self.grid = None
            self.source = Point(0, 0)
            self.target = Point(self.size.r - 1, self.size.c - 1)
            self.distance = None
            self.agent = None
            self.ghosts = SortedDict()
            for i in range(kwargs['num_ghosts']):
                self.ghosts[f'gh{i}'] = None
            self.reachable_points = None
            self.status = self.ONGOING
            self.path = None
            self.__init(kwargs['agent_name'])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        sep = ' ' * 4
        col_width = [0 for _ in range(self.size.c)]
        for c in range(self.size.c):
            if col_width[c] < len(str(c)): col_width[c] = len(str(c))
            for r in range(self.size.r):
                width = len(str(self.grid[r][c]))
                if col_width[c] < width: col_width[c] = width
        s = ''
        if self.size.r > 0 and self.size.c > 0:
            r_width = len(str(self.size.r))
            s += ''.ljust(r_width)
            for c in range(self.size.c):
                s += f'{sep}' + str(c).ljust(col_width[c])
            s += '\n'
            for r, row in enumerate(self.grid):
                s += str(r).ljust(r_width)
                for c, cell in enumerate(row):
                    s += f'{sep}' + str(cell).ljust(col_width[c])
                s += '\n'
        return s

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.size = deepcopy(self.size)
        d_copy.grid = deepcopy(self.grid)
        d_copy.source = deepcopy(self.source)
        d_copy.target = deepcopy(self.target)
        d_copy.distance = deepcopy(self.distance)
        d_copy.agent = deepcopy(self.agent)
        d_copy.ghosts = SortedDict()
        for k, v in self.ghosts.items():
            d_copy.ghosts[k] = deepcopy(v)
        d_copy.reachable_points = deepcopy(self.reachable_points)
        d_copy.status = self.status
        d_copy.path = deepcopy(self.path)
        return d_copy

    def move_ghosts(self):
        for ghost_name, ghost in self.ghosts.items():
            dr = [1, -1, 0, 0]
            dc = [0, 0, 1, -1]
            candidate_points = []
            for i in range(len(dr)):
                next_r, next_c = ghost.cur_point.r + dr[i], ghost.cur_point.c + dc[i]
                if 0 <= next_r < self.size.r and 0 <= next_c < self.size.c:
                    candidate_points.append(Point(next_r, next_c))
            picked_point = candidate_points[ri(0, len(candidate_points) - 1)]
            if self.grid[picked_point.r][picked_point.c].is_blocked():
                # select to move to the blocked cell or to stay on current cell
                # with same probability of 0.5
                # when picked neighbor is blocked
                if ri(0, 1):
                    ghost.update_point(picked_point)
                    self.__update_cell(ghost=ghost)
            else:
                ghost.update_point(picked_point)
                self.__update_cell(ghost=ghost)

    def move_agent(self):
        if self.agent.name == 1:
            # decide which is the next cell to move to for agent 1
            next_point = self.agent.next_point()
        elif self.agent.name == 2:
            # decide which is the next cell to move to for agent 2
            next_point = self.agent.next_point(self)
        elif self.agent.name == 3:
            # decide which is the next cell to move to for agent 3
            next_point = self.agent.next_point(self)
        elif self.agent.name == 4:
            # decide which is the next cell to move to for agent 4
            next_point = self.agent.next_point(self)
        else:
            # to be completed in the future
            pass
        self.agent.update_point(next_point)
        self.__update_cell(agent=self.agent)

    def is_in_grid(self, next_point):
        return 0 <= next_point.r < self.size.r and \
               0 <= next_point.c < self.size.c

    def is_valid_cell(self, next_point):
        if self.is_in_grid(next_point):
            next_cell = self.grid[next_point.r][next_point.c]
            return (not next_cell.is_blocked()) and \
                   self.distance[next_point.r][next_point.c] == INVALID
        return False

    def get_status(self):
        if self.status == self.ONGOING:
            return 'ONGOING'
        if self.status == self.SUCCESS:
            return 'SUCCESS'
        if self.status == self.COLLISION:
            return 'COLLISION'
        if self.status == self.TIMEOUT:
            return 'TIMEOUT'
        return 'UNKNOWN'

    def update_status(self, step):
        # specify upper time limit for each strategy to solve maze game
        if step > 3 * (self.size.r + self.size.c):
            # confirm that agent is alive in the maze but not on target cell over time limit
            self.status = self.TIMEOUT
        elif self.agent.cur_point == self.target:
            # confirm that agent successfully reaches the target cell within time limit
            self.status = self.SUCCESS
        elif self.grid[self.agent.cur_point.r][self.agent.cur_point.c].is_collision():
            # confirm that agent is killed by ghosts within time limit
            self.status = self.COLLISION

    # to be updated in the future
    def write(self):
        filename = f'{self.name}.grid'
        with open(filename, 'w') as f:
            for row in self.grid:
                for cell in row:
                    cell.write(f)

    # to be updated in the future
    def read(self):
        self.grid = []
        filename = f'{self.name}.grid'
        with open(filename, 'r') as f:
            for _ in range(self.size.r):
                self.grid.append([])
                for _ in range(self.size.c):
                    cell = Cell()
                    cell.read(f)
                    self.grid[-1].append(cell)

    def __init(self, agent_name):
        if not self.grid:
            self.__generate()
            self.__get_reachable_points()
            self.__init_ghosts()
            self.__init_agent(agent_name)
            self.update_status(0)

    def __init_agent(self, agent_name):
        if agent_name == 1:
            # initialize agent 1
            self.agent = Agent1(self.source, grid=self)
            self.__update_cell(agent=self.agent)
        elif agent_name == 2:
            # initialize agent 2
            kwargs = {
                'grid': self,
                'pre_point': None,
                'init_point': self.source,
                'ghost_info': None,
                'distance': None,
                'queue_history': None,
                'path_history_set': None,
                'path_history_list': None,
                'path': None
            }
            self.agent = Agent2(kwargs)
            self.__update_cell(agent=self.agent)
        elif agent_name == 3:
            # initialize agent 3
            kwargs = {
                'grid': self,
                'pre_point': None,
                'init_point': self.source,
                'ghost_info': None,
                'distance': None,
                'queue_history': None,
                'path_history_set': None,
                'path_history_list': None,
                'path': None
            }
            self.agent = Agent3(kwargs)
            self.__update_cell(agent=self.agent)
        elif agent_name == 4:
            # initialize agent 4
            kwargs = {
                'grid': self,
                'pre_point': None,
                'init_point': self.source,
                'distance': None,
                'path': None
            }
            self.agent = Agent4(kwargs)
            self.__update_cell(agent=self.agent)
        else:
            # to be completed in the future
            pass

    def __generate(self):
        self.grid = self.__random_grid()
        # validate maze generated by above function
        while not self.__is_reachable():
            self.grid = self.__random_grid()
        if DEBUG >= 2:
            print(str(self))
            for p in self.path:
                print(p)

    def __init_ghosts(self):
        for name in self.ghosts:
            # exclude source cell from the choice of possible initial cells for ghosts to stand on
            reachable_points_without_source = deepcopy(self.reachable_points)[1:]
            end_index = len(reachable_points_without_source) - 1
            self.ghosts[name] = Ghost(name=name, init_point=reachable_points_without_source[ri(0, end_index)])
            self.__update_cell(ghost=self.ghosts[name])

    def __random_grid(self):
        # create a maze with cells
        # with probability 0.28 being blocked and with probability 0.72 being unblocked
        grid = [[Cell(blocked=True) if ri(0, 99) < 28 else Cell()
                 for _ in range(self.size.c)] for _ in range(self.size.r)]
        grid[self.source.r][self.source.c].set_blocked(False)
        grid[self.target.r][self.target.c].set_blocked(False)
        return grid

    def __is_reachable(self):
        # use breadth first search from source cell
        # to create a list of list with optimal distances from source cell to each cell on maze
        # (where when the search visits blocked cells,
        # the value stored in corresponding blocked cell position in the list of list
        # will remain invalid, namely, -1)
        # ,and validate whether maze generated is valid or not
        self.distance = [[INVALID for _ in range(self.size.c)] for _ in range(self.size.r)]
        dr = [1, -1, 0, 0]
        dc = [0, 0, 1, -1]
        self.distance[self.source.r][self.source.c] = 0
        queue = deque([(self.source, 0)])
        while len(queue):
            cur_point, cur_distance = queue.popleft()
            for i in range(len(dr)):
                next_point = Point(cur_point.r + dr[i], cur_point.c + dc[i])
                if self.is_valid_cell(next_point):
                    self.distance[next_point.r][next_point.c] = cur_distance + 1
                    queue.append((next_point, cur_distance + 1))
        if DEBUG >= 2:
            self.__backtrack(self.target)
        # confirm that it's a valid maze if the value stored
        # in the corresponding target cell position in the list of list
        # is not -1 (representing invalid and meaning there's no path to target from source cell)
        return self.distance[self.target.r][self.target.c] != INVALID

    # is for debugging
    def __backtrack(self, cur_point):
        self.path = [['.' for _ in range(self.size.c)] for _ in range(self.size.r)]
        if self.distance[cur_point.r][cur_point.c] >= 0:
            dr = [1, -1, 0, 0]
            dc = [0, 0, 1, -1]
            self.path[cur_point.r][cur_point.c] = 'T'
            cur_dist = self.distance[cur_point.r][cur_point.c]
            while cur_dist > 0:
                for i in range(len(dr)):
                    next_point = Point(cur_point.r + dr[i], cur_point.c + dc[i])
                    if self.is_in_grid(next_point) and INVALID < self.distance[next_point.r][next_point.c] < cur_dist:
                        if self.distance[next_point.r][next_point.c] != 0:
                            self.path[next_point.r][next_point.c] = 'O'
                        cur_dist = self.distance[next_point.r][next_point.c]
                        cur_point = next_point
                        break
            self.path[cur_point.r][cur_point.c] = 'S'

    def __get_reachable_points(self):
        self.reachable_points = []
        for r, row in enumerate(self.distance):
            for c, d in enumerate(row):
                if d > INVALID:
                    self.reachable_points.append(Point(r, c))

    def __update_cell(self, ghost=None, agent=None):
        if ghost:
            # update record of ghosts' current standing cells
            if ghost.pre_point:
                pre_cell = self.grid[ghost.pre_point.r][ghost.pre_point.c]
                pre_cell.remove_ghost(ghost)
            cur_cell = self.grid[ghost.cur_point.r][ghost.cur_point.c]
            cur_cell.insert_ghost(ghost)
        if agent:
            # update record of agent's current standing cell
            if agent.pre_point:
                pre_cell = self.grid[agent.pre_point.r][agent.pre_point.c]
                pre_cell.remove_agent()
            cur_cell = self.grid[agent.cur_point.r][agent.cur_point.c]
            cur_cell.insert_agent(agent)


