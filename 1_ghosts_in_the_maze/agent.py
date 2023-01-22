from collections import deque, defaultdict
from copy import *

from grid import *
from util import *


class Agent1:
    def __init__(self, init_point=None, grid=None):
        if init_point is not None and grid is not None:
            self.name = 1
            self.pre_point = None
            self.cur_point = init_point
            self.path = None
            self.__init(grid)

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.name = self.name
        d_copy.pre_point = deepcopy(self.pre_point)
        d_copy.cur_point = deepcopy(self.cur_point)
        d_copy.path = deepcopy(self.path)
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

    def next_point(self):
        if len(self.path):
            return self.path.pop()
        return self.cur_point

    def __init(self, grid):
        cur_point = grid.target
        self.path = []
        if grid.distance[cur_point.r][cur_point.c] >= 0:
            dr = [1, -1, 0, 0]
            dc = [0, 0, 1, -1]
            self.path.append(cur_point)
            # use depth first search in the previously calculated list of list with optimal distances
            # from target cell position back to source cell position
            # and will prune the further search when reach source cell position the first time
            cur_dist = grid.distance[cur_point.r][cur_point.c]
            while cur_dist > 0:
                for i in range(len(dr)):
                    next_point = Point(cur_point.r + dr[i], cur_point.c + dc[i])
                    if grid.is_in_grid(next_point) and INVALID < grid.distance[next_point.r][next_point.c] < cur_dist:
                        if grid.distance[next_point.r][next_point.c] != 0:
                            self.path.append(next_point)
                        cur_dist = grid.distance[next_point.r][next_point.c]
                        cur_point = next_point
                        break


class Agent2:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.name = 2
            grid = kwargs['grid']
            self.pre_point = kwargs['pre_point']
            self.cur_point = kwargs['init_point']
            self.ghost_info = kwargs['ghost_info']
            if self.ghost_info is None:
                self.ghost_info = []
            self.distance = kwargs['distance']
            if self.distance is None:
                self.distance = [[INVALID for _ in range(grid.size.c)] for _ in range(grid.size.r)]
                self.distance[kwargs['init_point'].r][kwargs['init_point'].c] = 0
            self.queue_history = kwargs['queue_history']
            if self.queue_history is None:
                self.queue_history = deque()
            self.path_history_set = kwargs['path_history_set']
            if self.path_history_set is None:
                self.path_history_set = set()
                self.path_history_set.add(kwargs['init_point'])
            self.path_history_list = kwargs['path_history_list']
            if self.path_history_list is None:
                self.path_history_list = [kwargs['init_point']]
            self.path = kwargs['path']
            if self.path is None:
                self.path = []

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.name = self.name
        d_copy.pre_point = deepcopy(self.pre_point)
        d_copy.cur_point = deepcopy(self.cur_point)
        d_copy.ghost_info = deepcopy(self.ghost_info)
        d_copy.distance = deepcopy(self.distance)
        d_copy.queue_history = deepcopy(self.queue_history)
        d_copy.path_history_set = deepcopy(self.path_history_set)
        d_copy.path = deepcopy(self.path)
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

    def next_point(self, grid):
        # non re-plan situation: keep moving on the shortest path without ghosts
        if len(self.path) != 0 and not self.__is_ghost_in_path(grid):
            self.path_history_set.add(self.path[-1])
            self.path_history_list.append(self.path[-1])
            return self.path.pop()

        # clear distance greater than distance from source to current point
        # (when knowing there is any ghost on the previously calculated shortest path)
        while len(self.queue_history):
            point, _, _ = self.queue_history.pop()
            if point not in self.path_history_set:
                self.distance[point.r][point.c] = INVALID

        # use breadth first search again to categorize 3 different re-plan scenarios
        # when knowing there's ant ghost on the previously calculated shortest path
        dr = [1, 0, -1, 0]
        dc = [0, 1, 0, -1]
        shortest_ghost_distance = [INVALID for _ in range(len(dr))]
        # is used to maintain as well as record the initial direction of move on the path
        init_point = Point(self.cur_point.r, self.cur_point.c)
        init_distance = self.distance[self.cur_point.r][self.cur_point.c]
        queue = deque([(init_point, 0, init_distance)])
        is_target_reachable = False
        while len(queue) and not is_target_reachable:
            self.queue_history.append(queue[0])
            cur_point, init_step_i, cur_distance = queue.popleft()
            for i in range(len(dr)):
                next_point = Point(cur_point.r + dr[i], cur_point.c + dc[i])
                if cur_distance == init_distance:
                    init_step_i = i
                if grid.is_in_grid(next_point):
                    next_cell = grid.grid[next_point.r][next_point.c]
                    if not next_cell.is_blocked() and \
                            self.distance[next_point.r][next_point.c] == INVALID:
                        if next_cell.is_ghost_existed():
                            if shortest_ghost_distance[init_step_i] == INVALID:
                                shortest_ghost_distance[init_step_i] = cur_distance + 1 - init_distance
                            self.queue_history.append((next_point, init_step_i, cur_distance + 1))
                        else:
                            queue.append((next_point, init_step_i, cur_distance + 1))
                        self.distance[next_point.r][next_point.c] = cur_distance + 1
                if next_point == grid.target and not \
                        grid.grid[grid.target.r][grid.target.c].is_ghost_existed():
                    is_target_reachable = True
                    break
        while len(queue):
            self.queue_history.append(queue.pop())

        # move on the currently recalculated shortest path
        # from current cell position to target cell position if found one
        # (after knowing there's at least one ghost on the previously calculated shortest path)
        if is_target_reachable:
            # use depth first search to recalculate a shortest path
            # from target cell position to current cell position
            self.find_path_to_init_point(grid)
            self.path_history_set.add(self.path[-1])
            self.path_history_list.append(self.path[-1])
            return self.path.pop()

        # clear path after knowing there's at least one ghost on it for next round usage
        self.path = []

        # try to move away from the nearest ghosts when no path is found after recalculating
        # (including staying at current point when distances from all ghosts are the same)
        h_i = None
        if len(self.path_history_list) > 1:
            # calculate distance between agent and ghost that's on the previous path which agent 2 come from
            h_i = self.update_ghost_on_history_distance(grid, shortest_ghost_distance, dr, dc)
        # # calculate distance between agent and ghost that's not on the previous path which agent 2 come from
        non_nearest_ghost_directions = self.check_ghost_distance(shortest_ghost_distance)
        if len(non_nearest_ghost_directions):
            # move in the direction toward non nearest ghost
            next_direction = non_nearest_ghost_directions[0]
            next_point = Point(self.cur_point.r + dr[next_direction], self.cur_point.c + dc[next_direction])
            if h_i is not None and next_point == self.path_history_list[-2]:
                # allow forgetting previous cell and so move back to previous cell from which agent 2 come
                self.path_history_set.remove(self.cur_point)
                self.path_history_list.pop()
            else:
                self.path_history_set.add(next_point)
                self.path_history_list.append(next_point)
            return next_point
        else:
            if h_i is not None and shortest_ghost_distance[h_i] == INVALID:
                # allow forgetting previous cell and so move back to previous cell from which agent 2 come
                next_point = Point(self.cur_point.r + dr[h_i], self.cur_point.c + dc[h_i])
                self.path_history_set.remove(self.cur_point)
                self.path_history_list.pop()
                return next_point
        # stay on current standing cell when no previous cell from which agent came
        # or when all distances to ghosts in every possible direction are the same
        return self.cur_point

    def find_path_to_init_point(self, grid):
        # use depth first search to recalculate a shortest path
        # from target cell position to current cell position
        self.path = []
        for ghost in grid.ghosts.values():
            if ghost.cur_point not in self.path_history_set:
                self.distance[ghost.cur_point.r][ghost.cur_point.c] = INVALID
        if self.distance[grid.target.r][grid.target.c] != INVALID:
            self.path.append(grid.target)
            dr = [-1, 0, 1, 0]
            dc = [0, -1, 0, 1]
            init_distance = self.distance[self.cur_point.r][self.cur_point.c]
            cur_point = grid.target
            cur_distance = self.distance[cur_point.r][cur_point.c]
            while cur_distance > init_distance:
                for i in range(len(dr)):
                    next_point = Point(cur_point.r + dr[i], cur_point.c + dc[i])
                    if grid.is_in_grid(next_point):
                        next_distance = self.distance[next_point.r][next_point.c]
                        if INVALID != next_distance == (cur_distance - 1):
                            if next_distance > init_distance:
                                self.path.append(next_point)
                            cur_distance = next_distance
                            cur_point = next_point
                            break

    def update_ghost_on_history_distance(self, grid, shortest_ghost_distance, dr, dc):
        r = self.path_history_list[-2].r - self.cur_point.r
        c = self.path_history_list[-2].c - self.cur_point.c
        h_i = None
        for i in range(len(dr)):
            if dr[i] == r and dc[i] == c:
                h_i = i
                break
        for i, point in enumerate(self.path_history_list[::-1]):
            if grid.grid[point.r][point.c].is_ghost_existed():
                shortest_ghost_distance[h_i] = i
                break
        return h_i

    @staticmethod
    def check_ghost_distance(shortest_ghost_distance):
        non_nearest_ghost_directions = []
        directions = []
        shortest_distance = float('Inf')
        for i, distance in enumerate(shortest_ghost_distance):
            if distance != INVALID:
                directions.append(i)
                if distance < shortest_distance:
                    shortest_distance = distance
        if len(directions) > 1:
            for direction in directions:
                if shortest_distance < shortest_ghost_distance[direction]:
                    non_nearest_ghost_directions.append(direction)
        return non_nearest_ghost_directions

    def __is_ghost_in_path(self, grid):
        for point in self.path:
            if grid.grid[point.r][point.c].is_ghost_existed():
                return True
        return False


class Agent3:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.name = 3
            grid = kwargs['grid']
            self.pre_point = kwargs['pre_point']
            self.cur_point = kwargs['init_point']
            self.ghost_info = kwargs['ghost_info']
            if self.ghost_info is None:
                self.ghost_info = []
            self.distance = kwargs['distance']
            if self.distance is None:
                self.distance = [[INVALID for _ in range(grid.size.c)] for _ in range(grid.size.r)]
                self.distance[kwargs['init_point'].r][kwargs['init_point'].c] = 0
            self.queue_history = kwargs['queue_history']
            if self.queue_history is None:
                self.queue_history = deque()
            self.path_history_set = kwargs['path_history_set']
            if self.path_history_set is None:
                self.path_history_set = set()
                self.path_history_set.add(kwargs['init_point'])
            self.path_history_list = kwargs['path_history_list']
            if self.path_history_list is None:
                self.path_history_list = [kwargs['init_point']]
            self.path = kwargs['path']
            if self.path is None:
                self.path = []

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.name = self.name
        d_copy.pre_point = deepcopy(self.pre_point)
        d_copy.cur_point = deepcopy(self.cur_point)
        d_copy.ghost_info = deepcopy(self.ghost_info)
        d_copy.distance = deepcopy(self.distance)
        d_copy.queue_history = deepcopy(self.queue_history)
        d_copy.path_history_set = deepcopy(self.path_history_set)
        d_copy.path = deepcopy(self.path)
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

    def next_point(self, grid):
        dr = [1, 0, -1, 0]
        dc = [0, 1, 0, -1]
        survival_rate_list = [INVALID for _ in range(len(dr))]
        hung_rate_list = [INVALID for _ in range(len(dr))]
        # simulate several times for each next possible direction via strategy applied by agent 2
        # and choose to move in the direction with the highest success count
        for i in range(len(dr)):
            next_possible_point = Point(self.cur_point.r + dr[i], self.cur_point.c + dc[i])
            if grid.is_in_grid(next_possible_point):
                next_possible_cell = grid.grid[next_possible_point.r][next_possible_point.c]
                if not next_possible_cell.is_blocked() \
                        and not next_possible_cell.is_ghost_existed() \
                        and next_possible_point not in self.path_history_set:
                    survival_rate, hung_rate = \
                        self.survivability(grid, next_possible_point, num_simulation=10000)
                    # record success count for total number of simulations
                    survival_rate_list[i] = survival_rate
                    # record hung count for total number of simulations
                    hung_rate_list[i] = hung_rate
        largest_survival_directions, largest_survival_rate = self.find_largest_survivability(survival_rate_list)
        if len(largest_survival_directions) == 1:
            # move in the direction with the largest success count
            index = largest_survival_directions[0]
            next_point = Point(self.cur_point.r + dr[index], self.cur_point.c + dc[index])
            self.path_history_set.add(next_point)
            self.path_history_list.append(next_point)
            return next_point
        else:
            # mimic agent 2 to choose which direction to move in order to run away from nearest ghosts
            if largest_survival_rate != INVALID:
                for d, point in enumerate(self.path_history_list):
                    self.distance[point.r][point.c] = d
                next_point = self.update_distance(grid, dr, dc)
                self.path = []
                self.distance = [[INVALID for _ in range(grid.size.c)] for _ in range(grid.size.r)]
                self.distance[grid.source.r][grid.source.c] = 0
                return next_point
            else:
                shortest_distance_from_agent_to_ghost = self.find_distance_to_ghost_on_history(grid)
                if self.is_non_history_directions_blocked(grid, dr, dc):
                    if shortest_distance_from_agent_to_ghost == INVALID \
                            and len(self.path_history_list) >= 2:
                        hist_point = self.path_history_list[-2]
                        next_point = Point(hist_point.r, hist_point.c)
                        self.path_history_set.remove(self.cur_point)
                        self.path_history_list.pop()
                        return next_point
                else:
                    if shortest_distance_from_agent_to_ghost != 1 \
                            and len(self.path_history_list) >= 2:
                        hist_point = self.path_history_list[-2]
                        next_point = Point(hist_point.r, hist_point.c)
                        self.path_history_set.remove(self.cur_point)
                        self.path_history_list.pop()
                        return next_point
        return self.cur_point

    @staticmethod
    def find_largest_survivability(survival_rate_list):
        largest_survival_directions = []
        largest_survival_rate = max(survival_rate_list)
        for i in range(len(survival_rate_list)):
            if survival_rate_list[i] == largest_survival_rate:
                largest_survival_directions.append(i)
        return largest_survival_directions, largest_survival_rate

    def update_distance(self, grid, dr, dc):
        # mimic agent 2 to choose which direction to move in order to run away from nearest ghosts
        # (so this part is highly similar to the part in agent 2's next_point funtion
        # and can resort to the comment there when in doubt)
        shortest_ghost_distance = [INVALID for _ in range(len(dr))]
        init_point = Point(self.cur_point.r, self.cur_point.c)
        init_distance = self.distance[self.cur_point.r][self.cur_point.c]
        queue = deque([(init_point, 0, init_distance)])
        is_target_reachable = False
        while len(queue) and not is_target_reachable:
            cur_point, init_step_i, cur_distance = queue.popleft()
            for i in range(len(dr)):
                next_point = Point(cur_point.r + dr[i], cur_point.c + dc[i])
                if cur_distance == init_distance:
                    init_step_i = i
                if grid.is_in_grid(next_point):
                    next_cell = grid.grid[next_point.r][next_point.c]
                    if not next_cell.is_blocked() and \
                            self.distance[next_point.r][next_point.c] == INVALID:
                        if next_cell.is_ghost_existed():
                            if shortest_ghost_distance[init_step_i] == INVALID:
                                shortest_ghost_distance[init_step_i] = cur_distance + 1 - init_distance
                            self.queue_history.append((next_point, init_step_i, cur_distance + 1))
                        else:
                            queue.append((next_point, init_step_i, cur_distance + 1))
                        self.distance[next_point.r][next_point.c] = cur_distance + 1
                if next_point == grid.target and not \
                        grid.grid[grid.target.r][grid.target.c].is_ghost_existed():
                    is_target_reachable = True
                    break

        if is_target_reachable:
            self.find_path_to_init_point(grid)
            self.path_history_set.add(self.path[-1])
            self.path_history_list.append(self.path[-1])
            next_point = self.path.pop()
            return next_point

        h_i = None
        if len(self.path_history_list) > 1:
            h_i = self.update_ghost_on_history_distance(grid, shortest_ghost_distance, dr, dc)
        non_nearest_ghost_directions = self.check_ghost_distance(shortest_ghost_distance)
        if len(non_nearest_ghost_directions):
            next_direction = non_nearest_ghost_directions[0]
            next_point = Point(self.cur_point.r + dr[next_direction], self.cur_point.c + dc[next_direction])
            if h_i is not None and next_point == self.path_history_list[-2]:
                self.path_history_set.remove(self.cur_point)
                self.path_history_list.pop()
            else:
                self.path_history_set.add(next_point)
                self.path_history_list.append(next_point)
            return next_point
        else:
            if h_i is not None and shortest_ghost_distance[h_i] == INVALID:
                next_point = Point(self.cur_point.r + dr[h_i], self.cur_point.c + dc[h_i])
                self.path_history_set.remove(self.cur_point)
                self.path_history_list.pop()
                return next_point
        return self.cur_point

    def find_path_to_init_point(self, grid):
        self.path = []
        for ghost in grid.ghosts.values():
            if ghost.cur_point not in self.path_history_set:
                self.distance[ghost.cur_point.r][ghost.cur_point.c] = INVALID
        if self.distance[grid.target.r][grid.target.c] != INVALID:
            self.path.append(grid.target)
            dr = [-1, 0, 1, 0]
            dc = [0, -1, 0, 1]
            init_distance = self.distance[self.cur_point.r][self.cur_point.c]
            cur_point = grid.target
            cur_distance = self.distance[cur_point.r][cur_point.c]
            while cur_distance > init_distance:
                for i in range(len(dr)):
                    next_point = Point(cur_point.r + dr[i], cur_point.c + dc[i])
                    if grid.is_in_grid(next_point):
                        next_distance = self.distance[next_point.r][next_point.c]
                        if INVALID != next_distance == (cur_distance - 1):
                            if next_distance > init_distance:
                                self.path.append(next_point)
                            cur_distance = next_distance
                            cur_point = next_point
                            break

    def update_ghost_on_history_distance(self, grid, shortest_ghost_distance, dr, dc):
        r = self.path_history_list[-2].r - self.cur_point.r
        c = self.path_history_list[-2].c - self.cur_point.c
        h_i = None
        for i in range(len(dr)):
            if dr[i] == r and dc[i] == c:
                h_i = i
                break
        for i, point in enumerate(self.path_history_list[::-1]):
            if grid.grid[point.r][point.c].is_ghost_existed():
                shortest_ghost_distance[h_i] = i
                break
        return h_i

    @staticmethod
    def check_ghost_distance(shortest_ghost_distance):
        non_nearest_ghost_directions = []
        directions = []
        shortest_distance = float('Inf')
        for i, distance in enumerate(shortest_ghost_distance):
            if distance != INVALID:
                directions.append(i)
                if distance < shortest_distance:
                    shortest_distance = distance
        if len(directions) > 1:
            for direction in directions:
                if shortest_distance < shortest_ghost_distance[direction]:
                    non_nearest_ghost_directions.append(direction)
        return non_nearest_ghost_directions

    def survivability(self, grid, next_possible_point, num_simulation=100):
        survival_rate = 0
        hung_rate = 0
        for i in range(num_simulation):
            is_survival, is_hung = self.simulate(grid, next_possible_point)
            # increment success count by one when a simulation ends up successfully reaching target cell
            survival_rate += int(is_survival)
            # increment hung count by one when a simulation ended up timeout
            hung_rate += int(is_hung)
        return survival_rate, hung_rate

    def simulate(self, grid, next_possible_point):
        temp_grid = deepcopy(grid)
        kwargs = {
            'grid': temp_grid,
            'pre_point': deepcopy(self.cur_point),
            'init_point': next_possible_point,
            'ghost_info': deepcopy(self.ghost_info),
            'distance': deepcopy(self.distance),
            'queue_history': deepcopy(self.queue_history),
            'path_history_set': deepcopy(self.path_history_set),
            'path_history_list': deepcopy(self.path_history_list),
            'path': deepcopy(self.path)
        }
        temp_grid.grid[self.cur_point.r][self.cur_point.c].remove_agent()
        temp_grid.agent = Agent2(kwargs)
        temp_grid.agent.path_history_set.add(next_possible_point)
        temp_grid.agent.path_history_list.append(next_possible_point)
        # simulate until there's an outcome for each possible direction (to move) during every simulation
        for d, point in enumerate(temp_grid.agent.path_history_list):
            temp_grid.agent.distance[point.r][point.c] = d
        step = 0
        is_terminated = self.update_and_check_termination(temp_grid, step)
        while not is_terminated:
            step += 1
            is_terminated = self.update_and_check_termination(temp_grid, step)
        return temp_grid.get_status() == 'SUCCESS', temp_grid.get_status() == 'TIMEOUT'

    @staticmethod
    def update_and_check_termination(temp_grid, step):
        if step > 0:
            temp_grid.move_agent()
        temp_grid.move_ghosts()
        temp_grid.update_status(step)
        status = temp_grid.get_status()
        is_terminated = status != 'ONGOING'
        if DEBUG >= 2:
            if is_terminated:
                print(status)
        return is_terminated

    def is_non_history_directions_blocked(self, grid, dr, dc):
        is_blocked = True
        for i in range(len(dr)):
            next_possible_point = Point(self.cur_point.r + dr[i], self.cur_point.c + dc[i])
            if grid.is_in_grid(next_possible_point) \
                    and len(self.path_history_list) >= 2 \
                    and next_possible_point != self.path_history_list[-2]:
                next_possible_cell = grid.grid[next_possible_point.r][next_possible_point.c]
                if not next_possible_cell.is_blocked():
                    is_blocked = False
                    break
        return is_blocked

    def find_distance_to_ghost_on_history(self, grid):
        for d, point in enumerate(self.path_history_list[::-1]):
            if grid.grid[point.r][point.c].is_ghost_existed():
                return d
        return INVALID


class Agent4:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.name = 4
            grid = kwargs['grid']
            self.pre_point = kwargs['pre_point']
            self.cur_point = kwargs['init_point']
            self.distance = kwargs['distance']
            if self.distance is None:
                self.distance = deepcopy(grid.distance)
            self.path = kwargs['path']
            if self.path is None:
                self.path = []
            self.find_path_to_cur_point(grid, set())

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.name = self.name
        d_copy.pre_point = deepcopy(self.pre_point)
        d_copy.cur_point = deepcopy(self.cur_point)
        d_copy.distance = deepcopy(self.distance)
        d_copy.path = deepcopy(self.path)
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

    def next_point(self, grid):
        ghost_info_set = set()
        if len(self.path) != 0:
            # detect whether at the moment there's at least one ghost in the previously calculated path
            is_nearby_ghost_on_path = self.check_ghost_on_path(grid)
            if not is_nearby_ghost_on_path:
                return self.path.pop()
        # re-plan for next direction to move when knowing there is at least one ghost in the previously calculated path
        self.mark_nearby_ghost(grid, ghost_info_set)
        self.find_path_to_cur_point(grid, ghost_info_set)
        if len(self.path) != 0:
            # move on the recalculated shortest path from current cell to target cell
            # (after knowing there is at least one ghost in the previously calculated path)
            return self.path.pop()
        return self.cur_point

    def find_path_to_cur_point(self, grid, ghost_info_set):
        # allows agent to forget all its path history (while agent 2 won't forget path history)
        # and treat the current standing cell as if the cell was its source cell
        # and then use breadth first search to get newly optimal distance from current cell to other cells
        for point in grid.reachable_points:
            self.distance[point.r][point.c] = INVALID
        dr = [-1, 0, 0, 1]
        dc = [0, -1, 1, 0]
        queue = deque([(deepcopy(self.cur_point), 0)])
        self.distance[self.cur_point.r][self.cur_point.c] = 0
        is_target_reachable = False
        while len(queue) and not is_target_reachable:
            temp_point, temp_distance = queue.popleft()
            for i in range(len(dr)):
                next_temp_point = Point(temp_point.r + dr[i], temp_point.c + dc[i])
                next_temp_distance = temp_distance + 1
                if grid.is_in_grid(next_temp_point):
                    next_cell = grid.grid[next_temp_point.r][next_temp_point.c]
                    if not next_cell.is_blocked() \
                        and next_temp_point not in ghost_info_set \
                            and self.distance[next_temp_point.r][next_temp_point.c] == INVALID:
                        self.distance[next_temp_point.r][next_temp_point.c] = next_temp_distance
                        if next_temp_point == grid.target:
                            is_target_reachable = True
                            break
                        queue.append((next_temp_point, next_temp_distance))
        # recalculated a shortest path from current cell to target cell
        self.path = []
        if is_target_reachable:
            self.path.append(grid.target)
            temp_point = grid.target
            temp_distance = self.distance[temp_point.r][temp_point.c]
            while temp_distance > 0:
                for i in range(len(dr)):
                    next_temp_point = Point(temp_point.r + dr[i], temp_point.c + dc[i])
                    if grid.is_in_grid(next_temp_point):
                        next_temp_distance = self.distance[next_temp_point.r][next_temp_point.c]
                        if INVALID != next_temp_distance == (temp_distance - 1):
                            if next_temp_distance > 0:
                                self.path.append(next_temp_point)
                            temp_distance = next_temp_distance
                            temp_point = next_temp_point
                            break

    def check_ghost_on_path(self, grid):
        dr = [-1, 0, 0, 1]
        dc = [0, -1, 1, 0]
        is_nearby_ghost_on_path = False
        if grid.grid[self.path[-1].r][self.path[-1].c].is_ghost_existed():
            is_nearby_ghost_on_path = True
        if not is_nearby_ghost_on_path:
            for i in range(len(dr)):
                next_point = Point(self.path[-1].r + dr[i], self.path[-1].c + dc[i])
                if grid.is_in_grid(next_point) and grid.grid[next_point.r][next_point.c].is_ghost_existed():
                    is_nearby_ghost_on_path = True
                    break
        return is_nearby_ghost_on_path

    def mark_nearby_ghost(self, grid, ghost_info_set):
        # mark and so treat the ghost that were within two cells away from agent 4 current standing cell at that moment
        # and some of the surrounding cells (which were those within two cells away from agent 4) as walls
        dr = [-1, 0, 0, 1]
        dc = [0, -1, 1, 0]
        for i in range(len(dr)):
            next_point = Point(self.cur_point.r + dr[i], self.cur_point.c + dc[i])
            if grid.is_in_grid(next_point) and grid.grid[next_point.r][next_point.c].is_ghost_existed():
                ghost_info_set.add(next_point)
        dr = [-1, -1, 1, 1]
        dc = [-1, 1, -1, 1]
        for i in range(len(dr)):
            next_point = Point(self.cur_point.r + dr[i], self.cur_point.c + dc[i])
            if grid.is_in_grid(next_point) and grid.grid[next_point.r][next_point.c].is_ghost_existed():
                ghost_info_set.add(next_point)
                ghost_info_set.add(Point(self.cur_point.r + dr[i], self.cur_point.c))
                ghost_info_set.add(Point(self.cur_point.r, self.cur_point.c + dc[i]))
        dr = [-2, 0, 0, 2]
        dc = [0, -2, 2, 0]
        for i in range(len(dr)):
            next_point = Point(self.cur_point.r + dr[i], self.cur_point.c + dc[i])
            if grid.is_in_grid(next_point) and grid.grid[next_point.r][next_point.c].is_ghost_existed():
                ghost_info_set.add(next_point)
                ghost_info_set.add(Point(self.cur_point.r + dr[i] // 2, self.cur_point.c + dc[i] // 2))
