from copy import *
from collections import deque
import os

from cell import *
from util import *
from state import *


class Network:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.cell_list = None
            self.row_size = None
            self.col_size = None
            # store the number of unblocked cells
            self.num_unblocked_cells = None
            # store the total number of cells
            # (including both blocked and unblocked cells)
            self.num_cells = None
            self.drone_belief_vector = None
            self.transition_matrices = None
            self.drone_belief_vectors = None
            # store the shortest command sequence
            # taken to locate where the drone is
            self.shortest_cmd_seq = None
            self.__init(kwargs)

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.row_size = deepcopy(self.row_size)
        d_copy.col_size = deepcopy(self.col_size)
        d_copy.grid = deepcopy(self.cell_list)
        return d_copy

    # to be continued
    def write(self):
        pass

    # to be continued
    def read(self):
        pass

    def read_given_network(self, filename):
        self.cell_list = []
        self.num_unblocked_cells = 0
        with open(filename, 'r') as f:
            # read in and set up given network in a list of lists of cells
            for i, line in enumerate(f.readlines()):
                if line != '\n':
                    self.cell_list.append([])
                    for j, ch in enumerate(line.rstrip()):
                        if ch == '_':
                            self.cell_list[i].append(Cell(r=i, c=j,
                                                          blocked=False))
                            self.num_unblocked_cells += 1
                        else:
                            self.cell_list[i].append(Cell(r=i, c=j,
                                                          blocked=True))
        self.row_size = len(self.cell_list)
        self.col_size = 0 if len(self.cell_list) == 0 else len(self.cell_list[0])
        self.num_cells = self.row_size * self.col_size

        if DEBUG > -1:
            print(len(self.cell_list))
            for r_i in range(len(self.cell_list)):
                for cell in self.cell_list[r_i]:
                    print(r_i, cell.is_blocked(), sep=': ')

    def convert_network_map_to_cell_list(self, network_map):
        # convert internal network map generated in question 4 to cell list
        self.row_size, self.col_size = len(network_map), len(network_map[0])
        self.cell_list = [[Cell(r=r, c=c, blocked=network_map[r][c])
                           for c in range(self.col_size)]
                          for r in range(self.row_size)]
        self.num_unblocked_cells = 0
        for r in range(self.row_size):
            for c in range(self.col_size):
                self.num_unblocked_cells += 0 if network_map[r][c] else 1
        self.num_cells = self.row_size * self.col_size

    def __init(self, kwargs):
        # initialize network related data structure and function
        # for question 3
        if 'filename' in kwargs:
            self.read_given_network(kwargs['filename'])
        # initialize network related data structure and function
        # for question 4
        elif 'network_map' in kwargs:
            self.convert_network_map_to_cell_list(kwargs['network_map'])
        # initialize belief vector for where the drone might be
        self.init_drone_belief_vector()
        # initialize all transition matrices
        # (each of which is used for one direction command)
        self.init_transition_matrices()

    def init_drone_belief_vector(self):
        self.drone_belief_vector = []
        for r_i in range(self.row_size):
            for c_i, cell in enumerate(self.cell_list[r_i]):
                # initialize belief vector for where the drone is
                # by setting equal belief to all unblocked cell
                # and zero to all blocked cell
                if not cell.is_blocked():
                    self.drone_belief_vector.append(1 / self.num_unblocked_cells)
                else:
                    self.drone_belief_vector.append(0)
        self.drone_belief_vector = np.array(self.drone_belief_vector)

    def init_transition_matrices(self):
        self.transition_matrices = [np.array([[0 for _ in range(self.num_cells)]
                                                for _ in range(self.num_cells)])
                                                for _ in ACT]
        for r_i in range(self.row_size):
            for c_i, cell in enumerate(self.cell_list[r_i]):
                cur_index = c_i + r_i * self.col_size
                if not cell.is_blocked():
                    # construct one transition matrix for each direction command
                    # i.e., (up transition matrix, down transition matrix,
                    # left transition matrix, and right transition matrix)
                    for i in ACT:
                        next_r, next_c = cell.r + DR[i], cell.c + DC[i]
                        if 0 <= next_r < self.row_size and \
                                0 <= next_c < self.col_size:
                            next_cell = self.cell_list[next_r][next_c]
                            next_index = next_c + next_r * self.col_size
                            if not next_cell.is_blocked():
                                self.transition_matrices[i][next_index][cur_index] = 1
                                continue
                        self.transition_matrices[i][cur_index][cur_index] = 1

    # start from all unblocked cells and converge to one unblocked cell
    def run(self, filename=None):
        # check if the statistics of sequence of command have already generated
        if filename is not None and os.path.isfile(filename):
            return False
        # run iterative deepening DFS
        for max_d in range(3 * self.num_cells):
            # used to record how many nonzero belief there are at the end of search
            # (which is expected to be one for termination)
            min_targets = float('Inf')
            # used to record the command sequence corresponding to min_targets
            shortest_cmd_seq = None
            # initialize first node (i.e., first state) for DFS start traversing
            initial_state = State(kwargs={
                'drone_belief_vector': self.drone_belief_vector,
                'parent': None,
                'act': None
            })
            state_stack = [(initial_state, 0)]
            # used to store distinct nodes (i.e., states) already traversed
            # for later tree pruning
            state_dict = dict()
            state_dict[initial_state.hash_value] = 0
            while len(state_stack):
                state, d = state_stack.pop()
                # bookkeep so far the shortest sequence of commands
                # to reach the last or terminating node (i.e., state)
                if state.num_non_zero_points < min_targets or \
                        (state.num_non_zero_points == min_targets and
                         len(state.cmd_seq) < len(shortest_cmd_seq)):
                    # update the latest shortest sequence of commands
                    # and number of nonzero belief left
                    min_targets = state.num_non_zero_points
                    shortest_cmd_seq = state.cmd_seq
                    if filename is not None:
                        with open(filename, 'a') as f:
                            f.write(f'{min_targets}\n')
                            f.write(f'{len(state.cmd_seq)}: {state.cmd_seq}\n')
                    if DEBUG > -1:
                        print(min_targets)
                        print(len(state.cmd_seq), state.cmd_seq, sep=': ')
                if DEBUG > -1:
                    print('-' * 16)
                    state.print(col_size=self.col_size)
                # stop branching when depth d reach restricted depth max_d
                if d == max_d:
                    continue
                # keep branching for four directions (i.e., UP, DOWN, LEFT, RIGHT)
                # where 0 represents UP, 1 represents DOWN, 2 represents LEFT,
                # and 3 represents RIGHT
                for i in ACT:
                    temp_belief_vector = np.matmul(self.transition_matrices[i],
                                                   state.drone_belief_vector)
                    next_state = State(kwargs={
                        'drone_belief_vector': temp_belief_vector,
                        'parent': state,
                        'act': i
                    })
                    # prune out any state that have been seen
                    # and having depth larger than the previously seen one
                    if next_state.hash_value not in state_dict or \
                            (d + 1) < state_dict[next_state.hash_value]:
                        state_dict[next_state.hash_value] = d + 1
                        state_stack.append((next_state, d + 1))
                        next_state.cmd_seq = deepcopy(state.cmd_seq)
                        next_state.cmd_seq.append(i)
                        next_state.parent = state
                        # check if only one entry (in drone belief vector) left
                        # that had belief equal to 1
                        if next_state.is_final_state():
                            next_state.print(col_size=self.col_size)
                            if filename is not None:
                                with open(filename, 'a') as f:
                                    f.write(f'{next_state.num_non_zero_points}\n')
                                    f.write(f'{len(next_state.cmd_seq)}: {next_state.cmd_seq}\n')
                            self.shortest_cmd_seq = next_state.cmd_seq
                            return True
            if filename is not None:
                with open(filename, 'a') as f:
                    f.write('-'*16 + '\n')
        return False

    def initial_network_without_move(self, filename):
        # used for visualize and display self-generated internal network map
        visualization(self.drone_belief_vector, self.row_size, self.col_size, filename)

    def move_one_direction(self, direction, filename):
        # used for visualize the heatmap after issuing a "DOWN" command
        drone_belief_vector = np.matmul(self.transition_matrices[direction],
                                        self.drone_belief_vector)
        visualization(drone_belief_vector, self.row_size, self.col_size, filename)

    # # below was another (flawless in some case) idea
    # # for finding the shortest sequence of commands
    # def init_drone_belief_vector_2(self):
    #     self.drone_belief_vectors = []
    #     temp_drone_belief_vector = \
    #         np.array([0 for _ in range(self.row_size * self.col_size)], dtype='float64')
    #     # prepare and store all possible initial belief vectors
    #     # (assuming one entry having drone being probability of one and
    #     # zero for other entries)
    #     for r_i in range(self.row_size):
    #         for c_i, cell in enumerate(self.cell_list[r_i]):
    #             cur_index = c_i + r_i * self.col_size
    #             if not cell.is_blocked():
    #                 temp_drone_belief_vector[cur_index] = 1
    #                 self.drone_belief_vectors.append(deepcopy(temp_drone_belief_vector))
    #                 temp_drone_belief_vector[cur_index] = 0
    #
    # def init_transition_matrices_2(self):
    #     self.transition_matrices = [np.array([[0 for _ in range(self.num_cells)]
    #                                             for _ in range(self.num_cells)],
    #                                             dtype='float64')
    #                                             for _ in ACT]
    #     for r_i in range(self.row_size):
    #         for c_i, cell in enumerate(self.cell_list[r_i]):
    #             cur_index = c_i + r_i * self.col_size
    #             if not cell.is_blocked():
    #                 for i in ACT:
    #                     next_r, next_c = cell.r + DR[i], cell.c + DC[i]
    #                     if 0 <= next_r < self.row_size and \
    #                             0 <= next_c < self.col_size:
    #                         next_cell = self.cell_list[next_r][next_c]
    #                         next_index = next_c + next_r * self.col_size
    #                         if not next_cell.is_blocked():
    #                             self.transition_matrices[i][next_index][cur_index] = 1 / 2
    #                             self.transition_matrices[i][cur_index][cur_index] = 1 / 2
    #                             continue
    #                     self.transition_matrices[i][cur_index][cur_index] = 1
    #
    # # start from one unblocked cell and percolate the whole map (i.e., every unblocked cell)
    # def run_2(self, filename):
    #     if os.path.isfile(filename):
    #         return False
    #     # used to store all possible sequences of commands (for fill up all unblocked cells)
    #     # starting from all possible initial belief vectors
    #     all_possible_cmd_seq = []
    #     # start traversing from each state of
    #     # all states with possible initial belief vectors
    #     for belief_vector in self.drone_belief_vectors:
    #         # used to indicate whether all unblocked cells are percolated with non-zero belief
    #         is_done = False
    #         # run deepening iterative DFS
    #         max_d = 0
    #         while not is_done and max_d < (3 * self.num_cells):
    #             # used to record how many non-zero belief there are at the end of search
    #             # (which is expected to be the total number of all unblocked cells
    #             # when search is terminated)
    #             max_targets = 0
    #             # used to record the reversed command sequence corresponding to max_targets
    #             shortest_rev_cmd_seq = None
    #             # initialize first state for DFS start traversing
    #             initial_state = State(kwargs={
    #                 'drone_belief_vector': deepcopy(belief_vector),
    #                 'parent': None,
    #                 'act': None
    #             })
    #             state_stack = [(initial_state, 0)]
    #             # used to store distinct states traversed for later tree pruning
    #             state_dict = dict()
    #             state_dict[initial_state.hash_value] = 0
    #             while not is_done and len(state_stack):
    #                 state, d = state_stack.pop()
    #                 # bookkeep so far the shortest reversed sequence of commands
    #                 # to reach the last state
    #                 if state.num_non_zero_points > max_targets or \
    #                         (state.num_non_zero_points == max_targets and
    #                          len(state.cmd_seq) < len(shortest_rev_cmd_seq)):
    #                     max_targets = state.num_non_zero_points
    #                     shortest_rev_cmd_seq = state.cmd_seq
    #                     with open(filename, 'a') as f:
    #                         f.write(f'{max_targets}\n')
    #                         f.write(f'{len(state.cmd_seq)}: {state.cmd_seq}\n')
    #                     print(max_targets)
    #                     print(len(state.cmd_seq), state.cmd_seq, sep=': ')
    #                 if DEBUG > -1:
    #                     print('-' * 16)
    #                     state.print(col_size=self.col_size)
    #                 # stop branching when depth d reach restricted depth max_d
    #                 if d == max_d:
    #                     continue
    #                 # keep branching for four directions (i.e., up, down, left, right)
    #                 for i in ACT:
    #                     temp_belief_vector = np.matmul(self.transition_matrices[i],
    #                                                    state.drone_belief_vector)
    #                     next_state = State(kwargs={
    #                         'drone_belief_vector': temp_belief_vector,
    #                         'parent': state,
    #                         'act': i
    #                     })
    #                     # prune out any state that have been seen
    #                     # and having depth larger than the previously seen one
    #                     if next_state.hash_value not in state_dict or \
    #                             (d + 1) < state_dict[next_state.hash_value]:
    #                         state_dict[next_state.hash_value] = d + 1
    #                         state_stack.append((next_state, d + 1))
    #                         next_state.cmd_seq = deepcopy(state.cmd_seq)
    #                         next_state.cmd_seq.append(i)
    #                         next_state.parent = state
    #                         # check if the total number of non-zero entries (in drone belief vector)
    #                         # equal to the number of unblocked cells
    #                         if next_state.is_final_state_2(self.num_unblocked_cells):
    #                             next_state.print(col_size=self.col_size)
    #                             print(next_state.reverse_cmd_seq())
    #                             # record command sequence (reversed back from reversed command sequence)
    #                             all_possible_cmd_seq.append(next_state.reverse_cmd_seq())
    #                             with open(filename, 'a') as f:
    #                                 f.write(f'{next_state.num_non_zero_points}\n')
    #                                 f.write(f'{len(next_state.cmd_seq)}: {next_state.cmd_seq}\n')
    #                                 f.write(f'{next_state.reverse_cmd_seq()}\n')
    #                             is_done = True
    #                             break
    #             max_d += 1
    #             with open(filename, 'a') as f:
    #                 f.write('-' * 16 + '\n')
    #     with open(filename, 'a') as f:
    #         min_len = float('Inf')
    #         min_cmd_seq = None
    #         for cmd_seq in all_possible_cmd_seq:
    #             if len(cmd_seq) < min_len:
    #                 min_len = len(cmd_seq)
    #                 min_cmd_seq = cmd_seq
    #             f.write(f'{len(cmd_seq)}: {cmd_seq}\n')
    #         f.write('=' * 16 + '\n')
    #         f.write(f'{min_len}: {min_cmd_seq}\n')
    #     return True
