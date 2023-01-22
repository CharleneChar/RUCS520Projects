from collections import deque

from network import *


class Scheme:
    def __init__(self, row_size, col_size):
        self.row_size = row_size
        self.col_size = col_size
        self.shortest_cmd_dict = None

    def run(self):
        # used to store all possible internal network map
        # that have connected unblocked cells
        self.shortest_cmd_dict = dict()
        # used to store only unique internal network maps
        map_set = set()
        # determine the first cell as the starting cell to reset to unblocked
        init_config = ([[True
                           for c in range(self.col_size)]
                           for r in range(self.row_size)], 0, 0)
        stack = [init_config]
        # represent internal network map in binary as integer
        # for reduce memory usage
        temp_key = self.network_map_key(init_config[0])
        _, temp_r, temp_c = init_config
        map_set.add(temp_key)
        # start turning more cells to be unblocked cells by dfs
        while len(stack):
            network_map, r, c = stack.pop()
            key = self.network_map_key(network_map)
            # traverse only network maps that are unseen
            # and have all unblocked cells being connected
            if key not in self.shortest_cmd_dict and \
                    self.is_connected(network_map):
                # self.print_adj_matrix(key)
                network = Network(kwargs={
                    'network_map': network_map
                })
                network.run()
                # get and store the shortest sequence of commands
                cmd_seq = network.shortest_cmd_seq
                self.shortest_cmd_dict[key] = \
                    cmd_seq if cmd_seq is not None else []
            # explore all neighbor nodes and itself
            for dr, dc in zip(DR9, DC9):
                next_r, next_c = r + dr, c + dc
                if 0 <= next_r < self.row_size \
                        and 0 <= next_c < self.col_size:
                    if network_map[next_r][next_c]:
                        next_adj_matrix = deepcopy(network_map)
                        next_adj_matrix[next_r][next_c] = False
                        next_config = (next_adj_matrix, next_r, next_c)
                        next_key = self.network_map_key(next_adj_matrix)
                        if DEBUG > -1:
                            self.print_network_map(next_key)
                            print()
                        # prune out those seen internal network maps
                        if next_key not in map_set:
                            map_set.add(next_key)
                            stack.append(next_config)
        # print all possible shortest sequence of commands
        # and the longest among all in sum
        self.print_shortest_cmd_dict()

    def network_map_key(self, network_map):
        key = 0
        for r in range(self.row_size):
            for c in range(self.col_size):
                key <<= 1
                if network_map[r][c]:
                    key |= 1
        return key

    def is_connected(self, network_map):
        cur_r, cur_c = None, None
        for r in range(self.row_size):
            for c in range(self.col_size):
                if not network_map[r][c]:
                    cur_r, cur_c = r, c
                    break
        if cur_r is None:
            return False
        temp_network_map = deepcopy(network_map)
        queue = deque([(cur_r, cur_c)])
        temp_network_map[cur_r][cur_c] = True
        while len(queue):
            r, c = queue.pop()
            for dr, dc in zip(DR, DC):
                next_r, next_c = r + dr, c + dc
                if 0 <= next_r < self.row_size \
                        and 0 <= next_c < self.col_size \
                        and not temp_network_map[next_r][next_c]:
                    temp_network_map[next_r][next_c] = True
                    queue.append((next_r, next_c))
        for r in range(self.row_size):
            for c in range(self.col_size):
                if not temp_network_map[r][c]:
                    return False
        return True

    def print_shortest_cmd_dict(self):
        max_length = 0
        for key, cmd_seq in self.shortest_cmd_dict.items():
            if len(cmd_seq) > max_length:
                max_length = len(cmd_seq)
            if DEBUG > -1:
                print('-' * 16)
                self.print_network_map(key)
                print(f'{len(cmd_seq)}: {cmd_seq}')
        print('=' * 16)
        for key, cmd_seq in self.shortest_cmd_dict.items():
            if len(cmd_seq) == max_length:
                self.print_network_map(key)
                print(f'{max_length}: {cmd_seq}')
                print()

    def print_network_map(self, key):
        network_map = [[0
                       for c in range(self.col_size)]
                       for r in range(self.row_size)]
        for r in range(self.row_size - 1, -1, -1):
            for c in range(self.col_size - 1, -1, -1):
                network_map[r][c] = key & 1
                key >>= 1
        for row in network_map:
            print(''.join(map(str, ['X' if _ else 'O' for _ in row])))

