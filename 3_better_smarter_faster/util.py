from copy import *
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

INVALID = -1
DEBUG = -1


class ListDict(object):
    def __init__(self):
        self.item_to_position = {}
        self.items = []

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.item_to_position = self.item_to_position
        d_copy.items = self.items
        return d_copy

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def add_item(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove_item(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        return random.choice(self.items)

    def write(self, f):
        pass

    def read(self, f):
        pass


class State:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.agent_node_name = kwargs['agent_node_name']
            self.prey_node_name = kwargs['prey_node_name']
            self.predator_node_name = kwargs['predator_node_name']
            self.graph_size = kwargs['graph_size']

    def __str__(self):
        return f'{self.agent_node_name},{self.prey_node_name},{self.predator_node_name}'

    def __repr__(self):
        return f'Ag: {self.agent_node_name}, Py: {self.prey_node_name}, Pd: {self.predator_node_name}'

    def __hash__(self):
        return (self.agent_node_name - 1) * self.graph_size ** 2 + \
               (self.prey_node_name - 1) * self.graph_size + \
               (self.predator_node_name - 1)

    def __eq__(self, other):
        return isinstance(other, State) and \
               self.agent_node_name == other.agent_node_name and \
               self.prey_node_name == other.prey_node_name and \
               self.predator_node_name == other.predator_node_name and \
               self.graph_size == other.graph_size

    def write(self, f):
        f.write(f'{self.agent_node_name},{self.prey_node_name},{self.predator_node_name},{self.graph_size}\n')

    def read(self, f):
        self.agent_node_name, self.prey_node_name, self.predator_node_name, self.graph_size = \
            map(int, f.readline().rstrip().split(','))

    def is_prey_captured(self):
        return self.agent_node_name == self.prey_node_name

    def is_agent_captured(self):
        return self.agent_node_name != self.prey_node_name and self.agent_node_name == self.predator_node_name


class PartialState:
    def __init__(self, kwargs=None):
        if kwargs is not None:
            self.agent_node_name = kwargs['agent_node_name']
            self.predator_node_name = kwargs['predator_node_name']
            self.prey_belief_vector = kwargs['prey_belief_vector']

    def __str__(self):
        s = ''
        s += f'{self.agent_node_name}\n'
        s += f'{self.predator_node_name}\n'
        s += ','.join(map(str, self.prey_belief_vector)) + '\n'
        return s

    def __repr__(self):
        s = ''
        s += f'Ag: {self.agent_node_name}\n'
        s += f'Pd: {self.predator_node_name}\n'
        s += 'Py' + ','.join(map(str, self.prey_belief_vector)) + '\n'
        return s

    def write(self, f):
        f.write(f'{self.agent_node_name}\n')
        f.write(f'{self.predator_node_name}\n')
        f.write(','.join(map(str, self.prey_belief_vector)) + '\n')

    def read(self, f):
        self.agent_node_name = map(int, f.readline().rstrip().split(','))
        self.predator_node_name = map(int, f.readline().rstrip().split(','))
        self.prey_belief_vector = list(map(float, f.readline().rstrip().split(',')))

    def is_prey_captured(self):
        return self.prey_belief_vector[self.agent_node_name] == 1

    def is_agent_captured(self):
        return self.prey_belief_vector[self.agent_node_name] != 1 \
               and self.agent_node_name == self.predator_node_name


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


# to be completed in the future
def custom_plot(state, adj_list, plot_filename):
    # used for plotting graph
    # with open('rsc/0.graph', 'r') as f:
    #     graph_size, = map(int, f.readline().rstrip().split(','))
    #     for i in range(1, graph_size + 1):
    #         for j in map(int, f.readline().rstrip().split(',')):
    #             if i < j:
    #                 print(f'{i}-{j}')
    pass


def comp_plot():
    # plot performance of agents in partial information setting
    stats = {'avg_step': [9.899, 17.629, 10.086, 8.185]}
    # plot graph to display statistics
    plt.figure(figsize=(10, 6), dpi=120)
    colors = ['teal', 'palevioletred', 'steelblue']
    for i, count_stats in enumerate(stats):
        plt.plot(stats[count_stats], 'bo-', label=f'{count_stats}', color=colors[i])
    plt.xticks(np.arange(0, 4, 1), ['agent U*', 'agent 1', 'agent 2', 'agent V'])
    plt.xlabel('agent i strategy', labelpad=15)
    plt.title('Average Steps Over 3000 Simulations', y=1.05)
    plt.ylabel('steps')
    # zip joins x and y coordinates in pairs
    for x, y in zip(np.arange(0, 4 + 1, 1), stats['avg_step']):
        label = "{:.0f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(12, 5),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    plt.legend()
    plt.savefig(f'complete_info_comparison.png', bbox_inches='tight')
    plt.show()

    # plot performance of agents in partial information setting
    stats = {'avg_step': [18.535, 34.280, 27.257, 17.443, 17.630]}
    # plot graph to display statistics
    plt.figure(figsize=(10, 6), dpi=120)
    colors = ['teal', 'palevioletred', 'steelblue']
    for i, count_stats in enumerate(stats):
        plt.plot(stats[count_stats], 'bo-', label=f'{count_stats}', color=colors[i])
    plt.xticks(np.arange(0, 5, 1), ['agent U partial', 'agent 3', 'agent 4', 'agent V partial',
                                    'another agent u partial'])
    plt.xlabel('agent i strategy', labelpad=15)
    plt.title('Average Steps Over 3000 Simulations', y=1.05)
    plt.ylabel('steps')
    # zip joins x and y coordinates in pairs
    for x, y in zip(np.arange(0, 5 + 1, 1), stats['avg_step']):
        label = "{:.0f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(12, 5),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    plt.legend()
    plt.savefig(f'partial_info_comparison.png', bbox_inches='tight')
    plt.show()

