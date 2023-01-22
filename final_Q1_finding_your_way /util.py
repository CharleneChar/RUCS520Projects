import matplotlib.pyplot as plt
import random
import numpy as np
import os


INVALID = -1
DEBUG = -1

# represent UP command as 0, DOWN command as 1,
# LEFT command as 2, and RIGHT command as 3
UP = 0
DN = 1
LT = 2
RT = 3

ACT = (UP, DN, LT, RT)
REV_ACT = (DN, UP, RT, LT)

DR = (-1, 1, 0, 0)
DC = (0, 0, -1, 1)

DR9 = (-1, -1, -1, 0, 0, 0, 1, 1, 1)
DC9 = (-1, 0, 1, -1, 0, 1, -1, 0, 1)


def generate_network(filename, row_size, col_size):
    network = [[True for _ in range(col_size)] for _ in range(row_size)]
    visited = [[False for _ in range(col_size)] for _ in range(row_size)]
    r, c = random.randint(0, row_size - 1), random.randint(0, col_size - 1)
    while True:
        network[r][c] = False
        visited[r][c] = True
        act_choices = []
        for i in ACT:
            next_r, next_c = r + DR[i], c + DC[i]
            if 0 <= next_r < row_size and \
                    0 <= next_c < col_size and not visited[next_r][next_c]:
                act_choices.append(i)
        if len(act_choices) == 0:
            break
        random.shuffle(act_choices)
        act = act_choices[0]
        for i in act_choices[1:]:
            visited[r + DR[i]][c + DC[i]] = True
        r, c = r + DR[act], c + DC[act]
    with open(filename, 'w') as f:
        for row in network:
            f.write(''.join(['X' if _ else '_' for _ in row]) + '\n')


def visualization(drone_belief_vector, row_size, col_size, filename):
    if filename is not None and os.path.isfile(filename):
        return False
    drone_belief_matrix = [[] for _ in range(row_size)]
    for r_i in range(row_size):
        for c_i in range(col_size):
            cur_index = c_i + r_i * col_size
            drone_belief_matrix[r_i].append(drone_belief_vector[cur_index])
    fig, ax = plt.subplots()
    im = ax.imshow(drone_belief_matrix, cmap='gray', extent=(0, col_size, row_size, 0))
    ax.grid(color="dimgray", linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(0, col_size + 1, 1))
    ax.set_yticks(np.arange(0, row_size + 1, 1))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    fig.tight_layout()
    # plt.show()
    fig.savefig(filename)
    # print(drone_belief_vector)
    return True
