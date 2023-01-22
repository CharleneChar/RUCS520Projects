import time

from network import *
from scheme import *


def main():
    # used to count the amount of time needed to complete each task
    start = time.time()

    # solve question 3
    original_file = 'Thor23-SA74-VERW-Schematic(Classified).txt'
    # generate a smaller scale internal network map with the following:
    # generate_network('gen_test_7x7.txt', 7, 7)
    smaller_scaled_file = 'gen_test_7x7.txt'
    # feed file to the program as keyword argument (for the program to read it in)
    kwargs = {
        'filename': smaller_scaled_file,
    }
    network = Network(kwargs)
    network.initial_network_without_move('network_map.png')
    # get the heatmap after issuing "DOWN" (i.e., represented by an integer, 1)
    network.move_one_direction(1, 'visualization.png')
    # get the shortest sequence of commands and store result in a file
    network.run(filename=f'statistics_7x7.txt')

    # solve question 4
    scheme = Scheme(5, 5)
    # get the longest shortest sequence among all shortest sequence
    scheme.run()

    # used to print the amount of time needed to complete each task
    print(f'{time.time() - start:.2f} (s)')


if __name__ == '__main__':
    main()
