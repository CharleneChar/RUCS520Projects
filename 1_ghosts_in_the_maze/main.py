from maze import *


def main():
    """
    You can change the following setting to see different results for different settings.
    """
    kwargs = {
        # name which agent strategy to be applied
        # (e.g., if you'd like to see how all agent strategy perform in a same file,
        # you could code as follows, and if you'd like to see only agent 4 strategy performance,
        # you could code as 'agent_name_set': SortedSet([4])
        'agent_name_set': SortedSet([1, 2, 3, 4]),
        # specify how many mazes for each agent strategy to test on
        'num_grids': 25,
        # specify maze size (e.g., if it's an 8 by 8 maze, it would be coded as follows)
        'grid_size': Point(8, 8),
        # specify different number of ghosts
        # to see different performance of each agent strategy
        # under different number of ghosts
        'num_ghost_list': [i for i in range(0, 26, 1)]
    }
    # start maze game
    maze = Maze(kwargs)
    maze.run()


if __name__ == '__main__':
    main()
