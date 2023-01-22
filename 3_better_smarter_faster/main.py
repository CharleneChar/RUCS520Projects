from circle import *


def main():
    """
    You can change the following setting to see different results for different settings.
    """
    kwargs = {
        # name which agent strategy to be applied
        # (e.g., if you'd like to see how all agent strategy perform in a same file,
        # you could code as follows, and if you'd like to see only agent u star strategy performance,
        # you could code as 'agent_name_set': ['agent_u_star']
        # (note that agent_another_u_star is just another agent U*)
        # (note that agent_another_u_partial is just another agent U partial)
        'agent_name_set': ['agent_u_star', 'agent1', 'agent2', 'agent_v',
                           'agent_u_partial', 'agent3', 'agent4', 'agent_v_partial',
                           'agent_another_u_partial'],
        # specify how many circles (i.e, graphs with 50 nodes) for each agent strategy to test on
        # (note that this number is fixed in this project)
        'num_graphs': 1,
        # specify how many simulation for each circle (i.e., a graph with 50 nodes)
        'num_simu': 3000,
        # specify circle size (e.g., if it's a circle with 50 nodes, it would be coded as follows)
        # (note that this number is fixed in this project)
        'graph_size': 50,
    }
    # start circle game
    circle = Circle(kwargs)
    # specify filename to store performance results of circle game for later reference
    filename = f'statistics_circle_text_test_all.txt'
    circle.run(filename)


if __name__ == '__main__':
    main()
    # used for plot
    # comp_plot()
