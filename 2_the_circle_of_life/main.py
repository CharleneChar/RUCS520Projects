from circle import *


def main():
    """
    You can change the following setting to see different results for different settings.
    """
    kwargs = {
        # name which agent strategy to be applied
        # (e.g., if you'd like to see how all agent strategy perform in a same file,
        # you could code as follows, and if you'd like to see only agent 4 strategy performance,
        # you could code as 'agent_name_set': ['agent4']
        # (note that extended_agent_7 and extended_agent_8 are the agents not accounting for the defective drone
        # and revised_extended_agent_7 and revised_extended_agent_8 are the agents accounting for the defective drone)
        'agent_name_set': ['agent1', 'agent2', 'agent3', 'agent4', 'agent5', 'agent6', 'agent7', 'agent8'
                           , 'extended_agent7', 'extended_agent8'
                           , 'revised_extended_agent7', 'revised_extended_agent8', 'agent9'],
        # specify how many circles for each agent strategy to test on
        'num_graphs': 10000,
        # specify circle size (e.g., if it's a circle with 50 nodes, it would be coded as follows)
        'graph_size': 50,
        # specify whether to see bonus or not (only constructed well for agent 1 to 8)
        'is_bonus': False
    }
    # start circle game
    circle = Circle(kwargs)
    circle.run()


if __name__ == '__main__':
    main()
