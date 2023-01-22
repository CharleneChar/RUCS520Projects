import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main():
    # stats = {'predator_catch_agent': [85, 22, 141, 77, 973, 694, 1709, 1470],
    #          'agent_catch_prey': [9915, 9978, 9859, 9923, 9027, 9306, 8291, 8530],
    #          'simulation_hang': [0, 0, 0, 0, 0, 0, 0, 0]}
    # # predator_catch_agent and agent_catch_prey and simulation_hang
    # plot(stats, 'performance')
    #
    # stats = {'exact_prey_estimation': [0.125, 0.087, 1.000, 1.000, 0.063, 0.061],
    #          'exact_predator_estimation': [1.000, 1.000, 0.715, 0.833, 0.713, 0.718]}
    # # exact prey and predator estimation
    # plot(stats, 'exact_estimation')

    stats = {'predator_catch_agent': [4039, 3959, 2702, 2471, 2408],
             'agent_catch_prey': [5961, 6041, 7298, 7529, 7592],
             'simulation_hang': [0, 0, 0, 0, 0]}
    # predator_catch_agent and agent_catch_prey and simulation_hang
    plot(stats, 'performance_drone')

    stats = {'exact_prey_estimation': [0.056, 0.056, 0.054, 0.055, 0.055],
             'exact_predator_estimation': [0.49, 0.49, 0.54, 0.54, 0.54]}
    # exact prey and predator estimation
    plot(stats, 'exact_estimation_drone')


def plot(stats, plot_name):
    # plot graph to display statistics
    plt.figure(figsize=(10, 6), dpi=120)
    colors = ['palevioletred', 'steelblue', 'teal']
    for i, count_stats in enumerate(stats):
        plt.plot(stats[count_stats], 'bo-', label=f'{count_stats}', color=colors[i])

    # for i, difference_stats in enumerate(stats):
    #     plt.plot([j + 1 for j in range(len(stats['cur_difference']))],
    #              stats[difference_stats],
    #              'o',
    #              label=f'{difference_stats}', color=colors[-1])

    # colors = [random.choice(list(mcolors.CSS4_COLORS))
    #           for _ in range(len(mcolors.CSS4_COLORS))]
    # for i, loss_stats in enumerate(stats):
    #     plt.plot(stats[loss_stats],
    #              label=f'{loss_stats}', color=colors[i])

    if plot_name == 'performance_drone':
        plt.xticks(np.arange(0, 4 + 1, 1), ['extended_agent7', 'extended_agent8'
                                            , 'revised_extended_agent7', 'revised_extended_agent8', 'agent9'])
        plt.xlabel('agent with defective drone strategy', labelpad=15)
        plt.title('Counts Over 10000 Simulations', y=1.05)
        plt.ylabel('count')
        # zip joins x and y coordinates in pairs
        for x, y in zip(np.arange(0, 5 + 1, 1), stats['predator_catch_agent']):
            label = "{:.0f}".format(y)
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        # zip joins x and y coordinates in pairs
        for x, y in zip(np.arange(0, 5 + 1, 1), stats['agent_catch_prey']):
            label = "{:.0f}".format(y)
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 5),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        # zip joins x and y coordinates in pairs
        for x, y in zip(np.arange(0, 5 + 1, 1), stats['simulation_hang']):
            label = "{:.0f}".format(y)
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, -12),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
    else:
        plt.xticks(np.arange(0, 4 + 1, 1), ['extended_agent7', 'extended_agent8'
                                            , 'revised_extended_agent7', 'revised_extended_agent8', 'agent9'])
        plt.xlabel('agent with defective drone strategy', labelpad=15)
        plt.title('Accurate Estimation Rates', y=1.05)
        plt.ylabel('rate')
        plt.ylim((0, 1.2))
        # zip joins x and y coordinates in pairs
        for x, y in zip(np.arange(0, 5 + 1, 1), stats['exact_prey_estimation']):
            label = "{:.3f}".format(y)
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        # zip joins x and y coordinates in pairs
        for x, y in zip(np.arange(0, 5 + 1, 1), stats['exact_predator_estimation']):
            label = "{:.3f}".format(y)
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
    plt.legend()
    plt.savefig(f'{plot_name}.png', bbox_inches='tight')
    plt.show()


main()
