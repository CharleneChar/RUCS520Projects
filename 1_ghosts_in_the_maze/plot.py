# plot graph to display statistics
import matplotlib.pyplot as plt

stats = {
    'agent_1': [25, 23, 19, 15, 12, 5, 6, 6, 4, 2, 5, 1, 3],
    'agent_2': [25, 23, 19, 15, 15, 11, 5, 6, 1, 3, 1, 1, 2],
    'agent_3': [25, 19, 15, 12, 14, 8, 2, 3, 6, 4, 4, 2, 3],
    'agent_4': [25, 25, 19, 15, 18, 14, 12, 12, 8, 6, 3, 2, 2]
}
plt.figure(figsize=(15, 6), dpi=100)
colors = ['palevioletred', 'steelblue', 'teal', 'rosybrown']
for i, agent in enumerate(stats):
    plt.plot(stats[agent],
             label=f'{agent}', color=colors[i])
plt.xlabel('Number of Ghosts')
plt.ylim((0, 26))
plt.ylabel('Success Count')
plt.title('Agent Performance')
plt.legend()
plt.savefig('comparison_different_agents_under1000simulation', bbox_inches='tight')
plt.show()
