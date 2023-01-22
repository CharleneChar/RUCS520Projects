import time
from sortedcontainers import SortedSet


from grid import *
from util import *


class Maze:
    def __init__(self, kwargs):
        # is used to store success count
        self.agent_survivability = {name: None for name in kwargs['agent_name_set']}
        # is used to store hung count
        self.agent_hung_rate = {name: None for name in kwargs['agent_name_set']}
        self.num_grids = kwargs['num_grids']
        self.grid_size = kwargs['grid_size']
        self.grid = None
        self.num_ghost_list = kwargs['num_ghost_list']

    def run(self):
        # specify filename to store results of maze game for later reference
        filename = f'statistics_maze25_size8x8_simu10000_gh0to25.txt'
        with open(filename, 'w') as f:
            for agent_name, _ in self.agent_survivability.items():
                self.agent_survivability[agent_name] = [0 for _ in range(len(self.num_ghost_list))]
                self.agent_hung_rate[agent_name] = [0 for _ in range(len(self.num_ghost_list))]
                for num_ghost_i, num_ghost in enumerate(self.num_ghost_list):
                    start_time = time.time()
                    if self.num_grids > 0:
                        for _ in range(self.num_grids):
                            kwargs = {
                                'grid_size': self.grid_size,
                                'num_ghosts': num_ghost,
                                'agent_name': agent_name,
                            }
                            step = 0
                            status = 'ONGOING'
                            while status == 'ONGOING':
                                if step == 0:
                                    # initialize maze game
                                    self.grid = Grid(kwargs=kwargs)
                                else:
                                    # decide and move at each timestamp after initialization
                                    self.grid.move_agent()
                                    self.grid.move_ghosts()
                                self.grid.update_status(step)
                                status = self.grid.get_status()
                                # check and record outcome of maze game
                                if status != 'ONGOING':
                                    # increment success count by one when agent can reach target cell within time limit
                                    self.agent_survivability[agent_name][num_ghost_i] += int(status == 'SUCCESS')
                                    # increment hung count by one when agent is still in maze alive
                                    # but not on target cell over time limit
                                    self.agent_hung_rate[agent_name][num_ghost_i] += int(status == 'TIMEOUT')
                                step += 1

                                if DEBUG >= 1:
                                    if DEBUG >= 0:
                                        if status != 'ONGOING':
                                            print(status)
                                    print(self.grid)

                    # show and collect every experiment data
                    print(f'Agent{agent_name} in maze with {num_ghost} ghosts')
                    print(f'Survivability: {self.agent_survivability[agent_name][num_ghost_i]}')
                    print("--- %.2f seconds ---" % (time.time() - start_time))
                    f.write(f'Agent{agent_name} in maze with {num_ghost} ghosts\n')
                    f.write(f'Survivability: {self.agent_survivability[agent_name][num_ghost_i]}\n')
                    f.write("--- %.2f seconds ---" % (time.time() - start_time) + '\n')
            for agent_name in self.agent_survivability:
                f.write(f'all survivability of Agent{agent_name}: {self.agent_survivability[agent_name]}\n')
            for agent_name in self.agent_hung_rate:
                f.write(f'all hung rates of Agent{agent_name}: {self.agent_hung_rate[agent_name]}\n')

    # to be completed in the future
    def read(self):
        pass

    # to be completed in the future
    def write(self):
        pass
