import time


from graph import *
from util import *


class Circle:
    def __init__(self, kwargs):
        self.average_step = {name: None for name in kwargs['agent_name_set']}
        self.num_graphs = kwargs['num_graphs']
        self.num_simu = kwargs['num_simu']
        self.graph_size = kwargs['graph_size']
        self.graph = None
        self.step = 0
        self.s = 0

    # to be completed in the future
    def read(self):
        pass

    # to be completed in the future
    def write(self):
        pass

    def run(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'w') as f:
                f.write(f'')
        for agent_name, _ in self.average_step.items():
            self.average_step[agent_name] = 0
            mse = None
            start_time = time.time()
            for graph_name in range(self.num_graphs):
                kwargs = {
                    'agent_name': agent_name,
                    'graph_size': self.graph_size,
                    'graph_name': graph_name,
                    'is_object_kept': True
                }

                # used for debugging new agent
                if DEBUG > -2:
                    if agent_name == 'none':
                        self.graph = Graph(kwargs=kwargs)
                        continue

                self.graph = None
                # simulate the game under one same graph a specified number of times
                for simu_name in range(self.num_simu):
                    self.execute_game_once(agent_name, kwargs)

                # record the accuracy for model (using mse)
                if agent_name == 'agent_v' and self.graph is not None:
                    mse = self.graph.agent.model_mse
                if (agent_name == 'agent_v_partial' or agent_name == 'agent_another_u_partial')\
                        and self.graph is not None:
                    mse = self.graph.agent.model_mse / self.graph.agent.model_num_prediction

                # after one game terminate,
                # record all partial state's info generated along one game
                if agent_name == 'agent_u_partial' and self.graph is not None:
                    self.graph.agent.write(graph_name)

                # record state statistics comparison between agent u* and agent 1, agent 2
                if agent_name == 'agent_u_star' and self.graph is not None:
                    self.graph.get_state_with_different_choice_between_agents()
                    # get states with highest finite U*
                    # 1th: 26,22,25
                    # 2th: 27,22,25
                    largest_finite_u_star_state_list = self.graph.agent.get_state_with_largest_finite_u_star()
                    for i, largest_finite_u_star_state in enumerate(largest_finite_u_star_state_list):
                        print(f'{i+1}th: {largest_finite_u_star_state}')

            # show and collect every experiment data for one agent
            self.one_agent_performance_statistics(filename, agent_name, start_time, self.num_graphs, self.num_simu, mse)

        # record all agents' statistics
        self.overall_performance_statistics(filename)

    def execute_game_once(self, agent_name, kwargs):
        is_object_kept = kwargs['is_object_kept']
        self.step = 0
        status = 'ONGOING'
        while status != 'CAPTURING':
            if self.step == 0:
                # initialize circle game
                if is_object_kept and self.graph is not None:
                    self.graph.refresh(kwargs)
                else:
                    self.graph = Graph(kwargs=kwargs)
            else:
                # decide and move at each timestamp after initialization
                # until prey is caught by agent
                self.graph.move_agent()
                self.graph.update_status(self.step)
                status = self.graph.get_status()
                if status != 'CAPTURING':
                    self.graph.move_prey()
                    self.graph.update_status(self.step)
                    status = self.graph.get_status()
                    if status != 'CAPTURING':
                        self.graph.move_predator()
                        self.graph.update_status(self.step)
                        status = self.graph.get_status()

            # debug by printing out current node for agent, prey, and predator
            if DEBUG > -1:
                print(f'{self.step}: Ag-{self.graph.agent.cur_node_name}'
                      f', Py-{self.graph.prey.cur_node_name}'
                      f', Pd-{self.graph.predator.cur_node_name}')
                # print(self.graph)

            self.step += 1
        # compute average step taken to catch prey
        self.average_step[agent_name] += self.step

    def one_agent_performance_statistics(self, filename, agent_name, start_time,
                                         graph_count, simu_count_per_graph, mse=None):
        # count average step for one agent
        self.average_step[agent_name] /= (graph_count * simu_count_per_graph)
        print(f'{agent_name} in circle')
        print(f'Avg Steps taken to catch prey: {self.average_step[agent_name]:.3f}')
        if agent_name == 'agent_v' \
                or agent_name == 'agent_v_partial' \
                or agent_name == 'agent_another_u_partial':
            print(f'model accuracy (in terms of having mse): {mse:.3f}')
        print("--- %.2f seconds ---" % (time.time() - start_time))
        with open(filename, 'a') as f:
            f.write(f'{agent_name} in circle\n')
            f.write(f'Avg Steps taken to catch prey: {self.average_step[agent_name]:.3f}\n')
            if agent_name == 'agent_v' \
                    or agent_name == 'agent_v_partial' \
                    or agent_name == 'agent_another_u_partial':
                f.write(f'model accuracy (in terms of having mse): {mse:.3f}\n')
            f.write("--- %.2f seconds ---" % (time.time() - start_time) + '\n')

    def overall_performance_statistics(self, filename):
        # record average step for all agents
        with open(filename, 'a') as f:
            for agent_name in self.average_step:
                f.write(f'All Avg Steps taken to catch prey of {agent_name}: {self.average_step[agent_name]:.3f}\n')
