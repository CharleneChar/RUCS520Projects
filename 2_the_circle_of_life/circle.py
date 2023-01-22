import time
from sortedcontainers import SortedSet


from graph import *
from util import *


class Circle:
    def __init__(self, kwargs):
        # is used to store count of agent capturing prey (i.e., capturing count)
        self.capturing_count = {name: None for name in kwargs['agent_name_set']}
        # is used to store count of agent being captured by predator (i.e., captured count)
        self.captured_count = {name: None for name in kwargs['agent_name_set']}
        # is used to store hung count
        self.hung_count = {name: None for name in kwargs['agent_name_set']}
        self.correct_prey_estimation_count = {name: None for name in kwargs['agent_name_set']}
        self.correct_predator_estimation_count = {name: None for name in kwargs['agent_name_set']}
        self.total_step = {name: None for name in kwargs['agent_name_set']}
        self.total_survey_count = {name: None for name in kwargs['agent_name_set']}
        self.num_graphs = kwargs['num_graphs']
        self.graph_size = kwargs['graph_size']
        self.is_bonus = kwargs['is_bonus']
        self.graph = None

    # to be completed in the future
    def read(self):
        pass

    # to be completed in the future
    def write(self):
        pass

    def run(self):
        # specify filename to store results of circle game for later reference
        filename = f'statistics_circle_text.txt'
        with open(filename, 'w') as f:
            for agent_name, _ in self.capturing_count.items():
                self.capturing_count[agent_name] = 0
                self.captured_count[agent_name] = 0
                self.hung_count[agent_name] = 0
                self.correct_prey_estimation_count[agent_name] = 0
                self.correct_predator_estimation_count[agent_name] = 0
                self.total_step[agent_name] = 0
                self.total_survey_count[agent_name] = 0
                start_time = time.time()
                if self.num_graphs > 0:
                    for _ in range(self.num_graphs):
                        kwargs = {
                            'agent_name': agent_name,
                            'graph_size': self.graph_size,
                            'is_bonus': self.is_bonus
                        }
                        step = 0
                        status = 'ONGOING'
                        while status == 'ONGOING':
                            if step == 0:
                                # initialize circle game
                                self.graph = Graph(kwargs=kwargs)
                            else:
                                # decide and move at each timestamp after initialization
                                self.graph.move_agent()
                                self.graph.update_status(step)
                                status = self.graph.get_status()
                                self.graph.update_correct_estimation_count()
                                if status == 'ONGOING':
                                    self.graph.move_prey()
                                    self.graph.update_status(step)
                                    status = self.graph.get_status()
                                    if status == 'ONGOING':
                                        self.graph.move_predator()
                                        self.graph.update_status(step)
                                        status = self.graph.get_status()

                            # debug by printing out current node for agent, prey, and predator
                            if DEBUG > -1:
                                print(f'time {step}: agent at {self.graph.agent.cur_node_name}')
                                print(f'time {step}: prey at {self.graph.prey.cur_node_name}')
                                print(f'time {step}: predator at {self.graph.predator.cur_node_name}')

                            # check and record outcome of circle game
                            if status != 'ONGOING':
                                # increment capturing count by one when agent can capture prey within time limit
                                self.capturing_count[agent_name] += int(status == 'CAPTURING')
                                # increment captured count by one when predator can capture agent within time limit
                                self.captured_count[agent_name] += int(status == 'CAPTURED')
                                # increment hung count by one
                                # when agent, prey, and predator are all still in circle alive
                                # but over time limit
                                self.hung_count[agent_name] += int(status == 'TIMEOUT')
                                self.correct_prey_estimation_count[agent_name] += self.graph.correct_prey_count
                                self.correct_predator_estimation_count[agent_name] += self.graph.correct_predator_count
                                self.total_step[agent_name] += step
                                self.total_survey_count[agent_name] += self.graph.agent.survey_count
                            step += 1

                    # show and collect every experiment data
                    print(f'{agent_name} in circle')
                    print(f'Capturing count: {self.capturing_count[agent_name]}')
                    print(f'Captured count: {self.captured_count[agent_name]}')
                    print(f'Capturing-to-Captured ratio: '
                          f'{(self.capturing_count[agent_name] / self.captured_count[agent_name]):.3f}')
                    print(f'Hung count: {self.hung_count[agent_name]}')
                    print(f'Survey rate: {(self.total_survey_count[agent_name] /  self.total_step[agent_name]):.3f}')
                    print(f'correct_prey_estimation_rate: '
                          f'{(self.correct_prey_estimation_count[agent_name] / self.total_step[agent_name]):.3f}')
                    print(f'correct_predator_estimation_rate: '
                          f'{(self.correct_predator_estimation_count[agent_name] / self.total_step[agent_name]):.3f}')
                    print("--- %.2f seconds ---" % (time.time() - start_time))
                    f.write(f'{agent_name} in circle\n')
                    f.write(f'Capturing count: {self.capturing_count[agent_name]}\n')
                    f.write(f'Captured count: {self.captured_count[agent_name]}\n')
                    f.write(f'Capturing-to-Captured ratio: '
                            f'{(self.capturing_count[agent_name] / self.captured_count[agent_name]):.3f}')
                    f.write(f'Hung count: {self.hung_count[agent_name]}\n')
                    f.write(f'Survey rate: {(self.total_survey_count[agent_name] / self.total_step[agent_name]):.3f}')
                    f.write(f'correct_prey_estimation_rate: '
                            f'{self.correct_prey_estimation_count[agent_name] / self.total_step[agent_name]}')
                    f.write(f'correct_predator_estimation_rate: '
                            f'{self.correct_predator_estimation_count[agent_name] / self.total_step[agent_name]}')
                    f.write("--- %.2f seconds ---" % (time.time() - start_time) + '\n')
            for agent_name in self.capturing_count:
                f.write(f'all capturing counts of {agent_name}: {self.capturing_count[agent_name]}\n')
            for agent_name in self.captured_count:
                f.write(f'all captured counts of {agent_name}: {self.captured_count[agent_name]}\n')
            for agent_name in self.captured_count:
                f.write(f'all capturing-to-captured ratios of {agent_name}: '
                        f'{(self.capturing_count[agent_name] / self.captured_count[agent_name]):.3f}\n')
            for agent_name in self.hung_count:
                f.write(f'all hung counts of {agent_name}: {self.hung_count[agent_name]}\n')
            for agent_name in self.correct_prey_estimation_count:
                f.write(f'all correct prey estimation rates of {agent_name}: '
                        f'{self.correct_prey_estimation_count[agent_name] / self.total_step[agent_name]}\n')
            for agent_name in self.correct_predator_estimation_count:
                f.write(f'all correct predator estimation rates of {agent_name}: '
                        f'{self.correct_predator_estimation_count[agent_name] / self.total_step[agent_name]}\n')
