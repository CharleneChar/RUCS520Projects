from sortedcontainers import SortedDict
from copy import *


class Cell:
    UNBLOCKED = 'O'
    BLOCKED = 'X'
    AGENT = 'A'
    GHOST = 'G'
    UNKNOWN = '?'

    def __init__(self, blocked=False):
        self.blocked = blocked
        self.agent = None
        self.ghosts = SortedDict()

    def __repr__(self):
        state = f'blocked: {self.blocked} \n'
        state += f'a{self.agent.name}'
        state += f'# ghosts: {len(self.ghosts)} \n'
        state += ','.join([ghost.name for ghost in self.ghosts]) + '\n'
        return state

    def __str__(self):
        string = ''
        string += f'{Cell.BLOCKED if self.blocked else Cell.UNBLOCKED}'
        if self.agent:
            string += f'-a{self.agent.name}'
        if len(self.ghosts):
            string += '-'
            string += '-'.join(self.ghosts)
        return string

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.blocked = self.blocked
        d_copy.agent = deepcopy(self.agent)
        d_copy.ghosts = SortedDict()
        for k, v in self.ghosts.items():
            d_copy.ghosts[k] = deepcopy(v)
        return d_copy

    def set_blocked(self, blocked):
        self.blocked = blocked

    def write(self, f):
        f.write(f'{int(self.blocked)}\n')
        f.write(str(bool(self.agent)) + '\n')
        if self.agent:
            self.agent.write(f)
        f.write(str(len(self.ghosts)))
        if len(self.ghosts):
            for _, ghost in self.ghosts.items():
                ghost.write(f)

    def read(self, f):
        self.blocked = bool(f.readline().rstrip())
        if bool(f.readline().rstrip()):
            self.agent.read(f)
        if int(f.readline().rstrip()):
            for _, ghost in self.ghosts.items():
                ghost.read(f)

    def is_blocked(self):
        return self.blocked

    def is_agent_existed(self):
        return self.agent is not None

    def is_ghost_existed(self):
        return len(self.ghosts) != 0

    def insert_agent(self, agent):
        if not self.agent:
            self.agent = agent

    def is_collision(self):
        return self.is_agent_existed() and self.is_ghost_existed()

    def insert_ghost(self, ghost):
        if ghost.name not in self.ghosts:
            self.ghosts[ghost.name] = ghost

    def remove_agent(self):
        self.agent = None

    def remove_ghost(self, ghost):
        if ghost.name in self.ghosts:
            self.ghosts.pop(ghost.name)
