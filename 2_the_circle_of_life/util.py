from copy import *
import random

INVALID = -1
DEBUG = -1


class ListDict(object):
    def __init__(self):
        self.item_to_position = {}
        self.items = []

    def __deepcopy__(self, memodict=None):
        d_copy = type(self)()
        d_copy.item_to_position = self.item_to_position
        d_copy.items = self.items
        return d_copy

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def add_item(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove_item(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        return random.choice(self.items)

    def write(self, f):
        pass

    def read(self, f):
        pass
