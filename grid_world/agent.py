from abc import ABCMeta, abstractmethod

from grid_world.enum import Action


class QAgent:
    def __init__(self, size: int):
        self.q_table = (((size, size), Action), 0)
        self.epsilon = 0.1

    def update(self, before_state: tuple[int, int], action: Action, reward: float):
        self.q_table[before_state][action] = reward

    def choose_action(self, state: tuple[int, int]):
          for actions in self.q_table[state]:

