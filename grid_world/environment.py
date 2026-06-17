from abc import ABCMeta, abstractmethod

from grid_world.enum import Action

class Environment(metaclass=ABCMeta):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, at: Action) -> tuple[tuple, float, bool]:
        pass
