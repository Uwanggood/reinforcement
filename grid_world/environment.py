from abc import ABCMeta, abstractmethod

from grid_world.grid_world_enum import Action


class Environment(metaclass=ABCMeta):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, at: Action):
        pass
