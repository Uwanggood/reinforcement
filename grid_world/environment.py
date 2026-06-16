from abc import ABCMeta, abstractmethod

if __package__:
    from grid_world.grid_world_enum import Action
else:
    from grid_world_enum import Action


class Environment(metaclass=ABCMeta):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, at: Action) -> tuple[tuple, float, bool]:
        pass
