if __package__:
    from grid_world.grid_world_enum import Action
else:
    from grid_world_enum import Action


class RandomAgent:
    def choose_action(self) -> Action:
        return Action.random()
