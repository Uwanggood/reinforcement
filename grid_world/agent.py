from grid_world.environment import Environment
from grid_world.grid_world_enum import Action


class RandomAgent:
    def __init__(self, max_step=100) -> None:
        if max_step < 1:
            raise ValueError("max_step은 1 이상이어야 합니다.")
        self.current_step = 0
        self.iterable = True
        self.max_step = max_step
        self.t_r = 0.0

    def choose_action(self) -> Action:
        self.current_step += 1
        if self.current_step == self.max_step:
            self.iterable = False
        return Action.random()

    def run_episode(self, env: Environment):
        while self.iterable:
            action = self.choose_action()
            state, c_r, done = env.step(at=action)
            self.t_r += c_r
            if done:
                break
