from agent import RandomAgent
from grid_world.environment import Environment
from grid_world_enum import Action


def get_pos(at: Action):
    match at:
        case Action.UP:
            return -1, 0
        case Action.DOWN:
            return 1, 0
        case Action.LEFT:
            return 0, -1
        case Action.RIGHT:
            return 0, 1

class GridWorld(Environment):
    def __init__(self):
        self.done = False
        self.size = 5
        self.state = (0, 0)
        self.start = (0, 0)
        self.goal = (4, 4)

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, at: Action):
        a_row, a_col = get_pos(at)
        c_row, c_col = self.state
        n_row = a_row + c_row
        n_col = a_col + c_col
        hit_wall = False
        if n_row == self.size or n_row < 0:
            n_row = c_row
            hit_wall = True
        if n_col == self.size or n_col < 0:
            n_col = c_col
            hit_wall = True

        self.state = n_row, n_col

        self.done = True if self.state == self.goal else False
        c_r = 1.0 if self.done else -0.1 if hit_wall else -0.01

        return self.state, c_r, self.done


max_step = 500
env = GridWorld()
agent = RandomAgent(max_step=500)

t_r = 0.0
state = None
done = False

while agent.iterable:
    action = agent.choose_action()
    state, c_r, done = env.step(at=action)
    t_r += c_r
    if done:
        break

print(f"state : {state}, t_r: {t_r} done: {done} step: {agent.current_step}")
