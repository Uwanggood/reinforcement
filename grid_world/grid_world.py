if __package__:
    from grid_world.environment import Environment
    from grid_world.grid_world_enum import Action
    from grid_world.random_agent import RandomAgent
else:
    from environment import Environment
    from grid_world_enum import Action
    from random_agent import RandomAgent


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
        self.done = False
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


def run_episode(env: Environment, agent: RandomAgent, max_steps: int = 100) -> tuple[tuple, float, bool, int]:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    state = env.reset()
    total_reward = 0.0

    for step in range(1, max_steps + 1):
        action = agent.choose_action()
        state, reward, done = env.step(at=action)
        total_reward += reward

        if done:
            return state, total_reward, done, step

    return state, total_reward, False, max_steps


if __name__ == "__main__":
    env = GridWorld()
    agent = RandomAgent()

    state, total_reward, done, step = run_episode(env=env, agent=agent, max_steps=500)

    print(f"state : {state}, total_reward: {total_reward} done: {done} step: {step}")
