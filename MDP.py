import numpy as np


class MDP:
    def __init__(self, filename: str, start_pos: tuple[int, int], transition_probs: float):
        file = open(filename, 'r')
        self.states: list = [state.strip().split(",") for state in file]
        self.col: int = self.states.__len__()
        self.row: int = self.states[0].__len__()
        self.transition_probs: float = transition_probs
        self.actions: list = []


if __name__ == "__main__":
    MDP('environment.csv', (0, 0), 0.9)
