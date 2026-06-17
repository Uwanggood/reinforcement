import random
from enum import IntEnum

size =5

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @classmethod
    def random(cls) -> "Action":
        return random.choice(list(cls))

q_table =[[[0.0 for i in range(4)] for _ in range(size)] for _ in range(size)]
print(q_table)

