import numpy as np

# MDP 환경에 대한 정보
num_states = 4
num_actions = 2
transition_prob = np.array([[[0.7, 0.3], [0.3, 0.7]],
                            [[0.6, 0.4], [0.8, 0.2]],
                            [[0.3, 0.7], [0.9, 0.1]],
                            [[1.0, 0.0], [1.0, 0.0]]])
reward = np.array([[10, 0], [0, 5], [4, 4], [0, 0]])

# 벨만 최적 정책 알고리즘
gamma = 0.9
num_iterations = 100
q_function = np.zeros((num_states, num_actions))
for i in range(num_iterations):
    v_function = np.max(q_function, axis=1)
    v_function = v_function.reshape((num_states, 1))  # v_function shape을 (4, 1)로 변경
    for s in range(num_states):
        for a in range(num_actions):
            q_function[s][a] = np.sum(transition_prob[s][a] * (reward[s][a] + gamma * v_function))

optimal_policy = np.argmax(q_function, axis=1)

print("Bellman optimal policy:", optimal_policy)
