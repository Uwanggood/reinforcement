import numpy as np
import gym


def iterative_policy_evaluation(env, policy, gamma=1.0, theta=1e-9):
    # env.nS is not available in gym version 0.10.5
    num_states = env.observation_space.n
    v_function = np.zeros(num_states)
    while True:
        delta = 0
        for s in range(num_states):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * v_function[next_state])
            delta = max(delta, np.abs(v - v_function[s]))
            v_function[s] = v
        if delta < theta:
            break
    return v_function


def iterative_policy_improvement(env, policy, v_function, gamma=1.0):
    num_states = env.nS
    num_actions = env.nA
    new_policy = np.zeros([num_states, num_actions]) / num_actions
    for s in range(num_states):
        q_function = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                q_function[a] += prob * (reward + gamma * v_function[next_state])
        best_action = np.argmax(q_function)
        new_policy[s][best_action] = 1.0
    return new_policy


def policy_iteration(env, gamma=1.0, theta=1e-9):
    # 초기화
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.ones([num_states, num_actions]) / num_actions  # 모든 상태에서 균등한 확률로 행동을 선택하는 정책
    value_function = np.zeros(num_states)  # 모든 상태의 가치를 0으로 초기화

    while True:
        # 정책 평가 (policy evaluation)
        while True:
            delta = 0
            for s in range(num_states):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in env.P[s][a]:
                        v += action_prob * prob * (reward + gamma * value_function[next_state])
                delta = max(delta, np.abs(v - value_function[s]))
                value_function[s] = v
            if delta < theta:
                break

        # 정책 개선 (policy improvement)
        policy_stable = True
        for s in range(num_states):
            old_action = np.argmax(policy[s])
            action_values = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * value_function[next_state])
            best_action = np.argmax(action_values)
            if old_action != best_action:
                policy_stable = False
            policy[s] = np.eye(num_actions)[best_action]

        if policy_stable:
            break

    return policy, value_function


def value_iteration(env, gamma=1.0, theta=1e-9):
    # env.nS is not available in gym version 0.10.5
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    V_function = np.zeros(num_states)

    def one_step_lookahead(state, V):
        q_values = np.zeros(num_actions)
        for action in range(num_actions):
            for prob, next_state, reward, done in env.P[state][action]:
                q_values[action] += prob * (reward + gamma * V[next_state])
        return q_values

    while True:
        delta = 0
        for s in range(num_states):
            v = V_function[s]
            action_values = one_step_lookahead(s, V_function)
            V_function[s] = np.max(action_values)
            delta = max(delta, np.abs(v - V_function[s]))
        if delta < theta:
            break

    policy = np.zeros([num_states, num_actions])
    for s in range(num_states):
        action_values = one_step_lookahead(s, V_function)
        best_action = np.argmax(action_values)
        policy[s, best_action] = 1.0

    return V_function, policy


def main():
    env = gym.make("FrozenLake-v1")
    policy, value_function = policy_iteration(env)
    print("Policy Iteration")
    print(policy)
    print(value_function)
    policy = value_iteration(env)


if __name__ == "__main__":
    main()
