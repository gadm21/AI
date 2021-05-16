import numpy as np
import gym

env = gym.make("MountainCar-v0")

epochs = 3000
discount = 0.95
lr = 0.1
show_every = 200

discrete_sizes = [20] * len(env.observation_space.high)
discrete_win_size = (env.observation_space.high - env.observation_space.low) / discrete_sizes
q_table = np.random.uniform(low = -2, high = 0, size = (discrete_sizes + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_win_size
    return tuple(discrete_state.astype(np.int))

def update_q_table(old_state, old_action, new_state, reward):
    old_q = q_table[old_state+ (old_action, )]
    max_new_q = np.max(q_table[new_state])
    new_q = (1-lr) * old_q + lr * (reward + discount * max_new_q)
    q_table[discrete_state + (old_action,)] = new_q



for epoch in range(epochs):
    done = False

    if not epoch % show_every:
        print("epoch:", epoch)
        render = True
    else: render = False
    discrete_state = get_discrete_state(env.reset())

    while not done:
        # print("discrete_state:", discrete_state)
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render : 
            env.render()

        if not done:
            update_q_table(discrete_state, action, new_discrete_state, reward)
        elif new_state[0] >= env.goal_position:
            print("reached the goal:{} with state:{} at:{}".format(env.goal_position, new_state[0], epoch))
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    env.close()



