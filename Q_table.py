import gym
import numpy as np
import random

render_env = gym.make("MountainCar-v0", render_mode="human")
no_render_env = gym.make("MountainCar-v0")
env = render_env

learning_rate = 0.1
discount_factor = 0.95
episodes = 25_000
show_every = 1000
epsilon = 0.1

os_size = [20] * len(env.observation_space.high)
partitions = (env.observation_space.high - env.observation_space.low) / os_size
q_table = np.random.uniform(low = -2 , high = 0 , size =  os_size + [env.action_space.n])

def discr_obs (state):
    discrete_state = (state - env.observation_space.low) / partitions
    return (tuple(discrete_state.astype(np.int_)))


# while True:
#     action = random.choice([0,1,2])
#     new_state, reward, terminated, truncated, info = env.step(action)

for episode in range (episodes):
    if episode % show_every == 0:
        print(f"Episode: {episode}")
        env = render_env 
    else:
        env = no_render_env 

    discrete_state = discr_obs(env.reset(seed=123)[0])
    done = False
    while not done:
        randomize = random.randint (0,1)
        if randomize >= epsilon:
            action = random.choice([0,1,2])
            if (epsilon < 1):
                epsilon *= 1.2
        else:
            action = np.argmax(q_table[discrete_state]) 
            if (epsilon < 1): 
                epsilon *= 1.2

        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        """
        if render:
            env.render()
        """
        new_discr_state = discr_obs(new_state)
        if not done :
            next_reward = np.max(q_table[new_discr_state])

            q_table[discrete_state + (action,)] = (1 - learning_rate) * q_table[discrete_state + (action,)] + learning_rate * (reward + (discount_factor * next_reward))

            

        elif new_state[0] >= env.goal_position :
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discr_state
env.close()



