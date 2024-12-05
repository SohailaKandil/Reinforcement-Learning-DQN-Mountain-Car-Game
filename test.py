import gym
import numpy as np

# Create two environments: one for rendering and one without rendering
render_env = gym.make("MountainCar-v0", render_mode="human")
no_render_env = gym.make("MountainCar-v0")

learning_rate = 0.1
discount_factor = 0.95
episodes = 25000
show_every = 1000

os_size = [20] * len(render_env.observation_space.high)
partitions = (render_env.observation_space.high - render_env.observation_space.low) / os_size
q_table = np.random.uniform(low=-2, high=0, size=os_size + [render_env.action_space.n])

def discr_obs(state, env):
    discrete_state = (state - env.observation_space.low) / partitions
    return tuple(discrete_state.astype(np.int_))

for episode in range(episodes):
    # Use the render environment every 'show_every' episodes, otherwise use the no-render environment
    if episode % show_every == 0:
        print(f"Episode: {episode} (with rendering)")
        env = render_env
    else:
        env = no_render_env

    # Reset the selected environment and get the initial discrete state
    discrete_state = discr_obs(env.reset(seed=123)[0], env)
    done = False

    while not done:
        # Choose action based on Q-table (exploitation)
        action = np.argmax(q_table[discrete_state])
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the environment if using the render_env
        if episode % show_every == 0:
            env.render()

        # Discretize the new state
        new_discr_state = discr_obs(new_state, env)

        if not done:
            # Q-learning update
            next_reward = np.max(q_table[new_discr_state])
            q_table[discrete_state + (action,)] = (1 - learning_rate) * q_table[discrete_state + (action,)] + learning_rate * (reward + (discount_factor * next_reward))

        elif new_state[0] >= env.goal_position:
            # If the car reaches the goal, set the Q-value to 0
            q_table[discrete_state + (action,)] = 0

        # Move to the new state
        discrete_state = new_discr_state

# Close both environments after all episodes
render_env.close()
no_render_env.close()
