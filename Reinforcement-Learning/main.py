# Cart Pole Gym Environment

# Inputs: Numpy Array with 4 Values:
# 1. Initial Position of the Cart
# 2. Angular Velocity of the Pole
# 3. Angle of Pole w.r.t Vertial Position
# 4. Velocity of Cart

# Actions Avaliable: 0 or 1

# Import Dependencies
import gym

# Create Environment
env = gym.make('CartPole-v0')

# Reset Env. To Initial State
env.reset()

# Render Env. for Several Time Steps
for t in range(1000):
    env.render()
    # Take some random action from actions available
    env.step(env.action_space.sample())