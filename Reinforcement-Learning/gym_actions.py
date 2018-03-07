# Create a simple game policy
# If the Pole falls to the right, move the cart to the right and vice versa

import gym

env = gym.make('CartPole-v0')

# For Actions, we have two discrete numbers available to us "0" or "1".
print(env.action_space)

# We have 4 observations for this environment
# Cart Position, Cart Velocity, Pole Angle and Pole Angular Velocity
print(env.observation_space)

observation = env.reset()

for _ in range(1000):
    env.render()

    # Current Observation
    cart_pos, cart_vel, pole_angle, angle_vel = observation

    # Leans towards Right if Pole Bends Towards Right
    if (pole_angle > 0):
        action = 1

    # Lean Towards Left if Pole Bends Towards Left
    else:
        action = 0

    # Update the Environment with the Action Performed
    observation, reward, done, info = env.step(action)
