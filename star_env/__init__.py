import gym

gym.envs.registration.register(
        id='Star-v0',
        entry_point='star_env.star_env:StarEnv',
        max_episode_steps=100)
