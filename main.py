import gym
import star_env
from baselines import deepq


env = gym.make('Star-v0')
model = deepq.models.tabular()
episodes = [0]
def callback(lcl, glb):
    #num_episodes = len(lcl['episode_rewards'])
    #if num_episodes > episodes[0] and num_episodes >= 2:
    #    episodes[0] = num_episodes
    #    print('Episode reward: %f' % lcl['episode_rewards'][-2])
    return False

act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback)
