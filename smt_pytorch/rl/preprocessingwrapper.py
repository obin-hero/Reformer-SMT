from gym.spaces.box import Box
import gym
import torch

class ProcessObservationWrapper(gym.ObservationWrapper):
    ''' Wraps an environment so that instead of
            obs = env.step(),
            obs = transform(env.step())
        
        Args:
            transform: a function that transforms obs
            obs_shape: the final obs_shape is needed to set the observation space of the env
    '''
    def __init__(self, env, transform, obs_space):
        super().__init__(env)
        self.observation_space = obs_space
        self.transform = transform
        
    def observation(self, observation):
        return self.transform(observation)

import numpy as np
class EncodeObservatonWrapper(gym.ObservationWrapper):
    def __init__(self, env, encoder, observation_space):
        super().__init__(env)
        self.observation_space = observation_space
        self.observation_space.spaces.update({'feature': Box(-np.inf, np.inf, (8,16,16))})
        self.observation_space.spaces.update({'rgb': Box(0, 255, (3, 64, 64))})
        self.encoder = encoder

    def observation(self, obs):
        with torch.no_grad():
            obs['rgb'] = torch.tensor(obs['rgb']).cuda().transpose(1,3)
            rgb_tensor = obs['rgb'].float()/255.*2 - 1
            feature = self.encoder(rgb_tensor)
        obs['feature'] = feature
        return obs