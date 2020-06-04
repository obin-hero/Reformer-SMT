#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import envs.habitat_utils
import habitat
from habitat import Config, Dataset
import os
from gym.spaces.dict_space import OrderedDict
import gym
import numpy as np
from guppy import hpy
h = hpy()
class ObjectNavENV(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.task.habitat.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        self.pid = os.getpid()
        super().__init__(self._core_env_config, dataset)

        self.action_dim = config.action_dim
        height = width = config.img_size
        self.observation_space = OrderedDict({'image': gym.spaces.Box(0, 255, (4, height, width), dtype=np.uint8),
                                              'pose': gym.spaces.Box(-np.Inf, np.Inf, (4,), dtype=np.float32),
                                              'prev_action': gym.spaces.Box(0, 1, (self.action_dim,),
                                                                            dtype=np.float32)})


    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]

        return observations

    def step(self, *args):
        #self._previous_action = kwargs["action"]
        obs, reward, done, info = super().step(*args)

        return obs, reward, done, info


    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


if __name__ == '__main__':
    from envs.habitat_objectnav_env import ObjectNavENV
    from habitat.config.default import get_config as get_habitat_config
    from configs.default_cfg import get_config
    import habitat
    from habitat import make_dataset
    import numpy as np
    import envs.habitat_utils
    import time
    import psutil

    pid = os.getpid()
    current_process = psutil.Process(pid)

    habitat_cfg = get_habitat_config('configs/objectnav_mp3d.yaml')
    habitat_cfg.defrost()
    habitat_cfg.TASK.CUSTOM_OBJECT_GOAL_SENSOR = habitat.Config()
    habitat_cfg.TASK.CUSTOM_OBJECT_GOAL_SENSOR.TYPE = 'CustomObjectSensor'
    habitat_cfg.TASK.CUSTOM_OBJECT_GOAL_SENSOR.GOAL_SPEC = "OBJECT_IMG"
    habitat_cfg.freeze()
    cfg = get_config()
    cfg.defrost()
    cfg.task.habitat.TASK_CONFIG = habitat_cfg
    cfg.freeze()

    training_scenes = ['PX4nDJXEHrG', '5q7pvUzZiYa', 'S9hNv5qa7GM', 'ac26ZMwG7aT', '29hnd4uzFmX', '82sE5b5pLXE',
                       'p5wJjkQkbXX', 'B6ByNegPMKs', '17DRP5sb8fy', 'pRbA3pwrgk9', 'gZ6f7yhEvPG', 'HxpKQynjfin',
                       'ZMojNkEp431', '5LpN3gDmAk7', 'dhjEzFoUFzH', 'vyrNrziPKCB', 'sKLMLpTHeUy', '759xd9YjKW5',
                       'sT4fr6TAbpF', '1pXnuDYAj8r', 'E9uDoFAP3SH', 'GdvgFV5R1Z5', 'rPc6DW4iMge', 'D7N2EKCX4Sj',
                       'uNb9QFRL6hY', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 's8pcmisQ38h', 'EDJbREhghzL', 'YmJkqBEsHnH',
                       'XcA2TqTSSAj', '7y3sRwLe3Va', 'e9zR4mvMWw7', 'JeFG25nYj2p', 'VLzqgDo317F', 'kEZ7cmS4wCh',
                       'r1Q1Z4BcV1o', 'qoiz87JEwZ2', '1LXtFkjw3qL', 'VFuaQ6m2Qom', 'b8cTxDM8gDG', 'ur6pFq6Qu1A',
                       'V2XKFyX4ASd', 'Uxmj2M2itWa', 'Pm6F8kyY3z2', 'PuKPg4mmafe', '8WUmhLawc2A', 'ULsKaCPVFJR',
                       'r47D5H71a5s', 'jh4fc5c5qoQ', 'JF19kD82Mey', 'D7G3Y4RVNrH', 'cV4RVeZvu5T', 'mJXqzFtmKg4',
                       'i5noydFURQK', 'aayBHfsNo7d']

    def filter_fn(episode):
        if episode.info['geodesic_distance'] > 3.0:
            return False
        else:
            return True

    dataset = make_dataset(
        habitat_cfg.DATASET.TYPE, config=habitat_cfg.DATASET, **{'filter_fn': filter_fn}
    )
    env = ObjectNavENV(cfg, dataset=dataset)
    obs = env.reset()
    prev_time = time.time()
    for i in range(100000000):
        obs, reward, done, info = env.step(env.action_space.sample())
        if i % 1000 == 0 :
            memKB = current_process.memory_info()[0] / 2. ** 20
            print('step %d episode scene %s id %s time : %.4f, mem usage %9f'%(i, env.current_episode.scene_id.split('/')[-2], env.current_episode.episode_id, time.time() - prev_time, memKB))
            prev_time = time.time()
            #print(h.heap()[0:5])

        if done:
            obs = env.reset()
