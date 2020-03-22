# This code is a heavily modified version of a similar file in Habitat
import copy
from collections import deque
import cv2
from gym import spaces
import gzip
import numpy as np
import os
from PIL import Image
import random
from time import time
import torch
import torch.nn as nn
import warnings

from habitat.config.default import get_config
from habitat.utils.visualizations import maps
#from habitat.sims.habitat_simulator import SimulatorActions
from habitat.sims.habitat_simulator.actions import HabitatSimActions as SimulatorActions
import habitat

from rl.visdommonitor import VisdomMonitor
from rl.habitat_explore_env import *

import quaternion as q

PI = 3.141592

class HabitatPreprocessVectorEnv(habitat.VectorEnv):
    def __init__(
        self,
        make_env_fn,
        env_fn_args,
        preprocessing_fn=None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
        collate_obs_before_transform: bool = False,
    ):
        super().__init__(make_env_fn, env_fn_args, auto_reset_done, multiprocessing_start_method)
        obs_space = self.observation_spaces[0]

        # Preprocessing
        self.transform = None
        for i in range(self.num_envs):
            self.observation_spaces[i] = obs_space

        self.collate_obs_before_transform = collate_obs_before_transform
        self.keys = []
        shapes, dtypes = {}, {}

        for key, box in obs_space.spaces.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]

    def reset(self):
        observation_list = super().reset()
        if self.collate_obs_before_transform:
            self._save_init_obs(observation_list)
            self._save_all_obs(self.buf_init_obs)
        else:
            for e, obs in enumerate(observation_list):
                if self.transform is not None:
                    obs = self.transform(obs)
                self._save_obs(e, obs)
        return self._obs_from_buf()

    def step(self, action):
        start = time()
        results_list = super().step(action)
        for e, result in enumerate(results_list):
            self.buf_rews[e] = result[1]
            self.buf_dones[e] = result[2]
            self.buf_infos[e] = result[3]
        if self.collate_obs_before_transform:
            self._save_init_obs([r[0] for r in results_list])
            self._save_all_obs(self.buf_init_obs)
        else:
            for e, (obs, _, _, _) in enumerate(results_list):
                if self.transform is not None:
                    obs = self.transform(obs)
                self._save_obs(e, obs)

        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy())

    def _save_init_obs(self, all_obs):
        self.buf_init_obs = {}
        for k in all_obs[0].keys():
            if k is None:
                self.buf_init_obs[k] = torch.stack([torch.Tensor(o) for o in all_obs])
            else:
                self.buf_init_obs[k] = torch.stack([torch.Tensor(o[k]) for o in all_obs])

    def _save_obs(self, e, obs):
        try:
            for k in self.keys:
                if k is None:
                    self.buf_obs[k][e] = obs
                else:
                    self.buf_obs[k][e] = obs[k]
        except Exception as e:
            print(k, e)
            raise e

    def _save_all_obs(self, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k] = obs
            else:
                self.buf_obs[k] = obs[k]

    def _obs_from_buf(self):
        if self.keys==[None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs


def draw_top_down_map(info, heading, output_size):
    if info is None:
        return
    top_down_map = maps.colorize_topdown_map(info["top_down_map"]["map"])
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def pi_clip(value):
    if value > PI : value -= PI*2
    elif value < -PI : value += PI*2
    return value


def make_habitat_vector_env(task_cfg,
                            num_processes=2,
                            preprocessing_fn=None,
                            log_dir=None,
                            visdom_name='main',
                            visdom_log_file=None,
                            visdom_server='localhost',
                            visdom_port='8097',
                            vis_interval=200,
                            scenes=None,
                            val_scenes=['Greigsville', 'Pablo', 'Mosquito'],
                            num_val_processes=0,
                            swap_building_k_episodes=10,
                            gpu_devices=[0],
                            collate_obs_before_transform=True,
                            seed=42
                           ):

    habitat_path = os.path.dirname(os.path.dirname(habitat.__file__))
    if 'Explore' in task_cfg.ENV_NAME: file_name = 'config/explore_mp3d.yaml'
    elif 'TargetNav' in task_cfg.ENV_NAME : file_name = 'config/targetnav_mp3d.yaml'
    elif 'PathFollow' in task_cfg.ENV_NAME : file_name = 'config/pathfollow_mp3d.yaml'
    #habitat_task_config = os.path.join(habitat_path, file_name)
    habitat_cfg = get_config(config_paths=file_name)
    habitat_cfg.defrost()
    habitat_cfg.SIMULATOR.SCENE = os.path.join(habitat_path, habitat_cfg.SIMULATOR.SCENE)
    habitat_cfg.DATASET.DATA_PATH = os.path.join(habitat_path, habitat_cfg.DATASET.DATA_PATH)
    habitat_cfg.DATASET.SCENES_DIR = os.path.join(habitat_path, habitat_cfg.DATASET.SCENES_DIR)
    habitat_cfg.freeze()

    habitat_val_cfg = habitat_cfg
    habitat_val_cfg.defrost()
    habitat_val_cfg.DATASET.SPLIT = "val"
    habitat_val_cfg.freeze()

    env_cfgs = []
    task_cfgs = []

    # Assign specific buildings to each process
    if 'test-scenes' in habitat_cfg.DATASET.DATA_PATH:
        training_scenes = ['*']
    elif 'mp3d' in habitat_cfg.DATASET.DATA_PATH:

        training_scenes =  ['1pXnuDYAj8r', 'ULsKaCPVFJR', '2n8kARJN3HM', 'ur6pFq6Qu1A', 'PX4nDJXEHrG']


    validation_scenes = ['*']
    #                    ['TbHJrupSAjP', 'EU6Fwq7SyZv', 'pLe4wQe7qrG', 'x8F5xyUWy9e', 'oLBMNvg9in8',
    #                     'zsNo4HB9uLZ', 'X7HyMhZNoso', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'QUCTc6BB5sX', '8194nk5LbLH']


    train_process_scenes = [[] for _ in range(num_processes)]
    for i, scene in enumerate(training_scenes):
        train_process_scenes[i % len(train_process_scenes)].append(scene)


    if num_val_processes > 0:
        val_process_scenes = [[] for _ in range(num_val_processes)]
        for i, scene in enumerate(validation_scenes):
            val_process_scenes[i % len(val_process_scenes)].append(scene)

    total_num_processes = num_processes + num_val_processes
    for i in range(total_num_processes):
        env_cfg = copy.copy(habitat_cfg)
        env_cfg.defrost()
        if i < num_processes:
            env_cfg.DATASET.SPLIT = 'train'
            env_cfg.DATASET.CONTENT_SCENES = train_process_scenes[i]
        else:
            env_cfg.DATASET.SPLIT = 'val'
            val_i = i - (num_processes)
            env_cfg.DATASET.CONTENT_SCENES = val_process_scenes[val_i]
            env_cfg.DATASET.DATA_PATH =  os.path.join(habitat_path, 'data/datasets/pointnav/mp3d/v1/val/val.json.gz')
        print("Env {}:".format(i), env_cfg.DATASET.CONTENT_SCENES)

        env_cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_devices[i % len(gpu_devices)]
        env_cfg.freeze()
        env_cfgs.append(env_cfg)
        task_cfgs.append(task_cfg)

    should_record = [(i == 0 or i == (total_num_processes - num_processes)) for i in range(total_num_processes)]

    envs = HabitatPreprocessVectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(env_cfgs,
                    task_cfgs,
                    range(total_num_processes),
                    [log_dir for _ in range(total_num_processes)],
                    [visdom_name for _ in range(total_num_processes)],
                    [visdom_log_file for _ in range(total_num_processes)],
                    [vis_interval for _ in range(total_num_processes)],
                    [visdom_server for _ in range(total_num_processes)],
                    [visdom_port for _ in range(total_num_processes)],
                    [swap_building_k_episodes for _ in range(total_num_processes)],
                    should_record,
                    [seed + i for i in range(total_num_processes)],
                   )
            )
        ),
        preprocessing_fn=preprocessing_fn,
        collate_obs_before_transform=collate_obs_before_transform
    )
    envs.observation_space = envs.observation_spaces[0]
    envs.action_space = spaces.Discrete(3)
    envs.reward_range = None
    envs.metadata = None
    envs.is_embodied = True
    return envs


flatten = lambda l: [item for sublist in l for item in sublist]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def shuffle_episodes(env, swap_every_k=10):
    episodes = env.episodes
#     buildings_for_epidodes = [e.scene_id for e in episodes]
    episodes = env.episodes = random.sample([c for c in chunks(episodes, swap_every_k)], len(episodes) // swap_every_k)
    env.episodes = flatten(episodes)
    return env.episodes

def make_env_fn(habitat_cfg,
                task_cfg,
                rank,
                log_dir,
                visdom_name,
                visdom_log_file,
                vis_interval,
                visdom_server,
                visdom_port,
                swap_building_k_episodes,
                should_record,
                seed):

    env = eval(task_cfg.ENV_NAME)(habitat_cfg, task_cfg)
    env.episodes = shuffle_episodes(env, swap_every_k=swap_building_k_episodes)
    env.seed(seed)
    if should_record and visdom_log_file is not None:
        print("SETTING VISDOM MONITOR WITH VIS INTERVAL", vis_interval)
        env = VisdomMonitor(env,
                       directory=os.path.join(log_dir, visdom_name),
                       video_callable=lambda x: x % vis_interval == 0,
                       uid=str(rank),
                       server=visdom_server,
                       port=visdom_port,
                       visdom_log_file=visdom_log_file,
                       visdom_env=visdom_name)

    return env


