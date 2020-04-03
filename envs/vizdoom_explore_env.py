import gym
import numpy as np
from gym.spaces.dict_space import OrderedDict
import cv2
from vizdoom import *
from collections import deque
from shapely.geometry import Point, Polygon
from mazeexplorer import MazeExplorer
import os

import time
def print_time(log, start):
    print(log, time.time() - start)
    return time.time()
class ExplorerMultiWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        env.action_space = env.action_spaces[0]
        env.observation_space = env.observation_spaces[0]
        env.reward_range = [0., 5.0]
        env.metadata = {'render.modes': ['rgb_array']}
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        ob = self.env.reset()
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return ob, reward, done, info

class MazeExplorerEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, cfg, seed=0, **kwargs):
        super(MazeExplorerEnv,self).__init__(**kwargs)
        self._max_step = cfg.training.max_step
        if not os.path.exists('maps') : os.mkdir('maps')
        if not os.path.exists(os.path.join('maps',cfg.saving.version)): os.mkdir(os.path.join('maps',cfg.saving.version))

        self.env_cfg = cfg.task.explorer
        map_size = self.env_cfg.map_size
        self.env = MazeExplorer(unique_maps=True, number_maps=self.env_cfg.num_maps, keys=self.env_cfg.num_keys,
                                size=(map_size,map_size), random_spawn=True, random_textures=True, random_key_positions=True,
                                action_frame_repeat=4, actions="MOVE_FORWARD TURN_LEFT TURN_RIGHT MOVE_LEFT MOVE_RIGHT",
                                scaled_resolution=(64, 64), data_augmentation=True, seed=seed, episode_timeout=self._max_step*4,
                                complexity=.3, density=.3)


        self.game = self.env.env
        self.game.close()
        self.game.set_automap_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE)
        self.game.init()
        #self.game.add_game_args("+am_followplayer 1")
        self.ACTION_LIST = np.eye(5).astype(np.bool)
        self.action_space = gym.spaces.Discrete(len(self.ACTION_LIST))
        self.action_dim = self.action_space.n
        self.observation_space = OrderedDict({'image': gym.spaces.Box(0, 255, (4, 64, 64), dtype = np.uint8),
                                              'pose': gym.spaces.Box(-np.Inf, np.Inf, (4,), dtype=np.float32),
                                              'prev_action': gym.spaces.Box(0, 1, (self.action_dim,), dtype=np.float32),
                                              'episode': gym.spaces.Box(0,1,(1,),dtype=np.float32)})
        self._last_observation = None
        self._last_action = None
        self.time_t = -1
        self.episode_id = -1
        self.prev_pose = 0
        self.stuck_flag = 0
        self.success_num = 0
        self.total_reward = 0.0

    def step(self, action):
        if isinstance(action, dict): action = action['action']
        rgb, reward, done, info = self.env.step(action)
        if reward > 0.5 : self.success_num += 1
        self.time_t += 1
        if self.time_t >= self._max_step - 1: done = True
        state = self.env.env.get_state()
        obs = None if done else state
        self._last_observation = obs if obs is not None else self._last_observation
        image = np.concatenate([self.process_image(self._last_observation.screen_buffer),
                                self.process_image(self._last_observation.depth_buffer)],2)
        agent_pose = self._last_observation.game_variables
        pose_x, pose_y = agent_pose[0]/2000, agent_pose[1]/2000
        pose_yaw = agent_pose[-1]/360
        #print(agent_pose, self.stuck_flag, self.time_t,self.game.is_episode_finished())k
        self.total_reward += reward
        if self.prev_pose is not None:
            progress = np.sqrt(((pose_x - self.prev_pose[0])**2 + (pose_y - self.prev_pose[1])**2))
        else: progress = 0.0
        #print(pose_x, pose_y, pose_yaw, progress)
        if progress < 0.01:
            self.stuck_flag += 1
        else: 
            self.stuck_flag = 0 
        if self.stuck_flag > 20 :
            done = True
            self.stuck_flag = 0.0
        self.prev_pose = [pose_x, pose_y]
        self._last_action = action
        obs = {'image': image.transpose(2,1,0), 'pose': np.array([pose_x, pose_y, pose_yaw, self.time_t+1]), 'prev_action': np.eye(self.action_dim)[self._last_action]}
        # for debug
        obs['episode'] = self.episode_id * 6 + self.env.seed
        return obs, reward, done, {'episode_id': self.episode_id, 'step_id':self.time_t, 'success': self.success_num}

    def process_image(self, image, resize=True, ch3=False):
        if len(image.shape) > 2:
            image = image.transpose(1, 2, 0)
        if resize :
            image = cv2.resize(image, dsize=(64,64))
        if len(image.shape) == 2 :
            image = np.expand_dims(image, 2)
        if image.shape[2] == 1 and ch3:
            image = np.concatenate([image]*3,2)
        return image


    def reset(self):
        _ = self.env.reset()
        state = self.game.get_state()
        self._last_observation = state
        self.time_t = -1
        image = np.concatenate([self.process_image(state.screen_buffer), self.process_image(state.depth_buffer)],2)
        agent_pose = state.game_variables[[0,1,-1]]
        pose_x, pose_y = agent_pose[0]/2000, agent_pose[1]/2000
        pose_yaw = agent_pose[-1]/360
        self._last_action = None
        obs = {'image': image.transpose(2,1,0), 'pose': np.array([pose_x, pose_y, pose_yaw, self.time_t + 1]), 'prev_action': np.zeros(self.action_dim)}
        self.episode_id += 1
        self.prev_pose = None
        self.stuck_flag = 0
        self.total_reward = 0.0
        self.success_num =  0
        obs['episode'] = self.episode_id * 6 + self.env.seed
        return obs

    def set_seed(self, seed = None):
        self.seed = seed
        self.env.seed(seed)

    def close(self):
        self.env.close()

    def render(self, mode='rgb_array', close=False):
        state = self._last_observation
        if mode == 'rgb_array':
            obs = self.process_image(self.env._rgb_array, resize=False)
            map = state.automap_buffer.transpose(1,2,0)
            view_img = np.concatenate([obs[:,:,:3], map],1).astype(np.uint8)
            view_img = np.ascontiguousarray(view_img)
            cv2.putText(view_img, 'reward: %.3f'%(self.total_reward), (200, 80), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
            #view_img = cv2.resize(view_img, dsize=None, fx=2.0, fy=2.0)
            return view_img
        elif mode == 'human':
            obs = self.process_image(state.screen_buffer, resize=False)
            map = state.automap_buffer.transpose(1,2,0)
            view_img = np.concatenate([obs[:,:,:3],map],1)
            view_img = np.ascontiguousarray(view_img)
            cv2.putText(view_img, 'reward: %.3f'%(self.total_reward), (400, 150), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
            discovered = str(np.where(self.sectors)[0].tolist())
            cv2.putText(view_img, discovered, (400, 170), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
            return view_img
            #cv2.imshow('render', view_img[:,:,[2,1,0]])
            #cv2.waitKey(0)
            #pop up a window and render
        else:
            super(MazeExplorerEnv, self).render(mode=mode) # just raise an exception

if __name__== '__main__':
    run_mode = 'play'
    from configs.default_cfg import get_config
    if run_mode == 'auto':
        env = MazeExplorerEnv(get_config())
        env.reset()
        for i in range(100):
            obs, *_= env.step(env.action_space.sample())
            env.render('human')
    elif run_mode == 'play':
        env = MazeExplorerEnv(get_config())
        env.reset()
        for i in range(1000):
            im = env.render('rgb_array')
            cv2.imshow('render', im[:,:,[2,1,0]])
            key = cv2.waitKey(0)
            if key == ord('w'): action = 0
            elif key == ord('j'): action = 1
            elif key == ord('k'): action = 2
            elif key == ord('a'): action = 3
            elif key == ord('d'): action = 4
            elif key == ord('q'): break
            obs, reward, done, _ = env.step(action)
            print(reward, obs['pose'])
            if done:
                break


# MOVE_FORWARD TURN_LEFT TURN_RIGHT MOVE_LEFT MOVE_RIGHT
