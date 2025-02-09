import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import deepmind_lab
from gym import register, make
from gym.spaces.dict_space import OrderedDict
import cv2
import torch
from gym import ObservationWrapper
from collections import deque
LEVELS = ['lt_chasm', 'lt_hallway_slope', 'lt_horseshoe_color', 'lt_space_bounce_hard', \
          'nav_maze_random_goal_01', 'nav_maze_random_goal_02', 'nav_maze_random_goal_03', 'nav_maze_static_01', \
          'nav_maze_static_02', 'seekavoid_arena_01', 'stairway_to_melon']
import time
def print_time(log, start):
    print(log, time.time() - start)
    return time.time()
class DeepLabMultiWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        env.action_space = env.action_spaces[0]
        env.observation_space = env.observation_spaces[0]
        env.reward_range = [0., 10.0]
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

class DeepmindLabEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, cfg, scene='nav_maze_random_goal_01', colors = 'RGBD_INTERLEAVED', width = 64, height = 64, max_step = 512, **kwargs):
        super(DeepmindLabEnv, self).__init__(**kwargs)

        #if not scene in LEVELS:
        #    raise Exception('Scene %s not supported' % (scene))
        scene = cfg.task.deeplab_scene
        self._colors = colors
        self._lab = deepmind_lab.Lab(scene, [self._colors, 'DEBUG.POS.TRANS', 'DEBUG.POS.ROT', 'DEBUG.CAMERA_INTERLEAVED.TOP_DOWN'],
                                     dict(fps = str(60), width = str(width), height = str(height)))

        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        self.action_dim = self.action_space.n
        self.observation_space = OrderedDict({'image': gym.spaces.Box(0, 255, (4, height, width), dtype = np.uint8),
                                              'pose': gym.spaces.Box(-np.Inf, np.Inf, (4,), dtype=np.float32),
                                              'prev_action': gym.spaces.Box(0, 1, (self.action_dim,), dtype=np.float32)})

        self._last_observation = None
        self._last_action = None
        self._max_step = cfg.training.max_step
        self.time_t = -1
        self.episode_id = -1
        self.prev_pose = 0
        self.stuck_flag = 0
        self.success = 0
        self.total_reward = 0
        self.progress = 0.0

    def step(self, action):
        if isinstance(action, dict): action = action['action']
        reward = self._lab.step(ACTION_LIST[action], num_steps=4) * 0.1
        self.time_t += 1
        done = not self._lab.is_running()
        self.total_reward += reward
        if reward >= 1.0: 
             self.success = 1.0
             done = True
        if self.time_t >= self._max_step - 1: done = True
        #print(self.time_t, self._max_step)
        obs = None if done else self._lab.observations()
        self._last_observation = obs if obs is not None else self._last_observation
        image = self._last_observation[self._colors]
        pose_x, pose_y = self._last_observation['DEBUG.POS.TRANS'][0:2] / 400
        pose_yaw = self._last_observation['DEBUG.POS.ROT'][1]/180. * np.pi

        if self.prev_pose is not None:
            self.progress = np.sqrt(((pose_x - self.prev_pose[0])**2 + (pose_y - self.prev_pose[1])**2))
        else: self.progress = 0.0
        if self.progress < 0.01 :
            self.stuck_flag += 1
        else: self.stuck_flag = 0 
        if self.stuck_flag > 40 :
            done = True
            self.stuck_flag = 0.0
        self.prev_pose = [pose_x, pose_y]
        self._last_action = action
        obs = {'image': image.transpose(2,1,0), 'pose': np.array([pose_x, pose_y, pose_yaw, self.time_t+1]), 'prev_action': np.eye(self.action_dim)[self._last_action]}
        return obs, reward, done, {'episode_id': self.episode_id, 'step_id':self.time_t, 'success': self.success}


    def reset(self):
        self._lab.reset()
        self._last_observation = self._lab.observations()
        self.time_t = -1
        image = self._last_observation[self._colors]
        pose_x, pose_y = self._last_observation['DEBUG.POS.TRANS'][0:2] / 400
        pose_yaw = self._last_observation['DEBUG.POS.ROT'][1]/180. * np.pi
        #print(self._lab.observations()['DEBUG.POS.TRANS'],self._lab.observations()['DEBUG.POS.ROT'])
        #if done : print('------------------------------done!')
        self._last_action = None
        obs = {'image': image.transpose(2,1,0), 'pose': np.array([pose_x, pose_y, pose_yaw, self.time_t+1]), 'prev_action': np.zeros(self.action_dim)}
        self.episode_id += 1
        self.prev_pose = None
        self.stuck_flag = 0
        self.success = 0
        self.total_reward = 0
        self.progress = 0.0
        return obs

    def seed(self, seed = None):
        self.seed = seed
        self._lab.reset(seed=seed)

    def close(self):
        self._lab.close()

    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            obs = self._lab.observations()[self._colors]
            map = self._lab.observations()['DEBUG.CAMERA_INTERLEAVED.TOP_DOWN']
            view_img = np.concatenate([obs[:,:,:3],map],1)
            view_img = np.ascontiguousarray(view_img)
            view_img = cv2.resize(view_img, dsize=None, fx=2.0, fy=2.0)
            #print(view_img.shape)
            cv2.putText(view_img, 'step %d reward: %.2f'%(self.time_t, self.total_reward), (140, 110), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
            cv2.putText(view_img, 'progress %.3f'%(self.progress), (140,120), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 1)
            return view_img
        elif mode == 'human':
            obs = self._lab.observations()[self._colors]
            map = self._lab.observations()['DEBUG.CAMERA_INTERLEAVED.TOP_DOWN']
            view_img = np.concatenate([obs[:,:,:3],map],1)
            view_img = np.ascontiguousarray(view_img)
            view_img = cv2.resize(view_img, dsize=None, fx=2.0, fy=2.0)
            cv2.putText(view_img, 'step %d reward: %.2f' % (self.time_t, self.total_reward), (140, 110),
                        cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
            return view_img
            #cv2.imshow('render', view_img[:,:,[2,1,0]])
            #cv2.waitKey(0)
           #pop up a window and render
        else:
            super(DeepmindLabEnv, self).render(mode=mode) # just raise an exception

def _action(*entries):
  return np.array(entries, dtype=np.intc)

ACTION_LIST = [
    _action(-20,   0,  0,  0, 0, 0, 0), # look_left 0
    _action( 20,   0,  0,  0, 0, 0, 0), # look_right 1
    #_action(  0,  10,  0,  0, 0, 0, 0), # look_up
    #_action(  0, -10,  0,  0, 0, 0, 0), # look_down
    #_action(  0,   0, -1,  0, 0, 0, 0), # strafe_left 2
    #_action(  0,   0,  1,  0, 0, 0, 0), # strafe_right 3
    _action(  0,   0,  0,  1, 0, 0, 0), # forward 4
    #_action(  0,   0,  0, -1, 0, 0, 0), # backward 5
    #_action(  0,   0,  0,  0, 1, 0, 0), # fire
    #_action(  0,   0,  0,  0, 0, 1, 0), # jump
    #_action(  0,   0,  0,  0, 0, 0, 1)  # crouch
]

if __name__== '__main__':
    from configs.default_cfg import get_config
    run_mode = 'play'
    if run_mode == 'auto':
        env = DeepmindLabEnv('nav_maze_static_01')
        env.reset()
        for i in range(100):
            obs, *_= env.step(env.action_space.sample())
            env.render('human')
    elif run_mode == 'play':
        env = DeepmindLabEnv(get_config(),'nav_maze_static_01', width=256, height=256)
        env.reset()
        for i in range(10002):
            im = env.render('rgb_array')
            cv2.imshow('render', im[:,:,[2,1,0]])
            key = cv2.waitKey(0)
            if key == ord('a') or key == ord('2'): action = 2
            elif key == ord('d') or key == ord('3'): action = 3
            elif key == ord('w') or key == ord('4'): action = 4
            elif key == ord('s'): action = 5
            elif key == ord('j') or key == ord('0'): action = 0
            elif key == ord('l') or key == ord('1'): action = 1
            elif key == ord('q'): break
            obs, reward, done, _ = env.step(action)
            print(i, reward)
            #if done: break

