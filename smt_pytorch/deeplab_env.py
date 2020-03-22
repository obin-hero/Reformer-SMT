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

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

import time
def print_time(log, start):
    print(log, time.time() - start)
    return time.time()
class TensorStackWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob['image'])
        obs_image_numpy = np.array(self._get_ob())
        image_tensor = torch.from_numpy(obs_image_numpy).transpose(2,0).unsqueeze(0)
        image_tensor = image_tensor.float() / 255.0 * 2 - 1
        pose_tensor = torch.from_numpy(ob['pose']).float().unsqueeze(0)
        prev_action_tensor = torch.from_numpy(ob['prev_action']).float().unsqueeze(0)
        return {'image': image_tensor, 'pose': pose_tensor, 'prev_action': prev_action_tensor}

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob['image'])
        obs_image_numpy = np.array(self._get_ob())
        image_tensor = torch.from_numpy(obs_image_numpy).transpose(2,0).unsqueeze(0)
        image_tensor = image_tensor.float() / 255.0 * 2 - 1
        pose_tensor = torch.from_numpy(ob['pose']).unsqueeze(0).float()
        prev_action_tensor = torch.from_numpy(ob['prev_action']).float().unsqueeze(0)
        reward_tensor = torch.tensor(reward).unsqueeze(0)
        done_tensor = torch.tensor(done).unsqueeze(0)
        return_obs = {'image': image_tensor, 'pose': pose_tensor, 'prev_action': prev_action_tensor}
        return return_obs, reward_tensor, done_tensor, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))







class DeepmindLabEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, scene, colors = 'RGBD_INTERLEAVED', width = 64, height = 64, max_step = 512, **kwargs):
        super(DeepmindLabEnv, self).__init__(**kwargs)

        if not scene in LEVELS:
            raise Exception('Scene %s not supported' % (scene))

        self._colors = colors
        self._lab = deepmind_lab.Lab(scene, [self._colors, 'DEBUG.POS.TRANS', 'DEBUG.POS.ROT', 'DEBUG.CAMERA_INTERLEAVED.TOP_DOWN'],
                                     dict(fps = str(60), width = str(width), height = str(height)))

        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        self.action_dim = self.action_space.n
        self.observation_space = OrderedDict({'image': gym.spaces.Box(0, 255, (height, width, 4), dtype = np.uint8),
                                              'pose': gym.spaces.Box(-np.Inf, np.Inf, (3,), dtype=np.float32),
                                              'prev_action': gym.spaces.Box(0, 1, (self.action_dim,), dtype=np.float32)})

        self._last_observation = None
        self._last_action = None
        self._max_step = max_step
        self.time_t = 0

    def step(self, action):
        reward = self._lab.step(ACTION_LIST[action], num_steps=4)
        self.time_t += 1
        done = not self._lab.is_running()
        if self.time_t >= self._max_step :
            done = True
        obs = None if done else self._lab.observations()
        self._last_observation = obs if obs is not None else self._last_observation
        image = self._last_observation[self._colors]
        pose_x, pose_y = self._last_observation['DEBUG.POS.TRANS'][0:2] / 400
        pose_yaw = self._last_observation['DEBUG.POS.ROT'][1]/180. * np.pi
        #print(self._lab.observations()['DEBUG.POS.TRANS'],self._lab.observations()['DEBUG.POS.ROT'])
        #if done : print('------------------------------done!')
        self._last_action = action
        obs = {'image': image, 'pose': np.array([pose_x, pose_y, pose_yaw, self.time_t]), 'prev_action': np.eye(self.action_dim)[self._last_action]}
        return obs, reward, done, dict()


    def reset(self):
        self._lab.reset()
        self._last_observation = self._lab.observations()
        self.time_t = 0
        image = self._last_observation[self._colors]
        pose_x, pose_y = self._last_observation['DEBUG.POS.TRANS'][0:2] / 400
        pose_yaw = self._last_observation['DEBUG.POS.ROT'][1]/180. * np.pi
        #print(self._lab.observations()['DEBUG.POS.TRANS'],self._lab.observations()['DEBUG.POS.ROT'])
        #if done : print('------------------------------done!')
        self._last_action = None
        obs = {'image': image, 'pose': np.array([pose_x, pose_y, pose_yaw, self.time_t]), 'prev_action': np.zeros(self.action_dim)}
        return obs

    def seed(self, seed = None):
        self._lab.reset(seed=seed)

    def close(self):
        self._lab.close()

    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            obs = self._lab.observations()[self._colors]
            map = self._lab.observations()['DEBUG.CAMERA_INTERLEAVED.TOP_DOWN']
            view_img = np.concatenate([obs[:,:,:3],map],1)
            return view_img
        elif mode is 'human':
            obs = self._lab.observations()[self._colors]
            map = self._lab.observations()['DEBUG.CAMERA_INTERLEAVED.TOP_DOWN']
            view_img = np.concatenate([obs[:,:,:3],map],1)
            view_img = cv2.resize(view_img, dsize=None, fx=2.0, fy=2.0)
            cv2.imshow('render', view_img[:,:,[2,1,0]])
            cv2.waitKey(0)
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
    _action(  0,   0, -1,  0, 0, 0, 0), # strafe_left 2
    _action(  0,   0,  1,  0, 0, 0, 0), # strafe_right 3
    _action(  0,   0,  0,  1, 0, 0, 0), # forward 4
    _action(  0,   0,  0, -1, 0, 0, 0), # backward 5
    #_action(  0,   0,  0,  0, 1, 0, 0), # fire
    #_action(  0,   0,  0,  0, 0, 1, 0), # jump
    #_action(  0,   0,  0,  0, 0, 0, 1)  # crouch
]

if __name__== '__main__':
    run_mode = 'play'
    if run_mode == 'auto':
        env = DeepmindLabEnv('nav_maze_static_01')
        env.reset()
        for i in range(100):
            obs, *_= env.step(env.action_space.sample())
            env.render('human')
    elif run_mode == 'play':
        env = DeepmindLabEnv('nav_maze_static_01', width=256, height=256)
        env.reset()
        for i in range(1000):
            im = env.render('rgb_array')
            cv2.imshow('render', im[:,:,[2,1,0]])
            key = cv2.waitKey(0)
            if key == ord('a'): action = 2
            elif key == ord('d'): action = 3
            elif key == ord('w'): action = 4
            elif key == ord('s'): action = 5
            elif key == ord('j'): action = 0
            elif key == ord('l'): action = 1
            elif key == ord('q'): break
            obs, reward, done, _ = env.step(action)
            print(reward)

