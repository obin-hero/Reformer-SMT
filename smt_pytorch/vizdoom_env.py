import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym import register, make
from gym.spaces.dict_space import OrderedDict
import cv2
import torch
from gym import ObservationWrapper
import vizdoom
from vizdoom import *
from collections import deque
from shapely.geometry import Point, Polygon

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
class VizDoomMultiWrapper(gym.Wrapper):
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

class VizDoomEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, cfg, **kwargs):
        super(VizDoomEnv,self).__init__(**kwargs)
        self.game = DoomGame()
        self.game.set_sectors_info_enabled(True)

        self.game.load_config("configs/my_way_home.cfg")
        self._max_step = cfg.training.max_step
        self.game.set_episode_timeout(self._max_step*5)
        # Episodes can be recorder in any available mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_automap_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE)
        self.game.add_game_args("+am_followplayer 1")
        self.game.add_game_args(
            "+viz_am_scale 10")  # This CVAR controls scale of rendered map (higher valuer means bigger zoom).
        self.game.add_game_args(
            "+viz_am_center 1")  # This CVAR shows the whole map centered (overrides am_followplayer and viz_am_scale).
        self.game.add_game_args(
            "+am_backcolor 000000")  # Map's colors can be changed using CVARs, full list is available here: https://zdoom.org/wiki/CVARs:Automap#am_backcolor
        self.game.init()

        self.ACTION_LIST = np.eye(5).astype(np.bool)
        self.action_space = gym.spaces.Discrete(len(self.ACTION_LIST))
        self.action_dim = self.action_space.n
        self.observation_space = OrderedDict({'image': gym.spaces.Box(0, 255, (4, 64, 64), dtype = np.uint8),
                                              'pose': gym.spaces.Box(-np.Inf, np.Inf, (4,), dtype=np.float32),
                                              'prev_action': gym.spaces.Box(0, 1, (self.action_dim,), dtype=np.float32)})
        self._last_observation = None
        self._last_action = None
        self.time_t = -1
        self.episode_id = -1
        self.prev_pose = 0
        self.stuck_flag = 0
        self.sectors = None
        self.total_reward = 0.0

    def new_room(self,state):
        if state is None : return 0.0
        sectors = state.sectors
        agent_point = Point(state.game_variables[0], state.game_variables[1])
        for s_id in range(len(sectors)):
            if self.sectors[s_id] == 1.0: continue
            if agent_point.within(self.sectors_polygon[s_id]):
                self.sectors[s_id] = 1.0
                return 0.1
        return 0.0

    def step(self, action):
        if isinstance(action, dict): action = action['action']
        reward = self.game.make_action(self.ACTION_LIST[action].tolist(),5)
        self.time_t += 1
        done = (self.game.is_episode_finished())
        #print(reward
        reward = 3.0 if reward > 0.05 else 0.0
        if self.time_t >= self._max_step - 1: done = True
        state = self.game.get_state()
        obs = None if done else state
        self._last_observation = obs if obs is not None else self._last_observation
        image = np.concatenate([self.process_image(self._last_observation.screen_buffer),
                                self.process_image(self._last_observation.depth_buffer)],2)
        agent_pose = self._last_observation.game_variables
        pose_x, pose_y = agent_pose[0]/400, agent_pose[1]/400
        pose_yaw = agent_pose[-1]/360
        #print(agent_pose, self.stuck_flag, self.time_t,self.game.is_episode_finished())
        reward += self.new_room(state)
        self.total_reward += reward
        if self.prev_pose is not None:
            progress = np.sqrt(((pose_x - self.prev_pose[0])**2 + (pose_y - self.prev_pose[1])**2))
        else: progress = 0.0
        #print(pose_x, pose_y, pose_yaw, progress)
        if progress < 0.02 :
            self.stuck_flag += 1
        else: 
            self.stuck_flag = 0 
        if self.stuck_flag > 20 :
            done = True
            self.stuck_flag = 0.0
        self.prev_pose = [pose_x, pose_y]
        self._last_action = action
        obs = {'image': image.transpose(2,1,0), 'pose': np.array([pose_x, pose_y, pose_yaw, self.time_t]), 'prev_action': np.eye(self.action_dim)[self._last_action]}
        return obs, reward, done, {'episode_id': self.episode_id, 'step_id':self.time_t}

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
        self.game.new_episode()
        state = self.game.get_state()
        self._last_observation = state
        self.time_t = -1
        image = np.concatenate([self.process_image(state.screen_buffer), self.process_image(state.depth_buffer)],2)
        agent_pose = state.game_variables
        pose_x, pose_y = agent_pose[0], agent_pose[1]
        pose_yaw = agent_pose[-1]
        self._last_action = None
        obs = {'image': image.transpose(2,1,0), 'pose': np.array([pose_x, pose_y, pose_yaw, self.time_t]), 'prev_action': np.zeros(self.action_dim)}
        self.episode_id += 1
        self.prev_pose = None
        self.stuck_flag = 0
        self.sectors = np.zeros(len(state.sectors))
        self.sectors_polygon = self.build_polygon(state.sectors)
        self.total_reward = 0.0
        return obs

    def build_polygon(self, sectors):
        sectors_polygon = []
        for s in sectors:
            coords = []
            for line in s.lines:
                coords.append((line.x1, line.y1))
                coords.append((line.x2, line.y2))
            poly = Polygon(coords)
            sectors_polygon.append(poly)
        return sectors_polygon

    def seed(self, seed = None):
        self.seed = seed
        self.game.set_seed(seed)

    def close(self):
        self.game.close()

    def render(self, mode='rgb_array', close=False):
        state = self._last_observation
        if mode == 'rgb_array':
            obs = self.process_image(state.screen_buffer, resize=False)
            map = state.automap_buffer.transpose(1,2,0)
            view_img = np.concatenate([obs[:,:,:3],map],1).astype(np.uint8)
            view_img = np.ascontiguousarray(view_img)
            discovered = str(np.where(self.sectors)[0].tolist())
            cv2.putText(view_img, 'reward: %.3f'%(self.total_reward), (400, 150), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
            cv2.putText(view_img, discovered, (400, 170), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
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
            super(VizDoomEnv, self).render(mode=mode) # just raise an exception

if __name__== '__main__':
    run_mode = 'play'
    from default_cfg import get_config
    if run_mode == 'auto':
        env = VizDoomEnv(None)
        env.reset()
        for i in range(100):
            obs, *_= env.step(env.action_space.sample())
            env.render('human')
    elif run_mode == 'play':
        env = VizDoomEnv(get_config())
        env.reset()
        for i in range(1000):
            im = env.render('rgb_array')
            cv2.imshow('render', im[:,:,[2,1,0]])
            key = cv2.waitKey(0)
            if key == ord('a'): action = 0
            elif key == ord('d'): action = 1
            elif key == ord('w'): action = 2
            elif key == ord('q'): break
            obs, reward, done, _ = env.step(action)
            print(reward)
            if done:
                break


