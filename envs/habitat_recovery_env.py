import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import habitat_sim
from gym import register, make
from gym.spaces.dict_space import OrderedDict
import cv2
import torch
from gym import ObservationWrapper
from collections import deque


import time
def print_time(log, start):
    print(log, time.time() - start)
    return time.time()
from envs.habitat_utils.habitat_runner import DemoRunner
import quaternion as q

class HabitatMultiWrapper(gym.Wrapper):
    def __init__(self, cfg, env, k=4):
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


class HabitatRecoveryEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, cfg,**kwargs):
        super(HabitatRecoveryEnv, self).__init__(**kwargs)

        self.runner = DemoRunner()
        self.action_num = len(self.runner._cfg.agents[self.runner._sim_settings["default_agent"]].action_space)
        self.action_space = gym.spaces.Discrete(self.action_num)
        self.action_dim = self.action_space.n
        self.img_size = cfg.img_size
        self.observation_space = OrderedDict({'image': gym.spaces.Box(0, 255, (3, self.img_size, self.img_size), dtype = np.uint8),
                                              'pose': gym.spaces.Box(-np.Inf, np.Inf, (4,), dtype=np.float32),
                                              'prev_action': gym.spaces.Box(0, 1, (self.action_dim,), dtype=np.float32),
                                              'demo_im': gym.spaces.Box(0, 255, (30, 3, self.img_size, self.img_size), dtype = np.uint8),
                                              'demo_pose': gym.spaces.Box(-np.Inf, np.Inf, (30, 7)),
                                              'demo_act': gym.spaces.Box(-np.Inf, np.Inf, (30, 3)),
                                               })

        self._last_observation = None
        self._last_action = None
        self._max_step = cfg.training.max_step
        self.time_t = -1
        self.episode_id = -1
        self.prev_pose = 0
        self.stuck_flag = 0
        self.success = 0
        self.total_reward = 0

    def process_img(self, img):
        # img H, W, C
        resized_img = cv2.resize(img, dsize=(self.img_size, self.img_size))
        transposed = resized_img.transpose(2,0,1)
        return transposed

    def check_progress(self, curr_position):
        not_reached = np.where(self.progress_checker == False)[0]
        for i in not_reached[::-1]:
            if self.progress_checker[i]: continue
            else:
                dist = self.runner.geodesic_distance(curr_position, self.demo_pose[i][:3])
                if dist < 0.3 :
                    self.progress_checker[i:] = True
                    return 0.1
        return 0.0

    def step(self, action):
        if isinstance(action, dict): action = action['action']
        self.rgb, self.success = self.runner.step(action)
        pose_x, pose_y = self.runner.curr_state.position[0], self.runner.curr_state.position[2]
        pose_yaw = q.as_euler_angles(self.runner.curr_state.rotation)[1]
        self.time_t += 1
        reward = 1.0 if self.success else 0
        reward += self.check_progress(self.runner.curr_state.position)
        if self.success or (self.time_t >= self._max_step - 1): done = True
        else: done = False
        if self.prev_pose is not None: progress = np.sqrt(((pose_x - self.prev_pose[0])**2 + (pose_y - self.prev_pose[1])**2))
        else: progress = 0.0
        if progress < 0.02 : self.stuck_flag += 1
        else: self.stuck_flag = 0
        if self.stuck_flag > 20 :
            done = True
            self.stuck_flag = 0.0
        self.prev_pose = [pose_x, pose_y]
        self._last_action = action
        self.total_reward += reward
        obs = {'image': self.process_img(self.rgb), 'pose': np.array([pose_x, pose_y, pose_yaw, self.time_t+1]), 'prev_action': np.eye(self.action_dim)[self._last_action]}
        obs.update({'demo_im': self.demo_im, 'demo_act': self.demo_act, 'demo_pose': self.demo_pose})
        return obs, reward, done, {'episode_id': self.episode_id, 'step_id':self.time_t, 'success': self.success}

    def sample_episode(self, demo_pose):
        demo_start_pose, demo_start_orn = demo_pose[0,:3], demo_pose[0,3:]
        demo_end_pose, demo_end_orn = demo_pose[-1,:3], demo_pose[-1,3:]

        start_pose, end_pose = demo_end_pose, demo_start_pose
        look_back = q.as_rotation_vector(q.from_float_array(demo_end_orn))
        look_back[1] += np.pi
        look_back_quat = q.from_rotation_vector(look_back).components
        start_orn, end_orn = look_back_quat, demo_start_orn
        return start_pose, start_orn, end_pose, end_orn

    def reset_episode(self, **kwargs):
        scene, self.demo_im, self.demo_act, self.demo_pose = kwargs['scene'], kwargs['demo_im'], kwargs['demo_act'], kwargs['demo_pose']
        self.runner.init_episode(scene, *self.sample_episode(self.demo_pose.detach().numpy()))
        self.runner.init_common()

    def reset(self):
        self.rgb = self.runner.get_observations()['color_sensor'][:,:,:3]
        self.time_t = -1
        pose_x, pose_y = self.runner.curr_state.position[0], self.runner.curr_state.position[2]
        pose_yaw = q.as_euler_angles(self.runner.curr_state.rotation)[1]
        self._last_action = None
        obs = {'image': self.process_img(self.rgb), 'pose': np.array([pose_x, pose_y, pose_yaw, self.time_t+1]), 'prev_action': np.zeros(self.action_dim)}
        obs.update({'demo_im': self.demo_im, 'demo_act': self.demo_act, 'demo_pose': self.demo_pose})
        self.episode_id += 1
        self.prev_pose = None
        self.stuck_flag = 0
        self.success = 0
        self.total_reward = 0
        self.progress_checker = np.zeros(len(self.demo_im), dtype=np.bool)
        self.check_progress(self.runner.curr_state.position)
        return obs

    def seed(self, seed = None):
        self.seed = seed

    def close(self):
        self.runner._sim.close()

    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array' or mode == 'human':
            obs = self.rgb
            map = self.runner.get_map()
            blank_img = np.zeros_like(self.rgb)
            bs = self.rgb.shape[0]
            ratio = blank_img.shape[0]/max(map.shape[0],map.shape[1])
            map = cv2.resize(map, dsize=None, fx=ratio, fy=ratio)
            map_h, map_w, _ = map.shape
            start_h, start_w = int(bs/2 - map_h/2), int(bs/2 - map_w/2)
            blank_img[start_h:start_h+map_h, start_w:start_w+map_w ] = map
            view_img = np.concatenate([obs[:,:,:3],blank_img],1)
            view_img = np.ascontiguousarray(view_img)
            #view_img = cv2.resize(view_img, dsize=None, fx=2.0, fy=2.0)
            #print(view_img.shape)
            cv2.putText(view_img, 'step %d reward: %.2f'%(self.time_t, self.total_reward), (140, 110), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
            return view_img
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

