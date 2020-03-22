import habitat
import numpy as np

from habitat.utils.visualizations import maps
import cv2

# top_down_map = draw_top_down_map(info, observations['heading'], 256)
def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
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


class ExploreEnv(habitat.RLEnv):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, habitat_cfg, env_cfg):
        super(ExploreEnv, self).__init__(habitat_cfg)
        self.map_resolution = habitat_cfg.TASK.TOP_DOWN_MAP.MAP_RESOLUTION

    def get_reward_range(self):
        return [-1, 1]

    def reset(self):
        self.obs = super(ExploreEnv, self).reset()
        self.info = None
        self.past_fow = 0
        self.curr_fow = 0
        info = self.habitat_env.get_metrics()
        self.prev_pose = self.habitat_env.sim.get_agent_state().position
        self.default_z = self.prev_pose[1]
        self.stuck = 0
        self.time_step = 0
        self.visited_cells = 0
        self.done = False
        self.done_info = None
        self.total_reward = 0
        return self.obs

    def get_reward(self,obs):
        new_pixel = self.curr_fow - self.past_fow
        self.past_fow = self.curr_fow
        exploration_reward = np.clip(new_pixel, 0, 50)/500 # 0 ~ 0.1
        living_penalty = - 0.002
        collision = self.habitat_env.sim.previous_step_collided
        collision_penalty = - 0.0 * collision
        reward = exploration_reward + living_penalty + collision_penalty
        self.total_reward += reward
        return reward

    def get_done(self, obs):
        curr_pose = self.habitat_env.sim.get_agent_state().position
        progress = abs((curr_pose - self.prev_pose).sum())
        if progress <= 0.1 :
            self.stuck += 1
        else: self.stuck = 0
        self.stuck_info = {'curr_pose': curr_pose, 'prev_pose': self.prev_pose, 'progress': progress}
        height_change = abs(curr_pose[1] - self.default_z) # entered different floor
        self.prev_pose = curr_pose
        time_over = self.habitat_env.episode_over
        #collision = self.habitat_env.sim.previous_step_collided
        if self.map_size == 0 :
            self.map_size = 1
        stuck = True if self.stuck > 30 else False
        self.visited_cells = float(self.curr_fow)/self.map_resolution #/ float(self.map_size)
        all_covered = float(self.curr_fow)/float(self.map_size) > 0.9
        done = time_over or all_covered or stuck or (height_change>0.5)# or collision
        #done = False
        #print('done - time_over {} all_covered {}'.format(time_over, all_covered))
        return done, {'all_covered': (self.curr_fow, self.map_size),
                      'stuck': self.stuck, 'time_over':time_over,
                      'height_change': height_change}

    def step(self,action):
        self.obs = self.habitat_env.step(action)
        self.info = self.habitat_env.get_metrics()

        fow = self.info["top_down_map"]["fog_of_war_mask"]
        self.curr_fow = np.sum(fow)
        self.map_size = (self.info['top_down_map']['map'] != 0 ).sum()

        self.reward = self.get_reward(self.obs)
        self.done, self.done_info = self.get_done(self.obs)
        self.info.update({'visited_cells': self.visited_cells})
        self.time_step += 1
        return self.obs, self.reward, self.done, self.info

    def render(self, mode='human'):
        im = self.obs['rgb']
        if self.info is None:
            top_down_map = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            top_down_map = draw_top_down_map(self.info, self.obs['heading'], 256)
        im = cv2.resize(im, (256,256))
        top_down_map = cv2.resize(top_down_map,(256,256))
        curr_info = 'step %3d covered %.3f' % (self.time_step, self.visited_cells)
        view_img = np.concatenate([im, top_down_map], 1)
        cv2.putText(view_img, curr_info, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        if self.done_info is not None:
            corver_info = 'covered %5d/%5d'%(self.curr_fow, self.map_size)
            cv2.putText(view_img, corver_info, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
            stuck_info = 'stuck %02d , z_change %.2f'%(self.stuck, self.done_info['height_change'])
            cv2.putText(view_img, stuck_info, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
            cp, pp, progress = self.stuck_info['curr_pose'], self.stuck_info['prev_pose'], self.stuck_info['progress']
            stuck_info_detail =  'cp : %.2f, %.2f, %.2f, pp: %.2f, %.2f, %.2f,'%(cp[0], cp[1], cp[2],pp[0], pp[1], pp[2])
            cv2.putText(view_img, stuck_info_detail, (20, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)
            stuck_info_detail =  'progress : %.3f'%(progress)
            cv2.putText(view_img, stuck_info_detail, (20, 100), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)
            stuck_info_detail =  'reward : %.3f'%(self.total_reward)
            cv2.putText(view_img, stuck_info_detail, (20, 120), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1)
        return view_img
if __name__ == '__main__':
    cfg = habitat.get_config('config/explore_mp3d.yaml')
    cfg.defrost()
    cfg.DATASET.DATA_PATH = '../habitat-api/data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz'
    cfg.DATASET.SCENES_DIR = '../habitat-api/data/scene_datasets'
    cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    cfg.freeze()
    print(cfg)

    env = ExploreEnv(cfg, env_cfg=None)
    obs = env.reset()

    for episode in range(100):
        env.reset()
        images = []
        end = False
        action = 1#None
        past_fow = 0
        while not env.habitat_env.episode_over:
            if action is not None :
                best_action = action
            observations, reward, done, info = env.step(best_action)
            print('reward {} , done {}'.format(reward, done))
            if done : break
            im = observations["rgb"]
            top_down_map = draw_top_down_map(info, observations['heading'], 256)
            curr_fow = np.sum(info["top_down_map"]["fog_of_war_mask"])
            print(curr_fow - past_fow)
            view_img = np.concatenate([im/255., top_down_map/255.0],1)
            cv2.imshow('hi',view_img[:,:,[2,1,0]])
            key = cv2.waitKey(0)
            past_fow = curr_fow
            action = None
            if key == ord('q') :
                end = True
                break
            elif key == ord('p'):
                end = False
                break
            elif key == ord('w'): action = 'MOVE_FORWARD'
            elif key == ord('a'): action = 'TURN_LEFT'
            elif key == ord('d'): action = 'TURN_RIGHT'
        if end : break
