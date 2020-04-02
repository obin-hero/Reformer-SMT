import os
import random
from enum import Enum
import numpy as np
from PIL import Image
import habitat_sim
import habitat_sim.agent
import habitat_sim.bindings as hsim
import habitat_sim.utils as utils
import quaternion as q
from utils.habitat_settings import default_sim_settings, make_cfg
from utils.mapper.TopDownMap import TopDownMap

class DemoRunner:
    def __init__(self, settings=None):
        if settings is None : sim_settings = default_sim_settings
        self._shortest_path = hsim.ShortestPath()
        self._cfg = make_cfg(sim_settings)
        print(sim_settings)
        self._sim = habitat_sim.Simulator(self._cfg)
        self._sim_settings = sim_settings
        self.top_down_map = TopDownMap(sim=self._sim)
        self.scene = self._sim_settings['scene']

    def init_agent_state(self, agent_id):
        # initialize the agent at a random start state
        agent = self._sim.initialize_agent(agent_id)
        start_state = agent.get_state()

        if (start_state.position != self.init_position).any():
            start_state.position = self.init_position
            start_state.rotation = q.from_float_array(self.init_rotation)
            start_state.sensor_states = dict()  ## Initialize sensor

        agent.set_state(start_state)
        return start_state

    def compute_shortest_path(self, start_pos, end_pos):
        self._shortest_path.requested_start = start_pos
        self._shortest_path.requested_end = end_pos
        self._sim.pathfinder.find_path(self._shortest_path)
        #print("shortest_path.geodesic_distance", self._shortest_path.geodesic_distance)

    def geodesic_distance(self, position_a, position_b):
        self._shortest_path.requested_start = np.array(position_a, dtype=np.float32)
        self._shortest_path.requested_end = np.array(position_b, dtype=np.float32)
        self._sim.pathfinder.find_path(self._shortest_path)
        return self._shortest_path.geodesic_distance

    def euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def step(self, in_action):
        action_names = list(self._cfg.agents[self._sim_settings["default_agent"]].action_space.keys())
        action = action_names[in_action]
        observations = self._sim.step(action)
        color_obs = observations["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA").convert("RGB")
        current_position = self._sim.agents[0].get_state().position
        # self.dist_to_goal = np.linalg.norm(current_position - self.end_position)
        self.dist_to_goal = self.euclidean_distance(current_position, self.end_position)
        self.agent_episode_distance += self.euclidean_distance(current_position, self.previous_position)
        done = False
        if self.dist_to_goal / self.initial_dist_to_goal <= 0.1 or self.dist_to_goal <= 0.4:
            done = True
        self.previous_position = current_position
        self.top_down_map.update_metric()
        return np.asarray(color_img), done

    def get_observations(self):
        return self._sim.get_sensor_observations()

    def get_curposition(self):
        return self._sim.agents[0].get_state().position

    def init_episode(self, scene, init_position, init_rotation, end_position, end_rotation, orig_start_pose = None):
        self.scene = "data/scene_datasets/mp3d/{}/{}.glb".format(scene[0],scene[0])
        self.init_position = init_position
        self.init_rotation = init_rotation
        self.end_position = end_position
        self.end_rotation = end_rotation
        self.orig_start_pose = orig_start_pose

    def init_common(self):

        if self._sim_settings["scene"] != self.scene:
            self._sim_settings["scene"] = self.scene
            self._cfg = make_cfg(self._sim_settings)
            try: self._sim.close()
            except: pass
            self._sim = habitat_sim.Simulator(self._cfg)
            self.top_down_map._sim = self._sim

        obs = self._sim.reset()
        #random.seed(self._sim_settings["seed"])
        #self._sim.seed(self._sim_settings["seed"])

        start_state = self.init_agent_state(self._sim_settings["default_agent"])
        self.initial_dist_to_goal = np.linalg.norm(start_state.position - self.end_position)
        self.dist_to_goal = np.linalg.norm(start_state.position - self.end_position)
        self.top_down_map.reset_metric(start_pose = start_state.position, goal_pose=self.end_position, orig_start_pose = self.orig_start_pose)
        self.previous_position = start_state.position
        self.agent_episode_distance = 0

        return start_state, obs['color_sensor']

    def get_map(self):
        return self.top_down_map.draw_top_down_map()

import matplotlib.pyplot as plt
import cv2
import quaternion
if __name__ == '__main__':
    runner = DemoRunner()
    action_num = len(runner._cfg.agents[runner._sim_settings["default_agent"]].action_space)
    end_position = runner._sim.pathfinder.get_random_navigable_point()
    state = runner._sim.agents[0].get_state()
    runner.init_episode(runner.scene, state.position, state.rotation.components, end_position, state.rotation.components)
    runner.init_common()
    for i in range(100):
        rand_action = np.random.randint(action_num)
        #rand_action = 0
        obs_img, done = runner.step(rand_action)
        map = runner.get_map()
        view_img = np.concatenate([obs_img, map],1)
        cv2.imshow('hi',view_img[:,:,[2,1,0]])
        cv2.waitKey(0)
        #plt.imshow(view_img)
        #plt.show()
