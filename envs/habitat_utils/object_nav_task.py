#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode, ObjectViewLocation, ObjectGoal
import time
@registry.register_sensor(name="CustomObjectSensor")
class CustomObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    def __init__(
        self, sim, config: Config, dataset: Dataset, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dataset = dataset
        use_depth = 'DEPTH_SENSOR' in self._sim.config.AGENT_0.SENSORS
        use_rgb = 'RGB_SENSOR' in self._sim.config.AGENT_0.SENSORS
        self.channel = use_depth + 3 * use_rgb
        self.height = self._sim.config.RGB_SENSOR.HEIGHT if use_rgb else self._sim.config.DEPTH_SENSOR.HEIGHT
        self.width = self._sim.config.RGB_SENSOR.WIDTH if use_rgb else self._sim.config.DEPTH_SENSOR.WIDTH
        self.curr_episode_id = -1
        self.curr_scene_id = ''
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "objectgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        if self.config.GOAL_SPEC == 'OBJECT_IMG':
            return spaces.Box(low=0, high=1.0, shape=(self.height, self.width, self.channel+1), dtype=np.float32)
        max_value = (self.config.GOAL_SPEC_MAX_VAL - 1,)
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[int]:
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            if len(episode.goals) == 0:
                logger.error(
                    f"No goal specified for episode {episode.episode_id}."
                )
                return None
            if not isinstance(episode.goals[0], ObjectGoal):
                logger.error(
                    f"First goal should be ObjectGoal, episode {episode.episode_id}."
                )
                return None
            category_name = episode.object_category
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.GOAL_SPEC == "OBJECT_ID":
            return np.array([episode.goals[0].object_name_id], dtype=np.int64)
        elif self.config.GOAL_SPEC == "OBJECT_IMG":
            episode_id = episode.episode_id
            scene_id = episode.scene_id
            if (self.curr_episode_id != episode_id) and (self.curr_scene_id != scene_id):
                closest = episode.info['closest_goal_object_id']
                for goal in episode.goals:
                    if goal.object_id == int(closest):
                        viewpoint = goal.view_points[0].agent_state
                        break
                obs = self._sim.get_observations_at(episode.info['best_viewpoint_position'], viewpoint.rotation)
                rgb_array = obs['rgb']/255.0
                depth_array = obs['depth']
                category_array = np.ones_like(depth_array) * self._dataset.category_to_task_category_id[episode.object_category]
                self.goal_obs = np.concatenate([rgb_array, depth_array, category_array],2)
                self.curr_episode_id = episode_id
                self.curr_scene_id = scene_id
            return self.goal_obs
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )


