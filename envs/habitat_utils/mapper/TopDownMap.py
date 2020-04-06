import envs.habitat_utils.mapper.habitat_maps as maps
from habitat_sim import Simulator
import habitat_sim.bindings as hsim
from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
import cv2
MAP_THICKNESS_SCALAR: int = 1250
def quaternion_to_rotation(q_r, q_i, q_j, q_k):
    r"""
    ref: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    s = 1  # unit quaternion
    rotation_mat = np.array(
        [
            [
                1 - 2 * s * (q_j ** 2 + q_k ** 2),
                2 * s * (q_i * q_j - q_k * q_r),
                2 * s * (q_i * q_k + q_j * q_r),
            ],
            [
                2 * s * (q_i * q_j + q_k * q_r),
                1 - 2 * s * (q_i ** 2 + q_k ** 2),
                2 * s * (q_j * q_k - q_i * q_r),
            ],
            [
                2 * s * (q_i * q_k - q_j * q_r),
                2 * s * (q_j * q_k + q_i * q_r),
                1 - 2 * s * (q_i ** 2 + q_j ** 2),
            ],
        ],
        dtype=np.float32,
    )
    return rotation_mat


def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion

    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate

    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format
    """
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi

def quat_to_xy_heading(quat):
    direction_vector = np.array([0, 0, -1])

    heading_vector = quaternion_rotate_vector(quat, direction_vector)

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return np.array(phi)




class TopDownMap():
    r"""Top Down Map measure
    """

    def __init__(
        self, *args: Any, sim: Simulator, **kwargs: Any
    ):
        self._sim = sim
        self._grid_delta = 3
        self._step_count = None
        self._map_resolution = (1250, 1250)
        self._num_samples = 10000
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self._shortest_path_points = None
        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        self.line_thickness = int(
            np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
        )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "top_down_map"

    def _check_valid_nav_point(self, point: List[float]):
        self._sim.is_navigable(point)

    def get_original_map(self):
        top_down_map = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            True,
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        self._ind_x_min = range_x[0]
        self._ind_x_max = range_x[-1]
        self._ind_y_min = range_y[0]
        self._ind_y_max = range_y[-1]

        return top_down_map

    def draw_source_and_target(self, start_pose, goal_pose):
        # mark source point
        s_x, s_y = maps.to_grid(
            start_pose[0],
            start_pose[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        self._top_down_map[
            s_x - point_padding : s_x + point_padding + 1,
            s_y - point_padding : s_y + point_padding + 1,
        ] = maps.MAP_SOURCE_POINT_INDICATOR

        # mark target point
        t_x, t_y = maps.to_grid(
            goal_pose[0],
            goal_pose[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._top_down_map[
            t_x - point_padding : t_x + point_padding + 1,
            t_y - point_padding : t_y + point_padding + 1,
        ] = maps.MAP_TARGET_POINT_INDICATOR

    def reset_metric(self, *args: Any, start_pose,  goal_pose, orig_start_pose=None, **kwargs: Any):


        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()

        if orig_start_pose is not None:
            shortest_path = hsim.ShortestPath()
            shortest_path.requested_start = orig_start_pose
            shortest_path.requested_end = goal_pose
            self._sim.pathfinder.find_path(shortest_path)
            self._shortest_path_points = shortest_path.points
            self._shortest_path_points = [
                maps.to_grid(
                    p[0],
                    p[2],
                    self._coordinate_min,
                    self._coordinate_max,
                    self._map_resolution,
                )[::-1]
                for p in self._shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )



        agent_position =  self._sim.agents[0].get_state().position
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._previous_xy_location = (a_y, a_x)

        # draw shortest path
        shortest_path = hsim.ShortestPath()
        shortest_path.requested_start = agent_position
        shortest_path.requested_end = goal_pose
        self._sim.pathfinder.find_path(shortest_path)
        self._shortest_path_points = shortest_path.points
        self._shortest_path_points = [
            maps.to_grid(
                p[0],
                p[2],
                self._coordinate_min,
                self._coordinate_max,
                self._map_resolution,
            )[::-1]
            for p in self._shortest_path_points
        ]
        maps.draw_path(
            self._top_down_map,
            self._shortest_path_points,
            maps.MAP_SHORTEST_PATH_COLOR2,
            self.line_thickness,
        )


        self.update_fog_of_war_mask(np.array([a_x, a_y]))


        self.draw_source_and_target(start_pose,  goal_pose)
        self.update_metric()

    def _clip_map(self, _map):
        return _map[
            self._ind_x_min
            - self._grid_delta : self._ind_x_max
            + self._grid_delta,
            self._ind_y_min
            - self._grid_delta : self._ind_y_max
            + self._grid_delta,
        ]

    def update_metric(self):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.agents[0].get_state().position
        )

        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        clipped_house_map = self._clip_map(house_map)

        clipped_fog_of_war_map = None

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_map,
            "agent_map_coord": (
                map_agent_x - (self._ind_x_min - self._grid_delta),
                map_agent_y - (self._ind_y_min - self._grid_delta),
            ),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state =  self._sim.agents[0].get_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // 50, 245
            )

            thickness = int(
                np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
            )
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        return
        # if self._config.FOG_OF_WAR.DRAW:
        #     self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
        #         self._top_down_map,
        #         self._fog_of_war_mask,
        #         agent_position,
        #         self.get_polar_angle(),
        #         fov=self._config.FOG_OF_WAR.FOV,
        #         max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
        #         * max(self._map_resolution)
        #         / (self._coordinate_max - self._coordinate_min),
        #     )


    def draw_top_down_map(self, output_size=256):
        info = self._metric
        rotation = self._sim.agents[0].get_state().rotation
        heading = quat_to_xy_heading(rotation.inverse())
        top_down_map = maps.colorize_topdown_map(
            info["map"], info["fog_of_war_mask"]
        )
        original_map_size = top_down_map.shape[:2]
        map_scale = np.array(
            (1, original_map_size[1] * 1.0 / original_map_size[0])
        )
        new_map_size = np.round(output_size * map_scale).astype(np.int32)
        # OpenCV expects w, h but map size is in h, w
        top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

        map_agent_pos = info["agent_map_coord"]
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
