#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Thread
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import cv2
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict
import gym
try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    import torch.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
OBSERVATION_SPACE_COMMAND = "observation_space"
ACTION_SPACE_COMMAND = "action_space"
CALL_COMMAND = "call"
EPISODE_COMMAND = "current_episode"

from time import time

def tile_images(images) -> np.ndarray:
    r"""Tile multiple images into single image

    Args:
        images: list of images where each image has dimension
            (height x width x channels)

    Returns:
        tiled image (new_height x width x channels)
    """
    assert len(images) > 0, "empty list of images"
    np_images = np.asarray(images)
    n_images, height, width, n_channels = np_images.shape
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    # pad with empty images to complete the rectangle
    np_images = np.array(
        images
        + [images[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = np_images.reshape(
        new_height, new_width, height, width, n_channels
    )
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape(
        new_height * height, new_width * width, n_channels
    )
    return out_image

class VectorEnv:
    observation_spaces: List[SpaceDict]
    action_spaces: List[SpaceDict]
    _workers: List[Union[mp.Process, Thread]]
    _is_waiting: bool
    _num_envs: int
    _auto_reset_done: bool
    _mp_ctx: BaseContext
    _connection_read_fns: List[Callable[[], Any]]
    _connection_write_fns: List[Callable[[Any], None]]

    def __init__(
        self,
        make_env_fn,
        env_fn_args = None,
        auto_reset_done = True,
        multiprocessing_start_method = "forkserver",
    ) -> None:

        self._is_waiting = False
        self._is_closed = True

        assert (
            env_fn_args is not None and len(env_fn_args) > 0
        ), "number of environments to be created should be greater than 0"

        self._num_envs = len(env_fn_args)

        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        self._auto_reset_done = auto_reset_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args, make_env_fn
        )

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((OBSERVATION_SPACE_COMMAND, None))
        self.observation_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        for write_fn in self._connection_write_fns:
            write_fn((ACTION_SPACE_COMMAND, None))
        self.action_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        self._paused = []

    @property
    def num_envs(self):
        r"""number of individual environments.
        """
        return self._num_envs - len(self._paused)

    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment.
        """
        env = env_fn(*env_fn_args)
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # different step methods for habitat.RLEnv and habitat.Env
                    if isinstance(
                        env, gym.Env
                    ):
                        # habitat.RLEnv
                        observations, reward, done, info = env.step(**data)
                        if auto_reset_done and done:
                            observations = env.reset()
                        connection_write_fn((observations, reward, done, info))
                    else:
                        raise NotImplementedError

                elif command == RESET_COMMAND:
                    observations = env.reset()
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif (
                    command == OBSERVATION_SPACE_COMMAND
                    or command == ACTION_SPACE_COMMAND
                ):
                    if isinstance(command, str):
                        connection_write_fn(getattr(env, command))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None or len(function_args) == 0:
                        result = getattr(env, function_name)()
                    else:
                        result = getattr(env, function_name)(**function_args)
                    connection_write_fn(result)

                # TODO: update CALL_COMMAND for getting attribute like this
                elif command == EPISODE_COMMAND:
                    connection_write_fn(env.current_episode)
                else:
                    raise NotImplementedError

                command, data = connection_read_fn()

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt:
            print("Worker KeyboardInterrupt")
        finally:
            env.close()

    def _spawn_workers(
        self,
        env_fn_args,
        make_env_fn,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_connections, worker_connections = zip(
            *[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_envs)]
        )
        self._workers = []
        for worker_conn, parent_conn, env_args in zip(
            worker_connections, parent_connections, env_fn_args
        ):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(ps)
            ps.daemon = True
            ps.start()
            worker_conn.close()
        return (
            [p.recv for p in parent_connections],
            [p.send for p in parent_connections],
        )

    def reset(self):
        r"""Reset all the vectorized environments

        :return: list of outputs from the reset method of envs.
        """
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def reset_at(self, index_env: int):
        r"""Reset in the index_env environment in the vector.

        :param index_env: index of the environment to be reset
        :return: list containing the output of reset method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((RESET_COMMAND, None))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def step_at(self, index_env: int, action: Dict[str, Any]):
        r"""Step in the index_env environment in the vector.

        :param index_env: index of the environment to be stepped into
        :param action: action to be taken
        :return: list containing the output of step method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((STEP_COMMAND, action))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def async_step(self, data: List[Union[int, str, Dict[str, Any]]]) -> None:
        r"""Asynchronously step in the environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        """
        # Backward compatibility
        if isinstance(data[0], (int, np.integer, str)):
            data = [{"action": {"action": action}} for action in data]

        self._is_waiting = True
        for write_fn, args in zip(self._connection_write_fns, data):
            write_fn((STEP_COMMAND, args))

    def wait_step(self):
        r"""Wait until all the asynchronized environments have synchronized.
        """
        observations = []
        for read_fn in self._connection_read_fns:
            observations.append(read_fn())
        self._is_waiting = False
        return observations

    def step(self, data: List[Union[int, str, Dict[str, Any]]]) -> List[Any]:
        r"""Perform actions in the vectorized environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        :return: list of outputs from the step method of envs.
        """
        self.async_step(data)
        return self.wait_step()

    def close(self) -> None:
        if self._is_closed:
            return

        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()

        for write_fn in self._connection_write_fns:
            write_fn((CLOSE_COMMAND, None))

        for _, _, write_fn, _ in self._paused:
            write_fn((CLOSE_COMMAND, None))

        for process in self._workers:
            process.join()

        for _, _, _, process in self._paused:
            process.join()

        self._is_closed = True

    def pause_at(self, index: int) -> None:
        r"""Pauses computation on this env without destroying the env.

        :param index: which env to pause. All indexes after this one will be
            shifted down by one.

        This is useful for not needing to call steps on all environments when
        only some are active (for example during the last episodes of running
        eval episodes).
        """
        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()
        read_fn = self._connection_read_fns.pop(index)
        write_fn = self._connection_write_fns.pop(index)
        worker = self._workers.pop(index)
        self._paused.append((index, read_fn, write_fn, worker))

    def resume_all(self) -> None:
        r"""Resumes any paused envs.
        """
        for index, read_fn, write_fn, worker in reversed(self._paused):
            self._connection_read_fns.insert(index, read_fn)
            self._connection_write_fns.insert(index, write_fn)
            self._workers.insert(index, worker)
        self._paused = []

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        r"""Calls a function (which is passed by name) on the selected env and
        returns the result.

        :param index: which env to call the function on.
        :param function_name: the name of the function to call on the env.
        :param function_args: optional function args.
        :return: result of calling the function.
        """
        self._is_waiting = True
        self._connection_write_fns[index](
            (CALL_COMMAND, (function_name, function_args))
        )
        result = self._connection_read_fns[index]()
        self._is_waiting = False
        return result

    def call(
        self,
        function_names: List[str],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        r"""Calls a list of functions (which are passed by name) on the
        corresponding env (by index).

        :param function_names: the name of the functions to call on the envs.
        :param function_args_list: list of function args for each function. If
            provided, :py:`len(function_args_list)` should be as long as
            :py:`len(function_names)`.
        :return: result of calling the function.
        """
        self._is_waiting = True
        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(
            self._connection_write_fns, func_args
        ):
            write_fn((CALL_COMMAND, func_args_on))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def render(
        self, mode: str = "human", *args, **kwargs
    ) -> Union[np.ndarray, None]:
        r"""Render observations from all environments in a tiled image.
        """
        for write_fn in self._connection_write_fns:
            write_fn((RENDER_COMMAND, (args, {"mode": mode, **kwargs})))
        images = [read_fn() for read_fn in self._connection_read_fns]
        tile = tile_images(images)
        if mode == "human":
            
            try:
                from habitat.core.utils import try_cv2_import
                cv2 = try_cv2_import()
            except: pass

            cv2.imshow("vecenv", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {"forkserver", "spawn", "fork"}

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


import torch
class PreprocessVectorEnv(VectorEnv):
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

        for key, box in obs_space.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = {k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
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
        if self.keys == [None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs





if __name__ == '__main__':
    import gym
    from envs.make_env_fn import _make_env_fn
    num_env = 20
    envs = VectorEnv(_make_env_fn, [('nav_maze_static_01', i) for i in range(num_env)])
    obs = envs.reset()
    for i in range(100):
        envs.step([envs.action_spaces[i].sample() for i in range(num_env)])
        envs.render('human')
