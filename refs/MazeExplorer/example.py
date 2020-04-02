# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from mazeexplorer import MazeExplorer
import cv2

env_train = MazeExplorer(unique_maps=True, number_maps=10, keys=6, size=(20,20), random_spawn=True,
                                 random_textures=True, random_key_positions=True, action_frame_repeat=4,
                                 actions="MOVE_FORWARD TURN_LEFT TURN_RIGHT MOVE_LEFT MOVE_RIGHT", scaled_resolution=(64, 64),
                                 data_augmentation=False)

obs = env_train.reset()
for i in range(1000):
    obs, rewards, dones, info = env_train.step(env_train.action_space.sample())
    cv2.imshow('hi',obs[:,:,[2,1,0]])
    cv2.waitKey(0)

