import torch.utils.data as data
import numpy as np
import joblib
import torch
import time
import cv2
from augmentations import RPFAugmentation


class HabitatDemoDataset(data.Dataset):
    def __init__(self, cfg, data_list, transform=None):
        self.data_list = data_list
        self.img_size = cfg.img_size
        self.action_dim = cfg.action_dim
        self.max_demo_length = cfg.dataset.max_demo_length
        self.max_follow_length = cfg.dataset.max_follow_length
        self.transform = None if cfg.dataset.transform == 'none' else eval(cfg.dataset.transform)(size=self.img_size)
        self.return_action = cfg.dataset.return_action
        self.return_pose = cfg.dataset.return_pose
        self.sparsify = cfg.dataset.sparsify_demo
        self.is_train = cfg.is_train

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def get_dist(self, demo_position):
        return np.linalg.norm(demo_position[-1] - demo_position[0], ord=2)

    def pull_image(self, index):
        demo_data = joblib.load(self.data_list[index])
        scene = self.data_list[index].split('/')[-1].split('_')[0]
        data_index = "".join(self.data_list[index].split('_')[4:-1])

        demo_length = len(demo_data['rgb'])
        demo_indicies = [i for i in range(0, demo_length, self.sparsify)]  # sparsify demonstration
        # if (demo_length-1) not in demo_indicies: demo_indicies.append(demo_length-1)
        ## Normalize img data
        demo_im = np.array(demo_data['rgb'], dtype=np.float32)[demo_indicies, :, :, :]
        if self.transform is not None:
            demo_im = self.transform(demo_im)
        demo_im = demo_im.transpose(0, 3, 2, 1) * 2 / 255. - 1
        demo_length = np.minimum(len(demo_im), self.max_demo_length)

        demo_im_out = np.zeros([self.max_demo_length, 3, self.img_size, self.img_size])
        demo_im_out[:demo_length] = demo_im[:demo_length]
        return_tensor = [torch.from_numpy(demo_im_out).float()]


        demo_data['action'] = np.array(demo_data['action'], dtype=np.int8) - 1
        demo_act = np.eye(self.action_dim)[demo_data['action']].astype(np.float32)[demo_indicies]
        demo_act_out = np.ones([self.max_demo_length, self.action_dim]) * (-100)
        demo_act_out[:demo_length] = demo_act[:demo_length]
        return_tensor.extend([torch.from_numpy(demo_act_out).float()])

        demo_pose = np.concatenate([demo_data['position'], demo_data['rotation']], 1)[demo_indicies]
        demo_pose_out = np.zeros([self.max_demo_length, 7])
        demo_pose_out[:demo_length] = demo_pose[:demo_length]
        return_tensor.extend([torch.from_numpy(demo_pose_out).float()])

        return_tensor.extend([scene, data_index])
        return return_tensor


class HabitatDemoNoiseDataset(data.Dataset):
    def __init__(self, cfg, data_list):
        self.data_list = data_list
        self.img_size = cfg.img_size
        self.action_dim = cfg.action_dim
        self.max_demo_length = cfg.dataset.max_demo_length
        self.max_follow_length = cfg.dataset.max_follow_length
        self.transform = None if cfg.dataset.transform == 'none' else eval(cfg.dataset.transform)(size=self.img_size)
        self.return_action = cfg.dataset.return_action
        self.return_pose = cfg.dataset.return_pose
        self.sparsify = cfg.dataset.sparsify_demo
        self.is_train = cfg.is_train

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        follower_data = joblib.load(self.data_list[index])
        mode = 'train' if 'train' in str(self.data_list[index]) else 'valid'
        demo_data_name = str(self.data_list[index]).replace('RECOVERY', 'DEMON')
        demo_data_name = demo_data_name[:demo_data_name.find('.dat.gz') - 3] + '0.dat.gz'
        demo_data = joblib.load(demo_data_name)

        ## sparsify demo
        demo_length = len(demo_data['rgb'])
        demo_indicies = [i for i in range(0, demo_length, self.sparsify)]  # sparsify demonstration
        # if (demo_length-1) not in demo_indicies: demo_indicies.append(demo_length-1)

        ## img data
        demo_im = np.array(demo_data['rgb'], dtype=np.float32)[demo_indicies, :, :, :]
        follower_im = np.array(follower_data['rgb'], dtype=np.float32)
        # demo_im = np.stack([cv2.resize(imgs, (self.img_size, self.img_size)) for imgs in demo_im],0)
        # follower_im =  np.stack([cv2.resize(imgs, (self.img_size, self.img_size)) for imgs in follower_im],0)
        if self.transform is not None:
            follower_im = self.transform(follower_im)
            demo_im = self.transform(demo_im)
        follower_im = follower_im.transpose(0, 3, 2, 1) * 2 / 255. - 1
        demo_im = demo_im.transpose(0, 3, 2, 1) * 2 / 255. - 1

        ## action data
        demo_length = np.minimum(len(demo_im), self.max_demo_length)
        follower_length = np.minimum(len(follower_im), self.max_follow_length)

        demo_im_out = np.zeros([self.max_demo_length, 3, self.img_size, self.img_size])
        demo_im_out[:demo_length] = demo_im[:demo_length]
        follower_im_out = np.zeros([self.max_follow_length, 3, self.img_size, self.img_size])
        follower_im_out[:follower_length] = follower_im[:follower_length]

        return_tensor = [torch.from_numpy(demo_im_out).float(), torch.from_numpy(follower_im_out).float()]

        if self.return_action:
            follower_data['action'] = np.array(follower_data['action'], dtype=np.int8) - 1
            demo_data['action'] = np.array(demo_data['action'], dtype=np.int8) - 1
            follower_act = follower_data['action'].astype(np.float32)
            demo_act = np.eye(self.action_dim)[demo_data['action']].astype(np.float32)[demo_indicies]

            demo_act_out = np.ones([self.max_demo_length, self.action_dim]) * (-100)
            demo_act_out[:demo_length] = demo_act[:demo_length]

            follower_act_out = np.ones([self.max_follow_length]) * (-100)
            follower_act_out[:follower_length] = follower_act[:follower_length]
            return_tensor.extend([torch.from_numpy(demo_act_out).float(), torch.from_numpy(follower_act_out).long()])

        ## pose data
        if self.return_pose:
            follower_pose = np.concatenate([follower_data['position'], follower_data['rotation']], 1)
            demo_pose = np.concatenate([demo_data['position'], demo_data['rotation']], 1)[demo_indicies]
            demo_pose_out = np.zeros([self.max_demo_length, 7])
            demo_pose_out[:demo_length] = demo_pose[:demo_length]
            follower_pose_out = np.zeros([self.max_follow_length, 7])
            follower_pose_out[:follower_length] = follower_pose[:follower_length]
            return_tensor.extend([torch.from_numpy(demo_pose_out).float(), torch.from_numpy(follower_pose_out).float()])

        return return_tensor

