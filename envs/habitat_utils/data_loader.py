import torch.utils.data as data
import numpy as np
import joblib
import torch
import time
import cv2
from envs.habitat_utils.augmentations import RPFAugmentation

class HabitatDemoDataset(data.Dataset):
    def __init__(self,data_list, sparsify=1,transform=RPFAugmentation):
        self.data_list = data_list
        self.img_size = 64
        self.action_dim = 3
        self.max_demo_length = 30
        self.max_follow_length = 50
        self.transform = transform()
        self.return_action = True
        self.return_pose = True#cfg.dataset.return_pose
        self.sparsify = sparsify

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def get_dist(self, demo_position):
        return np.linalg.norm(demo_position[-1] - demo_position[0], ord=2)

    def pull_image(self, index):
        demo_data = joblib.load(self.data_list[index])
        scene_name = self.data_list[index].split('/')[-1].split('_')[0]
        data_index = "".join(self.data_list[index].split('_')[4:-1])

        demo_length = len(demo_data['rgb'])
        demo_indicies = [ i for i in range(0,demo_length,self.sparsify)] #sparsify demonstration
        #if (demo_length-1) not in demo_indicies: demo_indicies.append(demo_length-1)
        ## Normalize img data
        demo_im = np.array(demo_data['rgb'], dtype=np.float32)[demo_indicies, :, :, :]
        if self.transform is not None:
            demo_im = self.transform(demo_im)
        demo_im = demo_im.transpose(0, 3, 2, 1) * 2 / 255. - 1
        demo_length = np.minimum(len(demo_im), self.max_demo_length)

        demo_im_out = np.zeros([self.max_demo_length, 3, self.img_size, self.img_size])
        demo_im_out[:demo_length] = demo_im[:demo_length]
        return_tensor = [torch.from_numpy(demo_im_out).float()]

        if self.return_action:
            demo_data['action'] = np.array(demo_data['action'], dtype=np.int8) - 1
            demo_act = np.eye(self.action_dim)[demo_data['action']].astype(np.float32)[demo_indicies]
            demo_act_out = np.ones([self.max_demo_length, self.action_dim]) * (-100)
            demo_act_out[:demo_length] = demo_act[:demo_length]
            return_tensor.extend([torch.from_numpy(demo_act_out).float()])

        if self.return_pose:
            demo_pose = np.concatenate([demo_data['position'],demo_data['rotation']],1)[demo_indicies]
            demo_pose_out = np.zeros([self.max_demo_length, 7])
            demo_pose_out[:demo_length] = demo_pose[:demo_length]
            return_tensor.extend([torch.from_numpy(demo_pose_out).float()])

        return_tensor.extend([scene_name, data_index])
        if (demo_pose[0] == demo_pose[-1]).all():
            print('check this demo-pose ', demo_pose)
        return return_tensor
