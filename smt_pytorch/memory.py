"""
A neural memory module inspired by the Scene Memory Transformer paper.
"""
'''
-------> edited version for pytorch
'''

import typing
from typing import Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Each image modality is embedded into 64-dimensional vectors using a modified ResNet-18 [18]. 
We reduce the numbers of filters of all convolutional layers by a factor of 4 and use stride of 1 for the first two convolutional layers. 
We remove the global pooling to better capture thespatial information and directly apply the fully-connectedlayer at the end.  
Both pose and action vectors are embedded using a single 16-dimensional fully-connected layer.
'''

from resnet import resnet18
class Embedding(nn.Module):
    def __init__(self, cfg):
        super(Embedding, self).__init__()
        # 'rgb', 'depth', 'pose'
        assert len(cfg.network.inputs) > 0
        self.inputs = cfg.network.inputs

        if 'image' in self.inputs:
            self.use_image = True
            self.embed_image = resnet18(first_ch=4*cfg.network.num_stack, num_classes=64).cuda()
        else: self.use_image = False

        if 'prev_action' in self.inputs:
            self.use_action = True
            self.action_dim = cfg.action_dim
            self.embed_act = nn.Linear(self.action_dim,16).cuda()
        else: self.use_action = False

        if 'pose' in self.inputs:
            self.use_pose = True
            # p = (x/lambda, y/lambda, cos(th), sin(th), exp(-t))
            self.pose_dim = cfg.pose_dim
            self.embed_pose = nn.Linear(self.pose_dim,16).cuda()
        else: self.use_pose = False

        final_ch = 64 * (self.use_image) + 16 * (self.use_action + self.use_pose)
        self.final_embed = nn.Linear(final_ch, 128).cuda()

import time
class SceneMemory(object):
    # B * M (max_memory_size) * E (embedding)
    def __init__(self, cfg) -> None:
        self.B = cfg.training.num_envs
        self.max_memory_size = cfg.training.max_memory_size
        self.embedding_size = cfg.training.embedding_size
        self.embed_network = Embedding(cfg)
        self.gt_pose_buffer = torch.zeros([self.B, self.max_memory_size, 4],dtype=torch.float32).cuda() if self.embed_network.use_pose else None
        embedding_size_wo_pose = 64 + 16 * (self.embed_network.use_action)
        self.memory_buffer = torch.zeros([self.B, self.max_memory_size, embedding_size_wo_pose],dtype=torch.float32).cuda()
        self.memory_mask = torch.zeros([self.B, self.max_memory_size],dtype=torch.bool).cuda()
        self.reset_all()

    def freeze_embedding_network(self):
        self.embed_network.eval()

    def reset(self, reset_mask) -> None:
        assert(reset_mask.shape[0] == self.B)
        reset_memory = 1 - reset_mask.view(-1,1,1,1).float()
        self.memory_buffer = self.memory_buffer * reset_memory
        reset_memory_mask = 1 - reset_mask.view(-1,1).float()
        self.memory_mask = self.memory_mask * reset_memory_mask

    def reset_all(self) -> None:
        embedding_size_wo_pose = 64 + 16 * (self.embed_network.use_action)
        self.memory_buffer = torch.zeros([self.B, self.max_memory_size, embedding_size_wo_pose],dtype=torch.float32).cuda()
        self.memory_mask = torch.zeros([self.B, self.max_memory_size], dtype=torch.bool).cuda()

    def update_memory(self, obs, masks):
        new_embeddings = []
        new_embeddings.append(self.embed_network.embed_image(obs['image']))
        new_embeddings.append(self.embed_network.embed_act(obs['prev_action']))
        new_embedding = torch.cat(new_embeddings,1)
        self.memory_buffer = torch.cat([new_embedding.unsqueeze(1) * masks, self.memory_buffer[:, :-1]], 1)
        self.memory_mask = torch.cat([masks.bool(), self.memory_mask[:,:-1]],1)

        curr_rel_pose = torch.zeros([self.B, 1, 5]).cuda()
        curr_rel_pose[:, :, 2] = 1.0
        curr_rel_pose[:, :, -1] = torch.exp(-obs['pose'][:,-1]).unsqueeze(1)
        curr_pose_embedding = self.embed_network.embed_pose(curr_rel_pose)
        new_embeddings.append(curr_pose_embedding)

        self.gt_pose_buffer = torch.cat([obs['pose'].unsqueeze(1)*masks ,self.gt_pose_buffer[:,:-1]],1)
        relative_pose_buffer = self.get_relative_poses_embedding(obs['pose'])

        embedded_memory = []
        length = self.memory_mask.sum(dim=1)
        max_length = int(length.max())
        memory_buffer = torch.cat((self.memory_buffer[:,:max_length], relative_pose_buffer),-1)
        for i in range(max_length):
            embedded_memory.append(self.embed_network.final_embed(memory_buffer[:,i]))
        embedded_memory = torch.stack(embedded_memory,1)
        return embedded_memory, embedded_memory[:,0:1], self.memory_buffer[:,0], self.memory_mask[:,:max_length]

    def get_length(self):
        return self.memory_mask.sum(dim=1)

    def get_relative_poses_embedding(self, curr_pose):
        # curr_pose (x,y,orn,t)
        # shape B * 4
        curr_pose = curr_pose.unsqueeze(1) # shape B * 1 * 4
        curr_pose_x, curr_pose_y = curr_pose[:,:,0], curr_pose[:,:,1]
        gt_pose_x, gt_pose_y = self.gt_pose_buffer[:,:,0], self.gt_pose_buffer[:,:,1]
        curr_pose_yaw, gt_pose_yaw = curr_pose[:,:,2], self.gt_pose_buffer[:,:,2]

        del_x = gt_pose_x - curr_pose_x
        del_y = gt_pose_y - curr_pose_y
        th = torch.atan2(curr_pose_y, curr_pose_x)
        rel_x = del_x * torch.cos(-th) - del_y * torch.sin(th)
        rel_y = del_x * torch.sin(th) + del_y * torch.cos(th)
        rel_yaw = gt_pose_yaw - curr_pose_yaw
        exp_t = self.gt_pose_buffer[:,:,3]

        past_relative_poses = torch.stack([rel_x, rel_y, torch.cos(rel_yaw), torch.sin(rel_yaw), exp_t],2)
        past_pose_embeddings = []
        length = self.memory_mask.sum(dim=1)
        max_length = int(length.max())
        for i in range(max_length):
            if self.memory_mask[:,i].sum() == 0 : break
            past_pose_embeddings.append(self.embed_network.embed_pose(past_relative_poses[:,i]))
        pose_embedding = torch.stack(past_pose_embeddings, 1) * self.memory_mask[:,:i+1].unsqueeze(-1).float()
        return pose_embedding

    def get_relative_poses(self, curr_pose, prev_poses):
        # curr_pose (x,y,orn,t)
        # shape B * 4
        curr_pose = curr_pose.unsqueeze(1) # shape B * 1 * 4
        curr_pose_x, curr_pose_y = curr_pose[:,:,0], curr_pose[:,:,1]
        gt_pose_x, gt_pose_y = prev_poses[:,:,0], prev_poses[:,:,1]
        curr_pose_yaw, gt_pose_yaw = curr_pose[:,:,2], prev_poses[:,:,2]

        del_x = gt_pose_x - curr_pose_x
        del_y = gt_pose_y - curr_pose_y
        th = torch.atan2(curr_pose_y, curr_pose_x)
        rel_x = del_x * torch.cos(-th) - del_y * torch.sin(th)
        rel_y = del_x * torch.sin(th) + del_y * torch.cos(th)
        rel_yaw = gt_pose_yaw - curr_pose_yaw
        exp_t = prev_poses[:,:,3]

        relative_poses = torch.stack([rel_x, rel_y, torch.cos(rel_yaw), torch.sin(rel_yaw), exp_t],2)
        return relative_poses

    def embedd_observations(self, images, poses, prev_actions, memory_masks=None):
        # B * L * 3 * H * W : L will be 1
        images, poses, prev_actions = images.squeeze(1), poses.squeeze(1), prev_actions.squeeze(1)
        embedded_memory = []
        poses_x, poses_y, poses_th, time_t = poses[:,0], poses[:,1] , poses[:,2], poses[:,3]
        poses = torch.stack([poses_x, poses_y, torch.cos(poses_th), torch.sin(poses_th), torch.exp(-time_t)],1)
        relative_pose = self.get_relative_poses(poses, poses.unsqueeze(1)).squeeze(1).float()
        embeddings = [self.embed_network.embed_image(images),
                      self.embed_network.embed_act(prev_actions),
                      self.embed_network.embed_pose(relative_pose)]
        embeddings = torch.cat(embeddings, -1)
        embedded_memory.append(self.embed_network.final_embed(embeddings))
        if memory_masks is None:
            embedded_memory = torch.stack(embedded_memory, 1)
        else: embedded_memory = torch.stack(embedded_memory, 1) * memory_masks.view(-1,1,1)
        return embedded_memory, embedded_memory[:,0:1], memory_masks

    def embedd_with_pre_embeds(self, pre_embeddings, poses, memory_masks=None):
        L = pre_embeddings.shape[1]
        embedded_memory = []
        relative_pose = self.get_relative_poses(poses[:, 0], poses)
        for l in range(L):
            embeddings = [pre_embeddings[:,l:l+1], self.embed_network.embed_pose(relative_pose[:,l:l+1])]
            embeddings = torch.cat(embeddings, -1)
            embedded_memory.append(self.embed_network.final_embed(embeddings))
        if memory_masks is None:
            embedded_memory = torch.stack(embedded_memory, 1)
        else:
            embedded_memory = torch.stack(embedded_memory, 1) * memory_masks.view(-1,L,1,1)
        return embedded_memory, embedded_memory[:,0:1]
