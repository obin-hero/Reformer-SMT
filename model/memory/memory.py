"""
A neural memory module inspired by the Scene Memory Transformer paper.
"""
'''
-------> edited version for pytorch
'''

import torch
import torch.nn as nn

'''
Each image modality is embedded into 64-dimensional vectors using a modified ResNet-18 [18]. 
We reduce the numbers of filters of all convolutional layers by a factor of 4 and use stride of 1 for the first two convolutional layers. 
We remove the global pooling to better capture thespatial information and directly apply the fully-connectedlayer at the end.  
Both pose and action vectors are embedded using a single 16-dimensional fully-connected layer.
'''

from model.memory.resnet import resnet18
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


class SceneMemory(nn.Module):
    # B * M (max_memory_size) * E (embedding)
    def __init__(self, cfg) -> None:
        super(SceneMemory, self).__init__()
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
        self.memory_buffer = self.memory_buffer * reset_mask.view(-1,1,1).float()
        self.memory_mask = self.memory_mask * reset_mask.view(-1,1).bool()

    def reset_all(self) -> None:
        embedding_size_wo_pose = 64 + 16 * (self.embed_network.use_action)
        self.memory_buffer = torch.zeros([self.B, self.max_memory_size, embedding_size_wo_pose],dtype=torch.float32).cuda()
        self.memory_mask = torch.zeros([self.B, self.max_memory_size], dtype=torch.bool).cuda()
        self.gt_pose_buffer = torch.zeros([self.B, self.max_memory_size, 4],
                                          dtype=torch.float32).cuda() if self.embed_network.use_pose else None

    def update_memory(self, obs, masks):
        self.reset(masks)
        new_embeddings = []
        new_embeddings.append(self.embed_network.embed_image(obs['image']))
        new_embeddings.append(self.embed_network.embed_act(obs['prev_action']))
        new_embedding = torch.cat(new_embeddings,1)

        self.memory_buffer = torch.cat([new_embedding.unsqueeze(1), self.memory_buffer[:, :-1]], 1)
        self.memory_mask = torch.cat([torch.ones_like(masks, dtype=torch.bool), self.memory_mask[:,:-1]],1)
        self.gt_pose_buffer = torch.cat([(obs['pose']).unsqueeze(1),self.gt_pose_buffer[:,:-1]],1)

        length = self.memory_mask.sum(dim=1)
        max_length = int(length.max())
        relative_poses = self.get_relative_poses(obs['pose'], self.gt_pose_buffer[:,:max_length])
        embedded_memory = []
        for i in range(max_length):
            embedded_pose = self.embed_network.embed_pose(relative_poses[:,i])
            embedded_memory.append(self.embed_network.final_embed(torch.cat((self.memory_buffer[:,i], embedded_pose),1)))
        embedded_memory = torch.stack(embedded_memory,1) * self.memory_mask[:, :max_length].unsqueeze(-1).float()
        return embedded_memory, embedded_memory[:,0:1], self.memory_buffer[:,0], self.memory_mask[:,:max_length]

    def get_length(self):
        return self.memory_mask.sum(dim=1)

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
        #images, poses, prev_actions = images.squeeze(1), poses.squeeze(1), prev_actions.squeeze(1)

        relative_pose = self.get_relative_poses(poses[:, 0], poses)

        L = images.shape[1]
        embedded_memory = []
        for l in range(L):
            embeddings = [self.embed_network.embed_image(images[:,l]),
                          self.embed_network.embed_act(prev_actions[:,l]),
                          self.embed_network.embed_pose(relative_pose[:,l])]
            embeddings = torch.cat(embeddings, -1)
            embedded_memory.append(self.embed_network.final_embed(embeddings))
        if memory_masks is None:
            embedded_memory = torch.stack(embedded_memory, 1)
        else:
            embedded_memory = torch.stack(embedded_memory, 1) * memory_masks.view(-1,L,1)
        return embedded_memory, embedded_memory[:,0:1], memory_masks

    def embedd_with_pre_embeds(self, pre_embeddings, poses, memory_masks=None):
        L = pre_embeddings.shape[1]
        embedded_memory = []
        relative_pose = self.get_relative_poses(poses[:, 0], poses)
        for l in range(L):
            embeddings = [pre_embeddings[:,l], self.embed_network.embed_pose(relative_pose[:,l])]
            embeddings = torch.cat(embeddings, -1)
            embedded_memory.append(self.embed_network.final_embed(embeddings))
        if memory_masks is None:
            embedded_memory = torch.stack(embedded_memory, 1)
        else:
            embedded_memory = torch.stack(embedded_memory, 1) * memory_masks.view(-1,L,1)
        return embedded_memory, embedded_memory[:,0:1]
