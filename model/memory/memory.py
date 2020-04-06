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
        self.action_dim = cfg.action_dim
        self.embed_image = resnet18(first_ch=3*cfg.network.num_stack, num_classes=256)
        self.embed_act = nn.Linear(self.action_dim,16)

        final_ch = 256 + 16
        self.final_embed = nn.Sequential(nn.Linear(final_ch, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, 128))

    def forward(self, imgs, acts, masks):
        length = masks.sum(dim=1)
        max_length = int(length.max())
        embeddings = []
        for i in range(max_length):
            img_embedding = self.embed_image(imgs[:,i])
            act_embedding = self.embed_act(acts[:,i])
            embeds = self.final_embed(torch.cat((img_embedding, act_embedding),1))
            embeddings.append(embeds)
        embeddings = torch.stack((embeddings),1)
        return embeddings

class SceneMemory(nn.Module):
    # B * M (max_memory_size) * E (embedding)
    def __init__(self, cfg) -> None:
        super(SceneMemory, self).__init__()
        self.B = cfg.training.num_envs + cfg.training.valid_num_envs
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
        self.gt_pose_buffer = self.gt_pose_buffer * reset_mask.view(-1,1,1).float()

    def reset_all(self) -> None:
        embedding_size_wo_pose = 64 + 16 * (self.embed_network.use_action)
        self.memory_buffer = torch.zeros([self.B, self.max_memory_size, embedding_size_wo_pose],dtype=torch.float32).cuda()
        self.memory_mask = torch.zeros([self.B, self.max_memory_size], dtype=torch.bool).cuda()
        self.gt_pose_buffer = torch.zeros([self.B, self.max_memory_size, 4],
                                          dtype=torch.float32).cuda() if self.embed_network.use_pose else None

    def update_memory(self, obs, masks):
        if (masks == False).any(): self.reset(masks)
        new_embeddings = []
        new_embeddings.append(self.embed_network.embed_image(obs['image']))
        new_embeddings.append(self.embed_network.embed_act(obs['prev_action']))
        new_embedding = torch.cat(new_embeddings,1)

        self.memory_buffer = torch.cat([new_embedding.unsqueeze(1), self.memory_buffer[:, :-1]], 1)
        #self.memory_buffer[:, 0, -1] = obs['pose'][:, -1]
        #self.memory_buffer[:, 0, -2] = obs['episode'].squeeze()
        self.memory_mask = torch.cat([torch.ones_like(masks, dtype=torch.bool), self.memory_mask[:,:-1]],1)
        self.gt_pose_buffer = torch.cat([(obs['pose']).unsqueeze(1),self.gt_pose_buffer[:,:-1]],1)

        length = self.memory_mask.sum(dim=1)
        max_length = int(length.max())
        relative_poses = self.get_relative_poses(obs['pose'], self.gt_pose_buffer[:,:max_length])
        embedded_memory = []
        for i in range(max_length):
            embedded_pose = self.embed_network.embed_pose(relative_poses[:,i])
            # for debug
            #embedded_pose[:,-1] = obs['pose'][:,-1]
            #embedded_pose[:,-2] = obs['episode'].squeeze()
            embedded_memory.append(self.embed_network.final_embed(torch.cat((self.memory_buffer[:,i], embedded_pose),1)))
        embedded_memory = torch.stack(embedded_memory,1) * self.memory_mask[:, :max_length].unsqueeze(-1).float()

        #pad_length = self.max_memory_size - max_length
        #embedded_memory = torch.cat((embedded_memory, torch.zeros([embedded_memory.shape[0], pad_length, *embedded_memory.shape[2:]]).to(embedded_memory.device)),1)
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
        rel_x = del_x * torch.cos(th) - del_y * torch.sin(th)
        rel_y = del_x * torch.sin(th) + del_y * torch.cos(th)
        rel_yaw = gt_pose_yaw - curr_pose_yaw
        exp_t = torch.exp(-(prev_poses[:,0:1,3] - prev_poses[:,:,3]))

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
