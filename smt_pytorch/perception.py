import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from memory import SceneMemory
class Attblock(nn.Module):
    def __init__(self,n_head, d_model, d_k, d_v, dropout=0.1):
        super(Attblock, self).__init__()
        self.att_residual = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y, mask=None):
        enc_output, enc_slf_attn = self.att_residual(X,Y,Y, mask)
        H = self.layer_norm(enc_output)
        residual = H
        x = self.fc2(F.relu(self.fc1(H)))
        x = self.dropout(x)
        x += residual
        return x

import time
class Perception(nn.Module):
    def __init__(self,cfg):
        super(Perception, self).__init__()
        self.Encoder = Attblock(cfg.attention.n_head, cfg.attention.d_model, cfg.attention.d_k, cfg.attention.d_v, cfg.attention.dropout)
        self.Decoder = Attblock(cfg.attention.n_head, cfg.attention.d_model, cfg.attention.d_k, cfg.attention.d_v, cfg.attention.dropout)
        self.output_size = cfg.attention.d_model
        self.Memory = SceneMemory(cfg)
        self.Memory.reset_all()

    def cuda(self, device=None):
        super(Perception, self).cuda(device)
        self.Memory.embed_network = self.Memory.embed_network.cuda()

    def act(self, obs, masks, mode='train'): # with memory
        obs['image'] = obs['image'] / 255.0 * 2 - 1.0
        if mode == 'pretrain':# running with memory collecting
            embedded_memory, curr_embedding, memory_masks = self.Memory.embedd_observations(obs['image'].cuda(), obs['pose'].cuda(), obs['prev_action'].cuda(), masks)
            pre_embedding = None
        else:
            embedded_memory, curr_embedding, pre_embedding, memory_masks = self.Memory.update_memory(obs, masks)
        C = self.Encoder(embedded_memory, embedded_memory, memory_masks.unsqueeze(1))
        x = self.Decoder(curr_embedding, C, memory_masks.unsqueeze(1))
        return x.squeeze(1), pre_embedding


    def forward(self, observations, memory_masks=None, mode='train'): # without memory
        if mode == 'pretrain':
            observations['image'] = observations['image'].float() / 255.0 * 2 - 1.0
            images, poses, prev_actions = observations['image'], observations['pose'], observations['prev_action']
            embedded_memory, curr_embedding, _ = self.Memory.embedd_observations(images, poses, prev_actions, memory_masks)
        else:
            batch_pre_embedding, batch_pose = observations
            embedded_memory, curr_embedding = self.Memory.embedd_with_pre_embeds(batch_pre_embedding.cuda(),batch_pose.cuda(), memory_masks)
        if memory_masks is not None:
            memory_masks = memory_masks.unsqueeze(1)
        C = self.Encoder(embedded_memory, embedded_memory, memory_masks)
        x = self.Decoder(curr_embedding, C, memory_masks)
        return x.squeeze(1)
