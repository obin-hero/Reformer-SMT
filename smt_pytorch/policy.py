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

    def forward(self, X, Y):
        enc_output, enc_slf_attn = self.att_residual(X,Y,Y)
        H = self.layer_norm(enc_output)
        residual = H
        x = self.fc2(F.relu(self.fc1(H)))
        x = self.dropout(x)
        x += residual
        return x

import time
class Policy(nn.Module):
    def __init__(self,cfg):
        super(Policy, self).__init__()
        self.Encoder = Attblock(cfg.attention.n_head, cfg.attention.d_model, cfg.attention.d_k, cfg.attention.d_v, cfg.attention.dropout)
        self.Decoder = Attblock(cfg.attention.n_head, cfg.attention.d_model, cfg.attention.d_k, cfg.attention.d_v, cfg.attention.dropout)
        self.act = nn.Sequential(nn.Linear(cfg.attention.d_model, cfg.attention.d_model),
                                 nn.ReLU(),
                                 nn.Linear(cfg.attention.d_model, cfg.action_dim))
        self.Memory = SceneMemory(cfg)
        self.Memory.reset_all()

    def cuda(self, device=None):
        super(Policy, self).cuda(device)
        self.Memory.embed_network = self.Memory.embed_network.cuda()

    def run(self, obs, done, mode='train'):
        if mode == 'pretrain':# running with memory collecting
            embedded_memory, curr_embedding = self.Memory.embedd_observations(obs['image'], obs['pose'], obs['prev_action'], done)
            pre_embedding = None
        else:
            embedded_memory, curr_embedding, pre_embedding = self.Memory.update_memory(obs, done)
        C = self.Encoder(embedded_memory, embedded_memory)
        x = self.Decoder(curr_embedding, C)
        x = self.act(x)
        return x, pre_embedding

    def forward_with_obs(self, obs, done): # estimate Q value with given observations
        images, poses, prev_actions = obs
        embedded_memory, curr_embedding = self.Memory.embedd_observations(images, poses, prev_actions, done)
        C = self.Encoder(embedded_memory, embedded_memory)
        x = self.Decoder(curr_embedding, C)
        x = self.act(x)
        x = x.squeeze(1)
        return x#, embedded_memory

    def forward_with_embeddings(self, obs, done): # esitamate Q value with given pre-embedded memory
        batch_pre_embedding, batch_pose = obs
        embedded_memory, curr_embedding = self.Memory.embedd_with_pre_embeds(batch_pre_embedding.cuda(), batch_pose.cuda(), done)
        C = self.Encoder(embedded_memory, embedded_memory)
        x = self.Decoder(curr_embedding, C)
        x = self.act(x)
        x = x.squeeze(1)
        return x#, embedded_memory