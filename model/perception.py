import torch.nn as nn
import torch.nn.functional as F
from model.attention import MultiHeadAttention
from model.memory.memory import SceneMemory
class Attblock(nn.Module):
    def __init__(self,n_head, d_model, d_k, d_v, dropout=0.1):
        super(Attblock, self).__init__()
        self.att_residual = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y, mask=None):
        enc_output, attn = self.att_residual(X,Y,Y, mask)
        H = self.layer_norm(enc_output)
        residual = H
        x = self.fc2(F.relu(self.fc1(H)))
        x = self.dropout(x)
        x += residual
        return x, attn


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
        embedded_demo, _, demo_masks = self.Memory.embedd_observations(obs['demo_im'],obs['demo_pose'],obs['demo_act'])
        embedded_memory, curr_embedding, pre_embedding, memory_masks = self.Memory.update_memory(obs, masks)
        C = self.Encoder(embedded_demo, embedded_demo, demo_masks)
        x = self.Decoder(embedded_memory, C, memory_masks.unsqueeze(1))
        return x.squeeze(1), pre_embedding

    def forward(self, observations, memory_masks=None, mode='train'): # without memory
        observations['image'] = observations['image'].float() / 255.0 * 2 - 1.0
        images, poses, prev_actions = observations['image'], observations['pose'], observations['prev_action']
        embedded_memory, curr_embedding, _ = self.Memory.embedd_observations(images, poses, prev_actions, memory_masks)
        demo_im, demo_pose, demo_action = observations['demo_im'], observations['demo_pose'], observations['demo_act']
        embedded_demo, _ , demo_masks = self.Memory.embedd_observations(demo_im, demo_pose, demo_action)
        if memory_masks is not None:
            memory_masks = memory_masks.unsqueeze(1)
        C = self.Encoder(embedded_demo, embedded_demo, demo_masks)
        x = self.Decoder(embedded_memory, C, memory_masks)
        return x.squeeze(1)

from model.memory.memory import Embedding
import torch
class IL_Perception(nn.Module):
    def __init__(self,cfg):
        super(IL_Perception, self).__init__()
        self.action_dim = cfg.action_dim
        self.Encoder = Attblock(cfg.attention.n_head, cfg.attention.d_model, cfg.attention.d_k, cfg.attention.d_v, cfg.attention.dropout)
        self.Decoder_slf = Attblock(cfg.attention.n_head, cfg.attention.d_model, cfg.attention.d_k, cfg.attention.d_v, cfg.attention.dropout)
        self.Decoder_crs = Attblock(cfg.attention.n_head, cfg.attention.d_model, cfg.attention.d_k, cfg.attention.d_v, cfg.attention.dropout)
        self.output_size = cfg.attention.d_model
        self.Embedding = Embedding(cfg)

    def embed_demo(self, demo_im, demo_act, mask):
        self.embedded_demo = self.Embedding(demo_im, demo_act, mask)
        self.C, _ = self.Encoder(self.embedded_demo, self.embedded_demo, mask)
        return self.embedded_demo, self.C

    def forward(self, curr_im, prev_info=None, agent_embedded_memory=None, mask=None):
        B = curr_im.shape[0]
        if prev_info is not None:
            prev_im_embed, prev_act = prev_info
            prev_act_embed = self.Embedding.embed_act(prev_act)
            prev_embedding = self.Embedding.final_embed(torch.cat((prev_im_embed, prev_act_embed),1)).unsqueeze(1)
        else:
            prev_act = torch.zeros((B,self.action_dim)).cuda()

        embedded_curr_im = self.Embedding.embed_image(curr_im)
        blank_act = self.Embedding.embed_act(torch.zeros_like(prev_act))
        embedded_curr_state = self.Embedding.final_embed(torch.cat((embedded_curr_im, blank_act),1)).unsqueeze(1)

        if agent_embedded_memory is not None :
            agent_embedded_memory = torch.cat((prev_embedding, agent_embedded_memory), 1)
            curr_state_memory = torch.cat((embedded_curr_state, agent_embedded_memory),1)
        elif prev_info is not None:
            agent_embedded_memory = prev_embedding
            curr_state_memory = torch.cat((embedded_curr_state, agent_embedded_memory),1)
        else:
            curr_state_memory = embedded_curr_state
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, slf_attn = self.Decoder_slf(curr_state_memory, curr_state_memory, mask)
        x, crs_attn = self.Decoder_crs(x, self.C, mask)
        return x.sum(1), embedded_curr_im, agent_embedded_memory, slf_attn, crs_attn

