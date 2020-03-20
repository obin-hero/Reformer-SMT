import torch
import numpy as np
from torch.distributions.uniform import Uniform
from collections import deque
class ReplayBuffer(object):
    def __init__(self, cfg) -> None:
        self.max_episodes = cfg.replay_buffer.max_episode
        self.max_memory_size = cfg.training.max_memory_size
        self.max_steps = cfg.training.max_step
        self.mode = 'pretrain' # 'train'
        self.action_dim = cfg.action_dim

        # when pretraining
        self.pretrain_buffer = deque(maxlen=self.max_episodes*self.max_steps)
        # after embedding network is fixed
        pre_embedding_size = 64
        pre_embedding_size += 16 * ('prev_action' in cfg.network.inputs)
        self.pre_embeddings_buffer = torch.zeros([self.max_episodes, self.max_steps, pre_embedding_size], dtype=torch.float32)
        self.pose_buffer = torch.zeros([self.max_episodes, self.max_steps, 4], dtype=torch.float32)
        self.action_buffer = torch.zeros([self.max_episodes, self.max_steps, cfg.action_dim], dtype=torch.bool)
        self.reward_buffer = torch.zeros([self.max_episodes, self.max_steps, 1], dtype=torch.float32)
        self.done_buffer = torch.zeros([self.max_episodes, self.max_steps, 1], dtype=torch.bool)
        self.curr_episode = 0
    def size(self):
        if self.mode == 'pretrain': return len(self.pretrain_buffer)
        else: return self.curr_episode

    def add(self, experience, episode, step) -> None:
        episode = episode % self.max_episodes
        if self.mode == 'pretrain':
            self.pretrain_buffer.append(experience)
        elif self.mode == 'train':
            new_embeddings, pose, action, reward, done = experience
            self.pre_embeddings_buffer[episode, step].copy_(new_embeddings.squeeze())
            self.pose_buffer[episode, step].copy_(pose.squeeze())
            self.action_buffer[episode, step].copy_(torch.tensor(action))
            self.reward_buffer[episode, step].copy_(reward)
            self.done_buffer[episode, step].copy_(done)
            self.curr_episode = min((self.curr_episode + 1), self.max_episodes)
    def sample(self, batch_size: int, continuous: bool = True):

        if self.mode == 'pretrain':
            if batch_size > len(self.pretrain_buffer): batch_size = len(self.pretrain_buffer)
            indicies = np.random.choice(np.arange(len(self.pretrain_buffer)), size=batch_size, replace=False)
            images = []
            poses = []
            prev_actions = []
            next_images = []
            next_poses = []
            next_prev_actions = []
            actions = []
            rewards = []
            dones = []
            for i in indicies:
                obs, next_obs, action, reward, done = self.pretrain_buffer[i]
                images.append(obs['image'])
                poses.append(obs['pose'])
                prev_actions.append(obs['prev_action'])
                next_images.append(next_obs['image'])
                next_poses.append(next_obs['pose'])
                next_prev_actions.append(next_obs['prev_action'])
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

            images = torch.cat(images).cuda()
            poses = torch.cat(poses).cuda()
            prev_actions = torch.cat(prev_actions).cuda()
            next_images = torch.cat(next_images).cuda()
            next_poses = torch.cat(next_poses).cuda()
            next_prev_actions = torch.cat(next_prev_actions).cuda()
            actions = torch.tensor(actions).cuda().view(-1,1)
            rewards = torch.cat(rewards).cuda()
            dones = torch.cat(dones).cuda()
            return (images, poses, prev_actions), (next_images, next_poses, next_prev_actions), actions, rewards, dones
        elif self.mode == 'train':
            if batch_size > self.curr_episode:
                batch_size = self.curr_episode
            indicies = np.random.choice(np.arange(self.curr_episode), size=batch_size, replace=False)
            batch_pre_embedding = []
            batch_pose = []
            batch_action = []
            batch_reward = []
            batch_done =[]
            ts = Uniform(0, (~self.done_buffer[indicies, :]).sum(dim=1).float()).sample().squeeze(1).int()
            max_t = min(ts.max(), self.max_memory_size)
            for ii, ep in enumerate(indicies):
                ts_ii = int(ts[ii])
                mem_init_t = max(0, ts_ii-self.max_memory_size)
                padd_length = max(0, int(max_t - ts_ii))
                # return pre_embedding_(0:t+1), pose_(0:t+1), action_(t), reward_(t), done
                pre_embedding = torch.cat((self.pre_embeddings_buffer[ep,mem_init_t:ts_ii],
                                           torch.zeros([padd_length, self.pre_embeddings_buffer.shape[-1]])),0)
                batch_pre_embedding.append(pre_embedding.unsqueeze(0))
                poses = torch.cat((self.pose_buffer[ep,mem_init_t:ts_ii],
                                   torch.zeros([padd_length, self.pose_buffer.shape[-1]])),0)
                batch_pose.append(poses.unsqueeze(0))
                batch_action.append(self.action_buffer[ep, ts_ii].unsqueeze(0))
                batch_reward.append(self.reward_buffer[ep, ts_ii].unsqueeze(0))
                batch_done.append(self.done_buffer[ep, ts_ii].unsqueeze(0))

            batch_pre_embedding = torch.cat(batch_pre_embedding,0).cuda()
            batch_pose = torch.cat(batch_pose,0).cuda()
            batch_action = torch.cat(batch_action,0).cuda()
            batch_reward = torch.cat(batch_reward,0).cuda()
            batch_done = torch.cat(batch_done,0).cuda()
            return batch_pre_embedding, batch_pose, batch_action, batch_reward, batch_done
        #batch_state, batch_next_state, batch_action, batch_reward, batch_done = batch
    def clear(self):
        if self.mode == 'pretrain':
            self.buffer.clear()
        elif self.mode == 'train':
            self.pre_embeddings_buffer *= 0
            self.pose_buffer *= 0
            self.action_buffer *= 0
            self.reward_buffer *= 0
            self.done_buffer *= 0
