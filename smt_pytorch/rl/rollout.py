from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch
import random
import time

from rl.sensors import SensorDict
def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

import os
import torch.nn as nn
import psutil
process = psutil.Process(os.getpid())


class RolloutSensorDictReplayBuffer(object):
    def __init__(self, cfg, num_steps, num_processes, obs_shape, action_space, actor_critic, use_gae, gamma, tau,
                 max_episode_size = 1000, max_episode_step_size = 500, agent_memory_size=None):

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.max_episode_size = max_episode_size
        self.max_episode_step_size = max_episode_step_size
        self.agent_memory_size = agent_memory_size
        self.obs_shape = obs_shape
        self.sensor_names = set(obs_shape.keys())
        # this is for pretraining
        obs_dict = {}
        for k, ob_shape in obs_shape.items():
            if k == 'image':
                obs_dict[k] = torch.zeros(max_episode_size, 500, *ob_shape,dtype=torch.uint8)
            else: obs_dict[k] = torch.zeros(max_episode_size, 500, *ob_shape)
        self.observations = SensorDict(obs_dict)
        self.poses = torch.zeros(max_episode_size, max_episode_step_size, 4, requires_grad=False)
        self.rewards = torch.zeros(max_episode_size, max_episode_step_size, 1, requires_grad=False)
        self.value_preds = torch.zeros(max_episode_size, max_episode_step_size, 1, requires_grad=False)
        self.returns = torch.zeros(max_episode_size, max_episode_step_size, num_processes, 1, requires_grad=False)
        self.action_log_probs = torch.zeros(max_episode_size, max_episode_step_size, 1, requires_grad=False)
        self.actions = torch.zeros(max_episode_size, max_episode_step_size, 1, requires_grad=False)
        self.masks = torch.ones(max_episode_size, max_episode_step_size, 1, requires_grad=False)
        self.pre_embedding_size = 64 + 16 * ('prev_action' in cfg.network.inputs)
        self.pre_embeddings = torch.zeros(max_episode_size, max_episode_step_size, self.pre_embedding_size, requires_grad=False)

        self.actor_critic = actor_critic
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau

        self.num_steps = num_steps
        self.curr_episodes = torch.zeros(max_episode_size, requires_grad=False, dtype=torch.bool)
        self.curr_envs_episodes = np.zeros(num_processes)
        self.curr_steps = torch.zeros(max_episode_size, requires_grad=False, dtype=torch.int32)
        self.memory_occupied = 0

        self.action_dim = action_space.n

    def reset_episode(self, ep):
        for k in self.observations:
            self.observations[k][ep] *= 0.0
        self.poses[ep] *= 0.0
        self.rewards[ep] *= 0.0
        self.value_preds[ep] *= 0.0
        self.returns[ep] *= 0.0
        self.action_log_probs[ep] *= 0.0
        self.actions[ep] *= 0.0
        self.masks[ep] *= 0.0
        self.pre_embeddings[ep] *= 0.0


    def cuda(self):
        # self.observations = self.observations.apply(lambda k, v: v.cuda())
        # self.states = self.states.cuda()
        # self.rewards = self.rewards.cuda()
        # self.value_preds = self.value_preds.cuda()
        # self.returns = self.returns.cuda()
        # self.action_log_probs = self.action_log_probs.cuda()
        # self.actions = self.actions.cuda()
        # self.masks = self.masks.cuda()
        self.actor_critic = self.actor_critic.cuda()
    

    def insert(self, episodes, steps, current_obs, action, action_log_prob, value_pred, reward, mask, mode):
        modules = []
        inputs = []
        for p_num in range(self.num_processes):
            ep_id, step = (episodes[p_num]*self.num_processes + p_num)%self.max_episode_size, steps[p_num]
            if self.curr_episodes[ep_id] and step == 0: self.reset_episode(ep_id)
            modules.extend([self.actions[ep_id, step].copy_,
                           self.action_log_probs[ep_id, step].copy_,
                           self.value_preds[ep_id, step].copy_,
                           self.rewards[ep_id, step].copy_,
                           self.masks[ep_id, step].copy_])
            inputs.extend([action[p_num], action_log_prob[p_num], value_pred[p_num], reward[p_num], mask[p_num]])
            self.curr_episodes[ep_id] = True
            self.curr_steps[ep_id] = step

            if mode == 'train':
                poses, pred_embedding = current_obs
                modules.extend([self.poses[ep_id, step].copy_, self.pre_embeddings[ep_id, step].copy_])
                inputs.extend([poses[p_num], pred_embedding[p_num]])
        nn.parallel.parallel_apply(modules, inputs)
        if mode == 'pretrain':
            for p_num in range(self.num_processes):
                ep_id, step = (episodes[p_num]*self.num_processes + p_num)%self.max_episode_size, steps[p_num]
                modules = ([self.observations[k][ep_id, step].copy_ for k in self.observations])
                inputs = tuple([(current_obs[k].peek()[p_num],) for k in self.observations])
                nn.parallel.parallel_apply(modules, inputs)

        self.curr_envs_episodes = np.maximum(np.array(episodes), self.curr_envs_episodes)


    def get_current_observation(self):
        return self.observations.at(self.step).apply(lambda k, v: v.cuda())

    def get_current_state(self):
        return self.states[self.step].cuda()

    def get_current_mask(self):
        return self.masks[self.step].cuda()

    def after_update(self):
        pass


    def feed_forward_generator(self, num_mini_batch, on_policy=True, mode='train'):
        mini_batch_size = self.num_steps // num_mini_batch

        saved_episodes = self.curr_episodes.sum()
        if on_policy or saved_episodes < self.max_episode_size:
            # get trajectory from most recent episodes
            episode_id = self.curr_episodes.sum() - 1
            step_id = self.curr_steps[episode_id]
        else:
            saved_ep_inds = torch.where(self.curr_episodes)[0]
            while True:
                episode = np.random.choice(saved_ep_inds)
                step_id = np.random.choice(1, self.curr_steps[episode])

        if step_id >= self.num_steps:
            start_idx = (step_id - self.num_steps)
            step_indicies = [[episode_id, start_idx + i] for i in range(self.num_steps)]
        else:
            step_indicies = [[episode_id, i] for i in range(step_id)]
            for ep in range(episode_id-1,0,-1):
                if len(step_indicies) >= self.num_steps:
                    step_indicies = step_indicies[:self.num_steps+1]
                    break
                step = self.curr_steps[ep]
                step_indicies.extend([[ep, i] for i in range(step)])

        observations_sample = SensorDict(
            {k: torch.zeros(self.num_steps + 1, *ob_shape) for k, ob_shape in
             self.obs_shape.items()}).apply(lambda k, v: v.cuda())
        embeddings_sample = torch.zeros(self.num_steps + 1, self.agent_memory_size, self.pre_embedding_size).cuda()
        poses_sample = torch.zeros(self.num_steps+1, self.agent_memory_size, 4)
        rewards_sample = torch.zeros(self.num_steps, 1).cuda()
        values_sample = torch.zeros(self.num_steps + 1, 1).cuda()
        returns_sample = torch.zeros(self.num_steps + 1, 1).cuda()
        action_log_probs_sample = torch.zeros(self.num_steps, 1).cuda()
        actions_sample = torch.zeros(self.num_steps, 1).cuda()
        masks_sample = torch.ones(self.num_steps + 1, 1).cuda()

        # Fill the buffers and get values
        sample_idx = 0
        for step_info in step_indicies[:-1]:
            sample_ep, sample_step = step_info
            memory_start = max(sample_step + 1 - self.agent_memory_size, 0)
            memory_size = sample_step - memory_start
            if mode == 'pretrain':
                for k in self.observations:
                    observations_sample[k][sample_idx] = self.observations[k][sample_ep, sample_step].cuda()
                with torch.no_grad():
                    values_sample[sample_idx] = self.actor_critic.get_value(self.observations.dim2_att(sample_ep, [memory_start, sample_step+1]).apply(lambda k, v: v.cuda()), self.masks[sample_ep, sample_step].cuda(), mode)
            else:
                embeddings_sample[sample_idx, :memory_size+1] = self.pre_embeddings[sample_ep, memory_start:sample_step+1]
                poses_sample[sample_idx, :memory_size+1] = self.poses[sample_ep, memory_start:sample_step+1]
                with torch.no_grad():
                    embed_obs = (self.pre_embeddings[sample_ep, memory_start:sample_step+1].unsqueeze(0).cuda(),
                                 self.poses[sample_ep, memory_start:sample_step+1].unsqueeze(0).cuda())
                    values_sample[sample_idx] = self.actor_critic.get_value(embed_obs, self.masks[sample_ep, sample_step].cuda(), mode)
            rewards_sample[sample_idx] = self.rewards[sample_ep, sample_step].cuda()
            action_log_probs_sample[sample_idx] = self.action_log_probs[sample_ep, sample_step].cuda()
            actions_sample[sample_idx] = self.actions[sample_ep, sample_step].cuda()
            masks_sample[sample_idx] = self.masks[sample_ep, sample_step].cuda()
            sample_idx += 1

        # we need to compute returns and advantages on the fly, since we now have updated value function
        with torch.no_grad():
            sample_ep, sample_step = step_indicies[-1]
            memory_start = max(sample_step + 1 - self.agent_memory_size, 0)
            if mode == 'pretrain':
                next_value = self.actor_critic.get_value(
                    self.observations.dim2_att(sample_ep, [memory_start, sample_step+1]).apply(lambda k, v: v.cuda()),self.masks[sample_ep, sample_step].cuda(),mode)
            else :
                embed_obs = (self.pre_embeddings[sample_ep, memory_start:sample_step + 1].unsqueeze(0).cuda(),
                             self.poses[sample_ep, memory_start:sample_step + 1].unsqueeze(0).cuda())
                next_value = self.actor_critic.get_value(embed_obs, self.masks[sample_ep, sample_step+1].cuda(), mode)
        if self.use_gae:
            values_sample[-1] = next_value
            gae = 0
            for step in reversed(range(rewards_sample.size(0))):
                delta = rewards_sample[step] + self.gamma * values_sample[step + 1] * masks_sample[step + 1] - \
                        values_sample[step]
                gae = delta + self.gamma * self.tau * masks_sample[step + 1] * gae
                returns_sample[step] = gae + values_sample[step]

        observations_batch = {}
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_steps)), mini_batch_size, drop_last=False)
        advantages = returns_sample[:-1] - values_sample[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        for indices in sampler:
            if mode == 'pretrain':
                for k, sensor_ob in observations_sample.items():
                    observations_batch[k] = sensor_ob[:-1][indices]
            else:
                embeddings_batch = embeddings_sample[indices]
                poses_batch = poses_sample[indices]
                observations_batch = [embeddings_batch, poses_batch]
            actions_batch = actions_sample[indices]
            return_batch = returns_sample[indices]
            masks_batch = masks_sample[indices]
            old_action_log_probs_batch = action_log_probs_sample[indices]
            adv_targ = advantages[indices]
            yield observations_batch, actions_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ




class RolloutSensorDictStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size):
        '''
            num_steps:
            num_processes: number of parallel rollouts to store
            obs_shape: Dict from sensor_names -> sensor_obs_shape
            action_space:
            state_size: Internal state size
        '''
        self.sensor_names = set(obs_shape.keys())
        self.observations = SensorDict({
            k: torch.zeros(num_steps + 1, num_processes, *ob_shape)
            for k, ob_shape in obs_shape.items()
        })
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def cuda(self):
        self.observations = self.observations.apply(lambda k, v: v.cuda())
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        modules = [self.observations[k][self.step + 1].copy_ for k in self.observations]
        inputs = tuple([(current_obs[k].peek(),) for k in self.observations])
        nn.parallel.parallel_apply(modules, inputs)
        # for k in self.observations:
        # self.observations[k][self.step + 1].copy_(current_obs[k].peek())
        self.states[self.step + 1].copy_(state)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def get_current_observation(self):
        return self.observations.at(self.step)

    def get_current_state(self):
        return self.states[self.step]

    def get_current_mask(self):
        return self.masks[self.step]

    def after_update(self):
        for k in self.observations:
            self.observations[k][0].copy_(self.observations[k][-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                        self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                                     gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"* number of steps ({num_steps}) = {num_processes * num_steps} "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
        mini_batch_size = batch_size // num_mini_batch
        observations_batch = {}
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            for k, sensor_ob in self.observations.items():
                observations_batch[k] = sensor_ob[:-1].view(-1, *sensor_ob.size()[2:])[indices]
            states_batch = self.states[:-1].view(-1, self.states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            yield observations_batch, states_batch, actions_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)
            states_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                for k, sensor_ob in self.observations.items():
                    observations_batch[k].append(sensor_ob[:-1, ind])
                states_batch.append(self.states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            for k, v in observations_batch.items():
                observations_batch[k] = torch.stack(observations_batch[k], 1)
            actions_batch = torch.stack(actions_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            states_batch = torch.stack(states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for k, sensor_ob in observations_batch.items():
                observations_batch[k] = _flatten_helper(T, N, sensor_ob)
            actions_batch = _flatten_helper(T, N, actions_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                                                         old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield SensorDict(observations_batch), \
                  states_batch, actions_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ
class RolloutTensorStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        self.observations[self.step + 1].copy_(current_obs)
        self.states[self.step + 1].copy_(state)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                        self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                                     gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"* number of steps ({num_steps}) = {num_processes * num_steps} "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            observations_batch = self.observations[:-1].view(-1,
                                                             *self.observations.size()[2:])[indices]
            states_batch = self.states[:-1].view(-1, self.states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield observations_batch, states_batch, actions_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = []
            states_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                observations_batch.append(self.observations[:-1, ind])
                states_batch.append(self.states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            observations_batch = torch.stack(observations_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            states_batch = torch.stack(states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            observations_batch = _flatten_helper(T, N, observations_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                                                         old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield observations_batch, states_batch, actions_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ
