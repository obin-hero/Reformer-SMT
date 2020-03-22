import gym
import gym_deepmindlab
import matplotlib.pyplot as plt
import cv2
import time
import random
from default_cfg import get_config
from tensorboardX import SummaryWriter
cfg = get_config()

import torch
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
from perception import Policy
from torch.distributions.categorical import Categorical
from imageio import get_writer, imread
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
onlineQNetwork = Policy(cfg)
onlineQNetwork.cuda()
targetQNetwork = Policy(cfg)
targetQNetwork.cuda()
targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

replay_buffer = ReplayBuffer(cfg)

optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)
GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 64
UPDATE_STEPS = 4
epsilon = INITIAL_EPSILON
# epsilon = 0
learn_steps = 0
writer = SummaryWriter('logs')
begin_learn = False

episode_reward = 0
previous_episode_reward = episode_reward

VIS_INTERVAL = 100

from deeplab_env import DeepmindLabEnv, TensorStackWrapper
from multi_env import VectorEnv
from multi_env import _make_env_fn
envs = VectorEnv(_make_env_fn, [{'scene': 'nav_max_static_01', 'rank': i} for i in range(3)])
obs = envs.reset()
print(len(obs))
for i in range(obs):
    print(i.keys())
    print(i['image'].shape)
env = DeepmindLabEnv('nav_maze_static_01', max_step=cfg.training.max_step)
env = TensorStackWrapper(env, k=cfg.network.num_stack)
mode = 'pretrain'
replay_buffer.mode = mode
DEBUG_TIME = False
def print_time(log, start):
    print(log, time.time() - start)
    return time.time()
start_time = 0
for episode_id in range(50000):
    if episode_id > 10000:
        onlineQNetwork.Memory.freeze_embedding_network()
        targetQNetwork.Memory.freeze_embedding_network()
        mode = 'train'
    obs = env.reset()
    episode_reward = 0
    done = torch.tensor([False])
    onlineQNetwork.Memory.reset_all()
    s = time.time()
    if episode_id % VIS_INTERVAL == 0 :
        view_imgs = []
    step = 0
    step_time = 0
    while not done:
        p = random.random()
        with torch.no_grad():
            action, new_embeddings = onlineQNetwork.run(obs,done, mode=mode)
        action = torch.nn.functional.softmax(action)
        action = int(Categorical(action).sample().squeeze())
        if DEBUG_TIME: s = print_time('run 1step', s)
        real_env_time = time.time()
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        if DEBUG_TIME: s = print_time('env step', s)
        if mode == 'pretrain':
            replay_buffer.add((obs, next_obs, action, reward, done), episode_id, env.env.time_t-1)
        elif mode == 'train':
            replay_buffer.add((new_embeddings, obs['pose'], action, reward, done), episode_id, env.env.time_t-1)
        obs = next_obs
        if DEBUG_TIME: s = print_time('buffer add', s)
    if replay_buffer.size() > 500:
        if begin_learn is False:
            print('learn begin!')
            begin_learn = True
        for i in range(64):
            learn_steps += 1
            if learn_steps % UPDATE_STEPS == 0:
                targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
            batch = replay_buffer.sample(BATCH, False)
            if DEBUG_TIME: s = print_time('get batch', s)
            if mode == 'pretrain': # with memory size == 1
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = batch
                with torch.no_grad():
                    onlineQ_next = onlineQNetwork.forward_with_obs(batch_next_state, batch_done)
                    targetQ_next = targetQNetwork.forward_with_obs(batch_next_state, batch_done)
                x = onlineQNetwork.forward_with_obs(batch_state, batch_done).squeeze(1).gather(1, batch_action.long())
            elif mode == 'train':
                batch_next_pre_embedding, batch_next_pose, batch_action, batch_reward, batch_done = batch
                batch_pre_embedding, batch_pose = batch_next_pre_embedding[:, :-1], batch_next_pose[:, :-1]
                batch_next_state = (batch_next_pre_embedding, batch_next_pose)
                batch_state = batch_pre_embedding, batch_pose
                with torch.no_grad():
                    onlineQ_next = onlineQNetwork.forward_with_embeddings(batch_next_state, batch_done).squeeze(1)
                    targetQ_next = targetQNetwork.forward_with_embeddings(batch_next_state, batch_done).squeeze(1)
                x = onlineQNetwork.forward_with_embeddings(batch_state, batch_done).gather(1, batch_action.long())
            online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
            y = batch_reward + (1 - batch_done.float()).view(-1,1) * GAMMA * targetQ_next.gather(1, online_max_action.long())
            loss = F.mse_loss(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), global_step=learn_steps)
        if DEBUG_TIME: s = print_time('update', s)

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    step += 1
    if episode_id % VIS_INTERVAL == 0:
        video_writer = get_writer('test.mp4', fps=20)
        for im in view_imgs:
            video_writer.append_data(im)
        video_writer.close()
    writer.add_scalar('episode reward', episode_reward, global_step=episode_id)
    if episode_id % 10 == 0:
        torch.save(onlineQNetwork.state_dict(), 'doom-policy.para')
        print('Ep {}\tMoving average score: {:.2f} time : {}\t'.format(episode_id, episode_reward.item(), time.time()-start_time),step)
        start_time = time.time()






