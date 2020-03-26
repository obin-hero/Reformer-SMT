import os
import sys
from default_cfg import get_config
cfg = get_config()

devices = ",".join(str(e) for e in cfg.training.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = devices
import torch
torch.backends.cudnn.benchmark=True
import numpy as np
import time
from gym import logger
from ob_utils import mkdir
## ---------------------------------
from multi_env import vizdoom_make_env_fn, PreprocessVectorEnv
from vizdoom_env import VizDoomMultiWrapper, VizDoomEnv
from rl.stackedobservation import StackedSensorDictStorage
import tnt.torchnet as tnt
import torchvision
from rl.rollout import RolloutSensorDictReplayBuffer
from rl.preprocessingwrapper import EncodeObservatonWrapper
from rl.actor_critic_module import ActorCriticModule
from Policy import PolicyWithBase, RecurrentPolicyWithBase
from rl.ppo_replay import PPOReplay
from perception import Perception
## ----------------------------------

def save_network(net, file_name):
    torch.save(net.state_dict(),file_name)
def log_input_images(obs_unpacked, mlog, num_stack, key_names=['map'], meter_name='debug/input_images', step_num=0, env=None):
    # Plots the observations from the first process
    stacked = []
    for key_name in key_names:
        if key_name not in obs_unpacked:
            logger.debug(key_name, "not found")
            continue
        obs = obs_unpacked[key_name][0]
        #obs = (obs + 1.0) / 2.0
        obs = obs/255.
        obs_chunked = list(torch.chunk(obs, num_stack, dim=0))
        key_stacked = torchvision.utils.make_grid(obs_chunked, nrow=num_stack, padding=2)
        stacked.append(key_stacked)
    if len(stacked) == 0 : return
    stacked = torch.cat(stacked, dim=2)
    mlog.update_meter(stacked, meters={meter_name})
    mlog.reset_meter(step_num, meterlist={meter_name})


debug_time = False
def main(cfg):
    # save config and codes
    model_name = cfg.saving.version
    save_dir = os.path.join(cfg.saving.save_dir, 'saved_networks', model_name)
    image_dir = os.path.join(cfg.saving.save_dir, 'images', model_name)
    log_dir = os.path.join(cfg.saving.save_dir, 'logs/EXPLORATION', model_name)
    mkdir(save_dir)
    mkdir(image_dir)
    mkdir(log_dir)

    visdom_log_file = None if cfg.visdom.log_file == 'none' else cfg.visdom.log_file
    visdom_name, vis_interval = cfg.visdom.name, cfg.visdom.vis_interval
    visdom_server, visdom_port = cfg.visdom.server, cfg.visdom.port

    envs = PreprocessVectorEnv(vizdoom_make_env_fn,
                                  env_fn_args=[(cfg,
                                                i,
                                               log_dir,
                                               visdom_name,
                                               visdom_log_file,
                                               vis_interval,
                                               visdom_server,
                                               visdom_port) for i in range(cfg.training.num_envs)])
    envs = VizDoomMultiWrapper(envs)

    perception_model = Perception(cfg)
    actor_critic = RecurrentPolicyWithBase(perception_model, action_space=envs.action_space)
    agent = PPOReplay(actor_critic=actor_critic,
                      clip_param=cfg.RL.CLIP_PARAM,
                      ppo_epoch=cfg.RL.PPO_EPOCH,
                      num_mini_batch=cfg.RL.NUM_MINI_BATCH,
                      value_loss_coef=cfg.RL.VALUE_LOSS_COEF,
                      entropy_coef=cfg.RL.ENTROPY_COEF,
                      on_policy_epoch=cfg.RL.ON_POLICY_EPOCH,
                      off_policy_epoch=cfg.RL.OFF_POLICY_EPOCH,
                      lr=cfg.training.lr,
                      eps=cfg.RL.EPS,
                      max_grad_norm=cfg.RL.MAX_GRAD_NORM,
                      )
    actor_critic.cuda()
    actor_critic.train()


    if cfg.training.resume != 'none':
        state_dict = torch.load(os.path.join(save_dir, cfg.training.resume))
        actor_critic.load_state_dict(state_dict)
        print('loaded {}'.format(cfg.training.resume))
    if cfg.training.pretrain_load != 'none':
        state_dict = torch.load(os.path.join(save_dir, cfg.training.pretrain_load))
        actor_critic.preception_unit.Memory.embed_network.load_state_dict(state_dict)
        print('loaded {}'.format(cfg.training.pretrain_load))

    uuid = cfg.saving.version
    mlog = tnt.logger.TensorboardMeterLogger(env=uuid,
                                            log_dir=log_dir,
                                            plotstylecombined=True)

    loggable_metrics = ['metrics/rewards',
                        'diagnostics/dist_perplexity',
                        'diagnostics/lengths',
                        'diagnostics/max_importance_weight',
                        'diagnostics/value',
                        'losses/action_loss',
                        'losses/dist_entropy',
                        'losses/value_loss']
    core_metrics = ['metrics/rewards', 'diagnostics/lengths']
    debug_metrics = ['debug/input_images']

    for metric in ['metrics/visited_cells']:
        loggable_metrics.append(metric)
        core_metrics.append(metric)
    for meter in loggable_metrics:
        mlog.add_meter(meter, tnt.meter.ValueSummaryMeter())
    for debug_meter in debug_metrics:
        mlog.add_meter(debug_meter, tnt.meter.SingletonMeter(), ptype='image')

    flog = tnt.logger.FileLogger(os.path.join(log_dir, cfg.RL.RESULTS_LOG_FILE), overwrite=True)
    reward_only_flog = tnt.logger.FileLogger(os.path.join(log_dir, cfg.RL.REWARD_LOG_FILE), overwrite=True)


    # cfg training resume

    observation_space = envs.observation_space
    retained_obs_shape = {k: v.shape
                          for k, v in observation_space.items()
                          if k in cfg.network.inputs}
    total_processes = cfg.training.num_envs
    current_obs = StackedSensorDictStorage(total_processes, cfg.RL.NUM_STACK, retained_obs_shape)
    current_train_obs = StackedSensorDictStorage(total_processes, cfg.RL.NUM_STACK, retained_obs_shape)
    current_obs.cuda()
    current_train_obs.cuda()

    num_train_processes = cfg.training.num_envs
    episode_rewards = torch.zeros([num_train_processes, 1])
    episode_lengths = torch.zeros([num_train_processes, 1])

    # First observation
    obs = envs.reset()
    current_obs.insert(obs)
    mask_done = torch.FloatTensor([[1.0] for _ in range(num_train_processes)]).pin_memory()
    states = torch.zeros(num_train_processes, 128).pin_memory()

    assert cfg.replay_buffer.max_episode % num_train_processes == 0
    rollouts = RolloutSensorDictReplayBuffer(cfg,
                                             cfg.RL.NUM_STEPS,
                                             num_train_processes,
                                             current_obs.obs_shape,
                                             envs.action_space,
                                             actor_critic,
                                             cfg.RL.USE_GAE,
                                             cfg.RL.GAMMA,
                                             cfg.RL.TAU,
                                             cfg.replay_buffer.max_episode,
                                             cfg.training.max_step,
                                             cfg.training.max_memory_size)

    # Main loop
    start_time = time.time()
    num_updates = int(cfg.RL.NUM_FRAMES) // cfg.RL.NUM_STEPS * num_train_processes
    abs_time = 0
    start_epoch = 0
    training_mode = 'pretrain'
    rollouts.agent_memory_size = 1
    for epoch in range(start_epoch, num_updates, 1):
        if epoch > cfg.training.pretrain_epoch and training_mode == 'pretrain':
            training_mode = 'train'
            rollouts.agent_memory_size = cfg.training.max_memory_size
            #actor_critic.perception_unit.Memory.freeze_embedding_network()
            agent.change_optimizer()
            print('changed training mode')
            save_network(actor_critic.perception_unit.Memory.embed_network, os.path.join(save_dir, 'pretrain_ep%06d.pth'%(epoch)))

        for step in range(cfg.RL.NUM_STEPS):
            obs_unpacked = {k: current_obs.peek()[k].peek() for k in current_obs.peek()}
            if epoch == start_epoch and step < 10:
                log_input_images(obs_unpacked, mlog, num_stack=cfg.RL.NUM_STACK,
                                 key_names=[], meter_name='debug/input_images', step_num=step)

            with torch.no_grad():
                value, states, action, action_log_prob, pre_embedding = actor_critic.act(obs_unpacked,
                                                                                         states.cuda(),
                                                                                         mask_done.cuda(), mode=training_mode)

            cpu_actions = list(action.squeeze(1).cpu().numpy())
            obs, reward, done, info = envs.step(cpu_actions)
            #envs.render('human')
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()


            episode_rewards += reward
            episode_lengths += (1 + 0)
            mask_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            for i, (r, l, done_) in enumerate(zip(episode_rewards, episode_lengths, done)):  # Logging loop
                if done_:
                    phase = 'train' if i < num_train_processes else 'val'
                    mlog.update_meter(r, meters={'metrics/rewards'}, phase=phase)
                    mlog.update_meter(l, meters={'diagnostics/lengths'}, phase=phase)
            episode_rewards *= mask_done
            episode_lengths *= mask_done

            # Insert the new observation into RolloutStorage

            mask_done = mask_done.cuda()
            if training_mode == 'pretrain':
                for k in obs:
                    if k in current_train_obs.sensor_names:
                        current_train_obs[k].insert(obs[k][:num_train_processes], mask_done[:num_train_processes])
                rollouts.insert([i['episode_id'] for i in info],
                                [i['step_id'] for i in info],
                                current_train_obs.peek(),
                                states[:num_train_processes],
                                action[:num_train_processes],
                                action_log_prob[:num_train_processes],
                                value[:num_train_processes],
                                reward[:num_train_processes],
                                mask_done[:num_train_processes],
                                training_mode)
            else:
                rollouts.insert([i['episode_id'] for i in info],
                                [i['step_id'] for i in info],
                                (torch.tensor(obs['pose']), pre_embedding),
                                states[:num_train_processes],
                                action[:num_train_processes],
                                action_log_prob[:num_train_processes],
                                value[:num_train_processes],
                                reward[:num_train_processes],
                                mask_done[:num_train_processes],
                                training_mode)
            current_obs.insert(obs, mask_done)

            mlog.update_meter(value[:num_train_processes].mean().item(), meters={'diagnostics/value'}, phase='train')
        if cfg.is_train:
            value_loss, action_loss, dist_entropy, max_importance_weight, info = agent.update(rollouts, mode=training_mode)
            rollouts.after_update()  # For the next iter: initial obs <- current observation
            # Update meters with latest training info
            mlog.update_meter(dist_entropy, meters={'losses/dist_entropy'})
            mlog.update_meter(np.exp(dist_entropy), meters={'diagnostics/dist_perplexity'})
            mlog.update_meter(value_loss, meters={'losses/value_loss'})
            mlog.update_meter(action_loss, meters={'losses/action_loss'})
            mlog.update_meter(max_importance_weight, meters={'diagnostics/max_importance_weight'})

            # Main logging
        if (epoch) % cfg.saving.log_interval == 0 :
            n_steps_since_logging = cfg.saving.log_interval * num_train_processes * cfg.RL.NUM_STEPS
            total_num_steps = (epoch + 1) * num_train_processes * cfg.RL.NUM_STEPS
            print("Update {}, num timesteps {}, FPS {} - time {}".format(
                epoch + 1,
                total_num_steps,
                int(n_steps_since_logging / (time.time() - start_time)),
                time.time() - start_time
            ))
            for metric in core_metrics:  # Log to stdout
                for mode in ['train', 'val']:
                    if metric in core_metrics or mode == 'train':
                        mlog.print_meter(mode, total_num_steps, meterlist={metric})
            for mode in ['train', 'val']:  # Log to files
                results = mlog.peek_meter(phase=mode)
                reward_only_flog.log(mode, {metric: results[metric] for metric in core_metrics})
                if mode == 'train':
                    results['step_num'] = epoch + 1
                    flog.log('all_results', results)

                mlog.reset_meter(total_num_steps, mode=mode)
            start_time = time.time()

            # Save checkpoint
        if epoch % cfg.saving.save_interval == 0 :
            save_network(agent.actor_critic, os.path.join(save_dir, 'ep%06d.pth'%(epoch)))


if __name__=='__main__':
    main(cfg)
