import os
import sys
from config.default_config import get_config
from config.agent_config import get_config as get_agent_config

cfg = get_agent_config(sys.argv[1]) if sys.argv[1] is sys.argv[1] != 'default' else get_agent_config()
cfg.defrost()
for arg in sys.argv[2:]:
    exec(arg)
cfg.freeze()

dreamer_pretrain_dir = os.path.join(cfg.SAVING.SAVE_DIR, 'saved_networks', cfg.DREAMER.VERSION)
dreamer_cfg = get_config(os.path.join(dreamer_pretrain_dir,'config.yaml'))

devices = ",".join(str(e) for e in cfg.TRAINING.GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = devices
import torch
torch.backends.cudnn.benchmark=True
print(torch.backends.cudnn.benchmark)

torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)
print("current cpu random seed", torch.initial_seed())
print("current gpu random seed", torch.cuda.initial_seed())

from tensorboardX import SummaryWriter
import numpy as np
np.random.seed(cfg.SEED)
import time
## import custom codes -------------
from datasets.explore_dataset import *
from logger import TorchLogger
from utils.ob_utils import mkdir, load_torch_network
from model.MR_pose import *
from model.MR_LSTM import *
## ---------------------------------
## import task-specific utils--------
from utils.pf_load_data import HabitatDataset
from utils.augmentations import RPFAugmentation
from model.agent.Explorer import *
from rl.habitat_explore_env import ExploreEnv
from rl.habitat_vector_env import make_habitat_vector_env
from rl.stackedobservation import StackedSensorDictStorage
import tnt.torchnet as tnt
import torchvision
from rl.rollout import RolloutSensorDictReplayBuffer
from rl.preprocessingwrapper import EncodeObservatonWrapper
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
def main():
    # save config and codes
    model_name = cfg.SAVING.VERSION
    save_dir = os.path.join(cfg.SAVING.SAVE_DIR, 'saved_networks', model_name)
    image_dir = os.path.join(cfg.SAVING.SAVE_DIR, 'images', model_name)
    log_dir = os.path.join(cfg.SAVING.SAVE_DIR, 'logs/EXPLORATION', model_name)
    mkdir(save_dir)
    mkdir(image_dir)
    mkdir(log_dir)
    cfg_file = os.path.join(save_dir,'config.yaml')
    if os.path.exists(cfg_file) :
        past_cfg = get_agent_config()
        past_cfg.merge_from_file(cfg_file)
        if past_cfg != cfg :
            key = input('your config file for this version will be changed, do you want to proceed anyway? [y/n]')
            while key == 'y' or key=='n':
                if key == 'y' :
                    cfg_str = cfg.dump()
                    with open(cfg_file, "w") as f:
                        f.write(cfg_str)
                    break
                elif key == 'n' : return
    else:
        cfg_str = cfg.dump()
        with open(cfg_file,"w") as f:
            f.write(cfg_str)

    # for RL
    cfg.defrost()
    cfg.TRAINING.BATCH_SIZE = cfg.RL.NUM_PROCESSES + cfg.RL.NUM_VAL_PROCESSES
    cfg.freeze()

    s = time.time()
    dreamer = eval(dreamer_cfg.LEARNER.MAIN)(dreamer_cfg)
    dreamer.cuda()

    dreamer_pretrain_dir = os.path.join(cfg.SAVING.SAVE_DIR, 'saved_networks', cfg.DREAMER.VERSION)
    dreamer, _, _ = load_torch_network(dreamer, dreamer_pretrain_dir, cfg.DREAMER.PRETRAIN_EPOCH, True)

    runner = eval(cfg.TASK.EXPLORATION.NAME)(cfg, dreamer)
    runner.cuda()
    runner.train()
    runner.dreamer.eval()



    if debug_time:
        print('[TIME] Building network {} sec'.format(time.time() - s))
        dreamer.debug_time = True
        s = time.time()

    # load dreamer pretrain model

    envs = make_habitat_vector_env(cfg.TASK.EXPLORATION,
                                   num_processes=cfg.RL.NUM_PROCESSES,
                                   num_val_processes=cfg.RL.NUM_VAL_PROCESSES,
                                   log_dir=log_dir,
                                   vis_interval=cfg.SAVING.VIS_INTERVAL,
                                   visdom_log_file=cfg.SAVING.VISDOM.LOG_FILE,
                                   visdom_server=cfg.SAVING.VISDOM.SERVER,
                                   visdom_port=cfg.SAVING.VISDOM.PORT,
                                   visdom_name=cfg.SAVING.VISDOM.NAME
                                   )
    envs = EncodeObservatonWrapper(envs, runner.dreamer.encoder, envs.observation_space)

    uuid = cfg.SAVING.VERSION
    # mlog = tnt.logger.VisdomMeterLogger(title=uuid, env=uuid,
    #                                     server=cfg.SAVING.VISDOM.SERVER,
    #                                     port=cfg.SAVING.VISDOM.PORT,
    #                                     log_to_filename=cfg.SAVING.VISDOM.LOG_FILE)

    mlog = tnt.logger.TensorboardMeterLogger(env=uuid,
                                            log_dir=log_dir,
                                            plotstylecombined=True)

    loggable_metrics = ['metrics/rewards',
                        'metrics/intrinsic_reward',
                        'diagnostics/dist_perplexity',
                        'diagnostics/lengths',
                        'diagnostics/max_importance_weight',
                        'diagnostics/value',
                        'losses/action_loss',
                        'losses/dist_entropy',
                        'losses/value_loss']
    core_metrics = ['metrics/rewards', 'diagnostics/lengths','metrics/intrinsic_reward']
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


    if cfg.TRAINING.RESUME != -1:
        runner_pretrain_dir = os.path.join(cfg.SAVING.SAVE_DIR, 'saved_networks', cfg.SAVING.VERSION)
        runner = load_torch_network(runner, runner_pretrain_dir, cfg.TRAINING.RESUME, False)
        #start_step += 1
        start_epoch = cfg.TRAINING.RESUME
    else:   
        start_epoch = 0
        start_step = 0
        if cfg.TRAINING.LOAD_PRETRAINED: # selective load
            sd = torch.load(cfg.TRAINING.PRETRAIN_DIR)

            def key_match(pretrained_keys, current_key):
                for k in pretrained_keys:
                    if current_key in k: return True, k
                return False, None

            pretrained_dict = {}
            for k, v in runner.state_dict().items():
                exists, pair_key = key_match(sd.keys(), k)
                if exists:
                    pretrained_dict[k] = sd[pair_key]

            runner.load_state_dict(pretrained_dict, False)

    if debug_time: print('[TIME] Load network {} sec'.format(time.time() - s))

    print('resume ep {} '.format(start_epoch))
    print('=' * 50)
    print('VERSION : ', cfg.SAVING.VERSION)
    print('MAIN : ', cfg.TASK.EXPLORATION.NAME)
    print('DREAMER : ', cfg.DREAMER.VERSION)
    print('=' * 50)


    start = time.time()
    observation_space = envs.observation_space
    retained_obs_shape = {k: v.shape
                          for k, v in observation_space.spaces.items()
                          if k in cfg.RL.SENSORS}
    total_processes = cfg.RL.NUM_PROCESSES + cfg.RL.NUM_VAL_PROCESSES
    current_obs = StackedSensorDictStorage(total_processes, cfg.RL.NUM_STACK, retained_obs_shape)
    current_train_obs = StackedSensorDictStorage(total_processes, cfg.RL.NUM_STACK, retained_obs_shape)
    current_obs.cuda()
    current_train_obs.cuda()

    num_train_processes = cfg.RL.NUM_PROCESSES
    episode_rewards = torch.zeros([num_train_processes, 1])
    episode_lengths = torch.zeros([num_train_processes, 1])

    # First observation
    obs = envs.reset()
    current_obs.insert(obs)
    mask_done = torch.FloatTensor([[0.0] for _ in range(num_train_processes)]).pin_memory()
    states = torch.zeros(total_processes, cfg.RL.INTERNAL_STATE_SIZE).pin_memory()

    rollouts = RolloutSensorDictReplayBuffer( cfg.RL.NUM_STEPS,
                                                num_train_processes,
                                                current_obs.obs_shape,
                                                envs.action_space,
                                                cfg.RL.INTERNAL_STATE_SIZE,
                                                runner.policy,
                                                cfg.RL.USE_GAE,
                                                cfg.RL.GAMMA,
                                                cfg.RL.TAU,
                                                cfg.RL.REPLAY_BUFFER_SIZE,
                                               dreamer= runner.dreamer,
                                              episode_size=cfg.TASK.MEMORY.MAX_MEMORY_SIZE)

    # Main loop
    start_time = time.time()
    num_updates = int(cfg.RL.NUM_FRAMES) // cfg.RL.NUM_STEPS * cfg.RL.NUM_PROCESSES
    abs_time = 0
    for epoch in range(start_epoch, num_updates, 1):
        s = time.time()
        for step in range(cfg.RL.NUM_STEPS):
            obs_unpacked = {k: current_obs.peek()[k].peek() for k in current_obs.peek()}
            if epoch == start_epoch and step < 10:
                log_input_images(obs_unpacked, mlog, num_stack=cfg.RL.NUM_STACK,
                                 key_names=[], meter_name='debug/input_images', step_num=step)

            with torch.no_grad():
                value, action, action_log_prob, states, memory_mask, hidden = runner.policy.act(
                                                                                            obs_unpacked,
                                                                                            states.cuda(),
                                                                                            mask_done.cuda(),
                                                                                            abs_time=abs_time)
            abs_time = (abs_time+1)%cfg.RL.REPLAY_BUFFER_SIZE


            cpu_actions = list(action.squeeze(1).cpu().numpy()+1)
            obs, reward, done, info = envs.step(cpu_actions)
            #envs.render(mode='human')
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()


            episode_rewards += reward
            episode_lengths += (1 + 0)
            mask_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            for i, (r, l, done_) in enumerate(zip(episode_rewards, episode_lengths, done)):  # Logging loop
                if done_:
                    phase = 'train' if i < num_train_processes else 'val'
                    mlog.update_meter(r, meters={'metrics/rewards'}, phase=phase)
                    mlog.update_meter(l, meters={'diagnostics/lengths'}, phase=phase)
                    mlog.update_meter(info[i]["visited_cells"], meters={'metrics/visited_cells'}, phase=phase)
            episode_rewards *= mask_done
            episode_lengths *= mask_done

            # Insert the new observation into RolloutStorage

            mask_done = mask_done.cuda()
            for k in obs:
                if k in current_train_obs.sensor_names:
                    current_train_obs[k].insert(obs[k][:num_train_processes], mask_done[:num_train_processes])
            current_obs.insert(obs, mask_done)
            if memory_mask is not None:
                rollouts.insert(current_train_obs.peek(),
                                        states[:num_train_processes],
                                        action[:num_train_processes],
                                        action_log_prob[:num_train_processes],
                                        value[:num_train_processes],
                                        reward[:num_train_processes],
                                        mask_done[:num_train_processes],
                                        memory_mask[:num_train_processes],
                                        hidden[:num_train_processes])
            else:
                rollouts.insert(current_train_obs.peek(),
                                        states[:num_train_processes],
                                        action[:num_train_processes],
                                        action_log_prob[:num_train_processes],
                                        value[:num_train_processes],
                                        reward[:num_train_processes],
                                        mask_done[:num_train_processes])

            mlog.update_meter(value[:num_train_processes].mean().item(), meters={'diagnostics/value'}, phase='train')
            #print('step_time %03d : %.4f'%(step, time.time()-s))
            s = time.time()

        if rollouts.use_dream_reward:
            rollouts.pre_calculate_intrinsic_reward_all()
        if cfg.TRAINING.IS_TRAIN:
            if not cfg.RL.USE_REPLAY:
                # Moderate compute saving optimization (if no replay buffer):
                #     Estimate future-discounted returns only once
                with torch.no_grad():
                    next_value = runner.get_value(rollouts.observations.at(-1),
                                                        rollouts.states[-1],
                                                        rollouts.masks[-1],).detach()
                rollouts.compute_returns(next_value, cfg['learner']['use_gae'], cfg['learner']['gamma'],
                                         cfg['learner']['tau'])
            value_loss, action_loss, dist_entropy, max_importance_weight, info = runner.agent.update(rollouts)
            rollouts.after_update()  # For the next iter: initial obs <- current observation

            # Update meters with latest training info
            mlog.update_meter(dist_entropy, meters={'losses/dist_entropy'})
            mlog.update_meter(np.exp(dist_entropy), meters={'diagnostics/dist_perplexity'})
            mlog.update_meter(value_loss, meters={'losses/value_loss'})
            mlog.update_meter(action_loss, meters={'losses/action_loss'})
            mlog.update_meter(max_importance_weight, meters={'diagnostics/max_importance_weight'})
            if rollouts.use_dream_reward:
                mlog.update_meter(info['avg_intrinsic_reward'].mean(), meters={'metrics/intrinsic_reward'})

            # Main logging
        if (epoch) % cfg.SAVING.LOG_INTERVAL == 0 :
            n_steps_since_logging = cfg.SAVING.LOG_INTERVAL * num_train_processes * cfg.RL.NUM_STEPS
            total_num_steps = (epoch + 1) * num_train_processes * cfg.RL.NUM_STEPS
            print("Update {}, num timesteps {}, FPS {}".format(
                epoch + 1,
                total_num_steps,
                int(n_steps_since_logging / (time.time() - start_time))
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
        if epoch % cfg.SAVING.SAVE_INTERVAL == 0 :
            save_network(runner, os.path.join(save_dir, 'ep%06d.pth'%(epoch)))


#TODO
#1. Data augmentation # Transformation
#2. Long horizon after convergence
#3.


if __name__=='__main__':
    main()
