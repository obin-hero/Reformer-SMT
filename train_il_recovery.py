import os
import argparse

parser = argparse.ArgumentParser(description='Path Follow Agent with Transformer')
parser.add_argument('--data_dir', default='../recovery_dataset/train', type=str)
#parser.add_argument('--data_dir', default='/disk1/obin/habitat_nav_data_processed/train', type=str)
parser.add_argument('--eval_mode', default='simulator', type=str)
parser.add_argument('--resume', default='none', type=str)
#DATA_DIR = '/media/obin/34c21332-ae82-4ba2-a263-5a738c625fa3/habitat_nav_data_preprocessed_1213/train'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
from habitat_demo_noise_dataset import HabitatDemoNoiseDataset, HabitatDemoDataset
from torch.utils.data import DataLoader
from configs.default_cfg import get_config
import torch
#torch.backends.cudnn.enable = True
import time
import numpy as np
from ob_utils import mkdir
from torch.utils.tensorboard import SummaryWriter
import copy

from model.imitation_policy import ImitationPolicy


def main(cfg):

    DATA_DIR = cfg.training.data_dir
    train_data_list = [os.path.join(DATA_DIR,'RECOVERY',x) for x in sorted(os.listdir(os.path.join(DATA_DIR,'RECOVERY')))]
    valid_data_list = [os.path.join(DATA_DIR.replace('train','val'),'DEMON',x) for x in sorted(os.listdir(os.path.join(DATA_DIR.replace('train','val'),'DEMON')))]

    params = {'batch_size': cfg.training.batch_size,
              'shuffle': True,
              'num_workers': cfg.training.num_workers,
              'pin_memory': True}
    train_dataset = HabitatDemoNoiseDataset(cfg, train_data_list)
    train_dataloader = DataLoader(train_dataset, **params)
    train_iter = iter(train_dataloader)
    if cfg.eval_mode == 'dataset':
        valid_dataset = HabitatDemoNoiseDataset(cfg, valid_data_list)
    else:
        valid_dataset = HabitatDemoDataset(cfg, valid_data_list)
        valid_params = copy.deepcopy(params)
        valid_params['batch_size'] = 1
        valid_params['shuffle'] = False
    valid_dataloader = DataLoader(valid_dataset, **valid_params)
    valid_iter = iter(valid_dataloader)


    version_name = cfg.saving.version
    IMAGE_DIR = os.path.join(cfg.saving.save_dir, 'images', version_name)
    SAVE_DIR =  os.path.join(cfg.saving.save_dir, 'saved_networks',  version_name)
    LOG_DIR =  os.path.join(cfg.saving.save_dir, 'logs', version_name)
    mkdir(IMAGE_DIR)
    mkdir(SAVE_DIR)
    mkdir(LOG_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = ImitationPolicy(cfg)
    start_step = 0
    start_epoch = 0
    if cfg.resume != 'none':
        sd = torch.load(os.path.join(SAVE_DIR, cfg.resume))
        agent.load_state_dict(sd)
        start_epoch = int(cfg.resume[cfg.resume.index('epoch')+5:cfg.resume.index('iter')])
        start_step = int(cfg.resume[cfg.resume.index('iter')+4:cfg.resume.index('.pt')])
        print('load {}, start_ep {}, strat_step {}'.format(cfg.resume, start_epoch, start_step))

    print_every = cfg.saving.log_interval
    save_every = cfg.saving.save_interval
    eval_every = cfg.saving.eval_interval
    writer = SummaryWriter(log_dir=LOG_DIR)

    progress_loss = []
    action_loss = []
    start = time.time()
    temp = start
    step = start_step
    step_values = [10000, 50000, 100000]
    step_index = 0
    lr = cfg.training.lr
    def adjust_learning_rate(optimizer, step_index, lr_decay):
        lr = cfg.training.lr * (lr_decay ** step_index)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    agent.cuda()
    agent.train()
    for epoch in range(start_epoch, cfg.training.max_epoch):
        train_dataloader = DataLoader(train_dataset, **params)
        train_iter = iter(train_dataloader)
        for batch in train_iter:
            demo_im, follower_im, demo_act, follower_act = batch
            demo_im, follower_im, demo_act, follower_act = demo_im.cuda(), follower_im.cuda(), demo_act.cuda(), follower_act.cuda()

            results, loss_dict = agent((demo_im, demo_act), (follower_im, follower_act))

            action_loss.append(loss_dict['total_loss'])

            if step in step_values:
                step_index += 1
                lr = adjust_learning_rate(trainer.optim, step_index, cfg.training.lr_decay)

            if step % print_every == 0:
                action_loss_avg = np.array(action_loss).mean()
                print("time = %.2fm, epo %d, step %d, loss: %.4f, lr: %.4f, %ds per %d iters" % (
                    (time.time() - start) // 60, epoch + 1, step + 1, action_loss_avg, lr, time.time() - temp, print_every))
                action_loss = []
                temp = time.time()
                writer.add_scalars('train_loss', {'total': action_loss_avg},step)
            if step% save_every == 0 and step > 0 :
                agent.save(file_name = os.path.join(SAVE_DIR,'epoch%04diter%05d.pt'%(epoch,step)))

            if step % cfg.saving.vis_interval == 0:
                agent.visualize_results(results,
                                          file_name=os.path.join(IMAGE_DIR, 'epoch%04diter%05d' % (epoch, step)),
                                          vis_num=1)
            if step % eval_every == 0:
                agent.eval()
                if not hasattr(agent, 'runner'): agent.build_evaluation_tool()
                eval_start = time.time()
                success_rate = []
                spl = []
                total_progress_loss = []
                for j in range(cfg.saving.eval_iter):
                    try:
                        batch = next(valid_iter)
                    except:
                        valid_dataloader = DataLoader(valid_dataset, **valid_params)
                        valid_iter = iter(valid_dataloader)
                        batch = next(valid_iter)

                    demo_im, demo_act, demo_pose, scene, episode_id = batch
                    demo_im, demo_act, demo_pose = demo_im.cuda(), demo_act.cuda(), demo_pose

                    with torch.no_grad():
                        eval_results = agent.evaluate((demo_im, demo_act, demo_pose), (scene, episode_id))

                    success_rate.append(eval_results['success'])
                    spl.append(eval_results['spl'])


                success_rate_avg = np.array(success_rate).mean()
                spl_avg = np.array(spl).mean()
                progress_loss_avg = np.array(total_progress_loss).mean()
                print(
                    "==> validation time = %.2fm, step %d, success_rate: %.4f, spl: %.4f, progress_loss: %.4f"%
                    ((time.time() - eval_start) // 60, step + 1, success_rate_avg, spl_avg, progress_loss_avg))
                writer.add_scalars('validation', {'success_rate': success_rate_avg,
                                                'action': spl_avg,
                                                'progress': progress_loss_avg}, step)
                agent.visualize_results(eval_results,
                                          file_name=os.path.join(IMAGE_DIR, 'eval_epoch%04diter%05d' % (epoch, step)),
                                          vis_num=1)
                agent.train()
            step += 1
    print('===> end training')

if __name__ == '__main__':
    opts = parser.parse_args()
    cfg = get_config()
    cfg.defrost()
    cfg.training.data_dir = opts.data_dir
    cfg.eval_mode = opts.eval_mode
    cfg.resume = opts.resume
    cfg.freeze()
    main(cfg)
