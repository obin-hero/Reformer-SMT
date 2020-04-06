import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import imageio
from ob_utils import visualize_tensor
from envs.habitat_utils.habitat_runner import DemoRunner
import quaternion as q
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return


from model.perception import IL_Perception
class ImitationPolicy(nn.Module):
    def __init__(self, cfg, device='cuda', internal_state_size=128):
        super().__init__()
        self.perception_unit = IL_Perception(cfg)
        self.state_size = internal_state_size
        self.gru = nn.GRUCell(input_size=internal_state_size, hidden_size=internal_state_size)

        # Make the critic
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(self.perception_unit.output_size, 1))

        self.act_out = nn.Sequential(nn.Linear(self.state_size, 128),
                                         nn.ReLU(),
                                         nn.Linear(128, cfg.action_dim))
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        self.img_size = cfg.img_size
        self.max_follow_length = cfg.dataset.max_follow_length
        self.device = device

    def forward(self, demos, follows, train=True):

        demo_im, demo_act = demos
        follow_im, follow_act = follows

        demo_mask = demo_act.sum(-1) > -1
        follow_mask = follow_act > -1
        B = demo_im.shape[0]
        self.perception_unit.embed_demo(demo_im, demo_act, demo_mask.unsqueeze(-1))
        max_length = follow_mask.sum(1).max()
        prev_info = None
        agent_memory = None
        states = torch.zeros((B, self.state_size), dtype=torch.float32).cuda()

        results = {'pred_act': [], 'gt_act': [], 'crs_attn':[], 'slf_attn':[]}
        for t in range(max_length):
            x, embedded_curr_im, agent_memory, slf_attn, crs_attn = self.perception_unit(follow_im[:,t], prev_info, agent_memory)
            x = states  = self.gru(x, states)
            pred_act = self.act_out(x)
            results['pred_act'].append(pred_act)
            results['gt_act'].append(follow_act[:,t])
            results['slf_attn'].append(slf_attn.sum(1))
            results['crs_attn'].append(crs_attn)
            prev_info = (embedded_curr_im, pred_act)
        pred_acts = torch.stack(results['pred_act'], 1)

        self.optim.zero_grad()
        loss = F.cross_entropy(pred_acts.reshape(-1, pred_acts.size(-1)), follow_act[:, :max_length].reshape(-1))

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optim.step()

        with torch.no_grad():
            loss_dict = {'total_loss': loss.item()}
            results.update({'demo_im': demo_im, 'follow_im': follow_im})

        return results, loss_dict
    def save(self, file_name):
        torch.save(self.state_dict(),file_name)
    def process_img(self, img, name, info=None):
        numpy_img = np.clip(cv2.resize(img, dsize=(256, 256)) * 255, 0, 255).astype(np.uint8)
        cv2.putText(numpy_img, name, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        if info is not None:
            cv2.putText(numpy_img, info, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        return numpy_img
    def visualize_results(self, results, file_name=None, vis_num=4):
        # {'pred_act': [], 'gt_act': [], 'pred_progress': [], 'gt_progress': []} ({'demo_im': demo_im, 'follow_im': follow_im})
        demo_im, follower_im = results['demo_im'], results['follow_im']
        if 'gt_act' in results.keys():
            gt_acts = results['gt_act']
        else:
            gt_acts = None
        pred_acts = results['pred_act']
        B = demo_im.shape[0]
        rand_num = np.random.choice(np.arange(B), vis_num) if B > vis_num else [0]
        ret_imgs = []
        for idx in rand_num:
            view_imgs = []
            T = len(pred_acts)
            for t in range(T):
                if gt_acts is not None and gt_acts[t][idx] == -100: break
                if isinstance(follower_im[idx,t], np.ndarray): curr_im = follower_im[idx,t] * 0.5 + 0.5
                else: curr_im = visualize_tensor(follower_im[idx, t] * 0.5 + 0.5, method='array')
                numpy_img = self.process_img(curr_im, 'step: %02d' % t)
                if t < 30:
                    demo_reverse_im = visualize_tensor(demo_im[idx,29-t]*0.5 + 0.5, method='array')
                else:
                    demo_reverse_im = np.zeros_like(numpy_img)
                demo_img = self.process_img(demo_reverse_im, 'demo_step: %02d'%(30-t))
                if gt_acts is not None:
                    gt_act = int(gt_acts[t][idx])
                    if gt_act == 1:
                        gt_act = 1
                    elif gt_act == 2:
                        gt_act = -1
                    cv2.line(numpy_img, (128, 256), (int(128 - 40 * gt_act), 256 - 40), (255, 0, 0), 3)
                pred_act = torch.argmax(torch.softmax(pred_acts[t][idx], dim=-1))
                if pred_act == 1:
                    pred_act = 1
                elif pred_act == 2:
                    pred_act = -1
                cv2.line(numpy_img, (128, 256), (int(128 - 40 * pred_act), 256 - 40), (0, 255, 0), 3)
                if 'map' in results.keys():
                    map_img = results['map'][t][idx]
                    numpy_img = np.concatenate([numpy_img, map_img], 1)
                view_imgs.append(np.concatenate([demo_img, numpy_img],1))
            if 'success' in results.keys():
                sucess_info = 'Success' if results['success'] else 'Fail'
                view_imgs[-1] = cv2.putText(view_imgs[-1], sucess_info, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1.0,
                                            (255, 255, 255), 1)
            ret_imgs.append(view_imgs)

        if file_name is not None:
            for ii, imgs in enumerate(ret_imgs):
                imageio.mimsave(file_name + '_%03d.gif' % ii, imgs, duration=0.2)
            return None
        else:
            return ret_imgs

    def build_evaluation_tool(self):
        self.runner = DemoRunner()

    def evaluate(self, demos, info):
        demo_im, demo_act, demo_pose = demos
        scene, episode_id = info
        B = demo_im.shape[0]
        # set episode
        demo_pose_ = demo_pose.detach().cpu().numpy()
        start_pose, start_rotate = demo_pose_[0, -1, :3], demo_pose_[0, -1, 3:]
        end_pose, end_rotate = demo_pose_[0, 0, :3], demo_pose_[0, 0, 3:]
        look_back = q.as_rotation_vector(q.from_float_array(start_rotate))
        look_back[1] += np.pi
        look_back_quat = q.as_float_array(q.from_rotation_vector(look_back))
        self.runner.init_episode(scene[0], start_pose, look_back_quat, end_pose, end_rotate)
        start_state, obs_img = self.runner.init_common()
        map = self.runner.get_map()
        obs_img = cv2.resize(obs_img[:, :, :3], (self.img_size, self.img_size))
        demo_mask = (demo_pose.abs().sum(dim=2) > 0).to(demo_im.device)
        self.perception_unit.embed_demo(demo_im, demo_act, demo_mask)
        states = torch.zeros((B, self.state_size), dtype=torch.float32).cuda()
        results = {'pred_act': [], 'pred_progress': [], 'gt_progress': [], 'follow_im': [],
                   'map': [], 'slf_attn': [], 'crs_attn': []}
        prev_info = None
        agent_memory = None
        for t in range(self.max_follow_length):
            obs_img = cv2.resize(obs_img[:, :, :3], (self.img_size, self.img_size))
            obs_tensor = np.expand_dims(obs_img, 0).transpose(0, 3, 2, 1) / 255.0 * 2 - 1
            obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32).to(self.device)
            x, embedded_curr_im, agent_memory, slf_attn, crs_attn = self.perception_unit(obs_tensor, prev_info, agent_memory)
            x = states = self.gru(x, states)
            pred_act = self.act_out(x)
            action = pred_act.squeeze().argmax()
            new_obs_img, done = self.runner.step(action)
            new_map_img = self.runner.get_map()
            results['pred_act'].append(pred_act)
            results['follow_im'].append(obs_img / 255.0 * 2 - 1)
            results['slf_attn'].append(slf_attn.sum(1))
            results['crs_attn'].append(crs_attn)
            results['map'].append(np.expand_dims(map, 0))
            obs_img = new_obs_img
            map = new_map_img
            if done: break

        success = done
        agent_dist, shortest_dist = self.runner.agent_episode_distance, self.runner.geodesic_distance(start_pose,
                                                                                                      end_pose)
        if max(shortest_dist, agent_dist) == 0.:
            print('shortest dist', shortest_dist, 'agent_dist', agent_dist, 'start_pose', start_pose, 'end_pose',
                  end_pose)
            spl = 0
        else:
            spl = success * (shortest_dist / max(shortest_dist, agent_dist))

        pred_acts = torch.stack(results['pred_act'], 1)

        results['pred_act'] = pred_acts.transpose(1, 0)  # .detach().cpu().numpy()
        results['follow_im'] = np.expand_dims(np.stack(results['follow_im']), 0)

        results.update({'demo_im': demo_im, 'success': success, 'spl': spl})
        return results
