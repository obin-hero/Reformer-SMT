
from rl.visdommonitor import VisdomMonitor
import os

try: from envs.deeplab_env import DeepmindLabEnv
except: print('cant use deepmindlab')
def deeplab_make_env_fn(cfg, rank,
                log_dir,
                visdom_name = 'main',
                visdom_log_file = None,
                vis_interval = 200,
                visdom_server = 'localhost',
                visdom_port = '8097'):
    env = DeepmindLabEnv(cfg,None)
    env.seed(rank)
    env = VisdomMonitor(env,
                        directory=os.path.join(log_dir, visdom_name),
                        video_callable=lambda x: x % vis_interval == 0,
                        uid=str(rank),
                        server=visdom_server,
                        port=visdom_port,
                        visdom_log_file=visdom_log_file,
                        visdom_env=visdom_name)

    return env


try: from envs.vizdoom_env import VizDoomEnv
except: print('cant use vizdoom')
def vizdoom_make_env_fn(cfg, rank,
                log_dir,
                visdom_name = 'main',
                visdom_log_file = None,
                vis_interval = 200,
                visdom_server = 'localhost',
                visdom_port = '8097'):
    env = VizDoomEnv(cfg)
    env.seed(rank)
    env = VisdomMonitor(env,
                        directory=os.path.join(log_dir, visdom_name),
                        video_callable=lambda x: x % vis_interval == 0,
                        uid=str(rank),
                        server=visdom_server,
                        port=visdom_port,
                        visdom_log_file=visdom_log_file,
                        visdom_env=visdom_name)

    return env

try: from envs.vizdoom_explore_env import MazeExplorerEnv
except: print("can't use maze explorer")
def explorer_make_env_fn(cfg, rank,
                log_dir,
                visdom_name = 'main',
                visdom_log_file = None,
                vis_interval = 200,
                visdom_server = 'localhost',
                visdom_port = '8097'):
    env = MazeExplorerEnv(cfg, seed=rank)
    env.seed(rank)
    env = VisdomMonitor(env,
                        directory=os.path.join(log_dir, visdom_name),
                        video_callable=lambda x: x % vis_interval == 0,
                        uid=str(rank),
                        server=visdom_server,
                        port=visdom_port,
                        visdom_log_file=visdom_log_file,
                        visdom_env=visdom_name)

    return env

try: from envs.habitat_recovery_env import HabitatRecoveryEnv
except: raise
import gym
from envs.habitat_utils.data_loader import HabitatDemoDataset
from torch.utils.data.dataloader import DataLoader
class DatasetWrapper(gym.Wrapper):
    def __init__(self, visdom_env, cfg):
        gym.Wrapper.__init__(self, visdom_env)
        data_dir = cfg.task.recovery.data_dir
        data_list = [os.path.join(data_dir, x) for x in sorted(os.listdir(data_dir))]
        data_chunk = int(len(data_list)/cfg.training.num_envs)
        data_list = data_list[data_chunk * self.env.env.seed : data_chunk * (self.env.env.seed+1)]
        self.dataset = HabitatDemoDataset(data_list)
        self.data_params = {'batch_size': 1,
                  'shuffle': False,
                  'num_workers': 0,
                  'pin_memory': False}
        self.dataloader = DataLoader(self.dataset, **self.data_params)
        self.data_iter = iter(self.dataloader)

    def reset_episode(self, **kwargs):
        self.env.env.reset_episode(**kwargs)

    def reset(self):
        try: demo_im, demo_act, demo_pose, scene_info, episode_id = next(self.data_iter)
        except:
            self.dataloader = DataLoader(self.dataset, **self.data_params)
            self.data_iter = iter(self.dataloader)
            demo_im, demo_act, demo_pose, scene_info, episode_id = next(self.data_iter)
        args = {'scene':scene_info[0], 'start_pose':demo_pose[0,0], 'end_pose':demo_pose[0,-1],
                      'demo_im': demo_im[0], 'demo_act': demo_act[0], 'demo_pose': demo_pose[0]}
        self.env.env.reset_episode(**args)
        obs = self.env.reset()
        return obs


def habitat_make_env_fn(cfg, rank,
                log_dir,
                visdom_name = 'main',
                visdom_log_file = None,
                vis_interval = 200,
                visdom_server = 'localhost',
                visdom_port = '8097'):
    env = HabitatRecoveryEnv(cfg)
    env.seed(rank)
    env = VisdomMonitor(env,
                        directory=os.path.join(log_dir, visdom_name),
                        video_callable=lambda x: x % vis_interval == 0,
                        uid=str(rank),
                        server=visdom_server,
                        port=visdom_port,
                        visdom_log_file=visdom_log_file,
                        visdom_env=visdom_name)
    env = DatasetWrapper(env, cfg)

    return env

