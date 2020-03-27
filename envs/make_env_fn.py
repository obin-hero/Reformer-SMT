
from rl.visdommonitor import VisdomMonitor
import os

try: from envs.deeplab_env import DeepmindLabEnv
except: print('cant use deepmindlab')
def deeplab_make_env_fn(cfg, scene, rank,
                log_dir,
                visdom_name = 'main',
                visdom_log_file = None,
                vis_interval = 200,
                visdom_server = 'localhost',
                visdom_port = '8097'):
    env = DeepmindLabEnv(cfg,scene)
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


from envs.vizdoom_env import VizDoomEnv
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
