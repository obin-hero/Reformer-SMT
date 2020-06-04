
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

try: from envs.vizdoom_explore_env import MazeExplorerEnv
except:
    raise
    print("can't use maze explorer")
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

from envs.habitat_objectnav_env import ObjectNavENV
from habitat.config.default import get_config
import habitat
from habitat import make_dataset
import numpy as np
def habitat_objectnav_make_env_fn(cfg, rank,
                log_dir,
                visdom_name = 'main',
                visdom_log_file = None,
                vis_interval = 200,
                visdom_server = 'localhost',
                visdom_port = '8097'):

    habitat_cfg = get_config(cfg.task.habitat.config_file)

    training_scenes = ['PX4nDJXEHrG', '5q7pvUzZiYa', 'S9hNv5qa7GM', 'ac26ZMwG7aT', '29hnd4uzFmX', '82sE5b5pLXE', 'p5wJjkQkbXX', 'B6ByNegPMKs', '17DRP5sb8fy', 'pRbA3pwrgk9', 'gZ6f7yhEvPG', 'HxpKQynjfin', 'ZMojNkEp431', '5LpN3gDmAk7', 'dhjEzFoUFzH', 'vyrNrziPKCB', 'sKLMLpTHeUy', '759xd9YjKW5', 'sT4fr6TAbpF', '1pXnuDYAj8r', 'E9uDoFAP3SH', 'GdvgFV5R1Z5', 'rPc6DW4iMge', 'D7N2EKCX4Sj', 'uNb9QFRL6hY', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 's8pcmisQ38h', 'EDJbREhghzL', 'YmJkqBEsHnH', 'XcA2TqTSSAj', '7y3sRwLe3Va', 'e9zR4mvMWw7', 'JeFG25nYj2p', 'VLzqgDo317F', 'kEZ7cmS4wCh', 'r1Q1Z4BcV1o', 'qoiz87JEwZ2', '1LXtFkjw3qL', 'VFuaQ6m2Qom', 'b8cTxDM8gDG', 'ur6pFq6Qu1A', 'V2XKFyX4ASd', 'Uxmj2M2itWa', 'Pm6F8kyY3z2', 'PuKPg4mmafe', '8WUmhLawc2A', 'ULsKaCPVFJR', 'r47D5H71a5s', 'jh4fc5c5qoQ', 'JF19kD82Mey', 'D7G3Y4RVNrH', 'cV4RVeZvu5T', 'mJXqzFtmKg4', 'i5noydFURQK', 'aayBHfsNo7d']

    length = int(len(training_scenes)/cfg.training.num_envs)

    habitat_cfg.defrost()
    if rank < cfg.training.num_envs:
        habitat_cfg.DATASET.SPLIT = 'train'
    else:
        habitat_cfg.DATASET.SPLIT = 'val'

    cfg.defrost()
    habitat_cfg.TASK.CUSTOM_OBJECT_GOAL_SENSOR = habitat.Config()
    habitat_cfg.TASK.CUSTOM_OBJECT_GOAL_SENSOR.TYPE = 'CustomObjectSensor'
    habitat_cfg.TASK.CUSTOM_OBJECT_GOAL_SENSOR.GOAL_SPEC = "OBJECT_IMG"
    habitat_cfg.DATASET.CONTENT_SCENES = training_scenes[rank * length : (rank + 1) * length]
    habitat_cfg.freeze()
    cfg.task.habitat.TASK_CONFIG = habitat_cfg
    cfg.freeze()

    def filter_fn(episode):
        if episode.info['geodesic_distance'] > 3.0 : return False
        else : return True
    dataset = make_dataset(
        habitat_cfg.DATASET.TYPE, config=habitat_cfg.DATASET, **{'filter_fn':filter_fn}
    )
    env = ObjectNavENV(cfg, dataset=dataset)
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