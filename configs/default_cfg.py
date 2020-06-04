import yacs
from yacs.config import CfgNode

C = CfgNode()
C.seed = 0
C.action_dim = 5
C.pose_dim = 5
C.is_train = True
C.img_size = 64

C.task = CfgNode()
C.task.env_fn = 'vizdoom_make_env_fn'
C.task.deeplab_scene = 'nav_maze_random_goal_01'
C.task.explorer = CfgNode()
C.task.explorer.num_keys=6
C.task.explorer.map_size=20
C.task.explorer.num_maps=1000
C.task.habitat = CfgNode()
C.task.habitat.config_file = 'configs/objectnav_mp3d.yaml'

C.saving = CfgNode()
C.saving.version = 'base'
C.saving.save_dir = '.'
C.saving.log_interval = 10
C.saving.vis_interval = 200
C.saving.save_interval = 500
#C.saving.eval_interval = 500

C.training = CfgNode()
C.training.batch_size = 64
C.training.max_memory_size = 100
C.training.embedding_size = 128
C.training.max_step = 500
C.training.num_envs = 6
C.training.valid_num_envs = 0
C.training.gpu = [0]
C.training.lr = 5e-4
C.training.resume = 'none'
C.training.pretrain_load = 'none'
C.training.pretrain_epoch = 1000
C.training.pretrain_memory_size = 1

C.attention = CfgNode()
C.attention.n_head = 8
C.attention.d_model = 128
C.attention.d_k = 128
C.attention.d_v = 128
C.attention.dropout = 0.1

C.attention.lsh = CfgNode()
C.attention.lsh.bucket_size = 10
C.attention.lsh.n_hashes = 4
C.attention.lsh.add_local_attn_hash = False
C.attention.lsh.causal = True
C.attention.lsh.attn_chunks = 8
C.attention.lsh.random_rotations_per_head = False
C.attention.lsh.attend_across_buckets = True
C.attention.lsh.allow_duplicate_attention = True
C.attention.lsh.num_mem_kv = 0
C.attention.lsh.one_value_head = False
C.attention.lsh.full_attn_thres = 'none'
C.attention.lsh.return_attn = False
C.attention.lsh.post_attn_dropout = 0.1
C.attention.lsh.dropout = 0.1
C.attention.lsh.use_full_attn = False

C.network = CfgNode()
C.network.inputs = ['image', 'prev_action', 'pose']
C.network.num_stack = 1

C.replay_buffer = CfgNode()
C.replay_buffer.max_episode = 1002


C.visdom = CfgNode()
C.visdom.name = 'main'
C.visdom.log_file = 'none'
C.visdom.vis_interval = 200
C.visdom.server = 'localhost'
C.visdom.port = '8097'

C.RL = CfgNode()
C.RL.CLIP_PARAM = 0.1
C.RL.ENTROPY_COEF = 1e-4
C.RL.EPS = 1e-5
C.RL.GAMMA = 0.99
C.RL.INTERNAL_STATE_SIZE = 512
C.RL.NUM_STEPS = 512 # length of each rollout
C.RL.NUM_MINI_BATCH = 8 # size of ppo mini batch
C.RL.NUM_STACK = 1 # frames that each cell can see
C.RL.MAX_GRAD_NORM = 0.5
C.RL.PPO_EPOCH = 8
C.RL.RECURRENT_POLICY = True
C.RL.USE_GAE = True
C.RL.TAU = 0.95
C.RL.VALUE_LOSS_COEF = 1e-3
C.RL.USE_REPLAY = True
C.RL.REPLAY_BUFFER_SIZE = 10000
C.RL.ON_POLICY_EPOCH = 8
C.RL.OFF_POLICY_EPOCH = 8
C.RL.RESULTS_LOG_FILE= 'result_log.pkl'
C.RL.REWARD_LOG_FILE= 'rewards.pkl'
C.RL.NUM_FRAMES = 1e+9
C.RL.SLACK_REWARD = - 0.01
C.RL.SUCCESS_REWARD = 2.5
C.RL.REWARD_MEASURE = 'distance_to_goal'
C.RL.SUCCESS_MEASURE = "spl"



CONFIG_FILE_SEPARATOR = ","
def get_config(config_path=None, update_list=None):
    config = C.clone()
    if config_path:
        if isinstance(config_path, str):
            if CONFIG_FILE_SEPARATOR in config_path:
                config_paths = config_path.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_path]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if update_list:
        config.merge_from_list(update_list)

    config.freeze()
    return config
