import yacs
from yacs.config import CfgNode

C = CfgNode()
C.action_dim = 6
C.pose_dim = 5

C.training = CfgNode()
C.training.batch_size = 64
C.training.max_memory_size = 100
C.training.embedding_size = 128
C.training.max_step = 500
C.training.num_processes = 1

C.attention = CfgNode()
C.attention.n_head = 8
C.attention.d_model = 128
C.attention.d_k = 128
C.attention.d_v = 128
C.attention.dropout = 0.1

C.network = CfgNode()
C.network.inputs = ['rgb', 'depth', 'prev_action', 'pose']
C.network.num_stack = 1

C.replay_buffer = CfgNode()
C.replay_buffer.max_episode = 1000



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