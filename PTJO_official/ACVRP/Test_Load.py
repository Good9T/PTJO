##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../../..")
sys.path.insert(0, "../../../..")

##########################################################################################
# import
import logging
from utils import create_logger
from Tester_Load import Tester
##########################################################################################

env_params = {
    'dataset_folder':r"E:\PycharmProjcets\PTJO\ACVRP_dataset_from_ATSP"
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256 ** 0.5,
    'encoder_layer_num': 5,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16 ** 0.5,
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1 / 2) ** 0.5,
    'ms_layer2_init': (1 / 16) ** 0.5,
    'eval_type': 'argmax',
    'one_hot_seed_num': 220,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/train20',
        'epoch': 2000,
    },
    'test_episodes': 30,
    'test_batch_size': 1,
    'augmentation_enable': True,
    'aug_factor': 50,
    'aug_batch_size': 1,
}

logger_params = {
    'log_file': {
        'desc': 'test_dataset',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main
def main():
    create_logger(**logger_params)
    _print_config()
    tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)
    tester.run()

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

if __name__ == "__main__":
    main()