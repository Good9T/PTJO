import torch
import os
from logging import getLogger

from Env_Load import Env
from Model import Model

from utils import *

class Tester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            self.device = torch.device('cuda', cuda_device_num)
        else:
            self.device = torch.device('cpu')
        torch.set_default_device(self.device)

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        model_load = self.tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator = TimeEstimator()

    def run(self):
        score = AverageMeter()
        aug_score = AverageMeter()
        test_num_episode = self.tester_params['test_episodes']
        bench_dir = self.env_params['dataset_folder']

        all_files = sorted([os.path.join(bench_dir, f) for f in os.listdir(bench_dir)
                            if f.endswith(".atsp")])
        test_file_list = all_files[:test_num_episode]

        self.logger.info(f"{'File':<20} {'No-Aug':<12} {'Aug':<12}")
        self.logger.info("-" * 50)

        episode = 0
        aug_enable = self.tester_params['augmentation_enable']
        aug_factor = self.tester_params['aug_factor'] if aug_enable else 1

        while episode < test_num_episode:
            file_path = test_file_list[episode]
            single_score, single_aug_score = self._test_single_instance(file_path, aug_enable, aug_factor)
            score.update(single_score, 1)
            aug_score.update(single_aug_score, 1)

            fname = os.path.basename(file_path)
            self.logger.info(f"{fname:<20} {single_score:<12.4f} {single_aug_score:<12.4f}")

            episode += 1

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}]".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str))

        self.logger.info("-" * 50)
        self.logger.info(" *** Test Done *** ")
        self.logger.info(" NO-AUG SCORE (Avg): {:.4f} ".format(score.avg))
        self.logger.info(" AUGMENTATION SCORE (Avg): {:.4f} ".format(aug_score.avg))

    def _test_single_instance(self, file_path, aug_enable, aug_factor):
        self.model.eval()
        with torch.no_grad():
            # 原始视角
            self.env.load_problems(file_path, device=self.device, aug_enable=aug_enable, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)

            aug_reward = reward.reshape(aug_factor, 1, self.env.pomo_size)
            max_pomo_reward, _ = aug_reward.max(dim=2)
            no_aug_main = max_pomo_reward[0, :]
            best_aug_main, _ = max_pomo_reward.max(dim=0)

            self.env.load_trans_problems()
            trans_reset_state, _, _ = self.env.reset()
            self.model.pre_forward(trans_reset_state)

            trans_state, trans_reward, trans_done = self.env.pre_step()
            while not trans_done:
                t_selected, _ = self.model(trans_state)
                trans_state, trans_reward, trans_done = self.env.step(t_selected)

            trans_aug_reward = trans_reward.reshape(aug_factor, 1, self.env.pomo_size)
            trans_max_pomo_reward, _ = trans_aug_reward.max(dim=2)
            no_aug_trans = trans_max_pomo_reward[0, :]
            best_aug_trans, _ = trans_max_pomo_reward.max(dim=0)

            no_aug_dual = torch.max(no_aug_main, no_aug_trans)
            no_aug_score = -no_aug_dual.float().mean()

            best_aug_dual = torch.max(best_aug_main, best_aug_trans)
            aug_score = -best_aug_dual.float().mean()

            return no_aug_score.item(), aug_score.item()