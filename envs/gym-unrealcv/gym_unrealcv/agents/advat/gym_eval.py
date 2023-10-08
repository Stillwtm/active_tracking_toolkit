from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import logging
import numpy as np
# from environment import create_env
# from utils import setup_logger
# from model import build_model
# from player_util import Agent
# from utils import cv2_show

import cv2


parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument('--env', default='Track2D-BlockPartialNav-v0', metavar='ENV', help='environment to train on')
parser.add_argument('--num-episodes', type=int, default=100, metavar='NE', help='how many episodes in evaluation')
parser.add_argument('--load-model-dir', default=None, metavar='LMD', help='folder to load trained models')
parser.add_argument('--load-tracker', default=None, metavar='LCD', help='folder to load trained tracker')
parser.add_argument('--load-target', default=None, metavar='LCD', help='folder to load trained target')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--csv', default=None, metavar='SV', help='write to csv')
parser.add_argument('--render', dest='render', action='store_true', help='render test')
parser.add_argument('--network', default='cnn-lstm', metavar='M', help='Model type to use')
parser.add_argument('--stack-frames', type=int, default=1, metavar='SF', help='Choose whether to stack observations')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpu-id', type=int, default=-1, help='GPU to use [-1 CPU only] (default: -1)')
parser.add_argument('--obs', default='img', metavar='UE', help='unreal env')
parser.add_argument('--single', dest='single', action='store_true', help='single agent')
parser.add_argument('--clip', dest='clip_reward', action='store_true', help='clip reward')
parser.add_argument('--gray', dest='gray', action='store_true', help='gray image')
parser.add_argument('--crop', dest='crop', action='store_true', help='crop image')
parser.add_argument('--inv', dest='inv', action='store_true', help='inverse image')
parser.add_argument('--flip', dest='flip', action='store_true', help='flip image and action')
parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale image to [-1, 1]')
parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')
parser.add_argument('--rnn-out', type=int, default=128, metavar='LO', help='lstm output size')
if __name__ == '__main__':
    args = parser.parse_args()

    torch.set_default_tensor_type('torch.FloatTensor')

    log = {}
    setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
        args.log_dir, args.env))
    log['{}_mon_log'.format(args.env)] = logging.getLogger(
        '{}_mon_log'.format(args.env))

    gpu_id = args.gpu_id

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    d_args = vars(args)
    for k in d_args.keys():
        log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    env = create_env("{}".format(args.env), args)
    num_tests = 0
    reward_total_sum = 0
    eps_success = 0
    rewards_his = []
    len_lis = []
    player = Agent(None, env, args, None, device)
    player.model = build_model(env.observation_space, env.action_space, args, device)

    player.model = player.model.to(device)

    # load tracker and target
    if args.load_model_dir != None:
        saved_state = torch.load(
            args.load_model_dir,
            map_location=lambda storage, loc: storage)
        player.model.load_state_dict(saved_state, strict=False)

    # load tracker only
    if args.load_tracker != None:
        saved_state = torch.load(
            args.load_tracker,
            map_location=lambda storage, loc: storage)
        player.model.player0.load_state_dict(saved_state)

    # load target only
    if args.load_target != None:
        saved_state = torch.load(
            args.load_target,
            map_location=lambda storage, loc: storage)
        player.model.player1.load_state_dict(saved_state)

    try:
        player.model.eval()
        player.env.seed(args.seed)
        for i_episode in range(args.num_episodes):
            st_img = []
            player.reset()
            reward_sum = np.zeros(player.num_agents)
            while True:
                # for visualization
                if args.render:
                    if 'Unreal' in args.env:
                        st_img.append(cv2_show(env, False))
                    else:
                        env.render()
                player.action_test()
                reward_sum += player.reward

                if player.done:
                    num_tests += 1
                    rewards_his.append(reward_sum)
                    len_lis.append(player.eps_len)
                    if player.eps_len >= 500:
                        eps_success += 1
                    success_rate = eps_success / num_tests
                    reward_mean = np.array(rewards_his).mean(0)
                    reward_std = np.array(rewards_his).std(0)
                    len_mean = np.array(len_lis).mean()
                    len_std = np.array(len_lis).std()

                    h, w, _ = st_img[0].shape
                    # 创建视频编码器
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(f'./output{i_episode}.mp4', fourcc, 30.0, (w, h))

                    # 逐帧写入视频
                    for img in st_img:
                        video_writer.write(img)

                    # 释放视频编码器
                    video_writer.release()

                    break
        log['{}_mon_log'.format(args.env)].info(
            "El, {0}, R, {1}, R_mean: {2}, R_std: {3}, EL_mean: {4:.2f}, EL_std {5:.2f}, R_step: {6}, S_rate: {7}".format(
                player.eps_len, reward_sum, reward_mean, reward_std,
                len_mean, len_std, reward_mean / len_mean, success_rate))

        header = ['Env', 'Seed', 'R_mean', 'R_std', 'EL_mean', 'EL_std', 'S_rate']
        rows = [{'Env': args.env, 'Seed': args.seed, 'R_mean': float(reward_mean[0]), 'R_std': float(reward_std[0]), 'EL_mean': len_mean,
                 'EL_std': len_std, 'S_rate': float(success_rate)},
                ]
        if args.csv is not None:
            import csv
            if not os.path.exists(args.csv):
                with open(args.csv, 'w') as f:
                    f_csv = csv.DictWriter(f, header)
                    f_csv.writeheader()
                    f_csv.writerows(rows)
            else:
                with open(args.csv, 'a') as f:
                    f_csv = csv.DictWriter(f, header)
                    f_csv.writerows(rows)
        player.env.close()
        os._exit(0)
    except KeyboardInterrupt:
        print("Shutting down")
        player.env.close()