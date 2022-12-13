import numpy as np
import sys

import argparse
import os
import torch
from attrdict import AttrDict
from pathlib import Path

from scripts.evaluate_model import evaluate_helper
from scripts.goal import seek_goal
from scripts.model_loaders import get_combined_generator
from sgan.data.loader import data_loader
from sgan.data.trajectories import read_file
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path, plot_trajectories, plot_losses, save_plot_trajectory

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(args, loader, dset, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            print(obs_traj[::,0].T)
            print(obs_traj_rel[::,0].T)
            sys.exit(0)
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            ota = obs_traj.numpy()

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end,
                                               pred_traj_gt[-1].reshape(1, -1, 2))
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel)

                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
            #     # plot the lowest fde trajectory of the batch
            #     # print(f'len FDE list {len(fde)}, Tensor shape {fde[0].shape}')
            #     fde_unpacked = [torch.argmax(t).item() for t in fde]
            #     # print(*[t[0:10] for t in fde])
            #     min_fde = fde_unpacked.index(min(fde_unpacked))
            #     # print(f'Index of traj with smallest FDE: {min_fde}')
            #
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)

        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        generator = get_combined_generator(checkpoint)

        _args = AttrDict(checkpoint['args'])
        # _args.batch_size = 1
        dpath = get_dset_path(_args.dataset_name, args.dset_type)
        dset, loader = data_loader(_args, dpath)

        # plot_losses(checkpoint, train=True)
        # sys.exit(0)
        ade, fde = evaluate(_args, loader, dset, generator, args.num_samples)
        # print(f'Model: {os.path.basename(path)}, Dataset: {_args.dataset_name}, Pred Len: {_args.pred_len},'
        #       f' ADE: {ade:.2f}, FDE: {fde:.2f}')
        # seek_goal(dpath, loader, generator, agent_id=1, iters=50)
        print(f'No. of seqs: {len(dset)}')
        # seek_goal_simulated_data(generator, iters=50)


if __name__ == '__main__':
    args = parser.parse_args()
    # args.model_path = '/home/david/data/sgan-models/checkpoint_intention_with_model.pt'
    main(args)
