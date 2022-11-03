import pathlib

import time

import numpy as np
import sys

import argparse
import os
import torch

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path, plot_trajectories, save_plot_trajectory

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    # generator.cuda(device=_DEVICE_)
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def infer(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """

        # xs = range(0, 8)
        xs = torch.ones(8, dtype=torch.float) * 0.25
        # ys = range(0, 8)
        ys = torch.ones(8, dtype=torch.float) * 0.25
        pred_xs = torch.ones(8, dtype=torch.float) * 0.25
        pred_ys = torch.ones(8, dtype=torch.float) * 0.25

        obs_traj_rel = torch.tensor(list(zip(xs, ys)), dtype=torch.float).reshape(8, 1, 2)
        start_positions_obs = torch.zeros((obs_traj_rel.shape[1], 2), dtype=torch.float)
        obs_traj = relative_to_abs(obs_traj_rel, start_positions_obs)

        pred_traj_gt_rel = torch.tensor(list(zip(pred_xs, pred_ys)), dtype=torch.float).reshape(8, 1, 2)
        start_positions_preds = obs_traj[-1].clone().detach()

        pred_traj_gt = relative_to_abs(pred_traj_gt_rel, start_positions_preds)

        # non_linear_ped = torch.tensor()
        # loss_mask = torch.tensor()
        seq_start_end = torch.tensor([0, 1]).resize(1, 2)
        # print(obs_traj[::, 0:2])
        # print('*'*60)
        # obs_traj = torch.randn(*obs_traj.shape, device=_DEVICE_)
        # print(obs_traj[::, 0:2])
        ade, fde = [], []
        total_traj += pred_traj_gt.size(1)
        ota = obs_traj.numpy()
        ptga = pred_traj_gt.numpy()
        user_noise1 = torch.zeros((seq_start_end.shape[0], generator.noise_first_dim), device=_DEVICE_)
        # user_noise2 = torch.randn((seq_start_end.shape[0], generator.noise_first_dim), device=_DEVICE_)*2
        # user_noise3 = torch.ones((seq_start_end.shape[0], generator.noise_first_dim), device=_DEVICE_)
        # noises = [user_noise1, user_noise2, user_noise3]
        # for noise in noises:
        for i in range(user_noise1.shape[1]):
            for z in np.linspace(0, 1, 10):
                noise = user_noise1
                # noise[0, ::] = z
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, user_noise=noise)

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ptfa = pred_traj_fake.numpy()
                title = f'{i}/{noise}'
                save_plot_trajectory(title, ota, ptga, ptfa, seq_start_end.numpy())
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
                # Only need to plot one with z as all zeros
                if i == 0:
                    continue
            sys.exit(0)
            # print(f'len FDE list {len(fde)}, Tensor shape {fde[0].shape}')
            fde_unpacked = [torch.argmax(t).item() for t in fde]
            # print(*[t[0:10] for t in fde])
            min_fde = fde_unpacked.index(min(fde_unpacked))
            # print(f'Index of traj with smallest FDE: {min_fde}')
            # plot_trajectories(ota, ptga, ptfa, seq_start_end.numpy())
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / total_traj
        return ade, fde


def train_mlp(args, loader, generator, training_iters=1000, print_every=200):
    ade_outer, fde_outer = [], []
    total_traj = 0
    """
    Inputs:
    - obs_traj: Tensor of shape (obs_len, batch, 2)
    - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
    - seq_start_end: A list of tuples which delimit sequences within batch.
    - user_noise: Generally used for inference when you want to see
    relation between different types of noise and outputs.
    Output:
    - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
    """
    # xs = torch.ones(8, dtype=torch.float) * 0.25

    # ys = torch.ones(8, dtype=torch.float) * 0.25
    # pred_xs = torch.ones(8, dtype=torch.float) * 0.25
    # pred_ys = torch.ones(8, dtype=torch.float) * 0.25
    # pred_ys = torch.zeros(8, dtype=torch.float)

    # obs_traj_rel = torch.tensor(list(zip(xs, ys)), dtype=torch.float).reshape(8, 1, 2)
    # start_positions_obs = torch.zeros((obs_traj_rel.shape[1], 2), dtype=torch.float)
    # obs_traj = relative_to_abs(obs_traj_rel, start_positions_obs)

    # pred_traj_gt_rel = torch.tensor(list(zip(pred_xs, pred_ys)), dtype=torch.float).reshape(8, 1, 2)
    # start_positions_preds = obs_traj[-1].clone().detach()

    # pred_traj_gt = relative_to_abs(pred_traj_gt_rel, start_positions_preds)
    # seq_start_end = torch.tensor([0, 1]).resize(1, 2)
    # non_linear_ped = torch.tensor()
    # loss_mask = torch.tensor()
    # ================================================================================================
    # if generator.noise_mix_type == 'global':
    #     noise_shape = (seq_start_end.size(0),) + generator.noise_dim
    # else:
    #     noise_shape = (_input.size(0),) + generator.noise_dim
    mlp = torch.nn.Linear(in_features=2, out_features=loader.batch_size * generator.noise_first_dim, device=_DEVICE_)
    # mlp = torch.nn.Linear(in_features=2, out_features=generator.noise_first_dim, device=_DEVICE_)
    mlp_optimiser = torch.optim.Adam(mlp.parameters(), lr=5e-4)
    loss_func = torch.nn.L1Loss()
    # user_noise1 = torch.zeros((seq_start_end.shape[0], generator.noise_first_dim), device=_DEVICE_)
    start = time.perf_counter()
    ade, fde = [], []
    for j, batch in enumerate(loader):
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
         non_linear_ped, loss_mask, seq_start_end) = batch

        ade, fde = [], []
        total_traj += pred_traj_gt.size(1)
        ota = obs_traj.detach().numpy()
        ptga = pred_traj_gt.detach().numpy()

        for i in range(1, training_iters + 1):
            # print(f'seq_s_e shape: {seq_start_end.shape}')
            # print(f'seq 0 s + end: {seq_start_end[0]}')
            # print(f'seq 0 end: {seq_start_end[0][1]}')
            first_agent_idx = seq_start_end[0][0]
            first_agent_pred_gt = pred_traj_gt.clone()[::, first_agent_idx]
            # print(f'First agent pred gt: {first_agent_pred_gt}')
            first_agent_goal_pt = first_agent_pred_gt[-1]

            # print(f'First agent goal pt: {first_agent_goal_pt}')
            # print(f'Noise 1st dim: {generator.noise_first_dim}')
            # noise = mlp(first_agent_goal_pt).reshape(1, generator.noise_first_dim)
            out = mlp(first_agent_goal_pt)
            agent_noise = out.view(loader.batch_size, generator.noise_first_dim)
            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end,
                                           user_noise=agent_noise, injection_idx=None)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            first_agent_pred_pt = pred_traj_fake.clone()[::, first_agent_idx][-1]
            # if i == 1:
            #     print(f'Pred pt {first_agent_pred_pt} | Goal pt: {first_agent_goal_pt}')

            mlp_optimiser.zero_grad()
            loss = loss_func(first_agent_pred_pt, first_agent_goal_pt)
            loss.backward()
            mlp_optimiser.step()

            ptfa = pred_traj_fake.detach().numpy()

            ade.append(displacement_error(
                pred_traj_fake, pred_traj_gt, mode='raw'
            ))
            fde.append(final_displacement_error(
                pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
            ))
            if i % print_every == 0:
                # fde_unpacked = [torch.argmin(t).item() for t in fde]
                # min_fde = fde_unpacked.index(min(fde_unpacked))
                # print(f'Index of traj with smallest FDE: {min_fde}')
                # print(f'FDE : {fde[0][fde_unpacked[0]]} | len: {len(fde[0])}')
                # print(f'FDE unpacked : {fde_unpacked} | len: {len(fde_unpacked)}')
                title = f'mlp/epoch_{j}/iter_{i}'
                save_plot_trajectory(title, ota, ptga, ptfa, seq_start_end.detach().numpy())
                # Calculate ADE and FDE for the epoch
                ade_sum = evaluate_helper(ade, seq_start_end)
                fde_sum = evaluate_helper(fde, seq_start_end)

                ade_ = ade_sum / (total_traj * args.pred_len)
                fde_ = fde_sum / total_traj
                min, sec = divmod(time.perf_counter() - start, 60)
                print(
                    f'Elapsed: {min:02.0f}:{sec:02.0f} min Epoch {j}/{len(loader) - 1} (Training Iter {i}/{training_iters})'
                    f'| Training Loss {loss:.2f}\tADE: {ade_:.2f}, FDE: {fde_:.2f}')

        ade_sum = evaluate_helper(ade, seq_start_end)
        fde_sum = evaluate_helper(fde, seq_start_end)

        ade_outer.append(ade_sum)
        fde_outer.append(fde_sum)
    ade = sum(ade_outer) / (total_traj * args.pred_len)
    fde = sum(fde_outer) / total_traj
    # print(f'Model: {os.path.basename(args.model_name)}, Dataset: {args.dataset_name}, Pred Len: {args.pred_len},'
    #       f' ADE: {ade:.2f}, FDE: {fde:.2f}')
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
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        dpath = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, dpath)
        ade, fde = infer(_args, loader, generator, args.num_samples)
        print(f'Model: {os.path.basename(path)}, Dataset: {_args.dataset_name}, Pred Len: {_args.pred_len},'
              f' ADE: {ade:.2f}, FDE: {fde:.2f}')


def train(args):
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
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        dpath = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, dpath)
        train_mlp(_args, loader, generator)
        # print(f'Model: {os.path.basename(path)}, Dataset: {_args.dataset_name}, Pred Len: {_args.pred_len},'
        #       f' ADE: {ade:.2f}, FDE: {fde:.2f}')


if __name__ == '__main__':
    args = parser.parse_args()
    # main(args)
    train(args)
