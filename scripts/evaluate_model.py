import sys

import argparse
import os
import torch
from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator, IntentionForceGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path, plot_trajectories, plot_losses, save_plot_trajectory

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


def get_intention_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = IntentionForceGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
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


def get_discriminator(checkpoint):
    """Added for transfer learning restore from checkpoint in train_goal.py"""
    args = AttrDict(checkpoint['args'])
    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type).to(device=_DEVICE_)
    discriminator.load_state_dict(checkpoint['d_state'])
    # discriminator.cuda(device=_DEVICE_)
    discriminator.train()
    return discriminator


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


def load_next_seq(i, dset):
    # (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
    #  non_linear_ped, loss_mask) = dset[i]
    # ota = obs_traj.numpy()
    # for k, ped in enumerate(obs_traj):
    #     if k == 0:
    #         print(f'Ped {k} i {i} obs_traj[-1]\t\tX\tY\t{ped.T[-1]}')
    # ptga = pred_traj_gt.numpy()
    # for k, ped in enumerate(pred_traj_gt):
    #     if k == 0:
    #         print(f'Ped {k} i {i} pred_traj_gt[0]\tX\tY\t{ped.T[0]}')
    # print('-'*100)
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     non_linear_ped, loss_mask) = dset[i + 2]
    # for k, ped in enumerate(pred_traj_gt):
    #     if k == 0:
    #         print('Goal state 3*pred_len:')
    #         print(f'Ped {k} i {i + 2} pred_traj_gt[0]\tX\tY\t{ped.T[0]}')
    # print('-' * 100)
    # sys.exit(0)
    return pred_traj_gt[-1]

def evaluate(args, loader, dset, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    batch_no = 0
    first = True
    yes = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            # print(f'seq_start_end.shape {seq_start_end.shape}')
            # if obs_traj.shape[1] > 3:
            #     continue
            # if i < 18:
            #     continue
            # print(f'batch {i}')

            # goal_state = create_goal_states(obs_traj, pred_traj_gt, seq_start_end)
            goal_state = load_next_seq(i, dset)
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            ota = obs_traj.numpy()
            # for j, ped in enumerate(obs_traj.permute(1, 0, 2)):
            #     if j == 0:
            #         print(f'Ped {j} observed traj\tX\n\t\t\t\t\tY\n{ped[-1].T}')
            # ptga = pred_traj_gt.numpy()
            # for j, ped in enumerate(pred_traj_gt.permute(1, 0, 2)):
            #     if j == 0:
            #         print(f'Ped {j} pred traj gt\tX\n\t\t\t\t\tY\n{ped[0].T}')
            # for k, ped in enumerate(obs_traj.permute(1, 0, 2)):
            #     if k == 0:
            #         print(f'Ped {k} i {i} obs_traj[-1]\t\tX\tY\t{ped[-1].T}')
            # ptga = pred_traj_gt.numpy()
            # for k, ped in enumerate(pred_traj_gt):
            #     if k == 0:
            #         print(f'Ped {k} i {i} pred_traj_gt[0]\tX\tY\t{ped[0].T}')
            # print('^' * 100)
            # load_next_seq(i, dset)
            # print('*'*100)
        #     goal_experiment(batch_no, first, generator, goal_state, i, obs_traj, obs_traj_rel, ota, pred_traj_gt, ptga,
        #                     seq_start_end)
            if batch_no == 2:
                sys.exit(0)
        #     goal_state = pred_traj_gt[-1].reshape(1, -1, 2)
        #     for _ in range(num_samples):
        #         pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, goal_state)
        #         pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        #
        #         ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
        #         fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
        #     # plot the lowest fde trajectory of the batch
        #     # print(f'len FDE list {len(fde)}, Tensor shape {fde[0].shape}')
        #     fde_unpacked = [torch.argmax(t).item() for t in fde]
        #     # print(*[t[0:10] for t in fde])
        #     min_fde = fde_unpacked.index(min(fde_unpacked))
        #     # print(f'Index of traj with smallest FDE: {min_fde}')
        #
        #     ade_sum = evaluate_helper(ade, seq_start_end)
        #     fde_sum = evaluate_helper(fde, seq_start_end)
        #
        #     ade_outer.append(ade_sum)
        #     fde_outer.append(fde_sum)
            batch_no += 1
        # ade = sum(ade_outer) / (total_traj * args.pred_len)
        # fde = sum(fde_outer) / (total_traj)
        print(f'{yes}/{i}')
        return ade, fde


def goal_experiment(batch_no, first, generator, goal_state, i, obs_traj, obs_traj_rel, ota, pred_traj_gt, ptga,
                    seq_start_end):
    for j, x in enumerate(torch.linspace(0, 4, 8)):
        if first:
            goal_state = torch.zeros((1, obs_traj.shape[1], 2), device=_DEVICE_)
            goal_state[0, 0] = pred_traj_gt[-1, 0]
            first = False
        # if a ped enters or leaves scene shape of goal must change to reflect this
        elif goal_state.shape[1] != obs_traj.shape[1]:
            temp = goal_state[0, 0, 1]
            goal_state = torch.zeros((1, obs_traj.shape[1], 2), device=_DEVICE_)
            goal_state[0, 0] = pred_traj_gt[-1, 0]
            goal_state[0, 0, 1] = temp
        # goal_state[0, 0, 1] -= x
        goal_state[0, 0, 0] -= x
        # print(goal_state.shape)
        # pred_traj_gt[-1].reshape(1, -1, 2)
        pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, goal_state)
        ptfa = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        # title = f'x_{goal_state[0, 0, 0]:.2f}_y{goal_state[0, 0, 1]:.2f}'
        title = f'Batch {i} | {j} x {goal_state[0, 0, 0]:.2f} y {goal_state[0, 0, 1]:.2f}'
        # plot_trajectories(ota, ptga, ptfa, seq_start_end)
        save_plot_trajectory(title, ota, ptga, ptfa, seq_start_end)
    for i, ped in enumerate(obs_traj.permute(1, 0, 2)):
        if i == 0:
            print(f'Ped {i} observed traj\tX\n\t\t\t\t\tY\n{ped.T}')
    for i, ped in enumerate(pred_traj_gt.permute(1, 0, 2)):
        if i == 0:
            print(f'Ped {i} predicted gt\tX\n\t\t\t\t\tY\n{ped.T}')
    if batch_no == 3:
        sys.exit(0)
    batch_no += 1


def seek_goal(loader, generator, iters=8):
    batch_no = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            # print(f'seq_start_end.shape {seq_start_end.shape}')
            # if obs_traj.shape[1] > 3:
            #     continue
            # if i < 18:
            #     continue
            # print(f'batch {i}')

            ota = obs_traj.numpy()

            ptga = pred_traj_gt.numpy()
            goal_state = torch.zeros((1, obs_traj.shape[1], 2), device=_DEVICE_)
            goal_state[0, 0] = pred_traj_gt[-1, 0]
            goal_state[0, 0, 0] = 4
            goal_state[0, 0, 1] = 4
            for j in range(iters):
                # print(goal_state.shape)
                # pred_traj_gt[-1].reshape(1, -1, 2)
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, goal_state)
                ptfa = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                title = f'Batch {i} iter {j} | x {goal_state[0, 0, 0]:.2f} y {goal_state[0, 0, 1]:.2f}'
                # plot_trajectories(ota, ptga, ptfa, seq_start_end)
                save_plot_trajectory(title, ota, ptga, ptfa, seq_start_end)

                # Shift the obs traj along by one timestep using the pred_traj_fake_abs (ptfa)
                # as the next observed point
                obs_traj[:-1] = obs_traj[1::]
                obs_traj[-1] = ptfa[0]
                ota = obs_traj.numpy()
                obs_traj_rel[:-1] = obs_traj_rel[1::]
                obs_traj_rel[-1] = pred_traj_fake_rel[0]
                if j > iters//2:
                    goal_state[0, 0, 0] = 8
                    goal_state[0, 0, 1] = 8
                # ptga = pred_traj_gt.numpy()
            # for i, ped in enumerate(obs_traj.permute(1, 0, 2)):
            #     if i == 0:
            #         print(f'Ped {i} observed traj\tX\n\t\t\t\t\tY\n{ped.T}')
            # for i, ped in enumerate(pred_traj_gt.permute(1, 0, 2)):
            #     if i == 0:
            #         print(f'Ped {i} predicted gt\tX\n\t\t\t\t\tY\n{ped.T}')
            batch_no += 1
            if batch_no == 1:
                sys.exit(0)


def seek_goal_simulated_data(generator, iters=8, x=6, y=-18):
    with torch.no_grad():
        xs = torch.ones(8, device=_DEVICE_) * 0.25
        ys = torch.zeros(8,  device=_DEVICE_)
        pred_xs = torch.ones(8,  device=_DEVICE_) * 0.25
        # pred_ys = torch.ones(8,  device=_DEVICE_) * 0.25
        pred_ys = ys
        obs_traj_rel = torch.tensor(list(zip(ys, xs)), device=_DEVICE_).reshape(8, 1, 2)
        start_positions_obs = torch.zeros((obs_traj_rel.shape[1], 2), device=_DEVICE_)
        # start_positions_obs[0, 0] = 0
        # start_positions_obs[0, 1] = 0
        obs_traj = relative_to_abs(obs_traj_rel, start_positions_obs)
        pred_traj_gt_rel = torch.tensor(list(zip(pred_ys, pred_xs)),  device=_DEVICE_).reshape(8, 1, 2)
        start_positions_preds = obs_traj[-1].clone().detach()
        pred_traj_gt = relative_to_abs(pred_traj_gt_rel, start_positions_preds)
        seq_start_end = torch.tensor([0, 1], device=_DEVICE_).resize(1, 2)

        ota = obs_traj.numpy()
        ptga = pred_traj_gt.numpy()
        goal_state = torch.zeros((1, obs_traj.shape[1], 2), device=_DEVICE_)
        goal_state[0, 0] = pred_traj_gt[-1, 0]
        goal_state[0, 0, 0] = x
        goal_state[0, 0, 1] = y
        dir = f'{goal_state[0, 0, 0]}_{goal_state[0, 0, 1]:.2f}'

        suffix = f'| x {goal_state[0, 0, 0]:.2f} y {goal_state[0, 0, 1]:.2f}'
        for j in range(iters):
            # print(goal_state.shape)
            # pred_traj_gt[-1].reshape(1, -1, 2)
            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, goal_state)
            ptfa = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            title = f'{dir}/Iter {j} {suffix}'
            # plot_trajectories(ota, ptga, ptfa, seq_start_end)
            save_plot_trajectory(title, ota, ptga, ptfa, seq_start_end)

            # Shift the obs traj along by one timestep using the pred_traj_fake_abs (ptfa)
            # as the next observed point
            obs_traj[:-1] = obs_traj[1::].clone()
            obs_traj[-1] = ptfa[0]
            ota = obs_traj.numpy()
            obs_traj_rel[:-1] = obs_traj_rel[1::].clone()
            obs_traj_rel[-1] = pred_traj_fake_rel[0]
            if j > iters//2:
                goal_state[0, 0, 0] = -5
                goal_state[0, 0, 1] = 20
            # ptga = pred_traj_gt.numpy()
        # for i, ped in enumerate(obs_traj.permute(1, 0, 2)):
        #     if i == 0:
        #         print(f'Ped {i} observed traj\tX\n\t\t\t\t\tY\n{ped.T}')
        # for i, ped in enumerate(pred_traj_gt.permute(1, 0, 2)):
        #     if i == 0:
        #         print(f'Ped {i} predicted gt\tX\n\t\t\t\t\tY\n{ped.T}')




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
        if 'intention' in path:
            generator = get_intention_generator(checkpoint)
        else:
            generator = get_generator(checkpoint)
        # print(generator)
        _args = AttrDict(checkpoint['args'])
        dpath = get_dset_path(_args.dataset_name, args.dset_type)
        _args.batch_size = 1
        dset, loader = data_loader(_args, dpath)
        # plot_losses(checkpoint, train=False)
        # sys.exit(0)
        # ade, fde = evaluate(_args, loader, dset, generator, args.num_samples)
        # print(f'Model: {os.path.basename(path)}, Dataset: {_args.dataset_name}, Pred Len: {_args.pred_len},'
        #       f' ADE: {ade:.2f}, FDE: {fde:.2f}')
        # seek_goal(loader, generator, iters=100)
        seek_goal_simulated_data(generator, iters=50)


if __name__ == '__main__':
    args = parser.parse_args()
    # args.model_path = '/home/david/data/sgan-models/checkpoint_intention_with_model.pt'
    main(args)
