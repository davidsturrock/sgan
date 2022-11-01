import argparse
import os
import torch

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path, plot_trajectories

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


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            # batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            # print(obs_traj[::,0:2])
            # print('*'*60)
            # obs_traj = torch.randn(*obs_traj.shape, device=_DEVICE_)
            # print(obs_traj[::, 0:2])
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            ota = obs_traj.numpy()
            ptga = pred_traj_gt.numpy()
            user_noise = torch.ones((seq_start_end.shape[0], generator.noise_first_dim), device=_DEVICE_)
            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end)


                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ptfa = pred_traj_fake.numpy()

                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))
            # plot the lowest fde trajectory of the batch
            print(f'len FDE list {len(fde)}, Tensor shape {fde[0].shape}')
            fde_unpacked = [torch.argmax(t).item() for t in fde]
            # print(*[t[0:10] for t in fde])
            min_fde = fde_unpacked.index(min(fde_unpacked))
            print(f'Index of traj with smallest FDE: {min_fde}')
            plot_trajectories(ota, ptga, ptfa, seq_start_end.numpy())
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
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        dpath = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, dpath)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print(f'Model: {os.path.basename(path)}, Dataset: {_args.dataset_name}, Pred Len: {_args.pred_len},'
              f' ADE: {ade:.2f}, FDE: {fde:.2f}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
