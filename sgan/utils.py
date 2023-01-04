import random

import pathlib

import sys

import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess
import matplotlib
# Only use tkagg backend on laptop NOT HPC
if not torch.cuda.is_available():
    matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolours




def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    https://stackoverflow.com/a/25628397"""
    return plt.cm.get_cmap(name, n)


def plot_trajectories(obs_traj_abs, pred_traj_gt_abs, pred_traj_fake_abs, seq_start_end):
    for i, (s, e) in enumerate(seq_start_end[::, ::]):
        # print(f'i {i}, s {s},e {e}')
        cmap = list(mcolours.TABLEAU_COLORS.keys())
        if len(cmap) < e - s:
            cmap = list(mcolours.CSS4_COLORS.keys())
        fig, ax = plt.subplots()

        for j in range(s, e):
            colour = cmap.pop(0)
            l1 = ax.plot(obs_traj_abs[::, j, 0], obs_traj_abs[::, j, 1], c=colour,
                         linestyle='', marker='.')
            l2 = ax.plot(pred_traj_gt_abs[::, j, 0], pred_traj_gt_abs[::, j, 1], c=colour, linestyle='', marker='x')
            l3 = ax.plot(pred_traj_fake_abs[::, j, 0], pred_traj_fake_abs[::, j, 1], c=colour, linestyle='',
                         marker='*')

        plt.title('Trajectories in relative coordinates')
        plt.axis('square')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        plt.grid(which='both', axis='both', linestyle='-', linewidth=0.5)
        dot_line = mlines.Line2D([], [], color='black', linestyle='', marker='.',
                                 markersize=5, label='Obs_traj')
        x_line = mlines.Line2D([], [], color='black', linestyle='', marker='x',
                               markersize=5, label='Pred_traj_gt')
        star_line = mlines.Line2D([], [], color='black', linestyle='', marker='*',
                                  markersize=5, label='Pred_traj_fake')
        ax.legend(handles=[dot_line, x_line, star_line])
        # if e - s > 3:
        #     save_file = pathlib.Path('/home/david/Pictures/plots/sgan/BatchOneGoalChosen') / f'seq_{i}.png'
        #     save_file.parent.mkdir(exist_ok=True, parents=True)
        #     plt.savefig(save_file, bbox_inches='tight')
        plt.show()
        # plt.waitforbuttonpress()
        plt.close()


def save_plot_trajectory(title, obs_traj_abs, pred_traj_gt_abs, pred_traj_fake_abs, seq_start_end):
    for i, (s, e) in enumerate(seq_start_end[::, ::]):
        # print(f'i {i}, s {s},e {e}')
        cmap = list(mcolours.TABLEAU_COLORS.keys())
        if len(cmap) < e - s:
            cmap = list(mcolours.CSS4_COLORS.keys())
        fig, ax = plt.subplots()

        for j in range(s, e):
            colour = cmap.pop(0)
            l1 = ax.plot(obs_traj_abs[::, j, 0], obs_traj_abs[::, j, 1], c=colour,
                         linestyle='', marker='.')
            # l2 = ax.plot(pred_traj_gt_abs[::, j, 0], pred_traj_gt_abs[::, j, 1], c=colour, linestyle='', marker='x')
            l3 = ax.plot(pred_traj_fake_abs[::, j, 0], pred_traj_fake_abs[::, j, 1], markersize=3, c=colour,
                         linestyle='',
                         marker='*')

        # plt.title(f'Goal Pt {title.split("/")[1]}')
        plt.title(f'Goal Pt {title}')
        plt.axis('square')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        plt.grid(which='both', axis='both', linestyle='-', linewidth=0.5)
        dot_line = mlines.Line2D([], [], color='black', linestyle='', marker='.',
                                 markersize=5, label='Obs_traj')
        # x_line = mlines.Line2D([], [], color='black', linestyle='', marker='x',
        #                        markersize=5, label='Pred_traj_gt')
        star_line = mlines.Line2D([], [], color='black', linestyle='', marker='*',
                                  markersize=5, label='Pred_traj_fake')
        # ax.legend(handles=[dot_line, x_line, star_line])
        # ax.legend(handles=[dot_line, star_line])
        xlim = [-5, 15]
        ax.set_xlim(xlim)
        ylim = [-10, 10]
        ax.set_ylim(ylim)
        save_file = pathlib.Path('/home/administrator/code/sgan/plots') / f'{title}_seq_{i}.png'
        save_file.parent.mkdir(exist_ok=True, parents=True)

        plt.savefig(save_file)
        plt.close(fig)

class plotter:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.prev_x = []
        self.prev_y = []
        plt.title(f'Goal Pt {title}')
        plt.axis('square')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        plt.grid(which='both', axis='both', linestyle='-', linewidth=0.5)

    def display(self, obs_traj, pred):
        self.ax.clear()
        self.ax.plot(self.prev_x, self.prev_y, marker='x', c='dimgrey')
        self.ax.scatter(obs_traj[0], [1], c="red", marker="o", zorder=10)
        self.fig.canvas.draw()
        return 0
def live_plotter(title, obs_traj_abs, pred_traj_gt_abs, pred_traj_fake_abs, seq_start_end, first, save=True):
    for i, (s, e) in enumerate(seq_start_end[::, ::]):
        # print(f'i {i}, s {s},e {e}')
        cmap = list(mcolours.TABLEAU_COLORS.keys())
        if len(cmap) < e - s:
            cmap = list(mcolours.CSS4_COLORS.keys())
        fig, ax = plt.subplots()

        for j in range(s, e):
            colour = cmap.pop(0)
            l1 = ax.plot(obs_traj_abs[::, j, 0], obs_traj_abs[::, j, 1], c=colour,
                         linestyle='', marker='.')
            # l2 = ax.plot(pred_traj_gt_abs[::, j, 0], pred_traj_gt_abs[::, j, 1], c=colour, linestyle='', marker='x')
            l3 = ax.plot(pred_traj_fake_abs[::, j, 0], pred_traj_fake_abs[::, j, 1], markersize=3, c=colour,
                         linestyle='',
                         marker='*')

        # plt.title(f'Goal Pt {title.split("/")[1]}')
        plt.title(f'Goal Pt {title}')
        plt.axis('square')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        plt.grid(which='both', axis='both', linestyle='-', linewidth=0.5)
        dot_line = mlines.Line2D([], [], color='black', linestyle='', marker='.',
                                 markersize=5, label='Obs_traj')
        # x_line = mlines.Line2D([], [], color='black', linestyle='', marker='x',
        #                        markersize=5, label='Pred_traj_gt')
        star_line = mlines.Line2D([], [], color='black', linestyle='', marker='*',
                                  markersize=5, label='Pred_traj_fake')
        # ax.legend(handles=[dot_line, x_line, star_line])
        # ax.legend(handles=[dot_line, star_line])
        xlim = [-5, 15]
        ax.set_xlim(xlim)
        ylim = [-10, 10]
        ax.set_ylim(ylim)
        save_file = pathlib.Path('/home/administrator/code/sgan/plots') / f'{title}_seq_{i}.png'
        save_file.parent.mkdir(exist_ok=True, parents=True)
        if save:
            plt.savefig(save_file)
        if first:
            plt.show()
        # plt.waitforbuttonpress()
        # plt.close()


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return str(inspect.currentframe().f_back.f_lineno)


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm ** norm_type
                total_norm = total_norm ** (1. / norm_type)
            except:
                continue
    return total_norm


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, 'datasets', dset_name, dset_type)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def abs_to_relative(abs_traj):
    """
    Convert absolute tensor to a relative. First values are taken as zero to preserve input shape
    Inputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    Outputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    return torch.cat([torch.zeros((1, abs_traj.shape[1], 2)), torch.diff(abs_traj, dim=0)])
    # return torch.diff(abs_traj, dim=0)


def plot_losses(checkpoint, train: bool = True):
    key = 'train' if train else 'val'
    # g_losses = checkpoint['G_losses']['G_total_loss']
    g_losses = checkpoint[f'metrics_{key}']['g_l2_loss_rel']
    # d_losses = checkpoint['D_losses']['D_total_loss']
    d_losses = checkpoint[f'metrics_{key}']['d_loss']
    # print(checkpoint['metrics_val'].keys())
    fde = checkpoint[f'metrics_{key}']['fde']
    ade = checkpoint[f'metrics_{key}']['ade']
    fig, ax = plt.subplots(2, 2)
    # step = checkpoint['args']['checkpoint_every']
    # x_vals = list(range(0, len(g_losses) * step, step))
    l1 = ax[0, 0].plot(g_losses, 'g', linestyle='-', marker='.', markersize=0.1, label='Generator Losses')
    l2 = ax[0, 1].plot(d_losses, 'r', linestyle='-', marker='.', markersize=0.1, label='Discriminator Losses')
    l3 = ax[1, 0].plot(fde, 'b', linestyle='-', marker='.', markersize=0.1, label=f'{key} FDE')
    l4 = ax[1, 1].plot(ade, 'c', linestyle='-', marker='.', markersize=0.1, label=f'{key} ADE')
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.title('Metrics Graph')
    # plt.axis('square')
    for a in ax.flat:
        a.set_xlabel('Epoch')
        a.set_ylabel('Loss')
        a.legend()
    # plt.grid(which='both', axis='both', linestyle='-', linewidth=0.5)

    plt.show()
    plt.waitforbuttonpress()
    plt.close(fig)


def write_plots_to_video(directory='/home/administrator/code/sgan/plots/'):
    import cv2
    import glob
    import sys
    imgs = glob.glob(f'{directory}*.png')
    imgs = sorted(imgs, key=lambda f: int(''.join(filter(str.isdigit, f))))
    # print(*imgs, sep='\n')

    # sys.exit(0)
    imgs = [cv2.imread(img) for img in imgs]

    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{directory}plot.mp4', fourcc, 2.5, (width, height))

    for img in imgs:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
