import math

import sys
from pathlib import Path
import numpy as np
import cv2
# Husky import style
from utilities.VisualOdometry import generate_transform_path
# David Laptop import style
# from aru_sil_py.utilities.VisualOdometry import generate_transform_path
import torch

from sgan.utils import relative_to_abs, save_plot_trajectory, plot_trajectories, abs_to_relative
from sgan.data.trajectories import read_file
from scipy.spatial.transform import Rotation as R
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        ptfa = relative_to_abs(pred_traj_fake_rel)
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


def seek_goal(dpath, loader, generator, agent_id=0, iters=50, x=7, y=14):
    batch_no = 0
    print(f'Loader len: {len(loader)}')
    trajs = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):

            # if i != 18:
            #     continue
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            trajs += obs_traj.shape[1]
            # print(f'seq_start_end.shape {seq_start_end.shape}')
            # print(f'obs_traj.shape {obs_traj.shape}')
            # print(f'seq_start_end sample {seq_start_end}')
            with np.printoptions(precision=3, suppress=True):
                goal_state = torch.zeros((1, obs_traj.shape[1], 2), device=_DEVICE_)
                for i, last_obs in enumerate(obs_traj[-1]):
                    # for i, last_obs in enumerate(obs_traj[0]):

                    for filename in Path(dpath).rglob('*'):
                        data = read_file(filename)

                        goal_state[0, i] = get_goal_point(data, generator, last_obs)

            continue
            # if obs_traj.shape[1] > 3:
            #     continue
            # if i < 18:
            #     continue
            # print(f'batch {i}')

            ota = obs_traj.numpy()

            ptga = pred_traj_gt.numpy()
            goal_state = torch.zeros((1, obs_traj.shape[1], 2), device=_DEVICE_)
            goal_state[0, agent_id] = pred_traj_gt[-1, agent_id]
            goal_state[0, agent_id, 0] = x
            goal_state[0, agent_id, 1] = y
            dir = f'{x:.2f}_{y:.2f}_goal'

            suffix = f'| x {x:.2f} y {y:.2f}'
            for j in range(iters):
                # print(goal_state.shape)
                # pred_traj_gt[-1].reshape(1, -1, 2)
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, goal_state, 0.5)
                ptfa = relative_to_abs(pred_traj_fake_rel)
                title = f'{dir}/Iter {j} {suffix}'
                # plot_trajectories(ota, ptga, ptfa, seq_start_end)
                save_plot_trajectory(title, ota, ptga, ptfa, seq_start_end)

                ota = update_observations(agent_id, j, obs_traj, obs_traj_rel, pred_traj_fake_rel, pred_traj_gt,
                                          pred_traj_gt_rel, ptfa)
        print(f'No. of trajs: {trajs}')
        # if j > iters//2:
        #     goal_state[0, 0, 0] = 8
        #     goal_state[0, 0, 1] = 8

        # sys.exit(0)
        # ptga = pred_traj_gt.numpy()
        # for i, ped in enumerate(obs_traj.permute(1, 0, 2)):
        #     if i == 0:
        #         print(f'Ped {i} observed traj\tX\n\t\t\t\t\tY\n{ped.T}')
        # for i, ped in enumerate(pred_traj_gt.permute(1, 0, 2)):
        #     if i == 0:
        #         print(f'Ped {i} predicted gt\tX\n\t\t\t\t\tY\n{ped.T}')
        # batch_no += 1
        # if batch_no == 1:


def create_obs_traj(transforms):
    xvals, yvals = generate_transform_path(transforms)
    xy_vals = list(zip(xvals, yvals))
    if len(xy_vals) > 8:
        xy_vals = xy_vals[-8:]

    return torch.tensor(data=xy_vals, device=_DEVICE_, requires_grad=False).unsqueeze(1).float()


def seek_live_goal(obs_traj, generator, x, y, agent_id =0, title = 'live_exp'):
    with torch.no_grad():
        pred_traj_gt = torch.zeros(obs_traj.shape, device=_DEVICE_)
        obs_traj_rel = abs_to_relative(obs_traj)
        seq_start_end = torch.tensor([0, obs_traj.shape[1]], device=_DEVICE_).unsqueeze(0)
        # seq_start_end = torch.tensor([0, 1], device=_DEVICE_).unsqueeze(0)
        with np.printoptions(precision=3, suppress=True):
            goal_state = torch.zeros((1, obs_traj.shape[1], 2), device=_DEVICE_)
            # goal_state[0, agent_id] = pred_traj_gt[-1, agent_id]
            goal_state[0, agent_id, 0] = x
            goal_state[0, agent_id, 1] = y
            # print(goal_state.shape)

        ota = obs_traj.numpy()
        ptga = pred_traj_gt.numpy()

        pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, goal_state, 1)
        start_pos = obs_traj[-1]
        ptfa = relative_to_abs(pred_traj_fake_rel, start_pos=start_pos)
        # ptfa = relative_to_abs(pred_traj_fake_rel)

        # plot_trajectories(ota, ptga, ptfa, seq_start_end)

        save_plot_trajectory(title, ota, ptga, ptfa, seq_start_end)
    # for i, ped in enumerate(obs_traj.permute(1, 0, 2)):
    #     if i == agent_id:
    #         print(f'Ped {i} observed traj\tX\n\t\t\t\t\tY\n{ped.T}')
    # for i, ped in enumerate(pred_traj_gt.permute(1, 0, 2)):
    #     if i == agent_id:
    #         print(f'Ped {i} predicted gt\tX\n\t\t\t\t\tY\n{ped.T}')
    return ptfa, pred_traj_fake_rel

def pts_to_tfs(pred_traj_fake_rel):
    tfs = []
    # ptfr = pred_traj_fake_rel.numpy()[1:]
    ptfr = pred_traj_fake_rel.numpy()[1:]
    turn_angle = 0
    old_turn_angle = 0
    for i, xy in enumerate(ptfr):
        # if i + 1 >= len(ptfr):
        #     break
        # print(xy)
        if turn_angle != 0:
            old_turn_angle = turn_angle
        turn_angle = math.atan2(xy[0, 1], xy[0, 0]) * 180 / math.pi
        if turn_angle == old_turn_angle:
            turn_angle = 0
        r = R.from_euler('z', turn_angle, degrees=True)
        # r = R.align_vectors(np.append(xy, 0).reshape(1,-1), np.append(ptfr[i + 1], 0).reshape(1,-1))
        # print(r[0].as_matrix())
        rot = np.array(r.as_matrix())
        tf = np.eye(4)
        tf[0:3, 0:3] = rot
        # Assign x and y value respectively
        # tf[2, 3] = xy[0, 0]
        # tf[0, 3] = xy[0, 1]
        tf[2, 3] = np.linalg.norm(xy)
        tfs.append(tf)
        # print(f'Step {i}')
        # print(tf)
        # print(f'{turn_angle} deg')
        # print('-'*20)

    return tfs



def get_goal_point(data, generator, last_obs):
    x, y = (t.item() for t in last_obs)
    xmask = np.isclose(data[::, 2], x, atol=0.005)
    ymask = np.isclose(data[::, 3], y, atol=0.005)
    x_idxs = np.argwhere((xmask))
    y_idxs = np.argwhere((ymask))
    match_idx = np.intersect1d(x_idxs, y_idxs)
    if len(match_idx) > 1:
        match_idx = get_closest_match(data, last_obs, match_idx)
    else:
        match_idx = match_idx[0]
    # except IndexError:
    #     print('Index Error', match_idx)
    agent_id = data[match_idx, 1]

    subset = data[match_idx::]
    frames_w_agent = np.argwhere(subset[::, 1] == agent_id)

    # print(f'Agent ID {agent_id} Match Idx {match_idx} X {x:.3f} Y {y:.3f}')
    # print(f'Agent is in {subset[frames_w_agent].shape[0] - 1} further frames.')

    # print(subset[frames_w_agent])

    agent_final_frame_no = subset[frames_w_agent].shape[0] - 1
    # goal_idx = np.argwhere(data[::, 0:2] == np.array(agent_final_frame_no, agent_id))
    agent_final_frame = subset[frames_w_agent][-1]
    # print(agent_final_frame)
    goal_idx = int(np.argwhere(np.all(data == agent_final_frame, axis=1)))
    # print(goal_idx)
    # print(data[goal_idx])

    # sys.exit(0)
    # If the goal index is within dataset size and the agent id of the goal
    # line matches the matching line

    if subset[frames_w_agent].shape[0] - 1 > 3 * generator.goal.pred_len:
        print(f'Agent is in {subset[frames_w_agent].shape[0] - 1} further frames.')
        # print(f'Matching line: {data[match_idx]}')
        # print(f'3*pred_len line: {data[goal_idx]}')
        return torch.tensor(data[goal_idx, 2:])
    else:
        # print('Could not find 3*pred_len goal')
        return torch.zeros(1, 2)


def get_closest_match(data, last_obs, match_idx):
    """Returns closest matching line in array according to smallest l1 distance between x and y pairs"""

    # print(f'x {x:.4f} y {y:.4f} matching lines: {match_idx}')
    # print(data[match_idx])
    diff_dict = {}
    diff_dict[np.linalg.norm(data[match_idx[0]][2:] - last_obs.numpy())] = match_idx[0]
    diff_dict[np.linalg.norm(data[match_idx[1]][2:] - last_obs.numpy())] = match_idx[1]
    min_key = min(diff_dict.keys())
    return diff_dict[min_key]


def update_observations(agent_id, j, obs_traj, obs_traj_rel, pred_traj_fake_rel, pred_traj_gt, pred_traj_gt_rel, ptfa):
    """Shift the obs traj along by one timestep using the pred_traj_fake_abs (ptfa)
    as the next observed point
    for other agents use gt"""

    obs_traj[:-1] = obs_traj[1::]
    obs_traj_rel[:-1] = obs_traj_rel[1::]
    # If j is less than pred_len (12 for now) then use available gt
    if j < 12:
        obs_traj[-1] = pred_traj_gt[j]
        obs_traj_rel[-1] = pred_traj_gt_rel[j]
        # for goal agent use pred_traj_fake
        obs_traj[-1, agent_id] = ptfa[0, agent_id]
        obs_traj_rel[-1, agent_id] = pred_traj_fake_rel[0, agent_id]
    else:
        obs_traj[-1] = ptfa[0]
        obs_traj_rel[-1] = pred_traj_fake_rel[0]

    return obs_traj.numpy()


def seek_goal_simulated_data(generator, iters=8, x=6, y=-18):
    with torch.no_grad():
        xs = torch.ones(8, device=_DEVICE_) * 0.25
        ys = torch.zeros(8, device=_DEVICE_)
        pred_xs = torch.ones(8, device=_DEVICE_) * 0.25
        # pred_ys = torch.ones(8,  device=_DEVICE_) * 0.25
        pred_ys = ys
        obs_traj_rel = torch.tensor(list(zip(ys, xs)), device=_DEVICE_).reshape(8, 1, 2)
        start_positions_obs = torch.zeros((obs_traj_rel.shape[1], 2), device=_DEVICE_)
        # start_positions_obs[0, 0] = 0
        # start_positions_obs[0, 1] = 0
        obs_traj = relative_to_abs(obs_traj_rel)
        pred_traj_gt_rel = torch.tensor(list(zip(pred_ys, pred_xs)), device=_DEVICE_).reshape(8, 1, 2)
        start_positions_preds = obs_traj[-1].clone().detach()
        pred_traj_gt = relative_to_abs(pred_traj_gt_rel)
        seq_start_end = torch.tensor([0, 1], device=_DEVICE_).unsqueeze(0)

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
            ptfa = relative_to_abs(pred_traj_fake_rel)
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
            if j > iters // 2:
                goal_state[0, 0, 0] = -5
                goal_state[0, 0, 1] = 20
            # ptga = pred_traj_gt.numpy()
        # for i, ped in enumerate(obs_traj.permute(1, 0, 2)):
        #     if i == 0:
        #         print(f'Ped {i} observed traj\tX\n\t\t\t\t\tY\n{ped.T}')
        # for i, ped in enumerate(pred_traj_gt.permute(1, 0, 2)):
        #     if i == 0:
        #         print(f'Ped {i} predicted gt\tX\n\t\t\t\t\tY\n{ped.T}')


def create_goal_states(obs_traj, pred_traj_gt, seq_start_end):
    plot_trajectories(obs_traj, pred_traj_gt, pred_traj_gt * 0, seq_start_end.numpy())
    chosen_ped_id, x, y = manual_goal_select(obs_traj, pred_traj_gt)

    goal_point = torch.zeros((1, obs_traj.shape[1], 2), device=_DEVICE_)
    goal_point[0, chosen_ped_id, 0] = x
    goal_point[0, chosen_ped_id, 1] = y
    print(f'Goal Point:\n{goal_point}')
    return goal_point


def manual_goal_select(obs_traj, pred_traj_gt):
    chosen_ped_id = int(input(f'Choose pedestrian [0... {obs_traj.shape[1] - 1}]to give goal: '))
    for i, ped in enumerate(obs_traj.permute(1, 0, 2)):
        if i == chosen_ped_id:
            print(f'Ped {i} observed traj\n\t\t\tX\t\tY\n{ped}')
    for i, ped in enumerate(pred_traj_gt.permute(1, 0, 2)):
        if i == chosen_ped_id:
            print(f'Ped {i} predicted gt\n\t\t\tX\t\tY\n{ped}')
    x = int(input('Choose x goal: '))
    y = int(input('Choose y goal: '))
    return chosen_ped_id, x, y


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


if __name__ == '__main__':
    tf = np.eye(4)
    tf3 = np.eye(4)
    tf[2, 3] = 0.5
    tf[0, 3] = 0.5
    tf1 = [tf for _ in range(4)]
    tf3[2, 3] = 0.5
    tf3[0, 3] = 0.0
    tf2 = [tf3 for _ in range(4)]

    tfs = tf1 + tf2
    obs_traj = create_obs_traj(tfs)
    print(f'obs_traj:\n {obs_traj.T}')

    obs_traj_rel = abs_to_relative(obs_traj)
    print(f'obs_traj_rel:\n {obs_traj_rel.T}')
    print('-'*100)
    # check = relative_to_abs(obs_traj_rel, start_pos=obs_traj[0])

    # pred_traj_gt = torch.zeros(obs_traj.shape)
    # print(pred_traj_gt)
    with np.printoptions(precision=3, suppress=True):
        tfs = pts_to_tfs(obs_traj_rel)
        # print(tfs)
        check = create_obs_traj(tfs)
        print(f'Check obs_traj_rel:\n {check.T}')
        # print(tfs)
        # x,y = generate_transform_path(tfs)
        check = relative_to_abs(obs_traj_rel, start_pos=obs_traj[0])
        print(f'Recreated obs_traj: \n{check.T}')
        print(f'Recreated == Orig? {torch.equal(obs_traj,check)}')