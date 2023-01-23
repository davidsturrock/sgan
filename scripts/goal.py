import time

import math
import sys

from pathlib import Path
import numpy as np
import torch
from sgan.utils import relative_to_abs, save_trajectory_plot, plot_trajectories, abs_to_relative, plot_trajectory_plot, \
    save_figs, close_figs
from sgan.data.trajectories import read_file
from scipy.spatial.transform import Rotation as R

_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model_trajectories(dpath, loader, generator, model_name, iters=50, x=None, y=None, atol=0.5, coltol=0.2, goal_aggro=0.5):
    with torch.no_grad():
        successes = 0
        social_breach = 0
        fails = 0
        seqs = 0
        for i, batch in enumerate(loader):
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end = batch


            goal_state = create_goal_state(dpath=dpath, pred_len=generator.goal.pred_len,
                                           goal_obs_traj=obs_traj[::, [index[0] for index in seq_start_end]],
                                           pred_traj_gt=pred_traj_gt[::, [index[0] for index in seq_start_end]])

            for k, (s, e) in enumerate(seq_start_end):
                # save_directory = Path(f'/home/david/Pictures/plots/goal_test/{model_name}/Seq {k}').with_suffix('').__str__()
                # goal_state[0, k, 0] = x
                # goal_state[0, k, 1] = y
                # TODO consider how goal state in training changes each time but does not change each iter for eval
                obs_list = []
                pred_list = []
                for j in range(iters):
                    # print(goal_state[0 ,s].shape)
                    # print(obs_traj[-1, s].shape)
                    # goal_state[0, s, 0] = goal_x - obs_traj[-1, s, 0].item()
                    # goal_state[0, s, 1] = goal_y - obs_traj[-1, s, 1].item()
                    title = f'Goal Batch {i}  Seq {k} Iter {j} '
                    ptitle = f'Goal Batch {i}  Seq {k} Iter {j} ' \
                             f'Goal {goal_state[0, k, 0].item():.2f}m {goal_state[0, k, 1].item():.2f}m'

                    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, goal_state, goal_aggro=goal_aggro)
                    ptfa = relative_to_abs(pred_traj_fake_rel, start_pos=obs_traj[-1])
                    obs_list.append(obs_traj[5:, s:e].clone())
                    pred_list.append(ptfa[:3, s:e].clone())
                    # social compliance check
                    coll_pt = social_compliance_check(obs_traj[-1, s:e], tolerance=coltol)
                    # if j == 0:

                        # Save plot
                        # plot_trajectory_plot(obs_traj[5:, s:e], pred_traj_gt[20::, s:e], ptfa[:3, s:e],
                        #                      goal=goal_state[0, k], arrival_tol=atol, collision_point=coll_pt,
                        #                      col_tol=coltol,
                        #                      plot_title=ptitle, save_name=title, save_directory=save_directory,
                        #                      xlim=[-30, 30], ylim=[-30, 30])
                    # if x is None:
                    #     x = float(input('Goal x value: '))
                    #     goal_state[0, k, 0] = x
                    # if y is None:
                    #     y = float(input('Goal y value: '))
                    #     goal_state[0, k, 1] = y
                    # # Save plot
                    # save_trajectory_plot(obs_traj[5:, s:e], pred_traj_gt[20::, s:e], ptfa[:3, s:e],
                    #                      goal=goal_state[0, k], arrival_tol=atol, collision_point=coll_pt,
                    #                      col_tol=coltol,
                    #                      plot_title=ptitle, save_name=title, save_directory=save_directory,
                    #                      xlim=[-30, 30], ylim=[-30, 30])

                    if goal_arrival_check(observed_point=obs_traj[-1, s], goal=goal_state[0, k], tolerance=atol):
                        successes += 1
                        print('Goal Reached.')
                        save_directory = Path(f'/home/david/Pictures/plots/goal_test/{Path(model_name.with_suffix(""))}'
                                              f'/success/Seq {k}').__str__()
                        obs_list, pred_list = save_plots_empty_lists(atol, coll_pt, coltol, goal_state, k, obs_list,
                                                                     pred_list, ptitle, save_directory, title)
                        x, y = None, None
                        break
                    elif coll_pt is not None:
                        social_breach += 1
                        print('Social Breach.')
                        save_directory = Path(f'/home/david/Pictures/plots/goal_test/{Path(model_name.with_suffix(""))}'
                                              f'/social breach/Seq {k}').__str__()
                        obs_list, pred_list = save_plots_empty_lists(atol, coll_pt, coltol, goal_state, k, obs_list,
                                                                     pred_list, ptitle, save_directory, title)
                        x, y = None, None
                        break
                    obs_traj, obs_traj_rel = update_observations(s, e, j, obs_traj, pred_traj_gt, ptfa)

                    # update_goalstate(goal_state, iters, j, obs_traj)
                    if j == iters - 1:
                        print('Goal not reached in time.')
                        obs_list = []
                        pred_list = []
                        fails += 1
                        x, y = None, None
                seqs += 1
            if seqs >= 96:
                return successes, fails, social_breach, seqs
        return successes, fails, social_breach, seqs


def save_plots_empty_lists(atol, coll_pt, coltol, goal_state, k, obs_list, pred_list, ptitle, save_directory, title):
    for obs, pred in zip(obs_list, pred_list):
        save_trajectory_plot(obs, [], pred,
                             goal=goal_state[0, k], arrival_tol=atol, collision_point=coll_pt,
                             col_tol=coltol,
                             plot_title=ptitle, save_name=title, save_directory=save_directory)
        return [], []


def update_goalstate(goal_state, iters, j, obs_traj):
    # Reduce goals by husky's next planned step
    goal_x = goal_state[0, 0, 0].item() - obs_traj[-1, 0, 0]
    goal_y = goal_state[0, 0, 1].item() - obs_traj[-1, 0, 1]
    if j > iters // 2:
        goal_state[0, 0, 0] = 8
        goal_state[0, 0, 1] = 8


def social_compliance_check(obs_traj_abs, tolerance=0.2):
    """Returns centre point of circle where social compliance fails else returns None"""
    dists = obs_traj_abs - obs_traj_abs[0]
    # TODO revise. This will only pick first collision in order of peds. Others may exist.
    for i, ped in enumerate(dists[1:]):
        ped = ped.numpy()
        if np.linalg.norm(ped) < tolerance:
            print('NEAR COLLISION!')
            print(f'Agent is at {obs_traj_abs[0].numpy()}')
            print(f'Ped {i} at ({obs_traj_abs[i + 1].numpy()}) is {np.linalg.norm(ped):.2f}m from Agent.')
            return obs_traj_abs[0] - (obs_traj_abs[0] - obs_traj_abs[i + 1]) / 2
        return None


def goal_arrival_check(observed_point, goal, tolerance=0.5):
    """Returns centre point of circle where social compliance fails else returns None"""
    return np.linalg.norm(goal - observed_point) < tolerance


def count_suitable_target_agents_in_dataset(dpath, loader, generator):
    trajs = 0
    good_agents = 0
    with torch.no_grad():
        for batch in loader:
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end = batch

            trajs += obs_traj.shape[1]
            with np.printoptions(precision=3, suppress=True):
                for last_obs in obs_traj[-1]:
                    for filename in Path(dpath).rglob('*'):
                        data = read_file(filename)
                        if goal_point_exists(data, generator, last_obs):
                            good_agents += 1
        print(f'No. of trajs: {trajs}')
        print(f'No. of suitable agents: {good_agents}')


def seek_live_goal(obs_traj, generator, x, y, agent_id=0, title='live_exp'):
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

        save_trajectory_plot(title, ota, ptga, ptfa, 'figure')
    # for i, ped in enumerate(obs_traj.permute(1, 0, 2)):
    #     if i == agent_id:
    #         print(f'Ped {i} observed traj\tX\n\t\t\t\t\tY\n{ped.T}')
    # for i, ped in enumerate(pred_traj_gt.permute(1, 0, 2)):
    #     if i == agent_id:
    #         print(f'Ped {i} predicted gt\tX\n\t\t\t\t\tY\n{ped.T}')
    return ptfa, pred_traj_fake_rel


def pts_to_tfs(pred_traj_fake_rel):
    tfs = []
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

    return np.array(tfs)


def get_goal_point(data, pred_len, last_obs):
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

    if subset[frames_w_agent].shape[0] - 1 > 3 * pred_len:
        # print(f'Agent is in {subset[frames_w_agent].shape[0] - 1} further frames.')
        # print(f'Matching line: {data[match_idx]}')
        # print(f'3*pred_len line: {data[goal_idx]}')
        return torch.tensor(data[goal_idx, 2:])
    else:
        # print('Could not find 3*pred_len goal')
        return torch.zeros((1, 2), device=_DEVICE_)


def find_in_dataset(dpath, last_obs):
    """
    Search through .txt files for a matching last observed (x,y)
    """
    for filename in Path(dpath).rglob('*'):
        data = read_file(filename)
        match_idx = get_match_idx(data, last_obs)
        # If no matches in current file continue to next
        if len(match_idx) == 0:
            # print(f'No match in {Path(filename).name}')
            continue
        # This if-else just makes format of match_idx consistent
        if len(match_idx) > 1:
            # print(f'{len(match_idx)} matches in {Path(filename).name}')
            match_idx = get_closest_match(data, last_obs, match_idx)
            # print(f'Closest match is {match_idx}')
        else:
            match_idx = match_idx[0]
            # print(f'Single match ({match_idx}) in {Path(filename).name}')
        break
    return data, match_idx


def get_match_idx(data, last_obs):
    x, y = (t.item() for t in last_obs)
    xmask = np.isclose(data[::, 2], x, atol=0.005)
    ymask = np.isclose(data[::, 3], y, atol=0.005)
    x_idxs = np.argwhere(xmask)
    y_idxs = np.argwhere(ymask)

    return np.intersect1d(x_idxs, y_idxs)


def goal_point_exists(data, generator, last_obs) -> bool:
    x, y = (t.item() for t in last_obs)
    xmask = np.isclose(data[:, 2], x, atol=0.005)
    ymask = np.isclose(data[:, 3], y, atol=0.005)
    x_idxs = np.argwhere(xmask)
    y_idxs = np.argwhere(ymask)
    match_idx = np.intersect1d(x_idxs, y_idxs)
    if len(match_idx) > 1:
        match_idx = get_closest_match(data, last_obs, match_idx)
    else:
        match_idx = match_idx[0]
    agent_id = data[match_idx, 1]
    subset = data[match_idx:]
    frames_w_agent = np.argwhere(subset[:, 1] == agent_id)
    agent_final_frame = subset[frames_w_agent][-1]
    goal_idx = int(np.argwhere(np.all(data == agent_final_frame, axis=1)))
    return subset[frames_w_agent].shape[0] - 1 > 3 * generator.goal.pred_len


def get_closest_match(data, last_obs, match_idx):
    """Returns the closest matching line in array according to smallest l1 distance between x and y pairs"""
    last_obs = last_obs.cpu().numpy()
    diff_dict = {np.linalg.norm(data[match_idx[0]][2:] - last_obs): match_idx[0],
                 np.linalg.norm(data[match_idx[1]][2:] - last_obs): match_idx[1]}
    min_key = min(diff_dict.keys())
    return diff_dict[min_key]


def update_observations(s, e, pred_iteration, observed_trajectory, future_trajectory_ground_truth,
                        future_trajectory_prediction, pred_len=12):
    """Shift the observed trajectory along by one timestep using the network's prediction for agent and ground truth
     positions for pedestrians in the scene. Once available ground truth positions are exhausted, use the network
     predictions to update both agent and pedestrians.
     s and e are start and end indices of the sequence in the batch meaning update will only affect current sequence"""

    observed_trajectory[:-1, s:e] = observed_trajectory[1:, s:e]
    # If pred_iteration is less than pred_len, then use available ground truth position
    if pred_iteration < pred_len:
        observed_trajectory[-1, s:e] = future_trajectory_ground_truth[pred_iteration, s:e]
        # for goal agent use network prediction as next position
        observed_trajectory[-1, s] = future_trajectory_prediction[0, s]
        # obs_traj[-1, agent_id] = pred_traj_gt[j, agent_id]
    # Else ground truth positions for pedestrians has run out. Update all observations with the network's predictions
    else:
        observed_trajectory[-1, s:e] = future_trajectory_prediction[0, s:e]
    # Return updated observed trajectory in absolute and relative frames
    return observed_trajectory, abs_to_relative(observed_trajectory)


def seek_goal_simulated_data(generator, x, y, iters=60, x_start=0, y_start=0, arrival_tol=0.5):
    """
    iters = 5 * pred_len
    arrival_tol: distance in metres within which goal is considered reached by agent
    """
    with torch.no_grad():
        xs = torch.ones(8, device=_DEVICE_) * 0.25
        ys = torch.zeros(8, device=_DEVICE_)
        pred_xs = torch.ones(8, device=_DEVICE_) * 0.25
        # pred_ys = torch.ones(8,  device=_DEVICE_) * 0.25
        pred_ys = ys
        obs_traj_rel = torch.tensor(list(zip(ys, xs)), device=_DEVICE_).reshape(8, 1, 2)
        start_positions_obs = torch.zeros((obs_traj_rel.shape[1], 2), device=_DEVICE_)
        start_positions_obs[0, 0] = x_start
        start_positions_obs[0, 1] = y_start
        obs_traj = relative_to_abs(obs_traj_rel, start_positions_obs)
        pred_traj_gt_rel = torch.tensor(list(zip(pred_ys, pred_xs)), device=_DEVICE_).reshape(8, 1, 2)
        start_positions_preds = obs_traj[-1].clone().detach()
        pred_traj_gt = relative_to_abs(pred_traj_gt_rel, start_positions_preds)
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
            distance = (goal_state[0, 0] - obs_traj[-1, 0]).pow(2).sum(0).sqrt()
            print(f'Distance to goal: {distance.item():.2f}m')
            # print(goal_state.shape)
            # pred_traj_gt[-1].reshape(1, -1, 2)
            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, goal_state)
            ptfa = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            title = f'{dir}/Iter {j} {suffix}'
            # plot_trajectories(ota, ptga, ptfa, seq_start_end)
            save_trajectory_plot(title, ota, ptga, ptfa, 'figure')
            if distance < arrival_tol:
                print(f'Agent has arrived at goal. ({distance.item():.2f}m away)')
                break
            # Shift the obs traj along by one timestep using the pred_traj_fake_abs (ptfa)
            # as the next observed point
            obs_traj[:-1] = obs_traj[1::].clone()
            obs_traj[-1] = ptfa[0]
            ota = obs_traj.numpy()
            obs_traj_rel[:-1] = obs_traj_rel[1::].clone()
            obs_traj_rel[-1] = pred_traj_fake_rel[0]
            # if j == iters // 2 - 1:
            #     goal_state[0, 0, 0] = -5
            #     goal_state[0, 0, 1] = 20
            #     suffix = f'| x {goal_state[0, 0, 0]:.2f} y {goal_state[0, 0, 1]:.2f}'


def create_goal_state(dpath, pred_len, goal_obs_traj, pred_traj_gt=0, relative=True):
    """
    goal_obs_traj = obs_traj[::, [index[0] for index in seq_start_end]]
    """
    goal_state = torch.zeros((1, goal_obs_traj.shape[1], 2), device=_DEVICE_)

    for i in range(goal_state.shape[1]):
        last_obs = goal_obs_traj[-1, i]
        data, match_idx = find_in_dataset(dpath, last_obs)

        if not match_idx:
            print(f'No matches for {last_obs}found in any file in {dpath}')
            # return final predicted ground truth as goal state if no match is found
            goal_state[0, i] = pred_traj_gt[-1, i]

        agent_id = data[match_idx, 1]
        subset = data[match_idx::]
        frames_w_agent = np.argwhere(subset[::, 1] == agent_id)

        # If the goal index is within dataset size and the agent id of the goal
        # line matches the matching line
        if subset[frames_w_agent].shape[0] > 3 * pred_len:
            agent_goal_frame = subset[frames_w_agent][3 * pred_len]
            # print('3*pred_len goal chosen')
        else:
            agent_goal_frame = subset[frames_w_agent][-1]
            # print(f'Goal {subset[frames_w_agent].shape[0] - 1} frames ahead chosen')
        goal_idx = int(np.argwhere(np.all(data == agent_goal_frame, axis=1)))
        # print(f'Agent is in {subset[frames_w_agent].shape[0] - 1} further frames after frame {data[match_idx, 0]}.')
        # print(f'Match index: {match_idx} [Line no. {match_idx+1}]')
        # print(f'Goal index {goal_idx} [Line no. {goal_idx+1}]')

        # print(f'Obs pt: {last_obs}')
        relative_to = last_obs.clone().to(_DEVICE_)
        if relative:
            goal_state[0, i] = torch.tensor(data[goal_idx, 2:]) - relative_to
            # print(f'Rel Goal: {goal_state[0, i]}')
        else:
            goal_state[0, i] = torch.tensor(data[goal_idx, 2:])
            # print(f'Abs Goal: {goal_state[0, i]}')

    return goal_state
