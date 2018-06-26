"""
Created by Viki
2017-12-19 16:14
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
import random
import numpy as np
from src.pre_processing.read_data import get_link
from src.pre_processing.funs import cal_distance, cal_rough_speed_per_cell2, cal_rough_speed, output_data_js
from src.map_matching.CTrack import CTrack
from src.pre_processing.road_network import get_graph
from src.pre_processing.read_data import get_signals_from_csv


def deal_same_obs(obs):
    '''
    Deal with continuous same observations
    :param obs: DataFrame['dates', 'lon', 'lat']
    :return:
    '''
    # Check DataFrame
    if 'dates' not in obs.columns:
        if 'time' in obs.columns:
            obs['dates'] = obs['time']
        else:
            print('[Warning] obs DataFrame need "dates" columns. Error in "deal_same_obs function".')
    obs = obs.sort_values(by=['dates']) # Add 2018-04-17
    obs = obs.reset_index(drop=True)

    # Begin dealing
    inds = [[0]]
    res = obs
    for i in range(1, len(obs)):
        if obs['lon'][i-1] != obs['lon'][i] or obs['lat'][i-1] != obs['lat'][i]:
            inds.append([i])
        else:
            inds[len(inds) - 1].append(i)
    for ind in inds:
        n = len(ind)
        if n == 1:
            continue
        # mid_time = obs['dates'][ind[0]] + (obs['dates'][ind[n - 1]] - obs['dates'][ind[0]]) / n
        mid_time = obs['dates'][ind[0]] + (obs['dates'][ind[n - 1]] - obs['dates'][ind[0]]) / 2
        res = res.drop(ind[1:len(ind)])
        res['dates'][ind[0]] = mid_time
    res = res.reset_index(drop=True)
    return res


def deal_outlier(obs, max_speed=200*1000/3600, max_angle=10):
    '''
    Deal with outliers with speed and distance.
        1. If the angle < 1: drop the point
        2. If the angle < <max_angle> and the speed bigger than <max_speed>: drop the point
    :param obs: DataFrame['dates', 'lon', 'lat']
    :param max_speed: max distance. Bigger than it means unusual. (m/s)
    :param max_angle:
    :return:
    '''
    obs = obs.reset_index(drop=True)
    speed = cal_rough_speed_per_cell2(obs)
    dist = speed['dists']
    direction = speed['direction']
    res = obs
    for i in range(1, len(speed)):
        if abs(abs(direction[i] - direction[i - 1]) - 180) < 1:
            res = res.drop(i)
        elif abs(abs(direction[i] - direction[i - 1]) - 180) < max_angle:
            if (dist[i]+dist[i-1])/(speed.time[i]+speed.time[i-1]) > max_speed:
                res = res.drop(i)

    res = res.reset_index(drop=True)
    return res


def deal_pingpong(obs, tol=10):
    '''
    Deal with one of the ping pong problem:
    1. p1 -> p2 -> p1 => Replace by p1.
    2. p1 -> p2 -> p1 -> p2=> Replace by mean(p1, p2).
    :param obs: A DataFrame with ['dates', 'lon', 'lat']
    :param tol: If dist(p1, p1') < tol, take p1 and p1' as the same point. Unit: Meter(m).
    :return: A dealed obs DataFrame.
    '''
    obs = obs.reset_index(drop=True)

    i = 2
    stored = [[obs.lon[0], obs.lat[0]], [obs.lon[1], obs.lat[1]], 0, 1]
    res = obs

    while i < len(obs):
        pos = [obs.lon[i], obs.lat[i]]
        if cal_distance(stored[0], pos) < tol:
            if i+1 < len(obs):
                if cal_distance(stored[1], [obs.lon[i+1], obs.lat[i+1]]) < tol:
                    res = res.drop([stored[2], stored[3], i+1])
                    res.lon[i] = (stored[0][0] + stored[1][0]) / 2
                    res.lat[i] = (stored[0][1] + stored[1][1]) / 2
                    stored = [pos, pos, i, i]
                    i = i+2
                else:
                    res = res.drop([stored[2], stored[3]])
                    stored = [pos, [obs.lon[i+1], obs.lat[i+1]], i, i+1]
                    i = i+2
            else:
                res = res.drop([stored[2], stored[3]])
                i = i+1
        else:
            stored[0] = stored[1]
            stored[1] = pos
            i = i+1
    res = res.reset_index(drop=True)
    return res


def get_trajectory(signals, cells_loc, user):
    '''
    Trajecotry Segmentation
    Get cell trajectories.
    1. No longer than 20 min. If more than 20 min, part it.
    2. Every trajectory set has at least 5 samples. If less than 5, delete this trajectory set.
    3. Speed smaller than 200km/h. If a point faster than 200km/h,
    :param signals: Dataframe after clean
    :param cells_loc: Dataframe only have 'cell_id', 'lat', 'lon' features.
    :param user: A user_id
    :return: A list with several trajectories.
    '''
    signal = signals[signals['user_id'] == user]
    signal = signal.sort_values('dates')
    signal['cell_id'] = signal['cell_id'].astype(str)
    signal = pd.merge(signal, cells_loc, on='cell_id', how='left')
    signal = signal.dropna(0)
    signal = deal_same_obs(signal)
    signal = signal.reset_index(drop=True)

    max_seconds = 60.0 * 20
    min_set_len = 5
    max_distance = 3000

    ind_sets = []
    ind = [0]
    pre_time = signal['dates'][0]
    speed = cal_rough_speed_per_cell2(signal)
    for i in range(1, len(signal)):
        this_time = signal['dates'][i]
        if ((this_time - pre_time).total_seconds() > max_seconds) or (speed.dists[i-1] > max_distance):
            if len(ind) >= min_set_len:
                ind_sets.append(ind)
            ind = [i]
        else:
            ind.append(i)
        pre_time = this_time

    signal_sets = []
    for inds in range(len(ind_sets)):
        dealed_signal = deal_outlier(signal.iloc[ind_sets[inds]])
        dealed_signal = deal_pingpong(dealed_signal)
        if len(dealed_signal) >= 5:
            signal_sets.append(dealed_signal)

    return signal, ind_sets, signal_sets


def check_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    '''Hyper Parameters'''
    mon = 6
    dd = 7
    hh_start = 0
    hh_end = 24
    file_date = '2017%.2d%.2d' % (mon, dd)

    '''Get Data'''
    # get signals data
    filename = ('../../data/signals_pro/%.2d%.2d/%.2dt%.2d/signals_%.2dt%.2d_within_users.csv'
                % (mon, dd, hh_start, hh_end, hh_start, hh_end))
    signals = get_signals_from_csv(filename)
    signals = signals[['cell_id', 'user_id', 'dates']]

    # get cells data
    cells = pd.read_csv('../../res/[0926]cells_process.csv')
    cells_loc = cells[['cell_id', 'lat', 'lon']]

    users = signals['user_id'].unique()


    graph = get_graph('../../data/hefei_road/link_baidu.txt')
    user_id = users[10]
    signal, ind_sets, signal_sets = get_trajectory(signals, cells_loc, user_id)
    print(cal_rough_speed_per_cell2(signal))
    for i in range(len(signal_sets)):
        if i!=1: continue
        obs = pd.DataFrame({'time': signal_sets[i]['dates'], 'lon': signal_sets[i]['lon'], 'lat': signal_sets[i]['lat']})
        # print(obs)
        obs = deal_same_obs(obs)[0:8]
        check_directory('../../res/MapMatching/%s' % user_id)
        output_data_js(obs, '../../res/MapMatching/%s/raw_trajectory%d.js' % (user_id, i))
        obs = deal_outlier(obs)
        speed = cal_rough_speed_per_cell2(obs)
        print(speed)
        print([abs(speed['direction'][i] - speed['direction'][i-1]) for i in range(1, len(speed))])
        track = CTrack(obs, graph, smoothing=20, interpolate_interval=5)
        out = track.obs
        states = track.viterbi()
        track.plot_each_time_best_trajectory(track.delta)
        # print(cal_rough_speed_per_cell2(states))
        output_data_js(out, '../../res/MapMatching/%s/pre_trajectory%d.js' % (user_id, i))
        output_data_js(states, '../../res/MapMatching/%s/predict_trajectory%d.js' % (user_id, i))
    # print(np.array(signal_sets[4]['dates']))
    print(ind_sets)

