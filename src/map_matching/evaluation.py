__author__ = 'Victoria'

# 2017-12-04

from src.map_matching.CTrack import CTrack, get_obs_data, get_road_tests, output_data_js
from src.pre_processing.road_network import get_graph
from src.pre_processing.funs import cal_distance
import os
import re
import numpy as np


def evaluate_fn(pre, truth):
    '''
    A list of '$lon|$lat'
    :param pre:
    :param truth:
    :return:
    '''
    precision = sum([pos in truth for pos in pre]) / len(pre)
    recall = sum([pos in pre for pos in truth]) / len(truth)
    return precision, recall


def evaluate_fn_dist(pre, truth):
    '''
    A list of '$lon|$lat'
    :param pre:
    :param truth:
    :return: precision and recall
    '''
    pre = [[float(p.split('|')[0]), float(p.split('|')[1])] for p in pre]
    truth = [[float(p.split('|')[0]), float(p.split('|')[1])] for p in truth]
    precision = np.mean([min_dist(p, truth) for p in pre])
    recall = np.mean([min_dist(p, pre) for p in truth])
    return precision, recall


def min_dist(p, positions):
    '''
    Calculate minimize distance from point p to point set poisition
    :param p: [$lon, $lat]
    :param positions: [[$lon, $lat]]
    :return: precision, recall
    '''
    d = cal_distance(p, positions[0])
    for i in range(1, len(positions)):
        tmp = cal_distance(p, positions[i])
        d = tmp if tmp < d else d
    return d


def evaluate_test(trail, direction, n, graph, interpolate_interval=1,
                  smoothing=0, penalty=0.9, direction_window=15):
    df = get_road_tests(trail, direction, n)
    direction_c = u'正向' if direction == 1 else u'反向'

    # cell method
    cell_file = '../../res/MapMatching_test/pre_%s/wuhu_pre_线路%d_%s%d_smoothing=%d_window=%d.js'\
                % ('cell', trail, direction_c, n, smoothing, interpolate_interval)
    print('---Start %s---' % cell_file)
    if os.path.isfile(cell_file):
        pre = read_js(cell_file)
    else:
        obs_cell = get_obs_data(df, type='cell')
        track_cell = CTrack(obs_cell, graph, smoothing=smoothing, interpolate_interval=interpolate_interval,
                       penalty=penalty, direction_window=direction_window)
        states_cell = track_cell.viterbi()
        pre = ['%.5f|%.5f' % (states_cell.loc[i, 'lon'], states_cell.loc[i, 'lat']) for i in range(len(states_cell))]
        output_data_js(states_cell, cell_file)

    # gps method
    gps_file = '../../res/MapMatching_test/pre_%s/wuhu_pre_线路%d_%s%d_smoothing=%d_window=%d.js'\
               % ('GPS', trail, direction_c, n, smoothing, interpolate_interval)
    print('---Start %s---' % gps_file)
    if os.path.isfile(gps_file):
        truth = read_js(gps_file)
    else:
        obs_gps = get_obs_data(df, type='GPS')
        track_gps = CTrack(obs_gps, graph, smoothing=0, interpolate_interval=interpolate_interval,
                            penalty=penalty, direction_window=direction_window)
        states_gps = track_gps.viterbi()
        truth = ['%.5f|%.5f' % (states_gps.loc[i, 'lon'], states_gps.loc[i, 'lat']) for i in range(len(states_gps))]
        output_data_js(states_gps, gps_file)

    precision_per, recall_per = evaluate_fn(pre, truth)
    precision_dist, recall_dist = evaluate_fn_dist(pre, truth)
    print('trail=%d, direction=%d, n=%d, percentage: precision=%f, recall=%f' % (trail, direction, n, precision_per, recall_per))
    print('trail=%d, direction=%d, n=%d, distance: precision=%f, recall=%f' % (trail, direction, n, precision_dist, recall_dist))

    # Add result to evaluation.csv
    with open('../../res/MapMatching_test/evaluation.csv', 'a') as f:
        content = '%d, %d, %d, %d, %d, %f, %d, %f, %f, %f, %f\n' \
                  % (trail, direction, n, interpolate_interval, smoothing, penalty, direction_window,
                     precision_per, recall_per, precision_dist, recall_dist)
        f.write(content)

    return precision_per, recall_per, precision_dist, recall_dist


def get_test_obs(trail, direction, n):
    '''
    Get test data observations.
    :param trail: The number of trail.
    :param direction: [0/1]
                      If direction == 1: positive direction
                      else: negative direction
    :param n: The number of test.
    :return: A observation DataFrame [lon, lat, dates]
    '''
    res = get_road_tests(trail, direction, n)
    res['dates'] = res.time
    res = res.drop(['time'], axis=1)
    return res


def read_js(file):
    f = open(file)
    content = f.read()
    content = re.search(r'\[\[(.*?)\]\]', content).groups(0)[0]
    pos = content.split('], [')
    lon = []
    lat = []
    for p in pos:
        lon.append(float(p.split(', ')[0]))
        lat.append(float(p.split(', ')[1]))
    f.close()
    res1 = ['%.5f|%.5f' % (lon[i], lat[i]) for i in range(len(lon))]
    # res2 = [[lon[i], [lat[i]]] for i in range(len(lon))]
    return res1


if __name__ == '__main__':
    graph = get_graph('../../data/road_test/link_baidu_wuhu.txt')
    for n in range(1, 11):
        for direction in [-1, 1]:
            for trail in range(1, 9):
                evaluate_test(trail, direction, n, graph, smoothing=20, interpolate_interval=5)
    # df = get_road_tests(9, -1, 1)
    # print(df.head())
