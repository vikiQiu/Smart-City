'''
Author: Vikitoria
Date: 2018-04-17
'''

import json
import pandas as pd
import numpy as np

from src.map_matching.multipath_mm import MultiDTWIndx, multi_map_matching
from src.map_matching.CTrack import CTrack
from src.pre_processing.road_network import get_graph
from src.pre_processing.funs import cal_distance, get_datetime, output_data_js
from src.work.oneuser_alldata_0307 import trajectories_tojson


def multi_mm(user_number, path_need):
    with open('../../res/work/0311_random_trajectories/random_user%d.json' % user_number) as data_file:
        data = json.load(data_file)

    obses = []
    for i in path_need:
        tmp = data['trajectories'][i]
        dates = [get_datetime(t) for t in tmp['dates']]
        obs = pd.DataFrame({'lon': tmp['lon'], 'lat': tmp['lat'], 'dates': dates})
        obses.append(obs)

    # mdtw = MultiDTWIndx(obses, interval=5)
    # new_obs = mdtw.run_one_by_one()

    graph = get_graph('../../data/hefei_road/link_baidu.txt')
    multi_map_matching(obses, graph, save_name='random_user%d' % user_number,
                       mdtw_interval=5, mm_smoothing=10, mm_interpolate_interval=10)
    # track = CTrack(new_obs, graph, smoothing=10, interpolate_interval=10)
    # output_data_js(track.obs, '../../res/work/0417_multi_mm/random_user%d_processed.js'
    #                % (user_number))
    # states = track.viterbi()
    # output_data_js(states, '../../res/work/0417_multi_mm/random_user%d.js'
    #                % (user_number))


def multi_mm_similar(user_number, k):
    file_name = 'User %d Similar Trajectories Plot %d' % (user_number, k)
    with open('../../res/work/0422_random_trajectories_plot/similar_json/%s.json' % file_name) as data_file:
        data = json.load(data_file)

    obses = []
    for i in range(len(data['trajectories'])):
        tmp = data['trajectories'][i]
        dates = [get_datetime(t) for t in tmp['dates']]
        obs = pd.DataFrame({'lon': tmp['lon'], 'lat': tmp['lat'], 'dates': dates})
        obses.append(obs)

    mdtw = MultiDTWIndx(obses, interval=5)
    new_obs = mdtw.run_one_by_one()

    graph = get_graph('../../data/hefei_road/link_baidu.txt')
    track = CTrack(new_obs, graph, smoothing=10, interpolate_interval=10)
    output_data_js(track.obs, '../../res/work/0417_multi_mm/random_user%d_processed.js'
                   % (user_number))
    states = track.viterbi()
    output_data_js(states, '../../res/work/0417_multi_mm/random_user%d.js'
                   % (user_number))


if __name__ == '__main__':
    # multi_mm_similar(20, 1)
    multi_mm_similar(47, 3)
