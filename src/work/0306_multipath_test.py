__author__ = 'Victoria'
# 2018-03-06
# Not Done

from src.pre_processing.road_network import get_wuhu_graph
from src.map_matching.CTrack import CTrack, get_obs_data, get_road_tests, output_data_js
from src.map_matching.multipath_mm import Observations


def get_test_observations(trail):
    obses = []
    for direction in [-1, 1]:
        for n in range(1,11):
            obses.append(get_road_tests(trail, direction, n))
    return Observations(obses)


if __name__ == '__main__':
    obses = get_test_observations(1)
    print(obses.len_obses())