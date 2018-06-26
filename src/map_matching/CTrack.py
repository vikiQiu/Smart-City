__author__ = 'Victoria'
# 2017-11-13

import numpy as np
import pandas as pd
import os
import re
from datetime import timedelta
from math import exp, sqrt
import matplotlib.pyplot as plt
from src.map_matching.read_data import get_wuhu_cellSheet, get_road_tests
from src.pre_processing.road_network import Graph, get_graph, Vertex
from src.pre_processing.read_data import get_cellSheet
from src.pre_processing.funs import cal_distance, output_geo_js, output_data_js


class CTrack:
    '''
    The Viterbi algorithm to do map matching from an observables series to a road segments' series with HMM.
    '''
    def __init__(self,
                 obs: pd.DataFrame,
                 g: Graph,
                 interpolate_interval=1,
                 smoothing=0,
                 is_direction=False,
                 direction_window=5,
                 penalty=0.9,
                 interpolate=True):
        '''
        Initialize CTrack
        :param raw_obs: A DataFrame [time, lon, lat], time spacing may not be one second.
        :param G: Road network [Graph]
        :param interpolate_interval: Interval seconds of processed observables. Default = 1.
        :param smoothing: Window length of smoothing. If smoothing == 0, do not smooth.
        :param penalty: Direction penalty used in transition score.
        :param interpolate: Whether to interpolate trajectory.
        :var self.obs: The interpolated observables [DataFrame]
        :var self.N: The number of road segments
        :var self.T: The length of observables time. (The row number of self.obs)
        :var self.roads: DataFrame [index, road<Vertex>], index is pos<String: '$lon|$lat'>.
                         Roads has been already shrunken.
        '''
        if 'time' not in obs.columns:
            obs['time'] = obs['dates']
        obs = self.raw_obs = obs.sort_values('time')
        obs = self.interpolate(obs, interpolate_interval) if interpolate else obs
        obs = self.smoothing(obs, smoothing) if smoothing > 0 and interpolate else obs
        self.obs = obs
        self.G = g

        # Get roads
        vertices = self.G.get_all_vertex()
        roads = pd.DataFrame({'road': vertices, 'pos': [vertex.pos for vertex in vertices]})
        lon_tol = 0.01
        lat_tol = 0.01
        roads = self.shrinking_roads(roads, max(obs.lon), min(obs.lon),
                                     max(obs.lat), min(obs.lat), lon_tol, lat_tol)
        roads = roads.reset_index(drop=True)
        roads = roads.reset_index()
        self.roads = roads.set_index('pos')
        self.roads_set = [road for road in self.roads.road]

        self.N = len(self.roads)
        self.T = len(self.obs)
        self.is_direction = is_direction
        self.direction = self.get_directions(self.obs, direction_window)
        self.penalty = penalty

    def valid_roads(self, delta, n=500):
        '''
        Get the first n roads after sorted
        :param delta: the newest row of delat
        :param n: get n valid roads
        :return: A filtered DataFrame from self.roads
        '''
        ind = np.argsort(delta)[::-1][:n]
        return self.roads.iloc[ind]

    def viterbi(self):
        '''
        Use Viterbi algorithm to predict a possible state sequence according to observables.
        :return: A DataFrame
        '''
        # Initialize
        print('Total length = %d' % len(self.obs))
        delta = np.zeros([self.T, self.N])
        delta[0] = [self.emission_score(road, self.obs.lon[0], self.obs.lat[0]) for road in self.roads.road]
        print(delta[delta>0])

        # geographic interval
        geo_interval = 0.01

        for t in range(1, self.T):
            '''
            Calculate delta in time t.
            '''
            tmp_o = self.obs.iloc[t] # obs_t
            # Only calculate roads near the obs[t] whose longitude and latitude vary between 0.02 of obs_t
            # if valid == 'sort':
            #     valid_roads = self.valid_roads(delta[t-1], n=1000)

            flag = True
            num = 1

            while flag:
                valid_roads = self.shrinking_roads(self.roads, tmp_o.lon, tmp_o.lon, tmp_o.lat, tmp_o.lat,
                                                   geo_interval*num, geo_interval*num)

                for road in valid_roads.road:
                    '''
                    :var road: A Vertex.
                    :var i: The index of the Vertex road
                    :var neighbors: The achievable roads list to road including road itself. <[Vertex]>
                    delta[t, i] = max_j(delta[t-1, j] * a_ij * b_j(o_t))
                    a_ij = 1/len(neighbors)
                    b_j(o_t) = emission_score(j, o_t.lon, o_t.lat)
                    '''
                    i = self.roads.loc[road.pos]['index']
                    neighbors = [road]
                    neighbors.extend([r for r in road.get_connections()])
                    n = len(neighbors)
                    delta[t, i] = max([delta[t-1, self.roads.loc[neighbor.pos]['index']]
                                       * self.transition_score(neighbor, road, self.direction[t-1])
                                       * self.emission_score(neighbor, tmp_o.lon, tmp_o.lat)
                                       for neighbor in neighbors if neighbor in self.roads_set])
                    # print('time %d; road %d %s neighbor number %d; prob=%f' % (t, i, road.pos, n, delta[t, i]))

                    if max(delta[t, ]) > 0:
                        flag = False
                    else:
                        num += 1
            print('time %d; Max prob=%f' % (t, max(delta[t, ])))

        self.delta = delta
        states = [self.roads.iloc[np.argmax(delta[self.T-1])].road]
        for i in range(1, self.T):
            road = states[i-1]
            neighbors = [r for r in road.get_connections()]
            neighbors.append(road)
            neighbors = [r for r in neighbors if r.pos in self.roads.index]
            neighbors_delta = delta[self.T-1-i, [int(self.roads.loc[r.pos]['index']) for r in neighbors]]
            states.append(neighbors[np.argmax(neighbors_delta)])
        lon = [r.lon for r in reversed(states)]
        lat = [r.lat for r in reversed(states)]
        print('States Done.')
        return pd.DataFrame({'dates': self.obs.time, 'lon': lon, 'lat': lat})

    def get_states(self, t, delta):
        '''
        Get states according to given delta
        :param t: Length of states needed
        :param delta: Compute in Viterbi
        :return:
        '''
        states = [self.roads.iloc[np.argmax(delta[t - 1])].road]
        print('Calculate time %d states.' % t)
        for i in range(1, t):
            road = states[i - 1]
            neighbors = [r for r in road.get_connections()]
            neighbors.append(road)
            neighbors = [r for r in neighbors if r.pos in self.roads.index]
            neighbors_delta = delta[t - 1 - i, [int(self.roads.loc[r.pos]['index']) for r in neighbors]]
            states.append(neighbors[np.argmax(neighbors_delta)])
        lon = [r.lon for r in reversed(states)]
        lat = [r.lat for r in reversed(states)]
        print('Done.')
        return pd.DataFrame({'dates': self.obs.time[0:t], 'lon': lon, 'lat': lat})

    def plot_each_time_best_trajectory(self, delta):
        '''
        Plot the best trajectory varing process
        :return:
        '''
        plt.close()  # clf() # 清图  cla() # 清坐标轴 close() # 关窗口
        fig = plt.figure()
        # plt.show()
        ax = fig.add_subplot(1, 1, 1)
        plt.grid(True)  # 添加网格
        plt.ion()  # interactive mode on
        ax.plot(self.obs['lon'], self.obs['lat'])
        print('开始仿真')
        for t in range(1, len(self.obs)):
            print('Start plot %d.' % t)
            # The max probability states at time t
            tmp = self.get_states(t, delta)
            ax.plot(tmp['lon'], tmp['lat'])  # 折线图
            ax.set_title('Time %d' % t)
            plt.show()
            # ax.lines.pop(1)  删除轨迹
            # 下面的图,两船的距离
            plt.pause(0.05)
            print('plot %d done.' % t)

    def emission_score(self, road: Vertex, lon, lat):
        dist = cal_distance([road.lon, road.lat], [lon, lat]) / 1000
        # return exp(- dist**2) if dist < 2 else 0
        return 2+exp(- dist**2) #if dist < 2 else 0

    def shrinking_roads(self, roads, lon_max, lon_min, lat_max, lat_min, lon_tol=0.02, lat_tol=0.02):
        '''

        Get a smaller roads graph in a given rectangle.
        Longitude limitation: [lon_min - lon_tol, lon_max + lon_tol)
        Latitude limitation: [lat_min - lat_tol, lat_max + lat_tol)

        :param roads: A DataFrame with variable road<Vertex>
        :param lon_max: Maximum longitude.
        :param lon_min: Minimum longitude.
        :param lat_max: Maximum latitude.
        :param lat_min: Minimum latitude.
        :param lon_tol: Longitude tolerance.
        :param lat_tol: Latitude tolerance.
        :return: Filter roads not int he rectangle and return the rest DataFrame.
        '''
        valid_roads_ind = [lon_min-lon_tol < r.lon < lon_max+lon_tol
                           and lat_min-lat_tol < r.lat < lat_max+lat_tol for r in roads.road]
        return roads.iloc[valid_roads_ind]

    def get_times(self):
        return [t for t in self.obs.time]

    def interpolate(self, obs, interval=1):
        '''
        To interpolate observables within one second.
        :param obs: A DataFrame [time<datetime>, lon, lat]
        :param interval: Interval seconds. <Int>
        :return:
        '''
        res = obs
        for i in range(len(obs)-1):
            if (obs.time.iloc[i+1] - obs.time.iloc[i]).total_seconds != 1:
                res = pd.concat([res, self.interpolate_single(obs.iloc[i], obs.iloc[i+1])])

        res = res.sort_values('time')
        res = res.iloc[np.array(range(len(res))) % interval == 0]
        res = res.reset_index(drop=True)
        return res

    def interpolate_single(self, obs1, obs2):
        '''
        Interpolate data between time1 and time2 per second.
        :param obs1: The start row in obs DataFrame [time<datetime>, lon, lat]
        :param obs2: The end row in obs DataFrame [time<datetime>, lon, lat]
        :return: return the DataFrame need to interpolate.
        e.x: obs1 = [2016-03-30 10:56:18, 0, 0]
             obs2 = [2016-03-30 10:56:21, 3, 3]
             return = [[2016-03-30 10:56:19, 1, 1],
                       [2016-03-30 10:56:20, 2, 2]]
        '''
        # Number of row need to be interpolated
        n_interpolate = int((obs2.time - obs1.time).total_seconds()) - 1

        time = [obs1.time + timedelta(seconds=i+1) for i in range(n_interpolate)]
        lon = np.linspace(obs1.lon, obs2.lon, n_interpolate+2)[1:n_interpolate+1]
        lat = np.linspace(obs1.lat, obs2.lat, n_interpolate+2)[1:n_interpolate+1]
        return pd.DataFrame({'time': time, 'lon': lon, 'lat': lat})

    def smoothing(self, obs, window_len):
        '''
        Smoothing the observables recommending after interpolated.
        :param obs: A DataFrame [time<datetime>, lon, lat]
        :param window_len: Window length of moving average. Window_len = int(window_len / 2)
        :return: A DataFrame after smoothing.
        res[t] = obs[(t-half_len):(t+half_len)].mean()
        Attension: At the beginning and ending of the res, there is no enough observalbes.
                   But we also smooth them with a small window length.
        '''
        if window_len == 0:
            return obs
        half_len = int(window_len / 2)
        n = len(obs)
        if half_len >= n - half_len:
            print('Number of observables %d must bigger than window_len %d' % (n, window_len))
            return obs
        times = obs.time
        window_len = half_len * 2 + 1
        lon = [obs.lon[i:half_len].mean() for i in range(0, half_len)]
        lat = [obs.lat[i:half_len].mean() for i in range(0, half_len)]
        for t in range(n-window_len+1):
            lon.append(obs.lon[t:(t+window_len)].mean())
            lat.append(obs.lat[t:(t+window_len)].mean())
        lon.extend([obs.lon[i:n].mean() for i in range((n - half_len), n)])
        lat.extend([obs.lat[i:n].mean() for i in range((n - half_len), n)])
        out = pd.DataFrame({'time': times, 'lon': lon, 'lat': lat})
        return out

    def get_directions(self, obs, window_len):
        '''
        Get average directions with a given window length (seconds).
        :param obs: An observables DataFrame with ['time', 'lon', 'lat']
        :param window_len: Window length.
        To get the direction at time t, average obs[(t+1):(t+window_len)] and calculate the angle between it and obs[t]
        :return:
        '''
        if not self.is_direction:
            return [None]*(self.T - 1)
        direction = []
        for t in range(self.T - 2):
            x = obs.lon[(t+1):min(self.T, (t+window_len))].mean() - obs.lon[t]
            y = obs.lat[(t+1):min(self.T, (t+window_len))].mean() - obs.lat[t]
            direction.append(np.array([x, y]) / self.euclidean_distance(x, y))
        direction.append(direction[len(direction)-1])
        return direction

    def transition_score(self, road1, road2, direction=None, penalty=0.9):
        '''
        Get transition score from road1 to road2
        :param road1: The start road<Vertex>.
        :param road2: The end road<Vertex>.
        :param direction: The guide direction. If None, don't consider direction information.
        :param penalty: Penalty for wrong direction.
        :return: A scale value.
        '''
        if direction is not None:
            geox = road2.lon - road1.lon
            geoy = road2.lat - road1.lat
            geox, geoy = self.standardization(geox, geoy)
            dist = self.euclidean_distance((geox - direction[0]), (geoy - direction[1]))
            if dist < sqrt(2):
                return 1
            else:
                return 1
                # return 1 - penalty
        else:
            return 1

    def euclidean_distance(self, x, y):
        return sqrt(x**2 + y**2)

    def standardization(self, x, y):
        return np.array([x, y]) / self.euclidean_distance(x, y)


def get_obs_data(df, type='cell'):
    '''
    Get observables data from test data and cells data
    :param df: A DataFrame
    :param type: 'cell' or 'GPS'
                If type == 'cell', df is a DataFrame with ['time'<datetime>, 'cell_id']
                If type == 'GPS', df is a DataFrame with ['time'<datetime>, 'lon', 'lat']
    :return: An observalbes DataFrame with ['time'<datetime>, 'lon', 'lat']
    '''
    if type == 'cell':
        cells = get_wuhu_cellSheet()
        obs = df[['time', 'cell_id']]
        obs['cell_id'] = [str(cell) for cell in obs.cell_id]
        obs = obs.merge(cells, how='left')
        obs = obs[['time', 'lon', 'lat']]
    else:
        obs = df[['time', 'lon', 'lat']]
        obs.lon = obs.lon + 0.012
        obs.lat = obs.lat + 0.004
    obs = obs.dropna()
    return obs


def output_obs_data(trail, direction, n, type='cell', interpolate_interval=5, smoothing=20, interpolate=True):
    '''
    Output observables data after pre-processing. For test wuhu data
    :param trail: <Int> The i_th trail
    :param direction: <-1/1> -1 means reverse side; 1 means front.
    :param n: The n_th test of this trail. If None, return all the tests of the trail.
    :param type: 'cell' or 'GPS'. See details in function get_obs_data.
    :param interpolate_interval: Interval seconds of processed observables. Default = 1.
    :param smoothing: Window length of smoothing. If smoothing == 0, do not smooth.
    :param interpolate:
    :return: None.
    '''
    df = get_road_tests(trail, direction, n)
    graph = get_graph('../../data/road_test/link_baidu_wuhu.txt')
    obs = get_obs_data(df, type=type)
    track = CTrack(obs, graph, smoothing=smoothing, interpolate_interval=interpolate_interval, interpolate=interpolate)
    out = track.obs
    out['dates'] = out.time
    direction = u'正向' if direction == 1 else u'反向'
    if interpolate:
        output_data_js(out, '../../res/MapMatching_test/processed_%s/wuhu_线路%d_%s%d_smoothing=%d_window=%d.js'
                       % (type, trail, direction, n, smoothing, interpolate_interval))
    else:
        output_data_js(out, '../../res/MapMatching_test/raw_%s/wuhu_线路%d_%s%d.js' % (type, trail, direction, n))


def output_pre_road_data(trail, direction, n, type='cell', interpolate_interval=1,
                         smoothing=0, penalty=0.9, direction_window=15):
    '''
    Output predicted road data with CTrack. For test wuhu data.
    :param trail: <Int> The i_th trail
    :param direction: <-1/1> -1 means reverse side; 1 means front.
    :param n: The n_th test of this trail. If None, return all the tests of the trail.
    :param type: 'cell' or 'GPS'. See details in function get_obs_data.
    :param interpolate_interval: Interval seconds of processed observables. Default = 1.
    :param smoothing: Window length of smoothing. If smoothing == 0, do not smooth.
    :param penalty:
    :param direction_window:
    :return: None.
    '''
    df = get_road_tests(trail, direction, n)
    graph = get_graph('../../data/road_test/link_baidu_wuhu.txt')
    obs = get_obs_data(df, type=type)
    track = CTrack(obs, graph, smoothing=smoothing, interpolate_interval=interpolate_interval,
                   penalty=penalty, direction_window=direction_window)
    states = track.viterbi()
    direction = u'正向' if direction == 1 else u'反向'
    output_data_js(states, '../../res/MapMatching_test/pre_%s/wuhu_pre_线路%d_%s%d_smoothing=%d_window=%d.js'
                   % (type, trail, direction, n, smoothing, interpolate_interval))

def get_done_trail():
    my_path = '../../res/MapMatching_test/pre_cell'
    files = os.listdir(my_path)
    files = [file for file in files if '.js' in file and 'smoothing=20' in file]
    pred_trail = []
    for file in files:
        trail = int(re.search(r'线路(.+?)_', file).group(1))
        n = int(re.search(r'向(.+?)_', file).group(1))
        direction = 1 if re.search(r'_(.?)向', file).group(1) == '正' else -1
        pred_trail.append([trail, direction, n])
    return pred_trail


if __name__ == '__main__':
    # output_pre_road_data(6, -1, 8, smoothing=20, interpolate_interval=5)
    # output_pre_road_data(3, 1, 1, type='GPS', smoothing=5, interpolate_interval=5,
    #                      penalty=0)

    # trail7 逆向5有问题
    # pred_trail = get_done_trail()
    pred_trail = []
    for n in range(1, 11):
        for direction in [-1, 1]:
            for trail in range(1, 9):
                if [trail, direction, n] in pred_trail:
                    print('------ Pass(trail=%d, direction=%d, n=%d) ------' % (trail, direction, n))
                    continue
                else:
                    print('------ Start(trail=%d, direction=%d, n=%d) ------' % (trail, direction, n))
                # cell raw trajectory
                output_obs_data(trail, direction, n, interpolate=False, type='GPS')

                # processed trajectory
                # output_obs_data(trail, direction, n, smoothing=20, interpolate_interval=5)
                # output_obs_data(trail, direction, n, smoothing=0, interpolate_interval=5)

                # predict trajectory
                # output_pre_road_data(trail, direction, n, smoothing=0, interpolate_interval=5)
                # output_pre_road_data(trail, direction, n, interpolate_interval=5, smoothing=20)

