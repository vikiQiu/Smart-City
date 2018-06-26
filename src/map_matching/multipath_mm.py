'''
0408 Multiple DTW -> Get their index order
'''
__author__ = 'Victoria'
# 2018-03-05

import numpy as np
import pandas as pd
import os
import re
from datetime import timedelta
from math import exp, sqrt
from itertools import combinations, permutations
import json
import matplotlib.pyplot as plt
from src.map_matching.read_data import get_wuhu_cellSheet, get_road_tests
from src.pre_processing.road_network import Graph, get_graph, Vertex
from src.pre_processing.read_data import get_cellSheet
from src.pre_processing.funs import cal_distance, get_datetime, output_geo_js, output_data_js, interpolate
from src.map_matching.evaluation import get_test_obs
from src.map_matching.CTrack import CTrack
from src.work.oneuser_alldata_0307 import trajectories_tojson , save_json, save_js
from src.single_analysis.user_trajectory import deal_same_obs


class Multipath_HMM(CTrack):
    def __init__(self,
                 obses,
                 g: Graph,
                 interpolate_interval=1,
                 smoothing=0,
                 is_direction=False,
                 direction_window=5,
                 penalty=0.9,
                 fill_num=None,
                 interpolate=True):
        '''
        Initialize CTrack
        :param obses: A list of DataFrame [time, lon, lat], time spacing may not be one second.
        :param g: Road network [Graph]
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
        self.obses = obses
        self.mean_obs = Observations(obses).get_mean_obs(fill_num)
        super().__init__(self.mean_obs, g, interpolate_interval, smoothing,
                         is_direction, direction_window, penalty, interpolate)


class DTW:
    '''
    Dynamic Time Wrapping Algorithm.
    '''
    def __init__(self, ts1, ts2, is_norm=False):
        '''
        :param ts1: Time series 1 with m time periods. A list [[lon, lat]].
        :param ts2: Time series 2 with n time periods. A list.
        :param is_norm: whether normalization before dtw.
        '''
        if is_norm:
            self.ts1 = self.normalization(ts1)
            self.ts2 = self.normalization(ts2)
        else:
            self.ts1 = ts1
            self.ts2 = ts2
        self.ots1, self.ots2 = ts1, ts2
        self.m = len(ts1)
        self.n = len(ts2)
        self.directions = [[0, 1], [1, 1], [1, 0]]

    def normalization(self, ts):
        ts = np.matrix(ts).T
        m = ts.mean(axis=1)
        std = ts.std(axis=1)
        ts[0] = (ts[0] - m[0]) / std[0]
        ts[1] = (ts[1] - m[1]) / std[1]
        return ts.T.tolist()

    def dtw(self):
        mat = np.ones((self.m, self.n)) + np.inf
        mat_ind = {}
        start_nodes = [(0, 0)]
        mat[start_nodes[0]] = 0
        layer_cnt = 0
        end_node = (self.m-1, self.n-1)

        print('m = %d, n = %d' % (self.m, self.n))
        while len(start_nodes) != 0:
            next_nodes = []
            for node in start_nodes:
                for dir in self.directions:
                    succ = (node[0] + dir[0], node[1] + dir[1])
                    if 0 <= succ[0] < self.m and 0 <= succ[1] < self.n:
                        new_dist = mat[node] + cal_distance(self.ts1[succ[0]], self.ts2[succ[1]])
                        if succ in mat_ind.keys():
                            if new_dist < mat[succ]:
                                mat_ind[succ] = [node]
                                mat[succ] = new_dist
                            elif new_dist == mat[succ]:
                                mat_ind[succ].append(node)
                        else:
                            mat_ind[succ] = [node]
                            mat[succ] = new_dist
                        if succ not in next_nodes:
                            next_nodes.append(succ)
            start_nodes = next_nodes
            layer_cnt += 1
            print('Layer %d is finished' % layer_cnt)

        print('Minimum distance = %f' % mat[end_node])

        return mat, mat_ind

    def is_in_bound(self, i, j):
        return 0 <= i < self.m and 0 <= j < self.n

    def dtw_boundary(self):
        '''
        Get a rough upper boundary
        :return:
        '''
        min_dist = 0
        node = [0, 0]
        # for i in range(max(self.m, self.n)-1):
        #     nxt = [node[0] + 1, node[1] + 1]
        #     if nxt[0] >= self.m:
        #         nxt[0] = self.m - 1
        #     if nxt[1] >= self.n:
        #         nxt[1] = self.n - 1
        #     max_dist += cal_distance(self.ts1[nxt[0]], self.ts2[nxt[1]])
        #     node = nxt
        while node != [self.m - 1, self.n - 1]:
            value = np.inf
            for dir in self.directions:
                tmp = [node[0] + dir[0], node[1] + dir[1]]
                if not self.is_in_bound(tmp[0], tmp[1]):
                    continue
                tmp_value = cal_distance(self.ts1[tmp[0]], self.ts2[tmp[1]])
                if tmp_value < value:
                    nxt = tmp
                    value = tmp_value
            min_dist += value
            node = nxt

        return min_dist

    def get_dist_matrix(self, ts1, ts2):
        dist_mat = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                dist_mat[i, j] = cal_distance(ts1[i], ts2[j])
        return dist_mat

    @staticmethod
    def get_derivatives(ts):
        ts.insert(0, ts[0])
        ts.append(ts[-1])
        dts = [[(ts[i+1][0] - ts[i-1][0])/2, (ts[i+1][1] - ts[i-1][1])/2] for i in range(1, len(ts) - 1)]
        return dts

    def get_derivatives_dist_matrix(self, ts1, ts2):
        dts1 = self.get_derivatives(ts1)
        dts2 = self.get_derivatives(ts2)
        return self.get_dist_matrix(dts1, dts2)

    def quick_dtw(self, is_derivatives=False):
        mat = np.ones((self.m, self.n)) + np.inf
        if is_derivatives:
            dist_mat = self.get_derivatives_dist_matrix(self.ts1, self.ts2)
        else:
            dist_mat = self.get_dist_matrix(self.ts1, self.ts2)
        mat_ind = {}
        start_nodes = [(0, 0)]
        end_node = (self.m - 1, self.n - 1)
        mat[start_nodes[0]] = 0
        layer_cnt = 0

        max_dist = self.dtw_boundary()

        print('m = %d, n = %d, max_dist = %f' % (self.m, self.n, max_dist))
        while len(start_nodes) != 0:
            next_nodes = []
            for node in start_nodes:
                if max_dist < mat[node] < np.inf:
                    continue
                for dir in self.directions:
                    succ = (node[0] + dir[0], node[1] + dir[1])
                    if 0 <= succ[0] < self.m and 0 <= succ[1] < self.n:
                        new_dist = mat[node] + dist_mat[succ]
                        if new_dist > max_dist:
                            continue
                        if succ in mat_ind.keys():
                            if new_dist < mat[succ]:
                                mat_ind[succ] = [node]
                                mat[succ] = new_dist
                            elif new_dist == mat[succ]:
                                mat_ind[succ].append(node)
                        else:
                            mat_ind[succ] = [node]
                            mat[succ] = new_dist
                        if succ not in next_nodes:
                            next_nodes.append(succ)
            start_nodes = next_nodes
            layer_cnt += 1
            if len(mat[mat != np.inf]) != 0:
                print('Layer %d is finished. Max = %f' % (layer_cnt, mat[mat != np.inf].max()))

        print('Minimum distance = %f' % mat[end_node])

        return mat, mat_ind

    def shortest_path_dtw(self, is_derivatives=False):
        '''
        Solving DTW with shortest path algorithm
        :param is_derivatives:
        :return:
        '''
        mat = np.ones((self.m, self.n)) + np.inf
        if is_derivatives:
            dist_mat = self.get_derivatives_dist_matrix(self.ts1, self.ts2)
        else:
            dist_mat = self.get_dist_matrix(self.ts1, self.ts2)
        mat_ind = {}
        mat[0, 0] = 0

        for layer in range(1, self.m + self.n):
            for i in range(layer+1):
                node = (i, layer-i)
                if not self.is_in_bound(node[0], node[1]):
                    continue
                dist = dist_mat[node]
                for dir in self.directions:
                    lst_node = (node[0] - dir[0], node[1] - dir[1])
                    if not self.is_in_bound(lst_node[0], lst_node[1]):
                        continue
                    tmp_value = mat[lst_node] + dist
                    if tmp_value < mat[node]:
                        mat[node] = tmp_value
                        mat_ind[node] = [lst_node]
                    elif tmp_value == mat[node]:
                        mat_ind[node].append(lst_node)

        # print('Minimum distance = %f' % mat[self.m-1, self.n-1])

        return mat, mat_ind

    def get_trace(self, mat_ind):
        '''
        Now only get one best trace
        :param mat_ind:
        :return:
        '''
        trace = [(self.m-1, self.n-1)]
        node = trace[0]
        while node != (0, 0):
            last_node = mat_ind[node][0]
            trace.append(last_node)
            node = last_node

        # Print trace
        trace_mat = np.zeros((self.m, self.n))
        for t in trace:
            trace_mat[t[0], t[1]] = 1
        # print(trace_mat)

        return trace, trace_mat

    def get_new_ts(self, mat_ind=None):
        if mat_ind is None:
            _, mat_ind = self.shortest_path_dtw()
        trace, _ = self.get_trace(mat_ind)
        ts1, ts2 = self.ts1, self.ts2
        nts1, nts2 = [ts1[0]], [ts2[0]]
        for t in trace[::-1]:
            nts1.append(ts1[t[0]])
            nts2.append(ts2[t[1]])
        return nts1, nts2

    def get_average_dist(self, mat_ind=None):
        if mat_ind is None:
            _, mat_ind = self.shortest_path_dtw()
        nts1, nts2 = self.get_new_ts(mat_ind)
        dist = 0
        for t in range(len(nts1)):
            dist += cal_distance(nts1[t], nts2[t])
        return dist / len(nts1)

    def get_median_dist(self, mat_ind=None):
        if mat_ind is None:
            _, mat_ind = self.shortest_path_dtw()
        nts1, nts2 = self.get_new_ts(mat_ind)
        dist = []
        for t in range(len(nts1)):
            dist.append(cal_distance(nts1[t], nts2[t]))
        return np.median(dist)

    def get_middle_ts(self, mat_ind=None):
        nts1, nts2 = self.get_new_ts(mat_ind)
        return ((np.array(nts1) + np.array(nts2)) / 2).tolist()

    def plot(self, trace, title='DTW'):
        ts1 = np.matrix(self.ots1).T.tolist()
        ts2 = np.matrix(self.ots2).T.tolist()
        plt.title(title)
        plt.plot(ts1[0], ts1[1], color='blue', linewidth="0.8")
        plt.plot(ts2[0], ts2[1], color='red', linewidth="0.8")
        for line in trace:
            p1 = self.ots1[line[0]]
            p2 = self.ots2[line[1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth="0.4")
        plt.grid(True, linestyle="--", color="black", linewidth="0.4")
        plt.savefig('../../res/work/0401_ddtw/pic/%s.png' % title)
        plt.show()


class MultiDTW:
    def __init__(self, tses):
        '''
        Multiple trajectories dynamic time wrapping
        :param tses: A list of ts. [ts1, ..., tsn]. tsi = [[lon, lat]]
        :var self.k: Number of time series
        :var self.dim: Length of each time series
        :var self..directions: All direction allowed
        '''
        self.tses = tses
        self.k = len(tses)
        self.dim = tuple([len(ts) for ts in tses])
        self.directions = self.get_directions(len(tses))

    @staticmethod
    def dtw_tses_single_time(tses):
        '''
        Do DTW between each two time series, and return a list of new time series.
        :param tses:
        :return:
        '''
        inds = list(combinations(np.arange(len(tses)), 2))
        ntses = []
        for ind in inds:
            ts1 = tses[ind[0]]
            ts2 = tses[ind[1]]
            dtw = DTW(ts1, ts2)
            ntses.append(dtw.get_middle_ts())

    @staticmethod
    def get_directions(k):
        '''
        Get valid directions with dimension k.
        :param k:
        :return: A list.
        e.x: k = 2
        return [[1, 0], [0, 1], [1, 1]]
        k = 3
        return [[1, 0, 0], [0, 1, 0], [1, 1, 1]
        '''
        res = list(set(combinations([0, 1]*k, k)))
        res.remove(tuple([0]*k))
        return res

    def is_in_bound(self, pos):
        for k in range(len(self.dim)):
            if not 0 <= pos[k] < self.dim[k]:
                return False
        return True

    def cal_multi_distance(self, ind):
        '''
        ind is a tuple
        :param ind:
        e.x:
            ind = (1, 2, 4)
            Calculate distance between ts0_1, ts1_2, ts2_4.
        :return:
        '''
        ind = list(ind)
        combs = list(combinations(range(len(ind)), 2))
        dist = 0
        for comb in combs:
            ts1_ind = comb[0]
            ts2_ind = comb[1]
            ts1_t = ind[ts1_ind]
            ts2_t = ind[ts2_ind]
            dist += cal_distance(self.tses[ts1_ind][ts1_t], self.tses[ts2_ind][ts2_t])
        return dist / len(combs)

    def dtw_tses(self):
        '''
        Multiple DTW
        :return:
        '''
        mat = np.ones(self.dim) + np.inf
        mat_ind = {}
        mat[tuple([0]*self.k)] = 0

        for layer in range(1, sum(self.dim)):
            inds = self.get_ksum(layer)
            for node in inds:
                if not self.is_in_bound(node):
                    continue
                dist = self.cal_multi_distance(node)
                for dir in self.directions:
                    lst_node = tuple((np.array(node) - np.array(dir)).tolist())
                    if not self.is_in_bound(lst_node):
                        continue
                    tmp_value = mat[lst_node] + dist
                    if tmp_value < mat[node]:
                        mat[node] = tmp_value
                        mat_ind[node] = [lst_node]
                    elif tmp_value == mat[node]:
                        mat_ind[node].append(lst_node)
            if len(mat[mat != np.inf]) != 0:
                print('Layer %d is finished. Max = %f' % (layer, mat[mat != np.inf].max()))

        print('Minimum distance = %f' % mat[tuple([-1]*self.k)])

        return mat, mat_ind

    def get_trace(self, mat_ind=None):
        '''
        Now only get one best trace
        :param mat_ind:
        :return:
        '''
        if mat_ind is None:
            _, mat_ind = self.dtw_tses()
        trace = [tuple([k-1 for k in self.dim])]
        node = trace[0]
        while node != tuple([0]*self.k):
            last_node = mat_ind[node][0]
            trace.append(last_node)
            node = last_node

        return trace

    def plot(self, title='DTW of multiple TS', trace=None):
        if trace is None:
            trace = self.get_trace()
        plt.title(title)
        colors = ['red', 'blue', 'cyan', 'orange']
        lin_colors = ['black', 'brown', 'purple', 'gray']
        for i, ts in enumerate(self.tses):
            ts = np.matrix(ts).T.tolist()
            plt.plot(ts[0], ts[1], color=colors[i % len(colors)], linewidth="2")

        for i, line in enumerate(trace):
            combs = list(combinations(range(len(line)), 2))
            for comb in combs:
                ts1_ind = comb[0]
                ts2_ind = comb[1]
                ts1_t = line[ts1_ind]
                ts2_t = line[ts2_ind]
                p1 = self.tses[ts1_ind][ts1_t]
                p2 = self.tses[ts2_ind][ts2_t]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=lin_colors[i % len(lin_colors)], linewidth="0.5")
        plt.grid(True, linestyle="--", color="black", linewidth="0.4")
        plt.savefig('../../res/work/0401_ddtw/pic/%s.png' % title)
        plt.show()

    def get_ksum(self, n):
        res = get_ksum(n, self.k)
        return [tuple(x) for x in res]


class MultiDTWIndx:
    def __init__(self, obses, interval=1):
        '''
        Get all the obses' index order
        :param obses: Input a list of DataFrame -> [pd.DataFrame(['dates', 'lon', 'lat'])]
                      Must ordered by dates.
        :param interval: Interpolate interval. Default 1.
        '''
        self.obses = self.set_origin_destination(obses)
        self.tses, self.new_obses = self.interpolate(interval)
        self.interval = interval

    def set_origin_destination(self, obses):
        '''
        Set all the observations have the same origin point and destination point
        :param obses: [DataFrame['lon', 'lat', 'dates']]
        :return: A new processed obses
        '''
        origins = [[obs.lon[0], obs.lat[0]] for obs in obses]
        destinations = [[list(obs.lon)[-1], list(obs.lat)[-1]] for obs in obses]
        origin = [np.mean([x[0] for x in origins]), np.mean([x[1] for x in origins])]
        destination = [np.mean([x[0] for x in destinations]), np.mean([x[1] for x in destinations])]
        for i in range(len(obses)):
            obses[i].lon[0] = origin[0]
            obses[i].lat[0] = origin[1]
            obses[i].lon[len(obses[i])-1] = destination[0]
            obses[i].lat[len(obses[i]) - 1] = destination[1]
        return obses

    def interpolate(self, interval):
        '''
        Interpolate all the observations with the given interval.
        :param interval:
        :return:
        '''
        tses = []
        new_obses = []
        for obs in self.obses:
            ts, tmp = self.interpolate_single(obs, interval)
            new_obses.append(tmp)
            tses.append(ts)
        return tses, new_obses

    @staticmethod
    def interpolate_single(obs, interval):
        '''
        Interpolate one observation with the given interval.
        :param obs: DataFrame
        :param interval:
        :return: ts: [[lon, lat]]; new_obs: DataFrame
        '''
        new_obs = interpolate(obs, interval, origin_reserve=True)
        ts = [[new_obs.lon[i], new_obs.lat[i]] for i in range(len(new_obs))]
        return ts, new_obs

    def run(self):
        ts0 = self.tses[0]
        new_obs = self.new_obses[0]
        # new_obs.lon = [round(x, 7) for x in new_obs.lon]
        # new_obs.lat = [round(x, 7) for x in new_obs.lat]
        new_obs['traj'] = -1
        for i, ts in enumerate(self.tses):
            obs = self.obses[i]
            dtw = DTW(ts0, ts, is_norm=False)
            nts0, nts = dtw.get_new_ts()
            # nts = [[round(x[0], 7), round(x[1], 7)] for x in nts]
            # nts0 = [[round(x[0], 7), round(x[1], 7)] for x in nts0]
            # obs.lon = [round(x, 7) for x in obs.lon]
            # obs.lat = [round(x, 7) for x in obs.lat]
            for t in range(len(obs)):
                # if [obs.lon[t], obs.lat[t]] in nts:
                ind = nts.index([obs.lon[t], obs.lat[t]])
                ind2 = (new_obs.lon == nts0[ind][0]) & (new_obs.lat == nts0[ind][1])
                new_obs.lon[ind2] = obs.lon[t]
                new_obs.lat[ind2] = obs.lat[t]
                new_obs.traj[ind2] = i
            print(new_obs[round(new_obs.lon, 6) == 117.229595])
            print(len(new_obs))
        new_obs = new_obs[new_obs.traj != -1]
        return new_obs

    def run_one_by_one(self):
        '''
        Get the fused observations.
        Method:
            1. Take obses[0] and obses[1], interpolate them and get the ts format.
            2. Fuse them into a new observation(only have observation points, no interpolate points).
            3. Take the new observation and obses[k+1], repeat Step 1 and Step 2.
        :return: A fused observation: DataFrame['lon', 'lat', 'dates']
        '''
        new_obs = self.obses[0]

        for i in range(1, len(self.obses)):
            obs = self.obses[i]
            ts0, obs0 = self.interpolate_single(new_obs, self.interval)
            ts, _ = self.interpolate_single(obs, self.interval)
            dtw = DTW(ts0, ts, is_norm=False)
            nts0, nts = dtw.get_new_ts()

            obs0['traj'] = 0
            for t in range(len(obs)):
                ind = nts.index([obs.lon[t], obs.lat[t]])
                ind2 = (obs0.lon == nts0[ind][0]) & (obs0.lat == nts0[ind][1])
                obs0.lon[ind2] = obs.lon[t]
                obs0.lat[ind2] = obs.lat[t]
                obs0.traj[ind2] = 1
            obs0 = obs0[obs0.traj == 1]
            obs0 = obs0.drop(['traj'], axis=1)
            new_obs = pd.concat([new_obs, obs0])
            # in deal_same_obs already has sort_values() and reset_index() method
            new_obs = deal_same_obs(new_obs)

        return new_obs[['lon', 'lat', 'dates']]

    def plot(self, new_obs=None):
        if new_obs is None:
            new_obs = self.run_one_by_one()
        print(new_obs)
        plt.title('DTW Index Sort')
        for obs in self.obses:
            plt.plot(obs.lon, obs.lat, linewidth="3")

        plt.plot(new_obs.lon, new_obs.lat, color='red', linewidth="1")

        plt.show()


def get_ksum(n, k, tmp_number=[], res=[]):
    '''
    K natural numbers sums up is n. Already given res (a list of natural numbers)
    :param n: The sum of k numbers.
    :param k: Number of natural numbers.
    :param tmp_number: Now the given numbers.
    :param res: The Final result. Always input []
    :return:
    '''
    if len(tmp_number) == k - 1:
        tmp_number.append(n - sum(tmp_number))
        return [tmp_number]
    for i in range(n-sum(tmp_number)+1):
        ntmp_number = tmp_number.copy()
        ntmp_number.append(i)
        a = get_ksum(n, k, ntmp_number, res)
        if len(tmp_number) == k - 2:
            res.extend(a)
        else:
            res = a

    return res


class Observations:
    def __init__(self, obses):
        '''
        List of observations.
        :param obses: A list of DataFrame [dates, lon, lat].
        :var self.obses: A list of Observation.
        '''
        self.obses = []
        for i in range(len(obses)):
            self.obses.append(Observation(obses[i]))
        self.mean_obs = self.get_mean_obs()

    def plot_obses(self):
        for obs in self.obses:
            plt.plot(obs.lon, obs.lat)

    def len_obses(self):
        l = []
        for obs in self.obses:
            l.append(len(obs))
        return l

    def fill_with_dist(self, num):
        '''
        Let all observations filled with distanc
        :param num:
        :return:
        '''
        obses = self.obses
        res = []
        for obs in obses:
            res.append(obs.fill_with_dist(num))
        return res

    def plot_obs(self, fill_num=None):
        data = self.mean_obs
        fig = plt.figure()
        # plt.show()
        ax = fig.add_subplot(1, 1, 1)
        plt.grid(True)  # 添加网格
        plt.ion()  # interactive mode on
        for i in range(len(self.obses)):
            obs = self.obses[i].obs
            ax.plot(obs.lon, obs.lat)
            plt.pause(0.2)
        for i in range(0, len(data), 5):
            ax.plot(data.lon[:i], data.lat[:i], color='red')
            plt.pause(0.01)
        plt.pause(5)

    def get_mean_obs(self, fill_num=None):
        if fill_num is None:
            total_dist = []
            for obs in self.obses:
                total_dist.append(obs.get_total_dist())
            fill_num = np.mean(total_dist) // 10
            print('Fill num = %d' % fill_num)

        obses = self.fill_with_dist(fill_num)
        lon = []
        lat = []

        for t in range(len(obses[0])):
            lon.append(np.mean([obs.lon[t] for obs in obses]))
            lat.append(np.mean([obs.lat[t] for obs in obses]))

        return pd.DataFrame({'lon': lon, 'lat': lat, 'dates': obses[0].dates})

    def save_json(self, save_path, user_id='', is_json=True, fill_num=None, need_mean=True):
        obses = []

        for i in range(len(self.obses)):
            tmp = self.obses[i].obs
            obses.append(tmp)

        if need_mean:
            obses.append(self.get_mean_obs(fill_num))

        trajectories_tojson(obses, user_id, save_path, is_json=is_json)


class Observation:
    def __init__(self, obs):
        '''
        :param obs: A DataFrame observation [dates, lon, lat]
        '''
        self.obs = obs

    def get_obs_len(self):
        return len(self.obs)

    def get_total_dist(self):
        dist = 0
        for i in range(1, len(self.obs)):
            dist += cal_distance([self.obs.lon[i-1], self.obs.lat[i-1]],
                                 [self.obs.lon[i], self.obs.lat[i]])
        return dist

    def fill_with_dist(self, num):
        '''
        Fill the observation with distance.
        :param num: The num of final observation.
        :return: A new observation.
        '''
        total_dist = self.get_total_dist()
        each_dist = total_dist / (num - 1)

        dist = 0

        obs = self.obs
        lon = [obs.lon[0]]
        lat = [obs.lat[0]]
        dates = [obs.dates[0]]
        for i in range(1, len(obs)):
            p1 = [obs.lon[i-1], obs.lat[i-1]]
            p2 = [obs.lon[i], obs.lat[i]]
            d_have = cal_distance(p1, p2)
            d_need = each_dist - dist

            while d_have >= d_need:
                p_need = d_need / d_have
                p1 = [(1 - p_need) * p1[0] + p_need * p2[0], (1 - p_need) * p1[1] + p_need * p2[1]]
                lon.append(p1[0])
                lat.append(p1[1])
                dates.append(obs.dates[i-1] + (obs.dates[i] - obs.dates[i-1]) * p_need)

                dist = 0
                d_have = cal_distance(p1, p2)
                d_need = each_dist - dist

            dist += d_have

        return pd.DataFrame({'lon': lon, 'lat': lat, 'dates': dates})

    def plot_obs(self, fill_num=None, one_by_one=False):
        if fill_num is None:
            data = self.obs
        else:
            data = self.fill_with_dist(fill_num)
        fig = plt.figure()
        # plt.show()
        ax = fig.add_subplot(1, 1, 1)
        plt.grid(True)  # 添加网格
        plt.ion()  # interactive mode on
        if one_by_one:
            for i in range(len(data)):
                ax.plot(data.lon[:i], data.lat[:i])
                plt.pause(0.2)
        else:
            ax.plot(data.lon, data.lat)
        plt.pause(2)

    def cal_time_interval(self):
        total_time =  (max(self.obs.dates) - min(self.obs.dates)).total_seconds()
        return total_time / (len(self.obs) - 1)


def multi_map_matching(obses, graph, save_name,
                       mdtw_interval=5,
                       mm_smoothing=10,
                       mm_interpolate_interval=10,
                       save_dir='../../res/work/0417_multi_mm/'):
    mdtw = MultiDTWIndx(obses, interval=mdtw_interval)
    new_obs = mdtw.run_one_by_one()
    print(new_obs)
    output_data_js(new_obs,  save_dir+save_name+'_raw.js')
    track = CTrack(new_obs, graph, smoothing=mm_smoothing, interpolate_interval=mm_interpolate_interval)
    output_data_js(track.obs, save_dir+save_name+'_processed.js')
    states = track.viterbi()
    output_data_js(states, save_dir+save_name+'.js')
    return states


def single_map_matching(obs, graph, save_name,
                        mm_smoothing=10,
                        mm_interpolate_interval=10,
                        save_dir='../../res/work/0417_multi_mm/'):
    output_data_js(obs,  save_dir+save_name+'_raw.js')
    track = CTrack(obs, graph, smoothing=mm_smoothing, interpolate_interval=mm_interpolate_interval)
    output_data_js(track.obs, save_dir+save_name+'_processed.js')
    states = track.viterbi()
    output_data_js(states, save_dir+save_name+'.js')
    return states


def dist_trajectories(ts1, ts2):
    assert len(ts2) == len(ts1)
    dist = 0
    for i in range(len(ts1)):
        dist += cal_distance(ts1[i], ts2[i])
    return dist / len(ts1)


def test_observation(trail, direction, n):
    '''
    Test the class observation.
    :param trail: The number of trail.
    :param direction: [0/1]
                      If direction == 1: positive direction
                      else: negative direction
    :param n: The number of test.
    :return:
    '''
    obs = Observation(get_test_obs(trail, direction, n))
    print(obs.obs)
    print('-'*10)
    print(obs.fill_with_dist(5))
    print(obs.get_total_dist())
    obs.plot_obs(3)


def test_observations(trail, direction, fill_num):
    '''
    Test observation list.
    :param trail:
    :param direction:
    :param fill_num:
    :return:
    '''
    obses = []
    for n in range(1, 11):
        obses.append(get_test_obs(trail, direction, n))
    obses = Observations(obses)
    # obses.plot_obs(fill_num)
    new_obses = obses.get_mean_obs(fill_num)
    Observation(new_obses).plot_obs()
    print(new_obses)


def test_dtw(user_number, path_need):
    with open('../../res/work/0311_random_trajectories/random_user%d.json' % user_number) as data_file:
        data = json.load(data_file)

    obses = []
    for i in path_need[:2]:
        tmp = data['trajectories'][i]
        dates = [get_datetime(t) for t in tmp['dates']]
        obs = pd.DataFrame({'lon': tmp['lon'], 'lat': tmp['lat'], 'dates': dates})
        obs = interpolate(obs, interval=1)
        obs = [[obs.lon[i], obs.lat[i]] for i in range(len(obs))]
        obses.append(obs)

    dtw = DTW(obses[0], obses[1], is_norm=False)
    mat, mat_ind = dtw.shortest_path_dtw(is_derivatives=True)
    # mat, mat_ind = dtw.quick_dtw(is_derivatives=False)
    # mat, mat_ind = dtw.dtw()
    trace, trace_mat = dtw.get_trace(mat_ind)
    dtw.plot(trace, 'random_user%d_%d_%d DDTW Plot' % (user_number, path_need[0], path_need[1]))
    # ts1, ts2 = dtw.get_new_ts()
    # print(dist_trajectories(ts1, ts2))
    # return trace


def test_muti_dtw(user_number, path_need):
    with open('../../res/work/0311_random_trajectories/random_user%d.json' % user_number) as data_file:
        data = json.load(data_file)

    obses = []
    for i in path_need:
        tmp = data['trajectories'][i]
        dates = [get_datetime(t) for t in tmp['dates']]
        obs = pd.DataFrame({'lon': tmp['lon'], 'lat': tmp['lat'], 'dates': dates})
        obs = interpolate(obs, interval=60)
        obs = [[obs.lon[i], obs.lat[i]] for i in range(len(obs))]
        obses.append(obs)

    dtw = MultiDTW(obses)
    # mat, mat_ind = dtw.dtw_tses()
    dtw.plot()
    # mat, mat_ind = dtw.quick_dtw(is_derivatives=False)
    # mat, mat_ind = dtw.dtw()
    # trace, trace_mat = dtw.get_trace(mat_ind)
    # dtw.plot(trace, 'random_user%d_%d_%d DDTW Plot' % (user_number, path_need[0], path_need[1]))
    # ts1, ts2 = dtw.get_new_ts()
    # print(dist_trajectories(ts1, ts2))
    # return trace


def test_hefei(user_number, path_need):
    with open('../../res/work/0311_random_trajectories/random_user%d.json' % user_number) as data_file:
        data = json.load(data_file)

    obses = []
    for i in path_need:
        tmp = data['trajectories'][i]
        dates = [get_datetime(t) for t in tmp['dates']]
        obs = pd.DataFrame({'lon': tmp['lon'], 'lat': tmp['lat'], 'dates': dates})
        print(Observation(obs).cal_time_interval())
        obses.append(obs)

    obses = Observations(obses)
    # obses.plot_obs()
    obses.save_json('../../res/work/0315_similar_trajectories/random_user%d.json' % user_number,
                    data['user_id'], is_json=True)
    obses.save_json('../../res/work/0315_similar_trajectories/random_user%d.js' % user_number,
                    data['user_id'], is_json=False, fill_num=30)

    return obses


def obs_to_ts(obs, interval=1):
    '''
    Transform observation to ts
    :param obs: DataFrame['lon', 'lat', 'dates']
    :param interval: The interval to interpolate the observation
    :var ts: [[lon, lat]]
    :return: ts
    '''
    obs = interpolate(obs, interval=interval)
    ts = []
    for t in range(len(obs)):
        ts.append([obs.lon[t], obs.lat[t]])

    return ts


def test_multi_mm(user_number, path_need, smoothing=2, interpolate_interval=15, interpolate=True):
    '''
    Test Multi_mm class.
    :param user_number: Random User number use
    :param path_need: Path number used
    :param smoothing:
    :param interpolate_interval:
    :param interpolate:
    :return:
    '''

    # Get observations
    with open('../../res/work/0311_random_trajectories/random_user%d.json' % user_number) as data_file:
        data = json.load(data_file)

    obses = []
    for i in path_need:
        tmp = data['trajectories'][i]
        dates = [get_datetime(t) for t in tmp['dates']]
        obs = pd.DataFrame({'lon': tmp['lon'], 'lat': tmp['lat'], 'dates': dates})
        obses.append(obs)

    # Get graph
    graph = get_graph('../../data/hefei_road/link_baidu.txt')

    matched_obses = []
    mean_obs = Observations(obses).get_mean_obs(30)
    obses.append(mean_obs)
    for obs in obses:
        track = CTrack(obs, g=graph, smoothing=smoothing,
                       interpolate_interval=interpolate_interval,
                       interpolate=interpolate)
        matched_obs = track.viterbi()
        matched_obses.append(matched_obs)

    trajectories_tojson(matched_obses, data['user_id'],
                        '../../res/work/0315_similar_trajectories/multiHMM_random_user%d.js' % user_number,
                        is_json=False)
    trajectories_tojson(matched_obses, data['user_id'],
                        '../../res/work/0315_similar_trajectories/multiHMM_random_user%d.json' % user_number,
                        is_json=True)


def test_multi_index(user_number, path_need):
    with open('../../res/work/0311_random_trajectories/random_user%d.json' % user_number) as data_file:
        data = json.load(data_file)

    obses = []
    for i in path_need:
        tmp = data['trajectories'][i]
        dates = [get_datetime(t) for t in tmp['dates']]
        obs = pd.DataFrame({'lon': tmp['lon'], 'lat': tmp['lat'], 'dates': dates})
        obses.append(obs)

    mdtw = MultiDTWIndx(obses, interval=5)
    new_obs = mdtw.run_one_by_one()
    mdtw.plot(new_obs)
    obses.append(new_obs)
    trajectories_tojson(obses, data['user_id'],
                        '../../res/work/0411_dtw_index_sort/multiHMM_random_user%d.js' % user_number,
                        is_json=False)


# dijkstra算法实现，有向图和路由的源点作为函数的输入，最短路径最为输出
def dijkstra(graph, src=0):
    # 判断图是否为空，如果为空直接退出
    if graph is None:
        return None
    nodes = [i for i in range(len(graph))]  # 获取图中所有节点
    visited=[]  # 表示已经路由到最短路径的节点集合
    if src in nodes:
        visited.append(src)
        nodes.remove(src)
    else:
        return None
    distance={src:0}  # 记录源节点到各个节点的距离
    for i in nodes:
        distance[i]=graph[src][i]  # 初始化
    # print(distance)
    path={src:{src:[]}}  # 记录源节点到每个节点的路径
    k=pre=src
    while nodes:
        mid_distance=float('inf')
        for v in visited:
            for d in nodes:
                new_distance = graph[src][v]+graph[v][d]
                if new_distance < mid_distance:
                    mid_distance=new_distance
                    graph[src][d]=new_distance  # 进行距离更新
                    k=d
                    pre=v
        distance[k]=mid_distance  # 最短路径
        path[src][k]=[i for i in path[src][pre]]
        path[src][k].append(k)
        # 更新两个节点集合
        visited.append(k)
        nodes.remove(k)
        print(visited,nodes)  # 输出节点的添加过程
    return distance,path


def cal_avg_time_interval():
    users = []
    num = []
    intervals = []
    for user_number in range(8, 63):
        with open('../../res/work/0311_random_trajectories/random_user%d.json' % user_number) as data_file:
            data = json.load(data_file)
        for i in range(len(data['trajectories'])):
            tmp = data['trajectories'][i]
            dates = [get_datetime(t) for t in tmp['dates']]
            obs = pd.DataFrame({'lon': tmp['lon'], 'lat': tmp['lat'], 'dates': dates})

            users.append(user_number)
            num.append(i)
            intervals.append(Observation(obs).cal_time_interval())
        out = pd.DataFrame({'user': users, 'num': num, 'time_interval': intervals})
        out.to_csv('../../res/work/0506_big_eye/time_interval/time_interval.csv')

    import seaborn as sns
    sns.set_palette('deep', desat=.6)
    sns.set_context(rc={'figure.figsize': (8, 5)})
    plt.hist(intervals, bins=20)
    plt.grid(True, linestyle="--", color="black", linewidth=".5", alpha=.4)
    plt.title('Cellular Data Time Interval Histogram')
    plt.xlabel('Time interval(s)')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == '__main__':
    # test_observation(1,1,1)
    # test_observations(1, 1, 10)
    # test_hefei(8, [15, 25, 26])
    # test_multi_mm(8, [15, 25, 26])
    # test_dtw(8, [15, 25])
    # print(len(get_ksum(10, 3)))
    # test_muti_dtw(8, [15, 25, 26])
    # test_multi_index(8, [15, 25, 26])
    # test_multi_index(20, [15, 38, 2, 7, 19])
    cal_avg_time_interval()

    # graph_list = [[0, 2, 1, 4, 5, 1],
    #               [1, 0, 4, 2, 3, 4],
    #               [2, 1, 0, 1, 2, 4],
    #               [3, 5, 2, 0, 3, 3],
    #               [2, 4, 3, 4, 0, 1],
    #               [3, 4, 7, 3, 1, 0]]
    #
    # distance, path = dijkstra(graph_list, 0)  # 查找从源点0开始带其他节点的最短路径
    # print(distance, path)

