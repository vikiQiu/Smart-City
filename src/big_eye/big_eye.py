'''
Author: Vikitoria
Date: 2018-05-06
Purpose: Do all the big eye work
Input:
    1. raw signal data: pd.DataFrame['cell_id', 'user_id', 'dates']
    2. cells data: pd.DataFrame['cell_id', 'lon', 'lat']
    => signal data dictionary: {user_id: [pd.DataFrame['dates', 'lon', 'lat']]}
Work:
    1. Segment trajectories <signal_analysis/user_trajectory/get_trajectory>
    2. Trajectory preprocess
    3.
'''
import pandas as pd
import numpy as np
import random
import os
import re
import math
import sys
sys.path.append('/Users/Viki/Documents/yhliu/Smart City')

from src.single_analysis.user_trajectory import deal_same_obs, deal_outlier, deal_pingpong
from src.pre_processing.funs import cal_rough_speed_per_cell2
from src.pre_processing.read_data import get_signals_from_csv
from src.pre_processing.road_network import get_graph
from src.work.density_clustering_0415 import density_clustering, \
    similar_trajectories_clustering, plot_similar_trajectories
from src.work.oneuser_alldata_0307 import trajectories_tojson
from src.map_matching.multipath_mm import multi_map_matching, single_map_matching
from src.map_matching.read_data import get_road_tests
from src.map_matching.evaluation import min_dist


SIGNAL_DIR = '../../res/work/0307_oneuser_alldata/'
CELLS_PATH = '../../res/[0926]cells_process.csv'
USER_PATH = '../../res/work/0307_oneuser_alldata/rand_users_with_test.csv'
WORK_DIR = '../../res/work/0506_big_eye/'
GRAPH_PATH = '../../data/hefei_road/link_baidu.txt'


class BigEye:
    def __init__(self, signal_dir=SIGNAL_DIR,
                 cells_path=CELLS_PATH,
                 user_path=USER_PATH,
                 work_dir=WORK_DIR,
                 graph_path=GRAPH_PATH,
                 user_num=None,
                 max_seconds=60.0 * 30,
                 min_set_len=5, max_distance=3000,
                 max_angle=5, max_speed=200 * 1000 / 3600):
        '''
        A BigEye system environment
        :param signal_dir: The directory saves signal data. Must end with '/'
                           Signal data is a csv, with columns ['cell_id', 'user_id', 'dates']
                           E.x: '../../res/work/0307_oneuser_alldata/'
        :param cells_path: The path of cells csv.
                           Cells data is a csv, with columns ['cell_id', 'lon', 'lat']
                           E.x: '../../res/[0926]cells_process.csv'
        :param user_path: The path of random users csv.
                          Random users data is a csv, with columns ['user_id']
                          E.x: '../../res/work/0307_oneuser_alldata/rand_users.csv'
        :param user_num: If user == None, signals is from the same user.
        :param max_seconds:
        :param min_set_len:
        :param max_distance:
        :param max_angle:
        :param max_speed:
        '''
        self.signal_dir = signal_dir
        self.work_dir = work_dir
        self.graph_path = graph_path
        self.random_user_signal_dir = work_dir + 'random_user_signals/'
        self.check_dir(self.random_user_signal_dir)

        self.user_num = user_num
        self.max_seconds = max_seconds
        self.min_set_len = min_set_len
        self.max_distance = max_distance
        self.max_angle = max_angle
        self.max_speed = max_speed

        # Get cells
        self.cells = self.get_cells(cells_path)

        # Get graph
        self.graph = get_graph(self.graph_path)

        # Random users
        self.users = self.get_users(user_path)

    @staticmethod
    def check_dir(dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def get_cells(cells_path):
        '''
        Get cells data
        :return: A DataFrame['cell_id', 'lat', 'lon']
        '''
        if '.csv' in cells_path:
            cells = pd.read_csv(cells_path)
        else:
            cells = pd.read_csv(cells_path, sep='\t', header=None, names=['cell_id', 'lon', 'lat', 'radius'])
            cells.cell_id = cells.cell_id.astype(str)
        return cells[['cell_id', 'lat', 'lon']]

    @staticmethod
    def get_users(user_path):
        users = pd.read_csv(user_path)
        return users['user_id'].values.tolist()

    def get_test_signal_set(self, trail, direction, n=None):
        # Get singals files' name.
        n = '%d.csv' % n if n is not None else ''
        trail = u'线路' + str(trail)
        direction = u'正向' if direction == 1 else u'逆向' if direction == -1 else ''
        direction = direction + n
        files_ind = [trail in file and direction in file for file in os.listdir(self.signal_dir)]
        files = np.array(os.listdir(self.signal_dir))[files_ind]

        headers = ['dates', 'nothing', 'cell_id', 'user_id', 'service_type', 'web', 'gps_lon',
                   'gps_lat']

        signal_set = []
        for i, file in enumerate(files):
            signal = get_signals_from_csv(self.signal_dir + file, headers)
            signal = pd.merge(signal, self.cells, on='cell_id', how='left')
            if signal.gps_lon.dtype == 'O':
                signal = signal[(signal.gps_lon != 'null') & (signal.gps_lat != 'null')]
            signal = signal.drop(['nothing'], axis=1)
            signal = signal.dropna(axis=0, how='any')
            signal = self.tidy_signals(signal)

            signal_set.append(signal)
            print(file + ' finised.')
        print(len(signal_set))

        self.save_json_js(signal_set, '_test%s%s' % (trail, direction))
        return signal_set

    # def get_test_signal_set(self, trail, direction):


    def get_signals(self, user_num=None):
        file_name = 'random_users.csv' if user_num is None else 'random_user%d.csv' % user_num

        # If already get the user's data, just read from the data
        if file_name in os.listdir(self.random_user_signal_dir):
            signals = get_signals_from_csv(self.random_user_signal_dir+file_name)
            return signals

        # Get singals files' name.
        files = os.listdir(self.signal_dir)
        files = list(filter(re.compile(r'\d.csv').search, files))

        signals = ''
        for i, file in enumerate(files):
            signal = get_signals_from_csv(self.signal_dir + file)

            if user_num is not None:
                signal = signal[signal['user_id'] == self.users[user_num]]
            if i == 0:
                signals = signal
            else:
                signals = pd.concat([signals, signal])
            print(file+' finised.')

        file_name = 'random_users.csv' if user_num is None else 'random_user%d.csv' % user_num
        signals = self.tidy_signals(signals)
        signals.to_csv(self.random_user_signal_dir+file_name, index=False)

        return signals

    @staticmethod
    def tidy_signals(signals):
        '''
        1. Sort signals with <dates>
        2. Reset index of the DataFrame
        :param signals:
        :return:
        '''
        signals = signals.sort_values('dates')
        signals = signals.reset_index(drop=True)
        return signals

    def preprocess_trajectory(self, signal, save_json=False, save_dir='random_user_signal_set/'):
        '''
        Preprocess the trajectory of the given user.
        Process:
            1. Merge the signal and cells data
            2. Drop NA
            3. Deal with the same observation
            4. Segment trajectory
            5. Deal with outlier and pingpong
        :param signal: A DataFrame['dates', 'lon', 'lat']
        :param save_json: whether to save json and js
        :param save_dir: self.work_dir+save_dir
        :return:
        '''
        # Get the user's trajectory
        user_num = self.user_num
        if user_num is not None:
            signal = signal[signal.user_id == self.users[user_num]]

        # Tidy the signals
        signal = self.tidy_signals(signal)
        signal['cell_id'] = signal['cell_id'].astype(str)

        # merge cells data
        signal = pd.merge(signal, self.cells, on='cell_id', how='left')

        # Deal with the same observation
        signal = signal.dropna(axis=1, how='all')
        signal = signal.dropna(axis=0)
        signal = deal_same_obs(signal)
        signal = self.tidy_signals(signal)

        # Segment and deal with outlier and pingpong
        signal_set = self.segment_trajectory(signal, self.max_seconds, self.min_set_len,
                                             self.max_distance, self.max_angle, self.max_speed)

        if save_json:
            save_dir = self.work_dir+save_dir
            self.check_dir(save_dir+'json/')
            self.check_dir(save_dir + 'js/')
            file_name = 'random_users' if user_num is None else 'random_user%d' % user_num
            trajectories_tojson(signal_set, self.users[user_num], save_dir+'json/'+file_name+'.json', is_json=True)
            trajectories_tojson(signal_set, self.users[user_num], save_dir + 'js/'+file_name + '.js', is_json=False)

        return signal_set, signal

    @staticmethod
    def segment_trajectory(signal, max_seconds=60.0*30,
                           min_set_len=5, max_distance=3000,
                           max_angle=5, max_speed=200*1000/3600):
        '''
        Segment trajectory (suggested from the same user):
            1. No longer than <max_seconds/60> min. If more than <max_seconds/60> min, part it.
            2. No longer than <max_distance> m, else part it.
            3. Deal with outlier: Angle bigger than <max_angle> and the speed bigger than max_speed, drop it.
            4. Deal with Pingpong:
                1) p1 -> p2 -> p1 => Replace by p1.
                2) p1 -> p2 -> p1 -> p2=> Replace by mean(p1, p2).
            5. Every trajectory set has at least 5 different samples. If less than 5, delete this trajectory set.
        :param signal: An already tidied DataFrame['dates', 'lon', 'lat']
        :param max_seconds:
        :param min_set_len:
        :param max_distance:
        :param max_angle:
        :param max_speed:
        :return: A signal set(list).
        '''
        ind_sets = []
        ind = [0]
        pre_time = signal['dates'][0]
        speed = cal_rough_speed_per_cell2(signal)

        for i in range(1, len(signal)):
            this_time = signal['dates'][i]

            # Satisfy the first two segment conditions:
            if ((this_time - pre_time).total_seconds() > max_seconds) or (speed.dists[i - 1] > max_distance):
                if len(ind) >= min_set_len:
                    ind_sets.append(ind)
                ind = [i]
            else:
                ind.append(i)
            pre_time = this_time

        signal_sets = []
        for inds in range(len(ind_sets)):
            dealed_signal = deal_outlier(signal.iloc[ind_sets[inds]], max_speed, max_angle)
            dealed_signal = deal_pingpong(dealed_signal)

            # unique points
            tmp = dealed_signal.lon.astype(str) + '|' + dealed_signal.lat.astype(str)
            if len(tmp.unique()) >= 5:
                signal_sets.append(dealed_signal)

        return signal_sets

    @staticmethod
    def stay_points_clustering(signal_set, eps=600):
        '''
        Clustering stay points
        :param: signal_set
        :var labels: -1 means that it's a single point. d \in N means they are in the same cluster.
        :return: processed_signal_set, labels, signal_set
        '''
        # Get stay points from signal_set
        stay_points = []
        for signal in signal_set:
            stay_points.append([signal.lon[0], signal.lat[0]])
            stay_points.append([signal.lon.iloc[-1], signal.lat.iloc[-1]])

        # Stay points clustering
        labels = density_clustering(stay_points, eps)
        print(labels)

        #
        nsignal_set = signal_set.copy()
        for l in range(max(labels)):
            inds = [i for i in range(len(stay_points)) if labels[i] == l]
            if len(inds) == 0:
                continue
            lon, lat = np.mean(np.array(stay_points)[labels == l], axis=0)
            for ind in inds:
                nsignal_set[ind // 2].lon.iloc[-(ind % 2)] = lon
                nsignal_set[ind // 2].lat.iloc[-(ind % 2)] = lat

        return nsignal_set, labels, signal_set

    def find_similar_trajectories(self, signal_set, max_dist=300, interval=5,
                                  save_dir='similar_trajectories/'):
        '''
        Apply density clustering to the real signal data.
        Input: Give the user number and the directory stored this user's observation.
        Output: Get a Trajectory list, in which the similar trajectories will be put in the same list.

        If the origin point or destination point of the trajectory is a single point,
            the trajectory doesn't have a similar trajectory.
        If the origin point and destination point of the trajectory is the same point,
            the trajectory is not a valid trajectory. It will be removed.

        Define similar trajectories:
            1. The start point is in the same clustering
            2. The end point is in the same clustering
        :param signal_set:
        :param max_dist:
        :param interval:
        :param save_dir:
        :return:
        '''
        save_dir = self.work_dir+save_dir
        self.check_dir(save_dir)
        if len(signal_set) == 0:
            return []

        signal_set, labels, _ = self.stay_points_clustering(signal_set)
        res = []

        # Get single trajectories
        for i in range(len(signal_set)):
            if labels[2 * i] == -1 or labels[2 * i + 1] == -1:
                res.append(signal_set[i])

        # TODO: Find similar trajecotries
        for a in range(max(labels) + 1):
            for b in range(max(labels) + 1):
                if a == b:
                    continue
                the_obses = [signal_set[i] for i in range(len(signal_set)) if labels[2 * i] == a and labels[2 * i + 1] == b]
                if len(the_obses) == 0:
                    continue
                if len(the_obses) == 1:
                    res.extend(the_obses)
                    continue
                res.extend(similar_trajectories_clustering(the_obses, interval, max_dist))

        self.check_dir(save_dir+'similar_json')
        self.check_dir(save_dir+'similar_js')
        plot_similar_trajectories(res, self.user_num, save_dir, is_plot=False, save_json=True)

        return res

    def save_json_js(self, signal_set, user_num, save_dir='random_user_signal_set/'):
        save_dir = self.work_dir+save_dir
        self.check_dir(save_dir+'json/')
        self.check_dir(save_dir + 'js/')
        file_name = 'random_users' if user_num is None else 'random_user%s' % str(user_num)
        user_name = self.users[user_num] if type(user_num) == int else user_num
        trajectories_tojson(signal_set, user_name, save_dir+'json/'+file_name+'.json', is_json=True)
        trajectories_tojson(signal_set, user_name, save_dir + 'js/'+file_name + '.js', is_json=False)

    @staticmethod
    def average_interval(signal_set):
        '''
        Calculate average interval
        :param signal_set:
        :return: the average interval (s)
        '''
        intervals = []
        for signal in signal_set:
            # intervals.append((signal.dates.iloc[-1] - signal.dates[0]).total_seconds() / len(signal))
            speed = cal_rough_speed_per_cell2(signal)
            intervals.extend(speed.time.tolist())
        intervals = [x for x in intervals if 1 <= x <= 600]
        if len(intervals) > 50:
            intervals.sort()
            intervals = [x for x in intervals if intervals[4] <= x <= intervals[-4]]

        m = np.mean(intervals)
        std = np.std(intervals)

        print(intervals)
        print('Average Interval = %.2fs; std = %.2f' % (m, std))
        return m, std

    def sample(self, signal_set, time_interval, std):
        '''
        Sample the signal_set
        :param signal_set:
        :param time_interval:
        :return:
        '''
        nsignal_set = []
        for signal in signal_set:
            nsignal = self.sample_signal(signal, time_interval, std)
            nsignal_set.append(nsignal)
        return nsignal_set

    @staticmethod
    def sample_signal(signal, time_interval, std=10):
        '''
        Sample the signal
        :param signal:
        :param time_interval: Calculate the total sample number
        :return:
        '''
        time_interval = np.random.normal(time_interval, std)
        sample_n = math.ceil((signal.dates.iloc[-1] - signal.dates[0]).total_seconds() / time_interval) + 1
        ind = np.random.choice(np.arange(1, len(signal) - 1), sample_n - 2).tolist()
        ind.extend([0, len(signal) - 1])
        nsignal = signal.iloc[ind]
        nsignal = BigEye.tidy_signals(nsignal)
        return nsignal

    def multi_mm(self, signal_set, save_name, interval=5,
                 mm_smoothing=5, mm_interpolate_interval=10,
                 save_dir='multi_mm/'):
        save_dir = self.work_dir+save_dir
        self.check_dir(save_dir)
        states = multi_map_matching(signal_set, self.graph, save_name,
                                    interval, mm_smoothing, mm_interpolate_interval, save_dir)
        return states

    def single_mm(self, signal, save_name, mm_smoothing=5,
                  mm_interpolate_interval=10, save_dir='single_mm/'):
        save_dir = self.work_dir + save_dir
        self.check_dir(save_dir)
        states = single_map_matching(signal, self.graph, save_name,
                                     mm_smoothing, mm_interpolate_interval, save_dir)
        return states

    @staticmethod
    def evaluate_fn_dist(pre, truth):
        '''
        Calculate precision and recall with average min distance
        :param pre: DataFrame['lon', 'lat']
        :param truth:
        :return: precision and recall
        '''
        pre = [[pre.lon[i], pre.lat[i]] for i in range(len(pre))]
        truth = [[truth.lon[i], truth.lat[i]] for i in range(len(truth))]
        precision = np.mean([min_dist(p, truth) for p in pre])
        recall = np.mean([min_dist(p, pre) for p in truth])
        return precision, recall

    @staticmethod
    def get_test_gps(signal):
        gps = signal[['gps_lon', 'gps_lat', 'dates']]
        gps.rename(columns={'gps_lon': 'lon', 'gps_lat': 'lat'}, inplace=True)
        gps.lon = gps.lon.astype(float)
        gps.lat = gps.lat.astype(float)
        gps.lon = gps.lon + 0.012
        gps.lat = gps.lat + 0.004
        return gps


def test_big_eye(user_num=8):
    bigeye = BigEye(user_num=user_num)
    signals = bigeye.get_signals(user_num)
    signal_set, _ = bigeye.preprocess_trajectory(signals, save_json=True)
    bigeye.average_interval(signal_set)
    # print(signal_set)


def test_wuhu():
    user_num = 1000
    signal_dir = '../../data/road_test/wuhu/'
    cells_path = '../../data/cellIdSheets/cellIdSheet_baidu_wuhu.txt'
    graph_path = '../../data/road_test/link_baidu_wuhu.txt'
    bigeye = BigEye(signal_dir, cells_path, user_num=user_num, graph_path=graph_path)
    # signals = bigeye.get_signals(user_num, is_test=True)
    # signal_set, _ = bigeye.preprocess_trajectory(signals, user_num, save_json=True)
    signal_set = bigeye.get_test_signal_set(1, 1)
    signal_set = bigeye.sample(signal_set, time_interval=105, std=10)
    signal_set, _, _ = bigeye.stay_points_clustering(signal_set, eps=600)
    # bigeye.save_json_js(signal_set, user_num, 'test_stay_points/')
    # bigeye.find_similar_trajectories(signal_set)
    bigeye.multi_mm(signal_set[:9], save_name='test1', save_dir='test_multi_mm/')
    # print(bigeye.cells, bigeye.cells.dtypes)


def test_wuhu_index(trail=1, direction=1):
    user_num = 1001
    signal_dir = '../../data/road_test/wuhu/'
    cells_path = '../../data/cellIdSheets/cellIdSheet_baidu_wuhu.txt'
    graph_path = '../../data/road_test/link_baidu_wuhu.txt'
    bigeye = BigEye(signal_dir, cells_path, user_num=user_num, graph_path=graph_path)

    # Given one signal, sample N time.
    sample_n = 5
    m_precision = []
    trails = []
    directions = []
    for trail in range(1, 9):
        for direction in [-1, 1]:
            print('############# Trail %d Direction %d ##############' % (trail, direction))
            test_signal_set = bigeye.get_test_signal_set(trail, direction)
            n = len(test_signal_set)
            for k in range(n):
                signal = test_signal_set[k]
                signal = signal[['lon', 'lat', 'dates']]
                signal_set = [bigeye.sample_signal(signal, 105, 10) for _ in range(sample_n)]

                # origin and destination is the same point
                signal_set, _, _ = bigeye.stay_points_clustering(signal_set, eps=1000)
                m_res = bigeye.multi_mm(signal_set, save_name='test1', save_dir='test_multi_mm/')
                precisions = []
                for i, signal in enumerate(signal_set):
                    states = bigeye.single_mm(signal, save_name='test%d_%d_%d'%(trail, direction, i), save_dir='test_single_mm/')
                    precision, recall = bigeye.evaluate_fn_dist(states, m_res)
                    print(precision, recall)
                    precisions.append(precision)
                m_precision.append(np.mean(precisions))

            trails.extend([trail]*n)
            directions.extend([direction]*n)
            out = pd.DataFrame({'trail': trails, 'direction': directions, 'precision': m_precision})
            out.to_csv(WORK_DIR + 'test_res/precision.csv')


def evaluate_wuhu_multimm_sample(sample_n=5):
    '''
    For each test trajectory(like trail 1, direction 1, number 1),
    sample n sampled_trajectories with time interval ~ N(105, 10).

    mm_precision: Do single MM on the origin test trajectory and calculate the precision
    average_single_precision: Do single MM on the n sampled_trajectory and calculate the average precision
    fuse_mm_precision:
        1. Fuse the n sampled_trajectory into one trajectory
        2. Do single MM
        3. Calculate the precision
    mm_fuse_precision:
        1. Do single MM on each sampled_trajectory
        2. Fuse the MM results into one mm trajectory
        3. Do single MM on the mm trajectory
        4. Calculate the precision

    For each n_sample, output the 160 test trajectories results to the csv

    :param sample_n: sample number
    :return:
    '''
    user_num = 1001
    signal_dir = '../../data/road_test/wuhu/'
    cells_path = '../../data/cellIdSheets/cellIdSheet_baidu_wuhu.txt'
    graph_path = '../../data/road_test/link_baidu_wuhu.txt'
    bigeye = BigEye(signal_dir, cells_path, user_num=user_num, graph_path=graph_path)

    # Given one signal, sample N time.

    fuse_mm_precision = []
    mm_fuse_precision = []
    single_precisions = []
    mm_precision = []
    trails = []
    directions = []
    ks = []
    for trail in range(1, 9):
        for direction in [-1, 1]:
            print('############# Trail %d Direction %d ##############' % (trail, direction))
            test_signal_set = bigeye.get_test_signal_set(trail, direction)
            n = len(test_signal_set)
            for k in range(n):
                ks.append(k)
                signal = test_signal_set[k]

                # GPS MM
                gps = bigeye.get_test_gps(signal)
                gps_state = bigeye.single_mm(gps, 'test_gps%d_%d_%d' % (trail, direction, k), save_dir='test_single_mm/')

                # origin MM
                state = bigeye.single_mm(signal, 'test_cell%d_%d_%d' % (trail, direction, k), save_dir='test_single_mm/')
                precision, _ = bigeye.evaluate_fn_dist(state, gps_state)
                mm_precision.append(precision)

                # Sample 5 trajectories from signal data
                signal = signal[['lon', 'lat', 'dates']]
                signal_set = [bigeye.sample_signal(signal, 105, 10) for _ in range(sample_n)]

                # origin and destination is the same point
                signal_set, _, _ = bigeye.stay_points_clustering(signal_set, eps=1000)
                # Multi-MM result
                m_state = bigeye.multi_mm(signal_set, save_name='test%d_%d_%d'%(trail, direction, k), save_dir='test_multi_mm/')
                precision, _ = bigeye.evaluate_fn_dist(m_state, gps_state)
                fuse_mm_precision.append(precision)

                # Single-MM result
                s_precisions = []
                states = []
                for i, signal in enumerate(signal_set):
                    print('######### Trail %d Direction %d number %d sample %d ##########' % (trail, direction, k, i))
                    state = bigeye.single_mm(signal, save_name='test%d_%d_%d_%d'%(trail, direction, k, i), save_dir='test_single_mm/')
                    states.append(state)
                    precision, recall = bigeye.evaluate_fn_dist(state, gps_state)
                    print(precision, recall)
                    s_precisions.append(precision)
                single_precisions.append(np.mean(s_precisions))
                state = bigeye.multi_mm(states, save_name='test1_mm_first_%d_%d'%(trail, direction), save_dir='test_multi_mm/compare/')
                precision, recall = bigeye.evaluate_fn_dist(state, gps_state)
                mm_fuse_precision.append(precision)

                trails.append(trail)
                directions.append(direction)
                print(trails, directions, ks, fuse_mm_precision, mm_fuse_precision, single_precisions)
                out = pd.DataFrame({'trail': trails, 'direction': directions, 'num': ks,
                                    'mm_precision': mm_precision,
                                    'fuse_mm_precision': fuse_mm_precision,
                                    'average_single_precision': single_precisions,
                                    'mm_fuse_precision': mm_fuse_precision})
                out.to_csv(WORK_DIR + 'test_res/precision_%dsample.csv' % sample_n)


def evaluate_wuhu_multimm(sample_n=5, test_size=10):
    '''
    For each test trajectory(like trail 1, direction 1, number 1),
    :param sample_n: sample number
    :return:
    '''
    user_num = 1001
    signal_dir = '../../data/road_test/wuhu/'
    cells_path = '../../data/cellIdSheets/cellIdSheet_baidu_wuhu.txt'
    graph_path = '../../data/road_test/link_baidu_wuhu.txt'
    bigeye = BigEye(signal_dir, cells_path, user_num=user_num, graph_path=graph_path)

    # Given one signal, sample N time.

    fuse_mm_precision = []
    mm_fuse_precision = []
    single_precisions = []
    trails = []
    directions = []
    ks = []
    for trail in range(1, 9):
        for direction in [-1, 1]:
            # if (trail==1) & (direction==-1):
            #     continue
            print('############# Trail %d Direction %d ##############' % (trail, direction))
            # Get trail and direction signal set, totally 10 trajecotries.
            test_signal_set = bigeye.get_test_signal_set(trail, direction)

            # Get each trajectory map matching and precision
            mm_test_signal_set = []
            mm_test_precision = []
            n = len(test_signal_set)

            # GPS MM
            gps = bigeye.get_test_gps(test_signal_set[0])
            gps_state = bigeye.single_mm(gps, 'test_gps%d_%d' % (trail, direction), save_dir='test_single_mm/')

            for k in range(n):
                signal = test_signal_set[k]
                state = bigeye.single_mm(signal, 'test_cell%d_%d_%d' % (trail, direction, k),
                                         save_dir='test_single_mm/')
                precision, _ = bigeye.evaluate_fn_dist(state, gps_state)
                mm_test_signal_set.append(state)
                mm_test_precision.append(precision)

            # Get sample index
            inds = [sorted(np.random.choice(range(n), sample_n, replace=False)) for _ in range(2*test_size)]
            inds = np.unique(inds, axis=0)[:10]
            for testi, ind in enumerate(inds):
                signal_set = [test_signal_set[_] for _ in ind]
                mm_signal_set = [mm_test_signal_set[_] for _ in ind]
                ks.append(testi)
                single_precisions.append(np.mean([mm_test_precision[_] for _ in ind]))
                trails.append(trail)
                directions.append(direction)

                # origin and destination is the same point
                signal_set, _, _ = bigeye.stay_points_clustering(signal_set, eps=1000)
                # Multi-MM result
                m_state = bigeye.multi_mm(signal_set, save_name='sample%d_test%d_%d_%d' % (sample_n, trail, direction, testi),
                                          save_dir='test_multi_mm/fuse_mm/')
                precision, _ = bigeye.evaluate_fn_dist(m_state, gps_state)
                fuse_mm_precision.append(precision)

                # mm -> fuse -> mm
                t_state = bigeye.multi_mm(mm_signal_set, save_name='sample%d_test%d_%d_%d' % (sample_n, trail, direction, testi),
                                          save_dir='test_multi_mm/mm_fuse/')
                precision, _ = bigeye.evaluate_fn_dist(t_state, gps_state)
                mm_fuse_precision.append(precision)

                print(trails, directions, ks, fuse_mm_precision, mm_fuse_precision, single_precisions)
                out = pd.DataFrame({'trail': trails, 'direction': directions, 'num': ks,
                                    'fuse_mm_precision': fuse_mm_precision,
                                    'average_single_precision': single_precisions,
                                    'mm_fuse_precision': mm_fuse_precision})
                out.to_csv(WORK_DIR + 'test_res/precision_sample%d_test%d.csv' % (sample_n, test_size))


if __name__ == '__main__':
    # test_big_eye(47)
    # test_wuhu_index(1, 1)
    # evaluate_wuhu_multimm_sample(5)
    evaluate_wuhu_multimm(sample_n=5)


    pass