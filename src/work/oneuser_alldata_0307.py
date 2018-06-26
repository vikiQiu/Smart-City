__author__ = 'Victoria'
'''
Start time: 2018-03-07
End time: Not done
1. Get a sample of users. (maybe 1000 users)
2. Extract all data in all days
'''

import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from src.pre_processing.read_data import get_users_signals, get_signals_from_csv
from src.single_analysis.user_trajectory import get_trajectory
from src.pre_processing.funs import datetime_tostr
from src.pre_processing.funs import cal_distance


class Signals:
    def __init__(self,
                 file_date: str,
                 mon=None, dd=None,
                 hh_start=None, hh_end=None,
                 file_dir='../../data/hf_signals/'):
        self.mon = mon
        self.dd = dd
        self.hh_start = hh_start
        self.hh_end = hh_end
        self.file_date = file_date
        self.file_dir = file_dir

    def get_users_all_data(self, users, part_start=0, part_end=1000):
        '''
        Get all users' signal data in the file_date
        :param users: A user list.
        :param part_start:
        :param part_end:
        :return: A signal's DataFrame of given users.
        '''
        parts = [str("%.5i" % x) for x in range(part_start, part_end + 1)]
        signals = ''

        # Calculate execute time
        start_time = part_time = time.time()

        for i, part in enumerate(parts):
            if not os.path.exists(self.file_dir + 'hf_' + self.file_date + '/part-' + str(part)):
                break

            signal = get_users_signals(self.file_date, part, users, file_dir=self.file_dir)
            if i == 0:
                signals = signal
            else:
                signals = pd.concat([signals, signal])

            if i % 10 == 0:
                print('Finish part %s of date %s. Use time %.2fs' % (part, self.file_date, time.time() - part_time))
                part_time = time.time()

        if part_end == 1000 and part_start == 0:
            signals.to_csv('../../res/work/0307_oneuser_alldata/%s.csv' % self.file_date, index=False)
        else:
            signals.to_csv('../../res/work/0307_oneuser_alldata/%s_%dto%d.csv' %
                           (self.file_date, part_start, min(part_end, i+1)), index=False)

        # Total time
        print('All Done! Total time = %.2fs' % (time.time() - start_time))

        return signals


def test_users_data(file_date,
                    file_dir='../../data/hf_signals/',
                    write_dir='../../res/work/0307_oneuser_alldata/',
                    part_start=0,
                    part_end=1000):
    '''
    Test get all data of given users in one day
    Save the signals to the directory such as: '../../res/work/0307_oneuser_alldata/'
    :param file_date:
    :param file_dir:
    :param write_dir:
    :param part_start:
    :param part_end:
    :return:
    Example:
      test_users_data('20170607')
    '''
    # Initialize class Signal:
    s_model = Signals(file_date, file_dir=file_dir)

    # Get users id
    users = pd.read_csv(write_dir + 'rand_users.csv')
    users = users['user_id'].values.tolist()

    # Get users' signal data
    df = s_model.get_users_all_data(users, part_start, part_end)
    df.to_csv(write_dir+file_date+'.csv', index=False)
    return df


def test_all_users_data(file_dir='../../data/hf_signals/',
                        write_dir='../../res/work/0307_oneuser_alldata/'):
    '''
    Do all tests.
    :param file_dir:
    :param write_dir: Directory to write data.
    :return:
    Example:
      test_all_users_data('/Volumes/Elements/hf_signals/')
    '''
    file_dates = os.listdir(file_dir)

    for f in file_dates:
        file_date = f[3:]
        if file_date + '.csv' not in os.listdir(write_dir):
            test_users_data(file_date, file_dir, write_dir)


def get_all_users_trajectories(dir = '../../res/work/0307_oneuser_alldata/',
                               is_json=None,
                               is_plot=False):
    '''
    Get [user_trajectories].
    user_trajectories = [user's dealed trajectory].
    dealed trajectory: A DataFrame ['dates', 'lon', 'lat']
    :param dir: Directory stored signals data.
    :param is_json: Whether to save trajectories as json.
                    If None, don't save;
                    If True, save to json;
                    If False, save to js.
    :param is_plot: Whether to plot the trajectories. Boolean.
    :return: A dict: {'user_id': [dealed trajectory]
    '''
    # Get singals files' name.
    files = os.listdir(dir)
    files = list(filter(re.compile(r'\d.csv').search, files))

    # Get users
    users = pd.read_csv(dir + 'rand_users.csv')

    # Get cells data
    cells = pd.read_csv('../../res/[0926]cells_process.csv')
    cells_loc = cells[['cell_id', 'lat', 'lon']]

    # Return result
    res = {}

    for i, user in enumerate(users['user_id']):
        if i < 210:
            continue

        print('Begin to deal with user %d: %s' % (i, user))
        signal_set = get_user_trajectories(files, user, dir, cells_loc)
        res[user] = signal_set
        print('Total %d trajectories. Begin to plot trajectory.' % len(signal_set))

        if is_json is not None:
            if is_json:
                trajectories_tojson(signal_set, user,
                                    '../../res/work/0416_oneuser_alldata_alltime/json/random_user%d.json' % i, is_json)
            else:
                trajectories_tojson(signal_set, user,
                                    '../../res/work/0416_oneuser_alldata_alltime/js/random_user%d.js' % i, is_json)

        if is_plot:
            plot_trajectories(signal_set)

    return res


def get_user_trajectories(files, user_id,
                          dir='../../res/work/0307_oneuser_alldata/',
                          cells_loc=None):
    '''
    Get a given user's one day trajectory
    :param files: A file list: ['yyyymmdd.csv']
    :param user_id: string.
    :param dir: The directory stored the file
    :param cells_loc: cells location DataFrame. If None, it will load the DataFrame automatically.
    :return: A list of trajectory DataFrame.
    '''
    if cells_loc is None:
        # Get cells data
        cells = pd.read_csv('../../res/[0926]cells_process.csv')
        cells_loc = cells[['cell_id', 'lat', 'lon']]

    # Get all the signals
    signals = ''
    for i, file in enumerate(files):
        signal = get_signals_from_csv(dir + file)
        signal = signal[signal['user_id'] == user_id]
        if i == 0:
            signals = signal
        else:
            signals = pd.concat([signals, signal])

    # Transfer to trajectories
    if len(signals) == 0:
        return []
    else:
        _, _, signal_set = get_trajectory(signals, cells_loc, user_id)

    return signal_set


def trajectories_tojson(signal_set, user_id, save_path, is_json=None):
    '''
    Save the json file.
    :param signal_set: A list of trajectory DataFrame.
    :param user_id: String.
    :param save_path: The path to save json.
    :param is_json: Whether to save as json. If True, json; if False, js.
    :return:
    {
      user_id: String,
      trajectories: [{lon: float, lat: float, start_time, end_time}]
    }
    '''
    res = {'user_id': user_id}
    trajectories = []
    mean_lon = []
    mean_lat = []
    for i in range(len(signal_set)):
        signal = signal_set[i]
        # ind = [limit_time(t, 7, 10) or limit_time(t, 17, 20) for t in signal.dates]
        # signal = signal[ind]
        signal = signal.reset_index(drop=True)
        if len(signal) < 5:
            continue

        max_time = max(signal.dates)
        min_time = min(signal.dates)
        start_time = datetime_tostr(min_time, '%H:%M')
        end_time = datetime_tostr(max_time, '%H:%M')
        day = datetime_tostr(min_time, '%m/%d')
        delta_time = (max_time - min_time).total_seconds() / 60 # min
        N = len(signal)
        dist = cal_distance([signal.lon[0], signal.lat[0]], [signal.lon[N-1], signal.lat[N-1]]) # m
        if delta_time == 0:
            continue
        rough_speed = dist / delta_time / 60  # m/s

        trajectory = {'lon': signal.lon.tolist(),
                      'lat': signal.lat.tolist(),
                      'start': '[%d-S]%s(%.1fmin)%s' % (i, day, delta_time, start_time),
                      'end': '[%d-E]%s(%.1fm)%s' % (i, day, dist, end_time),
                      'start_time': start_time,
                      'end_time': end_time,
                      'day': day,
                      'delta_time': '%.1f' % delta_time,
                      'speed': '%.2f' % rough_speed,
                      'dist': '%.1f' % dist,
                      'dates': signal.dates.apply(datetime_tostr).tolist()}

        trajectories.append(trajectory)
        mean_lon.append(np.mean(signal.lon))
        mean_lat.append(np.mean(signal.lat))

    res['trajectories'] = trajectories
    res['lon'] = np.mean(mean_lon)
    res['lat'] = np.mean(mean_lat)

    if is_json:
        save_json(res, save_path)
    else:
        save_js(res, save_path)


def save_json(dict, save_path):
    output = json.dumps(dict)

    f = open(save_path, 'w')
    f.write(output)
    f.close()


def save_js(dict, save_path):
    output = json.dumps(dict)

    f = open(save_path, 'w')
    f.write('var data = ' + output)
    f.close()


def plot_trajectories(signal_set):
    '''
    Plot all trajectories in signal set according to time.
    :param signal_set: A list of trajactory.
    :return:
    '''
    plt.close()  # clf() # 清图  cla() # 清坐标轴 close() # 关窗口
    fig = plt.figure()
    # plt.show()
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(True)  # 添加网格
    plt.ion()  # interactive mode on
    for i in range(len(signal_set)):
        signal = signal_set[i]
        max_time = max(signal.dates)
        min_time = min(signal.dates)
        max_time_str = datetime_tostr(max_time, '%H:%M')
        min_time_str = datetime_tostr(min_time, '%H:%M')
        day = datetime_tostr(min_time, '%m/%d')
        delta_time = (max_time - min_time).total_seconds() / 60
        for j in range(len(signal)-1):
            lon1, lon2 = signal['lon'][j:(j+2)]
            lat1, lat2 = signal_set[i]['lat'][j:(j+2)]
            dist = cal_distance([lon1, lat1], [lon2, lat2])
            plt.title('[Trace %d %s(%.2f min)]%s to %s. T=%d, D=%.2fm.'
                      % (i, day, delta_time, min_time_str, max_time_str, j, dist))
            ax.plot(signal['lon'][j:(j+2)], signal_set[i]['lat'][j:(j+2)], 'C' + str(i % 10))
            if j == 0:
                ax.scatter([lon1, lon2], [lat1, lat2], s=10, color='red')
            else:
                # TODO: make the forward points to be black
                ax.scatter(signal.lon[:j], signal.lat[:j], color='black', s=10)
                ax.scatter([lon2], [lat2], s=10, color='red')
            plt.pause(0.5)
        plt.pause(1)


def save_trajectories_pic(signal_set, save_path, user, user_num):
    '''
    Plot all trajectories in signal set according to time.
    :param signal_set: A list of trajactory.
    :return:
    '''
    plt.close()  # clf() # 清图  cla() # 清坐标轴 close() # 关窗口
    # plt.show()
    for signal in signal_set:
        plt.plot(signal.lon, signal.lat)
    plt.savefig(save_path+'random_user%d.jpg' % user_num)


def limit_time(t, start_hh, end_hh):
    if start_hh <= t.hour <= end_hh:
        return True
    else:
        return False

        # TODO:
        # if t.hour == start_hh:
        #     if t.minute >= start_mm:
        #         return True
        #     else:
        #         return False
        # elif t.hour == end_hh:
        #     if t. == end_hh:


if __name__ == '__main__':
    # --- One file test ---
    # test_users_data('20170607', part_start=580)
    test_users_data('20170601')

    # --- Deal with all files ---
    # test_all_users_data('/Volumes/Elements/hf_signals/')

    # --- Get all trajectories ---
    # t = get_all_users_trajectories(is_json=True)
    # print(t)



