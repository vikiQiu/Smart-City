'''
Author: Vikitoria
Date: 2018-04-28
Puropose:
    1. Get random users raw trajectories during 20170601-20170620.
    2. Store the data in '../../res/work/0428_raw_trajectories_random_users/'
    3. Plot the trajectories in the same directory.
TIPS:
    1. The raw data of '201706dd.csv' stored in '../../res/work/0307_oneuser_alldata/'
    2. Similar with 'oneuser_alldata_0307.py'
'''

import pandas as pd
import datetime
import os
import re

from src.pre_processing.read_data import get_signals_from_csv
from src.work.oneuser_alldata_0307 import trajectories_tojson, save_trajectories_pic


def get_user_raw_trajectories(files, user, cells, user_num,
                              directory='../../res/work/0307_oneuser_alldata/',
                              save_dir='../../res/work/0428_raw_trajectories_random_users/',
                              segment_hour=3):
    '''
    Segment the raw trajectory with day
    :param files: A file list: ['yyyymmdd.csv']
    :param cells: pd.DataFrame['cell_id', 'lon', 'lat']
    :param user_num: The number of user
    :param user: A user_id -> string
    :param directory: The directory store the files
    :param save_dir: The directory to store the signal_set and pictures
    :param segment_hour: Default 3. Means segment the trajectory by day:03:00 - day+1:03:00
    :return: A list of pd.DataFrame['lon', 'lat', 'dates', 'user_id', 'cell_id']
    '''
    # Concat the user's all days signals together
    signals = ''
    for i, file in enumerate(files):
        signal = get_signals_from_csv(directory + file)
        signal = signal[signal['user_id'] == user]
        if i == 0:
            signals = signal
        else:
            signals = pd.concat([signals, signal])

    # Merge cells info
    signals = pd.merge(signals, cells, on='cell_id', how='left')

    # Get day column
    signals['dates2'] = signals.dates - datetime.timedelta(hours=segment_hour)
    signals['day'] = signals.dates.apply(lambda x: x.day)
    days = sorted(signals.day.unique())

    # Segment the trajectory with day
    signal_set = []
    datetime.timedelta()
    for day in days:
        signal = signals[signals.day == day]
        signal = signal.sort_values('dates')
        signal = signal[['lon', 'lat', 'dates', 'user_id', 'cell_id']]
        signal = signal.reset_index()
        signal_set.append(signal)

    # Save the signal_set as json and js
    trajectories_tojson(signal_set, user, save_dir+'json/random_user%d.json' % user_num, is_json=True)
    trajectories_tojson(signal_set, user, save_dir + 'js/random_user%d.js' % user_num, is_json=False)

    # Plot
    save_trajectories_pic(signal_set, save_dir+'pic/', user, user_num)

    return signal_set


def save_random_users_raw_trajectories(segment_hour=3,
                                       csv_dir='../../res/work/0307_oneuser_alldata/'):
    '''
    Save random users daily raw trajectories to json, js, pic.
    Only segment the trajectory during a day
    :param segment_hour: Default 3. Means segment the trajectory by day:03:00 - day+1:03:00
    :param csv_dir:
    :return:
    '''

    # Get singals files' name.
    files = os.listdir(csv_dir)
    files = list(filter(re.compile(r'\d.csv').search, files))

    # Get users
    users = pd.read_csv(csv_dir + 'rand_users.csv').user_id

    # Get cells data
    cells = pd.read_csv('../../res/[0926]cells_process.csv')
    cells_loc = cells[['cell_id', 'lat', 'lon']]

    for i, user in enumerate(users):
        get_user_raw_trajectories(files, user, cells_loc, i, segment_hour=segment_hour,
                                  directory=csv_dir)
        print('[User %d] %s Finished.' % (i, user))


if __name__ == '__main__':
    save_random_users_raw_trajectories()
