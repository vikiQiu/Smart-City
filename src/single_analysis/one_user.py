# 2017-10-09

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import random
import numpy as np
from src.pre_processing.read_data import get_link
from src.pre_processing.funs import cal_distance, cal_rough_speed_per_cell, cal_rough_speed, output_data_js

''' Set Hyper Parameters '''
mon = 6
dd = 6
hh_start = 14
hh_end = 15
file_date = '20170606'

''' Get Data '''
# get signals data (Already been preprocessed by
filename = ('../../data/signals_pro/%.2d%.2d/%.2dt%.2d/signals_%.2dt%.2d.csv'
            % (mon, dd, hh_start, hh_end, hh_start, hh_end))
signals = pd.read_csv(filename)
signals.cell_id = signals.cell_id.astype('str')
signals.dates = [np.datetime64(x) for x in signals.dates]

# get cells data
cells = pd.read_csv('../../res/[0926]cells_process.csv')
cells_loc = cells[['cell_id', 'lat', 'lon']]

'''Get Top 50 users'''
# get a user_id
g_signals = signals.groupby(['user_id', 'cell_id']).count()
g_signals = g_signals.groupby('user_id').count()
N = len(g_signals)
print('Total %d users' % N)
# g_signals.sort_values('dates', ascending=False).head(20)
n = 50
users = g_signals.sort_values('dates', ascending=False).index[0:n]


def plot_users_trail(users, signals, cells, save_pic=False, save_trails_js=False):
    '''
    save users' trail plot or save users' trail js file
    :param users: users id array
    :param signals: signals
    :param cells: get_cells
    :param save_pic: Boolean Whether to save the trail plot
    :param save_trails_js: Boolean Whether to save the trail js file
    :return:
    '''
    n = len(users)
    for i in range(n):
        if save_pic & save_trails_js:
            break
        user_id = users[i]
        signal = signals[signals.user_id == user_id]
        signal = signal.sort_values('dates')
        signal = pd.merge(signal, cells, on='cell_id', how='left')
        signal = signal.dropna(0)
        print('user %d' % i)
        if save_pic:
            plt.plot(signal.lon, signal.lat)
            plt.savefig('../../pic/%.2d%.2d/%.2dt%.2d/pic%d.jpg' % (mon, dd, hh_start, hh_end, i))
            plt.close('all')
        if save_trails_js:
            output_data_js(signal, '../../web/data/%.2d%.2d/%.2dt%.2d/user%d.js' % (mon, dd, hh_start, hh_end, i))


# plot_users_trail(users, signals, cells, save_pic=True)
# plot_users_trail(users, signals, cells, save_trails_js=True)

# get speed
user_ind = 20
# spd_per = cal_rough_speed_per_cell(signals[signals.user_id == users[user_ind]], cells)
# print(spd_per)

print('start')
# print(cal_rough_speed(signals[signals.user_id == users[user_ind]], cells))

# speed = cal_rough_speed(signals[signals.user_id == users[0]], cells)
# for i in range(1, n):
#     tmp = cal_rough_speed(signals[signals.user_id == users[i]], cells)
#     speed = pd.concat([speed, tmp], ignore_index=True)
# print(speed)

# speed.to_csv('../../res/speed/speed_%.2d%.2d_%.2dt_%.2d.csv' % (mon, dd, hh_start, hh_end), index=False)

# randomly sample 100 users
n_sample = 1000
rand_users = pd.DataFrame({'user_id': signals.user_id.unique()[random.sample(range(N), n_sample)]})
rand_users.to_csv('../../res/rand_users.csv', index=False)

