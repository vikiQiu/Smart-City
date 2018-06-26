__author__ = 'Victoria'

from src.pre_processing.read_data import get_signals, get_random_users
from src.pre_processing.funs import clean_signal, output_data_js
import datetime
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


class SignalModel:
    def __init__(self, mon, dd, hh_start, hh_end, yy=2017):
        '''
        Initialization
        :param mon: The month [int]
        :param dd: The day [int]
        :param hh_start: The hour start [int]
        :param hh_end: The hour end [int]
        '''
        self.mon = mon
        self.dd = dd
        self.hh_start = hh_start
        self.hh_end = hh_end
        self.yy = yy
        self.file_date = '2017%.2d%.2d' % (mon, dd)
        self.out_signal_dir = '../../data/signals_pro/%.2d%.2d/%.2dt%.2d' % (mon, dd, hh_start, hh_end)
        self.plot_dir = '../../pic/%.2d%.2d/%.2dt%.2d' % (mon, dd, hh_start, hh_end)
        self.js_dir = '../../web/data/%.2d%.2d/%.2dt%.2d' % (mon, dd, hh_start, hh_end)
        self.filename_before = ('%s/signals_origin_%.2dt%.2d.csv' % (self.out_signal_dir, hh_start, hh_end))
        self.filename_after = ('%s/signals_%.2dt%.2d.csv' % (self.out_signal_dir, hh_start, hh_end))
        self.filename_after_users = ('%s/signals_%.2dt%.2d_within_users.csv' % (self.out_signal_dir, hh_start, hh_end))
        self.filename_before_users = ('%s/signals_origin_%.2dt%.2d_within_users.csv' % (self.out_signal_dir, hh_start, hh_end))

    def save_data(self, part_start=0, part_end=1000, users=None, isReturn=False):
        '''
        save the signal data during [hh_start, hh_end] time in date 2017/mon/dd.
        data directory: data/signals_pro/$mon$dd/$hh_start t $hh_end
        :param part_start: The start part to search signals
        :param part_end: The end part to search signals
        :param users: If specify users, only remain signals of users. Default: None
        :param isReturn: Return signals or not. Default: False
        :return:
        '''
        parts = [str("%.5i" % x) for x in range(part_start, part_end+1)]
        signals = get_signals(self.file_date, parts[0])

        # 5.229469537734985s per part
        date_min = datetime.datetime(self.yy, self.mon, self.dd, self.hh_start)
        if self.hh_end == 24:
            date_max = datetime.datetime(self.yy, self.mon, self.dd + 1, 0)
        else:
            date_max = datetime.datetime(self.yy, self.mon, self.dd, self.hh_end)
        signals = signals[(signals.dates > date_min) & (signals.dates < date_max)]
        if users is not None:
            signals = signals[signals.user_id.isin(users)]

        for part in parts[1:]:
            if not os.path.exists('../../data/hf_signals/hf_' + self.file_date + '/part-' + str(part)):
                break
            signal = get_signals(self.file_date, part)
            signal = signal[(signal.dates > date_min) & (signal.dates < date_max)]
            if users is not None:
                signal = signal[signal.user_id.isin(users)]
            print("part=%s, num samples=%d, this part num=%d" % (part, len(signals), len(signal)))
            signals = pd.concat([signals, signal])

        self.check_dir(self.out_signal_dir)

        signals = signals.sort_values(['user_id', 'dates'])
        if users is None:
            signals.to_csv(self.filename_before, index=False)
        else:
            signals.to_csv(self.filename_before_users + 'within_users', index=False)
        print('nrow of signals before clean = %d' % len(signals))
        signals = clean_signal(signals)
        print('nrow of signals after clean = %d' % len(signals))
        if users is None:
            signals.to_csv(self.filename_after, index=False)
        else:
            signals.to_csv(self.filename_after_users, index=False)
        if isReturn:
            return signals

    def get_clean_signals(self, within_users=False):
        if not os.path.exists(self.filename_after_users if within_users else self.filename_after):
            print('%s does not exist. Please Save_data first.' % self.filename_after)
        else:
            if within_users:
                signals = pd.read_csv(self.filename_after_users)
            else:
                signals = pd.read_csv(self.filename_after)
            signals.cell_id = signals.cell_id.astype('str')
            signals.dates = [np.datetime64(x) for x in signals.dates]
            return signals

    def check_dir(self, dir_name):
        '''
        Check if the directory exists. If not exists, it will be create.
        :param dir_name: directory name [String]
        :return: No return
        '''
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def plot_users_trail(self, users=None, within_users=True, save_pic=False, save_trails_js=False):
        '''
        save users' trail plot or save users' trail js file
        :param users: users id array. If users is None, it will plot all the users. Default = None
        :param within_users: read signals within users or not. Default = True
        :param save_pic: Boolean Whether to save the trail plot
        :param save_trails_js: Boolean Whether to save the trail js file
        :return:
        '''
        cells = pd.read_csv('../../res/[0926]cells_process.csv')
        signals = self.get_clean_signals(within_users)
        if users is None:
            users = signals.user_id.unique()
        n = len(users)
        for i in range(n):
            if not (save_pic | save_trails_js):
                break
            user_id = users[i]
            signal = signals[signals.user_id == user_id]
            signal = signal.sort_values('dates')
            signal = pd.merge(signal, cells, on='cell_id', how='left')
            signal = signal.dropna(0)
            print('user %d' % i)
            if save_pic:
                self.check_dir(self.plot_dir)
                plt.plot(signal.lon, signal.lat)
                plt.title('user = %s' % user_id)
                plt.savefig('%s/pic%d.jpg' % (self.plot_dir, i))
                plt.close('all')
            if save_trails_js:
                self.check_dir(self.js_dir)
                output_data_js(signal, user_id, '%s/user%d.js' % (self.js_dir, i))


# signal_mol = SignalModel(6, 6, 19, 20)
# signal_mol.save_data(438, 439)

# Get a whole day specific users singals
signal_mol = SignalModel(6, 7, 0, 24)
rand_users = np.array(get_random_users().sort_values('user_id').user_id)
# signal_mol.save_data(users=rand_users)
print('start')
signal_mol.plot_users_trail(rand_users[0:200], save_pic=False, save_trails_js=True)



