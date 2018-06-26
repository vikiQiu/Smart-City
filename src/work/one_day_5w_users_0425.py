'''
Author: Vikitoria
Date: 2018-04-25
Purpose: Get signal data in the given date and given set of users.
         For the Data Structure Algorithm Course.
    1. Get all users
    2. Randomly choose 5w users in the users
    3. Get all signal data of the chosen users
'''

import os
import time
import pandas as pd
from src.pre_processing.read_data import get_signals
from src.work.oneuser_alldata_0307 import get_users_signals


def get_all_users(file_date='20170607',
                  dir_name='../../data/hf_signals/',
                  save_dir='../../res/work/0425_one_day_5w_users/',
                  part_start=0, part_end=1000):
    '''
    Get all the users appears in the file date
    :param dir_name: Directory name to store signal data.
    :param file_date: yyyymmdd. e.x: 20170607
    :param save_dir: The directory to store the users id.
    :var file_name = dir_name + 'hf_' + file_date + '/part-' + %.5f.
                     See detail in function get_signals().
    :var save_file_name = save_dir + 'all_users_' + file_date

    Example:
        get_all_users(file_date='20170607')
    :return: None
    '''
    parts = [str("%.5i" % x) for x in range(part_start, part_end + 1)]

    # Calculate execute time
    part_time = time.time()

    users = []
    for i, part in enumerate(parts):
        file_name = dir_name + 'hf_' + file_date + '/part-' + part
        if not os.path.exists(file_name):
            break

        signal = get_signals(file_date, part, dir_name)
        users.extend(signal.user_id.tolist())
        users = list(set(users))
        if i % 10 == 0:
            print('Finish part %s of date %s. Use time %.2fs' % (part, file_date, time.time() - part_time))
            part_time = time.time()

    res = pd.DataFrame({'user_id': users})
    res.to_csv('%sall_users_%s.csv' % (save_dir, file_date), index=False)

def random_users(file_date, user_num,
                 dir_name='../../res/work/0425_one_day_5w_users/'):
    '''
    Random choose users from the all_users_yyyymmdd.csv.
    :param file_date: 'yyyymmdd'.
        Users file name: 'all_users_yyyymmdd.csv'
    :param user_num: The number of users need
    :param dir_name: The directory stored the users csv.
    :return: The list of user_id
    '''
    file_name = dir_name + 'all_users_' + file_date + '.csv'
    users = pd.read_csv(file_name)
    users = users.sample(user_num)
    users.to_csv('%srandom_%dusers_%s.csv' % (dir_name, user_num, file_date), index=False)


def get_users_all_data(users_file,
                       file_date='20170607',
                       dir_name='../../data/hf_signals/',
                       save_dir='../../res/work/0425_one_day_5w_users/',
                       part_start=0, part_end=1000):
    '''
    Get all users' signal data in the file_date
    :param users_file: file name of the users.
    :param part_start:
    :param part_end:
    :return: A signal's DataFrame of given users.
    '''
    users = pd.read_csv(save_dir+users_file).user_id.tolist()

    parts = [str("%.5i" % x) for x in range(part_start, part_end + 1)]
    signals = ''

    # Calculate execute time
    start_time = part_time = time.time()

    for i, part in enumerate(parts):
        file_name = dir_name + 'hf_' + file_date + '/part-' + part
        if not os.path.exists(file_name):
            continue

        signal = get_users_signals(file_date, part, users, file_dir=dir_name)
        if i == 0:
            signals = signal
        else:
            signals = pd.concat([signals, signal])

        if i % 10 == 0:
            print('Finish part %s of date %s. Use time %.2fs' % (part, file_date, time.time() - part_time))
            part_time = time.time()

    signals.to_csv('%srandom_users_signals%s.csv' % (save_dir, file_date), index=False)

    # Total time
    print('All Done! Total time = %.2fs' % (time.time() - start_time))

    return signals


if __name__ == '__main__':
    file_date = '20170607'
    # get_all_users(file_date=file_date)
    # random_users(file_date, 50000)
    get_users_all_data(users_file='random_50000users_20170607.csv', file_date=file_date)