__author__ = 'Victoria'

# 2017-09-22

import datetime
import pandas as pd
import numpy as np


def get_signals(file_dates, part, file_dir='../../data/hf_signals/'):
    filename = file_dir + 'hf_' + str(file_dates) + '/part-' + str(part)
    f = open(filename)
    dates, cell_id, user_id, service_type, web = [], [], [], [], []

    for line in f.readlines():
        line_tmp = line.strip('()\n').split(',')
        dates.append(get_date_type(line_tmp[0]))
        cell_id.append(line_tmp[1])
        user_id.append(line_tmp[2])
        service_type.append(line_tmp[3])
        web.append(line_tmp[4])
    f.close()
    return pd.DataFrame({'dates': dates, 'cell_id': cell_id, 'user_id': user_id,
                         'service_type': service_type, 'web': web})


def get_datetime(t):
    if '/' in t:
        pattern = '%Y/%m/%d %H:%M:%S'
    elif '-' in t:
        pattern = '%Y-%m-%d %H:%M:%S'
    else:
        pattern = '%Y%m%d%H%M%S'

    return datetime.datetime.strptime(t, pattern)


def get_users_signals(file_dates: str, part: str, users, file_dir):
    '''
    Get the given users signals
    :param file_dates:
    :param part: The part of date
    :param users: pd.DataFrame['user_id']
    :param file_dir: File directory stored signals.
    :return: A DataFrame with columns ['dates', 'cell_id', 'user_id', 'service_type', 'web']
    '''
    df = get_signals(file_dates, part, file_dir=file_dir)
    if type(users) == list:
        users = pd.DataFrame({'user_id': users})
    df = pd.merge(df, users, on='user_id')
    return df


def get_date_type(date0):
    day, time = date0.split(' ')
    day = day.split('/')
    time = time.split(':')
    return datetime.datetime(int(day[0]), int(day[1]), int(day[2]),
                             int(time[0]), int(time[1]), int(float(time[2])), int(float(time[2]) % 1))


def get_cellSheet(dir='../../data/cellIdSheets/', cell_type='baidu', place='hf'):
    filename = dir + 'cellIdSheet_'+cell_type+'_' + place +'.txt'
    f = open(filename)
    cell_id, longitude, latitude, radius = [], [], [], []
    for line in f.readlines():
        line_tmp = line.split('\t')
        cell_id.append(line_tmp[0])
        longitude.append(float(line_tmp[1]))
        latitude.append(float(line_tmp[2]))
        radius.append(int(line_tmp[3]))
    f.close()
    return pd.DataFrame({'cell_id': cell_id, 'lon': longitude, 'lat': latitude, 'radius': radius})


def get_hefei_cellSheet(cell_type='baidu'):
    dir = '../../data/cellIdSheets/'

    return get_cellSheet(dir, cell_type, 'hf')


def get_link(filename='../../data/hefei_road/link_baidu.txt'):
    f = open(filename)
    links, lon, lat = [], [], []
    for line in f.readlines():
        line_tmp = line.split('\t')
        for pos in line_tmp[1].split('|'):
            links.append(line_tmp[0])
            lon.append(float(pos.split(',')[0]))
            lat.append(float(pos.split(',')[1]))
    f.close()
    return pd.DataFrame({'alinks': links, 'blon': lon, 'clat': lat})


def get_hefei_link():
    filename = '../../data/hefei_road/link_baidu.txt'
    f = open(filename)
    links, lon, lat = [], [], []
    for line in f.readlines():
        line_tmp = line.split('\t')
        for pos in line_tmp[1].split('|'):
            links.append(line_tmp[0])
            lon.append(float(pos.split(',')[0]))
            lat.append(float(pos.split(',')[1]))
    f.close()
    return pd.DataFrame({'alinks': links, 'blon': lon, 'clat': lat})


def get_signals_from_csv(filename, names=None):
    df = pd.read_csv(filename, names=names)
    df['cell_id'] = df['cell_id'].astype(str)
    # df['dates'] = [np.datetime64(x) for x in df.dates]
    df.dates = df.dates.astype(str)
    df['dates'] = [get_datetime(x) for x in df.dates]
    return df


def get_cells_process():
    return pd.read_csv('../../res/[0926]cells_process.csv')


def get_random_users():
    return pd.read_csv('../../res/rand_users.csv')


def get_given_minute_time_signals(dates, hh_start, min_start, hh_end, min_end, part_start, part_end):
    '''
    Get signals data during hh_start:min_start - hh_end:min_end in the given dates
    :param dates: [yyyy, mm, dd]. Type: [<int>]
    :param hh_start: Start hour <int>
    :param hh_end: End hour <int
    :param min_start: Start minute <int>
    :param min_end: End minute <int>
    :param part_start: Start part
    :param part_end: End part
    :return: A DataFrame
    '''

    yy, mm, dd = dates
    filedates = '%.4d%.2d%.2d' % (yy, mm, dd)
    time_start = datetime.datetime(yy, mm, dd, hh_start, min_start)
    time_end = datetime.datetime(yy, mm, dd, hh_end, min_end)

    for part in range(part_start, part_end+1):
        df = get_signals(filedates, part)
        df = df[time_start <= df.dates < time_end]
        if 'res' in locals().keys():
            res = pd.concat([res, df])
        else:
            res = df
        print('Finish part %d in date %s' % (part, filedates))



# example to get data
# links = get_link()
# cells = get_cellSheet()
# signals = get_signals('20170601', '00000')
