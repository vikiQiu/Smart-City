__author__ = "Victoria"
# 2017-11-12

import datetime
import pandas as pd
from os import listdir
import numpy as np
from src.pre_processing.read_data import get_link, get_cellSheet


def get_road_tests(trail=0, direction=0, n=None):
    '''
    get road test data of a trail with direction.
    :param trail: <Int> The i_th trail. If 0, don't consider trail and n.
    :param direction: <-1/1> -1 means reverse side; 1 means front; 0 means don't consider direction.
    :param n: The n_th test of this trail. If None, return all the tests of the trail.
    :return: A pandas DataFrame
    '''

    mypath = '../../data/road_test/wuhu/'
    headers = ['time', 'nothing', 'cell_id', 'user_id', 'service_type', 'web', 'lon', 'lat']

    n = '%d.csv' % n if n is not None else ''
    trail = u'线路' + str(trail)
    direction = u'正向' if direction==1 else u'逆向' if direction==-1 else ''
    direction = direction + n
    files_ind = [trail in file and direction in file for file in listdir(mypath)]
    files = np.array(listdir(mypath))[files_ind]
    print(files)
    # df = pd.read_csv(mypath + files[0], names = headers)
    for file in files:
        tmp = pd.read_csv(mypath + file, names=headers)
        tmp = remove_null(tmp, ['lon', 'lat'])
        if 'data' in locals():
            data = pd.concat([data, tmp])
        else:
            data = tmp
    data = data.drop(['nothing'], axis=1)
    data = data.dropna(axis=0, how='any')
    data = data.reset_index(drop=True)
    data.time = data.time.apply(get_time)
    data.lon = data.lon.apply(float)
    data.lat = data.lat.apply(float)
    return data


def remove_null(df: pd.DataFrame, columns):
    '''
    Remove samples if there is 'null' in given columns
    :param df: A DataFrame
    :param columns: A string list. Columns to check whether 'null' in them.
    :return:
    '''
    for col in columns:
        if df.dtypes[col] == 'O':
            df = df.iloc[[not x for x in df[col].isin(['null'])]]
    return df.reset_index(drop=True)


def get_wuhu_link():
    filename = '../../data/road_test/link_baidu_wuhu.txt'
    return get_link(filename)


def get_time(t):
    '''
    Transform yyyymmddhhmmss [Int] to a datetime type
    :param t: Time (yyyymmddhhmmss)
    :return: datetime
    e.x: df = [time, lon, lat]
    1. df.time = df.time.apply(get_time)
    2. df.time = [get_time(x) for x in df.time]
    '''
    yy = int(t // 1e10)
    mm = int(t % 1e10 // 1e8)
    dd = int(t % 1e8 // 1e6)
    hh = int(t % 1e6 // 1e4)
    min = int(t % 1e4 // 1e2)
    s = int(t % 1e2)
    return datetime.datetime(yy, mm, dd, hh, min, s)


def get_wuhu_cellSheet(cell_type='baidu'):
    dir = '../../data/cellIdSheets/'

    return get_cellSheet(dir, cell_type, 'wuhu')


if __name__ == '__main__':
    # df = get_road_tests(1, 1)
    # print(df)
    # print(df.lon[1])
    print(len(get_wuhu_link()))

