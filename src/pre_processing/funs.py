__author__ = 'Victoria'

# '2017-09-25'

from math import radians, cos, sin, asin, sqrt, degrees, atan2
import numpy as np
import pandas as pd
import datetime
from src.pre_processing.read_data import get_signals, get_cellSheet
from datetime import timedelta


def cal_distance(pos1, pos2):
    '''
    Calculate m from pos1 to pos2
    :param pos1: [lon, lat]
    :param pos2: [lon, lat]
    :return:
    '''
    lon1, lat1, lon2, lat2 = map(radians, [pos1[0], pos1[1], pos2[0], pos2[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # earth radius, km
    return c * r * 1000


# def radians(x):
#     '''
#     Chage angle to radians
#     :param x: Double 0-360
#     :return:
#     '''
#     if not (0 <= x <= 360):
#         print('[Warning] Angle must between 0 to 360, while %f given.' % x)
#     return x * np.pi / 180


def cal_direction(pos1, pos2):
    '''
    Calculate angle from pos1 to pos2
    :param pos1:
    :param pos2:
    :return:
    '''
    radLatA = radians(pos1[1])
    radLonA = radians(pos1[0])
    radLatB = radians(pos2[1])
    radLonB = radians(pos2[0])
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng


def KNN(list, query):
    min_dist = 9999.0
    NN = np.array([])
    for i in range(len(list)):
        pt = np.array(list[i:(i+1)])[0]
        dist = cal_distance(query, pt[1:3])
        if dist < min_dist:
            NN = pt
            min_dist = dist
    return NN, min_dist


def clean_signal(signals):
    if len(signals) == 0:
        return signals
    signals = signals.sort_values(['user_id', 'dates'])
    user_id = np.array(signals.user_id)
    cell_id = np.array(signals.cell_id)
    user_tmp = user_id[0]
    cell_tmp = cell_id[0]
    ind = [0]
    for i in range(1, len(signals)):
        if user_id[i] == user_tmp:
            if cell_id[i] != cell_tmp:
                cell_tmp = cell_id[i]
                ind.append(i)
            elif i != len(signals)-1:
                if cell_id[i] != cell_id[i+1] and user_id[i+1] == user_tmp:
                    ind.append(i)
        else:
            user_tmp = user_id[i]
            cell_tmp = cell_id[i]
            ind.append(i)
    return signals.iloc[ind]


def clean_one_signal(signal):
    ind = []
    cell_id = [x for x in signal.cell_id]
    cell_tmp = cell_id[0]
    for i in range(1, len(signal)):
        if cell_id[i] != cell_tmp:
            cell_tmp = cell_id[i]
            ind.append(i)
    return signal.iloc[ind]


def cal_rough_speed_per_cell(signal, cells):
    signal = pd.merge(signal, cells, on='cell_id', how='left')
    signal = signal.sort_values('dates')
    signal = signal.dropna(0)
    signal = clean_one_signal(signal)
    signal = signal.reset_index()
    cell_ids = [x for x in signal.cell_id]
    dist = []
    time = []
    n = len(signal)
    for i in range(n - 1):
        dist.append(cal_distance([signal.lon[i], signal.lat[i]], [signal.lon[i+1], signal.lat[i+1]]))
        time.append((signal.dates[i+1] - signal.dates[i]).total_seconds())
    speed = [dist[i] / time[i] if time[i] !=0 else 999999 for i in range(len(dist))]
    return pd.DataFrame({'start_cell': cell_ids[:(n-1)], 'end_cell': cell_ids[1:],
                         'time': time, 'dists': dist, 'speed': speed})

def cal_rough_speed_per_cell2(obs):
    '''
    Get m/s
    :param obs: DataFrame['dates', 'lon', 'lat']
    :return:
    '''
    dist = []
    time = []
    direction = []
    obs = obs.reset_index(drop=True)
    n = len(obs)
    for i in range(n - 1):
        dist.append(cal_distance([obs.lon[i], obs.lat[i]], [obs.lon[i + 1], obs.lat[i + 1]]))
        time.append((obs.dates[i + 1] - obs.dates[i]).total_seconds())
        direction.append(cal_direction([obs.lon[i], obs.lat[i]], [obs.lon[i + 1], obs.lat[i + 1]]))
    speed = [dist[i] / time[i] if time[i] != 0 else 999999 for i in range(len(dist))]
    dir_180 = [dir % 180 for dir in direction]
    return pd.DataFrame({'time': time, 'dists': dist, 'speed': speed, 'direction': direction, 'direction_180': dir_180})


def cal_rough_speed(signal, cells):
    spd = cal_rough_speed_per_cell(signal, cells)
    dist = spd.dists.sum()
    time = spd.time.sum()
    return pd.DataFrame({'distance': [dist], 'time': [time], 'speed': [dist / time]})


def output_geo_js(lon, lat, times, filename):
    N = len(lon)
    print(N)
    print(len(lat))
    print(len(times))
    content = '''data = {'points': ['''
    for i in range(N):
        content = '''%s[%.5f, %.5f, '%s']''' % (content, lon[i], lat[i], str(times[i]))
        if i != N - 1:
            content = '''%s, ''' % content
        else:
            content = '''%s]}''' % content
    f = open(filename, 'w')
    f.write(content)
    f.close()


def output_data_js(df, filename, user_id=-1, var_name='data'):
    '''
    output the longitude and latitude data as js file
    :param df: A pandas DataFrame, must contain 'lon' and 'lat' columns.
    :param filename: the filename to output js file
    :param var_name: Variable name in js file
    :return: No return, just write a js file.
    Js File example:
    var lon = 139
    var lat = 39
    var data = {'points': [[139, 39], [139.1, 39.1]]}
    '''
    lon = df.lon.median()
    lat = df.lat.median()
    max_lon = df.lon.max() * 1.5 - df.lon.min() * 0.5
    min_lon = df.lon.min() * 1.5 - df.lon.max() * 0.5
    max_lat = df.lat.max() * 1.5 - df.lat.min() * 0.5
    min_lat = df.lat.min() * 1.5 - df.lat.max() * 0.5
    N = len(df)
    content = '''var lon = %.5f\nvar lat = %.5f\nvar max_lon = %.5f\nvar min_lon = %.5f''' % (lon, lat, max_lon, min_lon) \
              + '''\nvar max_lat = %.5f\nvar min_lat = %.5f\nvar user_id = '%s'\n''' % (max_lat, min_lat, user_id)\
              + '''%s = {'points': [''' % var_name
    for i in range(N):
        if 'dates' in df.columns:
            content = '''%s[%.5f, %.5f, '%s']''' % (content, df.iloc[i].lon, df.iloc[i].lat, str(df.iloc[i].dates))
        else:
            content = '''%s[%.5f, %.5f]''' % (content, df.iloc[i].lon, df.iloc[i].lat)
        if i != N - 1:
            content = '''%s, ''' % content
        else:
            content = '''%s]}''' % content
    f = open(filename, 'w')
    f.write(content)
    f.close()


def output_road_js():
    filename = '../../data/hefei_road/link_baidu.txt'
    fileout = '../../web/data/road_links.js'
    f = open(filename)
    content = '''var links = [{'''
    for ind_line, line in enumerate(f.readlines()):
        line_tmp = line.split('\t')
        if ind_line != 0:
            content = content + ',{'
        content = '''%s 'link_id': %s, 'pos': [''' % (content, line_tmp[0])
        for ind_pos, pos in enumerate(line_tmp[1].split('|')):
            if ind_pos != 0:
                content = content + ','
            content = '%s [%.5f, %.5f]' % (content, float(pos.split(',')[0]), float(pos.split(',')[1]))
        content = content + ']}'
    f.close()
    fw = open(fileout, 'w')
    fw.write(content)
    fw.close()


def get_cell_distribution(dates, hh_start, min_start, hh_end, min_end, part_start, part_end):
    '''
    Get cell distribution during hh_start:min_start - hh_end:min_end in the given dates.
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
        df = get_signals(filedates, '%.5d' % part)
        df = df[(time_start <= df.dates) & (df.dates < time_end)]
        df.dates = [t.replace(second=0) for t in df.dates]

        # group by 'dates' and 'cell_id'
        tmp = df.groupby(['dates', 'cell_id'], as_index=False).count()
        tmp['count'] = tmp.user_id
        df = tmp[['dates', 'cell_id', 'count']]

        if 'res' in locals().keys():
            res = pd.concat([res, df])
        else:
            res = df
        print('Finish part %d in date %s. len=%d' % (part, filedates, len(res)))

    # Aggregate
    res = res.groupby(['dates', 'cell_id'], as_index=False).sum()
    res = res.sort_values('dates')
    filename = '../../res/cell_distribution_with_time/%s/%.2d%.2dto%.2d%.2d.csv' % \
               (filedates, hh_start, min_start, hh_end, min_end)
    res.to_csv(filename, index=False)
    return res


def output_cell_distribution(dates, hh_start, min_start, hh_end, min_end, part_start, part_end, isRead=False):
    '''
    Get cell distribution during hh_start:min_start - hh_end:min_end in the given dates.
    And output the res DataFrame to a js file.
    :param dates: [yyyy, mm, dd]. Type: [<int>]
    :param hh_start: Start hour <int>
    :param hh_end: End hour <int
    :param min_start: Start minute <int>
    :param min_end: End minute <int>
    :param part_start: Start part
    :param part_end: End part
    :param isRead: Read the res from csv or use method get_cell_distribution()
    :var res: A DataFrame [dates, cell_id, count]
    :return: A DataFrame ['cell_id', 'lon', 'lat', 'dates']
    '''
    yy, mm, dd = dates
    filedates = '%.4d%.2d%.2d' % (yy, mm, dd)
    if isRead:
        filename = '../../res/cell_distribution_with_time/%s/%.2d%.2dto%.2d%.2d.csv' % \
                   (filedates, hh_start, min_start, hh_end, min_end)
        res = pd.read_csv(filename)
    else:
        res = get_cell_distribution(dates, hh_start, min_start, hh_end, min_end, part_start, part_end)
    cells = get_cellSheet()
    print('Finish read data. Next merging.')

    res = res.groupby('cell_id', as_index=False).count()
    res['cell_id'] = res['cell_id'].astype(str)
    res = pd.merge(res, cells, 'left', on='cell_id')
    res = res.dropna(axis=0, how='any')
    output_file = '../../res/cell_distribution_with_time/%s/%.2d%.2dto%.2d%.2d_data.js' % \
                   (filedates, hh_start, min_start, hh_end, min_end)

    print('Output data js.')
    output_data_js(res, output_file, var_name='cells')
    return res


def get_datetime(t):
    if '/' in t:
        return datetime.datetime.strptime(t, '%Y/%m/%d %H:%M:%S')
    else:
        return datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')


def datetime_tostr(d: datetime.datetime, time_format='%Y-%m-%d %H:%M:%S'):
    '''
    Transfer datetime to string.
    :param d:
    :param time_format: datetime format return.
    :return: A String.
    '''
    return d.strftime(time_format)


def interpolate(obs, interval=1, origin_reserve=False):
    '''
    To interpolate observables within one second.
    :param obs: A DataFrame [dates<datetime>, lon, lat]
    :param interval: Interval seconds. <Int>
    :param origin_reserve: whether to reserve the origin points
    :return:
    '''
    res = obs
    for i in range(len(obs)-1):
        if (obs.dates.iloc[i+1] - obs.dates.iloc[i]).total_seconds != 1:
            res = pd.concat([res, interpolate_single(obs.iloc[i], obs.iloc[i+1])])

    res = res.sort_values('dates')
    res = res.iloc[np.array(range(len(res))) % interval == 0]
    if origin_reserve:
        res = pd.concat([res, obs])
        res = res.sort_values('dates')
        res = res.drop_duplicates()
    res = res.reset_index(drop=True)
    return res


def interpolate_single(obs1, obs2):
    '''
    Interpolate data between time1 and time2 per second.
    :param obs1: The start row in obs DataFrame [dates<datetime>, lon, lat]
    :param obs2: The end row in obs DataFrame [datse<datetime>, lon, lat]
    :return: return the DataFrame need to interpolate.
    e.x: obs1 = [2016-03-30 10:56:18, 0, 0]
         obs2 = [2016-03-30 10:56:21, 3, 3]
         return = [[2016-03-30 10:56:19, 1, 1],
                   [2016-03-30 10:56:20, 2, 2]]
    '''
    # Number of row need to be interpolated
    n_interpolate = int((obs2.dates - obs1.dates).total_seconds()) - 1

    time = [obs1.dates + timedelta(seconds=i+1) for i in range(n_interpolate)]
    lon = np.linspace(obs1.lon, obs2.lon, n_interpolate+2)[1:n_interpolate+1]
    lat = np.linspace(obs1.lat, obs2.lat, n_interpolate+2)[1:n_interpolate+1]
    return pd.DataFrame({'dates': time, 'lon': lon, 'lat': lat})


def get_0608_cell_distribution():
    # get_cell_distribution([2017, 6, 8], 0, 0, 8, 0, 1, 205)  # Get 8:00-12:00 cell distribution
    # get_cell_distribution([2017, 6, 8], 8, 0, 16, 0, 200, 405)  # Get 8:00-12:00 cell distribution
    # get_cell_distribution([2017, 6, 8], 16, 0, 23, 59, 395, 597)  # Get 8:00-12:00 cell distribution
    f1 = pd.read_csv('../../res/cell_distribution_with_time/20170608/0800to1600.csv')
    f2 = pd.read_csv('../../res/cell_distribution_with_time/20170608/1600to2359.csv')
    f3 = pd.read_csv('../../res/cell_distribution_with_time/20170608/0000to0800.csv')
    df = pd.concat([f1, f2, f3])
    df = df[['dates', 'cell_id', 'count']]

    cells = df.cell_id.unique()
    n_cell = len(cells)
    cells = pd.DataFrame({'cells': cells})
    cells.to_csv('../../res/cell_distribution_with_time/20170608/used_cells.csv')

    print('Unique cell_id in 20170608 = %d' % n_cell)
    df = df.groupby(['dates', 'cell_id'], as_index=False).sum()
    df = df.groupby(['dates'], as_index=False).agg(['mean', 'sum', 'count'])
    df = df.reset_index()
    df.dates = df.dates.apply(get_datetime)
    print('get hour')
    df['hour'] = [t.hour for t in df.dates]
    print('get minute')
    df['minute'] = [t.minute for t in df.dates]
    print(df.head())
    df.to_csv('../../res/cell_distribution_with_time/20170608/whole.csv')


def get_0609_cell_distribution():
    # get_cell_distribution([2017, 6, 9], 0, 0, 8, 0, 1, 205)  # Get 8:00-12:00 cell distribution
    # get_cell_distribution([2017, 6, 9], 8, 0, 16, 0, 200, 410)  # Get 8:00-12:00 cell distribution
    # get_cell_distribution([2017, 6, 9], 16, 0, 23, 59, 390, 595)  # Get 8:00-12:00 cell distribution
    f1 = pd.read_csv('../../res/cell_distribution_with_time/20170609/0800to1600.csv')
    f2 = pd.read_csv('../../res/cell_distribution_with_time/20170609/1600to2359.csv')
    f3 = pd.read_csv('../../res/cell_distribution_with_time/20170609/0000to0800.csv')
    df = pd.concat([f1, f2, f3])
    df = df[['dates', 'cell_id', 'count']]

    cells = df.cell_id.unique()
    n_cell = len(cells)
    cells = pd.DataFrame({'cells': cells})
    cells.to_csv('../../res/cell_distribution_with_time/20170609/used_cells.csv')

    # 5 min
    df2 = df[['dates', 'cell_id', 'count']]
    df2['hour'] = [t.hour for t in df2.dates]
    df2['minute'] = [t.minute // 5 for t in df2.dates]


    print('Unique cell_id in 20170609 = %d' % n_cell)
    df = df.groupby(['dates', 'cell_id'], as_index=False).sum()
    df = df.groupby(['dates'], as_index=False).agg(['mean', 'sum', 'count'])
    df = df.reset_index()
    df.dates = df.dates.apply(get_datetime)
    print('get hour')
    df['hour'] = [t.hour for t in df.dates]
    print('get minute')
    df['minute'] = [t.minute for t in df.dates]
    print(df.head())
    df.to_csv('../../res/cell_distribution_with_time/20170609/whole.csv')


def get_0607_cell_distribution():
    f1 = pd.read_csv('../../res/cell_distribution_with_time/20170607/0800to1200.csv')
    f2 = pd.read_csv('../../res/cell_distribution_with_time/20170607/1200to2359.csv')
    f3 = pd.read_csv('../../res/cell_distribution_with_time/20170607/0000to0800.csv')
    df = pd.concat([f1, f2, f3])
    df = df[['dates', 'cell_id', 'count']]

    cells = df.cell_id.unique()
    n_cell = len(cells)
    cells = pd.DataFrame({'cells': cells})
    cells.to_csv('../../res/cell_distribution_with_time/20170607/used_cells.csv')

    print('Unique cell_id in 20170607 = %d' % n_cell)
    df = df.groupby(['dates', 'cell_id'], as_index=False).sum()
    df = df.groupby(['dates'], as_index=False).agg(['mean', 'sum', 'count'])
    df = df.reset_index()
    df.dates = df.dates.apply(get_datetime)
    print('get hour')
    df['hour'] = [t.hour for t in df.dates]
    print('get minute')
    df['minute'] = [t.minute for t in df.dates]
    print(df.head())
    df.to_csv('../../res/cell_distribution_with_time/20170607/whole0607.csv')


if __name__ == '__main__':
    # x1 = [116.76602890593577, 31.788085925227357]
    # x2 = [116.76635890071597, 31.788105991196062]
    # x3 = [116.76660893600715, 31.788155977099493]
    # x4 = [116.76680894108648, 31.788135913065975]
    # x5 = [116.767169003356, 31.78814570835917]
    # x6 = [116.7675391258152, 31.788195374725404]
    # y1 = [116.740218, 31.882573]
    # y2 = [116.743275, 31.829832]
    # print(cal_distance(x1, x2))
    # print(cal_distance(x2, x3))
    # print(cal_distance(x3, x4))
    # print(cal_distance(x4, x5))
    # print(cal_distance(x5, x6))
    # print(cal_distance(y1, y1))

    # print (KNN(links, y1)) # 69.8S ['92335', 116.73993746905009, 31.883469113171184], dist=103.10361961604376

    # output_road_js()

    get_cell_distribution([2017, 6, 1], 0, 0, 8, 0, 1, 205) # Get 8:00-12:00 cell distribution
    # output_cell_distribution([2017, 6, 7], 12, 0, 12, 1, 295, 305)

    # f1 = pd.read_csv('../../res/cell_distribution_with_time/20170607/0000to0800.csv')
    # f2 = pd.read_csv('../../res/cell_distribution_with_time/20170607/0000to0959.csv')
    # df = pd.concat([f1, f2])
    # df.to_csv('../../res/cell_distribution_with_time/20170607/0000to0800.csv')
    # f1 = pd.read_csv('../../res/cell_distribution_with_time/20170607/0800to1200.csv')
    # f2 = pd.read_csv('../../res/cell_distribution_with_time/20170607/1200to2359.csv')
    # f3 = pd.read_csv('../../res/cell_distribution_with_time/20170607/0000to0800.csv')
    # df = pd.concat([f1, f2, f3])
    # df = df[['dates', 'cell_id', 'count']]
    # df = df.groupby(['dates', 'cell_id'], as_index=False).sum()
    # print(df.head())
    # # df = df.groupby(['dates'], as_index=False).agg(['mean', 'sum', 'count'])
    #
    # df.dates = df.dates.apply(get_datetime)
    # print('get hour')
    # df['hour'] = [t.hour for t in df.dates]
    # print('get minute')
    # df['minute'] = [t.minute for t in df.dates]
    # df.minute = df.minute // 30
    # df = df[['hour', 'minute', 'cell_id', 'count']]
    # df = df.groupby(['hour', 'minute', 'cell_id']).sum()
    # df = df.groupby(['hour', 'minute']).agg(['mean', 'sum', 'count'])
    # df.to_csv('../../res/cell_distribution_with_time/20170607/whole_part.csv')

    # wuhu_cells = get_cellSheet('../../data/road_test/cellId/', place='wuhu')
    # print(wuhu_cells.head())
    # output_data_js(wuhu_cells, '../../res/MapMatching_test/wuhu_cells.js', var_name='cells')

    get_0608_cell_distribution()


