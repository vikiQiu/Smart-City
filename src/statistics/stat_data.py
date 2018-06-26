import os
import json
import shutil
import random
import datetime
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.pre_processing.read_data import get_signals_from_csv
from src.pre_processing.funs import output_data_js

# from src.pre_processing.get_clean_signals import plot_us

file_path = '/Volumes/Elements/hf_signals/'
work_dir = '../../res/work/0610_stat_pic/'
data_file = work_dir+'data.json'


def get_stat(file_dates):
    files = os.listdir(file_path+'hf_'+file_dates+'/')
    if os.path.isfile(data_file):
        with open(data_file) as f:
            data = json.load(f)
            cells = data['cells']
            users = data['users']
            times = data['times']
            print('Read From history')
    else:
        cells = {}
        users = {}
        times = {}

    for file in files:
        filename = file_path+'hf_'+file_dates+'/'+file
        print(filename)
        if 'part' not in filename:
            continue
        # if '0000' not in filename:
        #     continue

        f = open(filename)
        dates, cell_id, user_id, service_type = [], [], [], []

        for line in f.readlines():
            line_tmp = line.strip('()\n').split(',')
            dates.append(get_date_type(line_tmp[0]))
            cell_id.append(line_tmp[1])
            user_id.append(line_tmp[2])
            service_type.append(line_tmp[3])
        f.close()
        hours = [str(x.hour) for x in dates]

        # cells_key = [e1 + e2 for (e1, e2) in zip(cell_id, service_type)]
        # users_key = [e1 + e2 for (e1, e2) in zip(user_id, service_type)]
        # times_key = [e1 + e2 for (e1, e2) in zip(hours, service_type)]
        # cells = cells + Counter(cells_key)
        # users = users + Counter(users_key)
        # times = times + Counter(times_key)

        for i in range(len(dates)):
            key = cell_id[i]+'/'+service_type[i]
            if key in cells.keys():
                cells[key] += 1
            else:
                cells[key] = 1

            key = hours[i]+'/'+service_type[i]
            if key in times.keys():
                times[key] += 1
            else:
                times[key] = 1

            key = user_id[i]+'/'+service_type[i]
            if key in users.keys():
                users[key] += 1
            else:
                users[key] = 1

        print(len(cells))
        print(len(users))
        print(len(times))
    out = {'cells': cells, 'users': users, 'times': times}
    with open(data_file, 'w') as f:
        out = json.dumps(out)
        f.write(out)
    return


def get_user_data(file_dates):
    users = pd.read_csv(work_dir+'attach/cells_pic/random_user.csv')
    users = users.user_id
    users = [x.split('@')[0] for x in users]
    users = users[:50]
    print(users)

    files = os.listdir(file_path + 'hf_' + file_dates + '/')

    for file in files:
        filename = file_path + 'hf_' + file_dates + '/' + file
        print(filename)
        if 'part' not in filename:
            continue
        # if '0000' not in filename:
        #     continue

        f = open(filename)
        dates, cell_id, user_id, service_type, web = [], [], [], [], []

        for line in f.readlines():
            line_tmp = line.strip('()\n').split(',')
            if line_tmp[1] not in users:
                continue
            dates.append(get_date_type(line_tmp[0]))
            cell_id.append(line_tmp[1])
            user_id.append(line_tmp[2])
            service_type.append(line_tmp[3])
            web.append(line_tmp[4])
        f.close()
        tmp = pd.DataFrame({'dates': dates, 'cell_id': cell_id, 'user_id': user_id,
                            'service_type': service_type, 'web': web})
        if 'out' in vars():
            out = pd.concat([out, tmp])
        else:
            out = tmp

        print(len(out))

    out.to_csv(work_dir+'attach/cells_data'+file_dates+'.csv', index=False)


def get_morning_attach(file_dates):
    '''Get morning abnormal attach'''
    files = os.listdir(file_path+'hf_'+file_dates+'/')
    users = {}
    cells = {}

    for file in files:
        filename = file_path+'hf_'+file_dates+'/'+file
        print(filename)
        if 'part' not in filename:
            continue
        # if '0000' not in filename:
        #     continue

        f = open(filename)
        dates, cell_id, user_id, service_type = [], [], [], []

        for line in f.readlines():
            line_tmp = line.strip('()\n').split(',')
            dates.append(get_date_type(line_tmp[0]))
            cell_id.append(line_tmp[1])
            user_id.append(line_tmp[2])
            service_type.append(line_tmp[3])
        f.close()
        hours = [str(x.hour) for x in dates]

        # cells_key = [e1 + e2 for (e1, e2) in zip(cell_id, service_type)]
        # users_key = [e1 + e2 for (e1, e2) in zip(user_id, service_type)]
        # times_key = [e1 + e2 for (e1, e2) in zip(hours, service_type)]
        # cells = cells + Counter(cells_key)
        # users = users + Counter(users_key)
        # times = times + Counter(times_key)

        for i in range(len(dates)):
            if service_type[i] != 'LTE_ATTACH':
                continue
            key = user_id[i]+'@'+service_type[i]
            if key in users.keys():
                users[key][int(hours[i])] += 1
            else:
                users[key] = [0 for _ in range(24)]

            key = cell_id[i] + '@' + service_type[i]
            if key in cells.keys():
                cells[key][int(hours[i])] += 1
            else:
                cells[key] = [0 for _ in range(24)]

        print(len(users), len(cells))

    with open(work_dir+'attach/user_hour_attach'+file_dates+'.json', 'w') as f:
        out = json.dumps(users)
        f.write(out)
    with open(work_dir+'attach/cells_hour_attach'+file_dates+'.json', 'w') as f:
        out = json.dumps(cells)
        f.write(out)
    return


def plot_user_hour_attach(file_date, name='user'):
    with open(work_dir+'attach/'+name+'_hour_attach'+file_date+'.json') as f:
        data = json.load(f)
    tmp_dir = 'attach/%s_pic/' % name
    n = len(data)
    if name == 'user':
        data = {k: v for k, v in data.items() if sum(v[:6])>5}
    else:
        data = {k: v for k, v in data.items() if sum(v[:6]) > 50}
    data = {k: v for k, v in data.items() if np.mean(v[:6]) > 1.2*np.mean(v[6:])}
    print('p=', len(data)/n)
    x = [_ for _ in range(24)]
    keys = random.sample(list(data.keys()), 100)

    with open(work_dir+'attach/'+name+'_hour_attach'+str(int(file_date)+1)+'.json') as f:
        data2 = json.load(f)
    with open(work_dir+'attach/'+name+'_hour_attach'+str(int(file_date)+2)+'.json') as f:
        data3 = json.load(f)

    tmp_dir = 'attach/%s_pic/' % name
    random.seed(100)
    for i, key in enumerate(keys):
        plt.clf()
        plt.plot(x, data[key])
        labels = [file_date]
        if key in data2.keys():
            plt.plot(x, data2[key])
            labels.append(str(int(file_date)+1))
        if key in data3.keys():
            plt.plot(x, data3[key])
            labels.append(str(int(file_date)+2))
        plt.xlabel('Hour')
        plt.ylabel('attach num')
        plt.title(key)
        plt.legend(labels=labels, loc='best')
        # plt.show()
        plt.savefig(work_dir+tmp_dir+'user'+str(i)+'.jpg')

    keys = pd.DataFrame({'user_id': keys})
    keys.to_csv(work_dir+'attach/%s_pic/random_user.csv'%name, index=False)


def plot_user_hour_type(file_date, name='user'):
    col = 'cell_id'
    df = get_signals_from_csv(work_dir+'attach/'+name+'_data'+file_date+'.csv')
    df['hour'] = [x.hour for x in df.dates]
    labels = list(pd.read_csv(work_dir+'attach/'+name+'_pic/random_user.csv').user_id)
    labels = [x.split('@')[0] for x in labels]

    types = ['LTE_ATTACH', 'LTE_DETACH', 'CSFB', 'Service_Req', 'TAU', 'Path_Switch', 'LTE_PAGING']

    for i, label in enumerate(labels):
        tmp = df[df[col] == label]
        if len(tmp) == 0:
            continue

        res = np.zeros((len(types), 24))
        x = [_ for _ in range(24)]

        plt.clf()
        for t_ind, t in enumerate(types):
            for h in range(24):
                res[t_ind, h] = sum((tmp['service_type']==t) & (tmp['hour']==h))
            plt.plot(x, res[t_ind])

        plt.xlabel('Hour')
        plt.ylabel('Frequency')
        plt.legend(labels=types, loc='best')
        plt.title(name+': '+label)
        # plt.show()
        plt.savefig(work_dir + 'attach/'+name+'_type_pic/' +name + str(i) + '.jpg')


def get_date_type(date0):
    day, time = date0.split(' ')
    day = day.split('/')
    time = time.split(':')
    return datetime.datetime(int(day[0]), int(day[1]), int(day[2]),
                             int(time[0]), int(time[1]), int(float(time[2])), int(float(time[2]) % 1))


def get_hist_data(d, filename='users'):
    '''

    :param d: A dictionary {'label/service_type': num}
    :param filename: label name. Such as 'cells', 'users' and so on.
    :return:
    '''
    df = pd.DataFrame({'label': [''.join(x.split('/')[:len(x.split('/'))-1]) for x in d],
                       'type': [x.split('/')[len(x.split('/'))-1] for x in d]})
    print('read done')
    df['value'] = 1
    # df['value'] = [d[x] for x in d]
    g = pd.pivot_table(df, index='label', values='value', columns='type', aggfunc=sum)
    g.to_csv(work_dir+filename+'.csv')
    g = g.sum()/len(g)
    print(g, list(g), list(g.index))
    return list(g), list(g.index)


def get_time_data():
    with open(work_dir+'data20170607.json') as f:
        data = json.load(f)
        data = data['times']
    df = pd.DataFrame({'label': [''.join(x.split('/')[:len(x.split('/')) - 1]) for x in data],
                       'type': [x.split('/')[len(x.split('/')) - 1] for x in data],
                       'value': [data[x] for x in data]})
    g = pd.pivot_table(df, index='label', values='value', columns='type', aggfunc=sum)
    g.to_csv(work_dir+'times.csv')


def user_type_plot(name='users'):
    # with open(data_file) as f:
    #     data = json.load(f)
    #     values = data[name]
    # y_vals, types = get_hist_data(values)

    t1 = ['LTE_ATTACH', 'LTE_DETACH', 'CSFB', 'Service_Req', 'TAU', 'Path_Switch', 'LTE_PAGING']
    t2 = ['Attach', 'Detach', 'CSFB', 'sRequests', 'TAU', 'pSwitch', 'Paging']
    g = pd.read_csv(work_dir+name+'.csv')
    # g = g.fillna(0)
    # g = g[g.TAU!=0]
    g = g[t1]
    types = g.columns
    types = [t2[t1.index(x)] for x in types]
    y_vals = list(g.sum() / len(g))
    print(g.sum()/len(g))

    # y_vals = [0.7 for _ in range(7)]
    # types = [t2[t1.index(x)] for x in types]

    plot_hist(y_vals, types, work_dir+'pic/'+name+'.pdf', 'Percent of '+name+' (%)')


def plot_hist(y_vals, types, save_name, ylabel):
    with PdfPages(save_name) as pdf:
        font_size='10'
        fig = plt.figure()
        ax=fig.add_subplot(111)

        print(types, [100*v for v in y_vals])
        ax.bar(types, [100*v for v in y_vals], align='center',color='#666666')

        rects = ax.patches
        for rect in rects:
            height = rect.get_height()
            value = height
            ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '%0.2f' % value,
                    ha='center', va='bottom',fontsize=10)

        plt.subplots_adjust(top=0.95, right=0.9, left=0.21, bottom=0.21)
        plt.xticks(sorted(types), size=10, fontweight='bold',rotation=90)
        plt.yticks(size=font_size,fontweight='bold')
        plt.ylabel(ylabel, fontsize=font_size,fontweight='bold')
        plt.ylim(0,110)
        # plt.show()
        plt.show()
        pdf.savefig(fig)


def get_test_users(file_dates='20170601'):
    filedir = '../../data/hf_signals/'
    u = pd.read_csv('../../data/xunfei_test/driver.txt', sep='\t')
    users = u['用户ID'].tolist()
    u = pd.read_csv('../../data/xunfei_test/passenger.txt', sep='\t')
    users.extend(list(u['用户ID']))
    users = list(set(users))
    print(users[:10], len(users))
    get_users(users, filedir + 'hf_' + file_dates + '/', 'test_user_signals%s.csv' % file_dates)


def get_users(users, data_dir, out_file, out_dir=work_dir):
    files = os.listdir(data_dir)
    for i, file in enumerate(files):
        filename = data_dir + file
        print(filename)

        f = open(filename)
        dates, cell_id, user_id, service_type = [], [], [], []

        for line in f.readlines():
            line_tmp = line.strip('()\n').split(',')
            dates.append(get_date_type(line_tmp[0]))
            cell_id.append(line_tmp[1])
            user_id.append(line_tmp[2])
            service_type.append(line_tmp[3])
        f.close()

        tmp = pd.DataFrame({'dates': dates, 'cell_id': cell_id, 'user_id': user_id,
                            'service_type': service_type})
        tmp = tmp[tmp.user_id.isin(users)]
        if 'df' in vars():
            df = pd.concat([df, tmp])
        else:
            df = tmp
        print(len(df))
    df.to_csv(out_dir+out_file, index=False)


def plot_users_trail(file_date, save_pic=True, save_trails_js=True, save_data=True, save_dir=work_dir+'attach/'):
    '''
    save users' trail plot or save users' trail js file
    :param users: users id array
    :param signals: signals
    :param cells: get_cells
    :param save_pic: Boolean Whether to save the trail plot
    :param save_trails_js: Boolean Whether to save the trail js file
    :return:
    '''

    cells = pd.read_csv('../../res/[0926]cells_process.csv')
    cells_loc = cells[['cell_id', 'lat', 'lon']]

    dirs = [save_dir+'user_pic/', save_dir+'user_trail/']

    signals = get_signals_from_csv(save_dir + 'user_data' + file_date + '.csv')

    labels = list(pd.read_csv(save_dir + 'user_pic/random_user.csv').user_id)
    users = [x.split('@')[0] for x in labels]

    n = len(users)
    for i in range(n):
        # if save_pic & save_trails_js:
        #     break
        user_id = users[i]
        signal = signals[signals.user_id == user_id]
        signal = signal.sort_values('dates')
        signal = pd.merge(signal, cells, on='cell_id', how='left')
        signal = signal.dropna(0)
        print('user %d' % i)
        if save_pic:
            plt.plot(signal.lon, signal.lat)
            plt.savefig('%suser_trail_%s/pic/user%d.jpg' % (save_dir, file_date, i))
            plt.close('all')
        if save_data:
            signal.to_csv('%suser_trail_%s/csv/user%d.csv' % (save_dir, file_date, i), index=False)
        if save_trails_js:
            output_data_js(signal, '%suser_trail_%s/data/user%d.js' % (save_dir, file_date, i))


def cells_distribution():
    '''
    We find near 30% cells never accept signal types besides paging.
    We want to get plot these cells on the map and find some rules.
    :return: Output cells JS
    '''
    cells_pos = pd.read_csv('../../data/cellIdSheets/cells_baidu_hf.csv')
    cells = pd.read_csv(work_dir+'cells.csv')
    cells['cell_id'] = cells.label
    cells = cells.fillna(0)
    cells.label = cells.apply(lambda x: 1 if sum(x[1:4])+sum(x[5:8]) == 0 else 0, axis=1)
    cells = cells[['cell_id', 'label']]
    cells.cell_id = cells.cell_id.apply(str)
    cells = pd.merge(cells, cells_pos)
    raw_points = [[cells.iloc[x].lon, cells.iloc[x].lat] for x in range(len(cells)) if cells.iloc[x].label==0]
    points = [[cells.iloc[x].lon, cells.iloc[x].lat] for x in range(len(cells)) if cells.iloc[x].label==1]
    out = {'raw_points': raw_points, 'points': points}

    with open(work_dir+'cells/cells_distributions.json', 'w') as f:
        out = json.dumps(out)
        f.write(out)
    return

    print(raw_points)
    print(cells.head())


def get_no_TAU_signals(file_dates='20170601'):
    '''
    对于没有tau信令数据的用户，提取出来他们的单天轨迹
    只对于20170601
    :return:
    '''
    tau_dir = 'no_tau/'
    user_file = work_dir+tau_dir+'random_users.csv'
    filedir = '../../data/hf_signals/'
    n_user = 1000
    if os.path.isfile(user_file):
        print('Read users from file.')
        users = pd.read_csv(user_file)
        users = list(users.label)
    else:
        u = pd.read_csv(work_dir+'users0601.csv')
        u = u.fillna(0)
        u = u[u.TAU == 0]
        u = u.reset_index(drop=True)
        ind = [_ for _ in range(len(u))]
        ind = [random.choice(ind) for _ in range(n_user)]
        u = u.iloc[ind]
        u.to_csv(user_file, index=False)
        users = list(u.label)

    print(users)
    get_users(users, filedir + 'hf_' + file_dates + '/', 'no_tau_user_signals_%s.csv' % file_dates, work_dir+tau_dir)


# def plot

if __name__ == '__main__':
    # for i in range(20170609, 20170621):
    #     get_stat(str(i))
    #     shutil.copy(data_file, '../../res/work/0610_stat_pic/data'+str(i)+'.json')

    # with open(work_dir+'data.json') as f:
    #     data = json.load(f)
    #     users = data['cells']
    # print(get_hist_data(users, 'cells'))

    # user_type_plot('cells')

    # get_test_users()

    get_no_TAU_signals('20170607')

    # get_time_data()

    # get_morning_attach('20170613')
    # plot_user_hour_attach(file_date='20170611', name='cells')
    # get_user_data('20170610')
    # plot_user_hour_type('20170610', 'cells')
    plot_users_trail('20170610')

    # cells_distribution()

