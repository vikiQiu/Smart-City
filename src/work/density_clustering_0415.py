import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from src.pre_processing.funs import cal_distance, get_datetime
from src.map_matching.multipath_mm import DTW, obs_to_ts
from src.work.oneuser_alldata_0307 import trajectories_tojson


def density_clustering(points, eps=600):
    '''
    Density Clustering of DBSCAN in sklearn
    :param points: [[lon, lat]]
    :return: labels of all points.
             If label == -1: Means the point is a single point.
             If label >= 0: Means the point is in the group label.
    '''
    n = len(points)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = cal_distance(points[i], points[j])
            dist_mat[i, j] = dist_mat[j, i] = dist
    labels = DBSCAN(eps=eps, metric='precomputed', min_samples=2).fit_predict(dist_mat)

    return labels


def get_obses_points(user_number, dir='../../res/work/0416_oneuser_alldata_alltime/json/'):
    '''
    Get A user's 20 days' all trajectories in obses and points.
    :param user_number: A int number.
    :param dir: The directory stores the trajectories.
                The file name must be 'random_user%d.json'
                The directory name must end with '/'
    :var obses: [DataFrame['lon', 'lat', 'dates']]
    :var points: [[lon, lat]]
    :return: (obses, points)
    '''
    with open('%s/random_user%d.json' % (dir, user_number)) as data_file:
        data = json.load(data_file)

    obses = []
    points = []
    for i in range(len(data['trajectories'])):
        tmp = data['trajectories'][i]
        if len(tmp['lon']) <= 7:
            continue
        dates = [get_datetime(t) for t in tmp['dates']]
        obs = pd.DataFrame({'lon': tmp['lon'], 'lat': tmp['lat'], 'dates': dates})
        obses.append(obs)
        points.append([obs.lon[0], obs.lat[0]])
        points.append([obs.lon[len(obs)-1], obs.lat[len(obs)-1]])

    return obses, points


def find_similar_trajectories(user_number,
                              max_dist=300,
                              interval=5,
                              dir='../../res/work/0416_oneuser_alldata_alltime/json/'):
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

    :param user_number: A int number.
    :param max_dist: Max distance to determine whether to trajectories are the similar trajectory.
    :param interval: Interval time to interpolate.
    :param dir: The directory stores the trajectories.
                The file name must be 'random_user%d.json'
                The directory name must end with '/'
    :return: res -> [[DataFrame]]
    '''
    obses, points = get_obses_points(user_number, dir)

    if len(obses) == 0:
        return []

    labels = density_clustering(points)
    res = []

    # Get single trajectories
    for i in range(len(obses)):
        if labels[2*i] == -1 or labels[2*i+1] == -1:
            res.append(obses[i])

    # TODO: Find similar trajecotries
    for a in range(max(labels)+1):
        for b in range(max(labels)+1):
            if a == b:
                continue
            the_obses = [obses[i] for i in range(len(obses)) if labels[2*i] == a and labels[2*i+1] == b]
            if len(the_obses) == 0:
                continue
            if len(the_obses) == 1:
                res.extend(the_obses)
                continue
            res.extend(similar_trajectories_clustering(the_obses, interval, max_dist))

    return res


def similar_trajectories_clustering(obses, interval, max_dist):
    '''
    For trajectories whose origin and destination points are from the same cluster as the other trajectory,
    cluster the trajectories by density clustering.
    :param obses: [DataFrame]
    :param interval: Interval time to interpolate.
    :param max_dist: Max distance to determine whether to trajectories are the similar trajectory.
    :return: [[DataFrame]
    '''
    n = len(obses)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            ts0 = obs_to_ts(obses[i], interval)
            ts1 = obs_to_ts(obses[j], interval)
            dtw = DTW(ts0, ts1)
            dist_mat[i, j] = dist_mat[j, i] = dtw.get_average_dist()
    # print(dist_mat)

    # density clustering
    labels = DBSCAN(eps=max_dist, metric='precomputed', min_samples=2).fit_predict(dist_mat)

    # Get the result
    reses = []
    reses.extend([obses[i] for i in range(len(obses)) if labels[i] == -1])
    for i in range(max(labels)+1):
        res = [obses[k] for k in range(len(obses)) if labels[k] == i]
        reses.append(res)
    return reses


def plot_similar_trajectories(obses, user_number,
                              dir='../../res/work/0422_random_trajectories_plot/',
                              is_plot=True, save_json=True):
    '''
    Plot similar trajectories
    :param obses: A list of observation: [DataFrame['lon', 'lat', 'dates']]
                  [obs0, obs1, [obs2, obs3]]
                  For similar trajectories, they will be in the same list.
                  The return of 'find_similar_trajectories'
    :param user_number:
    :return: None
    '''
    n=0
    for obs in obses:
        # whether obs is a list
        if type(obs) == list:
            if is_plot:
                # Plot
                plt.cla()
                plt.title('User %d: Similar Trajectories Plot %d' % (user_number, n))
                for i in range(len(obs)):
                    plt.plot(obs[i].lon, obs[i].lat)
                # plt.show()
                plt.savefig('%ssimilar_plot/User %d Similar Trajectories Plot %d.jpg' % (dir, user_number, n))
            n += 1

            # Save json
            if save_json:
                file_name = 'User %d Similar Trajectories Plot %d' % (user_number, n)
                trajectories_tojson(obs, user_number, dir+'similar_json/'+file_name+'.json', is_json=True)
                trajectories_tojson(obs, user_number, dir + 'similar_js/'+file_name+'.js', is_json=False)


def test_density_clustering(user_number):
    obses, points = get_obses_points(user_number)

    print(points)
    labels = density_clustering(points)
    points = np.array(np.matrix(points).T)
    ind = labels!=-1

    for i, obs in enumerate(obses):
        if labels[i*2] == -1 or labels[i*2+1] == -1:
            continue
        if labels[i*2] == labels[i*2+1]:
            continue
        plt.plot(obs.lon, obs.lat)
    plt.scatter(points[0][ind], points[1][ind], c=labels[ind])
    plt.show()


def test_find_similar_trajectories(user_number):
    print('------ user %d ---------' % user_number)
    obses = find_similar_trajectories(user_number, max_dist=300, interval=5)
    plot_similar_trajectories(obses, user_number)
    return obses


if __name__ == '__main__':
    # test_density_clustering(8)
    for user in range(300, 400):
        test_find_similar_trajectories(user)

