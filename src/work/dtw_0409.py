import numpy as np
from math import radians, cos, sin, asin, sqrt


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


class DTW:
    '''
    Dynamic Time Wrapping Algorithm.
    '''
    def __init__(self, ts1, ts2):
        '''
        :param ts1: Time series 1 with m time periods. A list [[lon, lat]].
        :param ts2: Time series 2 with n time periods. A list.
        '''
        self.ts1 = ts1
        self.ts2 = ts2
        self.ots1, self.ots2 = ts1, ts2
        self.m = len(ts1)
        self.n = len(ts2)
        self.directions = [[0, 1], [1, 1], [1, 0]]

    def is_in_bound(self, i, j):
        return 0 <= i < self.m and 0 <= j < self.n

    def get_dist_matrix(self, ts1, ts2):
        dist_mat = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                dist_mat[i, j] = cal_distance(ts1[i], ts2[j])
        return dist_mat

    @staticmethod
    def get_derivatives(ts):
        ts.insert(0, ts[0])
        ts.append(ts[-1])
        dts = [[(ts[i+1][0] - ts[i-1][0])/2, (ts[i+1][1] - ts[i-1][1])/2] for i in range(1, len(ts) - 1)]
        return dts

    def get_derivatives_dist_matrix(self, ts1, ts2):
        dts1 = self.get_derivatives(ts1)
        dts2 = self.get_derivatives(ts2)
        return self.get_dist_matrix(dts1, dts2)

    def shortest_path_dtw(self, is_derivatives=False):
        '''
        Solving DTW with shortest path algorithm
        :param is_derivatives:
        :return:
        '''
        mat = np.ones((self.m, self.n)) + np.inf
        if is_derivatives:
            dist_mat = self.get_derivatives_dist_matrix(self.ts1, self.ts2)
        else:
            dist_mat = self.get_dist_matrix(self.ts1, self.ts2)
        mat_ind = {}
        mat[0, 0] = 0

        for layer in range(1, self.m + self.n):
            for i in range(layer+1):
                node = (i, layer-i)
                if not self.is_in_bound(node[0], node[1]):
                    continue
                dist = dist_mat[node]
                for dir in self.directions:
                    lst_node = (node[0] - dir[0], node[1] - dir[1])
                    if not self.is_in_bound(lst_node[0], lst_node[1]):
                        continue
                    tmp_value = mat[lst_node] + dist
                    if tmp_value < mat[node]:
                        mat[node] = tmp_value
                        mat_ind[node] = [lst_node]
                    elif tmp_value == mat[node]:
                        mat_ind[node].append(lst_node)

        print('Minimum distance = %f' % mat[self.m-1, self.n-1])

        return mat, mat_ind

    def get_trace(self, mat_ind):
        '''
        Now only get one best trace
        :param mat_ind:
        :return:
        '''
        trace = [(self.m-1, self.n-1)]
        node = trace[0]
        while node != (0, 0):
            last_node = mat_ind[node][0]
            trace.append(last_node)
            node = last_node

        # Print trace
        trace_mat = np.zeros((self.m, self.n))
        for t in trace:
            trace_mat[t[0], t[1]] = 1
        # print(trace_mat)

        return trace, trace_mat

    def get_new_ts(self, mat_ind=None):
        if mat_ind is None:
            _, mat_ind = self.shortest_path_dtw()
        trace, _ = self.get_trace(mat_ind)
        ts1, ts2 = self.ts1, self.ts2
        nts1, nts2 = [ts1[0]], [ts2[0]]
        for t in trace[::-1]:
            nts1.append(ts1[t[0]])
            nts2.append(ts2[t[1]])
        return nts1, nts2

    def get_middle_ts(self, mat_ind=None):
        nts1, nts2 = self.get_new_ts(mat_ind)
        return ((np.array(nts1) + np.array(nts2)) / 2).tolist()

    def get_average_dist(self, mat_ind=None):
        nts1, nts2 = self.get_new_ts(mat_ind)
        dist = 0
        for t in len(nts1):
            dist += cal_distance(nts1[t], nts2[t])
        return dist / len(nts1)


if __name__ == '__main__':
    pass
    '''
    If you have ts1 and ts2
    dtw = DTW(ts1, ts2)
    dtw.get_average_dist()
    '''