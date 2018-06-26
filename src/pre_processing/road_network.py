__author__ = 'Victoria'

# 2017-10-10

from math import cos, sin, asin, sqrt
import random
import matplotlib.pyplot as plt


class Vertex:
    def __init__(self, lon: float, lat: float):
        '''
        init Vertex
        :param lon: Longitude of Vertex [float]
        :param lat: Latitude of Vertex [float]
        :var pos: Vertex position [str] = '%.6f|%.6f' % (lon, lat)
        :var adjacent: Adjacent Vertex dictionary: {Vertex[Vertex]: distance[float]}
        :var main: Main road or not
        '''
        self.pos = '%.4f|%.4f' % (lon, lat)
        self.lon = lon
        self.lat = lat
        self.adjacent = {}
        self.main = True

    def __str__(self):
        return self.pos

    def get_position(self):
        return self.pos

    def get_lon(self):
        return self.lon

    def get_lat(self):
        return self.lat

    def add_neighbor(self, neighbor):
        '''
        Add a neighbor to the Vertex
        :param neighbor: Another Vertex class adjacent to this Vertex
        :return:
        '''
        self.adjacent[neighbor] = self.distance_from(neighbor)

    def get_connections(self):
        '''
        Get neighbor vertexes list: [Vertex]
        :return:
        '''
        return self.adjacent.keys()

    def distance_from(self, node):
        '''
        Calculate distance from two Vertexes
        :param node: Another Vertex
        :return:
        '''
        lon1, lat1, lon2, lat2 = self.lon, self.lat, node.get_lon(), node.get_lat()
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # earth radius, km
        return c * r * 1000

    def next_step(self, node, k=0):
        '''
        Next kth nearest step to get end vertex node from the current node.
        :param node: End Vertex
        :param k: the kth good step
        :return: return the kth nearest Vertex
        '''
        connections = [neighbour for neighbour in self.get_connections()]
        assert k < len(connections), 'k=%d is larger than the adjacent number %d' % (k, len(connections))
        dists = []
        for i in range(len(connections)):
            dists.append(connections[i].distance_from(node))
        ind = sorted(range(len(dists)), key=dists.__getitem__)
        neighbour = connections[ind[k]]  # get the kth nearest neighbor
        return neighbour

    def reset_main(self):
        '''
        Reset self.main
        :return:
        '''
        if len(self.adjacent) > 2:
            self.main = True
        else:
            self.main = False


class Graph:
    def __init__(self):
        '''
        init Graph class
        :var vert_dict: {Vertex.pos: Vertex}
        :var num_vertices: number of Vertex in Graph
        :var link_dict: {link_id: [Vertex.pos]}
        :var key_vert: Vertex which is the main node.{Vertex.pos: Vertex}
        :var road_segment: {road_name: [Vertex.pos]}. road_name = '%sto%s' % (start_pos, end_pos)
        '''
        self.vert_dict = {}
        self.key_vert = {}
        self.link_dict = {}
        self.road_segment = {}

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, lon: float, lat: float, link):
        '''
        Add a vertex to graph if it not exists yet.
        Add the vertex to the link_dict
        :param lon:
        :param lat:
        :param link:
        :return: The Vertex added
        '''
        new_vertex = Vertex(lon, lat)
        pos = new_vertex.get_position()
        if pos not in self.vert_dict:
            self.vert_dict[pos] = new_vertex
        if link in self.link_dict:
            self.link_dict[link].append(self.vert_dict[pos])
        else:
            self.link_dict[link] = [self.vert_dict[pos]]
        return self.vert_dict[pos]

    def get_vertex(self, pos: str):
        '''
        Get a Vertex according to its position
        :param pos:
        :return:
        '''
        if pos in self.vert_dict:
            return self.vert_dict[pos]
        else:
            return None

    def get_vertex_dict(self):
        return self.vert_dict

    def get_link_dict(self):
        return self.link_dict

    def add_edge(self, frm: Vertex, to: Vertex):
        '''
        add edge between two vertexes
        :param frm: start vertex
        :param to: end vertex
        :return: no return
        '''
        assert frm.pos in self.vert_dict, 'start vertex %s not in graph.' % frm
        assert to.pos in self.vert_dict, 'end vertex %s not in praph.' % to
        dists = frm.distance_from(to)
        self.vert_dict[frm.pos].add_neighbor(self.vert_dict[to.pos])
        self.vert_dict[to.pos].add_neighbor(self.vert_dict[frm.pos])

    def get_vertex_position(self):
        return self.vert_dict.keys()

    def get_num_vertices(self):
        return len(self.vert_dict)

    def get_all_vertex(self):
        return [self.vert_dict.get(key) for key in self.vert_dict.keys()]

    def pos_is_in(self, node_pos):
        '''
        Judge is a Vertex.pos in the graph
        :param node_pos: A Vertex.pos
        :return: True / False
        '''
        return node_pos in self.vert_dict

    def vertex_is_in(self, node: Vertex):
        '''
        Judge is a Vertex node in the graph
        :param node: A Vertex
        :return: True / False
        '''
        return node.pos in self.vert_dict

    def reset_main_roads(self):
        '''
        Reset every Vertex whether it is a main road.
        Main Road has at least three adjcant.
        :return:
        '''
        for v in self.vert_dict:
            self.vert_dict[v].reset_main()

    def check_vertex(self, v: str):
        return v in self.vert_dict

    def refresh_key_vert(self):
        '''
        Refresh key_vert directory
        :return:
        '''
        for key in self.vert_dict.keys():
            if self.vert_dict[key].main:
                self.key_vert[key] = self.vert_dict[key]

    def refresh_road_segment(self):
        '''
        Refresh road segment
        :return:
        '''
        for link in self.link_dict.keys():
            roads = self.link_dict[link]
            start = roads[0] # Vertex.pos: str
            start_ind = 0

            for i in range(1, len(roads)):
                tmp_node = self.vert_dict[roads[i]] # Vertex

                if start is None:
                    start = roads[i]
                    start_ind = i
                    continue

                if tmp_node.main:
                    end = roads[i] # Vertex.pos: str
                    road_name = '%sto%s' % (start, end)
                    self.road_segment[road_name] = roads[start_ind:(i+1)]

                    start = None

                # TODO: What if the last road is also a secondary road?



def find_shortest_path(graph: Graph, start: str, end: str, paths=[], dists=0):
    '''
    // need a further improvement
    find a short way from start to end
    :param graph:
    :param start: a start node name
    :param end: an end node name
    :param paths: The previous path to start vertex
    :param dists: The previous distance to start vertex
    :return: a short path with node names in list
    '''

    start_vertex = graph.get_vertex(start)
    end_vertex = graph.get_vertex(end)
    visited = {}

    paths.append(start)
    if start == end:
        return [paths, dists]
    if start not in graph.get_vertex_position():
        print('start vertex %s not in the graph.' % start)
        return None

    start_vertex = graph.get_vertex(start)
    end_vertex = graph.get_vertex(end)
    n = len(start_vertex.get_connections())
    for i in range(n):
        node = start_vertex.next_step(end_vertex, i)
        if node.pos in paths:
            continue
        dists = dists + start_vertex.distance_from(node)
        return find_shortest_path(graph, node.pos, end, paths, dists)


def get_random_trail(graph: Graph, start: str, step: int):
    '''
    Go a random trail with given step number.
    :param graph:
    :param start: String. Vertex name.
    :param step:
    :return:
    '''
    if not graph.check_vertex(start):
        print('Start illegal. Not in the graph.')
        return

    paths = [start]
    node = start
    while len(paths) < step:
        neighbors = graph.get_vertex(node).adjacent
        for v in paths:
            if graph.get_vertex(v) in neighbors:
                del neighbors[graph.get_vertex(v)]
        node = random.choice(list(neighbors.keys())).pos
        paths.append(node)
    print_paths(paths)
    return paths


def get_graph(filename='../../data/hefei_road/link_baidu.txt'):
    '''
    Get a road graph through links(filename)
    :param filename: filename of road links
    :return:
    '''

    # Initialize class Graph
    graph = Graph()
    with open(filename) as f:
        for line in f.readlines():
            # e.x: line = 'road_id  lon1,lat1|lon2,lat2|...'
            line_tmp = line.split('\t')
            # get road id
            link = line_tmp[0]
            # get the segments on the road
            line_tmp_tmp = line_tmp[1].split('|')
            nodes = []
            for i in range(len(line_tmp_tmp)):
                pos = line_tmp_tmp[i]
                # to do: is the node already in graph
                new_vertex = graph.add_vertex(float(pos.split(',')[0]), float(pos.split(',')[1]), link)
                nodes.append(new_vertex)
                if i > 0:
                    graph.add_edge(nodes[i], nodes[i-1])
    graph.reset_main_roads()
    return graph


def get_wuhu_graph():
    graph = get_graph('../../data/road_test/link_baidu_wuhu.txt')
    return graph


def print_paths(paths):
    '''
    Print path
    :param paths: [Vertex.pos: str]
    :return:
    '''
    x = []
    y = []
    for v in paths:
        x.append(float(v.split('|')[0]))
        y.append(float(v.split('|')[1]))
    # plt.plot(x, y)
    # plt.savefig('a.jpg')

    plt.close()  # clf() # 清图  cla() # 清坐标轴 close() # 关窗口
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(True)  # 添加网格
    plt.ion()  # interactive mode on

    for i in range(len(x)):
        ax.plot(x[:i], y[:i])  # 折线图
        ax.set_title('Time %d' % i)
        plt.show()
        # ax.lines.pop(1)  删除轨迹
        # 下面的图,两船的距离
        plt.pause(0.5)
        print('plot %d done.' % i)


if __name__ == '__main__':
    graph = get_graph()
    graph.refresh_key_vert()
    print(len(graph.key_vert))
    # vertices = graph.get_vertex_position()
    # print(get_random_trail(graph, '117.3804|31.8838', 10))
    # print(vertices)
    # [path, dist] = find_shortest_path(graph, '117.3804|31.8838', '117.3747|31.8941')
    # print(path)
    # print(dist)

