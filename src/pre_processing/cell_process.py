__author__ = 'VikiQiu'

from src.pre_processing.read_data import get_link, get_cellSheet
from src.pre_processing.funs import cal_distance
from sklearn.neighbors import KDTree
import numpy as np
import time

links = get_link()
cells = get_cellSheet()
cells = cells[0:10]

start = time.time()
tree = KDTree(np.array(links[['blon', 'clat']]), leaf_size=1000)
end = time.time()

this_link = []
dists = []

for i in range(len(cells)):
    cell = cells[i:(i+1)]
    query = cell[['lon', 'lat']].as_matrix()[0]
    _, ind = tree.query(query)
    ind = ind[0][0]
    dist = cal_distance(query, links.loc[ind:ind, ['blon', 'clat']].values[0])
    dists.append(dist)
    this_link.append(links[ind:(ind+1)].values[0])
    if i % 1 == 0: print(i)

this_link = np.array(this_link).T
cells['link'] = this_link[0]
cells['link_lon'] = this_link[1]
cells['link_lat'] = this_link[2]
cells['link_dist'] = dists
print(cells.head(10))
#cells.to_csv('../../res/cells_process.csv', index=False)

print('prepare time', end-start)

