__author__ = 'Victoria'

'2017-10-16'

from src.pre_processing.read_data import get_cellSheet
from src.pre_processing.funs import output_data_js


def output_cells_js(cell_type='baidu', place='hf'):
    cells = get_cellSheet(cell_type=cell_type, place=place)
    filename = '../../web/cells_data_%s.js' % place
    output_cells_js(cells, filename)

# output_cells_js()




