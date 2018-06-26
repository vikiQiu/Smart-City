__author__ = 'Victoria'

# 2017/09/25

from src.pre_processing.read_data import get_link, get_signals, get_cellSheet
import datetime
import time
import numpy as np
import pandas as pd
from src.pre_processing.funs import clean_signal

part_start = 180
part_end = 560
mon = 5
dd = 31
hh_start = 19
hh_end = 20
file_date = '20170601'
filename_in = ('..\..\data\signals_pro\%.2d%.2d\%.2dt%.2d\signals_%.2dt%.2d_%.5d.csv'
               % (mon, dd, hh_start, hh_end, hh_start, hh_end, part_start))
filename_out = ('..\..\data\signals_pro\%.2d%.2d\%.2dt%.2d\signals_%.2dt%.2d_%.5d.csv'
               % (mon, dd, hh_start, hh_end, hh_start, hh_end, part_end))
signals_before = pd.read_csv(filename_in)
signals = pd.read_csv(filename_out)
print(len(signals_before))
print(len(signals))
signals = signals.sort_values(['user_id', 'dates'])
print(signals.head(20))
signals = pd.concat([signals_before, signals])
print(len(signals))
signals = signals.sort_values(['user_id', 'dates'])
print(signals.head(20))
signals = clean_signal(signals)
print(len(signals))
print(signals.head(20))
signals.to_csv(filename_out, index=False)


#links = get_link()
#cells = get_cellSheet()
#print (signals.groupby('user_id').count())
#cells_tmp = signals[signals.user_id=='++415muGS8rrv9J2uwt8IA=='].sort_values('dates').cell_id
#print(cells_tmp)
#print(cells[cells.cell_id.isin(cells_tmp)])

signals.sort_values
