import glob, json
import numpy as np
import pandas as pd
import dask.dataframe as dd
import s4_authors_stats as s4
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool
from scipy.stats import rankdata
from collections import defaultdict
from matplotlib.colors import LogNorm
from dask.diagnostics import ProgressBar
from scipy.stats import pearsonr, spearmanr

NCOLS = 4
SUFFIX = 2020
SUFFIX_STR = '_%d' % SUFFIX

header = '/mnt/e/MAG/mag-2021-01-05/advanced/'
fields_infos = 'FieldsOfStudy.txt'
fos_infos = pd.read_csv(header+fields_infos, header=None, sep='\t')[[0, 1, 2]]
fos_infos.columns = ['field_id', 'rank', 'normalized_name']


test = dd.read_csv('data/author_metrics_final_%s_sample20cent.csv' % SUFFIX, header=None, sep='\t')

test = test.dropna()


# test = test.sample(frac=0.2)
# with ProgressBar():
#     test.to_csv('data/author_metrics_final_%s_sample20cent.csv' % SUFFIX, sep='\t', header=None, index=None, single_file=True)


fos_list = ['95457728', '142362112', '138885662', '144024400', '17744445', '2908647359', '162324750', '33923547',
'144133560', '127313418', '15744967', '205649164', '127413603', '41008148', '39432304', '121332964', '71924100', 
'86803240', '185592680', '192562407']

import threading
lock = threading.Lock()

def is_fos_x(tfos, row):
    return tfos in row[0]

idxs = []
for fos in fos_list:
    i = test.apply(partial(is_fos_x, fos), axis=1, meta=bool)
    idxs.append(i)
    

def get_cit(row):
    c = row[0].split(',')[4]
    return int(float(c))
    
# def get_cit_dict(row, d):
#     print(row[0])
#     c = row[0].split(',')[4]
#     fields = row[0].split('\'')[1].split('\'')[0]
#     for f in fields.split(','):
#         with lock:
#             if f in d:
#                 d[f].append(c)
#             else:
#                 d[f] = [c]
#                 print('aqui', f)
rows = 5
cols = 4
f, ax = plt.subplots(rows, cols, sharey=True, figsize=(23,25))
count = 0

# dist = dict()

# with ProgressBar():
#     cits_fos = test.apply(get_cit_dict, axis=1, meta=object, d=dist).compute()
    
# for field, values in cits_fos.items():

#     percent = np.percentile(values, [20, 70, 80, 90])
#     #     xspace = np.logspace(np.log10(1), np.log10(10000), 10) # contar a quantidade de 0s
#     # percentil do top 5% com mais citações
#     # fixar range do y
#     # cumulativo TODO
#     i = count // cols
#     j = count % cols
#     print(field, i, j, percent)
#     hist, bins = np.histogram(values, bins=100)
#     maxhist = max(hist)
#     hist = hist/maxhist
#     ax[i,j].bar(bins[:-1], hist)
#     #     ax[i,j].hist(cits_fos, bins=100) #, cumulative=True)
#     #     ax[i,j].set_xscale('log')
#     title = fos_infos[fos_infos['field_id'] == int(field)].iloc[0, -1]
#     ax[i,j].set_title(title + (" (90-th percentile %d)" % (percent[-1])))
#     count += 1
# plt.savefig('paper_dist_p90.pdf')
# plt.show()


for field,field_authors_idx in zip(fos_list,idxs):
    with ProgressBar():
        cits_fos = test[field_authors_idx].apply(get_cit, axis=1, meta=(int)).compute()
    percent = np.percentile(cits_fos, [20, 70, 80, 90])
    xspace = np.logspace(np.log10(1), np.log10(10000), 10) # contar a quantidade de 0s
    # percentil do top 5% com mais citações
    # fixar range do y
    # cumulativo TODO
    i = count // cols
    j = count % cols
    print(field, i, j, percent)
    bla = cits_fos[cits_fos > 0]
    print('zeros', len(cits_fos) - len(bla))
#     hist, bins = np.histogram(bla, bins=10)
#     maxhist = max(hist)
#     hist = hist/maxhist
#     ax[i,j].bar(bins[:-1], hist)
#     print(hist)
#     print(bins)
#     print(hist/maxhist)
    weights = np.ones_like(bla) / len(bla)
    ax[i,j].hist(bla, weights=weights, bins=xspace, cumulative=True)
    ax[i,j].set_xscale('log')
    title = fos_infos[fos_infos['field_id'] == int(field)].iloc[0, -1]
    ax[i,j].set_title(title + (" (90-th percentile %d)" % (percent[-1])))
    count += 1

plt.savefig('cits_dist_p90_20percent_normalized_c.pdf')
plt.show()
    