import glob, json, tqdm
import numpy as np
import pandas as pd
import s4_authors_stats as s4
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import pearsonr, spearmanr
from matplotlib.colors import LogNorm

from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
SUFFIX = 2020
from s4_authors_stats import get_h_index
import numpy as np

authors_complete = dd.read_csv('data/authors_metrics_full_rev2_%d_a*' % SUFFIX, sep='\t', header=None)
authors_complete.columns = ['none', 'author_id', 'weights', 'fos', 'cits', 'birth_year', 'citation_count']

print("data loaded")

def sum_cits(row):
    c = sum(json.loads(row['citation_count']))
    return c

# authors_complete = authors_complete[authors_complete.apply(sum_cits, meta=(int), axis=1) > 0]


def _step_1(row):
    

    if type(row['citation_count']) == type(0.0):
        print(row['author_id'])
        print('citation list nan')
        return None
    c = json.loads(row['citation_count'])

    b = row['birth_year']

    if np.isnan(b):
        print(row['author_id'])
        ValueError('A very specific bad thing happened. OH NO (line 62)')

    h_index = get_h_index(c)
#     row['h_index'] = h_index
    total_cits = sum(c)
#     row['total_cits'] = total_cits
    papers = len(c)
#     row['n_papers'] = papers

    max_w = 0
    max_p = 0
    s = 0
    clist = []
#     weights = []
    author_colabs = json.loads(row['cits'][1:-1].replace('\"\"', '\"'))
    s = 0

    for colab, colab_infos in author_colabs.items():
        colab_temp = []
        for temp in colab_infos:
            if type(temp) == type([1]):
                colab_temp += temp
            else:
                colab_temp.append(temp)
        colab_infos = colab_temp
        if total_cits == 0:
            w = 0
        else:
            w = sum(colab_infos)/total_cits
            if max_w < w:
                max_w = w
                max_p = len(colab_infos)/papers
                clist = colab_infos

#             weights.append(w)
        s += w

    return row['author_id'],h_index, total_cits, papers, max_w, max_p, clist, b, row['fos'], c


from dask.diagnostics import ProgressBar
    

def step_1(pairs_authors_dd):
    with ProgressBar():
        res = pairs_authors_dd.apply(_step_1, axis=1, meta=("Result", object))
#                                      meta=[('h_index', int), ('total_cits', int),   
#                                                             ('n_papers', int), ('max_ws', float), ('max_colabs', float),
#                                                             ('colabs_list', object), ('birth_year', int),
#                                                             ('fos', object), ('cit_list', object)])
                                     
    print(res.head())
    return res


print(authors_complete.head())

# H, C, P, WS, WP, CL, Y, CH, FOS = step_1(authors_complete)
res = step_1(authors_complete)
res.name = 'temp'
with ProgressBar():
    res.to_csv('data/author_metrics_final_%s.csv' % SUFFIX, sep='\t', header=None, index=None, single_file=True)

print('saving results')

# open('results/hindex_rev2_full_%d.json' % SUFFIX,'w').write(json.dumps(H, indent=4))
# open('results/cits_rev2_full_%d.json' % SUFFIX,'w').write(json.dumps(C, indent=4))
# open('results/papers_rev2_full_%d.json' % SUFFIX,'w').write(json.dumps(P, indent=4))
# open('results/ws_rev2_full_%d.json' % SUFFIX,'w').write(json.dumps(WS, indent=4))
# open('results/wp_rev2_full_%d.json' % SUFFIX,'w').write(json.dumps(WP, indent=4))
# open('results/citlist_rev2_full_%d.json' % SUFFIX,'w').write(json.dumps(CL, indent=4))
# open('results/birth_rev2_full_%d.json' % SUFFIX, 'w').write(json.dumps(Y, indent=4))
# open('results/fos_rev2_full_%d.json' % SUFFIX,'w').write(json.dumps(FOS, indent=4))
# open('results/ch_rev2_full_%d.json' % SUFFIX, 'w').write(json.dumps(CH, indent=4))