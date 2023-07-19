import glob, json, tqdm, csv, sys
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

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
NCOLS = 4
SUFFIX = 2020
SUFFIX_STR = '_%d' % SUFFIX

header = '/mnt/e/MAG/mag-2021-01-05/advanced/'
fields_infos = 'FieldsOfStudy.txt'
fos_infos = pd.read_csv(header+fields_infos, header=None, sep='\t')[[0, 1, 2]]
fos_infos.columns = ['field_id', 'rank', 'normalized_name']


def to_process_papers(file):
    papers_fos_dict = defaultdict(lambda:set())
    papers_per_fos = dd.read_csv(file, header=None)[[1, 6]]
    papers_per_fos.columns = ['paper_id', 'parents_id']
    papers_per_fos = papers_per_fos.dropna()
    for idx,row in papers_per_fos.iterrows():
        for fos in row['parents_id'].split(','):
            papers_fos_dict[int(fos)].add(int(row['paper_id']))
    
    import pickle
    papers_fos_dict = dict(papers_fos_dict)
    with open(file.replace('paper', '1papers_per_fos'), 'wb') as handle:
        pickle.dump(papers_fos_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return papers_fos_dict


if __name__ == '__main__':
#     from s4_authors_stats import _step_1

#     authors_complete = dd.read_csv('data/AuthorsMetrics_split/authors_metrics_full_%d' % SUFFIX, sep='\t', header=None)
#     authors_complete.columns = ['author_id', 'cits', 'birth_year', 'citation_count', 'weights', 'fos']

#     def sum_cits(row):
#         c = sum(json.loads(row['citation_count']))
#         return c

#     authors_complete = authors_complete[authors_complete.apply(sum_cits, meta=(int), axis=1) > 0]
#     print(authors_complete.head())
    
    
#     ------------------------
    
    
#     files = glob.glob('data_temp/FOS_split/fields_papers_*.csv')
    

#     output_maps = process_map(to_process_papers, files, max_workers=16)
#     final_maps = defaultdict(set, output_maps[0])
#     for i in range(1, len(output_maps)):
#         m = output_maps[i]
#         for k, v in m.items():
#             final_maps[k] = final_maps[k] | v
#         output_maps[i] = {}
        
#     for k, v in final_maps.items():
#         print(k, len(v))

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
    NCOLS = 4
    SUFFIX = 2020
    SUFFIX_STR = '_%d' % SUFFIX

    header = '/mnt/e/MAG/mag-2021-01-05/advanced/'
    fields_infos = 'FieldsOfStudy.txt'
    fos_infos = pd.read_csv(header+fields_infos, header=None, sep='\t')[[0, 1, 2]]
    fos_infos.columns = ['field_id', 'rank', 'normalized_name']

    # pairs_authors_dd = dd.read_csv('data/pair_csv_%d_byAuthorIDprocessed_pairs.csv' % SUFFIX, sep='\t', header=None, names=['author_id', 'cits'])
    # pairs_authors_dd = dd.read_csv('data/pair_csv_10a_%dmerged_processed.csv' % SUFFIX, sep='\t', header=None, names=['author_id', 'cits'])
    '''
    pairs_authors_dd = dd.read_csv('data/PairAuthors2csv_split/pair_csv_year_rev1_%d_*' % SUFFIX, sep='\t', header=None, names=['author_id', 'cits'])
    pairs_authors_dd = pairs_authors_dd.set_index('author_id', sorted=True)
    pairs_authors_dd.head()

    print('aqui')
    authors_infos = dd.read_csv('data/authors_infos_full_10a_final_%d' % SUFFIX, sep='\t', header=None, 
                                names=['author_id', 'birth_year', 'citation_count'])
    authors_infos = authors_infos.set_index('author_id', sorted=True)
    authors_infos.head()

    
    pairs_authors_dd2 = pairs_authors_dd.merge(authors_infos, left_index=True, right_index=True)
    from dask.diagnostics import ProgressBar
    with ProgressBar():
        pairs_authors_dd2.to_csv('data/pairs_authors_dd2.csv', sep='\t', header=None, single_file=True)
    '''
    authors_fos = dd.read_csv('data/valid_authors_%d_fos_div_filter.csv' % SUFFIX, sep='\t', header=None, names=['author_id', 'weights', 'fos'], dtype={'fos':'str', 'author_id':'str'})

    files = glob.glob('data/pair_authors_dd2_*')
    
    print(sys.argv[1:])
    i = int(sys.argv[1])
    
    print(i)
    pairs_authors_dd2 = pd.read_csv(files[i], sep='\t', header=None, quoting=csv.QUOTE_NONE, names=['author_id', 'weights', 'birth_year', 'cits'], dtype={'author_id':'str'})
         
    print('aqui3')
    authors_complete = authors_fos.merge(pairs_authors_dd2, on='author_id')


    from dask.diagnostics import ProgressBar
    with ProgressBar():
        authors_complete.to_csv('data/authors_metrics_full_rev2_%d_' % SUFFIX + files[i].split('_')[-1], sep='\t', header=None, single_file=True)

