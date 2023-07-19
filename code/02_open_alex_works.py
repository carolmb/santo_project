import datetime
import pandas as pd
import openalexraw as oaraw

from pathlib import Path
from tqdm.auto import tqdm

openAlexPath = Path('../openalex-snapshot')
oa = oaraw.OpenAlex(
    openAlexPath = openAlexPath
)

authors = pd.read_csv('authors_10_200_07-11-2023.tsv', sep='\t')
valid_authors = authors['0']
valid_authors = set(valid_authors)

openAlexPath = Path('../openalex-snapshot')
oa = oaraw.OpenAlex(
    openAlexPath = openAlexPath
)

openAlexPath = Path('../openalex-snapshot')
oa = oaraw.OpenAlex(
    openAlexPath = openAlexPath
)

max_size = 5000000

fields = ['id', 'publication_date', 'title', 'referenced_works', 'authorships']
entityType = "works"
entitiesCount = oa.getRawEntityCount(entityType)
works = []
for entity in tqdm(oa.rawEntities(entityType),total=entitiesCount):
    authors = entity['authorships']
    for author in authors:
#         print(author)
        if 'id' in author['author']:
            if author['author']['id'] in valid_authors:
                row = []
                for field in fields:
                    row.append(entity[field])
                works.append(tuple(row))
                break
        else:
            print("no author", entity)
    
    if len(works) > max_size:
        today = datetime.datetime.today().strftime('%m-%d-%Y-%H-%M')
        works_df = pd.DataFrame(works)
        print(works_df.head())
        works_df.to_csv('works_valid_{}.tsv'.format(today), sep='\t')
        del works_df
        works = []
        
        
today = datetime.datetime.today().strftime('%m-%d-%Y-%H-%M')
works_df = pd.DataFrame(works)
works_df.to_csv('works_valid_{}.tsv'.format(today), sep='\t')