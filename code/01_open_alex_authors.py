# !pip install openalex-raw
import datetime
import pandas as pd
import openalexraw as oaraw

from pathlib import Path
from tqdm.auto import tqdm

openAlexPath = Path('../openalex-snapshot')
oa = oaraw.OpenAlex(
    openAlexPath = openAlexPath
)

entityType = "authors"
entitiesCount = oa.getRawEntityCount(entityType)

min_papers = 10
min_cits = 200

authors = []

for entity in tqdm(oa.rawEntities(entityType),total=entitiesCount):
    openAlexID = entity["id"]
    concepts = entity['x_concepts']
    works = entity['works_count']
    cits = entity['cited_by_count']
    if works >= min_papers and cits >= min_cits and len(concepts) > 0:
        filtered_concepts = []
        for c in concepts:
            if c['level'] == 0 and c['score'] > 50:
                filtered_concepts.append("{} {}".format(c['display_name'], c['score']))
        
        authors.append((openAlexID, works, cits, ','.join(filtered_concepts)))
        
today = datetime.datetime.today().strftime('%m-%d-%Y')
authors_df = pd.DataFrame(authors)
authors_df.to_csv('authors_{}_{}_{}.tsv'.format(min_papers, min_cits, today), sep='\t', index=None)