import pandas  as pd
import math
import operator
from cf import use_base

df = pd.read_csv('../data/u.data',
                 sep='\t',
                 # nrows=100,
                 names=['user_id', 'item_id', 'rating', 'timestamp'])
print(max(df['rating']))
d = dict()
for _, row in df.iterrows():
    user_id = str(row['user_id'])
    item_id = str(row['item_id'])
    rating = row['rating']
    if user_id not in d.keys():
        d[user_id] = {item_id: rating}
    else:
        d[user_id][item_id] = rating