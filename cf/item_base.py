import math
import pandas as pd

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

# 1. 计算物品与物品的相似度矩阵
C = dict()
N = dict()
for u, items in d.items():
    for i in items:
        # item拥有的user数据量
        if N.get(i, -1) == -1:
            N[i] = 0
        N[i] += 1
        if C.get(i, -1) == -1:
            C[i] = dict()
        for j in items:
            if i == j: continue
            if C[i].get(j, -1) == -1:
                C[i][j] = 0
            C[i][j] += 1

# 计算最终相似度矩阵
for i, related_items in C.items():
    for j, cij in related_items.items():
        C[i][j] += 2 * cij / ((N[i] + N[j]) * 0.1)

user_id = "196"
rank = dict()
Ru = d[user_id]
for i, rating in Ru.items():
    for j, sim in sorted(C[i].items(),
                         key=lambda x: x[1], reverse=True)[0:10]:
        # 过滤评论过得物品集合
        if j in Ru:
            continue
        elif rank[j].get(j, -1) == -1:
            rank[j] = 0
        rank[j] += sim * rating

print(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:10])
