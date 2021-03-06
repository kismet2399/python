import music.recall.item_base as ib
import music.recall.user_base as ub
import operator
import music.recall.config as conf

cf_rec_lst_outfile = conf.cf_rec_lst_outfile
# 不同召回的前缀标识
UCF_PREFIX = conf.UCF_PREFIX
ICF_PREFIX = conf.ICF_PREFIX

# load CF train data
train = {}
with open(conf.cf_train_data_path,'r',encoding='utf-8') as f:
    train = eval(f.read())
print('CF train data have loaded! Start compute user similarity ...')
# print(train)

reclst = dict()
'''
user base
'''
# 计算用户与用户的相似度矩阵并存储
user_user_sim = ub.user_sim(train)
print('compute done! saving user-user similarity matrix ... ')


# 对每个用户计算推荐物品集合 recall物品基于用户相似度
for user_id in train.keys():
    rec_item_list = ub.recommend(user_id, train, user_user_sim, 10)
    user_id = UCF_PREFIX + user_id
    reclst[user_id] = sorted(rec_item_list.items(), key=operator.itemgetter(1),reverse=True)[0:20]
print('User base done! Item base starting ...')

'''
item base
'''
# 计算歌曲与歌曲的相似度矩阵并存储
item_item_sim = ib.item_sim(train)

# 对每个用户计算推荐物品集合 recall物品基于物品相似度
for user_id in train.keys():
    item_list = ib.recommendation(train, user_id, item_item_sim, 10)
    user_id = ICF_PREFIX + user_id
    reclst[user_id] = sorted(item_list.items(), key=operator.itemgetter(1), reverse=True)[0:20]

with open(cf_rec_lst_outfile,'w',encoding='utf-8') as rcf:
    rcf.write(str(reclst))
