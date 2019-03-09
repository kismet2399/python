import music.recallSelf.item_base as ib
import music.recallSelf.user_base as ub
import operator
import music.recallSelf.config as conf

# 召回集合
cf_rec_lst_outfile = conf.cf_rec_lst_outfile
# 不同召回的前缀标识
UCF_PREFIX = conf.UCF_PREFIX
ICF_PREFIX = conf.ICF_PREFIX

# 读取训练集数据
train = conf.kismet_read(conf.cf_train_data_path)
# print(train)

reclst = dict()

'''
用户相识度召回
'''
# 用户相识度
user_user_sim = ub.user_sim(train)
# 用户召回集合
for user_id in train.keys():
    recommend = ub.recommend(user_id, train, user_user_sim, 10)
    u_key = UCF_PREFIX + user_id
    reclst[u_key] = sorted(recommend.items(), key=lambda x: x[1], reverse=True)[0:20]

'''
物品相识度召回
'''
# 物品相识度
item_item_sim = ib.item_sim(train)
# 物品召回集合
for user_id in train.keys():
    recommendation = ib.recommendation(train, user_id, item_item_sim, 10)
    u_key = ICF_PREFIX + user_id
    reclst[u_key] = sorted(recommendation.items(), key=operator.itemgetter(1), reverse=True)[0:20]

conf.kismet_write(reclst, cf_rec_lst_outfile)
