import music.recallSelf.config as conf
import math
import pandas as pd

'''
处理线上推荐请求
'''

a = conf.a
user_id = '014cacfaac9e34fcf2e7a73a163eb889'

# 1,载入模型特征和模型权重,交叉特征,商品信息和用户信息
category_feat_dict = conf.kismet_read(conf.user_feat_map_file)
cross_feat_dict = conf.kismet_read(conf.cross_file)
model_dict = conf.kismet_read(conf.model_file)

w = model_dict['w']
b = model_dict['b']

# 2,召回集合
rec_item_all = dict()
# 2.1 用户相似度召回
cf_rec_lst = conf.kismet_read(conf.cf_rec_lst_outfile)
rec_user_list = cf_rec_lst[conf.UCF_PREFIX + user_id]
for item, score in rec_user_list:
    rec_item_all[item] = float(score) * a

# 2.2 物品相似度召回
rec_item_list = cf_rec_lst[conf.ICF_PREFIX + user_id]
for item, score in rec_item_list:
    if rec_item_all.get(item, -1) == -1:
        rec_item_all[item] = float(score) * (1 - a)
    else:
        rec_item_all[item] += float(score) * (1 - a)

# 3 添加特征
rec_lst = []
user_df = conf.gen_user_profile()
item_df = conf.gen_music_meta()

age, gender, salary, province = '', '', '', ''
for _, row in user_df.loc[user_df['user_id'] == user_id, :].iterrows():
    age, gender, salary, province = row['age'], row['gender'], row['salary'], row['province']
    (age_idx, gender_idx, salary_idx, province_idx) = (category_feat_dict['age_' + age],
                                                       category_feat_dict['gender_' + gender],
                                                       category_feat_dict['salary_' + salary],
                                                       category_feat_dict['province_' + province])
for item_id in rec_item_all:
    for _, row in item_df.loc[item_df['item_id'] == int(item_id), :].iterrows():
        location, item_name = row['location'], row['item_name']
    location_idx = category_feat_dict['location_' + location]
    cross_value = float(cross_feat_dict.get(user_id + '_' + item_id, 0))
    # 4 计算分值
    wx_score = float(b)
    wx_score += w[age_idx] + w[gender_idx] + w[salary_idx] + w[province_idx] + w[location_idx]
    # 用连续特征score*cross_value
    wx_score += cross_value * w[-1]
    # sigmoid:p(y=1|x)=1/(1+exp(-wx))
    final_rec_score = 1 / (1 + math.exp(-wx_score))

    rec_lst.append((item_id, item_name, final_rec_score))

# 排序
rec_lst = sorted(rec_lst, key=lambda x: x[2], reverse=True)[:10]

# 打印
rec_list = [' ==> '.join([id, name, str(score)]) for id, name, score in rec_lst]
print('\n'.join(rec_list))
