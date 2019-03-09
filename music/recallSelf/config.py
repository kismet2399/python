import pandas as pd
import os

data_path = '../raw_data'

'''
原始数据
'''
music_meta = os.path.join(data_path, 'music_meta')
user_profile = os.path.join(data_path, 'user_profile.data')
user_watch_pref = os.path.join(data_path, 'user_watch_pref.sml')

'''
缓存的中间数据,开发时存储在redis中
'''
# 用户->商品->打分字典
cf_train_data_path = '../data/cf_train.data'

# 召回集合
cf_rec_lst_outfile = '../data/cf_reclst.data'

# 模型离散处理特征
user_feat_map_file = '../data/map/user_feat_map'
# 交叉特征
cross_file = '../data/map/cross_file'
# 训练后的模型数据
model_file = '../data/map/model_file'

# 不同召回集合的前缀
UCF_PREFIX = 'UCF_'
ICF_PREFIX = 'ICF_'

# user base在召回中的权重
a = 0.6


# 初始化元数据
# nrows 取出数据的行数
def gen_user_watch(nrows=None):
    return pd.read_csv(user_watch_pref,
                       sep='\001',
                       nrows=nrows,
                       names=['user_id', 'item_id', 'stay_seconds', 'hour'])


def gen_user_profile(nrows=None):
    return pd.read_csv(user_profile,
                       sep=',',
                       nrows=nrows,
                       names=['user_id', 'gender', 'age', 'salary', 'province'])


def gen_music_meta(nrows=None):
    df_music_meta = pd.read_csv(music_meta,
                                sep='\001',
                                nrows=nrows,
                                names=['item_id', 'item_name', 'desc', 'total_timelen', 'location', 'tags'])
    del df_music_meta['desc']
    return df_music_meta.fillna('-')


# 写入文件工具方法
def kismet_write(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(data))

# 读取文件的工具方法
def kismet_read(path):
    with open(path,'r',encoding='utf-8') as f:
        return eval(f.read())
