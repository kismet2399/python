import music.recall.config as conf

# 获取用户特征
user_feat_map_file = conf.user_feat_map_file

user_id = '016d5ef3210ad19b75ed5296f01b5090'
with open(user_feat_map_file, 'r', encoding='utf-8') as f:
    user_feat_map = eval(f.read())

vector_len = len(user_feat_map)


# 将一条样本处理成一个向量
def gen_user_item_vector(user_id, vector_len=vector_len):
    user_profile = conf.gen_user_profile()
    user_one_row = user_profile.loc[user_profile["user_id"] == user_id, :]
    user_feat_cols = user_one_row.columns[1:]
    feat_vector = [0.0 for i in range(vector_len)]
    for _, row in user_one_row.iterrows():
        for col_name in user_feat_cols:
            key = col_name + '_' + row[col_name]
            index = user_feat_map[key]
            feat_vector[index] += 1
    return feat_vector


# user_profile = conf.gen_user_profile()
# print(user_profile.loc[user_profile['user_id'] == user_id, :])
# print(gen_user_item_vector(user_id))
