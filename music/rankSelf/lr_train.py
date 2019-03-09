import music.recallSelf.gen_cf_data as gcd
import music.recallSelf.config as conf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

'''
训练模型
'''
cross_file = conf.cross_file
user_feat_map_file = conf.user_feat_map_file
model_file = conf.model_file

data = gcd.user_item_score(50000)
# 处理标签
data['label'] = data['score'].apply(lambda x: 1 if x > 1.0 else 0)
# 获取宽表信息包括用户特征和商品特征
data = data.merge(
    conf.gen_user_profile(), how='inner', on='user_id').merge(
    conf.gen_music_meta(), how='inner', on='item_id')

'''
特征处理
'''
user_feat = ['gender', 'age', 'salary', 'province']
item_feat = ['total_timelen', 'location']
# 离散特征
category_feat = user_feat + ['location']
# 连续特征
continuous_feat = ['score']

# 标签特征
labels = data['label']
del data['label']

# 特征处理
# 1,离散特征
# 1. 离散特征one-hot处理 （word2vec-> embedding[continuous]）
# ['gender','age','salary','province','location']
# 处理离散特征
df = pd.get_dummies(data[category_feat])
one_hot_columns = df.columns
# print(one_hot_columns)
# 2.连续特征不处理直接带入  【一般做离散化GBDT（xgboost）叶子节点做离散化 GBDT+LR】
# ['gender','age','salary','province','location','score']
df[continuous_feat] = data[continuous_feat].astype(float)
# 3.交叉特征(处理)
data['ui-key'] = data['user_id'].astype(str) + '_' + data['item_id'].astype(str)
cross_feat_map = dict()
for _, row in data[['ui-key', 'score']].iterrows():
    cross_feat_map[row['ui-key']] = row['score']
conf.kismet_write(cross_feat_map, cross_file)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(df.values, labels, test_size=0.3, random_state=2399)
# 定义逻辑回归模型
lr = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=0.1,
                        fit_intercept=True, intercept_scaling=1, class_weight=None,
                        random_state=None, solver='liblinear', max_iter=100,
                        multi_class='ovr', verbose=1, warm_start=False, n_jobs=-1)

model = lr.fit(X_train, y_train)
print("w:%s, b:%s" % (lr.coef_, lr.intercept_))
'''
模型验证
'''
# 1 误差平方和均方跟
print("Residual sum of square: %.2f" % np.mean((lr.predict(X_test) - y_test) ** 2))
# 2 准确率
print("准确率: %.2f" % lr.score(X_test, y_test))

'''
存储模型数据
'''
# 存储离散特征
feat_map = {}
for i in range(len(one_hot_columns)):
    feat_map[one_hot_columns[i]] = i

conf.kismet_write(feat_map, user_feat_map_file)

# 存储模型权重
model_weight = {'w': lr.coef_.tolist()[0], 'b': lr.intercept_.tolist()[0]}

conf.kismet_write(model_weight, model_file)

