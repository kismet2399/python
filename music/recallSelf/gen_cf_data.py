from music.recallSelf import config

train_data_path = config.cf_train_data_path


def user_item_score(action_num=100):
    '''
    将数据处理成 'user_id','item_id','rating'
    :param action_num: 处理的条数
    :return: 'user_id','item_id','rating' 的dataFrame
    '''
    user_watch = config.gen_user_watch(action_num)
    music_meta = config.gen_music_meta()
    df = user_watch.merge(music_meta, how='inner', on='item_id')
    del user_watch
    del music_meta
    # apply相当于spark rdd map操作  score = 200s/ 304s
    # axis为0表示对每一列进行操作,为1表示对每一行进行操作
    df['score'] = df.apply(lambda x: float(x['stay_seconds']) / float( x['total_timelen']), axis=1)
    data = df[['user_id', 'item_id', 'score']]
    # 处理多条数据的方式,通过累加(处理方式也可以是平均)
    data = data.groupby(['user_id', 'item_id']).score.sum().reset_index()
    return data


def train_from_df(df, col_name=['user_id', 'item_id', 'score']):
    '''
    将DataFrame转化成dict
    :param df:
    :param col_name:
    :return: 最终dict数据
    '''
    d = dict()
    for _, row in df.iterrows():
        user_id = str(row[col_name[0]])
        item_id = str(row[col_name[1]])
        score = row[col_name[2]]
        if user_id not in d.keys():
            d[user_id] = {item_id: score}
        else:
            d[user_id][item_id] = score

    return d

# main方法
if __name__=='__main__':
    data = user_item_score(50000)
    d = train_from_df(data, col_name=['user_id', 'item_id', 'score'])
    config.kismet_write(d,train_data_path)