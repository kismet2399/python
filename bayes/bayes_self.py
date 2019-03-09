import math

'''
训练，测试数据路径
'''
train_out_file = './mid_data/data.train'
test_out_file = './mid_data/data.test'

'''
模型存储路径
'''
Modelfile = './mid_data//bayes.Model'

'''
初始化参数
'''
DefautFreq = 0.1  # 平滑处理的默认频次
ClassDefaultProb = {}  # 类别的默认概率
ClassFeatDic = dict()  # xj和yi的矩阵 count
ClassFeatProb = dict()  # 概率
ClassFreq = dict()  # yi的数组 count
ClassProb = dict()  # yi的概率

WordDic = dict()


def load_data():
    '''
    加载数据
    :return:
    '''
    train_file = open(train_out_file, 'r', encoding='utf-8')
    sline = train_file.readline().strip()
    # 文章总数
    file_sum = 0
    while len(sline) > 0:
        # 截取文件名之前的数据
        pos = sline.find("#")
        if pos > 0:
            sline = sline[:pos]
        words = sline.strip().split(' ')
        if len(words) < 1:
            continue
        # 获取类别标识
        classid = int(words[0])
        if classid not in ClassFreq:
            ClassFeatDic[classid] = {}
            ClassFreq[classid] = 0
        ClassFreq[classid] += 1;
        file_sum += 1

        for word in words[1:]:
            if len(word) < 1:
                continue

            # word编码
            wid = int(word)
            if wid not in WordDic: WordDic[wid] = 1

            if wid not in ClassFeatDic[classid]:
                ClassFeatDic[classid][wid] = 0
            ClassFeatDic[classid][wid] += 1

        sline = train_file.readline().strip()

    train_file.close()
    print(file_sum, 'items loaded')
    print(len(ClassFreq), 'class!  ', len(WordDic), 'words! ')


def compute_model():
    '''
    将频次数据处理成概率,并增加平滑处理
    :return:
    '''
    # 1,计算文章总数-->计算分类概率
    file_sum = 0
    for feq in ClassFreq.values():
        file_sum += feq
    for classid, feq in ClassFreq.items():
        ClassProb[classid] = float(feq) / file_sum

    # 2,计算word的分类概率矩阵
    for classid, word_feq in ClassFeatDic.items():
        sum = 0  # 该类别文章单词总数
        for feq in word_feq.values():
            sum += feq
        # 平滑处理
        new_sum = float(sum + len(WordDic) * DefautFreq)

        ClassFeatProb[classid] = {}
        for word, feq in word_feq.items():
            ClassFeatProb[classid][word] = float(feq + DefautFreq) / new_sum
        # 该分类默认概率
        ClassDefaultProb[classid] = DefautFreq / new_sum


def save_model():
    out_file = open(Modelfile, 'w', encoding='utf-8')
    # 1,保存分类概率和单词在该分类的默认概率
    for classid, prob in ClassProb.items():
        out_file.write(str(classid) + ' ' + str(prob) + ' ' + str(ClassDefaultProb[classid]) + ' ')
    out_file.write('\n')
    # 2,保存分类下单词概率
    for classid, word_prob in ClassFeatProb.items():
        out_file.write(str(classid) + ' ')
        for word, prob in word_prob.items():
            out_file.write(str(word) + ' ' + str(prob) + ' ')
        out_file.write('\n')
    out_file.close()


def load_model():
    global WordDic
    WordDic = {}
    global ClassFeatProb
    ClassFeatProb = {}  # p(xj|yi)的矩阵
    global ClassDefaultProb
    ClassDefaultProb = {}  # 默认概率
    global ClassProb
    ClassProb = {}  # p(yi)

    infile = open(Modelfile, 'r', encoding='utf-8')
    # 1,加载分类概率数据
    sline = infile.readline().strip()
    items = sline.split(' ')
    if len(items) % 3 != 0:
        print('Model load error')
        return
    i = 0
    while i < len(items):
        classid = int(items[i])
        i += 1
        ClassProb[classid] = float(items[i])
        i += 1
        ClassDefaultProb[classid] = float(items[i])
        i += 1

    # 2,加载单词的分类概率数据
    sline = infile.readline().strip()
    while len(sline) > 0:
        items = sline.split(' ')
        classid = int(items[0])
        ClassFeatProb[classid] = {}
        i = 1
        while i < len(items):
            word = int(items[i])
            if word not in WordDic:
                WordDic[word] = 1
            i += 1
            ClassFeatProb[classid][word] = float(items[i])
            i += 1
        sline = infile.readline().strip()

    infile.close()
    print(len(ClassProb), 'classes! ', len(WordDic), "words!")


def pridict():
    global WordDic
    global ClassFeatProb
    global ClassDefaultProb
    global ClassProb

    true_label_list = []
    pred_label_list = []
    score_dict = {}
    # 加载测试数据
    infile = open(test_out_file, 'r', encoding='utf-8')
    sline = infile.readline().strip()
    while len(sline) > 0:
        pos = sline.find("#")
        if pos > 0:
            sline = sline[:pos]
        words = sline.strip().split(' ')
        # 记录实际分类
        true_label_list.append(int(words[0]))
        # 进行预测
        # 获取先验概率
        for classid, prob in ClassProb.items():
            score_dict[classid] = math.log(prob)
        # 计算不同分类的概率
        for classid in score_dict.keys():
            for word in words[1:]:
                if len(word) < 1:
                    continue
                wid = int(word)
                if wid not in WordDic:
                    continue
                if wid not in ClassFeatProb[classid]:
                    score_dict[classid] += math.log(ClassDefaultProb[classid])
                else:
                    score_dict[classid] += math.log(ClassFeatProb[classid][wid])
        max_prob = max(score_dict.values())
        for classid, prob in score_dict.items():
            if max_prob == prob:
                pred_label_list.append(classid)
        sline = infile.readline().strip()
    infile.close()
    print(len(true_label_list), len(pred_label_list))
    return true_label_list, pred_label_list


def evaluate(true_list, pred_list):
    i = 0
    for j in range(0, len(true_list)):
        if true_list[j] == pred_list[j]:
            i += 1

    accuracy = float(i) / float(len(true_list))
    print('Accuracy: ', accuracy)
    return accuracy


if __name__ == '__main__':
    load_data()
    compute_model()
    save_model()
    # load_model()
    true_label_list, pred_label_list = pridict()
    evaluate(true_label_list, pred_label_list)
