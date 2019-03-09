import os
import random

'''
朴素贝叶斯分类总体实现过程:
总体思路:
    1,train:yi的分类概率;word在yi的概率
    2,predict:获取文章切词后的words在不同yi下的概率并乘上yi的先验概率,取概率最大值
    
技巧:将分类和单词都进行编码成数字

data:分类的文章:文章内容-->classId:context

流程:
    1,数据处理;
        1>,文件名-->分类类型-->分类字典
        2>,单词-->编码成数字
        3>,测试集和训练集划分
    2,通过频次计算出yi和word在yi下的概率
    3,预测
    4,评估

'''

'''
将切分好的数据进行编码:
    1,文件名-->分类类型-->分类字典
    2,单词-->编码成数字
    3,测试集和训练集划分
'''

file_path = './raw_data'
# 训练集和测试集的输出文件路径
TrainOutFilePath = './mid_data/data.train'
TestOutFilePath = './mid_data/data.test'
train_file = open(TrainOutFilePath, 'w', encoding='utf-8')
test_file = open(TestOutFilePath, 'w', encoding='utf-8')

TrainingPercent = 0.8  # 划分数据集的概率，0.8为训练，0.2为test

# 分类字典
label_dict = {'business': 0, 'yule': 1, 'it': 2, 'sports': 3, 'auto': 4}

# 单词:index
word_index = {}
# 单词去重集和
word_list = []


def convert_data():
    # 存储分类信息
    file_sum = 0  # 文章总数
    tag = 0  # 文章类型

    for filename in os.listdir(file_path):
        if filename.find('business') != -1:
            tag = label_dict['business']
        elif filename.find('yule') != -1:
            tag = label_dict['yule']
        elif filename.find('it') != -1:
            tag = label_dict['it']
        elif filename.find('sports') != -1:
            tag = label_dict['sports']
        else:
            tag = label_dict['auto']

        file_sum += 1

        # 划分训练和测试集
        rd = random.random()

        outfile = train_file
        if rd > TrainingPercent:
            outfile = test_file

        # 1,写入标签信息
        outfile.write(str(tag) + ' ')

        # 2,写入文本信息
        infile = open(os.path.join(file_path, filename), 'r', encoding='utf-8')
        words = infile.read().strip().replace('\n', ' ').split(' ')
        for word in words:
            if len(word.strip()) < 1:
                continue
            if word_index.get(word, -1) == -1:
                word_list.append(word)
                word_index[word] = len(word_list)
            outfile.write(str(word_index[word]) + ' ')

        # 3,写入标题信息并换行
        outfile.write('#' + filename + '\n')
        infile.close()

        print(file_sum, 'files load')
        print(len(word_list), 'unique word found')


if __name__ == '__main__':
    convert_data()
    test_file.close()
    train_file.close()
