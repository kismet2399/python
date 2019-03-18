import jieba

seg_list = jieba.cut("他来到了网易杭研大厦", HMM=True)
print("/ ".join(seg_list))
seg_list = jieba.cut("他来到了网易杭研大厦", HMM=False)
print("/ ".join(seg_list))
seg_list = jieba.cut("我去过清华大学和北京大学。", HMM=True)
print("/ ".join(seg_list))
seg_list = jieba.cut("我去过清华大学和北京大学。",HMM=False)
print("/ ".join(seg_list))
seg_list = jieba.cut("杭研大厦",HMM=False)
print("/ ".join(seg_list))
