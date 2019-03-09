import math


data_set = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
cols = ['no surfacing', 'flippers']

i=-1
print([example[i] for example in data_set])


# n = len(data_set)
# label_cnt = {}
# for featVec in data_set:
#     cur_label = featVec[-1]
#     if cur_label not in label_cnt.keys():
#         label_cnt[cur_label] = 0
#     label_cnt[cur_label] += 1
#
# print('cur_label:',cur_label)
# print('label_cnt:',label_cnt)
#
# E = 0.0
# for key in label_cnt:
#     prob = float(label_cnt[key]) / n
#     E -= prob * math.log(prob)
# print('math.log(10):',math.log(10,10))
#
# i = 2
# a = [1,2,3,2]
# lst = a[:i]
# lst.extend(a[i+1:])
# print('lst:', lst)  # lst: [1, 3]
# a.remove(i)
# print(a)
# print(list(filter(lambda x: x !=2, a)))








