mod_path = '../data/model_file.txt'

STATUS_NUM = 4
# 1. 加载模型参数
mod_file = open(mod_path, 'r', encoding='utf-8')

pi = [0.0 for x in range(STATUS_NUM)]
A = [[0.0 for x in range(STATUS_NUM)] for x in range(STATUS_NUM)]
B = [{} for x in range(STATUS_NUM)]
# 1.1 初始概率
pi_tokens = mod_file.readline().split()
for i in range(STATUS_NUM):
    pi[i] = float(pi_tokens[i])

# 1.2 转移概率
for i in range(STATUS_NUM):
    A_tokens = mod_file.readline().split()
    for j in range(STATUS_NUM):
        A[i][j] = float(A_tokens[j])

# 1.3 发射概率
for i in range(STATUS_NUM):
    B_tokens = mod_file.readline().split()
    for j in range(0, len(B_tokens), 2):
        B[i][B_tokens[j]] = float(B_tokens[j + 1])


# print(pi)
# print(A)
# print(B)

# 2. 实现viterbi算法（预测最优路径）
def hmm_seg_func(ch=''):
    ch_num = len(ch)
    if ch_num == 0: return
    # 初始化动态模型
    status_matrix = [[[0.0, 0] for x in range(ch_num)] for x in range(STATUS_NUM)]

    # 初始概率
    for x in range(STATUS_NUM):
        # 当发射概率不存在时
        if ch[i] in B[i]:
            cur_p = B[i][ch[0]]
        else:
            cur_p = -1000000.0
        status_matrix[i][0][0] = -1000000.0 if pi[i] * cur_p == 0 else pi[i] * cur_p
        status_matrix[i][0][1] = i

    #其他情况
