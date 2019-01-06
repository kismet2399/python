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
    # 2.1 填充动态概率矩阵
    # 初始概率
    for i in range(STATUS_NUM):
        # 当发射概率不存在时
        if ch[i] in B[i]:
            cur_p = B[i][ch[0]]
        else:
            cur_p = -1000000.0
        if pi[i] == 0.0:
            cur_pi = -1000000.0
        else:
            cur_pi = pi[i]
        status_matrix[i][0][0] = cur_pi + cur_p
        status_matrix[i][0][1] = i

    # 其他情况
    for index in range(1, ch_num):
        # 计算最大概率并存储,其中i为下一层,j为上一层
        cur_ch = ch[index]
        for i in range(STATUS_NUM):
            cur_max_p = None
            cur_max_status = None
            for j in range(STATUS_NUM):
                cur_A = A[j][i] if A[j][i] != 0 else -1000000.0
                if ch[index] in B[i]:
                    cur_B = B[i][ch[index]]
                else:
                    cur_B = -1000000.0
                cur_p = cur_A + cur_B + status_matrix[j][index-1][0]
                if cur_max_p is None or cur_max_p < cur_p:
                    cur_max_p = cur_p
                    cur_max_status = j
            status_matrix[i][index][0] = cur_max_p
            status_matrix[i][index][1] = cur_max_status

    # 2.2 获取最大概率序列
    # 最后一列
    status_list = [0 for x in range(ch_num)]

    cur_max_p = None
    cur_max_status = None
    for i in range(STATUS_NUM):
        if cur_max_p is None or cur_max_p < status_matrix[i][-1][0]:
            cur_max_p = status_matrix[i][-1][0]
            cur_max_status = i
    status_list[-1] = cur_max_status

    # 往前递推
    for i in range(ch_num - 2, -1, -1):
        cur_status = status_matrix[cur_max_status][i + 1][1]
        status_list[i] = cur_status
        cur_max_status = cur_status

    d = {'0':'B','1':'M','2':'E','3':'S'}
    print(ch)
    print([d.get(str(i)) for i in status_list])

hmm_seg_func("我在八斗学习大数据和机器学习")
