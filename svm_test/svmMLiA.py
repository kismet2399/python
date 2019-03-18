from numpy import *
from time import sleep


def load_data_set(fileName):
    data_mat = []
    label_mat = []
    fr = open(fileName)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


# 选择i不等于j的a
def select_J_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo_simple(data_mat_in, class_labels, C, toler, max_iter):
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    b = 0
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    iter = 0   # 计数器
    # 每个a的更新都要小于max Iter
    while iter < max_iter:
        # 标识alpha有没有更新
        alpha_pairs_changed = 0
        # 扫训练集
        for i in range(m):
            # 模型的函数，已经对w，b求导了，是将w带入模型函数中的 fxi = sum(ai*yi*xi^T*x+b)
            # multiply(alphas, label_mat).T （100*100）
            fXi = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            # 预测值-真实值=Err误差
            Ei = fXi - float(label_mat[i])  # if checks if an example violates KKT conditions
            # 当错误率大于一定阈值，而且当前alpha在(0,C)
            if ((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or ((label_mat[i] * Ei > toler) and (alphas[i] > 0)):
                # 选择与i不一样的j，作为aj
                j = select_J_rand(i, m)
                # 套模型函数，计算aj的预测值
                fXj = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                # 计算aj的误差
                Ej = fXj - float(label_mat[j])
                # 保存当前的ai和aj
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                # 限定aj的定义域，分类讨论的H,L的值获取
                if label_mat[i] != label_mat[j]:  # yi！=yj
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:  # yi=yj
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L==H"); continue
                # eta 统计学习方法p127-p128 推到过程 ，公式-eta
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - \
                      data_matrix[i, :] * data_matrix[i, :].T - \
                      data_matrix[j, :] * data_matrix[j, :].T

                if eta >= 0:
                    print("eta>=0")
                    continue
                # 因为是-eta，所以aj的更新-=，不管aj的定义域，先求aj-=yj*（err_i=err_j）/eta
                alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                # aj在限定的定义域中
                alphas[j] = clip_alpha(alphas[j], H, L)
                # aj更新不在变化了
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("j not moving enough")
                    continue
                # 综上都在更新aj，当aj不能再更新的时候，更新ai。-yj*yi（aj_new-aj_old）
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])  # update i by the same amount as j
                # 将样本（xi，yi）带入模型函数得到的
                b1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


def kernel_trans(X, A, k_tup):  # calc the kernel or transform data to a higher dimensional space
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if k_tup[0] == 'lin':
        K = X * A.T  # linear kernel
    elif k_tup[0] == 'rbf':
        for j in range(m):
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        K = exp(K / (-1 * k_tup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


class Optstruct:
    def __init__(self, data_mat_in, class_labels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = data_mat_in
        self.labelMat = class_labels
        self.C = C
        self.tol = toler
        self.m = shape(data_mat_in)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], kTup)


def calc_ek(o_s, k):
    fXk = float(multiply(o_s.alphas, o_s.labelMat).T * o_s.K[:, k] + o_s.b)
    Ek = fXk - float(o_s.labelMat[k])
    return Ek


def select_j(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    max_k = -1;
    max_delta_e = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    valid_ecache_list = nonzero(oS.eCache[:, 0].A)[0]
    if len(valid_ecache_list) > 1:
        for k in valid_ecache_list:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i:
                continue  # don't calc for i, waste of time
            Ek = calc_ek(oS, k)
            delta_e = abs(Ei - Ek)
            if delta_e > max_delta_e:
                max_k = k;
                max_delta_e = delta_e;
                Ej = Ek
        return max_k, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = select_J_rand(i, oS.m)
        Ej = calc_ek(oS, j)
    return j, Ej


def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calc_ek(oS, k)
    oS.eCache[k] = [1, Ek]


# 更新ai,aj
def innerL(i, o_s):
    Ei = calc_ek(o_s, i)
    if ((o_s.labelMat[i] * Ei < -o_s.tol) and (o_s.alphas[i] < o_s.C)) or (
                (o_s.labelMat[i] * Ei > o_s.tol) and (o_s.alphas[i] > 0)):
        j, Ej = select_j(i, o_s, Ei)  # this has been changed from selectJrand
        alpha_i_old = o_s.alphas[i].copy();
        alpha_j_old = o_s.alphas[j].copy();
        if o_s.labelMat[i] != o_s.labelMat[j]:
            L = max(0, o_s.alphas[j] - o_s.alphas[i])
            H = min(o_s.C, o_s.C + o_s.alphas[j] - o_s.alphas[i])
        else:
            L = max(0, o_s.alphas[j] + o_s.alphas[i] - o_s.C)
            H = min(o_s.C, o_s.alphas[j] + o_s.alphas[i])
        if L == H:
            print("L==H");
            return 0
        eta = 2.0 * o_s.K[i, j] - o_s.K[i, i] - o_s.K[j, j]  # changed for kernel
        if eta >= 0:
            print("eta>=0");
            return 0
        o_s.alphas[j] -= o_s.labelMat[j] * (Ei - Ej) / eta
        o_s.alphas[j] = clip_alpha(o_s.alphas[j], H, L)
        updateEk(o_s, j)  # added this for the Ecache
        if abs(o_s.alphas[j] - alpha_j_old) < 0.00001:
            print("j not moving enough");
            return 0
        o_s.alphas[i] += o_s.labelMat[j] * o_s.labelMat[i] * (
        alpha_j_old - o_s.alphas[j])  # update i by the same amount as j
        updateEk(o_s, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = o_s.b - Ei - o_s.labelMat[i] * (o_s.alphas[i] - alpha_i_old) * o_s.K[i, i] - o_s.labelMat[j] * (
            o_s.alphas[j] - alpha_j_old) * o_s.K[i, j]
        b2 = o_s.b - Ej - o_s.labelMat[i] * (o_s.alphas[i] - alpha_i_old) * o_s.K[i, j] - o_s.labelMat[j] * (
            o_s.alphas[j] - alpha_j_old) * o_s.K[j, j]
        if (0 < o_s.alphas[i]) and (o_s.C > o_s.alphas[i]):
            o_s.b = b1
        elif (0 < o_s.alphas[j]) and (o_s.C > o_s.alphas[j]):
            o_s.b = b2
        else:
            o_s.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(data_mat_in, class_labels, C, toler, max_iter, k_tup=('lin', 0)):  # full Platt SMO
    o_s = Optstruct(mat(data_mat_in), mat(class_labels).transpose(), C, toler, k_tup)
    iter = 0
    entire_set = True;
    alpha_pairs_changed = 0
    while (iter < max_iter) and ((alpha_pairs_changed > 0) or (entire_set)):
        alpha_pairs_changed = 0
        if entire_set:  # go over all
            for i in range(o_s.m):
                alpha_pairs_changed += innerL(i, o_s)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
            iter += 1
        else:  # go over non-bound (railed) alphas
            non_bound_is = nonzero((o_s.alphas.A > 0) * (o_s.alphas.A < C))[0]
            for i in non_bound_is:
                alpha_pairs_changed += innerL(i, o_s)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
            iter += 1
        if entire_set:
            entire_set = False  # toggle entire set loop
        elif alpha_pairs_changed == 0:
            entire_set = True
        print("iteration number: %d" % iter)
    return o_s.b, o_s.alphas


def calcWs(alphas, data_arr, class_labels):
    X = mat(data_arr);
    label_mat = mat(class_labels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_mat[i], X[i, :].T)
    return w


def test_rbf(k1=1.3):
    data_arr, label_arr = load_data_set('testSetRBF.txt')
    b, alphas = smoP(data_arr, label_arr, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    dat_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    sv_ind = nonzero(alphas.A > 0)[0]
    sVs = dat_mat[sv_ind]  # get matrix of only support vectors
    label_sv = label_mat[sv_ind]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(dat_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(sVs, dat_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * multiply(label_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]):
            error_count += 1
    print("the training error rate is: %f" % (float(error_count) / m))
    data_arr, label_arr = load_data_set('testSetRBF2.txt')
    error_count = 0
    dat_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    m, n = shape(dat_mat)
    for i in range(m):
        kernel_eval = kernel_trans(sVs, dat_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * multiply(label_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]): error_count += 1
    print("the test error rate is: %f" % (float(error_count) / m))


def img2vector(filename):
    return_vect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def loadImages(dirName):
    from os import listdir
    hw_labels = []
    training_file_list = listdir(dirName)  # load the training set
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]  # take off .txt
        class_num_str = int(file_str.split('_')[0])
        if class_num_str == 9:
            hw_labels.append(-1)
        else:
            hw_labels.append(1)
        training_mat[i, :] = img2vector('%s/%s' % (dirName, file_name_str))
    return training_mat, hw_labels


def testDigits(k_tup=('rbf', 10)):
    data_arr, label_arr = loadImages('trainingDigits')
    b, alphas = smoP(data_arr, label_arr, 200, 0.0001, 10000, k_tup)
    dat_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    sv_ind = nonzero(alphas.A > 0)[0]
    sVs = dat_mat[sv_ind]
    labelSV = label_mat[sv_ind]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(dat_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(sVs, dat_mat[i, :], k_tup)
        predict = kernel_eval.T * multiply(labelSV, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]): error_count += 1
    print("the training error rate is: %f" % (float(error_count) / m))
    data_arr, label_arr = loadImages('testDigits')
    error_count = 0
    dat_mat = mat(data_arr);
    label_mat = mat(label_arr).transpose()
    m, n = shape(dat_mat)
    for i in range(m):
        kernel_eval = kernel_trans(sVs, dat_mat[i, :], k_tup)
        predict = kernel_eval.T * multiply(labelSV, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]): error_count += 1
    print("the test error rate is: %f" % (float(error_count) / m))
