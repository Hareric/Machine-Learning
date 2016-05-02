# coding=utf-8
#        ┏┓　　　┏┓+ +
# 　　　┏┛┻━━━┛┻┓ + +
# 　　　┃　　　　　　 ┃ 　
# 　　　┃　　　━　　　┃ ++ + + +
# 　　 ████━████ ┃+
# 　　　┃　　　　　　 ┃ +
# 　　　┃　　　┻　　　┃
# 　　　┃　　　　　　 ┃ + +
# 　　　┗━┓　　　┏━┛
# 　　　　　┃　　　┃　　　　　　　　　　　
# 　　　　　┃　　　┃ + + + +
# 　　　　　┃　　　┃　　　　Codes are far away from bugs with the animal protecting　　　
# 　　　　　┃　　　┃ + 　　　　神兽保佑,代码无bug　　
# 　　　　　┃　　　┃
# 　　　　　┃　　　┃　　+　　　　　　　　　
# 　　　　　┃　 　　┗━━━┓ + +
# 　　　　　┃ 　　　　　　　┣┓
# 　　　　　┃ 　　　　　　　┏┛
# 　　　　　┗┓┓┏━┳┓┏┛ + + + +
# 　　　　　　┃┫┫　┃┫┫
# 　　　　　　┗┻┛　┗┻┛+ + + +
"""
Author = Eric_Chan
Create_Time = 2016/04/28
k-近邻算法简单实例
"""

import numpy as np
import operator

# 训练集
data_set = np.array([[1., 1.1],
                     [1.0, 1.0],
                     [0., 0.],
                     [0, 0.1]])
labels = ['A', 'A', 'B', 'B']


def classify_knn(in_vector, training_data, training_label, k):
    """
    :param in_vector: 待分类向量
    :param training_data: 训练集向量
    :param training_label: 训练集标签
    :param k: 选择最近邻居的数目
    :return: 分类器对 in_vector 分类的类别
    """
    data_size = training_data.shape[0]  # .shape[0] 返回矩阵的行数
    diff_mat = np.tile(in_vector, (data_size, 1)) - data_set  # np.tile(array, (3, 2)) 对 array 进行 3×2 扩展为二维数组
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)  # .sum(axis=1) 矩阵以列求和
    # distances = sq_distances ** 0.5  # 主要是为了求最近点的个数,所以没有必要对全部值进行求平方根
    distances_sorted_index = sq_distances.argsort()  # .argsort() 对array进行排序 返回排序后对应的索引
    class_count_dict = {}  # 用于统计类别的个数
    for i in range(k):
        label = training_label[distances_sorted_index[i]]
        try:
            class_count_dict[label] += 1
        except KeyError:
            class_count_dict[label] = 1
    class_count_dict = sorted(class_count_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return class_count_dict[0][0]


if __name__ == '__main__':
    vector = [0, 0]
    print classify_knn(in_vector=vector, training_data=data_set, training_label=labels, k=3)
