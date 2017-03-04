# coding=utf-8
from GBDT import pandas_read
from sklearn.externals import joblib
import pandas as pd
import os
import numpy as np


def load_file(file_name, charset='utf-8'):
    """
    读取文件，按列返回列表
    :param file_name: 文件路径
    :param charset: 文本内容decode的编码，默认为utf-8
    :return: 文本内容列表
    """
    f1 = open(file_name)
    line = f1.readline().decode(charset).strip()
    line_list = []
    while line:
        line_list.append(line)
        line = f1.readline().decode(charset).strip()
    return line_list


def write_file(file_name, line_list, charset='utf-8', mode='w'):
    """
    新建文件将line_list中每个元素按行写入
    :param mode: 打开文件的规格， 'w' 表示新建， 'a' 表示添加， 'r' 表示只读
    :param file_name: 新建文件的文件名和路径
    :param line_list: 写入文件的列表
    :param charset: 写入文件是encode的编码， 默认为utf-8
    :return: void
    """
    f1 = open(file_name, mode=mode)
    for line in line_list:
        line = line.encode(charset)
        f1.write(line + '\n')
    f1.flush()
    f1.close()


class RoadProcess:
    user_info = pd.DataFrame()
    gbdt = None

    def road_filter(self, name_list):
        """
        :param name_list:
        :return: is_save
        """
        if name_list.__len__() < 3:
            return False
        try:
            infos = self.user_info.loc[name_list]
        except KeyError:  # 用户数据不存在该用户 默认不保存该路径
            print 'KeyError'
            return False
        infos = infos.dropna()
        try:
            predict = self.gbdt.predict(infos)
            num = predict.sum()
        except ValueError:  # 用户数据有误 默认不保存该路径
            print "存在有误数据"
            print name_list
            print infos
            print '-----------------'
            return False
        if num >= 3:
            # print infos
            # print predict, num
            print '~'.join(name_list)
            # print '--------------------'
            return True
        else:
            return False

    def file_pro(self, line_list):
        """

        :param line_list:
        :return: new line list
        """
        new_line_list = []
        for line in line_list:
            name_list = line.split(',')
            if self.road_filter(name_list):

                new_line_list.append('~'.join(name_list))
        return new_line_list

    def run(self, path):
        file_dirs = os.walk(path)
        for root, dirs, files in file_dirs:
            for file in files:
                if file == 'path.txt':
                    lines = load_file(root+'/'+file)
                    new_lines = self.file_pro(lines)
                    write_file(root+'/'+'path_new.txt', new_lines)
                    print root, 'Finish!\n*******************************************'

    def __init__(self):
        column_list = ['用户ID', '用户名', '关注数', '粉丝数', '博客数', '是否大v', '首页转发数', '首页评论数', '首页点赞数']
        self.user_info = pandas_read('dataSet/path/转发用户/林丹出轨转发用户ID 关注数 '
                                     '粉丝数 博客数 是否大v 首页转发数 首页评论数 首页点赞数.txt', column_list, '用户名', encoding='utf-8')
        self.user_info = self.user_info.append(pandas_read('dataSet/path/转发用户/罗一笑转发用户 ID 关注数 '
                                                           '粉丝数 博客数 是否大v 首页转发数 首页评论数 首页点赞数.txt', column_list, '用户名', encoding='utf-8'))
        self.user_info = self.user_info.append(pandas_read('dataSet/path/转发用户/裸贷转发用户 ID 关注数 '
                                                           '粉丝数 博客数 是否大v 首页转发数 首页评论数 首页点赞数.txt', column_list, '用户名', encoding='utf-8'))
        self.user_info.drop('用户ID', axis=1, inplace=True)
        # 消除重复用户
        self.user_info = self.user_info.reset_index().drop_duplicates(subset='用户名', keep='last').set_index('用户名')
        self.gbdt = joblib.load('gbdt.pkl')


if __name__ == '__main__':
    a = RoadProcess()
    a.run('dataSet/path')
