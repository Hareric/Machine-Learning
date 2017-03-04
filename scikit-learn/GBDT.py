# coding=utf-8
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.externals import joblib


def pandas_read(file_name, column_list, index_column=None, split_with='\t', encoding=None):
    """
        使用panda.read_csv读取数据
    :param file_name: 文件路径
    :param column_list: 属性列表名
    :param index_column: 作为index的属性名,若为空则使用增序索引
    :param split_with:
    :return: DataFrame
    """
    data_frame = pd.DataFrame(pd.read_csv(file_name, header=None, sep=split_with, encoding=encoding))
    data_frame.columns = column_list
    if index_column is not None:
        # 构建索引
        data_frame.index = data_frame[index_column]
        # 删除构建索引所使用的列数据
        data_frame.drop(index_column, axis=1, inplace=True)
        # 删除重复索引的数据
        data_frame = data_frame.reset_index().drop_duplicates(subset=index_column, keep='last').set_index(index_column)
    return data_frame.sort_index()


if __name__ == '__main__':
    column_list = ['用户名', '关注的人数', '粉丝人数', '发布的微博数',
                   '是否微博认证', '首页转发数', '首页评论数', '首页点赞数']
    # 训练集
    train_X = pandas_read('dataSet/微博用户样本数据/UserInfoI-r-00000.txt', column_list,
                          index_column='用户名', encoding='utf-8')
    train_y_0 = pandas_read('dataSet/微博用户样本数据/0.txt', ['用户名'], index_column='用户名', encoding='GBK')
    train_y_0['是否为重要用户'] = 0
    train_y_1 = pandas_read('dataSet/微博用户样本数据/1.txt', ['用户名'], index_column='用户名', encoding='GBK')
    train_y_1['是否为重要用户'] = 1
    train_y = train_y_0.append(train_y_1)
    # 获取X和y都存在的有效用户数据
    common_user = set(train_y.index.values).intersection(train_X.index.values)
    train_X = train_X.loc[common_user]
    train_y = train_y.loc[common_user]
    print train_X.head(5)
    print train_y.head(5)

    # 测试集
    test_X = pandas_read('dataSet/微博用户样本数据/testX.txt', column_list,
                         index_column='用户名', encoding='utf-8')
    test_y = pandas_read('dataSet/微博用户样本数据/testy.txt', ['用户名', '是否为重要用户'],
                         index_column='用户名', encoding='utf-8')

    # GBDT训练及测试
    gbdt = GradientBoostingClassifier()
    gbdt.fit(train_X, train_y)
    predicted = gbdt.predict(test_X)
    print metrics.classification_report(test_y, predicted)

    # 保存模型
    joblib.dump(gbdt, 'gbdt.pkl')