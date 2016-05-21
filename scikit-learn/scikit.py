# coding=utf-8
import sklearn
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import time

if __name__ == '__main__':
    feature_matrix = np.load('dataSet/feature_matrix_save.npy')
    class_list = np.load('dataSet/class_result_save.npy')

    model = ExtraTreesClassifier()
    model.fit(feature_matrix, class_list)
    # display the relative importance of each attribute
    # print model.
    # model = GaussianNB()
    # model.fit(feature_matrix, class_list)
    # print(model)
    # # make predictions
    # expected = class_list
    # predicted = model.predict(feature_matrix)
    # # summarize the fit of the model
    # print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))




    tfidf_transformer = TfidfTransformer()
    feature_matrix = tfidf_transformer.fit_transform(feature_matrix)

    #
    # model = GaussianNB()
    # model.fit(feature_matrix, class_list)
    # print(model)
    # # make predictions
    # expected = class_list
    # predicted = model.predict(feature_matrix)
    # # summarize the fit of the model
    # print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB().fit(feature_matrix, class_list)
    print clf