# -*- coding:utf-8 -*-

from sklearn.datasets import load_iris
from sklearn import cross_validation
from sklearn.metrics import classification_report, accuracy_score
from operator import itemgetter
import numpy as np
import math
from collections import Counter

# 1) 给定两个数据点，计算它们之间的欧氏距离
def get_euclidean_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a-b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))

# 2) 给定训练集和测试实例，使用get_euclidean_distance计算所有成对距离
def get_neighbours(training_set, test_instance, k):
    distances = [get_tuple_distance(training_instance, test_instance) for training_instance in training_set]
    # index 1是training_instance和test_instance之间的计算距离
    sorted_distances = sorted(distances, key=itemgetter(1))
    # 仅提取训练实例
    sorted_training_instances = [tuple[0] for tuple in sorted_distances]
    # 选择前k个元素
    return sorted_training_instances[:k]

def get_tuple_distance(training_instance, test_instance):
    return (training_instance, get_euclidean_distance(test_instance, training_instance[0]))

# 3) 给定一个测试用例的最近邻的数组，将他们的类统计到对测试用例类
def get_majority_vote(neighbours):
    # index 1是类别
    classes = [neighbours[1] for neighbour in neighbours]
    count = Counter(classes)
    return count.most_common()[0][0]

# 设置主方法
def main():
    # 读取数据和创建训练集和测试集
    # random_state = 1只是种子，可以允许训练集和测试集划分的重现性
    iris = load_iris()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)

    # 为了方便起见，重新组合训练集和测试集数据
    train = np.array(zip(X_train, y_train))
    test = np.array(zip(X_test, y_test))

    # 生成预测
    predictions = []

    # 让我们任意设置k等于5，这意味着要预测新实例的类
    k = 5

    # 对于测试集中的每个实例，获得最邻近的预测类
    for x in range(len(X_test)):

        print ('分类测试实例的号码' + str(x) + ':')
        neighbours = get_neighbours(training_set=train, test_instance=test[x][0], k=5)
        majority_vote = get_majority_vote(neighbours)
        predictions.append(majority_vote)
        print ('预测的类别:' + str(majority_vote) + ',实际的类别:' + str(test[x][1]))

    # 评价分类器的表现
    print ('\n模型的总体准确率:' + str(accuracy_score(y_test, predictions)) + "\n")
    report = classification_report(y_test, predictions, target_names=iris.target_names)
    print ('分类器的具体细节: \n\n' + report)

if __name__ == "__main__":
    main()