# -*- coding: utf-8 -*-


from sklearn import datasets  # 从sklearn自带数据库中加载鸢尾花数据
from sklearn.model_selection import train_test_split  # 引入train_test_split函数
from sklearn.neighbors import KNeighborsClassifier  # 引入KNN分类器
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()  # 将鸢尾花数据存在iris中
iris_X = iris.data  # 指定训练数据iris_X
iris_y = iris.target  # 指定训练目标iris_y
# print(iris_X[:2,:])   # 查看前两个例子的所有特征值
# print(iris_y)  # 查看目标标签名称

# 使用train_test_split（）函数将数据集分成用于训练的data和用于测试的data
x_train, x_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

knn = KNeighborsClassifier()  # 调用KNN分类器
knn.fit(x_train, y_train)  # 训练KNN分类器
y_pred = knn.predict(x_test)
print(y_pred)  # 预测值
print(y_test)  # 真实值
print("准确率：", accuracy_score(y_pred, y_test))  # 计算准确率