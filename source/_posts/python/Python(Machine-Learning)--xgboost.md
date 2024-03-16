---
title: Python(Machine Learning)--XGBoost
tags:
  - Python
  - 机器学习
categories:
  - Python
  - Machine Learning
cover: /img/XGBoost-cover.svg
top_img: /img/XGBoost-cover.svg
abbrlink: c46d5dae
date: 2024-01-25 22:15:00
description:
---

第113天： Python XGBoost 算法项目实战https://mp.weixin.qq.com/s?__biz=MzkxNDI3NjcwMw==&mid=2247493423&idx=1&sn=f04891095c8d95c491e1575f7895bb77&chksm=c1724f1ff605c609baab543b7515109a75470dad1eb228a15e29bd1ef3918964b5b782f60c13&scene=21#wechat_redirect

~/Downloads/集成学习公开课0709 XGBoost (B站节选版).ipynb

[XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/index.html)

[Collection of examples for using xgboost.spark estimator interface](https://xgboost.readthedocs.io/en/stable/python/examples/spark_estimator_examples.html)

[Demo for using process_type with prune and refresh](https://xgboost.readthedocs.io/en/stable/python/examples/update_process.html)

[Demo for training continuation](https://xgboost.readthedocs.io/en/stable/python/examples/continuation.html#demo-for-training-continuation)

# Overview

Simple example



# 参数

eta = 0.1
max_depth = 8
num_round = 500
nthread = 16
tree_method = exact
min_child_weight = 100

```python
watchlist = [(xgb_train, ``'train'``), (xgb_val, ``'val'``), (xgb_test, ``'test'``)]
```





# Scikit-Learn API

```python
from xgboost import XGBClassifier
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
# create model instance
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
```



# 可视化



# 继续训练

XGBoost提供两种增量训练的方式，

- 一种是在当前迭代树的基础上增加新树，原树不变；

- 一种是当前迭代树结构不变，重新计算叶节点权重，同时也可增加新树。

在指定的模型上继续训练。

```
# 指定基础模型参数，xgb_model
# 传入新的增量训练数据，xgb_train_new
model_new ``=` `xgb.train(params, xgb_train_new, num_boost, xgb_model``=``model_path)
```

# 分布式学习

## XGBoost with PySpark

从1.7.0版本开始，xgboost已经封装了pyspark API，因此不需要纠结spark版本对应的jar包 xgboost4j 和 xgboost4j-spark 的下载问题了，也不需要下载调度包 sparkxgb.zip。









