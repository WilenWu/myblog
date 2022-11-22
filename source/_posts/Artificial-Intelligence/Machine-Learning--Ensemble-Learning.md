---
title: 机器学习--集成学习
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/data-analysis.png
top_img: /img/data-chart3.jpg
abbrlink: aaff9a95
description:
date:
---



# 集成方法

随机森林超参数：决策树个数、最大深度、信息度量方式（gini、entropy）、特征选择方法（auto、all、sqrt、log2、onethird）

Adaboost分类超参数：数据标准化（归一化、标准化、无处理）、分类器个数、最大迭代次数、正则化参数（控制模型复杂度）、收敛容差

## Tree ensembles

事实证明，只需要改变一个训练样本，可拆分的最高信息增益对应的特征就可能发生改变，因此在根节点会产生一个不同的划分，生成一颗完全不同的树。

因此，单个决策树对数据的微小变化非常敏感。让算法变得更健壮(robust)的一个方法是构建不知一颗决策树，这称之为树集成(Tree ensemble)。

- Using multiple decision trees 多颗树投票决定预测结果
- Sampling with replacement 使用有放回抽样创建新的随机训练集

**Random forest algorithm** 随机森林

Given training set of size $m$

For $b=1$ to $B$

- 使用有放回抽样创建一个大小为 $m$ 的新训练集 ，
- 在新训练集上训练集一颗决策树

让这些树投票决定预测结果。事实证明，让 $B$ 变大不会影响性能，但过了某个点后，你会发现收益递减。这种算法称为袋装决策树(bagged decision tree)。

即使使用放回抽样，有时总是在根节点上使用相同的划分，在根节点附近也有相似的特征组成。所以尝试对算法做进一步的修改，随机每个节点的特征选择，这可能会获得更准确的预测结果。

通常做法是每个节点上选择一个特征进行划分，如果总共有 $n$ 个特征可用，随机选择 $k<n$ 个特征子集，允许算法从这 $k$ 个特征子集中选择最大信息增益的特征来划分。当特征数量大时，通常会选择 $k=\sqrt{n}$ 。这种算法称为随机森林(Random forest)

**XGBoost**

Given training set of size $m$

For $b=1$ to $B$

- 抽样创建一个大小为 $m$ 的新训练集 ，代替从所有的样本等概率抽样（$1/m$），更倾向于选出之前训练的决策树分类错误的样本。（具体概率的数学细节相当复杂）
- 在新训练集上训练集一颗决策树

让这些树投票决定预测结果。种算法称为极端梯度增强 (XGBoost , eXtreme Gradient Boosting)

- 开源
- 增强树的实现，非常快速和高效
- 默认的拆分准则和停止拆分的准则都有很好的选择
- 内置正则化防止过拟合

```python
from xgboost import XGBClassifier 
model = XGBClassifier()
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
```

## Bagging回归

## Voting回归