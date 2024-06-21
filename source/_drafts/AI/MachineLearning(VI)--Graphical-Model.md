---
title: 机器学习--概率图模型
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/data-analysis.png
top_img: /img/data-chart3.jpg
abbrlink:
description:
date:
---

概率图模型（Probabilistic Graphical Model，PGM），简称图模
Model，GM），是指一种用图结构来描述多元随机变量之间条件
模型，从而给研究高维空间中的概率模型带来了很大的便捷性．

结构化概率模型使用图（在图论中 ‘‘结点’’ 是通过 ‘‘边
变量之间的相互作用。每一个结点代表一个随机变量。每
作用。这些直接相互作用隐含着其他的间接相互作用，但
被显式地建模。
使用图来描述概率分布中相互作用的方法不止一种。
最为流行和有用的方法。图模型可以被大致分为两类：基于有向无环图的模型和基于无向图的模型。

https://zhuanlan.zhihu.com/p/391186508?utm_id=0


# 贝叶斯网

有向图模型（directed graphical model）是一种结构化概
络（belief network）或者 贝叶斯网络（Bayesian network

有向图模型(Directed Graphical Model)，也称为贝叶斯网络(Bayesian Network)或信念网络(Belief Network，BN)，是一类用有向图来描述随机向量 概率分布的模型.

**贝叶斯网**（Bayesian network）亦称信念网（belief network），不要求给定类的所有特征都条件独立，而是允许指定哪些特征条件独立。贝叶斯网借助有向无环图（Directed Acyclic Graph，DAG）来刻画特征之间的依赖关系，并使用条件概率表（Conditional Probability Table，CPT）来描述属特征的联合概率分布。 

有向图模型
work）或信念
分布的模型．
有向图模型
work）或信念
分布的模型正式

贝叶斯网借助有向无环图（Directed Acyclic Graph，DAG）来刻画特征之间的依赖关系，并使用条件概率表（Conditional Probability Table，CPT）来描述特征的联合概率分布。

贝叶斯网 (Bayesian network)亦称"信念网" (belief network)，它借助有向 无环图(Directed Acyclic Graph，简称 DAG)来刻画属性之间的依赖关系

用条件概 率表 (Conditional Probability Table， 简称 CPT)来描述属 性的 联合概 率分布 .



贝叶斯网用图形表示一组随机变量之间的概率关系，有两个主要成分：

1. **有向无环图**（Directed Acyclic Graph，DAG ）：表示变量之间的依赖关系；
2. **概率表**（Probability Table）：把各结点和它的直接父结点关联起来。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/DAG.svg)



对于贝叶斯网络精确推理方法有变量消除法、信念传播法，但其精确推理的复杂度为NP-hard。请参考西瓜书中内容。

此外还以通过采样的方法来实现近似推理，例如直接采样法，加权采样法，Gibbs采样。

直接采样会产生很多与估计无关的样本

![](v2-38802c39c257b5946bc2c13d075565d8_1440w.webp)

使用加权采样可以避免这种无关的采样，但是当某些概率值较小时如果采样次数较少可能会相对真实分布发生大的偏差。

![](v2-ddba9e861db272cb0f2d48a2524c687c_1440w.webp)

Gibbs 采样是Metropolis-Hasting算法的特殊情况。其定义建议分布是当前变量𝑥𝑖的满条件分布，并且接受率为1。其在BN中的伪代码如下所示。我们的每次采样都会在特征空间当中相对坐标轴平行移动。

![](v2-d2528ca4c855fddd957348a2293bc956_1440w.webp)

# 隐马尔可夫模型

# 马尔可夫随机场
有向图模型为我们提供了一种描述结构化概率模型的语
是 无向模型（undirected Model），也被称为 马尔可夫随
MRF）或者是 马尔可夫网络（Markov network）(Kind
名字所说的那样，无向模型中所有的边都是没有方向的。

# 条件随机场

# 学习与推断

# 近似推断