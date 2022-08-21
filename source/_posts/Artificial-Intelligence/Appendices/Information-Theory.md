---
title: 信息论
katex: true
categories:
  - Artificial Intelligence
  - Appendices
tags:
  - 机器学习
cover: /img/data-analysis.png
top_img: /img/data-chart3.jpg
abbrlink: 8a8bfa9b
date: 2022-07-31 19:04:42
description:
---

# 概述

信息论：信息论是运用概率论与数理统计的方法研究信息、信息熵、通信系统、数据传输、密码学、数据压缩等问题的应用数学学科。

# 信息熵

自信息
$$
I(x)=-\log P(x)
$$
信息熵（information entropy）：描述随机变量 $X$ 的不确定性，信息熵越大越不确定。对于 $x\in X$ ，其出现概率为 $P(x)$
$$
H(P)=E[I(X)]=-\sum P(x)\log P(x)
$$
# 交叉熵

交叉熵（cross entropy）主要用于度量两个概率分布间的差异性信息。

设 $P(x),Q(x)$ 是随机变量 $X$ 上的两个概率分布，其中 $P$ 为真实分布，$Q$ 为非真实分布，则在离散和连续随机变量的情形下，交叉熵的定义为
$$
H(P,Q)=-\sum P(x)\log Q(x) \\
H(P,Q)=-\int P(x)\log Q(x)\mathrm dx
$$

# 相对熵

相对熵（relative entropy）：又被称为 KL 散度（Kullback-Leibler divergence）是两个概率分布间差异的非对称性度量。在信息理论中，相对熵等价于两个概率分布的信息熵的差值。

相对熵是一些优化算法，例如最大期望算法（Expectation-Maximization algorithm, EM）的损失函数。此时参与计算的一个概率分布为真实分布，另一个为理论（拟合）分布，相对熵表示使用理论分布拟合真实分布时产生的信息损耗 。

设 $P(x),Q(x)$ 是随机变量 $X$ 上的两个概率分布，则在离散和连续随机变量的情形下，相对熵的定义分别为
$$
KL(P\|Q)=\sum P(x)\log\cfrac{P(x)}{Q(x)} \\
KL(P\|Q)=\int P(x)\log\cfrac{P(x)}{Q(x)}\mathrm dx
$$
由吉布斯不等式可知，相对熵是恒大于等于0的。当且仅当两分布相同时，相对熵等于0。
$$
KL(P\|Q)=H(P,Q)-H(P)
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/KL-divergence.png)
