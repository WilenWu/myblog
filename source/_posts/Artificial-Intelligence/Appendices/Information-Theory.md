---
title: 机器学习中的信息论
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

# 自信息

信息的大小跟随机事件的概率有关，越小概率的事情发生产生的信息量越大。如“今天早上太阳升起”如此肯定，没什么信息量，但"今天早上有日食"信息量就很丰富。我们想要通过这种基本想法来量化信息。特别地

-  非常可能发生的事件信息量要比较少，并且极端情况下，确定发生的事件没有信息量。
- 较不可能发生的事件具有更高的信息量。
- 独立事件应具有增量的信息。例如，投掷的硬币两次正面朝上传递的信息量，应该是投掷一次硬币正面朝上的信息量的两倍。

为了满足上述三个性质，定义**自信息**（self-information）来量化一个事件的信息。若离散型随机变量 $X$ 的概率分布为 $P(x)$，定义自信息
$$
I(x)=\log\frac{1}{P(x)}=-\log P(x)
$$

# 信息熵

自信息只度量单个事件的信息量，我们可以使用自信息的期望度量整个概率分布中的复杂程度，称为**信息熵**（information entropy）
$$
H(X)=\mathbb E(I)=-\sum_{x\in X} P(x)\log P(x)
$$
其中 $P(x)$ 代表随机事件 $X=x$ 的概率。若 $P(x)=0$ ，则定义 $0\log 0=0$ 。通常对数以2为底或以e为底，这时熵的单位分别称作 bit 或 nat 。由定义可知，熵只依赖 $X$ 的分布，而与 $X$ 的取值无关，因此熵也记作 $H(P)$ 。信息熵也表示对离散随机变量 $X$ 进行编码所需的最小字节数。

熵越大，随机变量的不确定性越大。若随机变量可以取 $n$ 个离散值，则
$$
0\leqslant H(P)\leqslant \log n
$$
随机变量只取两个值的熵曲线如下图

![information entropy](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/information_entropy.svg)

# 条件熵

**条件熵**（condition entropy）用来度量在已知随机变量 $X$ 的条件下随机变量 $Y$ 的不确定性，定义为 $Y$ 的条件概率分布 $P(Y|X)$ 的熵对 $X$ 的期望
$$
\begin{aligned}
H(Y|X)&=\sum_{x\in X} P(x)H(Y|x) \\
&=-\sum_{x\in X}\sum_{y\in Y}P(x,y)\log P(y|x)
\end{aligned}
$$
其中，$P(x,y)$ 是随机变量 $(X,Y)$ 联合概率分布。其实条件熵可以理解为利用随机变量 $X$ 对 $Y$ 分组后，计算熵 $H(Y)$ 的加权平均值。事件的条件熵一般小于熵，例如，知道西瓜的色泽（青绿，乌黑，浅白）后，西瓜质量的不确定性就会减少了。

# 交叉熵

如果对于同一个随机变量 $X$ 有两个单独的概率分布$P(x)$和 $Q(x)$，我们使用**交叉熵**（cross entropy）来度量两个概率分布间的差异性信息
$$
H(P,Q)=-\sum_{x\in X} P(x)\log Q(x)
$$
交叉熵表示离散随机变量 $X$ 使用基于 $Q$ 的编码对来自 $P$ 的变量进行编码所需的字节数。
# KL散度

**KL散度**（Kullback-Leibler divergence）亦称相对熵（relative entropy）或信息散度（information divergence），可用于度量两个概率分布之间的差异。给定两个概率分布$P(x)$和 $Q(x)$ 
$$
KL(P\|Q)=\sum_{x\in X} P(x)\log\frac{P(x)}{Q(x)}
$$
若将KL散度的定义展开，可得
$$
KL(P\|Q)=H(P,Q)-H(P)
$$
其中 $H(P)$ 为熵，$H(P,Q)$ 为交叉熵。因此，KL散度可认为是使用基于 $Q$ 的编码对来自 $P$ 的变量进行编码所需的**额外**字节数。

**性质**：

1. 非负性：由吉布斯不等式可知，$KL(P\|Q)\geqslant 0$，当且仅当 $P\equiv Q$ 时，$KL(P\|Q)= 0$。
2. 不对称性：相对熵是两个概率分布的不对称性度量，即$KL(P\|Q)\neq KL(Q\|P)$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/KL-divergence.png" style="zoom:50%;" />

相对熵是一些优化算法，例如最大期望算法（Expectation-Maximization algorithm, EM）的损失函数。此时参与计算的一个概率分布为真实分布，另一个为理论（拟合）分布，相对熵表示使用理论分布拟合真实分布时产生的信息损耗 。

# 互信息

互信息（Mutual Information）是信息论里一种有用的信息度量，它可以看成是一个随机变量中包含的关于另一个随机变量的信息量，或者说是一个随机变量由于已知另一个随机变量而减少的不肯定性。

两个离散随机变量 $X$ 和 $Y$ 的互信息可以定义为：
$$
I(X;Y)=\sum_{x\in X}\sum_{y\in Y} P(x,y)\log\frac{P(x,y)}{P(x)P(y)}
$$
其中 $P(x,y)$ 是 $X$ 和 $Y$ 的联合概率分布函数，而$P(x)$和$P(y)$分别是 $X$ 和 $Y$ 的边缘概率分布函数。互信息是联合分布$P(x,y)$与边缘分布$P(x)P(y)$的相对熵。

**性质**：

1. 对称性：$I(X;Y)=I(Y;X)$
2. 非负性：$I(X;Y)\geqslant 0$，当且仅当 $X$ 和 $Y$ 独立时，$I(X;Y)= 0$

按照熵的定义展开可以得到：
$$
\begin{aligned} I(X;Y) &=H(X)-H(X|Y) \\
&=H(Y)-H(Y|X) \\
&=H(X)+H(Y)-H(X,Y)
\end{aligned}
$$
其中$H(X)$和$H(Y)$ 是$X$ 和 $Y$ 熵，$H(X|Y)$和$H(Y|X)$是条件熵，而$H(X,Y)$是联合熵。

![信息关系图](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/mutual_information.svg)
