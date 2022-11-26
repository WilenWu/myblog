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

信息的大小跟随机事件的概率有关，越小概率的事情发生产生的信息量越大。如"今天早上太阳升起"如此肯定，没什么信息量，但"今天早上有日食"信息量就很丰富。我们想要通过这种基本想法来量化信息。特别地

-  非常可能发生的事件信息量要比较少，并且极端情况下，确定发生的事件没有信息量。
- 较不可能发生的事件具有更高的信息量。
- 独立事件应具有增量的信息。例如，投掷的硬币两次正面朝上传递的信息量，应该是投掷一次硬币正面朝上的信息量的两倍。

为了满足上述三个性质，定义**自信息**（self-information）来量化一个事件的信息。若离散型随机变量 $X=x$ 发生的概率 $\mathbb P(X=x)=P(x)$，自信息为
$$
I(x)=\log\frac{1}{P(x)}=-\log P(x)
$$

# 信息熵

自信息只度量单个事件的信息量，我们可以使用自信息的期望度量整个概率分布中的复杂程度，称为**信息熵**（information entropy）
$$
H(X)=\mathbb E(I)=-\sum_i P(x_i)\log P(x_i)
$$
其中 $P(x_i)$ 代表随机事件 $X=x_i$ 的概率。若 $P(x_i)=0$ ，则定义 $0\log 0=0$ 。通常对数以2为底或以e为底，这时熵的单位分别称作 bit 或 nat 。由定义可知，熵只依赖 $X$ 的分布，而与 $X$ 的取值无关，因此熵也记作 $H(P)$ 。

熵越大，随机变量的不确定性越大。若随机变量可以取 $n$ 个离散值，则
$$
0\leqslant H(P)\leqslant \log n
$$
当随机变量只取两个值时，熵曲线如下图

![information entropy](Information-Theory.assets/information%20entropy.svg)

信息熵表示对离散随机变量 $X$ 进行编码所需的最小字节数。

# 条件熵

设随机变量 $(X,Y)$ 联合概率分布为 $\mathbb P(X=x_i,Y=y_j)=P(x_i,y_j)$ 。条件熵（condition entropy）用来度量在已知随机变量 $X$ 的条件下随机变量 $Y$ 的不确定性，定义为 $Y$ 的条件概率分布 $P(Y|X)$ 的熵对 $X$ 的期望
$$
\begin{aligned}
H(Y|X)&=\sum_i P(x_i)H(Y|X=x_i) \\
&=-\sum_i\sum_jP(x_i,y_j)\log P(y_j|x_i)
\end{aligned}
$$
其实条件熵可以理解为利用随机变量 $X$ 对 $Y$ 分组后，计算熵 $H(Y)$ 的加权平均值。事件的条件熵一般小于熵，例如，知道西瓜的色泽(青绿,乌黑,浅白)后，西瓜质量的不确定性就会减少了。

# 交叉熵

如果对于同一个随机变量 $X$ 有两个单独的概率分布$P(x)$和 $Q(x)$，我们使用**交叉熵**（cross entropy）来度量两个概率分布间的差异性信息
$$
H(P,Q)=-\sum_i P(x_i)\log Q(x_i)
$$
交叉熵表示离散随机变量 $X$ 使用基于 $Q$ 的编码对来自 $P$ 的变量进行编码所需的字节数。
# KL散度

KL散度（Kullback-Leibler divergence）亦称相对熵（relative entropy）或信息散度（information divergence），可用于度量两个概率分布之间的差异。给定两个概率分布$P(x)$和 $Q(x)$ 
$$
KL(P\|Q)=\sum_i P(x_i)\log\frac{P(x_i)}{Q(x_i)}
$$
若将KL散度的定义展开，可得
$$
KL(P\|Q)=H(P,Q)-H(P)
$$
其中 $H(P)$ 为熵，$H(P,Q)$ 为交叉熵。因此，KL散度可认为是使用基于 Q 的编码对来自 P 的变量进行编码所需的**额外**字节数。由吉布斯不等式可知，相对熵是恒大于等于0的。当且仅当两分布相同时，相对熵等于0。
$$

$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/KL-divergence.png" style="zoom:50%;" />

相对熵是一些优化算法，例如最大期望算法（Expectation-Maximization algorithm, EM）的损失函数。此时参与计算的一个概率分布为真实分布，另一个为理论（拟合）分布，相对熵表示使用理论分布拟合真实分布时产生的信息损耗 。
