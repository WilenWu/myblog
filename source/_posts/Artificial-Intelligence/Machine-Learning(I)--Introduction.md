---
title: 机器学习(I)--引言
date: 2023-05-02 14:36
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-overview.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 8638abf9
description:
---

# 前言

**机器学习** (machine learning, ML)的主要目的是设计和分析一些学习算法，让计算机可以从数据（经验）中自动分析并获得规律，之后利用学习到的规律对未知数据进行预测，从而帮助人们完成一些特定任务，提高开发效率。

首先我们以一个生活中的例子来介绍机器学习中的一些基本概念。首先，我们从市场上随机选取一些西瓜，列出每个西瓜的**特征**(feature)，包括颜色、大小、形状、产地、品牌等，以及我们需要预测的**标签**(label)。特征也可以称为**属性**(attribute)，标签也称为**目标**(target)。我们可以将一个标记好特征以及标签的西瓜看作一个**样本**(sample)，也经常称为**示例**(instance)。

一组样本构成的集合称为**数据集**(data set)。一般将数据集分为两部分：**训练集**(training set)和**测试集**(test set)。训练集中的样本是用来训练模型的，而测试集中的样本是用来检验模型好坏的。

我们通常会用
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}
$$
表示$N$个样本的数据集，每个特征$\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 是包含$p$个特征的向量，称为**特征向量**(feature vector)。而标签通常用标量$y$来表示。

我们可以把机器学习过程看作在某个函数集合中进行搜索的过程，应用某个评价准则（evaluation criterion），找到与训练集匹配最优的模型。模型所属的函数集合称为**假设空间**（hypothesis space），评价准则要优化的函数称为**代价函数**（cost function）。

机器学习算法可以按照不同的标准来进行分类，比如按照目标函数的不同可分为线性模型和非线性模型。但一般来说，我们会按照训练样本提供的信息以及反馈方式的不同，将机器学习算法分为监督学习、无监督学习、强化学习。有时还包括半监督学习、主动学习。

- **监督学习**(supervised learning)：如果每个样本都有标签，机器学习的目的是建立特征和标签之间的映射关系，那么这类机器学习称为监督学习。根据标签数值类型的不同，监督学习又可以分为回归问题和分类问题。**回归**(regression)问题中的标签是连续值，**回归分类**(regression)问题中的标签是离散值。
- **无监督学习**(unsupervised learning)：是指从不包含目标标签的训练样本中自动学习到一些有价值的信息。典型的无监督学习问题有聚类、密度估计、特征学习、降维等。
- **强化学习**(reinforcement learning)：并不是训练于一个固定的数据集上。强化学习会和环境进行交互，在没有人类操作者指导的情况下，通过试错来学习。

监督学习需要每个样本都有标签，而无监督学习则不需要标签．一般而言，监督学习通常需要大量的有标签数据集，这些数据集一般都需要由人工进行标注，成本很高．因此，也出现了很多弱监督学习（Weakly Supervised Learning）和**半监督学习**（Semi-Supervised Learning，SSL）的方法，希望从大规模的无标注数据中充分挖掘有用的信息，降低对标注样本数量的要求．强化学习和监督学习的不同在于，强化学习不需要显式地以“输入/输出对”的方式给出训练样本，是一种在线的学习机制。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/machine-learning-map.png)

在实际任务中使用机器学习模型一般会包含以下几个**步骤**：

1. **数据预处理**：对数据的原始形式进行初步的缺失值处理、数据规范化、数据变换等。
2. **特征工程**：主要有特征选择、特征提取、降维等。 
4. **模型训练及评估**：机器学习的核心部分，学习模型并逐渐调优。

**没有免费午餐定理** (No Free Lunch Theorem，NFL) 是由Wolpert 和Macerday在最优化理论中提出的。没有免费午餐定理证明：对于基于迭代的最优化算法，不存在某种算法对所有问题（有限的搜索空间内）都有效。如果一个算法对某些问题有效，那么它一定在另外一些问题上比纯随机搜索算法更差。也就是说，不能脱离具体问题来谈论算法的优劣，任何算法都有局限性．必须要**具体问题具体分析**。

没有免费午餐定理对于机器学习算法也同样适用，不存在一种机器学习算法适合于任何领域或任务。

**奥卡姆剃刀** (Occam's razor) 是一种常用的、自然科学研究中最基本的原则。主张选择与经验观察一致的最简单假设。如果有两个性能相近的模型，我们应该选择更简单的模型数据探索。

# 描述统计

数据集的特征通常分为连续（continuous）和离散（discrete）两种。描述统计（summary statistics）是用量化的单个数或少量的数来概括数据的各种特征，主要包括集中趋势分析、离散程度分析和相关分析三大部分。

**平均数**（mean）：是描述数据集中趋势的测度值。均值容易受极值的影响，当数据集中出现极值时，所得到的的均值结果将会出现较大的偏差。
$$
\mu=\bar x=\frac{1}{N} \sum_{i=1}^N x_i
$$
**中位数**（median）：是集中趋势的测量。中位数是按顺序排列的一组数据中居于中间位置的数，大于该值和小于该值的样本数各占一半。中位数不受极值影响，因此对极值缺乏敏感性。通常用 $m_{0.5}$ 来表示中位数。

**四分位数**（quartile）：是指在统计学中把所有数值由小到大排列并分成四等份，处于三个分割点位置的数值，即为四分位数。四分位数分为上四分位数（处在75%位置上的数值，即最大的四分位数）、下四分位数（处在25%位置上的数值，即最小的四分位数）、中间的四分位数（即为中位数）。三个四分位数通常用 $Q_1,Q_2,Q_3$表示。

**四分卫间距**（InterQuartile Range，IQR）：第三四分位数与第一四分位数的差距，与方差、标准差一样，表示变量的离散程度。
$$
\text{IQR}=Q_3-Q_1
$$
**极差**（range）：连续变量最大值和最小值之间的差值。极差是描述数据分散程度的量，极差描述了数据的范围，但无法描述其分布状态。且对异常值敏感，异常值的出现使得数据集的极差有很强的误导性。
$$
R=\max(x)-\min(x)
$$
**方差**（variance）与**标准差**（standard deviation）：是描述数据的离散程度。在正态分布中，95% 的样本在平均值的两倍标准差范围内。
$$
\sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2
$$
**变异系数**（coefficient  of variation，CV）：标准差与期望的比值。变异系数是个无量纲的量，可以比较度量单位不同的数据集之间的离散程度的差异。
$$
\text{CV}=\frac{\sigma}{\mu}
$$
**偏度**（skewness）：用来衡量分布的不对称性。任何对称分布偏度为 0。具有显著的正偏度的分布有很长的右尾，具有显著的负偏度的分布有很长的左尾。一般当偏度大于两倍标准差时，认为分布不具有对称性。
$$
S=\frac{1}{N} \sum_{i=1}^N (\frac{x_i - \mu}{\sigma})^3
$$
**峰度**（kurtosis）：以正态分布为标准，用来衡量量分布的陡峭程度。完全符合正态分布的数据峰度值为 0，且正态分布曲线被称为基线。峰度为正表示数据比正态分布峰要陡峭，称为尖峰态（leptokurtic）。峰度为负表示数据比正态分布峰要平缓，称为平峰态（platykurtic）。
$$
K=\frac{1}{N} \sum_{i=1}^N (\frac{x_i - \mu}{\sigma})^4-3
$$

**频率**（frequency）：是指离散变量每个值出现的次数与总次数的比值，用来表示变量的分布情况。

**众数**（mode）：是指离散变量频率最大的数值。描述数据的集中趋势，不受极值影响。

**列联表**（contingency table）是由两个或更多变量进行交叉分类的频数分布表。列联表的目的是寻找变量间的关系。

**协方差**（covariance）用于衡量两个变量的总体误差，可作为描述两个变量相关程度的量
$$
\text{cov}(x,y)=\frac{1}{N}\sum_{i=1}^N(x_i-\bar x)(y_i-\bar y)
$$
# 范数

在机器学习中，我们经常使用被称为**范数**（norm）的函数衡量向量大小。向量 $\mathbf x\in\R^n$ 的L~p~ 范数定义如下
$$
\|\mathbf x\|_p=(|x_1|^p+|x_2|^p+\cdots+|x_n|^p)^{1/p}
$$

其中 $p\in\R$ 且 $p\geqslant 1$ 。直观上来说，向量 $\mathbf x$ 的范数衡量从原点到点 $\mathbf x$ 的距离。

- 当 $p=1$ 时，L~1~ 范数表示从原点到点 $\mathbf x$ 的 Manhattan 距离
- 当 $p=2$ 时，L~2~ 范数表示从原点到点 $\mathbf x$ 的 Euclidean 距离。L~2~ 范数在机器学习中出现地十分频繁，经常简化表示为$\|\mathbf x\|$，略去了下标2。平方L~2~ 范数也经用来衡量向量的大小，可以简单地通过点积 $\mathbf x^T\mathbf x$ 计算。

另外一个经常在机器学习中出现的范数是 $L_\infty$ 范数，也被称为**最大范数**（max norm）。这个范数表示向量中具有最大幅值的元素的绝对值
$$
\|\mathbf x\|_\infty=\max\limits_{1\leqslant i\leqslant n}|x_i|
$$
同时结合L~1~和L~2~的混合范数称为弹性网（Elastic-Net）
$$
R(\mathbf x)=\rho \|\mathbf x\|_1+ \frac{(1-\rho)}{2} \|\mathbf x\|_2^2
$$
有时候我们可能也希望衡量矩阵的大小。在深度学习中，最常见的做法是使用 **Frobenius 范数**（Frobenius norm）
$$
\mathbf \|\mathbf A\|_{F}=\sqrt{\sum_{ij}a^2_{ij}}=\sqrt{\text{tr}(\mathbf A^T\mathbf A)}
$$

# 独立性检验

假设检验（Hypothesis Testing），或者叫做显著性检验（Significance Testing）是数理统计学中根据一定假设条件由样本推断总体的一种方法。

## 卡方检验

**卡方检验**（Pearson Chi-square test）：用于检验==两个分类变量==是否相互独立。原假设为两分类变量独立，即 $P(XY)=P(X)P(Y)$。假设两分类变量的类别数分别为 $r$ 和 $c$，将样本分成$rc$个区间，其根本思想就是在于比较原假设期望频数和实际频数的吻合程度
$$
\chi^2=\sum_{j=1}^c\sum_{i=1}^r\frac{(A_{ij}-E_{ij})^2}{E_{ij}}=\sum_{j=1}^c\sum_{i=1}^r\frac{(A_{ij}-np_{ij})^2}{np_{ij}}\sim \chi^2((r-1)(c-1))
$$
其中 $A_{ij}$ 是小区间 $i,j$ 的样本数，$E_{ij}$是第 $i,j$ 个小区间的期望频数，$n$ 为总样本数，$p_{ij}$是小区间 $i,j$ 的期望频率。当$n$比较大时，$\chi^2$ 统计量服从 $(r-1)(c-1)$ 个自由度的卡方分布。

| 实际分布  | $y_1$ | $y_2$ | total |
| --------- | ----- | ----- | ----- |
| $x_1$     | 15    | 85    | 100   |
| $x_2$     | 95    | 5     | 100   |
| **total** | 110   | 90    | 200   |

变量 $X$ 的期望频率 $P(x_1)=100/200=0.5$
变量 $Y$ 的期望频率 $P(y_1)=110/200=0.55,P(y_2)=0.45$

| 期望频数  | $y_1$ | $y_2$ | total |
| --------- | ----- | ----- | ----- |
| $x_1$     | 55    | 45    | 100   |
| $x_2$     | 55    | 45    | 100   |
| **total** | 110   | 90    | 200   |

$\chi^2=\cfrac{(15-55)^2}{55}+\dfrac{(85-45)^2}{45}+\dfrac{(95-55)^2}{55}+\dfrac{(5-45)^2}{45}=129.3>10.828$ ，显著相关。

## CMH检验

Cochran-Mantel-Haenszel 检验，简称CMH检验。 其原假设是，两个分类变量在第三个变量的每一层中都是条件独立的。

## Kruskal-Wallis 检验

**Kruskal-Wallis 检验**：用于检验有序分类变量和分类变量的相关性。若分类变量有 $k$ 个类别，原假设 $H_0:\mu_1=\mu_2=\cdots=\mu_k$。检验统计量
$$
H=\frac{12}{n(n+1)}\sum_{i=1}^kn_i(\bar R_i-\bar R)^2\sim \chi^2(k-1)
$$
其中，$n$为总的样本数，$n_i$为分类变量第 $i$ 个类别的样本数，$R_i=\text{rank}(x_i)$ 是有序分类变量的秩。

## 单因素方差分析

方差分析（Analysis of Variance，简称ANOVA）用于检验连续变量和分类变量的相关性。对于仅有一个分类变量， 又称为单因素方差分析（one-way ANOVA）。

方差分析认为方差的基本来源有两个：

(1) 组间差异：变量在各组的均值与总均值的偏差平方和，记作 SSA
(2) 随机误差：测量误差个体间的差异，各组的均值与该组内变量值的偏差平方和，记作 SSE

| 方差     | 自由度 df | 平方和 SS                                             | 均方 MS         |
| -------- | --------- | ----------------------------------------------------- | --------------- |
| 组间方差 | k-1       | $SSA=\sum_{i=1}^kn_i(\bar x_i-\bar x)^2$              | $MSA=SSA/(k-1)$ |
| 组内方差 | n-k       | $SSE=\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\bar x_i)^2$ | $MSE=SSE/(n-k)$ |
| 总方差   | n-1       | $SST=SSA+SSE$                                         | $MS=SST/(n-k)$  |

方差分析主要通过 **F检验**（F-test）来进行效果评测，原假设 $H_0:\mu_1=\mu_2=\cdots=\mu_k$。检验统计量
$$
F=\frac{MSA}{MSE}\sim F(k-1,n-k)
$$
其中，$n=\sum_{i=1}^kn_i$ 为总样本数， $k$ 为分类变量的类别数。

# 相关系数

## 相关系数

**相关系数**（correlation coefficient）反映了两个变量间的相关性，常用的相关系数如下：

(1) **Pearson相关系数**：衡量了两个连续变量之间的线性相关程度
$$
r=\frac{\text{cov}(x,y)}{\sigma_x\sigma_y}=\frac{\sum_{i=1}^n(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_{i=1}^n(x_i-\bar x)^2\sum_{i=1}^n(y_i-\bar y)^2}}
$$
其取值范围 $-1\leqslant r\leqslant1$ 。通常情况下通过以下相关系数取值范围判断变量的相关强度：0.8-1.0 极强相关；0.6-0.8 强相关；0.4-0.6 中等程度相关；0.2-0.4 弱相关；0.0-0.2 极弱相关或无相关。

如果 $x,y$ 都服从正态分布，可以用 F-test来检验Pearson相关系数的显著性
$$
F=\frac{r^2(n-2)}{1-r^2}\sim F(1,n-2)
$$
也可以用 t 检验 $t=\sqrt{F}\sim t(n-2)$

(2) **Spearman秩相关系数**：衡量两个有序分类变量之间的相关程度。定义和Pearson相关系数一样，但其值不考虑具体数值的影响，只和变量的秩相关
$$
r=1-\frac{6\sum_{i=1}^Nd_i^2}{n(n^2-1)}
$$
其中，$d_i$ 表示数据点 $(x_i,y_i)$ 的秩的差值 $d_i=\text{rank}(x_i)-\text{rank}(y_i)$ 。其取值范围 $-1\leqslant r\leqslant1$ 。

和Pearson相关系数一样，可以使用F检验  $F=\dfrac{r^2(n-2)}{1-r^2}\sim F(1,n-2)$ 或 t 检验 $t=\sqrt{F}\sim t(n-2)$ 来检验显著性。

(3) **Kendall’s Tau相关系数**：也是一种非参数的秩相关度量，是基于样本数据对  $(x_i,y_i)$ 之间的关系来衡量两个有序分类变量之间的相关程度。
$$
\tau=\frac{2}{n(n-1)}\sum_{i<j}\text{sign}(x_i-x_j)\text{sign}(y_i-y_j)
$$

简单的说就是将两个变量进行排序，使用同序对（Concordant pairs）和异序对（Discordant pairs）个数的差值度量。其取值范围 $-1\leqslant r\leqslant1$ 。同序对计数为 1，异序对计数为 -1，则数据对的判段逻辑如下
$$
p(i,j)=\text{sign}(x_i-x_j)\text{sign}(y_i-y_j)
$$

## Cramer's V系数

**Cramer's V系数**：衡量两分类变量的相关程度
$$
V=\frac{\chi^2}{n*\min\{r-1,c-1\}}
$$
其取值范围 $0\leqslant V\leqslant1$ ，越接近1相关性越强。卡方值
$$
\chi^2=\sum_{j=1}^c\sum_{i=1}^r\frac{(A_{ij}-np_{ij})^2}{np_{ij}}
$$
其中，两分类变量的类别数分别为 $r$ 和 $c$。 $A_{ij}$ 是小区间 $i,j$ 的样本数，$E_{ij}$是第 $i,j$ 个小区间的期望频数，$n$ 为总样本数，$p_{ij}$是小区间 $i,j$ 的期望频率。 

## IV 和 WOE

**IV**（Information Value，信息价值）用来评估分箱特征对二分类变量的预测能力，IV 值越大表示该变量预测能力越强。IV 值的计算相当于WOE值的一个加权求和。
$$
\text{IV}=\sum(\frac{P_i}{P_T}-\frac{N_i}{N_T})*\text{WOE}_i
$$
其中，$P_i,N_i$ 为第 $i$ 个分箱中的正负样本数，$P_T,N_T$ 为总的正负样本数。

**WOE**（Weight of Evidence，证据权重）用来衡量该组与整体的差异。
$$
\text{WOE}_i=\log(\frac{P_i}{P_T}/\frac{N_i}{N_T})
$$

| age       | positive | negative | WOE                        | weight |
| --------- | -------- | -------- | -------------------------- | ------ |
| <28       | 25       | 75       | $\log(0.25/0.375)=-0.4055$ | -0.125 |
| [28, 60]  | 60       | 40       | $\log(0.6/0.2)=1.0986$     | 0.4    |
| >=60      | 15       | 85       | $\log(0.15/0.425)=-1.0415$ | -0.275 |
| **total** | 100      | 200      |                            |        |

$\text{IV}=(-0.4055)*(-0.125)+1.0986*0.4+(-1.0415)*(-0.275)=0.7765$

## 方差膨胀因子

**方差膨胀因子**（Variance Inflation Factor，VIF）：通常用来衡量一个变量和其他变量间的多重共线性
$$
\text{VIF}=\frac{1}{1-R^2}
$$
多重共线性指目标变量可有其他变量线性组合。$R^2$ 指以目标变量拟合的线性模型的决定系数（R-Square），其取值范围为$[0,1]$，一般来说，R-Square 越大，表示模型拟合效果越好。
$$
R^2=1-\cfrac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y_i)^2}
$$

- 当 $VIF<10$时，不存在多重共线性；
- 当 $10\leqslant VIF<100$ 时，存在较强的多重共线性；
- 当 $VIF\geqslant 100$ 时，存在严重的多重共线性。

## 互信息

**互信息**（Mutual Information）：用于衡量两个分类变量相互依赖的程度
$$
I(X;Y)=\sum_{x\in X}\sum_{y\in Y} P(x,y)\log\frac{P(x,y)}{P(x)P(y)}
$$
其中 $P(x,y)$ 是 $X$ 和 $Y$ 的联合概率分布，而$P(x)$和$P(y)$分别是 $X$ 和 $Y$ 的边缘概率分布。

| 实际分布 (概率分布) | $y_1$      | $y_2$      | total (margin) |
| ------------------- | ---------- | ---------- | -------------- |
| $x_1$               | 15 (7.5%)  | 85 (42.5%) | 100 (50%)      |
| $x_2$               | 95 (47.5%) | 5 (2.5%)   | 100 (50%)      |
| **total (margin)**  | 110 (55%)  | 90 (45%)   | 200 (100%)     |

$I(x_1,y_1)=0.075*\log\dfrac{0.075}{0.5*0.55}=-0.0974$
$I(x_1,y_2)=0.425*\log\dfrac{0.425}{0.5*0.45}=0.2703$
$I(x_2,y_1)=0.475*\log\dfrac{0.475}{0.5*0.55}=0.2596$
$I(x_2,y_2)=0.025*\log\dfrac{0.025}{0.5*0.45}=-0.0549$
互信息 $I(X,Y)=-0.0974+0.2703+0.2596-0.0549=0.3776$

# 距离

距离描述两个样本间的相似关系。

**欧几里得距离**（Euclidean distance）
$$
d(\mathbf x,\mathbf y)=\sqrt{\sum_{i=1}^n(x_i-y_i)^2}
$$
**切比雪夫距离**（Chebyshev Distance）
$$
d(\mathbf x,\mathbf y)=\max(|x_i-y_i|)
$$
**闵可夫斯基距离**（Minkowski distance）
$$
d(\mathbf x,\mathbf y)=\left(\sum_{i=1}^n|x_i-y_i|^p\right)^{1/p}
$$
其中 $p\in\R$ 且 $p\geqslant 1$ 。下面是三个常见的例子

- 当 $p=1$ 时，表示 Manhattan 距离，绝对值距离
- 当 $p=2$ 时，表示 Euclidean 距离（L~2~ 范数）。
- 当 $p\to\infty$ 时，表示 Chebyshev 距离。

**坎贝拉距离**（Canberra distance）也称 Lance and Williams distance。
$$
d(\mathbf x,\mathbf y)=\sum_{i=1}^n\frac{|x_i-y_i|}{|x_i|+|y_i|}
$$
**Mahalanobis 距离**：是一种有效的计算两个未知样本集的相似度的方法。下式中 $\Sigma^{-1}$ 是协方差矩阵的逆。
$$
d(\mathbf x,\mathbf y)=(\mathbf x-\mathbf y)\Sigma^{-1}(\mathbf x-\mathbf y)^T
$$
**余弦距离**（Cosine distance）机器学习中借用这一概念来衡量两个样本之间的差异
$$
\cos(\mathbf x,\mathbf y)=\frac{\mathbf x\cdot\mathbf y}{\|\mathbf x\|\cdot\|\mathbf y\|}
$$
**杰卡德距离**（Jaccard distance）：用来衡量两个集合差异性的一种指标，它是杰卡德相似系数的补集，适用于集合相似性度量，字符串相似性度量 。
$$
d(X,Y)=1-J(X,Y)=1-\frac{|X\cap Y|}{|X\cup Y|}
$$

# 可视化

## 散点图

**散点图**（scatter plot）：是科研绘图中最常见的图形类型之一，通常用来发现两变量间的关系与相关性

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/scatterplot.svg" alt="散点图" style="zoom: 80%;" />

## 折线图

**折线图**（line plot）：显示随时间而变化的连续数据

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/lineplot.svg" style="zoom:80%;" />

## 直方图

**直方图**（histogram）：是一种统计报告图。是对连续变量的概率分布的估计

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/histplot.svg" alt="直方图" style="zoom:80%;" />

## 核密度估计

**核密度估计**（kernel density estimate，kde）：是一种用于估计概率密度函数的非参数方法，可看作直方图的拟合曲线。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/kdeplot.svg" alt="核密度图" style="zoom:80%;" />

## 箱线图

**箱线图**（box plot）：是由一组或多组连续型数据的「最小观测值」、第一四分位数、中位数、第三分位数和「最大观测值」来反映数据的分布情况的统计图。

![箱线图](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/boxplot.svg)

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/grouped_boxplot.svg" style="zoom:80%;" />

## 小提琴图

**小提琴图**（violin plot）：本质上是由核密度图和箱线图两种基本图形结合而来的，主要用来显示数据的分布形状。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/grouped_violinplot.svg" alt="grouped_violinplot" style="zoom:80%;" />

## 热力图

**热力图**（heatmap）：是一种通过色块将三维数据使用二维可视化的方法

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/heatmap.svg" alt="热力图" style="zoom:80%;" />

------

> **参考文献**：
> 周志华.《机器学习》（西瓜书）
> 李航.《统计学习方法》

