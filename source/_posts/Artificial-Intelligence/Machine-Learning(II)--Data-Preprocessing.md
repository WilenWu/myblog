---
title: 机器学习--数据预处理
date: 2023-05-03 14:36
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-data-preprocessing.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 5faa8f40
description:
---

Preprocessing
Feature extraction and normalization.

Applications: Transforming input data such as text for use with machine learning algorithms.
Algorithms: preprocessing, feature extraction, and more...



# 类别不平衡

针对非平衡的数据集，为了使得模型在这个数据集上学习的效果更加好，需要改变原数据集中结构分布比例的不合理，通过丢弃降低多值对应数量或者复制增加低值对应数量，让不同值下的样本数量大致相同。

平衡方法：random、smote

## 抽样

在数据挖掘和机器学习领域，从更大的数据集中抽样是很常见的做法。举例来说，你可能希
望选择两份随机样本，使用其中一份样本构建预测模型，使用另一份样本验证模型的有效性。
sample()函数能够让你从数据集中（有放回或无放回地）抽取大小为n的一个随机样本。

- 无放回抽样：等概率抽样
- 有放回抽样
- 分层抽样：在每个分层中无放回抽样

抽样 sampling
 随机采样  random  sampling  受总体中个体间自相关关系的影响  
 系统采样  systematic sampling  受个体特点规律性的影响
 随机-系统采样  systematic-random  sampling   不受上面的影响 
 多层次采样 strtifed  sampling   适用于可分层的总体

#  标准化和归一化

数据集的**标准化**是scikit-learn中实现**许多机器学习估计器**的**普遍要求**；如果个别特征看起来或多或少不像标准正态分布数据：**均值和单位方差为零**的高斯分布，则它们的性能可能不好。

在实践中，我们通常会忽略分布的形状，而只是通过删除每个特征的平均值来实现特征数据中心化，然后除以非常数特征的标准差来缩放数据。

**数据特征缩放（Feature Scaling）**是一种将数据的不同变量或特征的方位进行标准化的方法。在数据处理中较为常用，也被称之为数据标准化（Data Normalization）。

特征的均值 $\mu$ 和标准差 $\sigma$  如下
$$
\begin{align}
\mu &= \frac{1}{N} \sum_{i=1}^N x_i\\
\sigma^2 &= \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2
\end{align}
$$


**Min-Max Normalization**：使用最大最小值将特征归一化到 $[0,1]$ 区间
$$
z_i=\frac{x_i-x_\min}{x_\max-x_\min}
$$
**Mean Normalization**：使用平均值 $\mu$ 将特征归一化到 $[-1,1]$ 区间
$$
z_i=\frac{x_i-\mu}{x_\max-x_\min}
$$
**Standardization**：又称 Z-score Normalization。使用平均值 $\mu$ 和标准差 $\sigma$ 将样本特征值转换为均值为0，标准差为1的数据分布
$$
z_i=\frac{x_i-\mu}{\sigma}
$$
**RobustScaler**

数据集的标准化是许多机器学习估计器的普遍要求。通常，这是通过去除均值并缩放到单位方差来完成的。但是，离群值通常会以负面方式影响样本均值/方差。在这种情况下，中位数和四分位数间距通常会产生更好的结果。

使用对异常值鲁棒的统计信息来缩放特征。

此缩放器删除中位数，并根据分位数范围（默认值为IQR：四分位间距）缩放数据。IQR是第一个四分位数（25%分位数）和第3个四分位数（75%分位数）之间的范围。

**L2 Normalization** 使用特征向量的$l_2$范数 $\|\mathbf x\|$ 将特征归一化到 $[-1,1]$ 区间，且归一化后 $\displaystyle\sum_{i=1}^Nz_i^2=1$
$$
z_i=\frac{x_i}{\|\mathbf x\|}=\frac{x_i}{\sum_{i=1}^Nx_i^2}
$$


# Non-linear transformation



# 特征重编码

## one-hot编码

对有序（order）分类变量，可通过连续化将其转化为连续值，对于$k$分类变量可转化为 $k$ 维0-1向量

## 离散化

**分箱**

k-bins

分箱可按分箱宽度，按分箱数、按分位数、按平均值/标准差四种分箱方式

变量的重编码

重编码涉及根据同一个变量和/或其他变量的现有值创建新值的过程。举例来说，你可能想：

- 将一个连续型变量修改为一组类别值；
- 将误编码的值替换为正确值；
- 基于一组分数线创建一个表示及格/不及格的变量。

**二值化**





# 缺失值

缺失值：缺失值通常以符号NA（Not Available，不可用）表示

R 并不把无限的或者不可能出现的数值标记成缺失值。正无穷和负无穷分别用Inf和–Inf所标记。因此5/0返
回Inf。不可能的值（比如说，sin(Inf)）用NaN符号来标记（not a number，不是一个数）

 **缺失值模式**
 完全随机缺失  MCAR
 随机缺失  MAR
 非随机缺失NMAR
 缺失值插补
 配对删除 
 简单抽样填补
 均值/中位数/众数填补
 回归填补，多重插补

# 异常值

 **异常检测**  outlier
 正态分布  3倍标准差原则 
 多元正态  马氏距离 

截尾均值，即丢弃了最大5%和最小5%的数据和所有缺失值后的算术平均数。

# 特征选择

特征工程：

- 通常通过变换或者合并原始特征设计新的特征（多项式特征等）。

降维包括**特征抽取**（Feature Extraction）和**特征选择**（Feature Selection）两种途径。

**特征选择**（Feature Selection）是选取原始特征集合的一个有效子集，使得基于这个特征子集训练出来的模型准确率最高。简单地说，特征选择就是保留有用特征，移除冗余或无关的特征。

**子集搜索**一种直接的特征选择方法为子集搜索（Subset Search）。假设原始特征数为 $p$，则共有 $2^p$ 个候选子集。特征选择的目标是选择一个最优的候选子集。最暴力的做法是测试每个特征子集，看机器学习模型哪个子集上的准确率最高。但是这种方式效率太低。常用的方法是采用贪心的策略：由空集合开始，每一轮添加该轮最优的特征，称为**前向搜索**（Forward Search）；或者从原始特征集合开始，每次删除最无用的特征，称为**反向搜索**（Backward Search）。

子集搜索方法可以分为过滤式方法和包裹式方法：

- **过滤式方法**（Filter Method）是不依赖具体机器学习模型的特征选择方法。每次增加最有信息量的特征，或删除最没有信息量的特征。特征的信息量可以通过信息增益（Information Gain）来衡量，即引入特征后条件
  分布 $P_\theta(y|\mathbf x)$ 的不确定性（熵）的减少程度。
- **包裹式方法**（Wrapper Method）是使用后续机器学习模型的准确率作为评价来选择一个特征子集的方法。每次增加对后续机器学习模型最有用的特征，或删除对后续机器学习任务最无用的特征。这种方法是将机器学习模型包裹到特征选择过程的内部。

此外，我们还可以通过 $l_1$ **正则化**来实现特征选择．由于 $l_1$正则化会导致稀疏特征，因此间接实现了特征选择。

# 特征提取

特征提取（Feature Extraction）是构造一个新的特征空间，并将原始特征投影在新的空间中得到新的表示。以线性投影为例，令 $\mathbf x\in\R^p$ 为原始特征向量， $\mathbf x'\in\R^k$ 为经过线性投影后得到的在新空间中的特征向量，有
$$
\mathbf x'=\mathbf{Wx}
$$
其中，$\mathbf W=\R^{p\times k}$为映射矩阵。

特征抽取又可以分为监督和无监督的方法。监督的特征学习的目标是抽取对一个特定的预测任务最有用的特征，比如==线性判别分析==（Linear Discriminant Analysis，LDA）。而无监督的特征学习和具体任务无关，其目标通常是减少冗余信息和噪声，比如==主成分分析==（Principal Component Analysis，PCA）和==自编码器==（Auto-Encoder，AE）。

特征选择和特征抽取的优点是可以用较少的特征来表示原始特征中的大部分相关信息，去掉噪声信息，并进而提高计算效率和减小**维度灾难**（Curse of Dimensionality）。对于很多没有正则化的模型，特征选择和特征抽取非常必要。经过特征选择或特征抽取后，特征的数量一般会减少，因此特征选择和特征抽取也经常称为**维数约减或降维**（Dimension Reduction）。



------



变量选择旨在从原始变量中选出一些最有效变量以降低数据集的维度，用于提高学习算法性能。变量选择适用于有监督学习。

算法：选择变量选择的方法可根据特征变量和响应变量的类型不同给出对应的变量选择方法。

- 特征变量和响应变量都是字符型时，变量选择方法可选皮尔逊卡方、似然比卡方、cramer、ambda
- 特征变量为数值型，响应变量为字符型时，变量选择方法支持F检验
- 特征变量为字符型，响应变量为数值型时，变量选择方法支持F检验
- 特征变量和响应变量都是数值型时，变量选择方法支持t检验

变量选择给出各变量的重要性，用户可根据重要性选择变量

WOE编码

WOE编码是对原始自变量的一种编码形式，表示自变量取某个值的时候对响应比例的一种影响。输出WOE值和IV值，可根据IV值进行变量筛选。

# 特征构造

密度，多项式特征

生成多项式特征

通常，考虑输入数据的非线性特征会增加模型的复杂性。多项式特征是一种简单而常用的方法，它可以获取特征的高阶和相互作用项。



# 降维

Dimensionality reduction
Reducing the number of random variables to consider.

Applications: Visualization, Increased efficiency
Algorithms: PCA, feature selection, non-negative matrix factorization, and more...
