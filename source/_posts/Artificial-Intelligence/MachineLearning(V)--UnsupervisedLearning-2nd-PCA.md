---
title: 机器学习(V)--无监督学习(二)主成分分析
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-unsupervised-learning.png
top_img: /img/artificial-intelligence.jpg
abbrlink: c929642b
date: 2024-06-15 13:36:00
description: 
---

当数据的维度很高时，很多机器学习问题变得相当困难，这种现象被称为**维度灾难**（curse of dimensionality）。

在很多实际的问题中，虽然训练数据是高维的，但是与学习任务相关也许仅仅是其中的一个低维子空间，也称为一个**低维嵌入**，例如：数据属性中存在噪声属性、相似属性或冗余属性等，对高维数据进行**降维**（dimension reduction）能在一定程度上达到提炼低维优质属性或降噪的效果。

常见的降维方法除了特征选择以外，还有维度变换，即将原始的高维特征空间映射到低维**子空间**（subspace），并尽量保证数据信息的完整性。常见的维度变换方法有 

- 主成分分析：（PCA）是一种无监督的降维方法，其目标通常是减少冗余信息和噪声。
- 因子分析：（FA）是找到当前特征向量的公因子，用公因子的线性组合来描述当前的特征向量。
- 线性判别分析：（LDA）本身也是一个分类模型，是为了让映射后的样本有最好的分类性能。
- 流行学习：（manifold learning）复杂的非线性方法。

# 主成分分析

**主成分分析**(Principal Component Analysis，PCA)是一种最常用的数据降维方法，这一方法利用一个正交变换，将原始空间中的样本投影到新的低维空间中。简单来说就是，PCA采用一组新的基来表示样本点，其中每一个基向量都是原来基向量的线性组合，通过使用尽可能少的新基向量来表出样本，从而达到降维的目的。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/PCA.png" style="zoom: 50%;" />

## 方差最大化
以往矩阵的样本矩阵 $X$ 中，我们用一行表示一个样本点 。由于向量通常是指列向量，所以在PCA相关的文献中，常常用$X$的每一列来表示一个样本，即给定样本矩阵

$$
X=\begin{pmatrix}\mathbf x_1&\mathbf x_2&\cdots&\mathbf x_N\end{pmatrix}=
\begin{pmatrix}
x_{11}&x_{12}&\cdots&x_{1N} \\
x_{21}&x_{22}&\cdots&x_{2N} \\
\vdots&\vdots&\ddots&\vdots \\
x_{p1}&x_{p2}&\cdots&x_{pN} \\
\end{pmatrix}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $j$ 个样本的特征向量为 $\mathbf x_j=(x_{1j},x_{2j},\cdots,x_{pj})^T$ 。

假定投影变换后得到的新坐标系变换矩阵为
$$
W_{p\times p}=\begin{pmatrix}\mathbf w_1&\cdots&\mathbf w_{p}\end{pmatrix}
$$
其中 $\mathbf w_j$ 是标准正交基向量，即 $\|\mathbf w_j\|_2=1,\ \mathbf w_i^T\mathbf w_j=0(i\neq j)$ 。样本点 $\mathbf x_j$ 在新坐标系中的样本
$$
\mathbf z_j=W^T\mathbf x_j
$$

正交变换后得到的样本矩阵
$$
Z=W^T X
$$

若要用一个超平面对空间中所有高维样本进行恰当的表达，那么它大概应具有这样的性质：

- **最近重构性**：样本点到超平面的距离足够近，即尽可能在超平面附近；
- **最大可分性**：样本点在超平面上的投影尽可能地分散开来，即投影后的坐标具有区分性。

有趣的是，最近重构性与最大可分性虽然从不同的角度出发，但最终得到了完全相同的优化函数。

在此，我们从最大可分性出发推导，样本在新空间每维特征上尽可能的分散，因此要求 $Z$​ 各行方差最大化。

新空间第 $i$ 行的特征向量 $\mathbf z_i=(z_{i1},\cdots,z_{ip})$ ，各行方差之和
$$
\frac{1}{N}\sum_{i=1}^p(\mathbf z_i-\bar z_i)(\mathbf z_i-\bar z_i)^T
=\frac{1}{N}\text{tr }(Z-\bar{\mathbf z})(Z-\bar{\mathbf z})^T
$$
新空间各特征的样本均值
$$
\bar{\mathbf z}=\frac{1}{N}\sum_{j=1}^N\mathbf z_j=\mathbf W^T\bar{\mathbf x}
$$
其中 $\bar{\mathbf x}$ 是原空间各特征的样本均值
$$
\bar{\mathbf x}=\frac{1}{N}\sum_{j=1}^N\mathbf x_j
$$
为方便求解，主成分分析中，首先对给定数据进行了中心化，即 $\Sigma_j\mathbf x_j=0$，所以新空间中各特征的样本均值 $\bar{\mathbf z}=0$。于是，转换后各行方差之和简化为

$$
\frac{1}{N}\text{tr}(ZZ^T)=\frac{1}{N}\text{tr}(W^TXX^TW)
$$
从最大可分性出发，应该使投影后样本点的方差最大化。同时 $\mathbf w_i$​ 是标准正交基，从而保证了转换后的特征相互独立，且系数唯一。于是优化目标可写为
$$
\max_W\text{tr}(W^TXX^TW) \\
\text{s.t.}\quad W^TW=I
$$

## 特征值分解

接着使用拉格朗日乘子法求解上面的优化问题，得到
$$
XX^TW=W\Lambda
$$
其中拉格朗日乘子 $\Lambda=\text{diag}(\lambda_1,\cdots,\lambda_p)$。上式可拆解为
$$
XX^T\mathbf w_i=\lambda_i\mathbf w_i\quad (i=1,2,\cdots,p)
$$
从上式可知，$\mathbf w_i$ 是协方差矩阵 $XX^T$ 的特征向量，$\lambda_i$ 为特征值。因此，主成分分析可以转换成一个矩阵特征值分解问题。我们的目标函数可变为
$$
\text{tr}(W^TXX^TW)=\text{tr}(W^TW\Lambda)=\text{tr}(\Lambda)=\sum_{i=1}^p\lambda_i
$$
可知，特征值 $\lambda$ 是投影后样本的方差。于是，要取得降维后的方差最大值，只需对协方差矩阵 $XX^T$ 进行特征值分解，将求得的特征值排序：$\lambda_1\geqslant\cdots\geqslant\lambda_p$ ，再取前 $p'$ 个特征值对应的特征向量构成变换矩阵
$$
W=\begin{pmatrix}\mathbf w_1&\cdots&\mathbf w_{p'}\end{pmatrix}
$$
于是，可得到的变换后的样本矩阵
$$
Z=W^T X
$$

这些变换后线性无关的特征称为**主成分**(Principal Components)。

> 实践中常通过对 $X$​ 进行奇异值分解来代替协方差矩阵的特征值分解。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/pca_of_iris.png" style="zoom: 75%;" />

## 主成分维数的选择

降维后低维空间的维数 $p'$ 通常是由用户事先指定，或通过在 $p'$ 值不同的低维空间中对 k 近邻分类器(或其他开销较小的学习器)进行交叉验证来选取较好的 $p'$值。对 PCA，还可从重构的角度设置一个重构阈值，例如 $t= 95\%$， 然后选取使下式成立的最小 $p'$ 值：
$$
\frac{\sum_{i=1}^{p'}\lambda_i}{\sum_{i=1}^p\lambda_i}\geqslant t
$$
主成分分析是一种无监督学习方法，可以作为监督学习的数据预处理方法，用来去除噪声并减少特征之间的相关性，但是它并不能保证投影后数据的类别。

## 优缺点总结

PCA算法的主要优点有：

- 仅仅需要以方差衡量信息量，不受数据集以外的因素影响。
- 各主成分之间正交，可消除原始数据成分间的相互影响的因素。
- 计算方法简单，主要运算是特征值分解，易于实现。

PCA算法的主要缺点有：

- 主成分各个特征维度的含义具有一定的模糊性，不如原始样本特征的解释性强。
- 方差小的非主成分也可能含有对样本差异的重要信息，因降维丢弃可能对后续数据处理有影响。

# KernelPCA

## 高维映射

在前面的讨论中，从高维空间到低维空间的函数映射是线性的，然而，在不少现实任务中，我们的样本数据点不是线性分布。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/original_S-curve_samples.png" style="zoom: 67%;" />

KernelPCA 算法其实很简单，只是将原始数据映射到高维空间，然后在这个高维空间进行PCA降维。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/KernelPCA_top_img.jpg" style="zoom: 50%;" />

给定样本矩阵

$$
X=\begin{pmatrix}\mathbf x_1&\cdots&\mathbf x_N\end{pmatrix}=
\begin{pmatrix}
x_{11}&\cdots&x_{1N} \\
x_{21}&\cdots&x_{2N} \\
\vdots&\ddots&\vdots \\
x_{p1}&\cdots&x_{pN} \\
\end{pmatrix}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $j$ 个样本的特征向量为 $\mathbf x_j=(x_{1j},x_{2j},\cdots,x_{pj})^T$ 。

将每个样本通过非线性函数 $\phi$ 映射到高维空间 $\R^q$，得到高维空间的数据矩阵
$$
\phi(X)=(\phi(\mathbf x_1),\cdots,\phi(\mathbf x_N))
$$
接下来在高维空间中对 $\phi(X)$ 进行PCA降维，在此之前，需要对其进行预处理。让我们暂时先假设矩阵 $\phi(X)$ 已经过中心化处理，即已有 $\sum_{j=1}^N\phi(\mathbf x_j)=0$。

> 因为 $\phi(X)$的中心化涉及到核函数的运算，比较复杂，本文仅简单介绍下 KernelPCA 的思路。

于是，要取得降维后的方差最大值，只需对高维空间中的协方差矩阵进行特征值分解，取最大的前 $p'$ 个特征值
$$
\phi(X)\phi(X)^TW=W\Lambda
$$
其中，$W$ 为变换矩阵，拉格朗日乘子 $\Lambda=\text{diag}(\lambda_1,\cdots,\lambda_p)$。然而，由于我们没有显式的定义映射$\phi$，所以上式无法直接求解，必须考虑换一种方法来求解这个特征值问题。

## 核函数

由于在对 $\phi(X)$ 进行PCA降维时，我们只需找较大的特征值对应的特征向量，而不需要计算0特征值对应的特征向量。因此，考虑当 $\lambda_i\neq 0$ 时的情况，特征值分解方程两边右乘 $\Lambda^{-1}$​
$$
W=\phi(X)\phi(X)^TW\Lambda^{-1}=\phi(X)C
$$
其中
$$
C=\phi(X)^TW\Lambda^{-1}
$$
将 $W$ 带回到特征值分解方程，得到
$$
\phi(X)\phi(X)^T\phi(X)C=\phi(X)C\Lambda
$$
进一步，等式两边都左乘矩阵 $\phi(X)^T$ 
$$
\phi(X)^T\phi(X)\phi(X)^T\phi(X)C=\phi(X)^T\phi(X)C\Lambda
$$
定义矩阵 
$$
K=\phi(X)^T\phi(X)=[\phi(\mathbf x_i)^T\phi(\mathbf x_j)]_{N\times N}
$$
则 $K$ 为 $N\times N$ 的对称半正定矩阵。上式简化为
$$
KC=C\Lambda
$$
为求解上述方程，我们需要求解以下特征值问题的非零特征值：
$$
K\mathbf c=\lambda\mathbf c
$$
求解上述问题涉及到计算 $\mathbf\phi(\mathbf x_i)^T\mathbf\phi(\mathbf x_j)$， 这是样本 $\mathbf x_i$ 与 $\mathbf x_j$ 映射到特征空间之后的内积。由于特征空间维数可能很高，甚至可能是无穷维，因此直接计算  $\mathbf\phi(\mathbf x_i)^T\mathbf\phi(\mathbf x_j)$ 通常是困难的。为了避开这个障碍，引入**核函数**（kernel function） 
$$
\kappa(\mathbf x_1,\mathbf x_2)=\mathbf\phi(\mathbf x_1)^T\mathbf\phi(\mathbf x_2)
$$
即  $\mathbf x_i$ 与 $\mathbf x_j$  在特征空间的内积等于它们在原始样本空间中通过核函数计算的结果，这称为**核技巧**（kernel trick）。核函数 $\kappa$ 的实现方法通常有比直接构建 $\mathbf\phi(\mathbf x)$ 再算点积高效很多。

于是，通过核函数计算矩阵 $K$​ ，从而进行特征值分解。将求得的特征值排序：$\lambda_1\geqslant\cdots\geqslant\lambda_N$ ，再取前 $p'$ 个特征值对应的特征向量构成矩阵
$$
C=\begin{pmatrix}\mathbf c_1&\cdots&\mathbf c_{p'}\end{pmatrix}
$$

> 注意，这里的特征向量 $\mathbf c_j$ 只是 $K$​​​ 的特征向量，不是高维空间中协方差矩阵的特征向量。

可以看到，矩阵 $K$ 需要计算所有样本间的核函数，因此该算法的计算开销十分大。

## 主成分计算

然而，至此仍然无法显式的计算变换矩阵 $W=\phi(X)C$。假设在新空间中使用标准正交基，即 $W^TW=I$
$$
W^TW=C^T\phi(X)^T\phi(X)C=C^TKC=C^TC\Lambda
$$
 因此，求出 $K$​ 的特征值和对应的特征向量后，令
$$
\mathbf c_j\gets \frac{\mathbf c_j}{\sqrt{\lambda_j}} 
$$
即可将相应的基向量归一化。

降维后的主成分矩阵
$$
Z=W^T\phi(X)=C^T\phi(X)^T\phi(X)=C^TK
$$
可以看到，主成分的计算也可由核函数的计算完成。

在PCA中，主成分的数量受特征数量的限制，而在KernelPCA中，主成分的数量受样本数量的限制。许多现实世界的数据集都有大量样本，在这些情况下，找到所有具有完整KernelPCA的成分是浪费计算时间，因为数据主要由前几个成分（例如n_components<=100）描述。换句话说，在KernelPCA拟合过程中特征分解的协方差矩阵的有效秩比其大小小得多。在这种情况下，近似特征求解器可以以非常低的精度损失提供加速。

# 因子分析

因子分析（Factor Analysis，FA）和PCA类似，也采用矩阵分解方法来降维。它通过寻找一组潜在的因子来解释已观测到变量间的关系。

## 基本形式

假设$p$维样本$\mathbf x=(x_1,x_2,\cdots,x_p)^T$ 可由几个相互独立的潜在因子（latent factor）线性近似
$$
\mathbf x=\mu+W\mathbf f+\mathbf e
$$

因子向量 $\mathbf f=(f_1,f_2,\cdots,f_k)^T, (k\leqslant p)$，$\mathbf f$的分量$f_i$称为**公共因子**（common factor）。随机向量$\mathbf e=(e_1,e_2,\cdots,e_p)^T$ ，$\mathbf e$的分量$e_i$称为**特殊因子**（specific factor）。矩阵$W=(w_{ij})_{p\times k}$ 称为**因子载荷**矩阵（factor loading），元素$w_{ij}$是$\mathbf x$的第$i$个分量$x_i$在第$j$个因子$f_j$上的载荷。

通常假设公共因子服从正态分布
$$
\mathbf f\sim N(0,\mathbf I) 
$$

且相互独立。为确保公共因子唯一，假定他们都是单位方差。

特殊因子服从正态分布
$$
\mathbf e\sim N(0,\Lambda), \quad\Lambda=\text{diag}(\sigma^2_1,\sigma^2_2,\cdots,\sigma^2_p)
$$

容易知道
$$
\mathbf x\sim N(\mu,WW^T+\Lambda)
$$

于是$var(x_i)=\sum_{j=1}^kw_{ij}^2+\sigma_i^2$ ，可以看出的$x_i$方差由两部分构成。记 $g_i^2=\sum_{j=1}^kw_{ij}^2$ ，反映了公共因子$f_j$对$x_i$的影响。

因为特殊因子的方差阵是对角阵，所以样本任意两个分量的协方差 $cov(x_s,x_t)=\sum_{i=1}^kw_{si}w_{ti}$ 仅由公共因子决定， 不含特殊因子影响。

## 参数估计

因子模型的未知参数是载荷矩阵$W$和特殊因子方差$\Lambda$，可以用主成分法、极大似然法、主因子法估计。