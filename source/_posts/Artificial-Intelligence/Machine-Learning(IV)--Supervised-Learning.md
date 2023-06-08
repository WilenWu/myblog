---
title: 机器学习(IV)--监督学习
date: 2022-11-27 21:40
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-supervised-learning.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 3c1747d9
description: 
---

# 引言

每个样本都有标签的机器学习称为监督学习。根据标签数值类型的不同，监督学习又可以分为回归问题和分类问题。分类和回归是监督学习的核心问题。

- **回归**(regression)问题中的标签是连续值。
- **分类**(regression)问题中的标签是离散值。分类问题根据其类别数量又可分为二分类（binary classification）和多分类（multi-class classification）问题。

# 线性回归

## 基本形式

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in \R$ 。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/linear-regression.svg" alt="线性回归" style="zoom:50%;" />

**Model**：线性模型假设目标变量是特征的线性组合。因此，我们试图拟合函数
$$
f_{\mathbf{w},b}(\mathbf{x})=w_1x_1+w_2x_2+\cdots+w_px_p+b=\sum_{j=1}^p w_jx_j+b
$$
称为多元线性回归 (multiple linear regression)。一般写作向量形式
$$
f_{\mathbf{w},b}(\mathbf{x})=\mathbf{w}^T\mathbf{x}+b
$$
特征向量 $\mathbf x=(x_1,x_2,\cdots,x_p)^T$，参数 $\mathbf{w}=(w_1,w_2,\cdots,w_p)^T$ 称为系数 (coefficients) 或权重 (weights)，标量 $b$ 称为偏置项(bias) 。求得参数 $\mathbf{w},b$ 后，模型就得以确定。$\mathbf w$ 可以直观表达了各特征在预测中的重要性，因此线性模型有很好的可解释性(comprehensibility) 。

为了计算方便，通常定义 $x_0=1,w_0=b$ 。线性回归模型可简写为
$$
f_{\mathbf{w}}(\mathbf{x}) = \mathbf{w}^T\mathbf{x}
$$
其中，特征向量 $\mathbf x=(x_0,x_1,x_2,\cdots,x_p)^T$，权重向量 $\mathbf{w}=(w_0,w_1,w_2,\cdots,w_p)^T$

## 最小二乘法

**普通最小二乘法** (ordinary least squares, OLS) 使用残差平方和来估计参数，使得数据的实际观测值和模型预测值尽可能接近。

**loss function** ：衡量单个样本预测值 $f_{\mathbf{w}}(\mathbf{x})$ 和真实值 $y$ 之间差异的
$$
\text{loss}=(f_{\mathbf{w}}(\mathbf{x})-y)^2
$$
**cost function** ：衡量样本集的差异 
$$
J(\mathbf{w}) = \frac{1}{2N} \sum\limits_{i=1}^{N} \left(f_{\mathbf{w}}(\mathbf{x}_i) - y_i\right)^2
$$
为了建立一个不会因训练集变大而变大的代价函数，我们计算均方误差而不是平方误差。额外的 1/2 是为了让后面的计算更简洁些。矩阵形式为
$$
J(\mathbf{w})=\cfrac{1}{2N}\|\mathbf{Xw-y}\|_2^2=\cfrac{1}{2N}(\mathbf{Xw-y})^T(\mathbf{Xw-y})
$$
其中，$\mathbf X$ 称为**设计矩阵**（design matrix）
$$
\mathbf{X}=\begin{pmatrix}
1&x_{11}&x_{12}&\cdots&x_{1p} \\
1&x_{21}&x_{22}&\cdots&x_{2p} \\
\vdots&\vdots&\ddots&\vdots \\
1&x_{N1}&x_{N2}&\cdots&x_{Np} \\
\end{pmatrix},
\quad \mathbf{w}=\begin{pmatrix}w_0\\ w_1\\ \vdots\\w_p\end{pmatrix},
\quad \mathbf{y}=\begin{pmatrix}y_1\\ y_2\\ \vdots\\y_N\end{pmatrix}
$$
最后，模型参数估计等价于求解
$$
\arg\min\limits_{\mathbf w} J(\mathbf{w})
$$

**参数估计** ：(parameter estimation) 由于 $J(\mathbf w)$ 为凸函数，且 Hessian 矩阵 $\nabla^2 J(\mathbf w)$ 正定。可以使用凸优化方法求解
$$
\nabla J(\mathbf w)=\frac{\partial J}{\partial\mathbf w}=2\mathbf X^T(\mathbf{Xw-y})=\mathbf 0
$$
此方程被称为**正规方程**（normal equation）。如果 $\mathbf X^T \mathbf X$ 为满秩矩阵(full-rank matrix)或正定矩阵(positive definite matrix)，则可求得最优解
$$
\mathbf w^*=(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y
$$

**最小二乘法的特点** 

现实任务中 $\mathbf X^T\mathbf X$ 可能不可逆，原因如下

- 特征之间可能线性相关
- 特征数量大于样本总数 ($p>N$)

最小二乘法的优点

- 求解参数不需要迭代计算
- 不需要调试超参数

最小二乘法的缺点

- 仅适用于线性回归，无法推广到其他学习算法
- 假设 $N\geqslant p$ ，这个算法的复杂度为 $O(Np^2)$
- 如果样本特征的数量太大 (>10k)，模型将执行的非常慢

## 极大似然估计

对于线性回归来说，也可以假设其为以下模型
$$
f_{\mathbf{w}}(\mathbf{x}) = \mathbf{w}^T\mathbf{x}+e
$$
其中，特征向量 $\mathbf x=(x_0,x_1,x_2,\cdots,x_p)^T$，权重向量 $\mathbf{w}=(w_0,w_1,w_2,\cdots,w_p)^T$ 。$e$ 为随机误差，通常假设其服从正态分布 $e∼\mathcal N(0, σ^2)$ 。所以的概率密度函数为
$$
\mathbb P(e)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{e^2}{2\sigma^2})
$$
且样本间的误差相互独立 $\mathrm{cov}(e_i,e_j)=0$

**极大似然估计**：(maximum likelihood estimate, MLE) 使得样本误差的联合概率（也称似然函数）取得最大值。为求解方便，对样本联合概率取对数似然函数
$$
\begin{aligned}
\ln L(\mathbf w) &=\ln\prod_{i=1}^N\mathbb P(e_i)=\sum_{i=1}^N\ln\mathbb P(e_i) \\
&=\sum_{i=1}^N\ln\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(f_{\mathbf{w}}(\mathbf{x}_i) - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2})  \\
&=N\ln\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{2\sigma^2}\sum_{i=1}^N(f_{\mathbf{w}}(\mathbf{x}_i) - \mathbf{w}^T\mathbf{x}_i)^2 \\
\end{aligned}
$$
最后，最大化对数似然函数等价于求解
$$
\arg\max\limits_{\mathbf w} \ln L(\mathbf{w})=\arg\min\limits_{\mathbf w} \sum_{i=1}^N(f_{\mathbf{w}}(\mathbf{x}_i) - \mathbf{w}^T\mathbf{x}_i)^2
$$

上式与最小二乘法等价。

## 正则化

**Ridge** (岭回归) 通过引入 $\ell_2$ 范数正则化(regularization) 项来解决普通最小二乘的过拟合问题

Cost function
$$
J(\mathbf{w})=\cfrac{1}{2N}\|\mathbf{Xw-y}\|_2^2+ \alpha \|\mathbf w\|^2
$$
其中，正则化参数 $\alpha>0$ 通过缩小特征权重来控制模型复杂度，值越大，收缩量越大，这样，系数对共线性的鲁棒性就更强了。

最小化代价函数可求得解析解
$$
\mathbf{w=(X^T X}+\alpha \mathbf{I)^{-1}X^T y}
$$
其中 $\mathbf I$ 是 $p+1$ 维单位阵。利用$\ell_2$ 范数进行正则化不仅可以抑制过拟合，同时叶避免了 $\mathbf{X^T X}$ 不可逆的问题。

**Lasso** (Least Absolute Shrinkage and Selection Operator) 是一个估计稀疏系数的线性模型。它在某些情况下是有用的，因为它倾向于给出非零系数较少的解，从而有效地减少了给定解所依赖的特征数。 它由一个带有 $\ell_1$ 范数正则项的线性模型组成。

Cost function
$$
J(\mathbf{w})=\cfrac{1}{2N}\left(\|\mathbf{Xw-y}\|_2^2+ \alpha \|\mathbf w\|_1\right)
$$
Lasso 中一般采用坐标下降法来实现参数估计。由于Lasso回归产生稀疏模型，因此也可以用来进行特征选择。

**Elastic-Net** 是一个训练时同时用 $\ell_1$ 和  $\ell_2$ 范数进行正则化的线性回归模型。这种组合允许学习稀疏模型，其中很少有权重是非零类。当多个特征存在相关时，弹性网是很有用的。Lasso很可能随机挑选其中之一，而弹性网则可能兼而有之。在这种情况下，要最小化的目标函数

Cost function
$$
J(\mathbf{w})=\cfrac{1}{2N}\left(\|\mathbf{Xw-y}\|_2^2+ \alpha\rho \|\mathbf w\|_1+ \frac{\alpha(1-\rho)}{2} \|\mathbf w\|_2^2\right)
$$

Elastic-Net 使用坐标下降法来估计参数。

## 广义线性回归

线性模型往往不能很好地拟合数据，我们可以在线性方程后面引入一个非线性变换，拟合许多功能更为强大的非线性模型(non-linear model)。例如，对数线性回归 (log-linear regression)
$$
\ln f_{\mathbf{w},b}(\mathbf{x})=\mathbf{w}^T\mathbf{x}+b
$$
这个非线性变换称为激活函数(activation function)。更一般地，考虑激活函数 $y=g(z)$，令 
$$
f_{\mathbf{w},b}(\mathbf{x})=g(\mathbf{w}^T\mathbf{x}+b)
$$
这样得到的模型称为广义线性模型 (Generalized Linear Models, GLM)，激活函数的反函数 $z=g^{-1}(y) $ 称为联系函数 (link function)。广义线性模型的参数估计常通过加权最小二乘法或极大似然估计。

## 多项式回归

为了更好的拟合数据，机器学习中一个常见模式是使用非线性函数对数据进行变换来创建新的特征。例如，可以通过构造**多项式特征**(**polynomial features**)来扩展简单的线性回归。在标准线性回归的情况下，您可能拟合一个二维特征的模型
$$
f_{\mathbf w}(\mathbf x)=w_0+w_1x_1+w_2x_2
$$
如果我们想用抛物面来拟合数据而不是平面，我们可以用二阶多项式组合特征，这样模型看起来就像这样
$$
f_{\mathbf w}(\mathbf x)=w_0+w_1x_1+w_2x_2+w_3x_1x_2+w_4x_1^2+w_5x_2^2
$$
其实，得到的多项式回归依旧是线性模型：只需引入新的特征向量进行转换
$$
\begin{matrix}
\text{from}&x_1&x_2&x_1x_2&x_1^2&x_2^2 \\
\to&z_1&z_2&z_3&z_4&z_5
\end{matrix}
$$
下面是一个应用于一维数据的例子，使用了不同程度的多项式特征：

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/polynomial-features.svg" style="zoom: 67%;" />

# 线性分类

## Logistic 回归

### 基本形式

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in \{0,1\}$ 。逻辑回归试图预测正样本的概率，那我们需要一个输出 $[0,1]$ 区间的激活函数。假设二分类数据集不同类别的特征值服从均值不同、方差相同的正态分布
$$
\begin{cases}
\mathbb P(\mathbf x|y=1)∼\mathcal N(\mathbf \mu_1, \mathbf\Sigma) \\
\mathbb P(\mathbf x|y=0)∼\mathcal N(\mathbf \mu_0, \mathbf\Sigma)
\end{cases}
$$
其中，协方差矩阵$\mathbf\Sigma$ 为对称阵。利用贝叶斯定理，正样本条件概率
$$
\mathbb P(y=1|\mathbf x)=\frac{\mathbb P(\mathbf x|y=1)\mathbb P(y=1)}{\mathbb P(\mathbf x|y=0)\mathbb P(y=0)+\mathbb P(\mathbf x|y=1)\mathbb P(y=1)}
$$
令 
$$
\begin{aligned}
z&=\ln\frac{\mathbb P(\mathbf x|y=1)\mathbb P(y=1)}{\mathbb P(\mathbf x|y=0)\mathbb P(y=0)} \\
&=\frac{1}{2}[(\mathbf x-\mathbf μ_0)^T\mathbf\Sigma^{-1}(\mathbf x-\mathbf μ_0)-(\mathbf x-\mathbf μ_1)^T\mathbf\Sigma^{-1}(\mathbf x-\mathbf μ_1)]+\ln\frac{\mathbb P(y=1)}{\mathbb P(y=0)} \\
&=(\mathbf μ_1-\mathbf μ_0)^T\mathbf\Sigma^{-1}\mathbf x+\ln\frac{\mathbb P(y=1)}{\mathbb P(y=0)}
\end{aligned}
$$
由于 $\mathbb P(y=1)$ 和 $\mathbb P(y=0)$ 是先验概率为常数，上式可简化为
$$
z=\mathbf w^T\mathbf x+b
$$
于是
$$
\mathbb P(y=1|\mathbf x)=\frac{1}{1+e^{-z}}
$$
上式称为 Sigmoid 函数（S型曲线），也称 logistic 函数。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/logistic-function.png" style="zoom: 50%;" />

**Model**: 逻辑回归 (logistic regression, logit regression) 通过引入Sigmod 函数将输入值映射到 $[0,1]$ 来实现分类功能。
$$
f_{\mathbf{w},b}(\mathbf{x}) = g(\mathbf{w}^T \mathbf{x}+b)
$$
其中
$$
g(z) = \frac{1}{1+e^{-z}}
$$
式中特征向量 $\mathbf x=(x_1,x_2,\cdots,x_p)^T$，参数 $\mathbf{w}=(w_1,w_2,\cdots,w_p)^T$ 称为系数 (coefficients) 或权重 (weights)，标量 $b$ 称为偏置项(bias) 。

为计算方便，引入 $x_0=1,w_0=b$ 。模型简写为
$$
f_{\mathbf{w}}(\mathbf{x}) = \frac{1}{1+\exp(-\mathbf{w}^T\mathbf{x})}
$$
其中，特征向量 $\mathbf x=(x_0,x_1,x_2,\cdots,x_p)^T$，权重向量 $\mathbf{w}=(w_0,w_1,w_2,\cdots,w_p)^T$

可以通过引入阈值（默认0.5）实现分类预测
$$
\hat y=\begin{cases}
1 &\text{if } f_{\mathbf{w}}(\mathbf{x})\geqslant 0.5 \\
0 &\text{if } f_{\mathbf{w}}(\mathbf{x})<0.5
\end{cases}
$$
模型的输出为正样本的概率
$$
\begin{cases}
\mathbb P(y=1|\mathbf x)=f_{\mathbf{w}}(\mathbf{x}) \\
\mathbb P(y=0|\mathbf x)=1-f_{\mathbf{w}}(\mathbf{x}) 
\end{cases}
$$

可简记为
$$
\mathbb P(y|\mathbf x)=[f_{\mathbf{w}}(\mathbf{x})]^{y}[1-f_{\mathbf{w}}(\mathbf{x})]^{1-y}
$$

### 极大似然估计

logistic 回归若采用均方误差作为 cost function，是一个非凸函数(non-convex)，会存在许多局部极小值，因此我们尝试极大似然估计。

**极大似然估计**：(maximum likelihood estimate, MLE)  使得观测样本出现的概率最大，也即使得样本联合概率（也称似然函数）取得最大值。

为求解方便，对样本联合概率取对数似然函数
$$
\begin{aligned}
\log L(\mathbf w) & =\log\prod_{i=1}^{N} \mathbb P(y_i|\mathbf x_i)=\sum_{i=1}^N\log \mathbb P(y_i|\mathbf x_i) \\
&=\sum_{i=1}^{N}[y_i\log f_{\mathbf{w}}(\mathbf{x}_i)+(1-y_i)\log(1-f_{\mathbf{w}}(\mathbf{x}_i))] 
\end{aligned}
$$
因此，可定义 **loss function**  
$$
\begin{aligned}
\text{loss}&=-y\log f_{\mathbf{w}}(\mathbf{x})-(1-y)\log(1-f_{\mathbf{w}}(\mathbf{x})) \\
&=-y\mathbf{w}^T\mathbf{x}+\log(1+e^{\mathbf{w}^T\mathbf{x}})
\end{aligned}
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/loss-logistic-regression.png)
最大化似然函数等价于最小化 **cost function**
$$
J(\mathbf w)=\frac{1}{N}\sum_{i=1}^{N}(-y_i\mathbf{w}^T\mathbf{x}+\log(1+e^{\mathbf{w}^T\mathbf{x}}))
$$
**参数估计** ：(parameter estimation) $J(\mathbf w)$ 是关于参数 $\mathbf w$ 的高阶可导连续凸函数，经典的数值优化算法如梯度下降法 (gradient descent method) 、牛顿法 (Newton method) 等都可求得其最优解
$$
\arg\min\limits_{\mathbf w} J(\mathbf{w})
$$

### 决策边界

**决策边界**：逻辑回归模型 $f_{\mathbf{w},b}(\mathbf{x})=g(z)= g(\mathbf{w}^T \mathbf{x}+b)$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/sigmoid-decision-boundary.png" style="zoom: 50%;" />

在 logistic 回归模型中，$z=\mathbf{w}^T\mathbf{x}+b$ 。对于 sigmoid 函数（如上图），$g(z)\geqslant 0.5 \text{ for } z\geqslant 0$ 。因此，模型预测
$$
\hat y=\begin{cases}
1 &\text{if } \mathbf{w}^T\mathbf{x}+b\geqslant 0 \\
0 &\text{if } \mathbf{w}^T\mathbf{x}+b<0
\end{cases}
$$
由此可见，logistic 回归输出一个线性决策边界 (linear decision boundary) 
$$
\mathbf{w}^T\mathbf{x}+b=0
$$
我们也可以创建多项式特征拟合一个非线性边界。例如，模型 
$f(x_1,x_2) = g(x_1^2+x_2^2-36)\text{ where } g(z) = \cfrac{1}{1+e^{-z}}$ 
决策边界方程为 $x_1^2+x_2^2-36=0$

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/DecisionBoundary.svg)

## 感知机

感知机（Perceptron）是线性二分类模型，适用于线性可分的数据集。

**Model**：感知机选取符号函数为激活函数
$$
f_{\mathbf{w},b}(\mathbf{x})=\text{sign}(\mathbf{w}^T\mathbf{x}+b)
$$
这样就可以将线性回归的结果映射到两分类的结果上了。符号函数 
$$
\text{sign}(z)=\begin{cases}+1 & \text{if }z\geqslant 0\\ -1 & \text{if }z<0\end{cases}
$$
为计算方便，引入 $x_0=1,w_0=b$ 。模型简写为
$$
f_{\mathbf{w}}(\mathbf{x}) = \text{sign}(\mathbf{w}^T\mathbf{x})
$$
其中，特征向量 $\mathbf x=(x_0,x_1,x_2,\cdots,x_p)^T$，权重向量 $\mathbf{w}=(w_0,w_1,w_2,\cdots,w_p)^T$

cost function：误分类点到分离超平面的总距离
$$
J(\mathbf w)=-\sum_{\mathbf x_i\in M}y_i\mathbf{w}^T\mathbf{x}_i
$$
其中，$M$ 是错误分类集合。

基于梯度下降法对代价函数的最优化算法，有原始形式和对偶形式。算法简单且易于实现。

损失函数的梯度
$$
\nabla J(\mathbf w)=-\sum_{\mathbf x_i\in M}y_i\mathbf{x}_i
$$


感知机有无穷多个解，其解由于不同的初始值或不同的迭代顺序而有所不同。

Perceptron 是另一种适用于大规模学习的简单分类算法。

- 它不需要设置学习率
- 它不需要正则项
- 它只用错误样本更新模型

最后一个特点意味着Perceptron的训练速度略快于带有合页损失(hinge loss)的SGD，因此得到的模型更稀疏。

**被动感知算法** (Passive Aggressive Algorithms) 是一种大规模学习的算法。和感知机相似，因为它们不需要设置学习率。然而，与感知器不同的是，它们包含正则化参数。

## 线性判别分析

线性判别分析（Linear Discriminant Analysis，LDA）亦称 Fisher 判别分析。其基本思想是：将训练样本投影到一条直线上，使得同类的样例尽可能近，不同类的样例尽可能远。如图所示：

## 最近邻算法

KNN

# 决策树

## 树的生成

**决策树**（Decision Tree）是一种用于分类和回归的有监督学习方法。其目标是创建一个模型，通过学习从数据特性中归纳出一组分类规则来预测目标变量的值。下图是一颗决策树

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/decision-tree.png" alt="decision-tree" style="zoom:50%;" />

决策树是一种由**节点**（node）和**有向边**（directed edge）组成的树形结构。从**根节点**（root node）开始，包含若干**内部节点**（internal node）和**叶节点**（leaf node）。其中每个叶节点对应一种分类结果，其他每个节点表示一个特征的判断条件，每个分支代表一个判断结果的输出。

其实决策树可以看做一个if-then规则的集合。我们从决策树的根结点到每一个都叶结点构建一条规则，并且我们将要预测的实例都可以被一条路径或者一条规则所覆盖。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/decision-tree-nodes.svg)

**Hunt 算法**：决策树学习旨在构建一个泛化能力好，并且复杂度小的决策树。因为从可能的决策树中直接选取最优决策树是 NP 完全问题，可构造的决策树的数目达指数级，找出最佳决策树在计算上时不可行的。现实中采用启发式方法，在合理的时间学习一颗次优的决策树。

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in\{c_1,c_2,\cdots,c_K\}$ ，有 $K$ 个类别。

Hunt 算法以递归方式建立决策树，使得各分支结点所包含的样本尽可能属于同一类别。设节点 $t$ 处的数据集为 $D_t$ ，样本量为 $N_t$ 。决策树的生成流程如下：

1. 在根节点从所有训练样本开始；
2. 在节点 $t$ 处，选择一个特征 $x_t$ 将数据集 $D_t$ 划分成更小的子集；
3. 对于每个子节点，递归的调用此算法，只到满足停止条件。

从上述步骤可以看出，决策树生成过程中有两个重要的问题

- 如何选择最优的划分特征：常用的算法有 ID3、C4.5 和 CART
- 什么时候停止划分：
  - 当一个节点100%是一个类别时；
  - 当分裂一个节点导致树超过最大深度时 (maximum depth)；
  - 如果分裂一个节点导致的纯度提升低于阈值；
  - 如果一个节点的样本数低于阈值。

> 限制决策树深度和设置阈值的一个原因是通过保持树的小巧而不容易导致过拟合

**决策树的特点**

决策树的一些优点：

- 决策树是一种非参数模型。换句话说，它不要求任何先验假设，不假定类和特征服从一定概率分布。
- 决策树可以被可视化，简单直观。
- 对于异常点的容错能力好，健壮性高。

决策树的缺点包括：

- 决策树算法非常容易过拟合，导致泛化能力不强，可以通过剪枝改进。
- 决策树可能是不稳定的。事实证明，只需要改变极少量训练样本，信息增益最大的特征就可能发生改变，会生成一颗完全不同的树。可以通过集成学习来缓解这个问题。
- 寻找最优的决策树是一个NP完全问题，我们一般是通过启发式算法（如贪婪算法），容易陷入局部最优。可以通过集成学习之类的方法来改善。
- 有些比较复杂的边界关系，决策树很难学习。

## 特征二元化

**连续特征离散化**：待划分的特征分为离散型和连续型两种。对于离散型的特征，按照特征值进行划分，每个特征值对应一个子节点；对于连续型的数据，由于可取值数目不再有限，一般需要离散化，常用二分法处理。

假定第 $j$ 个特征 $x^{(j)}$ 是连续变量，若样本中 $x$ 有 $K$ 个值，选取这些值的 $K-1$ 个中点值作为候选切分值。定义候选值是 $s$ 切分的两个区域
$$
R_1(j,s)=\{(\mathbf x,y)|x^{(j)}\leqslant s\} \quad \text{and}\quad R_2(j,s)=\{(\mathbf x,y)|x^{(j)}> s\}
$$
以基尼指数为例，求解最优切分值
$$
\arg\min_{s}[w_1\text{Gini}(R_1(j,s))+w_2\text{Gini}(R_2(j,s))]
$$
其中，$w_1,w_2$ 是 区域 $R_i,R_2$ 的样本数占比。

然后，我们就可以像离散特征一样来使用。需注意的是，与离散特征不同，若当前结点为连续特征，该特征还可作为其后代结点的划分特征。

**one-hot encoding**：某些算法（CART）只产生二元划分。如果一个离散特征可以取 $K$ 个值，可以通过创建 $K$ 个取值为0或1的二元特征来替换。如下图示例

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/one-hot-encoding.png" alt="one-hot-encoding" style="zoom: 50%;" />

## 划分特征选择

显然，决策树学习的关键在于划分数据集，我们希望不断地选取局部最优的特征，将无序的数据变得更加有序，即结点的**纯度** (purity) 越来越高。由于纯度的度量方法不同，也就导致了学习算法的不同，常用的算法有 ID3 、C4.5和 CART。

### 信息增益

**信息熵**（information entropy）是度量数据集纯度的最常用的指标。给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in\{c_1,c_2,\cdots,c_K\}$ ，有 $K$ 个类别。经验分布为
$$
\mathbb P(y=c_k)=P_k
$$
则信息熵为
$$
H(D)=-\sum_{k=1}^KP_k\log P_k
$$
注意，计算信息熵时约定 $0\log 0 = 0$。由定义可知，熵只依赖于 $y$ 的分布，与取值无关，所以也可将熵记作 $H(P)$ 。

对于二分类问题，目标变量 $y_i\in \{0,1\}$ 。正样本比例为 $P_1\ (0\leqslant P_1\leqslant 1)$ ，则负样本比例 $P_0=1-P_1$ 。信息熵可写为
$$
H(P_1)=-P_1\log P_1-(1-P_1)\log (1-P_1)
$$
二元变量的熵曲线如下图

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/information_entropy.svg" style="zoom:67%;" />

**条件熵**（condition entropy）用来表示离散特征 $x$ 划分后的数据集 $D$ 纯度。使用划分后子集的熵的加权平均值来度量
$$
H(D|x)=\sum_{m=1}^M w_mH(D_m)
$$
其中，离散特征值 $x$ 有 $M$ 个值。 $w_m=N_m/N$ 代表离散特征 $x$ 划分后的子集 $D_m$ 的样本数占比， $H(D_m)$ 代表子集$D_m$的信息熵。条件熵一般小于熵，例如，知道西瓜的色泽（青绿,乌黑,浅白）后，西瓜质量的不确定性就会减少了。

**信息增益**（Information Gain）表示使用特征 $x$ 的信息进行划分而使数据集 $D$ 纯度提升的程度
$$
\text{Gain}(D,x)=H(D)-H(D|x)
$$

以二元离散特征 $x$ 为例，将二分类数据集 $D$ 划分为 $D^{\text{left}}$和 $D^{\text{left}}$ 两个子集，则信息增益为
$$
\text{Gain}(D,x)=H(P_1)-\left(w^{\text{left}}H(P_1^{\text{left}})+w^{\text{right}}H(P_1^{\text{right}})\right)
$$
其中 $P_1$ 表示子集中正样本的比例，$w$ 表示子集的样本数占比。

**ID3**（Iterative Dichotomiser 3, 迭代二分器 3）算法在迭代中选取信息增益最大的特征进行划分
$$
\arg\max\limits_{x}\text{Gain}(D,x)
$$
其中特征 $x\in\{x_i,x_2,\cdots,x_p\}$ 。对于所有的节点来说，节点处数据集的熵是个不变的值，所以最大化信息增益等价于最小化条件熵。

以吴恩达老师的==猫分类数据集==为例：

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/cat-classification-example.png" alt="cat-classification-example" style="zoom:60%;" />

根节点的熵为：$H(P_1^{\text{root}})=H(0.5)=-\cfrac{1}{2}\log \cfrac{1}{2}-\cfrac{1}{2}\log \cfrac{1}{2}=1$

然后，计算各特征的信息增益：

Ear shape: $H(0.5)-(\cfrac{3}{10}H(0.67)+\cfrac{4}{10}H(0.75)+\cfrac{3}{10}H(0))=0.4$
Face shape: $H(0.5)-(\cfrac{7}{10}H(0.57)+\cfrac{3}{10}H(0.33))=0.03$
Whiskers: $H(0.5)-(\cfrac{4}{10}H(0.75)+\cfrac{6}{10}H(0.33))=0.12$
Weight: $H(0.5)-(\cfrac{4}{10}H(1)+\cfrac{6}{10}H(0.17))=0.61$

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/information-gain.svg)

显然，Weight ⩽ 9 的信息增益最大，于是Weight被选为在根节点划分的特征。类似的，再对每个分支结点进行上述操作，进一步划分，最终得到整颗决策树。 

### 信息增益率

从信息增益的公式中，其实可以看出，信息增益准则偏向取值较多的特征。原因是当特征的取值较多时，根据此特征划分更容易得到纯度更高的子集，因此划分之后的条件熵更低。为减少这种偏好可能带来的不利影响，C4.5通过**信息增益率**（Information Gain Rate）来选择最优划分特征
$$
\text{Gain\_rate}(D,x)=\frac{\text{Gain}(D,x)}{\text{IV}(x)}
$$

其中 $\text{IV}(x)$ 称为特征 $x$ 的固有值（intrinsic value）
$$
\text{IV}(x)=-\sum_{m=1}^Mw_m\log w_m
$$
其中，离散特征值 $x$ 有 $M$ 个值。 $w_m=N_m/N$ 代表离散特征 $x$ 划分后的子集 $D_m$ 的样本数占比。$\text{IV}(x)$ 可看作数据集 $D$ 关于 $x$ 的信息熵，特征 $x$ 的取值越多，通常 $\text{IV}(x)$ 越大。

需注意的是，信息增益率准对可取值数目较少的特征有所偏好。因此， C4.5算法并不是直接选择增益率最大的特征划分，而是使用了一个启发式：先从候选特征中找出信息增益高于平均水平的特征，再从中选择增益率最高的划分。

### 基尼指数

**基尼指数**（Gini Index）给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in\{c_1,c_2,\cdots,c_K\}$ ，有 $K$ 个类别。经验分布为
$$
\mathbb P(y=c_k)=P_k
$$
基尼指数可表示数据集 $D$ 的纯度
$$
\text{Gini}(D)=\sum_{k=1}^KP_k(1-P_k)=1-\sum_{k=1}^KP_k^2
$$

直观来说，基尼指数反应了从数据集中随机抽取两个样本，其类别不一致的概率。因此，基尼指数越小，则数据集的纯度越高。

对于二分类问题，目标变量 $y_i\in \{0,1\}$ 正样本比例为 $P_1\ (0\leqslant P_1\leqslant 1)$ ，则负样本比例 $P_0=1-P_1$ 。二分类变量的基尼指数可写为
$$
\text{Gini}(P_1)=2P_1(1-P_1)
$$
数据集 $D$ 在离散特征 $x$ 划分后的基尼指数定义为
$$
\text{Gini}(D,x)=\sum_{m=1}^Mw_m\text{Gini}(D_m)
$$
可理解为划分后子集基尼指数的加权平均值。其中，离散特征 $x$ 有$M$ 个值， $w_m=N_m/M$ 代表离散特征 $x$ 划分后的子集 $D_m$ 的样本数占比， $\text{Gini}(D_m)$ 代表子集$D_m$的基尼指数。

**CART**（Classification and Regression Trees）是使用划分后基尼指数最小的特征作为最优划分特征
$$
\arg\min\limits_{x}\text{Gini}(D,x)
$$
同时，CART使用**二叉树**准则减少对取值较多特征的偏向，并且可以分类也可以回归，也提供了优化的剪枝策略。

## 回归树

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in\R$ 。

假设回归树将特征空间划分为 $J$ 个互不相交的区域 $R_1,R_2,\cdots,R_J$ ，每个区域 $R_j$ 对应树的一个叶结点，并且在每个叶节点上有个固定的输出值 $c_j$ 。规则为
$$
\mathbf x\in R_j \implies T(\mathbf x)=c_j
$$
那么树可以表示为
$$
T(\mathbf x;\Theta)=\sum_{j=1}^Jc_j\mathbb I(\mathbf x\in R_j)
$$
参数 $\Theta=\{(R_1,c_1),(R_2,c_2),\cdots,(R_J,c_J)\}$ 表示树的区域划分和对应的值，$J$ 表示树的复杂度（即叶节点的个数）。

回归树使用平方误差最小的值作为最优输出值。易知，区域 $R_j$ 上的最优值 $c_j$ 对应此区域上所有目标变量 $y_i$ 的平均值
$$
c_j=\bar y,\quad  y_i\in R_j
$$
回归树使用**加权均方误差**选择最优划分特征。由于输出值 $c_j$ 为区域 $R_j$ 目标变量的平均值，所以区域 $R_j$ 的均方误差等价于方差。设节点 $t$ 处的区域数据集为 $D_t$ ，样本数为 $N_t$ ，则划分特征为
$$
\arg\min\limits_{x}\sum_{m=1}^Mw_{tm}\text{var}(D_{tm})
$$
其中，离散特征 $x$ 有$M$ 个值。$D_{tm}$ 为节点 $t$ 处特征 $x$ 划分的子集，$w_{tm}=N_{tm}/N_t$ 为子集的样本数占比。

## 剪枝处理

递归生成的决策树往往过于复杂，从而过拟合。对决策树进行简化的过程称为**剪枝**（pruning），剪枝的基本策略有预剪枝和后剪枝。剪枝过程中一般使用验证集评估决策树泛化能力的提升。

决策树学习的**代价函数**定义为
$$
C_\alpha(T)=C(T)+\alpha|T|
$$
其中，$|T|$ 是决策树 $T$ 中叶节点个数，$\alpha$ 是平衡树的复杂度和不纯度的超参数。$C(T)$ 是叶节点不纯度的加权平均值。以基尼指数为例，给定数据集 $D$ ，样本数为 $N$ ，则
$$
C(T)=\sum_tw_t\text{Gini}(D_t)
$$
其中，$D_t$ 为叶节点 $t$ 处的数据集，$w_t=N_t/N$ 为叶节点 $t$ 处的样本数占比。

**预剪枝**：（pre-pruning）是指在决策树生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前结点标记为叶结点。

预剪枝使得决策树的很多分支都没有展开，限制减少了决策树的时间开销，同时也给预剪枝决策树带来了欠拟含的风险。

**后剪枝**：（post-pruning）先从训练集生成一棵完整的决策树，然后剪掉一些叶结点或叶结点以上的子树，若能带来决策树泛化性能提升，则将其父结点作为新的叶结点，从而递归的简化生成的决策树。

一般情形下，后剪枝决策树的欠拟合风险很小，泛化性能往往优于预剪枝决策树，但决策树的时间开销要大得多。

**CART 剪枝**：首先从生成的决策树 $T_0$ 底端开始不断剪枝，直到根节点，形成一个子树序列  $\{T_0,T_1,\cdots,T_n\}$。然后通过交叉验证法选择最优子树。

## 决策边界

若我们把每个特征视为坐标空间中的一个坐标轴，则每个样本对应一个数据点，两个不同类之间的边界称为决策边界（decision boundary）。决策树所形成的分类边界有一个明显的特点：由于节点测试只涉及单个特征，它的决策边界由若干个与坐标轴平行的分段组成。这就限制了决策树对连续特征之间复杂关系的建模能力。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/DecisionBoundary2.svg)

**斜决策树**（oblique decision tree）在每个节点，不再是仅对某个特征，而是对特征的线性组合进行测试
$$
\sum_{j=1}^pw_jx_j+b=0
$$
尽管这种技术有更强的表达能力，并且能够产生更紧凑的决策树，但为找出最佳测试条件的计算可能相当复杂。

# 支持向量机

**支持向量机**（support vector machine, SVM）是一种用于分类、回归和异常检测的有监督学习方法。按照数据集的特点，可分为3类：

1. 当数据集线性可分时，通过硬间隔最大化（不容许错误分类），学习线性可分支持向量机，又叫硬间隔支持向量机（hard-margin SVM）
2. 当数据集近似线性可分时，通过软间隔最大化（容许一定误差），学习线性支持向量机（linear SVM），又叫软间隔支持向量机（soft-margin SVM）
3. 当数据集线性不可分时，通过核技巧（kernel method）及软间隔最大化，学习非线性支持向量机（non-linear SVM）

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in\{-1,+1\}$ 。

## 间隔最大化

本节讨论线性可分的数据集，即存在无数超平面，可以将正样本和负样本正确分类。从几何角度，具有最大间隔的超平面有更好的泛化性能，因为该超平面对训练样本局部扰动的容忍性最好。SVM 采用**最大间隔超平面**（maximal margin hyperplane）作为决策边界，此时的超平面是存在且唯一的。

分离的超平面可以写为如下形式
$$
\mathbf{w}^T\mathbf{x}+b=0
$$
其中 $\mathbf w=(w_1,w_2,\cdots,w_p)^T$ 为法向量，决定了超平面的方向；$b$ 为位移项，决定了超平面与原点之间的距离。显然，超平面可被法向量 $\mathbf w$ 和位移 $b$ 确定，下面我们将其记为 $(\mathbf w,b)$ 。

**Model**：
$$
f_{\mathbf{w},b}(\mathbf{x})=\text{sign}(\mathbf{w}^T\mathbf{x}+b)
$$
符号函数 
$$
\text{sign}(z)=\begin{cases}+1 & \text{if }z\geqslant 0\\ -1 & \text{if }z<0\end{cases}
$$
**最大间隔**：

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/SVM-maximum-margin.svg" style="zoom:80%;" />

距离超平面 $(\mathbf w,b)$ 最近的样本，被称为**支持向量** （support vector）。如上图，设超平面两侧平行边界的方程分别为
$$
\begin{aligned}
B_1: & \mathbf{w}^T\mathbf{x}+b=\gamma \\
B_2: & \mathbf{w}^T\mathbf{x}+b=-\gamma
\end{aligned}
$$
超平面 $B_1,B_2$ 间的距离称为**间隔**（margin）
$$
\text{margin}=\frac{2\gamma}{\|\mathbf w\|}
$$
假设样本能被超平面正确分类，则
$$
\begin{cases}
\mathbf{w}^T\mathbf{x}_i+b\geqslant \gamma  & \text{if }y_i=+1\\
\mathbf{w}^T\mathbf{x}_i+b\leqslant- \gamma & \text{if }y_i=-1
\end{cases}
$$
可简写为
$$
y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant \gamma \quad i=1,2,\cdots,N
$$

**参数估计**：由于超平面的系数经过同比例缩放不会改变这个平面，我们不妨给出约束 $\gamma=1$，从而得到唯一系数。那么最大化间隔可表示为：
$$
\begin{aligned}
\max\limits_{\mathbf w,b}&\frac{2}{\|\mathbf w\|} \\
\text{s.t.}&\quad y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant 1, \quad i=1,2,\cdots,N
\end{aligned}
$$

s.t. 是 subject to (such that) 的缩写，表示约束条件。约束为分类任务的要求。

显然，为了最大化间隔，仅需最大化 $\|\mathbf w\|^{-1}$，这等价于最小化 $\|\mathbf w\|^2$ 。于是上式可重写为
$$
\begin{aligned}
\min\limits_{\mathbf w,b}&\frac{1}{2}\|\mathbf w\|^2 \\
\text{s.t.}&\quad y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant 1, \quad i=1,2,\cdots,N
\end{aligned}
$$
这就是 SVM 的基本形式，是一个包含 $N$ 个约束的凸优化问题。

## 对偶问题

支持向量机通常将原始问题（primal problem）转化成拉格朗日对偶问题（dual problem）来求解。首先引入 Lagrange 函数
$$
L(\mathbf w,b,\mathbf\alpha)=\frac{1}{2}\mathbf w^T\mathbf w+\sum_{i=1}^N\alpha_i(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))
$$
参数 $\alpha_i\geqslant 0$ 称为拉格朗日乘子（Lagrange multiplier）。根据拉格朗日对偶性，原始问题的对偶问题为
$$
\max_{\mathbf\alpha}\min_{\mathbf w,b}L(\mathbf w,b,\mathbf\alpha)
$$
令 $L(\mathbf w,b,\mathbf\alpha)$ 对 $\mathbf w$ 和 $b$ 的偏导数为 0 可以得到
$$
\sum_{i=1}^N\alpha_iy_i=0\\
\mathbf w=\sum_{i=1}^N\alpha_iy_i\mathbf x_i
$$
将上式带入拉格朗日函数，我们就可以消去 $\mathbf w$ 和 $b$ ，得到对偶最优化问题
$$
\begin{aligned}
\max\limits_{\mathbf\alpha}&\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\mathbf x_i^T\mathbf x_j \\
\text{s.t.}&\quad \sum_{i=1}^N\alpha_iy_i=0 \\
&\quad\alpha_i\geqslant 0, \quad i=1,2,\cdots,N
\end{aligned}
$$
原问题和对偶问题等价的充要条件为其满足 KKT（Karush-Kuhn-Tucker）条件
$$
\begin{cases}
\alpha_i\geqslant 0  \\
1-y_i(\mathbf{w}^T\mathbf{x}_i+b) \leqslant 0  \\
\alpha_i(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))=0 
\end{cases}
$$
解出 $\mathbf\alpha$ 后，求出 $\mathbf w$ 和 $b$ 即可得到模型
$$
\begin{aligned}
f_{\mathbf{w},b}(\mathbf{x})&=\text{sign}(\mathbf{w}^T\mathbf{x}+b) \\
&=\text{sign}(\sum_{i=1}^N\alpha_iy_i\mathbf x_i^T\mathbf x+b)
\end{aligned}
$$
由 KKT 互补条件可知，对任意训练样本 $(\mathbf x_i,y_i)$， 总有 $\alpha_i=0$ 或 $y_i(\mathbf{w}^T\mathbf{x}_i+b)=1$。

- 若 $\alpha_i=0$ ，则该样本将不会在式 $f_{\mathbf{w},b}(\mathbf{x})$ 的求和中出现，也就不会对模型有任何影响
- 若 $\alpha_i>0$，则必有 $y_i(\mathbf{w}^T\mathbf{x}_i+b)=1$ ，所对应的样本点位于最大间隔边界上，被称为支持向量。这显示出 SVM 的一个重要性质：训练完成后，大部分的训练样本都不需保留，最终模型仅与支持向量有关。

不难发现，求解 $\mathbf\alpha$ 是一个二次规划问题，可使用通用的二次规划算法来求解。然而，该问题的规模正比于训练样本数，这会在实际任务中造成很大的开销。为了避开这个障碍，人们通过利用问题本身的特性，提出了很多高效算法， SMO (Sequential Minimal Optimization) 是其中典型代表。

## 软间隔与正则化

在前面的讨论中，假定训练样本是线性可分的，即所有样本都必须严格满足约束
$$
y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant 1
$$
这称为**硬间隔**（hard margin）。然而，现实任务中的数据很难线性可分，这时，我们为每个样本引入**松弛变量**（slack variable） $\xi_i>0$ ，允许某些样本不满足约束，约束条件修改为
$$
y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant 1-\xi_i
$$
使用**软间隔**（soft margin）最大化求解。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/SVM-soft-margin.png" style="zoom:67%;" />

当然，在最大化间隔的同时，不满足约束的样本应尽可能少。于是，引入损失
$$
\xi_i=\max\{0,1-y_i(\mathbf{w}^T\mathbf{x}_i+b)\}
$$
每个样本都有一个对应的松弛变量， 用以表征该样本不满足严格约束的程度。上式称为hinge损失函数
$$
\text{hinge\_loss}(z)=\max\{0,1-z\}
$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/hinge-loss.svg" style="zoom: 80%;" />

因此，软间隔SVM的优化目标可写为
$$
\begin{aligned}
\min\limits_{\mathbf w,b,\mathbf\xi}&\frac{1}{2}\|\mathbf w\|^2+C\sum_{i=1}^N\xi_i \\
\text{s.t.}&\quad y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant 1-\xi_i \\
&\quad \xi_i\geqslant 0, \qquad i=1,2,\cdots,N
\end{aligned}
$$
其中，$C$ 为惩罚参数，用于控制惩罚强度。这便是线性 SVM 的基本式。

原始问题的拉格朗日函数为
$$
L(\mathbf w,b,\mathbf\xi,\mathbf\alpha,\mathbf\eta)=\frac{1}{2}\mathbf w^T\mathbf w+C\sum_{i=1}^N\xi_i+\sum_{i=1}^N\alpha_i(1-\xi_i-y_i(\mathbf{w}^T\mathbf{x}_i+b))-\sum_{i=1}^N\eta_i\xi_i
$$
参数 $\alpha_i\geqslant 0,\eta_i$ 称为拉格朗日乘子。令 $L(\mathbf w,b,\mathbf\xi,\mathbf\alpha,\mathbf\eta)$ 对 $\mathbf w,b$ 和 $\mathbf \xi$ 的偏导数为 0 可以得到
$$
\sum_{i=1}^N\alpha_iy_i=0\\
\mathbf w=\sum_{i=1}^N\alpha_iy_i\mathbf x_i \\
\alpha_i+\eta_i=C
$$
将上式带入拉格朗日函数，得到**拉格朗日对偶问题**
$$
\begin{aligned}
\max\limits_{\mathbf\alpha}&\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\mathbf x_i^T\mathbf x_j \\
\text{s.t.}&\quad \sum_{i=1}^N\alpha_iy_i=0 \\
&\quad 0\leqslant\alpha_i\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
上式与线性可分对偶问题唯一不同的是对 $\alpha_i$ 的约束。对软间隔支持向量机， KKT 条件要求
$$
\begin{cases}
\alpha_i\geqslant 0 ,\quad \eta_i \geqslant 0 \\
1-\xi_i-y_i(\mathbf{w}^T\mathbf{x}_i+b)\leqslant 0  \\
\alpha_i(1-\xi_i-y_i(\mathbf{w}^T\mathbf{x}_i+b))=0 \\
\xi_i\geqslant 0 ,\quad \eta_i\xi_i=0
\end{cases}
$$
解出 $\mathbf\alpha,\mathbf\eta$ 后，求出 $\mathbf w$ 和 $b$ 即可得到模型
$$
\begin{aligned}
f_{\mathbf{w},b}(\mathbf{x})&=\text{sign}(\mathbf{w}^T\mathbf{x}+b) \\
&=\text{sign}(\sum_{i=1}^N\alpha_iy_i\mathbf x_i^T\mathbf x+b)
\end{aligned}
$$
由 KKT 条件可知，对任意训练样本 $(\mathbf x_i,y_i)$， 总有 $\alpha_i=0$ 或 $y_i(\mathbf{w}^T\mathbf{x}_i+b)=1-\xi_i$。

- 若 $\alpha_i=0$ ，则该样本将不会在式 $f_{\mathbf{w},b}(\mathbf{x})$ 的求和中出现，也就不会对模型有任何影响
- 若 $\alpha_i>0$，则必有 $y_i(\mathbf{w}^T\mathbf{x}_i+b)=1-\xi_i$ ，该样本为支持向量
- 若 $\alpha_i<C$，则 $\eta_i>0$，进而有 $\xi_i=0$ ，即该样本恰在最大间隔边界上
- 若 $\alpha_i=C$，则有 $\eta_i=0$，此时若  $\xi_i\leqslant 1$  则该样本落在最大间隔内部，若 $\xi_i>1$ 则该样本被错误分类

由此可看出，软间隔支持向量机的最终模型仅与支持向量有关，即通过采用hinge 损失函数仍保持了稀疏性。

## 核方法

现实任务中，原始样本空间内也许并不存在一个能正确划分两类样本的超平面。对这样的问题，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分。如下图，异或问题就不能线性可分

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/SVM-XOR.png" style="zoom:67%;" />

> 可证明，如果原始空间是有限维， 即属性数有限，那么一定存在一个高维特征空间使样本可分。

令 $\mathbf{\phi(x)}$ 表示将 $\mathbf x$ 映射后的特征向量，于是，在特征空间中划分超平面所对应的模型可表示为
$$
f_{\mathbf{w},b}(\mathbf{x})=\text{sign}(\mathbf{w}^T\mathbf{\phi(x)}+b)
$$
那么最大化间隔可表示为
$$
\begin{aligned}
\min\limits_{\mathbf w,b}&\frac{1}{2}\|\mathbf w\|^2+C\sum_{i=1}^N\xi_i \\
\text{s.t.}&\quad y_i(\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b)\geqslant 1-\xi_i \\
&\quad \xi_i\geqslant 0, \qquad i=1,2,\cdots,N
\end{aligned}
$$
其对偶问题是
$$
\begin{aligned}
\max\limits_{\mathbf\alpha}&\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\mathbf\phi(\mathbf x_i)^T\mathbf\phi(\mathbf x_j) \\
\text{s.t.}&\quad \sum_{i=1}^N\alpha_iy_i=0 \\
&\quad 0\leqslant\alpha_i\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
求解上述问题涉及到计算 $\mathbf\phi(\mathbf x_i)^T\mathbf\phi(\mathbf x_j)$， 这是样本 $\mathbf x_i$ 与 $\mathbf x_j$ 映射到特征空间之后的内积。由于特征空间维数可能很高，甚至可能是无穷维，因此直接计算  $\mathbf\phi(\mathbf x_i)^T\mathbf\phi(\mathbf x_j)$ 通常是困难的。为了避开这个障碍，引入**核函数**（kernel function） 
$$
K(\mathbf x_1,\mathbf x_2)=\mathbf\phi(\mathbf x_1)^T\mathbf\phi(\mathbf x_2)
$$
即  $\mathbf x_i$ 与 $\mathbf x_j$  在特征空间的内积等于它们在原始样本空间中通过核函数计算的结果，这称为**核技巧**（kernel trick）。核函数 $K$ 的实现方法通常有比直接构建 $\mathbf\phi(\mathbf x)$ 再算点积高效很多。

于是，对偶问题可重写为
$$
\begin{aligned}
\max\limits_{\mathbf\alpha}&\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j K(\mathbf x_i,\mathbf x_j) \\
\text{s.t.}&\quad \sum_{i=1}^N\alpha_iy_i=0 \\
&\quad 0\leqslant\alpha_i\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
可写为矩阵形式
$$
\begin{aligned}
\min\limits_{\mathbf\alpha}&\frac{1}{2}\mathbf\alpha^T\mathbf Q\mathbf\alpha-\mathbf e^T\mathbf\alpha \\
\text{s.t.}&\quad \mathbf y^T\mathbf\alpha=0 \\
&\quad 0\leqslant\alpha_i\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
其中，$\mathbf e$ 是一个全1的 $N$ 维向量，$\mathbf Q$ 是一个 $N\times N$ 的半正定矩阵，$Q_{ij}=y_iy_j K(\mathbf x_i,\mathbf x_j)$。

求解后即可得到
$$
\begin{aligned}
f_{\mathbf{w},b}(\mathbf{x})&=\text{sign}(\mathbf{w}^T\mathbf{\phi(x)}+b) \\
&=\text{sign}(\sum_{i=1}^N\alpha_iy_iK(\mathbf x_i,\mathbf x)+b)
\end{aligned}
$$
通过前面的讨论可知，我们希望样本在特征空间内线性可分，因此特征空间的好坏对支持向量机的性能至关重要。需注意的是，在不知道特征映射的形式时，我们并不知道什么样的核函数是合适的，而核函数也仅是隐式地定义了
这个特征空间。于是，"核函数选择"成为支持向量机的最大变数，若核函数选择不合适，则意味着将样本映射到了一个不合适的特征空间，很可能导致性能不佳。

下面介绍几种常用的核函数

(1) 线性核函数（linear kernel function）
$$
K(\mathbf x_1,\mathbf x_2)=\mathbf x_1^T\mathbf x_2
$$
(2) 多项式核函数（polynomial kernel function）
$$
K(\mathbf x_1,\mathbf x_2)=(\mathbf x_1^T\mathbf x_2+1)^p
$$
(3) 高斯核函数（Gaussian kernel function）：也被称为径向基函数（radial basis function, RBF），是最常用的核函数。$\sigma>0$ 为高斯核的带宽（width）。
$$
K(\mathbf x_1,\mathbf x_2)=\exp(\frac{\|\mathbf x_1-\mathbf x_2\|^2}{2\sigma^2})
$$
(4) 拉普拉斯核函数（Laplace kernel function）
$$
K(\mathbf x_1,\mathbf x_2)=\exp(\frac{\|\mathbf x_1-\mathbf x_2\|}{2\sigma^2})
$$
(5) Sigmoid 核函数（Sigmoid kernel function）
$$
K(\mathbf x_1,\mathbf x_2)=\tanh(\beta\mathbf x_1^T\mathbf x_2+\theta)
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/svm-kernel-iris.png)

## 支持向量回归

支持向量分类方法可以推广到解决回归问题，这种方法称为支持向量回归（Support Vector Regression，SVR）。

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in\R$ 。

**Model**：相比于线性回归用一条线来拟合训练样本， SVR采用一个以 $f(\mathbf x)$ 为中心，宽度为 $2\epsilon$ 的间隔带，来拟合训练样本。预测函数仍为
$$
f_{\mathbf{w},b}(\mathbf{x})=\mathbf{w}^T\mathbf\phi(\mathbf x)+b
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/SVR.svg)

落在带子内的样本不计算损失，不在带子内的样本则以偏离的程度作为损失，然后以最小化损失的方式迫使间隔带从样本最密集的地方（中心地带）穿过，进而达到拟合训练样本的目的。

为适应每个样本，引入松弛变量
$$
\xi_i=\max\{0,|\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b-y_i|-\epsilon\}
$$
并将 $\xi_i\geqslant 0$ 作为惩罚项加入优化，惩罚那些偏离 $\epsilon$ 带的样本，$\xi_i$ 表示样本远离带的程度。

因此SVR的优化问题可以写为
$$
\begin{aligned}
\min\limits_{\mathbf w,b}&\frac{1}{2}\|\mathbf w\|^2+C\sum_{i=1}^N\xi_i \\
\text{s.t.}&\quad -\epsilon-\xi_i \leqslant \mathbf{w}^T\mathbf\phi(\mathbf x_i)+b\leqslant \epsilon+\xi_i \\
&\quad \xi_i\geqslant 0, \qquad i=1,2,\cdots,N
\end{aligned}
$$
> 注释：
>
> - 当样本在带内时，一定满足约束条件，因此代价函数中惩罚项取最小值 $\xi_i=0$
> - 当样本在带外时，为满足样本位置和惩罚项最小值，约束条件则变为 $|\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b|=\epsilon+\xi_i$

这里我们使用了**ϵ-不敏感损失函数**（epsilon-insensitive）, 即小于ϵ的误差被忽略了。
$$
\text{loss}_\epsilon(z)-=\max\{0,|z|-\epsilon\}
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/SVR-loss.svg)

> 在这里不用均方误差的目的是为了和软间隔支持向量机的优化目标保持形式上的一致，这样就可以导出对偶问题引入核函数

如果考虑两边采用不同的松弛程度，可重写为
$$
\begin{aligned}
\min\limits_{\mathbf w,b}&\frac{1}{2}\|\mathbf w\|^2+C\sum_{i=1}^N(\xi_i+\xi_i') \\
\text{s.t.}&\quad -\epsilon-\xi_i' \leqslant \mathbf{w}^T\mathbf\phi(\mathbf x_i)+b\leqslant \epsilon+\xi_i \\
&\quad \xi_i\geqslant 0,\xi_i'\geqslant 0, \qquad i=1,2,\cdots,N
\end{aligned}
$$
对偶问题为
$$
\begin{aligned}
\max\limits_{\mathbf\alpha,\mathbf\alpha'}&\sum_{i=1}^Ny_i(\alpha_i'-\alpha_i)-\epsilon(\alpha_i'+\alpha_i) \\
&-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N(\alpha_i'-\alpha_i)(\alpha_j'-\alpha_j)K(\mathbf x_i,\mathbf x_j) \\
\text{s.t.}&\quad \sum_{i=1}^N(\alpha_i'-\alpha_i)=0 \\
&\quad 0\leqslant\alpha_i,\alpha_i'\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
可写为矩阵形式
$$
\begin{aligned}
\min\limits_{\mathbf\alpha,\mathbf\alpha'}&\frac{1}{2}(\mathbf\alpha-\mathbf\alpha')^T\mathbf Q(\mathbf\alpha-\mathbf\alpha')+\epsilon\mathbf e^T(\mathbf\alpha+\mathbf\alpha')-\mathbf y^T(\mathbf\alpha-\mathbf\alpha') \\
\text{s.t.}&\quad \mathbf e^T(\mathbf\alpha-\mathbf\alpha')=0 \\
&\quad 0\leqslant\alpha_i,\alpha_i'\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
其中，$\mathbf e$ 是一个全1的 $N$ 维向量，$\mathbf Q$ 是一个 $N\times N$ 的半正定矩阵，$Q_{ij}=K(\mathbf x_i,\mathbf x_j)$。

上述过程中需满足KKT 条件，即要求
$$
\begin{cases}
\alpha_i(\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b-y_i-\epsilon-\xi_i)=0 \\
\alpha_i'(y_i-\mathbf{w}^T\mathbf\phi(\mathbf x_i)-b-\epsilon-\xi_i')=0 \\
\alpha_i\alpha_i'=0 ,\quad \xi_i\xi_i'=0 \\
(C-\alpha_i)\xi_i=0 ,\quad (C-\alpha_i')\xi_i'=0
\end{cases}
$$
可以看出

- 当且仅当 $\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b-y_i-\epsilon-\xi_i=0$ 时，$\alpha_i$ 能取非零值
- 当且仅当 $y_i-\mathbf{w}^T\mathbf\phi(\mathbf x_i)-b-\epsilon-\xi_i'=0$ 时，$\alpha_i'$ 能取非零值

换言之，仅当样本不落入间隔带中，相应的 $\alpha_i$ 和 $\alpha_i'$ 才能取非零值。

- 此外，约束 $\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b-y_i-\epsilon-\xi_i=0$ 和 $y_i-\mathbf{w}^T\mathbf\phi(\mathbf x_i)-b-\epsilon-\xi_i'=0$ 不能同时成立，因此 $\alpha_i$ 和 $\alpha_i'$ 中至少有一个为零。

预测函数为
$$
f_{\mathbf{w},b}(\mathbf{x})=\sum_{i=1}^N(\alpha_i-\alpha_i')K(\mathbf x_i,\mathbf x)+b
$$
能使上式中的 $\alpha_i-\alpha_i'\neq 0$ 的样本即为 SVR 的支持向量，它们必落在 $\epsilon$ 间隔带之外。显然，SVR 的支持向量仅是训练样本的一部分，即其解仍具有稀疏性。

# 高斯过程

#  朴素贝叶斯

# 多分类和多标签

## 多类别分类任务

**Multiclass classification**：目标变量包含两个以上离散值的分类任务 $y\in\{0,1,2,\cdots,K\}$。每个样本只能标记为一个类。例如，使用从一组水果图像中提取的特征进行分类，其中每一幅图像都可能是一个橙子、一个苹果或一个梨。每个图像就是一个样本，并被标记为三个可能的类之一。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/multiclass-classification.svg" style="zoom: 67%;" />

- One-Vs-Rest (OVR) 也称为one-vs-all，为每个类分别拟合一个二分类模型，这是最常用的策略，对每个类都是公平的。这种方法的一个优点是它的可解释性，每个类都可以查看自己模型的相关信息。

- One-Vs-One (OVO) 是对每一对类分别拟合一个二分类模型。在预测时，选择得票最多的类别。在票数相等的两个类别中，它选择具有最高总分类置信度的类别，方法是对由底层二分类器计算的对分类置信度进行求和。

  由于它需要拟合 $\frac{K(K-1)}{2}$ 个分类器，这种方法通常比one-vs-rest要慢，原因就在于其复杂度 O(K^2^) 。然而，这个方法也有优点，比如说是在没有很好的缩放样本数的核方法中。这是因为每个单独的学习问题只涉及一小部分数据，而对于一个 one-vs-rest，完整的数据集将会被使用 K 次。

**One-Vs-Rest**：为每个类分别拟合一个二分类模型
$$
f^i_{\mathbf{w},b}(\mathbf{x})=\mathbb P(y=i|\mathbf x;\mathbf w,b)
$$
模型预测值，一种方法是选择概率最大的类别
$$
\hat y=\arg\max\limits_{i} f^i_{\mathbf{w},b}(\mathbf{x})
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/one-vs-all.svg)

## Lasso和Elastic-Net

包含多个目标变量的回归任务称为 **Multioutput regression**

**Multi-task Lasso** 是一个估计多任务的稀疏系数的线性模型， $\mathbf Y$ 是一个  $N\times N_{tasks}$ 矩阵。约束条件是，对于所有回归问题（也叫任务），所选的特征是相同的。它混合使用 $\ell_1\ell_2$ 范数作为正则化项。

Cost function
$$
J(\mathbf{W})=\cfrac{1}{2N}\left(\|\mathbf{XW-Y}\|^2_{Fro}+ \alpha \|\mathbf W\|_{21}\right)
$$
其中 Fro 表示Frobenius范数
$$
\mathbf \|\mathbf A\|_{Fro}=\sqrt{\sum_{ij}a^2_{ij}}=\sqrt{\text{tr}(\mathbf A^T\mathbf A)}
$$
混合 $\ell_1\ell_2$ 范数
$$
\mathbf \|\mathbf A\|_{21}=\sum_i\sqrt{\sum_{j}a^2_{ij}}
$$
Multi-task Lasso 也采用坐标下降法来估计参数。

**Multi-task Elastic-Net** 是一个估计多任务的稀疏系数的线性模型， $\mathbf Y$ 是一个  $N\times N_{tasks}$ 矩阵。约束条件是，对于所有回归问题（也叫任务），所选的特征是相同的。它使用混合的 $\ell_1\ell_2$ 范数和$\ell_2$作为正则化项。

Cost function
$$
J(\mathbf{W})=\cfrac{1}{2N}\left(\|\mathbf{XW-Y}\|^2_{Fro}+ \alpha\rho \|\mathbf W\|_{21}+ \frac{\alpha(1-\rho)}{2} \|\mathbf W\|_{Fro}^2\right)
$$
Multi-task Elastic-Net 也采用坐标下降法来估计参数。

## 多标签分类任务

包含多个目标变量的分类任务称为 **Multilabel classification**

# 集成学习

## 集成学习

**集成学习**（ensemble learning）通过构建**基学习器**（base learner）集合 $\{h_1,h_2,\cdots,h_M\}$，并组合基学习器的结果来提升预测结果的准确性和泛化能力。其中，基学习器通常采用弱学习算法，组合形成强学习算法。

> 准确率仅比随机猜测略高的学习算法称为**弱学习算法**，准确率很高并能在多项式时间内完成的学习算法称为**强学习算法**。

下面以二分类问题和回归问题为例，说明集成弱学习器为什么能够改善性能。

(1) 对于二分类问题，假设 $M$ 个弱分类模型，集成分类器采用多数表决的方法来预测类别，仅当基分类器超过一半预测错误的情况下，集成分类器预测错误。
$$
H(\mathbf x)=\text{sign}\left(\frac{1}{M}\sum_{m=1}^Mh_m(\mathbf x)\right)
$$
假设基分类器之间相互独立，且错误率相等为 $\epsilon$ 。则集成分类器的错误率为
$$
\epsilon_{\text{ensemble}} =\sum_{k=0}^{\lfloor M/2\rfloor}\complement^k_M(1-\epsilon)^k\epsilon^{M-k}
$$
取25个基分类器，误差率均为 0.35 ，计算可得集成分类器的误差为 0.06 ，远低于基分类器的误差率。注意，当 $\epsilon>0.5$ 时，集成分类器比不上基分类器。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/ensemble-classifier-error.svg" style="zoom:67%;" />

令$\epsilon=0.5-\gamma$ ，其中 $\gamma$ 度量了分类器比随机猜测强多少。则由Hoeffding 不等式可知
$$
\epsilon_{\text{ensemble}}\leqslant \exp(-2M\gamma^2)
$$
上式指出，随着基分类器的个数的增加集成错误率呈指数下降，从而快速收敛。但前提是基分类器之间相互独立。

(2) 对于回归问题，假设 $M$ 个弱回归模型，集成模型以均值输出
$$
H(\mathbf x)=\frac{1}{M}\sum_{m=1}^Mh_m(\mathbf x)
$$
每个基模型的误差服从均值为零的正态分布
$$
\epsilon_m\sim N(0,\sigma^2)
$$
若不同模型误差间的协方差均为 $\text{Cov}(\epsilon_i,\epsilon_j)=c$ 。则集成模型误差平方的期望是


$$
\begin{aligned}
\mathbb E(\epsilon_{\text{ensemble}}^2)
&=\mathbb E\left[\left(\frac{1}{M}\sum_{m=1}^M\epsilon_m\right)^2\right] \\
&=\frac{1}{M^2}\mathbb E\left[\sum_{i=1}^M\left(\epsilon_i^2+\sum_{j\neq i}\epsilon_i\epsilon_j\right)\right]  \\
&=\frac{1}{M}\sigma^2+\frac{M-1}{M}c
\end{aligned}
$$

在误差完全相关即 $c=\sigma^2$ 的情况下，误差平方减少到 $\sigma^2$ ，所以，模型平均没有任何帮助。在误差彼此独立即 $c=0$ 的情况下，该误差平方的期望仅为 $\sigma^2/M$ 。

上述示例容易得出，集成学习的基学习器要有足够的**准确性**和**差异性**。集成方法主分成两种：

- Bagging：是一种并行方法。通过在训练集上的有放回抽样来获得基学习器间的差异性。最典型的代表就是随机森林。
- Boosting：是一种串行迭代过程。自适应的改变训练数据的权重分布，构建一系列基分类器。最经典的包括AdaBoost算法和GBDT算法。

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。

## Bagging

### Bagging

Bagging（Bootstrap aggregating，袋装算法）是一种并行式的集成学习方法。通过在基学习器的训练集中引入随机化后训练。若数据集$D$有$N$ 个样本，则随机有放回采样出包含$N$个样本的数据集（可能有重复），同样的方法抽取$M$个训练集，这样在训练的时候每个训练集都会有不同。最后训练出 $M$ 个基学习器集成。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/bagging_algorithm.png" style="zoom:80%;" />

可以看出Bagging主要通过**样本的扰动**来增加基学习器之间的差异性，因此Bagging的基学习器应为那些对训练集十分敏感的不稳定学习算法，例如：神经网络与决策树等。从偏差-方差分解来看，Bagging算法主要关注于降低方差，即通过多次重复训练提高泛化能力。

由于抽样都来自于同一个数据集，且是有放回抽样，所以$M$个数据集彼此相似，而又因随机性而稍有不同。Bagging训练集中有接近36.8%的样本没有被采到
$$
\lim_{N\to\infty}(1-\frac{1}{N})^N=\frac{1}{e}\approx 0.368
$$
Bagging方法有许多不同的变体，主要是因为它们提取训练集的随机子集的方式不同：

- 如果使用无放回抽样，我们叫做 Pasting
- 如果使用有放回抽样，我们称为 Bagging
- 如果抽取特征的随机子集，我们叫做随机子空间 (Random Subspaces) 
- 最后，如果基学习器构建在对于样本和特征抽取的子集之上时，我们叫做随机补丁 (Random Patches) 

### 随机森林

对于决策树，事实证明，只需要改变一个训练样本，最高信息增益对应的特征就可能发生改变，因此在根节点会产生一个不同的划分，生成一颗完全不同的树。因此单个决策树对数据集的微小变化异常敏感。

**随机森林**（Random Forest）是Bagging的一个拓展体，它的基学习器固定为决策树，在基学习器构造过程中引入随机：

1. 采用有放回抽样的方式添加样本扰动，但有时在根节点附近也有相似的特征组成。
2. 因此进一步引入了特征扰动，每一个分裂过程从待选的 $n$ 个特征中随机选出包含 $k$ 个特征的子集，从这个子集中选择最优划分特征，一般推荐 $k=\log_2(n)$ 或 $k=\sqrt{n}$ 。
3. 每棵树都会完整成长而不会剪枝

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/random-forest.svg)

**随机森林优势**

- 它能够处理很高维度（特征很多）的数据，并且不用做特征选择
- 容易做成并行化方法，速度比较快
- 只在特征集的一个子集中选择划分，因此训练效率更高

## Boosting

### AdaBoost

**Boosting**（提升方法）是一种串行迭代过程。先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基学习器。如此迭代，最后将这些弱学习器组合成一个强学习器。Boosting族算法最著名、使用最为广泛的就是AdaBoost。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/boosting-flowchart.svg)

**AdaBoost** （Adaptive Boosting，自适应提升）的核心思想是用反复调整的数据来训练一系列的弱学习器，由这些弱学习器的加权组合，产生最终的预测结果。

具体说来，整个Adaboost 迭代算法分为3步：

1. **训练弱学习器**：在连续的提升（boosting）迭代中，那些在上一轮迭代中被预测错误的样本的权重将会被增加，而那些被预测正确的样本的权重将会被降低。然后，权值更新过的样本集被用于训练弱学习器。随着迭代次数的增加，那些难以预测的样例的影响将会越来越大，每一个随后的弱学习器都将会被强迫关注那些在之前被错误预测的样例。初始化时，所有样本都被赋予相同的权值 $1/N$ 。
2. **计算弱学习器权重**：在每一轮迭代中加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数，从而得到 $M$ 个弱学习器 $h_1,h_2,\cdots,h_M$。各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，使其在最终的分类函数中起着较大的决定作用，而降低分类误差率大的弱分类器的权重，使其在最终的分类函数中起着较小的决定作用。这样，每个弱分类器 $h_m$ 都有对应的权重 $\alpha_m$ 。
3. **组合成强学习器**：最后的强学习器由生成的多个弱学习器加权求和产生。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/AdaBoost_example.svg)

可以看出：**AdaBoost的核心步骤就是计算基学习器权重和样本权重分布**。AdaBoost 算法有多种推导方式，比较容易理解的是基于加法模型（additive model）的**前向分布算法**（forward stagewise algorithm）。

给定二分类数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标 $y_i\in\{-1,+1\}$

基分类器的加权组合即为加法模型
$$
f(\mathbf x)=\sum_{m=1}^M\alpha_mh_m(\mathbf x)
$$
其中 $\alpha_m$ 表示基分类器 $h_m$ 的重要性。最终的强分类器为
$$
H(\mathbf x)=\mathrm{sign}(f(\mathbf x))=\mathrm{sign}\left(\sum_{m=1}^M\alpha_mh_m(\mathbf x)\right)
$$
$f(\mathbf x)$ 的符号决定了实例 $\mathbf x$ 的类别。

给定损失函数 $L(y,f(\mathbf x))$ ，学习模型 $f(\mathbf x)$ 所要考虑的问题是最小化代价函数
$$
\min_{\alpha_m,h_m}\sum_{i=1}^NL\left(y_i,\sum_{m=1}^M\alpha_mh_m(\mathbf x_i)\right)
$$
通常这是一个复杂的全局优化问题，前向分布算法使用其简化版求解这一问题：既然是加法模型，每一步只学习一个弱学习器及其系数，且不调整已经加入模型中的参数和系数来向前逐步建立模型，这能够得到上述优化的近似解。这样，前向分布算法将同时求解 $m=1$ 到 $M$ 所有参数 $\alpha_m,\theta_m$ 的优化问题简化为逐步求解 $\alpha_m,\theta_m$ 的优化问题。

假设经过 $m-1$ 轮迭代，已经得到之前所有弱分类器的加权和 
$$
f_{m-1}(\mathbf x)=\alpha_1h_1(\mathbf x)+\cdots+\alpha_{m-1}h_{m-1}(\mathbf x)
$$
在第 $m$ 轮迭代求解 $\alpha_m,h_m$ 得到
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+\alpha_m h_m(\mathbf x)
$$
则每一步只需优化如下代价函数
$$
(\alpha_m,h_m)=\arg\min_{\alpha,h}\sum_{i=1}^NL\left(y_i,f_{m-1}(\mathbf x)+\alpha h(\mathbf x_i)\right)
$$
其中，在第$m$轮迭代中，$f_{m-1}(\mathbf x)$ 相当于一个定值。AdaBoost 每步采用**指数损失函数**（exponential loss function）
$$
L(y,f(\mathbf x))=\exp(-yf(\mathbf x))
$$

于是，优化函数可变为
$$
\begin{aligned}
(\alpha_m,h_m)&=\arg\min_{\alpha,h}\sum_{i=1}^NL(y_i,f_{m-1}(\mathbf x_i)+\alpha h(\mathbf x_i)) \\
&=\arg\min_{\alpha,h}\sum_{i=1}^N\exp[-y_i(f_{m-1}(\mathbf x_i)+\alpha h(\mathbf x_i))] \\
&=\arg\min_{\alpha,h}\sum_{i=1}^N  w_m^{(i)}\exp[-y_i\alpha h(\mathbf x_i)] \\

\end{aligned}
$$
其中 $w_m^{(i)}=\exp(-y_if_{m-1}(\mathbf x_i))$ 。$w_m^{(i)}$ 不依赖于 $\alpha$ 和 $h$ ，所以与优化无关。

由 AdaBoost 基分类器 $h(\mathbf x_i)\in\{-1,+1\}$ ，且 $y_i\in\{-1,+1\}$ 则
$$
-y_ih(\mathbf x_i)=\begin{cases}
+1 & \text{if }h(\mathbf x_i)\neq y_i \\
-1 & \text{if }h(\mathbf x_i)= y_i
\end{cases}
$$

所以，优化函数进一步化为

$$
\begin{aligned}
(\alpha_m,h_m)&=\arg\min_{\alpha,h}\left\{\sum_{i=1}^N  w_m^{(i)}e^{-\alpha}\mathbb I(h_m(\mathbf x_i)=y_i)+\sum_{i=1}^N  w_m^{(i)}e^{\alpha}\mathbb I(h_m(\mathbf x_i)\neq y_i)\right\} \\
&=\arg\min_{\alpha,h}\left\{(e^{\alpha}-e^{-\alpha})\sum_{i=1}^N  w_m^{(i)}\mathbb I(h_m(\mathbf x_i)\neq y_i)+e^{-\alpha}\sum_{i=1}^N  w_m^{(i)}\right\} \\
\end{aligned}
$$

上式可以得到AdaBoost算法的几个关键点：

**(1) 基学习器**。对于任意 $\alpha>0$ ，基分类器的解为
$$
h_m=\arg\min_{h}\sum_{i=1}^N w_m^{(i)}\mathbb I(h(\mathbf x_i)\neq y_i)
$$
这是第 $m$ 轮加权错误率最小的基分类器。

**(2) 各基学习器的系数** 。将已求得的 $h_m$ 带入优化函数
$$
\alpha_m=\arg\min_{\alpha}\left\{(e^{\alpha}-e^{-\alpha})\epsilon_m+e^{-\alpha}\right\}
$$
其中， $\epsilon_m$ 正是基分类器 $h_m$ 在加权训练集 $D_m$ 的错误率
$$
\epsilon_m=\frac{\displaystyle\sum_{i=1}^N  w_m^{(i)} \mathbb I(h_m(\mathbf x_i)\neq y_i)}{\displaystyle\sum_{i=1}^N w_m^{(i)}}
$$
这里 $ w_m^{(i)}$ 是第 $m$ 轮迭代中样本 $(\mathbf x_i,y_i)$ 的权重 ，因为Adaboost更新样本权值分布时做了规范化，所示上式中的分母为1。权重依赖于 $f_{m-1}(\mathbf x)$ ，随着每一轮迭代而发生改变。

对 $\alpha$ 求导并使导数为 0，即可得到基分类器 $h_m$ 的权重
$$
\alpha_m=\frac{1}{2}\ln(\frac{1-\epsilon_m}{\epsilon_m})
$$

由上式可知，当 $\epsilon_m\leqslant 0.5$ 时，$\alpha_m\geqslant 0$，并且 $\alpha_m$ 随 $\epsilon_m$ 的减小而增大 。所以，分类误差率越小的基分类器在最终分类器中的作用越大。如下图

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/AdaBoost_alpha.svg)

**(3) 下一轮样本权值**。由
$$
\begin{cases}
f_m(\mathbf x)=f_{m-1}(\mathbf x)+\alpha_m h_m(\mathbf x) \\
 w_m^{(i)}=\exp(-y_if_{m-1}(\mathbf x_i))
\end{cases}
$$
可得到
$$
w_{m+1}^{(i)}= w_m^{(i)}\exp(-\alpha_my_ih_m(\mathbf x_i))
$$
为了确保  $\mathbf w_{m+1}$ 成为一个概率分布 $\sum_{i=1}^Nw_{m+1}^{(i)}=1$， 权重更新变为
$$
w_{m+1}^{(i)}=\frac{w_m^{(i)}}{Z_m}\exp(-\alpha_my_ih_m(\mathbf x_i))
$$
其中， $Z_m$ 是正规因子。对原始式中所有的权重都乘以同样的值，对权重更新没有影响。
$$
Z_m=\sum_{i=1}^Nw_m^{(i)}\exp(-\alpha_my_ih_m(\mathbf x_i))
$$
上式可拆解为 
$$
w_{m+1}^{(i)}=\frac{w_m^{(i)}}{Z_m}\times\begin{cases}
\exp(-\alpha_m) & \text{if }h_m(\mathbf x_i)=y_i \\
\exp(\alpha_m) & \text{if }h_m(\mathbf x_i)\neq y_i 
\end{cases}
$$

上式给出的权值更新公式增加那些被错误分类的样本的权值，并减少那些被正确分类的样本的权值。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/AdaBoost_algorithm.png" style="zoom:80%;" />

从偏差-方差分解来看：AdaBoost 算法主要关注于降低偏差，每轮的迭代都关注于训练过程中预测错误的样本，很容易受过拟合的影响。

### 提升树

以二叉决策树为基函数的提升方法称为**提升树**（boosting tree）。提升树模型可以表示为决策树的加法模型：
$$
f_M(\mathbf x)=\sum_{m=1}^MT(\mathbf x;\Theta_m)
$$
其中，$T(\mathbf x;\Theta_m)$ 表示决策树，$\Theta_m$ 为决策树的参数， $M$ 为树的个数。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Tree_Ensemble_Model.png" style="zoom: 67%;" />

提升树采用前向分布算法实现学习的优化过程。，初始树 $f_0(\mathbf x)=0$ ，第 $m$ 轮迭代的模型是
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+T(\mathbf x;\Theta_m)
$$
其中，$f_{m-1}(\mathbf x)$ 是上一轮模型。本轮的目标是找到一个二叉树 $T(\mathbf x;\Theta_m)$，最小化代价函数
$$
\min_{\Theta_m}\sum_{i=1}^NL(y_i,f_{m-1}(\mathbf x_i)+T(\mathbf x_i;\Theta_m))
$$

对于不同问题的提升树学习算法，其主要区别在于损失函数的不同。主要包括用平法误差损失函数的回归问题，用指数损失函数的分类问题，以及用一般损失函数的一般决策问题。

(1) **回归问题**：提升树每一步采用平方误差损失函数
$$
L(y,f(\mathbf x))=(y-f(\mathbf x))^2
$$
第 $m$ 轮样本的损失为
$$
\begin{aligned}
L(y,f_m(\mathbf x))=&L(y,f_{m-1}(\mathbf x)+T(\mathbf x;\Theta_m)) \\
=&(y-f_{m-1}(\mathbf x)-T(\mathbf x;\Theta_m))^2 \\
=&(r_m-T(\mathbf x;\Theta_m))^2
\end{aligned}
$$
这里
$$
r_m=y-f_{m-1}(\mathbf x)
$$
是上一轮模型$f_{m-1}(\mathbf x)$拟合数据的残差（residual）。所以，对回归问题的提升树来说，最小化损失函数相当于决策树（弱学习器）简单拟合残差。

举一个通俗的例子，假如有个人30岁，我们首先用20岁去拟合，发现损失有10岁，这时我们用6岁去拟合剩下的损失，发现差距还有4岁，第三轮我们用3岁拟合剩下的差距，差距就只有一岁了。如果我们的迭代轮数还没有完，可以继续迭代下面，每一轮迭代，拟合的岁数误差都会减小。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GBDT-example.svg)

(2) **分类问题**：每一步采用指数损失函数，提升树算法只是 Adaboost算法的特例。

(3) **一般损失函数**：当损失函数是平方损失和指数函数时，每一步优化时很简单的，但对一般损失函数而言，往往每一步优化并不那么容易。针对这一问题，Freidman提出梯度提升（gradient boosting）算法。

### GBDT

**梯度提升树**（Gradient Boosted Decision Tree，GBDT）又叫MART（Multiple Additive Regression Tree），是提升树的一种改进算法，适用于任意可微损失函数。可用于各种领域的回归和分类问题，包括Web搜索、排名和生态领域。

GBDT加法模型表示为：
$$
f_M(\mathbf x)=\sum_{m=1}^MT(\mathbf x;\Theta_m)
$$
其中，$T(\mathbf x;\Theta_m)$ 表示决策树，$\Theta_m$ 为决策树的参数， $M$ 为树的个数。

最优化代价函数
$$
J(f)=\sum_{i=1}^NL(y_i,f(\mathbf x_i))
$$
GBDT使用前向分布算法迭代提升。假设第 $m$ 轮迭代的模型是
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+T(\mathbf x;\Theta_m)
$$
其中，$f_{m-1}(\mathbf x)$ 是上一轮迭代得到的模型。使用一阶泰勒展开[^taylor]，第 $m$ 轮第 $i$ 个样本的损失函数的近似值为
$$
L(y_i,f_m(\mathbf x_i))\approx L(y_i,f_{m-1}(\mathbf x_i))+T(\mathbf x_i;\Theta_m) g_{mi}
$$
其中
$$
g_{mi}=\left[\frac{\partial L(y_i,f(\mathbf x_i))}{\partial f(\mathbf x_i)}\right]_{f=f_{m-1}}
$$
本轮代价函数变为
$$
J(f_m)\approx J(f_{m-1})+\sum_{i=1}^NT(\mathbf x_i;\Theta_m) g_{mi}
$$
我们希望随着每轮迭代，损失会依次下降 $J(f_m)-J(f_{m-1})<0$，且本轮损失最小化，则有
$$
\min_{\Theta_m}\sum_{i=1}^NT(\mathbf x_i;\Theta_m) g_{mi}=\min_{\Theta_m}\mathbf T(\Theta_m)\cdot \mathbf g_m=\min_{\Theta_m}\|\mathbf T(\Theta_m)\|_2\|\mathbf g_m\|_2\cos\theta
$$
其中，$\mathbf T(\Theta_m)=(T(\mathbf x_1;\Theta_m),T(\mathbf x_2;\Theta_m),\cdots,T(\mathbf x_N;\Theta_m))$ 为第 $m$ 轮的强学习器在 $N$ 个数据点上的提升向量，$\mathbf g_m=(g_{m1},g_{m2},\cdots,g_{mN})$ 为损失函数在 $f_{m-1}(\mathbf x)$ 处的梯度向量，$\theta$ 为两向量夹角。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GBDT_residual.svg" style="zoom: 80%;" />

如果被拟合的决策树 $T(\mathbf x_i;\Theta_m)$ 预测的值与负梯度 $-g_{mi}$ 成正比 ($\cos\theta=-1$)，则取得最小值。因此，在每次迭代时，使用负梯度$-g_{mi}$ 来拟合决策树。$m$ 轮的弱学习器可表示为
$$
T(\mathbf x;\Theta_m)\approx -\left[\frac{\partial L(y,f(\mathbf x))}{\partial f(\mathbf x)}\right]_{f=f_{m-1}}
$$
负梯度被称为广义残差或**伪残差**（pseudo residual）。梯度在每次迭代中都会被更新，试图在**局部最优**方向求解，这可以看作是**函数空间中的某种梯度下降**，不同的损失函数将会得到不同的负梯度。下表总结了通常使用的损失函数的梯度

| Setting        | Loss Function                       | Gradient                                               |
| :------------- | ----------------------------------- | ------------------------------------------------------ |
| Regression     | $\frac{1}{2}(y_i-f(\mathbf x_i))^2$ | $y_i-f(\mathbf x_i)$                                   |
| Regression     | $\mid y_i-f(\mathbf x_i)\mid$       | $\text{sign} (y_i-f(\mathbf x_i))$                     |
| Regression     | Huber                               |                                                        |
| Classification | Deviance                            | $k$th component: $\mathbb I(y_i=c_k)-P_k(\mathbf x_i)$ |

对于平方误差损失，负梯度恰恰是普通的残差。因为GBDT每次迭代要拟合的梯度值是连续值，所以限定了基学习器只能使用**CART回归树**，且树的生成使用加权均方误差选择最优划分特征。即使对于分类任务，基学习器仍然是CART回归树。

**回归树流程**：

(1) 默认情况下，初始决策树选择使损失最小化的常数（对于均方误差损失，这是目标值的经验平均值 $\bar y$）。
$$
f_0(\mathbf x)=\arg\min_c\sum_{i=1}^NL(y_i,c)
$$
(2) 对每步迭代 $m=1,2,\cdots,M$

​    (a) 计算损失函数的负梯度
$$
r_{mi}=-\left[\frac{\partial L(y_i,f(\mathbf x_i))}{\partial f(\mathbf x_i)}\right]_{f=f_{m-1}}\quad i=1,2,\cdots,N
$$
​    (b) 对 $r_{mi}$ 拟合一个回归树
$$
T_m(\mathbf x)=\arg\min_{h}\sum_{i=1}^NL(r_{mi},h(\mathbf x_i))
$$
​    得到第 $m$ 棵树的叶节点区域 $R_{mj},\ j=1,2,\cdots,J_m$。

​    (c) 计算每个叶节点的输出值 $j=1,2,\cdots,J_m$
$$
c_{mj}=\arg\min_{c}\sum_{\mathbf x_i\in R_{mj}}L(y_i,f_{m-1}(\mathbf x_i)+c)
$$
(3) 更新
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+\sum_{j=1}^{J_m}c_{mj}\mathbb I(\mathbf x\in R_{mj})
$$
(4) 得到最终模型
$$
H(\mathbf x)=f_M(\mathbf x)
$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GBRT_algorithm.png" style="zoom:80%;" />

**二分类问题**：采取了和逻辑回归同样的方式，即利用sigmoid函数将输出值归一化
$$
H(\mathbf x)=\frac{1}{1+e^{-f_M(\mathbf x)}}
$$
模型的输出为正样本的概率 $\mathbb P(y=1|\mathbf x)$ 。损失函数同样交叉熵损失
$$
\begin{aligned}
L(y,H(\mathbf{x}))=-y\log H(\mathbf{x})-(1-y)\log(1-H(\mathbf{x}))
\end{aligned}
$$
负梯度
$$
-\frac{\partial L(y,H(\mathbf{x}))}{\partial f_M(\mathbf{x})}
=-\frac{\partial L(y,H(\mathbf{x}))}{\partial H(\mathbf{x})}\cdot \frac{\mathrm{d}H(\mathbf{x})}{\mathrm{d}f_M(\mathbf{x})}
=y-H(\mathbf x)
$$
这个负梯度即残差，表示真实值和预测正样本概率的差值。

**多分类问题**：采用softmax函数映射。假设类别的数量为 $K$ ，损失函数为
$$
L=-\sum_{k=1}^K y_k\log\mathbb P(y_k|\mathbf x)
$$
**优缺点**：Boosting共有的缺点为训练是按顺序的，难以并行，这样在大规模数据上可能导致速度过慢，所幸近年来XGBoost和LightGBM的出现都极大缓解了这个问题，后文详述。

### 特征重要性

在数据挖掘应用中，只有一小部分特征变量会对目标变量有显著的影响，研究每个特征变量在预测目标变量时的相对重要性或者贡献是很有用的。

单个决策树本质上通过选择合适的分割点来进行特征选择，这些信息可以用来度量每个特征的重要性。基本思想是：在树的分割点中使用某特征越频繁，该特性就越重要。对于单个决策树 $T$
$$
\mathcal I_\ell^2(T)=\sum_{t=1}^{J-1}\imath_t^2\mathbb I(v(t)=\ell)
$$
作为特征变量 $x_\ell$ 重要性的度量。这个求和是对树的 $J-1$ 个中间结点进行的。在每个中间结点 $t$ ，其中一个特征变量 $x_{v(t)}$ 会将这个结点区域分成两个子区域，每一个子区域用单独的常值拟合目标变量。特征变量的选择要使得在整个区域上有最大的纯度提升 $\imath_t^2$。变量 $x_\ell$ 的**平方相对重要度**（squared relative importance）是在所有的结点中，选择其作为分离变量时纯度提升的平方之和。

这种重要性的概念可以通过简单地平均每个树的基于不纯度的特征重要性来扩展到决策树集成器上
$$
\mathcal I_\ell^2=\frac{1}{M}\sum_{m=1}^M\mathcal I_\ell^2(T_m)
$$
考虑到平均的标准化影响，这个度量会比单个树对应的度量式更稳定。

### XGBoost

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/xgboost.svg" width="25%;" align="right"/> XGBoost（eXtreme Gradient Boosting）是基于GBDT的一种优化算法，于2016年由陈天奇在论文[《 XGBoost：A Scalable Tree Boosting System》](https://arxiv.org/pdf/1603.02754.pdf)中正式提出，在速度和精度上都有显著提升，因而近年来在 Kaggle 等各大数据科学比赛中都得到了广泛应用。

- 进行二阶泰勒展开，优化损失函数，提高计算精度
- 损失函数添加了正则项，避免过拟合
- 采用 Blocks存储结构，可以并行计算

XGBoost 同样为加法模型
$$
f_M(\mathbf x)=\sum_{m=1}^MT_m(\mathbf x)
$$
其中，$T_m(\mathbf x)$ 表示树模型， $M$ 为树的个数。

**目标函数推导**：XGBoost 优化的目标函数由损失函数和正则化项两部分组成
$$
\mathcal L=\sum_{i=1}^N l(y_i,f_M(\mathbf x_i))+\sum_{m=1}^M\Omega(T_m)
$$
其中，$\Omega(T_m)$ 表示第 $m$ 棵树 $T_m$ 的复杂度。

第 $m$ 轮迭代的目标函数为
$$
\mathcal L_m=\sum_{i=1}^N l(y_i,f_{m}(\mathbf x_i))+\sum_{k=1}^m\Omega(T_k)
$$
接下来，分三步简化目标函数。

(1) XGBoost遵从前向分布算法，第 $m$ 轮迭代的模型
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+T_m(\mathbf x)
$$
其中，$f_{m-1}(\mathbf x)$ 是上一轮迭代得到的模型。第 $m$ 轮第 $i$ 个样本的损失函数
$$
l(y_i,f_m(\mathbf x_i))=l(y_i,f_{m-1}(\mathbf x_i)+T_m(\mathbf x_i))
$$
XGBoost 使用**二阶泰勒展开**[^taylor]，损失函数的近似值为
$$
l(y_i,f_m(\mathbf x_i))\approx l(y_i,f_{m-1}(\mathbf x_i))+g_iT_m(\mathbf x_i)+\frac{1}{2}h_iT_m^2(\mathbf x_i)
$$
其中，$g_i$为一阶导，$h_i$ 为二阶导
$$
g_{i}=\left[\frac{\partial l(y_i,f(\mathbf x_i))}{\partial f(\mathbf x_i)}\right]_{f=f_{m-1}},\quad h_{i}=\left[\frac{\partial^2 l(y_i,f(\mathbf x_i))}{\partial^2 f(\mathbf x_i)}\right]_{f=f_{m-1}}
$$
上一轮模型 $f_{m-1}(\mathbf x)$ 已经确定，所以上一轮损失 $l(y_i,f_{m-1}(\mathbf x_i))$ 即为常数项，其对函数优化不会产生影响。移除常数项，所以第 $m$ 轮目标函数可以写成
$$
\mathcal L_m\approx \sum_{i=1}^N \left[g_iT_m(\mathbf x_i)+\frac{1}{2}h_iT_m^2(\mathbf x_i)\right]+\sum_{k=1}^m\Omega(T_k)
$$
(2) 将正则化项进行拆分
$$
\sum_{k=1}^m\Omega(T_k)=\Omega(T_m)+\sum_{k=1}^{m-1}\Omega(T_k)=\Omega(T_m)+\text{constant}
$$
因为 $m-1$ 棵树的结构已经确定，所以可记 $\sum_{k=1}^{m-1}\Omega(T_k)$ 为常数。移除常数项，目标函数可进一步简化为
$$
\mathcal L_m\approx \sum_{i=1}^N \left[g_iT_m(\mathbf x_i)+\frac{1}{2}h_iT_m^2(\mathbf x_i)\right]+\Omega(T_m)
$$
(3) 定义树：沿用之前对树结构的定义。假设树$T_m$ 将样本划分到 $J$ 个互不相交的区域 $R_1,R_2,\cdots,R_J$ ，每个区域 $R_j$ （本质是树的一个分支）对应树的一个叶结点，并且在每个叶节点上有个固定的输出值 $c_j$ 。每个样本只属于其中一个区域，那么树可以表示为 
$$
T_m(\mathbf x;\Theta)=\sum_{j=1}^Jc_j\mathbb I(\mathbf x\in R_j)
$$
参数 $\Theta=\{(R_1,c_1),(R_2,c_2),\cdots,(R_J,c_J)\}$ 表示树的区域划分和对应的值，$J$ 表示叶节点的个数。

然后，定义树的复杂度 $\Omega$ ：包含叶子节点的数量 $J$ 和叶子节点权重向量的 $\ell_2$ 范数
$$
\Omega(T_m)=\gamma J+\frac{1}{2}\lambda\sum_{j=1}^Jc_j^2
$$
这样相当于使叶结点的数目变小，同时限制叶结点上的分数，因为通常分数越大学得越快，就越容易过拟合。

因为每个叶子节点的输出值是相同的，可将目标函数中的样本按叶子节点分组计算，得到最终目标函数
$$
\begin{aligned}
\mathcal L_m &\approx \sum_{j=1}^J \left[(\sum_{\mathbf x_i\in R_j}g_i)c_j+\frac{1}{2}(\sum_{\mathbf x_i\in R_j}h_i+\lambda)c^2_j\right]+\gamma J \\
&=\sum_{j=1}^J \left[G_jc_j+\frac{1}{2}(H_j+\lambda)c^2_j\right]+\gamma J
\end{aligned}
$$
其中
$$
G_j=\sum_{\mathbf x_i\in R_j}g_i,\quad H_j=\sum_{\mathbf x_i\in R_j}h_i
$$

- $G_j$ 是树 $T_m$ 划分的叶子节点 $j$ 所包含的所有样本的**一阶偏导数**之和；
- $H_j$ 是树 $T_m$ 划分的叶子节点 $j$ 所包含的所有样本的**二阶偏导数**之和。

当树 $T_m$ 的区域划分确定时，$G_j,H_j$ 可视为常数。于是，目标函数只包含叶子节点输出值 $c_j$ 。 易知 $H_j+\lambda>0$，目标函数对 $c_j$ 求一阶导数，并令其为 0，可得最优解

$$
c_j^*=-\frac{G_j}{H_j+\lambda}
$$
所以目标函数最优值为
$$
\mathcal L_m^*=-\frac{1}{2}\sum_{j=1}^J\frac{G_j^2}{H_j+\lambda}+\gamma J
$$
下图给出目标函数计算的例子

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/tree_structure_score.png" style="zoom: 50%;" />

**学习树结构**：经过一系列推导后，我们的目标变为确定树 $T_m$ 的结构，然后计算叶节点上的值 $c_j^*$ 。

在树的生成过程中，最佳分裂点是一个关键问题。$\mathcal L_m^*$ 可以作为决策树 $T_m$ 结构的评分函数（scoring function），该值越小，树结构越好。XGboost 支持多种分裂方法：

(1) Exact Greedy Algorithm：现实中常使用贪心算法，遍历每个候选特征的每个取值，计算分裂前后的增益，并选择增益最大的候选特征和取值进行分裂。

XGBoost 提出了一种新的增益计算方法，采用目标函数的分裂增益。类似于CART基尼系数增益，对于目标函数来说，分裂后的增益为 $\mathcal L_{split}=\mathcal L_{before}-\mathcal L_{after}$ 。因此，定义
$$
\text{Gain}=\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}-\gamma
$$

> 上式去除了常数因子 1/2 ，不影响增益最大化。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/split_find.png" style="zoom: 80%;" />

上式中的$\gamma$ 是一个超参数，具有双重含义，一个是对叶结点数目进行控制；另一个是限制树的生长的阈值，当增益大于阈值时才让节点分裂。所以xgboost在优化目标函数的同时相当于做了预剪枝。

(2) Approximate Algorithm：贪心算法可以得到最优解，但当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低。近似算法主要针对贪心算法这一缺点给出了近似最优解，不仅解决了这个问题，也同时能提升训练速度

首先根据特征分布的分位数提出候选划分点，然后将连续型特征映射到由这些候选点划分的buckets中，然后汇总统计信息找到所有区间的最佳分裂点。

对于每个特征，只考察分位点可以减少计算复杂度。近似算法在提出候选切分点时有两种策略：

- Global：学习每棵树前就提出候选切分点，并在每次分裂时都采用这种分割；
- Local：每次分裂前将重新提出候选切分点。

下图给出不同种分裂策略的AUC变化曲线，横坐标为迭代次数，纵坐标为测试集AUC，eps 为近似算法的精度，其倒数为桶的数量。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/xgboost_eps_parameter.png" style="zoom:50%;" />

> Global 策略在候选点数多时（eps 小）可以和 Local 策略在候选点少时（eps 大）具有相似的精度。此外我们还发现，在 eps 取值合理的情况下，分位数策略可以获得与贪婪算法相同的精度。

(3) Weighted Quantile Sketch：加权分位数缩略图。XGBoost不是简单地按照样本个数进行分位，而是以二阶导数值 $g_i$ 作为样本的权重进行划分。实际上，当使用平方损失函数时，我们可以看到$g_i$就是样本的权重。

(4) Sparsity-aware Split Finding：稀疏感知法。实际工程中一般会出现输入值稀疏的情况。比如数据的缺失、one-hot编码都会造成输入数据稀疏。在计算分裂增益时不会考虑带有缺失值的样本，这样就减少了时间开销。在分裂点确定了之后，将带有缺失值的样本分别放在左子树和右子树，比较两者分裂增益，选择增益较大的那一边作为默认分裂方向。

**Shrinkage（收缩率）**：是一种简单的正则化策略，即对每一轮学习的结果乘以因子 $\nu(0<\nu<1)$ 进行缩放，就像梯度下降的学习率，降低单棵树的影响，为后面生成的树留下提升模型性能的空间。于是上文的迭代变为
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+\nu T_m(\mathbf x)
$$
一般学习率$\nu$ 要和弱学习器个数 $M$ 结合起来使用。较小的$\nu$ 值要求较多的弱学习器以保持一个恒定的训练误差。经验证据表明，较小的学习率会有更好的测试误差，并且需要更大的 $M$ 与之对应。建议将学习速率设置为一个小常数（例如，$\nu\leqslant 0.1$)，并通过 early stopping 策略选择 $M$ 。

**Subsampling（子采样）**：随机梯度提升（stochastic gradient boosting）是将梯度提升（gradient boosting）和 bagging 相结合，既能防止过拟合，又能减少计算量。在每次迭代中，基学习器是通过无放回抽取训练集子集拟合。通常设置采样率 $\eta=0.5$ 。

下图表明了shrinkage 与 subsampling 对于模型拟合好坏的影响。我们可以明显看到指定收缩率比没有收缩拥有更好的表现。而将子采样和收缩率相结合能进一步的提高模型的准确率。相反，使用子采样而不使用收缩的结果很差。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GBDT_subsample.png" style="zoom: 80%;" />

**Column Block for Parallel Learning**：（分块并行）：决策树的学习最耗时的一个步骤就是在每次寻找最佳分裂点是都需要对特征的值进行排序。而 XGBoost 在训练之前对根据特征对数据进行了排序，然后保存到块结构中，并在每个块结构中都采用了稀疏矩阵存储格式（Compressed Sparse Columns Format，CSC）进行存储，后面的训练过程中会重复地使用块结构，可以大大减小计算量。

这种块结构存储的特征之间相互独立，方便计算机进行并行计算。在对节点进行分裂时需要选择增益最大的特征作为分裂，这时各个特征的增益计算可以同时进行，这也是 Xgboost 能够实现分布式或者多线程计算的原因。

**优点**：

1. 精度高：GBDT 只用到一阶泰勒展开，而 XGBoost 对损失函数进行了二阶泰勒展开。另外 XGBoost 工具支持自定义损失函数，只要函数可一阶和二阶求导。
2. 灵活性更强：传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这时xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。
3. 正则化：在目标函数中加入了正则项，用于控制模型的复杂度，防止过拟合，从而提高模型的泛化能力。
4. Shrinkage：相当于学习速率。XGBoost 在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。
5. 列抽样：借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算。
6. 节点分裂方式：传统 GBDT 采用的是均方误差作为内部分裂的增益计算指标，XGBoost是经过优化推导后的目标函数。
7. 缺失值处理：在计算分裂增益时不会考虑带有缺失值的样本，这样就减少了时间开销。在分裂点确定了之后，将带有缺失值的样本分别放在左子树和右子树，比较两者分裂增益，选择增益较大的那一边作为默认分裂方向。
8. 并行化：由于 Boosting 本身的特性，无法像随机森林那样树与树之间的并行化。XGBoost 的并行主要体现在特征粒度上，在对结点进行分裂时，由于已预先对特征排序并保存为block 结构，每个特征的增益计算就可以开多线程进行，极大提升了训练速度。

**缺点**：

- 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，但在节点分裂过程中仍需要遍历数据集；
- 预排序过程的空间复杂度过高，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，相当于消耗了两倍的内存。

### LightGBM

[【白话机器学习】算法理论+实战之LightGBM算法](https://cloud.tencent.com/developer/article/1651704)：浅显易懂，讲的特别棒！
[深入理解LightGBM](https://mp.weixin.qq.com/s/zejkifZnYXAfgTRrkMaEww)：原论文精讲

LightGBM（Light Gradient Boosting Machine）是GBDT模型的另一个进化版本， 主要用于解决GBDT在海量数据中遇到的问题，以便更好更快的用于工业实践中。从 LightGBM 名字我们可以看出其是轻量级（Light）的梯度提升机器（GBM）。

 LightGBM 可以看成是XGBoost的升级加强版，它延续了xgboost的那一套集成学习的方式，但是它更加关注模型的训练速度，相对于xgboost， 具有训练速度快和内存占用率低的特点。下面我们就简单介绍下LightGBM优化算法。

**直方图算法**（histogram algorithm）基于Histogram的决策树算法，是替代XGBoost的预排序（pre-sorted）算法的。简单来说，就是把连续的浮点特征值离散化成$k$个整数，形成一个一个的箱体（bins）。并根据特征值所在的bin对其进行梯度累加和个数统计，构造一个宽度为$k$的直方图。然后根据直方图的离散值，遍历寻找最优的分割点。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Histogram_algorithm.png)

对于连续特征来说，分箱处理就是特征工程中的离散化。对于分类特征来说，则是每一种取值放入一个分箱 (bin)，且当取值的个数大于最大分箱数时，会忽略那些很少出现的分类值。

内存占用更小： 在Lightgbm中默认的分箱数 (bins) 为256。XGBoost需要用32位的浮点数去存储特征值，并用32位的整形去存储索引，而 LightGBM只需要用8位去存储直方图，从而极大节约了内存存储。

计算代价更小：相对于 XGBoost 中预排序每个特征都要遍历数据，复杂度为 O(#data\*#featrue) ，而直方图只需要遍历每个特征的直方图，复杂度为 O(#bins\*#featrue) 。而我们知道 #data\>\>#bins

直方图算法还能够做差加速：当节点分裂成两个时，右边叶子节点的直方图等于其父节点的直方图减去左边叶子节点的直方图。从而大大减少构建直方图的计算量。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/hist_diff.png" style="zoom: 67%;" />

**单边梯度采样**（Gradient-based One-Side Sampling，GOSS） ：GOSS算法从减少样本的角度出发，排除大部分小梯度的样本，仅用剩下的样本计算信息增益，它是一种在减少数据量和保证精度上平衡的算法。

我们观察到GBDT中每个数据都有不同的梯度值，样本的梯度越小，样本的训练误差就越小。因此LightGBM提出GOSS算法，它保留梯度大的样本，对梯度小的样本进行随机采样，相比XGBoost遍历所有特征值节省了不少时间和空间上的开销。

算法介绍：首先选出梯度绝对值最大的 $a\times100\%$ 的样本，然后对剩下的样本，再随机抽取 $b\times100\%$ 的样本。最后使用已选的数据来计算信息增益。但是这样会引起分布变化，所以对随机抽样的那部分样本权重放大$(1-a)/b$。

作者通过公式证明了GOSS不会损失很大的训练正确率，并且GOSS比随机采样要好，也就是a=0的情况。

**互斥特征绑定**（Exclusive Feature Bundling，EFB）：高维数据通常是稀疏的，这种稀疏性启发我们设计一种无损的方法来减少特征的维度。

许多特征是互斥的（即特征不会同时为非零值，像one-hot），LightGBM根据这一特点提出了EFB算法将互斥的特征合并成一个特征，从而将特征的维度降下来。相应的，构建histogram的时间复杂度叶从O(\#data\*\#feature) 变为 O(\#data\*\#bundle)  ，这里 \#bundle 是融合绑定后特征包的个数。

**决策树生长策略**：带深度限制的Leaf-wise的叶子生长策略

XGBoost 采用 Level-wise （按层生长）策略，该策略遍历一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上Level-wise是一种低效的算法，因为它不加区分的对待同一层的叶子，实际上很多叶子的分裂增益较低，没必要进行搜索和分裂，因此带来了很多没必要的计算开销。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/level_wise_tree_growth.png)

LightGBM采用 leaf-wise（按叶子生长）策略，以降低模型损失最大化为目的，对当前叶所有叶子节点中分裂增益最大的叶子节点进行分裂。leaf-wise的缺点是会生成比较深的决策树，为了防止过拟合，可以在模型参数中设置树的深度。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/leaf_wise_tree_growth.png)

**直接支持类别特征**

实际上大多数机器学习工具都无法直接支持类别特征，一般需要把类别特征，通过 one-hot 编码，降低了空间和时间的效率。

LightGBM优化了对类别特征的支持，可以直接输入类别特征，不需要额外的0/1展开。LightGBM采用 many-vs-many 的切分方式将类别特征分为两个子集，实现类别特征的最优切分。

算法流程如下图所示，在枚举分割点之前，先把直方图按照每个类别对应的label均值进行排序；然后按照排序的结果依次枚举最优分割点。当然，这个方法很容易过拟合，所以LightGBM里面还增加了很多对于这个方法的约束和正则化。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/LGB_many_vs_many.png" style="zoom: 67%;" />

**并行计算**：LightGBM支持特征并行、数据并行和投票并行

传统的特征并行主要思想是在并行化决策树中寻找最佳切分点，在数据量大时难以加速，同时需要对切分结果进行通信整合。LightGBM则是使用分散规约(Reduce scatter)，将任务分给不同的机器，降低通信和计算的开销，并利用直方图做加速训练，进一步减少开销。

特征并行是并行化决策树中寻找最优划分点的过程。特征并行是将对待特征进行划分，每个worker找到局部的最佳切分点，使用点对点通信找到全局的最佳切分点。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/LGB_feature_parallelization.png" style="zoom: 67%;" />

数据并行的目标是并行化真个决策树学习过程。每个worker中拥有部分数据，独立的构建局部直方图，合并后得到全局直方图，在全局直方图中寻找最优切分点进行分裂。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/LGB_data_parallelization.png" style="zoom: 67%;" />

LightGBM 采用一种称为 PV-Tree 的算法进行投票并行（Voting Parallel），其实本质上也是一种数据并行。PV-Tree 和普通的决策树差不多，只是在寻找最优切分点上有所不同。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/LGB_voting_based_parallel.png" style="zoom:67%;" />

## 投票方法

在训练好基学习器后，如何将这些基学习器的输出结合起来产生集成模型的最终输出，下面将介绍一些常用的结合策略。

对于回归问题，输出基回归模型的平均值。

**简单平均法**（simple averaging）：
$$
H(\mathbf x)=\frac{1}{M}\sum_{m=1}^Mh_m(\mathbf x)
$$
**加权平均法**（weighted averaging）：
$$
H(\mathbf x)=\sum_{m=1}^Mw_mh_m(\mathbf x)\\
\text{s.t.}\quad w_m>0,\sum_{m=1}^Mw_m=1
$$
对于分类问题，最常见的组合方法是硬投票和软投票。类别标签 $y\in\{c_1,c_2,\cdots,c_K\}$ 。

**硬投票**（hard voting）：即多数投票（ majority voting）。基学习器 $h_m$ 输出类别标签 $h_m(\mathbf x)\in\{c_1,c_2,\cdots,c_K\}$，预测结果中出现最多的类别。
$$
H(\mathbf x)=\arg\max_{c_k}\sum_{m=1}^Mh_m(\mathbf x|y=c_k)
$$
例如，给定样本的预测是

| classifier   | class 1 | class 2 | class 3 |
| ------------ | ------- | ------- | ------- |
| classifier 1 | 1       | 0       | 0       |
| classifier 2 | 1       | 0       | 0       |
| classifier 3 | 0       | 1       | 0       |
| sum          | 2       | 1       | 0       |

这里预测的类别为 class 1。

**软投票**（soft voting）：基学习器 $h_m$ 输出类别概率 $h_m(\mathbf x)\in[0,1]$，会选出基学习器的加权平均概率最大的类别。
$$
H(\mathbf x)=\arg\max_c\sum_{m=1}^Mw_mh_m(\mathbf x|y=c)\\
\text{s.t.}\quad w_m>0,\sum_{m=1}^Mw_m=1
$$
用一个简单的例子说明，其中3个分类器的权重相等 

| classifier       | class 1 | class 2 | class 3 |
| ---------------- | ------- | ------- | ------- |
| classifier 1     | 0.2     | 0.5     | 0.3     |
| classifier 2     | 0.6     | 0.3     | 0.1     |
| classifier 3     | 0.3     | 0.4     | 0.3     |
| weighted average | 0.37    | 0.4     | 0.23    |

这里预测的类别为 class 2，因为它具有最高的加权平均概率。

实际中，软投票和硬投票可以得出完全不同的结论。相对于硬投票，软投票考虑到了预测概率这一额外的信息，因此可以得出比硬投票法更加准确的结果。

## Stacking

stacking是指训练一个模型用于组合基学习器的方法，组合的学习器称为元学习器（meta learner）。
$$
H(\mathbf x)=H(h_1(\mathbf x),h_2(\mathbf x),\cdots,h_M(\mathbf x);\Theta)
$$

1. 首先，训练$M$个不同的基学习器，最好每个基学习器都基于不同的算法（KNN、SVM、RF等等），以产生足够的差异性。
2. 然后，每一个基学习器的输出作为组合学习器的特征来训练一个模型，以得到一个最终的结果。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/stacking_algorithm.png" style="zoom:80%;" />

若直接使用基学习器的训练集来生成元学习器的训练集，则过拟合风险会比较大；因此一般通过交叉验证，用基学习器未使用的样本来产生元学习器的训练样本。

以 k-folds 交叉验证为例

1. 初始训练集$D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}$被随机划分为 $k$ 个大小相似的集合 $\{D_1,D_2,\cdots,D_k\}$ 。令 $D_j$ 和 $D-D_j$ 分别表示第 $j$ 折的测试集和训练集。
2. 给定$M$个基学习算法，初级学习器 $h_m^{(j)}$ 通过在 $D-D_j$ 上使用第$m$个学习算法而得。
3. 对 $D_j$ 中每个样本 $\mathbf x_i$，令 $z_{im}=h_m^{(j)}(\mathbf x_i)$ ，则由 $\mathbf x_i$ 产生元学习器的训练集特征 $\mathbf z_i=(z_{i1},z_{i2},\cdots,z_{iM})$，目标值为 $y_i$。
4.  于是，在整个交叉验证过程结束后，从这$M$个基学习器产生的元学习器的训练集是 $D'=\{(\mathbf z_1,y_1),(\mathbf z_2,y_2),\cdots,(\mathbf z_N,y_N)\}$ 。

有研究表明，元学习器通常采用概率作为输入特征，用多响应线性回归（MLR）算法效果较好。

# 附录

## 超平面几何

超平面方程为
$$
\mathbf{w}^T\mathbf{x}+b=0
$$
其中 $\mathbf x=(x_1,x_2,\cdots,x_p)^T$ ，$\mathbf w=(w_1,w_2,\cdots,w_p)^T$ 。

超平面的方程不唯一，同比例缩放 $\mathbf w,b$ ，仍是同一个超平面。若缩放倍数为负数，会改变法向量方向。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/hyperplane.svg)

**法向量**：取超平面上任意两点 $\mathbf x_1,\mathbf x_2$ ，则
$$
\mathbf{w}^T\mathbf{x}_1+b=0 \\
\mathbf{w}^T\mathbf{x}_2+b=0
$$
上面两式相减可得
$$
\mathbf{w}^T(\mathbf{x}_1-\mathbf{x}_2)=\mathbf{w}\cdot(\mathbf{x}_1-\mathbf{x}_2)=0
$$
即向量 $\mathbf w$ 与超平面垂直，所以超平面 $\mathbf{w}^T\mathbf{x}+b=0$ 的法向量为  $\mathbf w$ 。

**任意点到超平面的距离**：取超平面外任意一点 $\mathbf x$ ，在超平面上的投影为 $\mathbf x_1$ ，距离为 $d>0$ ，则
$$
\mathbf{w}\cdot\mathbf{x}_1+b=0 \\
\mathbf{w}\cdot(\mathbf{x}-\mathbf{x}_1)=\|\mathbf w\|\cdot d
$$
因此 $\mathbf x$ 到超平面距离
$$
d=\frac{|\mathbf{w}^T\mathbf{x}+b|}{\|\mathbf w\|}
$$



[^taylor]: 泰勒展开式 $f(x+\Delta x)=f(x)+f'(x)\Delta x+\dfrac{1}{2}f''(x)(\Delta x)^2+\cdots$



> **参考文献**：
> 周志华.《机器学习》（西瓜书）
> 李航.《统计学习方法》
> Andrew Ng.《Machine Learning》.Stanford online
> 《Deep Learning》（花书）
