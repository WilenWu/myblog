---
title: 机器学习--半监督学习
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/data-analysis.png
top_img: /img/data-chart3.jpg
description: 机器学习算法导论
abbrlink: 6f17ebe2
date: 2018-05-09 21:31:41
---

# 线性模型

## 基本形式

对有序（order）分类变量，可通过连续化将其转化为连续值，对于$k$分类变量可转化为 $k$ 维0-1向量

## 多元线性回归

在预测任务中，给定样本集 $D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_m,y_m)\}$，其中$\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{id})$ 为第 $i$ 个样本的特征向量，$y_i\in \R$ 是目标变量。

**模型假设**：拟合多元线性回归模型（multivariate linear regression）
$$
\hat y=w_0+w_1x_1+w_2x_2+\cdots+w_dx_d = \mathbf{w}^T\mathbf{\hat x}
$$
其中 $\mathbf{\hat x}=(1,x_1,x_2,\cdots,x_d)^T$， 特征向量权重 $\mathbf{w}=(w_0,w_1,\cdots,w_d)^T, w_0$ 为偏置项(bias) 。求得 $\mathbf{w}=(w_0,w_1,\cdots,w_d)^T$ 后，模型就得以确定。$\mathbf w$ 可以直观表达了各属性在预测中的重要性，因此线性模型有很好的可解释性(comprehensibility) 。

对于每个样本，真实值 $y_i$ 和预测值 $\hat y_i$ 间存在误差 $e_i=y_i-\hat y_i$ ，矩阵形式为
$$
\mathbf e =\mathbf y-\mathbf{\hat y}=\mathbf{y- Xw}
$$
其中 
$$
\mathbf{X}=\begin{pmatrix}
1&x_{11}&x_{12}&\cdots&x_{1d} \\
1&x_{21}&x_{22}&\cdots&x_{2d} \\
\vdots&\vdots&\ddots&\vdots \\
1&x_{m1}&x_{m2}&\cdots&x_{md} \\
\end{pmatrix},
\quad \mathbf{w}=(w_0,w_1,\cdots,w_d)^T
$$
**假设条件**：误差满足Gauss-Markov假设

1. 误差满足高斯分布 $e_i∼N(0, σ^2)$
2. 误差同分布 $\mathrm{var}(e_i)= σ^2$ ,
3. 误差独立性 $\mathrm{cov}( e_i ,e_j )=0 \quad (i  ≠ j)$ 

**最小二乘法**：(least square method, LSM) 我们的目标是通过减小响应变量的真实值和预测值的差值来获取模型参数（偏置项和特征向量权重）。具体而言，即使用均方误差定义代价函数
$$
J(\mathbf{w})=\cfrac{1}{m}\|\mathbf{y-Xw}\|_2^2=\cfrac{1}{m}(\mathbf{y-Xw})^T(\mathbf{y-Xw})
$$
$$
\mathbf w^*=\argmin_{\substack{\mathbf w}}(\mathbf{y-Xw})^T(\mathbf{y-Xw})
$$
**极大似然估计**：(maximum likelihood estimate, MLE) 使得观测样本出现的概率最大，也即使得误差联合概率（似然函数）取得最大值。

由于误差满足高斯分布，单样本误差概率为
$$
P(e_i)=\cfrac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{e_i^2}{2\sigma^2})=\cfrac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y_i-\hat y_i)^2}{2\sigma^2})
$$
为求解方便，对误差联合概率取对数似然函数
$$
\begin{aligned}
\displaystyle\ln L(\mathbf w) & =\ln\prod_{i=1}^{m} P(e_i)=\sum_{i=1}^m\ln P(e_i) \\
&=\sum_{i=1}^{m}\ln\cfrac{1}{\sqrt{2\pi}\sigma}+\sum_{i=1}^{m}\ln\exp(-\frac{(y_i-\hat y_i)^2}{2\sigma^2}) \\
&=\sum_{i=1}^{m}\ln\cfrac{1}{\sqrt{2\pi}\sigma}-\frac{1}{2\sigma^2}(\mathbf{y-Xw})^T(\mathbf{y-Xw})
\end{aligned}
$$
因此可定义代价函数
$$
J(\mathbf w)=\arg\max\ln L(\mathbf w)=\arg\min(\mathbf{y-Xw})^T(\mathbf{y-Xw})
$$
最后得到的代价函数与最小二乘法一致。

**参数估计** ：(parame estimation) 使用凸优化方法求解代价函数，即
$$
\nabla J(\mathbf w)=\frac{\partial J}{\partial\mathbf x}=2\mathbf X^T(\mathbf{Xw-y})=\mathbf 0
$$
且 Hessian 矩阵 $\nabla^2 J(\mathbf w)$ 正定。可求得最优解
$$
\mathbf w^*=(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y
$$
最终学得线性回归模型为
$$
\hat y=\mathbf{\hat x}^T(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y
$$
然而，现实任务中 $\mathbf X^T\mathbf X$ 往往不是满秩矩阵。例如在许多任务中我们会遇到大量的变量，其数目甚至超过样例数，导致 $\mathbf X$ 的列数多于行数， $\mathbf X^T\mathbf X$ 显然不满秩。此时可解出多个解， 它们都能使均方误差最小化。选择哪一个解作为输出，将由学习算法的归纳偏好决定， 常见的做法是引入正则化(regularization) 项。

## 广义线性回归

许多功能更为强大的非线性模型(nonlinear model)可通过引入层级结构或高维映射转化为线性模型。例如，对数线性回归 (log-linear regression)
$$
\ln \hat y=\mathbf{w}^T\mathbf{\hat x}
$$
![](Machine-Learning-B.assets/log-linear.png)

更一般地，考虑单调可微函数 $z=g(y)$，令 
$$
y=g^{-1}(\mathbf{w}^T\mathbf{\hat x})
$$
这样得到的模型称为广义线性模型 (generalized linear model)，其中函数 $g(z) $ 称为联系函数 (link function)。广义线性模型的参数估计常通过加权最小二乘法或极大似然估计。

## 逻辑回归

$$
\hat y=g(\mathbf{w}^T\mathbf{x}+b),\text{where }g(z)=\cfrac{1}{1+e^{-z}} \\
\text{Given }\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_m,y_m)\},\text{want }\hat y^{(i)}\approx y^{(i)}
$$



假设给定二分类样本集 $D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_m,y_m)\}$，其中$\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{id})$ 为第 $i$ 个样本的特征向量，输出标记 $y_i\in \{0,1\}$ 。

由于线性回归模型产生的预测值 $z=\mathbf{w}^T\mathbf{\hat x} \in\R$ ，需要引入 Sigmod 函数将输入值映射到 $[0,1]$ 来实现分类功能。对数几率函数 (logistic function) 即是一种 Sigmoid 函数
$$
y=\cfrac{1}{1+e^{-z}}
$$
![](Machine-Learning-B.assets/logistic-function.png)

即若预测值 $z$ 大于零就判为正例，小于零则判为反例，预测值为临界值零则可任意判别。

**模型假设**：拟合逻辑回归模型 (logistic regression，logit regression)
$$
\hat y=\cfrac{1}{1+e^{-\mathbf{w}^T\mathbf{\hat x}}}
$$
其中 $\mathbf{\hat x}=(1,x_1,x_2,\cdots,x_d)^T$，特征向量权重 $\mathbf{w}=(w_0,w_1,\cdots,w_d)^T, w_0$ 为偏置项(bias) 。

输出值可视为正样本的概率 $P_1(y_i=1)=\hat y$，则负样本的概率 $P_0(y_i=0)=1-\hat y$。可联合写为
$$
P(y_i)=P_1^{y_i}P_0^{1-y_i}
$$
**极大似然估计**：(maximum likelihood estimate, MLE) 使得观测样本出现的概率最大，也即使得样本联合概率（似然函数）取得最大值。为求解方便，取对数似然函数
$$
\ln L(\mathbf w)=\ln\prod_{i=1}^{m} P(y_i)=\sum_{i=1}^m(y_i\mathbf w^T\mathbf{\hat x}_i-\ln(1+e^{\mathbf w^T\mathbf{\hat x}_i}))
$$
因此代价函数
$$
J(\mathbf w)=\arg\min\sum_{i=1}^m(-y_i\mathbf w^T\mathbf{\hat x}_i+\ln(1+e^{\mathbf w^T\mathbf{\hat x}_i}))
$$
**最大期望算法**：（Expectation-Maximization algorithm, EM）与真实分布最接近的模拟分布即为最优分布，因此可以通过最小化交叉熵来求出最优分布。

真实分布可写为 
$$
P(y_i)=1
$$
模拟分布可写为 
$$
Q(y_i)=P_1^{y_i}P_0^{1-y_i}
$$
交叉熵为
$$
H(P,Q)=-\sum_{i=1}^m P(y_i)\ln Q(y_i)=\sum_{i=1}^m(-y_i\mathbf w^T\mathbf{\hat x}_i+\ln(1+e^{\mathbf w^T\mathbf{\hat x}_i}))
$$
因此代价函数
$$
J(\mathbf w)=\arg\min\sum_{i=1}^m(-y_i\mathbf w^T\mathbf{\hat x}_i+\ln(1+e^{\mathbf w^T\mathbf{\hat x}_i}))
$$
与极大似然估计的代价函数相同。

《机器学习方法》第6章

《深度学习》5.5

**优化算法**：常用梯度下降法 (gradient descent method)、拟牛顿法 (quasi Newton method) 估计参数

## 线性判别分析

假设给定二分类样本集 $D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_m,y_m)\}$，其中$\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{id})$ 为第 $i$ 个样本的特征向量，输出标记 $y_i\in \{0,1\}$ 。

线性判别分析（Linear Discriminant Analysis，LDA）亦称 Fisher 判别分析。其基本思想是：将训练样本投影到一条直线上，使得同类的样例尽可能近，不同类的样例尽可能远。如图所示：

![](Machine-Learning-B.assets/LDA.png)

## 多分类学习

## 数据平衡

# 决策树

## 基本流程

## 核心方法

划分选择：（信息论：自信息，信息熵）、信息增益、信息增益率、基尼系数

剪枝处理：预剪枝、后剪枝

连续与缺失值

《统计-5》《花书-3.13》信息论







