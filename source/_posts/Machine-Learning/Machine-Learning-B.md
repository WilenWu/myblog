---
title: 机器学习导论(上册)
katex: true
categories:
  - 机器学习
  - 'Machine Learning'
tags:
  - 机器学习
cover: /img/data-analysis.png
top_img: /img/data-chart3.jpg
description: 机器学习算法导论
abbrlink: 
date: 2018-05-09 21:31:41
---


<!-- more -->

# 引言

机器学习三要素：模型，学习准则(Loss)，优化算法(Optimizer)。

损失函数(loss function)或代价函数(cost function)

克罗内克函数（Kronecker delta）
$$
\delta_{ij}=\begin{cases}
0 & \text{if } i\neq j  \\
1 & \text{if } i= j
\end{cases}
$$


# 模型评估与选择

# 线性回归

给定样本集 $D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_m,y_m)\}$，其中 $\mathbf x_i=(x_i^{(1)},x_i^{(2)},\cdots,x_i^{(d)})^T$ 为单样本特征向量。

拟合回归方程
$$
\begin{align}
f(\mathbf x) &=w_0+w_1x^{(1)}+w_2x^{(2)}+\cdots+w_dx^{(d)} \\
&=\sum_{i=0}^{d}w_ix^{(2)}=\mathbf w^T\mathbf x
\end{align}
$$
$\mathbf x=(1,x^{(1)},x^{(2)},\cdots,x^{(d)})^T,\quad \mathbf w=(w_0,w_1,w_2,\cdots,w_d)^T$ ，其中 $w_0$ 为偏置项(bias) 

误差：对于每个样本
$$
e_i=y_i-\mathbf w^T\mathbf x_i
$$
**Gauss-Markov假设**：

1.   $e∼N(0, σ^2)$
2.   $\mathrm{var}(e)= σ^2$ , 误差同分布
3.   $\mathrm{cov}( e_i ,e_j )=0 (i  ≠ j)$ , 误差独立性 

损失函数：
$$
L=\frac{1}{2}\|\mathbf{Xw-y}\|_2^2=\frac{1}{2}(\mathbf{Xw-y})^T(\mathbf{Xw-y})
$$
算法：最小二乘法（,MLS）、最大似然估计（）
