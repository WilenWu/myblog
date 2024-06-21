---
title: 机器学习(V)--无监督学习(七)核密度估计
date: 2023-06-07 22:20
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-unsupervised-learning.png
top_img: /img/artificial-intelligence.jpg
abbrlink: a6d4f9a8
description:
---

# 核密度估计

**核密度估计**（kernel density estimate，kde）：是一种用于估计概率密度函数的非参数方法，可看作直方图的拟合曲线。

我们知道，对概率密度函数（Probability Density Function，PDF）$f(x)$ 的定义为
$$
\mathbb P(a<x\leqslant b)=\int_a^b f(x)\mathrm{d} x
$$
则累积分布函数（Cumulative Distribution Function，CDF）$F(x)$ 为
$$
F(x)=\int_{-\infty}^x f(x)\text{d} x=\mathbb P(X\leqslant x)
$$
概率密度函数就是概率分布函数的一阶导数，根据微分思想，则有
$$
f(x_0)=F'(x_0)=\lim_{h\to 0}\frac{F(x_0+h)-F(x_0-h)}{2h}
$$
引入经验累积分布函数，来近似描述概率 $\mathbb P(X\leqslant x)$
$$
F_n(x)=\frac{1}{n}\sum_{i=1}^n\mathbb I(x_i\leqslant x)
$$
其中 $\mathbb I$ 为指示函数。于是有
$$
f(x) =\lim_{h\to 0}\frac{1}{2nh}\sum_{i=1}^n\mathbb I(x-h\leqslant x_i\leqslant x+h)
$$
即在 $x$ 的邻域 $[x-h,x+h]$ 内样本频率估计概率密度。在实际计算中，必须给定 $h$ 值， $h$ 值不能太大也不能太小。太大不满足 $h\to 0$ 的条件，太小使用的样本数据点太少，误差会很大。因此，关于 $h$ 值的选择有很多研究，该值也被称为核密度估计中的带宽（bandwidth）。

**核函数**：确定带宽后，上式可改写为
$$
\begin{aligned}
f_h(x) & =\frac{1}{2nh}\sum_{i=1}^n\mathbb I(x-h\leqslant x_i\leqslant x+h) \\
& =\frac{1}{2nh}\sum_{i=1}^n\mathbb I(|x-x_i|\leqslant h) \\
& =\frac{1}{2nh}\sum_{i=1}^n\mathbb I(\frac{|x-x_i|}{h}\leqslant 1) 
\end{aligned}
$$
记 $K(t)=\frac{1}{2}\mathbb I(t\leqslant 1)$ 则概率密度函数变为
$$
f_h(x)=\frac{1}{nh}\sum_{i=1}^nK(\frac{|x-x_i|}{h})
$$
其中 $K(t)$ 称为**核函数**。概率密度函数的积分
$$
\begin{aligned}
\int f_h(x)\mathrm dx &= \frac{1}{nh}\sum_{i=1}^n\int K(\frac{|x-x_i|}{h})\mathrm dx \\
&=\frac{1}{n}\sum_{i=1}^n\int K(t)dt \\
&=\int K(t)dt
\end{aligned}
$$
因而只要核函数的积分等于1，就能保证估计出来的密度函数积分等于1。因此，我们考虑使用已知的概率密度函数作为核函数，几种常用的核函数有

Gaussian kernel $K(x;h)\propto \exp(-\frac{x^2}{2h^2})$

Tophat kernel $K(x;h)\propto 1$ if $x<h$

Epanechnikov kernel $K(x;h)\propto 1-\frac{x^2}{h^2}$

Exponential kernel $K(x;h)\propto \exp(-\frac{x}{h})$

Linear kernel $K(x;h)\propto 1-\frac{x}{h}$  if $x<h$

Cosine kernel $K(x;h)\propto \cos(\frac{\pi x}{2h})$  if $x<h$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/kde_kernels.png" style="zoom:80%;" />

通常使用高斯核，即标准正态分布的密度函数。直觉上，高斯核就是一个加权平均，离$x$越近的$x_i$其权重越高。而最开始的估计方式则是在区间内权重相等，区间外权重为0。

**带宽**：作为一个平滑参数，平衡结果中的偏差和方差。大带宽导致非常平滑（即高偏差）的密度分布，小带宽导致不平滑（即高方差）的密度分布。如果使用高斯核函数，理论上的最优带宽为
$$
h=\left(\frac{4\sigma^5}{3n}\right)^{1/5}\approx 1.06 \sigma n^{1/5}
$$
其中，$\sigma$为样本标准差。这种近似称为正态分布近似。