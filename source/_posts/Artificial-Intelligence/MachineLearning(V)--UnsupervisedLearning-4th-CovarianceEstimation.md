---
title: 机器学习(V)--无监督学习(四)协方差估计
date: 
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-unsupervised-learning.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 5a518d6f
description:
---

# 协方差估计

对于 $p$ 维多元正态分布 $\mathbf x\sim N(\mathbf\mu,\mathbf\Sigma)$ 的概率密度函数为
$$
f(\mathbf x)=\frac{1}{\sqrt{(2\pi)^p|\mathbf\Sigma|}}\exp\left(-\frac{1}{2}(\mathbf x-\mathbf\mu)^T\mathbf\Sigma^{-1}(\mathbf x-\mathbf\mu)\right)
$$
其中 $|\mathbf\Sigma|$ 表示协方差矩阵的行列式。$d(\mathbf x,\mathbf\mu)=(\mathbf x-\mathbf\mu)^T\Sigma^{-1}(\mathbf x-\mathbf\mu)$ 为数据点 $\mathbf x$ 与均值之间 $\mathbf\mu$ 的 Mahalanobis 距离。