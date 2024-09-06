---
title: 机器学习(VII)--强化学习(六)策略梯度方法
date: 2024-08-29 17:23
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/reinforcement_learning_cover.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 3644c10c
description: 
---

# 策略梯度方法

## 基本思路

策略梯度(Policy Gradients)方法特别适用于处理高维或连续的动作和状态空间，而这些在基于值的方法中通常很难处理。

From table to function

$$
\pi(a|s,\theta)\approx \pi(a|s)
$$

如果一个策略很好，那么对于所有的状态 $s$，状态价值 $v_\pi(s)$ 的均值应当很大。因此我们的目标是最大化

$$
J(\theta)=\mathbb E_{S\sim d(S)}[v_{\pi}(S)] 
=\sum_{s\in\mathcal S}d(s)v_{\pi}(s) 
$$

从上式可以看出，目标函数只和策略函数相关。其中 $d(s)$ 是稳态分布(stationary distribution)。向量 $\mathbf d_{\pi}=[d_{\pi}(s_1),\cdots,d_{\pi}(s_n)]^T$ 满足

$$
\mathbf d_{\pi}^T=\mathbf d_{\pi}^TP_{\pi}\quad \text{where }(P_{\pi})_{ij}=\mathbb P(s_j|s_i)
$$

可以使用梯度上升来最大化目标函数

$$
\nabla_{\theta}J(\theta)=\sum_{s\in\mathcal S}d(s)\sum_{a\in\mathcal A}\nabla_{\theta}\pi(a|s,\theta)q_{\pi}(s,a)\\
$$

因为

$$
\nabla_{\theta}\ln\pi(a|s,\theta)=\frac{\nabla_{\theta}\pi(a|s,\theta)}{\pi(a|s,\theta)}
$$

带入目标梯度可得到

$$
\nabla_{\theta}J(\theta)=\mathbb E_{S\sim d(S),A\sim\pi(A|S,\theta)}[\nabla_{\theta}\ln\pi(A|S,\theta)q_{\pi}(S,A)]
$$

由于上式要求 $\pi>0$ ，策略函数一般使用 softmax functions，是一种探索性策略。

$$
\pi(a|s,\theta)=\frac{\exp(h(s,a,\theta))}{\sum_{a'\in\mathcal A}\exp(h(s,a',\theta))},\quad a\in\mathcal A
$$

## REINFORCE

我们可以使用蒙特卡洛近似估计值函数 $q(s,a)$，并结合随机梯度上升算法，更新策略参数，这称为REINFORCE算法。

$$
\theta_{t+1}=\theta_{t}+\alpha\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/reinforce_algorithm.png" style="zoom:50%;" />

## REINFORCE with baseline

REINFORCE 算法的一个主要缺点是不同路径之间的方差很大，导致训练不稳定，这是在高维空间中使用蒙特卡罗方法的通病。一种减少方差的通用方法是引入一个和动作 $A$ 无关的baseline减小梯度的方差。

$$
\mathbb E_{S\sim d,A\sim\pi}[\nabla_{\theta}\ln\pi(A|S,\theta)q_{\pi}(S,A)]=\mathbb E_{S\sim d,A\sim\pi}[\nabla_{\theta}\ln\pi(A|S,\theta)(q_{\pi}(S,A)-b(S))]
$$

最优baseline

$$
b^*(s)=\frac{\mathbb E_{A\sim\pi}[\|\nabla_{\theta}\ln\pi(A|s,\theta_t)\|^2q_{\pi}(s,A)]}{\mathbb E_{A\sim\pi}[\|\nabla_{\theta}\ln\pi(A|s,\theta_t)\|^2},\quad s\in\mathcal S
$$

目标函数为过于复杂，一般使用

$$
b(s)=\mathbb E_{A\sim\pi}[q_{\pi}(s,A)]=v_{\pi}(s),\quad s\in\mathcal S
$$

随机梯度提升

$$
\theta_{t+1}=\theta_{t}+\alpha\nabla_{\theta}\ln\pi(a_t|s_t,\theta_{t})\delta_t(s_t,a_t)
$$

其中

$$
\delta_t(s_t,a_t)=q_t(s_t,a_t)-v_t(s_t)
$$