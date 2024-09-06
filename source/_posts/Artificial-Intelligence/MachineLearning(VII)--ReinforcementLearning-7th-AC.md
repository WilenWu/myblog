---
title: 机器学习(VII)--强化学习(七)Actor-Critic
date: 2024-08-29 17:24
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/reinforcement_learning_cover.png
top_img: /img/artificial-intelligence.jpg
abbrlink: a14a3278
description: 
---

# Actor-Critic

## QAC

Actor-Critic 算法结合了值函数近似和策略函数近似：

- Actor 负责更新策略函数 $\pi(a|s,\theta)$

$$
\theta_{t+1}=\theta_{t}+\alpha_{\theta}\nabla_{\theta}\ln\pi(a_t|s_t,\theta_{t})q_t(s_t,a_t)
$$

- Critic 负责通过Sarsa算法更新值函数 $q(s,a,w)$

$$
w_{t+1}=w_{t}+\alpha_w[r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_{t}) -q(s_t,a_t,w_{t})]\nabla_{w}q(s_t,a_t,w_{t})
$$

基本的 Actor-Critic算法也被称为 Q actor-critic (QAC) ，适合连续状态空间和连续动作空间。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Q_Actor_Critic_Algorithm.png" style="zoom:50%;" />

**路径寻找**：如下图，在一个连续的平面世界中，包括平坦区域、草地和边界。状态空间是连续的 $\mathcal S=[0,6]\times[0,6]$ ，动作空间也是连续的，可以360度自由行走 $\mathcal A=[0,2\pi]$。如果Agent在平坦区域每单位路程奖励 -1 ；在草坪的每单位路程奖励 -2；在目标区域的奖励为 0。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/flat_world_free.svg)

[Jupyter Notebook Code](/ipynb/reinforcement_learning.html#Flat-World)

## A2C

之前我们推导出了带基线的策略梯度

$$
\nabla_{\theta}J(\theta)=\mathbb E_{S\sim d,A\sim\pi}[\delta_{\pi}\nabla_{\theta}\ln\pi(A|S,\theta)]
$$

其中

$$
\delta_{\pi}=q_{\pi}(S,A)-v_{\pi}(S)
$$

被称作**优势函数**(Advantage Function)，如果优势函数大于零，则说明该动作比平均动作好，如果优势函数小于零，则说明当前动作还不如平均动作好。基于上面公式得到的Actor-Critic 方法被称为 Advantage actor-critic (A2C)。

基于贝尔曼方程

$$
q_{\pi}(s,a)-v_{\pi}(s)=\mathbb E[R_{t+1}+\gamma v_{\pi}(S_{t+1})-v_{\pi}(S_t)|S_t=s,A_t=a]
$$

于是策略更新变为

$$
\theta_{t+1}=\theta_{t}+\alpha_{\theta}\delta_t\nabla_{\theta}\ln\pi(a_t|s_t,\theta_{t})
$$

其中

$$
\delta_t= r_{t+1}+\gamma v_{t}(s_{t+1})-v_t(s_t)
$$

值函数 $v(s,w)$ 使用时序差分算法估计

$$
w_{t+1}=w_t+\alpha_w\delta_t\nabla_wv(s_t,w_t)
$$

 算法流程如图

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Advantage_Actor_Critic_Algorithm.png" style="zoom:50%;" />

**Off-policy AC**：若使用行为策略 $\beta$ 采集经验样本，则每步需要乘以重要性权重修正
$$
\delta_t=\frac{\pi(a_t|s_t,\theta_t)}{\beta(a_t|s_t)}[r_{t+1}+\gamma v_{t}(s_{t+1})-v_t(s_t)]
$$

## DPG

确定策略梯度 (Deterministic Policy Gradient, DPG) 是最常用的连续控制方法，顾名思义使用确定性策略

$$
a=\mu(s,\theta)
$$

目标函数

$$
J(\theta)=\mathbb E[v_{\mu}(S)]=\sum_{s\in\mathcal S}d(s)v_{\mu}(s)
$$

则确定性策略梯度

$$
\begin{aligned}
\nabla_{\theta}J(\theta)&=\mathbb E_{S\sim d}[\nabla_{\theta}v_{\mu}(S)] \\
&=\mathbb E_{S\sim d}[\nabla_{\theta}q_{\mu}(S,\mu(S,\theta))] \\
&=\mathbb E_{S\sim d}[\nabla_{\theta}\mu(S,\theta)\nabla_aq_{\mu}(S,a)|_{a=\mu(S,\theta)}]
\end{aligned}
$$

由此我们得到更新 $\theta$ 的随机梯度提升

$$
\theta_{t+1}=\theta_{t}+\alpha_{\theta}\nabla_{\theta}\mu(s_t,\theta_t)\nabla_aq_{\mu}(s_t,a)|_{a=\mu(s_t,\theta_t)}
$$

使用时序差分近似值函数 $q(s,a,w)$ 的更新公式为

$$
w_{t+1}=w_{t}+\alpha_w[r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_{t}) -q(s_t,a_t,w_{t})]\nabla_{w}q(s_t,a_t,w_{t})
$$

算法流程如图 

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Deterministic_Policy_Gradien_Algorithm.png" style="zoom:50%;" />
