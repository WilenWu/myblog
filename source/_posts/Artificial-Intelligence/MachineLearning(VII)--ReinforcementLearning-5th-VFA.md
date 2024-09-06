---
title: 机器学习(VII)--强化学习(五)值函数近似
date: 2024-08-29 17:22
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/reinforcement_learning_cover.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 2bd55622
description: 
---

# 值函数近似

前面我们一直假定强化学习任务是在有限状态空间上进行，值函数则使用有限状态的表来表示。然而，现实强化学习任务，所面临的状态空间往往是连续的，有无穷多个状态，显然无法用表格来记录。我们不妨直接学习连续状态空间的连续值函数。由于此时的值函数难以像有限状态那样精确记录每个状态的值，因此这样值函数的求解被称为**值函数近似** (value function approximation)。

下面介绍一个简单的示例，它能帮助我们更加理解连续空间的强化学习过程。

**路径寻找**：如下图，在一个连续的平面世界中，包括平坦区域、草地和边界。状态空间是连续的 $\mathcal S=[0,6]\times[0,6]$ ，但动作空间是离散的，仍然只有5种行走方向 $\mathcal A=\{\uparrow,\leftarrow,\downarrow,\rightarrow,\circlearrowleft\}$。如果Agent在平坦区域每单位路程奖励 -1 ；在草坪的每单位路程奖励 -2；在目标区域的奖励为 0。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/flat_world.svg)

**From table to function**

$$
\hat v(s,w)\approx v_{\pi}(s)
$$

我们需要学习一个参数 $w$ 来使得函数 $\hat v(s,w)$ 可以逼近值函数 $v_{\pi}(s)$ 。常用期望误差来作为损失函数：

$$
J(w)=\mathbb E[(\hat v(S,w)-v_{\pi}(S))^2]
$$

为了使误差最小化，采用梯度下降法

$$
\nabla_wJ(w)=\mathbb E[2(\hat v(s,w)-v_{\pi}(s))\nabla_w\hat v(s,w)]
$$

于是可得到对于单个样本随机梯度下降的更新规则

$$
w_{t+1}=w_t+\alpha[v_{\pi}(s_t)-\hat v(s_t,w_t)]\nabla_{w}\hat v(s_t,w_t)
$$

我们并不知道策略的真实值函数 $v_{\pi}$。如果采用蒙特卡罗方法近似总回报 $G_t$

$$
w_{t+1}=w_t+\alpha[g_t-\hat v(s_t,w_t)]\nabla_{w}\hat v(s_t,w_t)
$$

如果采用时序差分学习方法近似 $R_{t+1}+\gamma v(S_{t+1})$

$$
w_{t+1}=w_t+\alpha[r_{t+1}+\gamma \hat v(s_{t+1},w_t) -\hat v(s_t,w_t)]\nabla_{w}\hat v(s_t,w_t)
$$

> 注意：上式中看到了离散化的时刻，可看作 agent 与环境每隔 $\delta t$ 时间交互一次。

**线性函数**：值函数一般采用线性函数估计
$$
\hat v(s,w)=w^T\phi(s)
$$

例如在平面世界中估计值函数

$$
\hat v(s,w)=ax+by+c=
\begin{bmatrix}a&b&c\end{bmatrix}
\begin{bmatrix}x\\y\\1\end{bmatrix}
$$

其中 $\phi(s)=[x,y,1]^T$ 称为特征向量 (feature vector)，$w=[a,b,c]$ 称为参数向量 (parameter vector)。 

线性函数使用SGD来更新参数时非常方便，梯度 

$$
\nabla_{w}\hat v(s,w)=\phi(s)
$$

单个样本随机梯度下降

$$
w_{t+1}=w_t+\alpha[r_{t+1}+\gamma w_t^T\phi(s_{t+1})- w_t^T\phi(s_{t})]\phi(s_{t})
$$

下面是基于线性值面数估计的例子

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/TD-Linear_example_p1.png" style="zoom:50%;" />

n-order多项式估计

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/TD-Linear_example_p2.png" style="zoom:50%;" />

**Q-learning**：TD 算法估计的是动作值函数

$$
\hat q(s,a,w)\approx q_{\pi}(s,a)
$$

更新参数的一般形式是

$$
w_{t+1}=w_t+\alpha[r_{t+1}+\gamma \max_{a\in\mathcal A}\hat q(s_{t+1},a,w_t) -\hat q(s_t,a_t,w_t)]\nabla_{w}\hat q(s_t,a_t,w_t)
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Q-learning_with_VF.png" style="zoom:50%;" />