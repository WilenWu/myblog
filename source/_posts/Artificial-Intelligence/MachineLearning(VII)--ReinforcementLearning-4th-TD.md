---
title: 机器学习(VII)--强化学习(四)时序差分方法
date: 2024-08-29 17:21
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/reinforcement_learning_cover.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 363ac257
description: 
---

# 时序差分方法

## 基本思路

蒙特卡罗方法一般需要拿到完整的轨迹，才能对策略进行评估并更新模型，因此效率也比较低。**时序差分学习**（Temporal-Difference Learning）方法是蒙特卡罗方法的一种改进，其基本思想就是将蒙特卡罗的批量式更新改为增量式更新，在生成轨迹中，每执行一步策略后就进行值函数更新，做到更高效的免模型学习。

蒙特卡罗方法值函数的估计

$$
v(s)=\mathbb E[G_t|S_t=s]=\frac{1}{n}\sum_{k=1}^ng_k(s)
$$

若直接根据上式计算平均奖赏，则需记录 $n$ 个奖赏值。显然，更高效的做法是对均值进行增量式计算，即每执行一步就立即更新 $v(s)$。不妨用下标来表示执行的次数，初始时 $v_1(s)=0$

$$
v_{k+1}(s)=v_k(s)+\frac{1}{k}[g_k(s)-v_k(s)]
$$

更一般的，将 $1/k$ 替换为步长 $\alpha$ 

$$
v_{k+1}(s)=v_k(s)+\alpha[g_k(s)-v_k(s)]
$$

显然，下一轮的估计值是本轮估计值和样本值的加权平均

$$
v_{k+1}(s)=(1-\alpha)v_k(s)+\alpha g_k(s)
$$

更新步长 $\alpha$ 越大，则越靠后的累积奖赏越重要。可以证明 $v_k(s)$ 收敛

$$
\lim\limits_{k\to\infty} v_k(s)=\mathbb E[v(s)]
$$

由于 $g_k$ 为一次试验的完整轨迹所得到的总回报。为了提高效率，我们可以使用贝尔曼方程

$$
v_{\pi}(s)=\mathbb E[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_t=s]
$$

来近似估计。即使用样本集 $\{s_t,r_{t+1},v(s_{t+1})\}_{k=1}^n$ 增量更新

$$
\underbrace{v(s_t)}_{\text{new estimate}}\gets \underbrace{v(s_t)}_{\text{current estimate}}+\alpha[\overbrace{\underbrace{r_{t+1}+\gamma v(s_{t+1})}_{\text{TD target}}-v(s_t)}^{\text{TD error}}]
$$

其中真实观测称为 **TD target**

$$
\tilde v_t=r_{t+1}+\gamma v(s_{t+1})
$$

模型估计与真实观测之差称为 **TD error**

$$
\delta=\tilde v_t-v(s_t)=r_{t+1}+\gamma v(s_{t+1})-v(s_t)
$$

TD 算法的目的是通过迭代使得TD error最小，这实际上是损失函数为 $(v_t-v(s_t))^2$ 的单样本随机梯度下降。

相比于蒙特卡洛方法，TD算法是增量式的，它不需要等到整条轨迹采集完成才能计算，而是在接收到一个经验样本 (experience sample) 后能立即更新值函数。

## SARSA

动作值函数 $q(s,a)$ 的贝尔曼方程为

$$
q_{\pi}(s,a)=\mathbb E[R_{t+1}+\gamma q_{\pi}(S_{t+1},A_{t+1})|S_t=s,A_t=a],\quad \forall (s,a)
$$

这里我们学习的不是状态价值函数，而是动作价值函数 $q(s,a)$。更新该值的方法跟更新状态价值 $v(s,a)$的方式一致：

$$
q(s_t,a_t)\gets q(s_t,a_t)+\alpha[r_{t+1}+\gamma q(s_{t+1},a_{t+1})-q(s_t,a_t)]
$$

这种策略学习方法称为SARSA算法，其名称代表了每次更新值函数时需知道数据 $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$ 。

算法流程如下图：

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Sarsa_Algorithm.png" style="zoom:50%;" />

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/an_example_for_demonstrating_Sarsa.png" style="zoom:50%;" />

时序差分学习方法和蒙特卡罗方法的主要不同为：蒙特卡罗方法需要一条完整的路径才能知道其总回报，也不依赖马尔可夫性质；而时序差分学习方法只需要一步，其总回报需要通过马尔可夫性质来进行近似估计。

**n-step Sarsa**：回顾下 action value的定义

$$
q_{\pi}(s,a)=\mathbb E[G_t|S_t=s,A_t=a]
$$

实际上折扣回报可以分解为不同的格式

$$
\begin{aligned}
G_t&=R_{t+1}+\gamma q_{\pi}(S_{t+1},A_{t+1}) \\
&=R_{t+1}+\gamma R_{t+2}+\gamma^2 q_{\pi}(S_{t+2},A_{t+2}) \\
&=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^n q_{\pi}(S_{t+n},A_{t+n}) \\
&=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+2}+\cdots
\end{aligned}
$$

当 $n=1$ 时，便是Sarsa算法，当 $n=\infty$ 时，则对应了MC算法，对于一般 $n$值对应的算法称为 n-step Sarsa。

## Q-Learning

Q-learning 是一种 off-policy 的时序差分学习方法。与SARSA算法不同的是，Sarsa的目标是估计给定策略的动作值函数，而Q-learning目标是直接估计最优动作值函数，因此更新后的q-value 是关于目标策略的。

动作值函数的贝尔曼最优方程为

$$
q_{\pi}(s,a)=\mathbb E[R_{t+1}+\gamma q_{\pi}\max_{a'}q(S_{t+1},a')|S_t=s,A_t=a],\quad \forall (s,a)
$$

Q-learning的单个样本的更新规则

$$
q(s_t,a_t)\gets q(s_t,a_t)+\alpha[r_{t+1}+\gamma \max_{a'} q(s_{t+1},a')-q(s_t,a_t)]
$$

off-policy 算法流程如下图：

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Q-learning_off-polic.png" style="zoom:50%;" />

[Jupyter Notebook Code](/ipynb/reinforcement_learning.html#Q-Learning)


# 附录

## 悬崖行走问题

**悬崖行走问题**：(Cliff Walking) 在一个网格世界中，每个网格表示一个状态。其中网格 $(2,1)$ 到 $(11,1)$ 是悬崖(Cliff)。有一个Agent，目标是如何安全地从左下角的开始位置 $S$，走到右下角的目标位置 $G$。Agent 在每一个状态都可以选择4种行走方向：⬆️ ⬅️ ⬇️ ➡️。但每走一步，都有一定的概率滑落到周围其他格子。如果掉入悬崖或到达目标状态都会结束动作并回到起点，也就是说掉入悬崖或者达到目标状态是终止状态。Agent 每走一步的奖励是 −1，掉入悬崖的奖励是 −100，到达终止状态奖励为 0。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/cliff_walking.svg" />

[Jupyter Notebook Code](/ipynb/reinforcement_learning.html#Cliff-Walking)
