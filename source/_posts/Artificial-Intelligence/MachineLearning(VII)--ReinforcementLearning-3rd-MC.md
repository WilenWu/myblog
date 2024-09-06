---
title: 机器学习(VII)--强化学习(三)蒙特卡洛方法
date: 2024-08-29 17:20
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/reinforcement_learning_cover.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 5d47a626
description: 
---

# 蒙特卡洛方法

在现实的强化学习任务中，模型参数（状态转移函数与奖赏函数）往往很难得知，因此我们一般需要智能体和环境进行交互，并收集一些样本，然后再根据这些样本来求解马尔可夫决策过程最优策略，这便是无模型学习。

## 探索式开端

**蒙特卡洛方法**（Monte Carlo method）实际上是一种无模型状态的策略迭代方法，它基于随机采样的方法来估计值函数，从而评估策略。

策略迭代算法估计的是状态值函数 $v$， 而最终的策略是通过动作值函数 $q$ 来获得。当模型己知时，从 $v$ 到 $q$ 有很简单的转换方法，而当模型未知时，这也会出现困难。于是，我们将估计对象从 $v$ 转变为 $q$。

**探索式开端** (exploring starts) ：策略选代算法要求每对 state-action 分别进行估计，因此需要从每对 state-action 出发，随机采样 $n$ 条轨迹，计算累积奖励数据集 $\{g^{(i)}_{\pi}(s,a)\}_{i=1}^n$ 去近似值函数

$$
q_{\pi}(s,a)=\mathbb E[G_t|S_t=s,A_t=a]\approx\frac{1}{n}\sum_{i=1}^ng^{(i)}_{\pi}(s,a)
$$

在近似估计出 $q_{\pi}(s,a)$ 之后，就可以进行策略改进。然后在新的策略下重新通过采样来估计q-value，并不断重复，直至收敛。

算法流程如下：

1. 初始化策略 $\pi$ 

2. 重复以下步骤直至收敛
   
   - 策略评估 (Policy evaluation)：以每一对 $(s,a)$ 起始，随机生成多个完整的轨迹，估计值函数
     
     $$
     q_{\pi}(s,a)=\frac{1}{n}\sum_{i=1}^ng^{(i)}_{\pi}(s,a)
     $$
   
   - 策略改进 (Policy improvement)：根据当前值函数更新策略
     
     $$
     \pi(a|s)\gets\begin{cases}1,&\text{if }a=\arg\max\limits_{a}q_{\pi}(s,a) \\ 0, &\text{otherwise}\end{cases}
     $$

同时为充分利用采样的数据，对采样轨迹中出现的每一对 state-action ，记录其后的奖赏值之和，作为该 $(s,a)$ 的一次累积奖赏。如果一对 state-action 在一个序列中出现了多次，则有两种对应方法：

- 一种是只计算这个 state-action 对在序列中第一次出现的回报，来估计值函数，称为 **first-visit MC method**
- 另一种是这个 state-action 每次出现时都计算回报，来估计值函数，称为**every-visit MC method**

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/multiple_subepisodes.png" style="zoom:35%;" />

算法流程如下图：

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Monte_Carlo_ES.png" style="zoom:50%;" />

**示例**：假设我们有三个轨迹状态-奖励序列，折扣率 $\gamma=1$
$$
A,1,B,1,B,1,B,0,A,2,C \\
A,1,B,2,C \\
B,0,A,1,B,2,C \\
$$
使用 every-visit MC method
$$
v(A)=(5+2+3+3)/4=3.25 \\
v(B)=(4+3+2+2+3+2)/6=2.67
$$

## 利用与探索

可以看出，欲较好地获得值函数的估计，就需要多条不同的采样轨迹。然而，如果采用确定性策略 $\pi$，即对于某个状态只会输出一个动作。若使用这样的策略进行采样，则只能得到多条相同的轨迹，而无法计算其他动作 $a'$ 的动作价值，因此也无法进一步改进策略。这种情况仅仅是对当前策略的**利用**（exploitation），而缺失了对环境的**探索**（exploration），即试验的轨迹应该尽可能覆盖所有的状态和动作，以找到更好的策略。

仅利用价值最大的动作的策略，叫做 greedy 策略。$\epsilon$-greedy 策略基于一个概率来对探索和利用进行平衡。具体而言：在每个状态下，以 $\epsilon$ 的概率进行探索，即以均匀概率随机选择一个动作；以 $1-\epsilon$ 的概率进行利用，即选择当前最优的动作。

$$
\pi(a|s)\gets\begin{cases}1-\epsilon+\dfrac{\epsilon}{|\mathcal A|},&\text{for the greedy action} \\ \dfrac{\epsilon}{|\mathcal A|}, &\text{for the other }|\mathcal A|-1\text{ actions}\end{cases}
$$

算法流程如下图：算法中奖赏均值采用增量式计算，每采样出一条轨迹，就根据该轨迹涉及的所有 state-action 对来对值函数进行更新。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Monte_Carlo_e-Greedy.png" style="zoom:50%;" />

下面是一个MC $\epsilon$-Greedy 的示例

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/MC_e-greedy_example.png" style="zoom:50%;" />

## 重要性采样

之前介绍的算法中引入 $\epsilon$ -greedy 策略仅是为了便于采样评估，而在改进策略时并不需要探索。实际上算法流程中可以使用两个不同的策略：一个用于生成样本的策略称为**行为策略** (behaviour policy)；另一个用于改进的策略称为**目标策略**(target policy)。

- on-policy 是指目标策略和行为策略相同的算法。它实际上是做了一个折中，学到的不是一个最优的策略，而是一个带有探索的接近最优策略 (near-optimal policy)。
- off-policy 是指目标策略和行为策略不同的算法。行为策略需要保持随机性和更多的探索性，而目标策略可以是确定性的，在每个状态下选择贪心动作。off-policy可以通过引入重要性采样来实现对目标策略策略的优化。

**重要性采样**：假设随机变量的实际分布 $X\sim p$，而抽样来自另一个分布 $q$ ，则实际分布的期望

$$
\mathbb E_{X\sim p}[X]=\sum_{x\in\mathcal X}p(x)x=\sum_{x\in\mathcal X}q(x)\frac{p(x)}{q(x)}x=\mathbb E_{X\sim q}\left[\frac{p(X)}{q(X)}X\right]
$$

其中 

$$
w(x)=\frac{p(x)}{q(x)}
$$

称为重要性权重 (importance weight)。可通过从概率分布 $q$ 上的采样$\{x_i\}_{i=1}^n$ 来估计实际期望

$$
\bar x=\frac{1}{n}\sum_{i=1}^n\frac{p(x_i)}{q(x_i)}x_i
$$

**off-policy MC**：回到我们的问题上来，若改用行为策略 $\beta$ 的采样轨迹来评估策略 $\pi$ 。比如给定初始状态 $s_0=s$ ，接下来轨迹为

$$
s_0,a_0,s_1,a_1,\cdots,s_T
$$

其在策略 $\pi$ 下发生的概率为：

$$
\mathbb P(s_0,a_0,s_1,a_1,\cdots,s_T)=\prod_{k=0}^{T-1}\pi(a_k|s_k)p(s_{k+1}|s_k,a_k)
$$

所以，该轨迹发生的重要性权重为：

$$
w(s)=\frac{\prod_{k=0}^{T-1}\pi(a_k|s_k)p(s_{k+1}|s_k,a_k)}{\prod_{k=0}^{T-1}\beta(a_k|s_k)p(s_{k+1}|s_k,a_k)}=\prod_{k=0}^{T-1}\frac{\pi(a_k|s_k)}{\beta(a_k|s_k)}
$$

从上式可以看出，重要性权重和转移概率无关，仅依赖于两个策略和状态动作轨迹。由于状态动作轨迹是通过策略 $\beta$ 生成的，我们要计算的是策略 $\pi$ 下的状态值函数。所以，在得到累积奖励之后，需要乘以重要性权重来预测策略 $\pi$ 下的奖励值

$$
v_{\pi}(s)=\frac{1}{n}\sum_{k=1}^{n}w^{(k)}(s)g^{(k)}(s)
$$

原始的重要性采样是无偏估计，但是方差很大。因此使用加权重要性采样

$$
v_{\pi}(s)=\frac{\sum_{k=1}^{n}w^{(k)}(s)g^{(k)}(s)}{\sum_{k=1}^{n}w^{(k)}(s)}
$$

加权重要性采样是有偏的，但方差更小。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Monte_Carlo_off-policy.png" style="zoom:67%;" />



