---
title: 机器学习(VII)--强化学习(一)马尔可夫决策过程
date: 2024-08-29 17:18:30
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/reinforcement_learning_cover.png
top_img: /img/artificial-intelligence.jpg
abbrlink: a36576b2
description: 
---

**强化学习** (Reinforcement Learning，RL) 可以描述为智能体 (Agent) 与环境 (Environment) 的交互中不断学习以完成特定目标（比如取得最大奖励值）的过程。

[【强化学习的数学原理】- bilibili](https://www.bilibili.com/video/BV1r3411Q7Rr/)
[Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)

# 马尔可夫决策过程

## 基本要素

下面介绍一个简单的示例，它能帮助我们更加直观的理解强化学习过程。

**路径寻找**：如下图，在一个网格世界中，包括平坦网格、草地和边界。有一个Agent，目标是如何快速地从起始位置，走到右下角的目标位置，其中草坪会减慢行走速度。每一步可以选择5种行走方向：$\uparrow,\leftarrow,\downarrow,\rightarrow,\circlearrowleft$，Agent 要做的是寻找好的行走路径。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/grid_world_example.svg)

强化学习任务通常使用**马尔可夫决策过程**(Markov Decision Process, MDP)来描述：智能体处于环境中，不断交互感知环境的**状态**，并根据反馈的**奖励**学习选择一个合适的**动作**，来最大化长期总收益。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/reinforcement_learning.svg" style="zoom:80%;">

描述强化学习的基本要素包括：

**状态** (state)： $s\in\mathcal S$  是智能体感知到的环境的描述。例如网格世界中每一个网格就是一种状态。

**动作** (action)： $a\in\mathcal A$  是对智能体行为的描述。例如网格世界中每一步的行走方向。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/states_actions.svg" style="zoom:80%;" />

**奖励** (reward)：$r$ 是智能体根据当前状态 $s$ 做出动作 $a$ 之后，环境反馈给智能体一个数值，这个奖励也经常和下一个时刻的状态 $s'$ 有关。奖励往往由我们自己来定义，奖励定义得好坏非常影响强化学习的结果。确定性的奖励一般用表格形式来呈现。比如，网格世界中，如果Agent进入草地或者碰到边界的 $r=-2$；进入平坦网格的奖励 $r=-1$ ；进入目标网格的奖励为 $1$。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/obtaining_rewards.svg)

**马尔可夫性质**：

$$
\mathbb P(s_{t+1}|s_0,a_0,s_1,a_1,\cdots,s_t,a_t)=\mathbb P(s_{t+1}|s_t,a_t)
$$

智能体要做的是通过在环境中不断地尝试而学得一个**策略** (policy) $\pi$ ，根据这个策略，在环境状态 $s$ 就能决定下一步动作 $a$ 。常见的策略表示方法有以下两种：

- **确定性策略**表示在状态 $s$ 下确定选择动作 $a$

$$
\pi(s)=a
$$

- **随机性策略**表示状态 $s$ 下选择动作 $a$ 的概率

$$
\pi(a|s)=\mathbb P(A_t=a|S_t=s)\quad \text{s.t. }\sum_{a\in\mathcal A} \pi(a|s)=1
$$

策略的目标是指导智能体选择最优动作，从而最大化累积奖励。学习最优策略是强化学习的核心任务之一。通常情况下，强化学习一般使用随机性策略，因为确定性策略只是随机性策略的特例，概率质量全部集中在一个动作 $a$ 上。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/policy_RL.svg" style="zoom:80%;" />

**状态转移**(state-transition)：是在智能体根据当前状态 $s$ 做出一个动作 $a$ 之后，环境在下一个时刻的状态 $s'$ 。状态转移可以是确定的，可以用表格形式呈现

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/deterministic_state_transition.svg" style="zoom:80%;" />

状态转移也可能是随机的，比如网格 $c_3$ 是冰面，有概率滑落到相邻网格。我们通常认为状态转移是随机的，用条件概率表示

$$
p(s'|s,a)=\mathbb P(S_{t+1}=s'|S_t=s,A_t=a)
$$

## 值函数

强化学习任务中，一个策略的优劣取决于长期执行这一策略后得到的累积奖励，因此，学习的目的就是要找到能使长期累积奖励最大化的策略。有了目标，接下来定义累积奖励。

智能体已经观测到的所有的状态、动作、奖励链称为一个**轨迹**(Trajectory)。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Markov_decision_process.svg" style="zoom:80%;" />

但不是所有任务都有一个明确的终止时间 $T$，有些任务时连续的，它的终止时刻定义为 $\infty$，所以需要定义一种计算累计奖励统一表示。即对于有限任务，在其终止时刻 $T$ 之后，终止状态 $s_T$ 循环不变。

假设环境中有一个或多个特殊的终止状态(Terminal State)，当到达终止状态时，一个智能体和环境的交互过程就结束了。这一轮交互的过程称为一个回合(Episode)或试验(Trial)。一般的强化学习任务(比如下棋、游戏)都属于这种回合式任务(Episodic Task)。从当前时刻 $t$ 开始的累积奖励称为**回报** (Return)

$$
G_t=r_{t+1}+r_{t+2}+\cdots+r_T
$$

如果环境中没有终止状态(比如终身学习的机器人)，即 $T=\infty$，称为持续 式任务(Continuing Task)，其总回报也可能是无穷大。为了解决这个问题，我们可以引入一个折扣率来降低远期回报的权重。**折扣回报**(Discounted Return) 定义为

$$
G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots
$$

这里的  $\gamma\in[0,1]$  叫做折扣率。如果 $\gamma$ 接近于0，则累积奖励更注重当前得到的即时奖励。如果 $\gamma$ 接近于1，则累积奖励更注重未来的奖励。折扣率是个超参数，需要手动调，折扣率的设置会影响强化学习的结果。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/return_RL.svg" style="zoom:80%;" />

因为在游戏尚未结束的 $t$ 时刻，策略和状态转移都有一定的随机性，$G_t$ 是一个随机变量，其随机性来自于 $t$ 时刻之后的所有状态与动作，所以使用期望回报来评估策略 $\pi$ 在当前时刻后的累积奖励。

**状态值函数** (state value)：定义从状态 $S_t=s$ 开始，执行策略 $\pi$ 得到的期望回报

$$
v_{\pi}(s)=\mathbb E_{\pi}[G_t|S_t=s]=\sum_{a\in\mathcal A}\pi(a|s)[G_t|S_t=s]
$$

**状态-动作值函数** (state-action value)：定义从状态 $S_t=s$ 开始，指定动作 $A_t=a$ 之后，执行策略 $\pi$ 得到的期望回报

$$
q_{\pi}(s,a)=\mathbb E[G_t|S_t=s,A_t=a]=\sum_{s'\in\mathcal S}p(s'|s,a)[G_t|S_t=s,A_t=a]
$$

易知，$v_{\pi}(s)$ 是 $q_{\pi}(s,a)$ 关于动作 $a$ 的期望

$$
v_{\pi}(s)=\sum_{a\in\mathcal A}\pi(a|s)q_{\pi}(s,a)=\mathbb E_\pi[q_{\pi}(s,a)]
$$

值函数可以看作对策略 $\pi$ 的评估，因此我们就可以根据值函数来优化策略。

## 贝尔曼方程

**状态价值贝尔曼方程**：首先考虑一个轨迹，根据回报的定义，易知

$$
G_t=R_{t+1}+\gamma G_{t+1}
$$

状态值函数可以拆解为

$$
v_{\pi}(s)=\mathbb E_{\pi}[R_{t+1}|S_t=s]+\gamma\mathbb E_{\pi}[G_{t+1}|s_t=s]
$$

分别代表 $t$ 时刻状态为 $s$ 时，执行所有动作的可能奖励和 $t+1$ 时刻的所有可能回报。

其中，等式第一部份

$$
\mathbb E_{\pi}[R_{t+1}|S_t=s]=\sum_{a\in\mathcal A}\pi(a|s)\sum_{s'\in\mathcal S}p(s'|s,a)r
$$

根据马尔可夫性质，等式第二部份

$$
\begin{aligned}
\mathbb E[G_{t+1}|S_t=s]&=\mathbb E[G_{t+1}|S_{t+1}=s'] \\
&=\sum_{s'\in\mathcal S}\mathbb E[G_{t+1}|S_{t+1}=s']p(s'|s)\\
&=\sum_{s'\in\mathcal S}v_{\pi}(s')\sum_{a\in\mathcal A}p(s'|s,a)\pi(a|s)
\end{aligned}
$$

最终得到

$$
\textcolor{red}{v_{\pi}(s)}=\sum_{a\in\mathcal A}\pi(a|s)\sum_{s'\in\mathcal S}p(s'|s,a)[r+\gamma\textcolor{red}{v_{\pi}(s')}],\quad \forall s\in\mathcal S
$$

其中，$r$ 表示即时奖励。 $v_{\pi}(s')$ 表示下一时刻状态 $S_{t+1}=s'$ 的状态值函数。上式称为**贝尔曼方程** (Bellman equation)。贝尔曼方程提供了计算值函数的递归公式，是求解最优策略和值函数的基础。

贝尔曼方程的期望形式如下：

$$
v_{\pi}(s)=\mathbb E_{a\sim\pi(a|s),s'\sim p(s'|s,a)}[r+\gamma v_{\pi}(s')],\quad \forall s\in\mathcal S
$$

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/state_action_value.svg)

令

$$
r_{\pi}(s)=\sum_{a\in\mathcal A}\pi(a|s)\sum_{s'\in\mathcal S}p(s'|s,a)r \\
p_{\pi}(s'|s)=\sum_{a\in\mathcal A}\pi(a|s)p(s'|s,a)
$$

对任意的状态 $s_i$ ，贝尔曼方程可重写为

$$
v_{\pi}(s_i)=r_{\pi}(s_i)+\gamma\sum_{s_j\in\mathcal S}p_{\pi}(s_j|s_i)v_{\pi}(s_j)
$$

于是得到贝尔曼方程的矩阵形式

$$
\mathbf v_{\pi}=\mathbf r_{\pi}+\gamma P_{\pi}\mathbf v_{\pi}
$$

其中

$$
\mathbf v_{\pi}=\begin{bmatrix}v_{\pi}(s_1) \\ \vdots \\ v_{\pi}(s_n)\end{bmatrix},\ 
\mathbf r_{\pi}=\begin{bmatrix}r_{\pi}(s_1) \\ \vdots \\ r_{\pi}(s_n)\end{bmatrix},\ 
P_{\pi}=\begin{bmatrix}
p_{\pi}(s_1|s_1)&\cdots& p_{\pi}(s_n|s_1)\\ 
\vdots &\ddots &\vdots \\ 
p_{\pi}(s_1|s_n)&\cdots& p_{\pi}(s_n|s_n)\\ 
\end{bmatrix}
$$

下面是一个简单的求解状态价值的示例

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Bellman_equation_matrix.svg" style="zoom:80%;" />

下面使用状态价值评价策略

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/policy_and_stats_values.svg" style="zoom:80%;" />

**行为值函数的贝尔曼方程**：比较关系式

$$
v_{\pi}(s)=\sum_{a\in\mathcal A}\pi(a|s)q_{\pi}(s,a)
$$

和状态值函数的贝尔曼方程我们可以得到关于行为值函数的贝尔曼方程

$$
\textcolor{red}{q_{\pi}(s,a)}=\sum_{s'\in\mathcal S}p(s'|s,a)[r+\gamma\textcolor{red}{v_{\pi}(s')}]
$$

期望形式为

$$
q_{\pi}(s,a)=\mathbb E_{s'\sim p(s'|s,a)}[r+\gamma v_{\pi}(s')]
$$

我们可以通过 $v_{\pi}(s)$ 计算出所有的 $q_{\pi}(s,a)$ ，这些 $q$ 值量化每个动作的好坏。有了这些 $q$ 值，智能体就可以根据 $q$ 来做决策，选出最好的动作。比如，在状态 $s_1$ 时应该选择动作$q$ 值最大的动作 $\downarrow$ 。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/q_value_Bellman_equation.svg" style="zoom:80%;" />

## 贝尔曼最优方程

**最优策略**：对某个策略的累积奖赏进行评估后，若发现它并非最优策略，则当然希望对其进行改进。理想的策略应能最大化累积奖赏

$$
\pi^*=\arg\max_{\pi}v_{\pi}(s),\quad\forall s\in\mathcal S
$$

一个强化学习任务可能有多个最优策略，所有最优策略拥有同样的值函数，称为最优值函数

$$
v^*(s)=\max_{\pi}v_{\pi}(s) \\
q^*(s,a)=\max_{\pi}q_{\pi}(s,a)
$$

**贝尔曼最优方程** (Bellman optimality equations) ：最优值函数对应的贝尔曼方程
$$
\begin{aligned}
v^*(s)=&\max_{\pi}\sum_{a\in\mathcal A}\pi(a|s)\sum_{s'\in\mathcal S}p(s'|s,a)[r+\gamma v_{\pi}(s')] \\
=&\max_{\pi}\sum_{a\in\mathcal A}\pi(a|s)q_{\pi}(s,a)
\end{aligned}
$$

因为 $\sum\pi(a|s)=1$ ，易知最优的策略

$$
\pi^*(a|s)=\begin{cases}1,&\text{if }a=a^*(s) \\ 0, &\text{otherwise}\end{cases}
$$

其中动作

$$
a^*(s)=\arg\max_{a\in\mathcal A} q^*(s,a)
$$

最优的状态值函数

$$
v^*(s)=\max_{a\in\mathcal A}q^*(s,a)
$$

上述贝尔曼最优方程的唯一解就是最优值函数。

然后，可以得到 $q$ 值的贝尔曼最优方程为

$$
q^*(s,a)=\sum_{s'\in\mathcal S}p(s'|s,a)[r+\gamma v^*(s')]
$$

进一步写为

$$
q^*(s,a)=\sum_{s'\in\mathcal S}p(s'|s,a)[r+\gamma\max_{a\in\mathcal A}q^*(s',a')]
$$


# 附录

## 模仿学习

强化学习任务中多步决策的搜索空间巨大，基于累积奖赏来学习很多步之前的合适决策非常困难，而直接模仿人类专家的 state-action 可显著缓解这一困难，我们称其为直接模仿学习。

假定我们获得了一批人类专家的决策轨迹数据，我们可将所有轨迹上的所有 state-action 对抽取出来，构造出一个新的数据集合

$$
D=\{(s_1,a_1),(s_2,a_2),\cdots,(s_N,a_N)\}
$$

有了这样的数据，就相当于告际机器在什么状态下应选择什么动作，于是可利用监督学习来学得符合人类专家决策轨迹数据的策略。即把状态作为特征，动作作为标记；然后，对这个新构造出的数据集合 $D$ 使用分类(对于离散动作)或回归(对于连续动作)算法即可学得策略模型。学得的这个策略模型可作为机器进行强化学习的初始策略，再通过强化学习方法基于环境反馈进行改进，从而获得更好的策略。


## 多臂赌博机问题

**多臂赌博机问题**：(K-Armed Bandit Problem) 给定 $K$ 个赌博机，拉动每个赌博机的拉杆，赌博机会按照一个事先设定的概率掉出钱，每个赌博机掉钱的概率分布不一样。现给定有限的机会次数 $T$，如何玩这些赌博机才能使得累积收益最大化。多臂赌博机问题在广告推荐、投资组合等领域有着非常重要的应用。