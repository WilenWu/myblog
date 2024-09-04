---
title: 机器学习(VII)--强化学习
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

# 动态规划

从贝尔曼方程可知，如果知道马尔可夫决策过程的状态转移概率和奖励函数，我们直接可以通过贝尔曼方程来迭代计算其值函数。这种模型已知的强化学习算法也称为**基于模型的方法**(Model-Based)，这里的模型就是指马尔可夫决策过程。**无模型方法** (Model-Free) 则需要近似估计模型。

**动态规划** (dynamic programming) 是一种基于模型的方法，它通过递推方式求解最优策略和最优值函数。

## 策略迭代

策略迭代 (policy iteration) 的核心思想是通过交替进行策略评估和策略改进来找到最优策略。

**策略改进**：贝尔曼最优方程揭示了非最优策略的改进方式：将策略选择的动作改变为当前最优的动作。对于一个策略 $\pi(a|s)$，其值函数为 $q_{\pi}(s,a)$，不妨令的改进后的策略为 $\pi'$

$$
\pi'(s)=\arg\max_{a\in\mathcal A} q_{\pi}(s,a)
$$

即 $\pi'$ 为一个确定性的策略。显然， $q_{\pi}(s,\pi'(s))\geq v_{\pi}(s)$，由 q-value 的贝尔曼方程可计算出递推不等式

$$
\begin{aligned}
v_{\pi}(s) & \leqslant q_{\pi}(s,\pi'(s)) \\
&=\mathbb E_{\pi'}[R_t+\gamma v_{\pi}(S_{t+1})|S_t=s] \\
&\leqslant \mathbb E_{\pi'}[R_t+\gamma q(S_{t+1},\pi'(S_{t+1})|S_t=s] \\
&=\mathbb E_{\pi'}[R_t+\gamma R_{t+1}+\gamma^2 v_{\pi}(S_{t+2})|S_t=s] \\
&\cdots \\
&\leqslant\mathbb E_{\pi'}[R_t+\gamma R_{t+1}+\gamma^2 R_{t+2}+\cdots|S_t=s] \\
&=v_{\pi'}(s)
\end{aligned}
$$

可见，值函数对于策略的每次改进都是单调递增的。我们可以通过下面方式来学习最优策略：先随机初始化一个策略，计算该策略的值函数，并根据值函数来设置新的策略，然后一直反复迭代直到收敛。此时就满足了贝尔曼最优方程，即找到了最优策略。

1. 初始化策略 $\pi$ 

2. 重复以下步骤直至收敛
   
   - 策略评估 (Policy evaluation)：利用贝尔曼方程对当前策略计算值函数
     
     $$
     \mathbf v_{\pi}=\mathbf r_{\pi}+\gamma P_{\pi}\mathbf v_{\pi}
     $$
   
   - 计算动作值函数
     
     $$
     q_{\pi}(s,a)=\sum_{s'\in\mathcal S}p(s'|s,a)[r+\gamma v_{\pi}(s')]
     $$
   
   - 策略改进 (Policy improvement)：根据当前值函数更新策略
     
     $$
     \pi(a|s)\gets\begin{cases}1,&\text{if }a=\arg\max\limits_{a}q_{\pi}(s,a) \\ 0, &\text{otherwise}\end{cases}
     $$

实际中常使用迭代方法求解贝尔曼方程，算法流程如图

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Policy_Iteration_Algorithm.png" style="zoom:50%;" />

下面是一个策略迭代的示例

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/policy_iteration_RL.svg)

## 值迭代

策略迭代算法中的策略评估和策略改进是交替轮流进行，其中策略评估也是通过一个内部迭代来进行计算，其计算量比较大。事实上，我们不需要每次计算出每次策略对应的精确的值函数，也就是说内部迭代不需要执行到完全收敛。

**值迭代** (Value Iteration) 算法将策略评估和策略改进两个过程合并，直接优化贝尔曼最优方程，迭代计算最优值函数。

1. 初始化状态价值 $v(s)$ 

2. 值更新 (Value update)：迭代计算最优值函数
   
   $$
   v(s)\gets \max_{a} \sum_{s'\in\mathcal S}p(s'|s,a)[r+\gamma v(s')]
   $$

3. 策略更新 (Policy update)：根据最优值函数更新策略
   
   - 计算动作值函数
     
     $$
     q(s,a)=\sum_{s'\in\mathcal S}p(s'|s,a)[r+\gamma v(s')]
     $$
   
   - 更新策略
     
     $$
     \pi(a|s)\gets\begin{cases}1,&\text{if }a=\arg\max\limits_{a}q(s,a) \\ 0, &\text{otherwise}\end{cases}
     $$

算法流程如图

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Value_Iteration_Algorithm.png" style="zoom:50%;" />

策略迭代算法是根据贝尔曼方程来更新值函数，并根据当前的值函数来改进策略。而值迭代算法是直接使用贝尔曼最优方程来更新值函数，收敛时的值函数就是最优的值函数，其对应的策略也就是最优的策略。

下面是一个值迭代的示例

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/value_iteration_RL.svg)

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

[Jupyter Notebook Code](/ipynb/reinforcement_learning.html#Grid-World)

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

## 悬崖行走问题

**悬崖行走问题**：(Cliff Walking) 在一个网格世界中，每个网格表示一个状态。其中网格 $(2,1)$ 到 $(11,1)$ 是悬崖(Cliff)。有一个Agent，目标是如何安全地从左下角的开始位置 $S$，走到右下角的目标位置 $G$。Agent 在每一个状态都可以选择4种行走方向：⬆️ ⬅️ ⬇️ ➡️。但每走一步，都有一定的概率滑落到周围其他格子。如果掉入悬崖或到达目标状态都会结束动作并回到起点，也就是说掉入悬崖或者达到目标状态是终止状态。Agent 每走一步的奖励是 −1，掉入悬崖的奖励是 −100，到达终止状态奖励为 0。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/cliff_walking.svg" />

[Jupyter Notebook Code](/ipynb/reinforcement_learning.html#Cliff-Walking)
