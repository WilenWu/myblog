---
title: 机器学习(VII)--强化学习(二)动态规划
date: 2024-08-29 17:19
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/reinforcement_learning_cover.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 6e55c5aa
description: 
---

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