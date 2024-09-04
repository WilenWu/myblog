---
title: 机器学习(VII)--深度学习DQN
date: 
katex: true
categories:
  - Artificial Intelligence
  - Deep Learning
tags:
  - 机器学习
  - 深度学习
cover: /img/reinforcement_learning_cover.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 
description: 
---

深度强化学习(Deep Reinforcement Learning)是将强化学习和深度学习结合在一起，用强化学习来定义问题和优化目标，用深度学习来解决策略和值函数 的建模问题，然后使用误差反向传播算法来优化目标函数.深度强化学习在一定 程度上具备解决复杂问题的通用智能，并在很多任务上都取得了很大的成功。

# Deep Q-learning

Deep Q-Network (DQN) 是Q-Learning在深度学习的扩展。

然而，Q-Learning的目标函数存在两个问题：一是目标不稳定，参数学习的目标依赖于参数本身；二是样本之间有很强的相关性。为了解决这两个问题，DQN采取两个措施：一是目标网络冻结（Freezing Target Networks），即在一个时间段内固定目标中的参数，来稳定学习目标；二是经验回放（Experience Replay），即构建一个经验池（Replay Buffer）来去除数据相关性。经验池是由智能体最近的经历组成的数据集。

训练时，随机从经验池中抽取样本来代替当前的样本用来进行训练。这样，就打破了和相邻训练样本的相似性，避免模型陷入局部最优。经验回放在一定程度上类似于监督学习，先收集样本，然后在这些样本上进行训练。

TD 算法的目的是通过更新参数 w 使得损失函数最小

用两者的平方差作为损失函数

$$
J(w)=(q(S,A)-\hat G)^2
$$

梯度下降法

$$
\nabla_w J(w)=\delta \nabla_wq(s,a;\pi)
$$

梯度下降法更新

$$
w\gets w-\alpha\delta \nabla_wq(s,a;\pi)
$$

前面说到，我们有了目标价值，还有价值估计函数，希望让估计的价值接近目标价值。这就相当于一个监督学习的问题，所以，ANN也是可以直接利用到这个问题上的。输入是状态的特征表示，输出是这个状态的估计价值。参数是通过ANN训练得到的。

experience replay