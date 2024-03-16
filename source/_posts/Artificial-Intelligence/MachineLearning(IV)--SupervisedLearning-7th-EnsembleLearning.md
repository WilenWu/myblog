---
title: 机器学习(IV)--监督学习(七)集成学习
date: 2022-11-27 21:40
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-supervised-learning.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 6eef4242
description: 
---

# 集成学习

## 集成学习

**集成学习**（ensemble learning）通过构建**基学习器**（base learner）集合 $\{h_1,h_2,\cdots,h_M\}$，并组合基学习器的结果来提升预测结果的准确性和泛化能力。其中，基学习器通常采用弱学习算法，组合形成强学习算法。

> 准确率仅比随机猜测略高的学习算法称为**弱学习算法**，准确率很高并能在多项式时间内完成的学习算法称为**强学习算法**。

下面以二分类问题和回归问题为例，说明集成弱学习器为什么能够改善性能。

(1) 对于二分类问题，假设 $M$ 个弱分类模型，集成分类器采用多数表决的方法来预测类别，仅当基分类器超过一半预测错误的情况下，集成分类器预测错误。
$$
H(\mathbf x)=\text{sign}\left(\frac{1}{M}\sum_{m=1}^Mh_m(\mathbf x)\right)
$$
假设基分类器之间相互独立，且错误率相等为 $\epsilon$ 。则集成分类器的错误率为
$$
\epsilon_{\text{ensemble}} =\sum_{k=0}^{\lfloor M/2\rfloor}\complement^k_M(1-\epsilon)^k\epsilon^{M-k}
$$
取25个基分类器，误差率均为 0.35 ，计算可得集成分类器的误差为 0.06 ，远低于基分类器的误差率。注意，当 $\epsilon>0.5$ 时，集成分类器比不上基分类器。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/ensemble-classifier-error.svg" style="zoom:67%;" />

令$\epsilon=0.5-\gamma$ ，其中 $\gamma$ 度量了分类器比随机猜测强多少。则由Hoeffding 不等式可知
$$
\epsilon_{\text{ensemble}}\leqslant \exp(-2M\gamma^2)
$$
上式指出，随着基分类器的个数的增加集成错误率呈指数下降，从而快速收敛。但前提是基分类器之间相互独立。

(2) 对于回归问题，假设 $M$ 个弱回归模型，集成模型以均值输出
$$
H(\mathbf x)=\frac{1}{M}\sum_{m=1}^Mh_m(\mathbf x)
$$
每个基模型的误差服从均值为零的正态分布
$$
\epsilon_m\sim N(0,\sigma^2)
$$
若不同模型误差间的协方差均为 $\text{Cov}(\epsilon_i,\epsilon_j)=c$ 。则集成模型误差平方的期望是


$$
\begin{aligned}
\mathbb E(\epsilon_{\text{ensemble}}^2)
&=\mathbb E\left[\left(\frac{1}{M}\sum_{m=1}^M\epsilon_m\right)^2\right] \\
&=\frac{1}{M^2}\mathbb E\left[\sum_{i=1}^M\left(\epsilon_i^2+\sum_{j\neq i}\epsilon_i\epsilon_j\right)\right]  \\
&=\frac{1}{M}\sigma^2+\frac{M-1}{M}c
\end{aligned}
$$

在误差完全相关即 $c=\sigma^2$ 的情况下，误差平方减少到 $\sigma^2$ ，所以，模型平均没有任何帮助。在误差彼此独立即 $c=0$ 的情况下，该误差平方的期望仅为 $\sigma^2/M$ 。

上述示例容易得出，集成学习的基学习器要有足够的**准确性**和**差异性**。集成方法主分成两种：

- Bagging：是一种并行方法。通过在训练集上的有放回抽样来获得基学习器间的差异性。最典型的代表就是随机森林。
- Boosting：是一种串行迭代过程。自适应的改变训练数据的权重分布，构建一系列基分类器。最经典的包括AdaBoost算法和GBDT算法。

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。

## Bagging

### Bagging

Bagging（Bootstrap aggregating，袋装算法）是一种并行式的集成学习方法。通过在基学习器的训练集中引入随机化后训练。若数据集$D$有$N$ 个样本，则随机有放回采样出包含$N$个样本的数据集（可能有重复），同样的方法抽取$M$个训练集，这样在训练的时候每个训练集都会有不同。最后训练出 $M$ 个基学习器集成。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/bagging_algorithm.png" style="zoom:80%;" />

可以看出Bagging主要通过**样本的扰动**来增加基学习器之间的差异性，因此Bagging的基学习器应为那些对训练集十分敏感的不稳定学习算法，例如：神经网络与决策树等。从偏差-方差分解来看，Bagging算法主要关注于降低方差，即通过多次重复训练提高泛化能力。

由于抽样都来自于同一个数据集，且是有放回抽样，所以$M$个数据集彼此相似，而又因随机性而稍有不同。Bagging训练集中有接近36.8%的样本没有被采到
$$
\lim_{N\to\infty}(1-\frac{1}{N})^N=\frac{1}{e}\approx 0.368
$$
Bagging方法有许多不同的变体，主要是因为它们提取训练集的随机子集的方式不同：

- 如果使用无放回抽样，我们叫做 Pasting
- 如果使用有放回抽样，我们称为 Bagging
- 如果抽取特征的随机子集，我们叫做随机子空间 (Random Subspaces) 
- 最后，如果基学习器构建在对于样本和特征抽取的子集之上时，我们叫做随机补丁 (Random Patches) 

### 随机森林

对于决策树，事实证明，只需要改变一个训练样本，最高信息增益对应的特征就可能发生改变，因此在根节点会产生一个不同的划分，生成一颗完全不同的树。因此单个决策树对数据集的微小变化异常敏感。

**随机森林**（Random Forest）是Bagging的一个拓展体，它的基学习器固定为决策树，在基学习器构造过程中引入随机：

1. 采用有放回抽样的方式添加样本扰动，但有时在根节点附近也有相似的特征组成。
2. 因此进一步引入了特征扰动，每一个分裂过程从待选的 $n$ 个特征中随机选出包含 $k$ 个特征的子集，从这个子集中选择最优划分特征，一般推荐 $k=\log_2(n)$ 或 $k=\sqrt{n}$ 。
3. 每棵树都会完整成长而不会剪枝

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/random-forest.svg)

**随机森林优势**

- 它能够处理很高维度（特征很多）的数据，并且不用做特征选择
- 容易做成并行化方法，速度比较快
- 只在特征集的一个子集中选择划分，因此训练效率更高

## Boosting

### AdaBoost

**Boosting**（提升方法）是一种串行迭代过程。先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基学习器。如此迭代，最后将这些弱学习器组合成一个强学习器。Boosting族算法最著名、使用最为广泛的就是AdaBoost。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/boosting-flowchart.svg)

**AdaBoost** （Adaptive Boosting，自适应提升）的核心思想是用反复调整的数据来训练一系列的弱学习器，由这些弱学习器的加权组合，产生最终的预测结果。

具体说来，整个Adaboost 迭代算法分为3步：

1. **训练弱学习器**：在连续的提升（boosting）迭代中，那些在上一轮迭代中被预测错误的样本的权重将会被增加，而那些被预测正确的样本的权重将会被降低。然后，权值更新过的样本集被用于训练弱学习器。随着迭代次数的增加，那些难以预测的样例的影响将会越来越大，每一个随后的弱学习器都将会被强迫关注那些在之前被错误预测的样例。初始化时，所有样本都被赋予相同的权值 $1/N$ 。
2. **计算弱学习器权重**：在每一轮迭代中加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数，从而得到 $M$ 个弱学习器 $h_1,h_2,\cdots,h_M$。各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，使其在最终的分类函数中起着较大的决定作用，而降低分类误差率大的弱分类器的权重，使其在最终的分类函数中起着较小的决定作用。这样，每个弱分类器 $h_m$ 都有对应的权重 $\alpha_m$ 。
3. **组合成强学习器**：最后的强学习器由生成的多个弱学习器加权求和产生。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/AdaBoost_example.svg)

可以看出：**AdaBoost的核心步骤就是计算基学习器权重和样本权重分布**。AdaBoost 算法有多种推导方式，比较容易理解的是基于加法模型（additive model）的**前向分布算法**（forward stagewise algorithm）。

给定二分类数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标 $y_i\in\{-1,+1\}$

基分类器的加权组合即为加法模型
$$
f(\mathbf x)=\sum_{m=1}^M\alpha_mh_m(\mathbf x)
$$
其中 $\alpha_m$ 表示基分类器 $h_m$ 的重要性。最终的强分类器为
$$
H(\mathbf x)=\mathrm{sign}(f(\mathbf x))=\mathrm{sign}\left(\sum_{m=1}^M\alpha_mh_m(\mathbf x)\right)
$$
$f(\mathbf x)$ 的符号决定了实例 $\mathbf x$ 的类别。

给定损失函数 $L(y,f(\mathbf x))$ ，学习模型 $f(\mathbf x)$ 所要考虑的问题是最小化代价函数
$$
\min_{\alpha_m,h_m}\sum_{i=1}^NL\left(y_i,\sum_{m=1}^M\alpha_mh_m(\mathbf x_i)\right)
$$
通常这是一个复杂的全局优化问题，前向分布算法使用其简化版求解这一问题：既然是加法模型，每一步只学习一个弱学习器及其系数，且不调整已经加入模型中的参数和系数来向前逐步建立模型，这能够得到上述优化的近似解。这样，前向分布算法将同时求解 $m=1$ 到 $M$ 所有参数 $\alpha_m,\theta_m$ 的优化问题简化为逐步求解 $\alpha_m,\theta_m$ 的优化问题。

假设经过 $m-1$ 轮迭代，已经得到之前所有弱分类器的加权和 
$$
f_{m-1}(\mathbf x)=\alpha_1h_1(\mathbf x)+\cdots+\alpha_{m-1}h_{m-1}(\mathbf x)
$$
在第 $m$ 轮迭代求解 $\alpha_m,h_m$ 得到
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+\alpha_m h_m(\mathbf x)
$$
则每一步只需优化如下代价函数
$$
(\alpha_m,h_m)=\arg\min_{\alpha,h}\sum_{i=1}^NL\left(y_i,f_{m-1}(\mathbf x)+\alpha h(\mathbf x_i)\right)
$$
其中，在第$m$轮迭代中，$f_{m-1}(\mathbf x)$ 相当于一个定值。AdaBoost 每步采用**指数损失函数**（exponential loss function）
$$
L(y,f(\mathbf x))=\exp(-yf(\mathbf x))
$$

于是，优化函数可变为
$$
\begin{aligned}
(\alpha_m,h_m)&=\arg\min_{\alpha,h}\sum_{i=1}^NL(y_i,f_{m-1}(\mathbf x_i)+\alpha h(\mathbf x_i)) \\
&=\arg\min_{\alpha,h}\sum_{i=1}^N\exp[-y_i(f_{m-1}(\mathbf x_i)+\alpha h(\mathbf x_i))] \\
&=\arg\min_{\alpha,h}\sum_{i=1}^N  w_m^{(i)}\exp[-y_i\alpha h(\mathbf x_i)] \\

\end{aligned}
$$
其中 $w_m^{(i)}=\exp(-y_if_{m-1}(\mathbf x_i))$ 。$w_m^{(i)}$ 不依赖于 $\alpha$ 和 $h$ ，所以与优化无关。

由 AdaBoost 基分类器 $h(\mathbf x_i)\in\{-1,+1\}$ ，且 $y_i\in\{-1,+1\}$ 则
$$
-y_ih(\mathbf x_i)=\begin{cases}
+1 & \text{if }h(\mathbf x_i)\neq y_i \\
-1 & \text{if }h(\mathbf x_i)= y_i
\end{cases}
$$

所以，优化函数进一步化为

$$
\begin{aligned}
(\alpha_m,h_m)&=\arg\min_{\alpha,h}\left\{\sum_{i=1}^N  w_m^{(i)}e^{-\alpha}\mathbb I(h_m(\mathbf x_i)=y_i)+\sum_{i=1}^N  w_m^{(i)}e^{\alpha}\mathbb I(h_m(\mathbf x_i)\neq y_i)\right\} \\
&=\arg\min_{\alpha,h}\left\{(e^{\alpha}-e^{-\alpha})\sum_{i=1}^N  w_m^{(i)}\mathbb I(h_m(\mathbf x_i)\neq y_i)+e^{-\alpha}\sum_{i=1}^N  w_m^{(i)}\right\} \\
\end{aligned}
$$

上式可以得到AdaBoost算法的几个关键点：

**(1) 基学习器**。对于任意 $\alpha>0$ ，基分类器的解为
$$
h_m=\arg\min_{h}\sum_{i=1}^N w_m^{(i)}\mathbb I(h(\mathbf x_i)\neq y_i)
$$
这是第 $m$ 轮加权错误率最小的基分类器。

**(2) 各基学习器的系数** 。将已求得的 $h_m$ 带入优化函数
$$
\alpha_m=\arg\min_{\alpha}\left\{(e^{\alpha}-e^{-\alpha})\epsilon_m+e^{-\alpha}\right\}
$$
其中， $\epsilon_m$ 正是基分类器 $h_m$ 在加权训练集 $D_m$ 的错误率
$$
\epsilon_m=\frac{\displaystyle\sum_{i=1}^N  w_m^{(i)} \mathbb I(h_m(\mathbf x_i)\neq y_i)}{\displaystyle\sum_{i=1}^N w_m^{(i)}}
$$
这里 $ w_m^{(i)}$ 是第 $m$ 轮迭代中样本 $(\mathbf x_i,y_i)$ 的权重 ，因为Adaboost更新样本权值分布时做了规范化，所示上式中的分母为1。权重依赖于 $f_{m-1}(\mathbf x)$ ，随着每一轮迭代而发生改变。

对 $\alpha$ 求导并使导数为 0，即可得到基分类器 $h_m$ 的权重
$$
\alpha_m=\frac{1}{2}\ln(\frac{1-\epsilon_m}{\epsilon_m})
$$

由上式可知，当 $\epsilon_m\leqslant 0.5$ 时，$\alpha_m\geqslant 0$，并且 $\alpha_m$ 随 $\epsilon_m$ 的减小而增大 。所以，分类误差率越小的基分类器在最终分类器中的作用越大。如下图

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/AdaBoost_alpha.svg)

**(3) 下一轮样本权值**。由
$$
\begin{cases}
f_m(\mathbf x)=f_{m-1}(\mathbf x)+\alpha_m h_m(\mathbf x) \\
 w_m^{(i)}=\exp(-y_if_{m-1}(\mathbf x_i))
\end{cases}
$$
可得到
$$
w_{m+1}^{(i)}= w_m^{(i)}\exp(-\alpha_my_ih_m(\mathbf x_i))
$$
为了确保  $\mathbf w_{m+1}$ 成为一个概率分布 $\sum_{i=1}^Nw_{m+1}^{(i)}=1$， 权重更新变为
$$
w_{m+1}^{(i)}=\frac{w_m^{(i)}}{Z_m}\exp(-\alpha_my_ih_m(\mathbf x_i))
$$
其中， $Z_m$ 是正规因子。对原始式中所有的权重都乘以同样的值，对权重更新没有影响。
$$
Z_m=\sum_{i=1}^Nw_m^{(i)}\exp(-\alpha_my_ih_m(\mathbf x_i))
$$
上式可拆解为 
$$
w_{m+1}^{(i)}=\frac{w_m^{(i)}}{Z_m}\times\begin{cases}
\exp(-\alpha_m) & \text{if }h_m(\mathbf x_i)=y_i \\
\exp(\alpha_m) & \text{if }h_m(\mathbf x_i)\neq y_i 
\end{cases}
$$

上式给出的权值更新公式增加那些被错误分类的样本的权值，并减少那些被正确分类的样本的权值。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/AdaBoost_algorithm.png" style="zoom:80%;" />

从偏差-方差分解来看：AdaBoost 算法主要关注于降低偏差，每轮的迭代都关注于训练过程中预测错误的样本，很容易受过拟合的影响。

### 提升树

以二叉决策树为基函数的提升方法称为**提升树**（boosting tree）。提升树模型可以表示为决策树的加法模型：
$$
f_M(\mathbf x)=\sum_{m=1}^MT(\mathbf x;\Theta_m)
$$
其中，$T(\mathbf x;\Theta_m)$ 表示决策树，$\Theta_m$ 为决策树的参数， $M$ 为树的个数。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Tree_Ensemble_Model.png" style="zoom: 67%;" />

提升树采用前向分布算法实现学习的优化过程。，初始树 $f_0(\mathbf x)=0$ ，第 $m$ 轮迭代的模型是
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+T(\mathbf x;\Theta_m)
$$
其中，$f_{m-1}(\mathbf x)$ 是上一轮模型。本轮的目标是找到一个二叉树 $T(\mathbf x;\Theta_m)$，最小化代价函数
$$
\min_{\Theta_m}\sum_{i=1}^NL(y_i,f_{m-1}(\mathbf x_i)+T(\mathbf x_i;\Theta_m))
$$

对于不同问题的提升树学习算法，其主要区别在于损失函数的不同。主要包括用平法误差损失函数的回归问题，用指数损失函数的分类问题，以及用一般损失函数的一般决策问题。

(1) **回归问题**：提升树每一步采用平方误差损失函数
$$
L(y,f(\mathbf x))=(y-f(\mathbf x))^2
$$
第 $m$ 轮样本的损失为
$$
\begin{aligned}
L(y,f_m(\mathbf x))=&L(y,f_{m-1}(\mathbf x)+T(\mathbf x;\Theta_m)) \\
=&(y-f_{m-1}(\mathbf x)-T(\mathbf x;\Theta_m))^2 \\
=&(r_m-T(\mathbf x;\Theta_m))^2
\end{aligned}
$$
这里
$$
r_m=y-f_{m-1}(\mathbf x)
$$
是上一轮模型$f_{m-1}(\mathbf x)$拟合数据的残差（residual）。所以，对回归问题的提升树来说，最小化损失函数相当于决策树（弱学习器）简单拟合残差。

举一个通俗的例子，假如有个人30岁，我们首先用20岁去拟合，发现损失有10岁，这时我们用6岁去拟合剩下的损失，发现差距还有4岁，第三轮我们用3岁拟合剩下的差距，差距就只有一岁了。如果我们的迭代轮数还没有完，可以继续迭代下面，每一轮迭代，拟合的岁数误差都会减小。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GBDT-example.svg)

(2) **分类问题**：每一步采用指数损失函数，提升树算法只是 Adaboost算法的特例。

(3) **一般损失函数**：当损失函数是平方损失和指数函数时，每一步优化时很简单的，但对一般损失函数而言，往往每一步优化并不那么容易。针对这一问题，Freidman提出梯度提升（gradient boosting）算法。

### GBDT

**梯度提升树**（Gradient Boosted Decision Tree，GBDT）又叫MART（Multiple Additive Regression Tree），是提升树的一种改进算法，适用于任意可微损失函数。可用于各种领域的回归和分类问题，包括Web搜索、排名和生态领域。

GBDT加法模型表示为：
$$
f_M(\mathbf x)=\sum_{m=1}^MT(\mathbf x;\Theta_m)
$$
其中，$T(\mathbf x;\Theta_m)$ 表示决策树，$\Theta_m$ 为决策树的参数， $M$ 为树的个数。

最优化代价函数
$$
J(f)=\sum_{i=1}^NL(y_i,f(\mathbf x_i))
$$
GBDT使用前向分布算法迭代提升。假设第 $m$ 轮迭代的模型是
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+T(\mathbf x;\Theta_m)
$$
其中，$f_{m-1}(\mathbf x)$ 是上一轮迭代得到的模型。使用一阶泰勒展开[^taylor]，第 $m$ 轮第 $i$ 个样本的损失函数的近似值为
$$
L(y_i,f_m(\mathbf x_i))\approx L(y_i,f_{m-1}(\mathbf x_i))+T(\mathbf x_i;\Theta_m) g_{mi}
$$
其中
$$
g_{mi}=\left[\frac{\partial L(y_i,f(\mathbf x_i))}{\partial f(\mathbf x_i)}\right]_{f=f_{m-1}}
$$
本轮代价函数变为
$$
J(f_m)\approx J(f_{m-1})+\sum_{i=1}^NT(\mathbf x_i;\Theta_m) g_{mi}
$$
我们希望随着每轮迭代，损失会依次下降 $J(f_m)-J(f_{m-1})<0$，且本轮损失最小化，则有
$$
\min_{\Theta_m}\sum_{i=1}^NT(\mathbf x_i;\Theta_m) g_{mi}=\min_{\Theta_m}\mathbf T(\Theta_m)\cdot \mathbf g_m=\min_{\Theta_m}\|\mathbf T(\Theta_m)\|_2\|\mathbf g_m\|_2\cos\theta
$$
其中，$\mathbf T(\Theta_m)=(T(\mathbf x_1;\Theta_m),T(\mathbf x_2;\Theta_m),\cdots,T(\mathbf x_N;\Theta_m))$ 为第 $m$ 轮的强学习器在 $N$ 个数据点上的提升向量，$\mathbf g_m=(g_{m1},g_{m2},\cdots,g_{mN})$ 为损失函数在 $f_{m-1}(\mathbf x)$ 处的梯度向量，$\theta$ 为两向量夹角。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GBDT_residual.svg" style="zoom: 80%;" />

如果被拟合的决策树 $T(\mathbf x_i;\Theta_m)$ 预测的值与负梯度 $-g_{mi}$ 成正比 ($\cos\theta=-1$)，则取得最小值。因此，在每次迭代时，使用负梯度$-g_{mi}$ 来拟合决策树。$m$ 轮的弱学习器可表示为
$$
T(\mathbf x;\Theta_m)\approx -\left[\frac{\partial L(y,f(\mathbf x))}{\partial f(\mathbf x)}\right]_{f=f_{m-1}}
$$
负梯度被称为广义残差或**伪残差**（pseudo residual）。梯度在每次迭代中都会被更新，试图在**局部最优**方向求解，这可以看作是**函数空间中的某种梯度下降**，不同的损失函数将会得到不同的负梯度。下表总结了通常使用的损失函数的梯度

| Setting        | Loss Function                       | Gradient                                               |
| :------------- | ----------------------------------- | ------------------------------------------------------ |
| Regression     | $\frac{1}{2}(y_i-f(\mathbf x_i))^2$ | $y_i-f(\mathbf x_i)$                                   |
| Regression     | $\mid y_i-f(\mathbf x_i)\mid$       | $\text{sign} (y_i-f(\mathbf x_i))$                     |
| Regression     | Huber                               |                                                        |
| Classification | Deviance                            | $k$th component: $\mathbb I(y_i=c_k)-P_k(\mathbf x_i)$ |

对于平方误差损失，负梯度恰恰是普通的残差。因为GBDT每次迭代要拟合的梯度值是连续值，所以限定了基学习器只能使用**CART回归树**，且树的生成使用加权均方误差选择最优划分特征。即使对于分类任务，基学习器仍然是CART回归树。

**回归树流程**：

(1) 默认情况下，初始决策树选择使损失最小化的常数（对于均方误差损失，这是目标值的经验平均值 $\bar y$）。
$$
f_0(\mathbf x)=\arg\min_c\sum_{i=1}^NL(y_i,c)
$$
(2) 对每步迭代 $m=1,2,\cdots,M$

    (a) 计算损失函数的负梯度
$$
r_{mi}=-\left[\frac{\partial L(y_i,f(\mathbf x_i))}{\partial f(\mathbf x_i)}\right]_{f=f_{m-1}}\quad i=1,2,\cdots,N
$$
    (b) 对 $r_{mi}$ 拟合一个回归树
$$
T_m(\mathbf x)=\arg\min_{h}\sum_{i=1}^NL(r_{mi},h(\mathbf x_i))
$$
    得到第 $m$ 棵树的叶节点区域 $R_{mj},\ j=1,2,\cdots,J_m$。
    
    (c) 计算每个叶节点的输出值 $j=1,2,\cdots,J_m$
$$
c_{mj}=\arg\min_{c}\sum_{\mathbf x_i\in R_{mj}}L(y_i,f_{m-1}(\mathbf x_i)+c)
$$
(3) 更新
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+\sum_{j=1}^{J_m}c_{mj}\mathbb I(\mathbf x\in R_{mj})
$$
(4) 得到最终模型
$$
H(\mathbf x)=f_M(\mathbf x)
$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GBRT_algorithm.png" style="zoom:80%;" />

**二分类问题**：采取了和逻辑回归同样的方式，即利用sigmoid函数将输出值归一化
$$
H(\mathbf x)=\frac{1}{1+e^{-f_M(\mathbf x)}}
$$
模型的输出为正样本的概率 $\mathbb P(y=1|\mathbf x)$ 。损失函数同样交叉熵损失
$$
\begin{aligned}
L(y,H(\mathbf{x}))=-y\log H(\mathbf{x})-(1-y)\log(1-H(\mathbf{x}))
\end{aligned}
$$
负梯度
$$
-\frac{\partial L(y,H(\mathbf{x}))}{\partial f_M(\mathbf{x})}
=-\frac{\partial L(y,H(\mathbf{x}))}{\partial H(\mathbf{x})}\cdot \frac{\mathrm{d}H(\mathbf{x})}{\mathrm{d}f_M(\mathbf{x})}
=y-H(\mathbf x)
$$
这个负梯度即残差，表示真实值和预测正样本概率的差值。

**多分类问题**：采用softmax函数映射。假设类别的数量为 $K$ ，损失函数为
$$
L=-\sum_{k=1}^K y_k\log\mathbb P(y_k|\mathbf x)
$$
**优缺点**：Boosting共有的缺点为训练是按顺序的，难以并行，这样在大规模数据上可能导致速度过慢，所幸近年来XGBoost和LightGBM的出现都极大缓解了这个问题，后文详述。

### 特征重要性

在数据挖掘应用中，只有一小部分特征变量会对目标变量有显著的影响，研究每个特征变量在预测目标变量时的相对重要性或者贡献是很有用的。

单个决策树本质上通过选择合适的分割点来进行特征选择，这些信息可以用来度量每个特征的重要性。基本思想是：在树的分割点中使用某特征越频繁，该特性就越重要。对于单个决策树 $T$
$$
\mathcal I_\ell^2(T)=\sum_{t=1}^{J-1}\imath_t^2\mathbb I(v(t)=\ell)
$$
作为特征变量 $x_\ell$ 重要性的度量。这个求和是对树的 $J-1$ 个中间结点进行的。在每个中间结点 $t$ ，其中一个特征变量 $x_{v(t)}$ 会将这个结点区域分成两个子区域，每一个子区域用单独的常值拟合目标变量。特征变量的选择要使得在整个区域上有最大的纯度提升 $\imath_t^2$。变量 $x_\ell$ 的**平方相对重要度**（squared relative importance）是在所有的结点中，选择其作为分离变量时纯度提升的平方之和。

这种重要性的概念可以通过简单地平均每个树的基于不纯度的特征重要性来扩展到决策树集成器上
$$
\mathcal I_\ell^2=\frac{1}{M}\sum_{m=1}^M\mathcal I_\ell^2(T_m)
$$
考虑到平均的标准化影响，这个度量会比单个树对应的度量式更稳定。

### XGBoost

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/xgboost.svg" width="25%;" align="right"/> XGBoost（eXtreme Gradient Boosting）是基于GBDT的一种优化算法，于2016年由陈天奇在论文[《 XGBoost：A Scalable Tree Boosting System》](https://arxiv.org/pdf/1603.02754.pdf)中正式提出，在速度和精度上都有显著提升，因而近年来在 Kaggle 等各大数据科学比赛中都得到了广泛应用。

- 进行二阶泰勒展开，优化损失函数，提高计算精度
- 损失函数添加了正则项，避免过拟合
- 采用 Blocks存储结构，可以并行计算

XGBoost 同样为加法模型
$$
f_M(\mathbf x)=\sum_{m=1}^MT_m(\mathbf x)
$$
其中，$T_m(\mathbf x)$ 表示树模型， $M$ 为树的个数。

**目标函数推导**：XGBoost 优化的目标函数由损失函数和正则化项两部分组成
$$
\mathcal L=\sum_{i=1}^N l(y_i,f_M(\mathbf x_i))+\sum_{m=1}^M\Omega(T_m)
$$
其中，$\Omega(T_m)$ 表示第 $m$ 棵树 $T_m$ 的复杂度。

第 $m$ 轮迭代的目标函数为
$$
\mathcal L_m=\sum_{i=1}^N l(y_i,f_{m}(\mathbf x_i))+\sum_{k=1}^m\Omega(T_k)
$$
接下来，分三步简化目标函数。

(1) XGBoost遵从前向分布算法，第 $m$ 轮迭代的模型
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+T_m(\mathbf x)
$$
其中，$f_{m-1}(\mathbf x)$ 是上一轮迭代得到的模型。第 $m$ 轮第 $i$ 个样本的损失函数
$$
l(y_i,f_m(\mathbf x_i))=l(y_i,f_{m-1}(\mathbf x_i)+T_m(\mathbf x_i))
$$
XGBoost 使用**二阶泰勒展开**[^taylor]，损失函数的近似值为
$$
l(y_i,f_m(\mathbf x_i))\approx l(y_i,f_{m-1}(\mathbf x_i))+g_iT_m(\mathbf x_i)+\frac{1}{2}h_iT_m^2(\mathbf x_i)
$$
其中，$g_i$为一阶导，$h_i$ 为二阶导
$$
g_{i}=\left[\frac{\partial l(y_i,f(\mathbf x_i))}{\partial f(\mathbf x_i)}\right]_{f=f_{m-1}},\quad h_{i}=\left[\frac{\partial^2 l(y_i,f(\mathbf x_i))}{\partial^2 f(\mathbf x_i)}\right]_{f=f_{m-1}}
$$
上一轮模型 $f_{m-1}(\mathbf x)$ 已经确定，所以上一轮损失 $l(y_i,f_{m-1}(\mathbf x_i))$ 即为常数项，其对函数优化不会产生影响。移除常数项，所以第 $m$ 轮目标函数可以写成
$$
\mathcal L_m\approx \sum_{i=1}^N \left[g_iT_m(\mathbf x_i)+\frac{1}{2}h_iT_m^2(\mathbf x_i)\right]+\sum_{k=1}^m\Omega(T_k)
$$
(2) 将正则化项进行拆分
$$
\sum_{k=1}^m\Omega(T_k)=\Omega(T_m)+\sum_{k=1}^{m-1}\Omega(T_k)=\Omega(T_m)+\text{constant}
$$
因为 $m-1$ 棵树的结构已经确定，所以可记 $\sum_{k=1}^{m-1}\Omega(T_k)$ 为常数。移除常数项，目标函数可进一步简化为
$$
\mathcal L_m\approx \sum_{i=1}^N \left[g_iT_m(\mathbf x_i)+\frac{1}{2}h_iT_m^2(\mathbf x_i)\right]+\Omega(T_m)
$$
(3) 定义树：沿用之前对树结构的定义。假设树$T_m$ 将样本划分到 $J$ 个互不相交的区域 $R_1,R_2,\cdots,R_J$ ，每个区域 $R_j$ （本质是树的一个分支）对应树的一个叶结点，并且在每个叶节点上有个固定的输出值 $c_j$ 。每个样本只属于其中一个区域，那么树可以表示为 
$$
T_m(\mathbf x;\Theta)=\sum_{j=1}^Jc_j\mathbb I(\mathbf x\in R_j)
$$
参数 $\Theta=\{(R_1,c_1),(R_2,c_2),\cdots,(R_J,c_J)\}$ 表示树的区域划分和对应的值，$J$ 表示叶节点的个数。

然后，定义树的复杂度 $\Omega$ ：包含叶子节点的数量 $J$ 和叶子节点权重向量的 $\ell_2$ 范数
$$
\Omega(T_m)=\gamma J+\frac{1}{2}\lambda\sum_{j=1}^Jc_j^2
$$
这样相当于使叶结点的数目变小，同时限制叶结点上的分数，因为通常分数越大学得越快，就越容易过拟合。

因为每个叶子节点的输出值是相同的，可将目标函数中的样本按叶子节点分组计算，得到最终目标函数
$$
\begin{aligned}
\mathcal L_m &\approx \sum_{j=1}^J \left[(\sum_{\mathbf x_i\in R_j}g_i)c_j+\frac{1}{2}(\sum_{\mathbf x_i\in R_j}h_i+\lambda)c^2_j\right]+\gamma J \\
&=\sum_{j=1}^J \left[G_jc_j+\frac{1}{2}(H_j+\lambda)c^2_j\right]+\gamma J
\end{aligned}
$$
其中
$$
G_j=\sum_{\mathbf x_i\in R_j}g_i,\quad H_j=\sum_{\mathbf x_i\in R_j}h_i
$$

- $G_j$ 是树 $T_m$ 划分的叶子节点 $j$ 所包含的所有样本的**一阶偏导数**之和；
- $H_j$ 是树 $T_m$ 划分的叶子节点 $j$ 所包含的所有样本的**二阶偏导数**之和。

当树 $T_m$ 的区域划分确定时，$G_j,H_j$ 可视为常数。于是，目标函数只包含叶子节点输出值 $c_j$ 。 易知 $H_j+\lambda>0$，目标函数对 $c_j$ 求一阶导数，并令其为 0，可得最优解

$$
c_j^*=-\frac{G_j}{H_j+\lambda}
$$
所以目标函数最优值为
$$
\mathcal L_m^*=-\frac{1}{2}\sum_{j=1}^J\frac{G_j^2}{H_j+\lambda}+\gamma J
$$
下图给出目标函数计算的例子

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/tree_structure_score.png" style="zoom: 50%;" />

**学习树结构**：经过一系列推导后，我们的目标变为确定树 $T_m$ 的结构，然后计算叶节点上的值 $c_j^*$ 。

在树的生成过程中，最佳分裂点是一个关键问题。$\mathcal L_m^*$ 可以作为决策树 $T_m$ 结构的评分函数（scoring function），该值越小，树结构越好。XGboost 支持多种分裂方法：

(1) Exact Greedy Algorithm：现实中常使用贪心算法，遍历每个候选特征的每个取值，计算分裂前后的增益，并选择增益最大的候选特征和取值进行分裂。

XGBoost 提出了一种新的增益计算方法，采用目标函数的分裂增益。类似于CART基尼系数增益，对于目标函数来说，分裂后的增益为 $\mathcal L_{split}=\mathcal L_{before}-\mathcal L_{after}$ 。因此，定义
$$
\text{Gain}=\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}-\gamma
$$

> 上式去除了常数因子 1/2 ，不影响增益最大化。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/split_find.png" style="zoom: 80%;" />

上式中的$\gamma$ 是一个超参数，具有双重含义，一个是对叶结点数目进行控制；另一个是限制树的生长的阈值，当增益大于阈值时才让节点分裂。所以xgboost在优化目标函数的同时相当于做了预剪枝。

(2) Approximate Algorithm：贪心算法可以得到最优解，但当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低。近似算法主要针对贪心算法这一缺点给出了近似最优解，不仅解决了这个问题，也同时能提升训练速度

首先根据特征分布的分位数提出候选划分点，然后将连续型特征映射到由这些候选点划分的buckets中，然后汇总统计信息找到所有区间的最佳分裂点。

对于每个特征，只考察分位点可以减少计算复杂度。近似算法在提出候选切分点时有两种策略：

- Global：学习每棵树前就提出候选切分点，并在每次分裂时都采用这种分割；
- Local：每次分裂前将重新提出候选切分点。

下图给出不同种分裂策略的AUC变化曲线，横坐标为迭代次数，纵坐标为测试集AUC，eps 为近似算法的精度，其倒数为桶的数量。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/xgboost_eps_parameter.png" style="zoom:50%;" />

> Global 策略在候选点数多时（eps 小）可以和 Local 策略在候选点少时（eps 大）具有相似的精度。此外我们还发现，在 eps 取值合理的情况下，分位数策略可以获得与贪婪算法相同的精度。

(3) Weighted Quantile Sketch：加权分位数缩略图。XGBoost不是简单地按照样本个数进行分位，而是以二阶导数值 $g_i$ 作为样本的权重进行划分。实际上，当使用平方损失函数时，我们可以看到$g_i$就是样本的权重。

(4) Sparsity-aware Split Finding：稀疏感知法。实际工程中一般会出现输入值稀疏的情况。比如数据的缺失、one-hot编码都会造成输入数据稀疏。在计算分裂增益时不会考虑带有缺失值的样本，这样就减少了时间开销。在分裂点确定了之后，将带有缺失值的样本分别放在左子树和右子树，比较两者分裂增益，选择增益较大的那一边作为默认分裂方向。

**Shrinkage（收缩率）**：是一种简单的正则化策略，即对每一轮学习的结果乘以因子 $\nu(0<\nu<1)$ 进行缩放，就像梯度下降的学习率，降低单棵树的影响，为后面生成的树留下提升模型性能的空间。于是上文的迭代变为
$$
f_m(\mathbf x)=f_{m-1}(\mathbf x)+\nu T_m(\mathbf x)
$$
一般学习率$\nu$ 要和弱学习器个数 $M$ 结合起来使用。较小的$\nu$ 值要求较多的弱学习器以保持一个恒定的训练误差。经验证据表明，较小的学习率会有更好的测试误差，并且需要更大的 $M$ 与之对应。建议将学习速率设置为一个小常数（例如，$\nu\leqslant 0.1$)，并通过 early stopping 策略选择 $M$ 。

**Subsampling（子采样）**：随机梯度提升（stochastic gradient boosting）是将梯度提升（gradient boosting）和 bagging 相结合，既能防止过拟合，又能减少计算量。在每次迭代中，基学习器是通过无放回抽取训练集子集拟合。通常设置采样率 $\eta=0.5$ 。

下图表明了shrinkage 与 subsampling 对于模型拟合好坏的影响。我们可以明显看到指定收缩率比没有收缩拥有更好的表现。而将子采样和收缩率相结合能进一步的提高模型的准确率。相反，使用子采样而不使用收缩的结果很差。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GBDT_subsample.png" style="zoom: 80%;" />

**Column Block for Parallel Learning**：（分块并行）：决策树的学习最耗时的一个步骤就是在每次寻找最佳分裂点是都需要对特征的值进行排序。而 XGBoost 在训练之前对根据特征对数据进行了排序，然后保存到块结构中，并在每个块结构中都采用了稀疏矩阵存储格式（Compressed Sparse Columns Format，CSC）进行存储，后面的训练过程中会重复地使用块结构，可以大大减小计算量。

这种块结构存储的特征之间相互独立，方便计算机进行并行计算。在对节点进行分裂时需要选择增益最大的特征作为分裂，这时各个特征的增益计算可以同时进行，这也是 Xgboost 能够实现分布式或者多线程计算的原因。

**优点**：

1. 精度高：GBDT 只用到一阶泰勒展开，而 XGBoost 对损失函数进行了二阶泰勒展开。另外 XGBoost 工具支持自定义损失函数，只要函数可一阶和二阶求导。
2. 灵活性更强：传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这时xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。
3. 正则化：在目标函数中加入了正则项，用于控制模型的复杂度，防止过拟合，从而提高模型的泛化能力。
4. Shrinkage：相当于学习速率。XGBoost 在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。
5. 列抽样：借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算。
6. 节点分裂方式：传统 GBDT 采用的是均方误差作为内部分裂的增益计算指标，XGBoost是经过优化推导后的目标函数。
7. 缺失值处理：在计算分裂增益时不会考虑带有缺失值的样本，这样就减少了时间开销。在分裂点确定了之后，将带有缺失值的样本分别放在左子树和右子树，比较两者分裂增益，选择增益较大的那一边作为默认分裂方向。
8. 并行化：由于 Boosting 本身的特性，无法像随机森林那样树与树之间的并行化。XGBoost 的并行主要体现在特征粒度上，在对结点进行分裂时，由于已预先对特征排序并保存为block 结构，每个特征的增益计算就可以开多线程进行，极大提升了训练速度。

**缺点**：

- 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，但在节点分裂过程中仍需要遍历数据集；
- 预排序过程的空间复杂度过高，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，相当于消耗了两倍的内存。

### LightGBM

[【白话机器学习】算法理论+实战之LightGBM算法](https://cloud.tencent.com/developer/article/1651704)：浅显易懂，讲的特别棒！
[深入理解LightGBM](https://mp.weixin.qq.com/s/zejkifZnYXAfgTRrkMaEww)：原论文精讲

LightGBM（Light Gradient Boosting Machine）是一种高效的 Gradient Boosting 算法， 主要用于解决GBDT在海量数据中遇到的问题，以便更好更快的用于工业实践中。从 LightGBM 名字我们可以看出其是轻量级（Light）的梯度提升机器（GBM）。

 LightGBM 可以看成是XGBoost的升级加强版。如果说XGB为GBDT类算法在提升计算精度上做出了里程碑式的突破，那么LGBM则是在计算效率和内存优化上提出了开创性的解决方案，一举将GBDT类算法计算效率提高了近20倍、并且计算内存占用减少了80%，这也最终使得GBDT类算法、这一机器学习领域目前最高精度的预测类算法，能够真正应用于海量数据的建模预测。下面我们就简单介绍下LightGBM优化算法。

**直方图算法**（histogram algorithm）基于Histogram的决策树算法，是替代XGBoost的预排序（pre-sorted）算法的。简单来说，就是把连续的浮点特征值离散化成$k$个整数，形成一个一个的箱体（bins）。并根据特征值所在的bin对其进行梯度累加和个数统计，构造一个宽度为$k$的直方图。然后根据直方图的离散值，遍历寻找最优的分割点。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Histogram_algorithm.png)

对于连续特征来说，分箱处理就是特征工程中的离散化。对于分类特征来说，则是每一种取值放入一个分箱 (bin)，且当取值的个数大于最大分箱数时，会忽略那些很少出现的分类值。

内存占用更小： 在Lightgbm中默认的分箱数 (bins) 为256。XGBoost需要用32位的浮点数去存储特征值，并用32位的整形去存储索引，而 LightGBM只需要用8位去存储直方图，从而极大节约了内存存储。

计算代价更小：相对于 XGBoost 中预排序每个特征都要遍历数据，复杂度为 O(#data\*#featrue) ，而直方图只需要遍历每个特征的直方图，复杂度为 O(#bins\*#featrue) 。而我们知道 #data\>\>#bins

直方图算法还能够做差加速：当节点分裂成两个时，右边叶子节点的直方图等于其父节点的直方图减去左边叶子节点的直方图。从而大大减少构建直方图的计算量。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/hist_diff.png" style="zoom: 67%;" />

**单边梯度采样**（Gradient-based One-Side Sampling，GOSS） ：GOSS算法从减少样本的角度出发，排除大部分小梯度的样本，仅用剩下的样本计算信息增益，它是一种在减少数据量和保证精度上平衡的算法。

我们观察到GBDT中每个数据都有不同的梯度值，样本的梯度越小，样本的训练误差就越小。因此LightGBM提出GOSS算法，它保留梯度大的样本，对梯度小的样本进行随机采样，相比XGBoost遍历所有特征值节省了不少时间和空间上的开销。

算法介绍：首先选出梯度绝对值最大的 $a\times100\%$ 的样本，然后对剩下的样本，再随机抽取 $b\times100\%$ 的样本。最后使用已选的数据来计算信息增益。但是这样会引起分布变化，所以对随机抽样的那部分样本权重放大$(1-a)/b$。

作者通过公式证明了GOSS不会损失很大的训练正确率，并且GOSS比随机采样要好，也就是a=0的情况。

**互斥特征绑定**（Exclusive Feature Bundling，EFB）：高维数据通常是稀疏的，这种稀疏性启发我们设计一种无损的方法来减少特征的维度。

许多特征是互斥的（即特征不会同时为非零值，像one-hot），LightGBM根据这一特点提出了EFB算法将互斥的特征合并成一个特征，从而将特征的维度降下来。相应的，构建histogram的时间复杂度叶从O(\#data\*\#feature) 变为 O(\#data\*\#bundle)  ，这里 \#bundle 是融合绑定后特征包的个数。

**决策树生长策略**：带深度限制的Leaf-wise的叶子生长策略

XGBoost 采用 Level-wise （按层生长）策略，该策略遍历一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上Level-wise是一种低效的算法，因为它不加区分的对待同一层的叶子，实际上很多叶子的分裂增益较低，没必要进行搜索和分裂，因此带来了很多没必要的计算开销。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/level_wise_tree_growth.png)

LightGBM采用 leaf-wise（按叶子生长）策略，以降低模型损失最大化为目的，对当前叶所有叶子节点中分裂增益最大的叶子节点进行分裂。leaf-wise的缺点是会生成比较深的决策树，为了防止过拟合，可以在模型参数中设置树的深度。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/leaf_wise_tree_growth.png)

**直接支持类别特征**

实际上大多数机器学习工具都无法直接支持类别特征，一般需要把类别特征，通过 one-hot 编码，降低了空间和时间的效率。

LightGBM优化了对类别特征的支持，可以直接输入类别特征，不需要额外的0/1展开。LightGBM采用 many-vs-many 的切分方式将类别特征分为两个子集，实现类别特征的最优切分。

算法流程如下图所示，在枚举分割点之前，先把直方图按照每个类别对应的label均值进行排序；然后按照排序的结果依次枚举最优分割点。当然，这个方法很容易过拟合，所以LightGBM里面还增加了很多对于这个方法的约束和正则化。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/LGB_many_vs_many.png" style="zoom: 67%;" />

**并行计算**：LightGBM支持特征并行、数据并行和投票并行

传统的特征并行主要思想是在并行化决策树中寻找最佳切分点，在数据量大时难以加速，同时需要对切分结果进行通信整合。LightGBM则是使用分散规约(Reduce scatter)，将任务分给不同的机器，降低通信和计算的开销，并利用直方图做加速训练，进一步减少开销。

特征并行是并行化决策树中寻找最优划分点的过程。特征并行是将对待特征进行划分，每个worker找到局部的最佳切分点，使用点对点通信找到全局的最佳切分点。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/LGB_feature_parallelization.png" style="zoom: 67%;" />

数据并行的目标是并行化真个决策树学习过程。每个worker中拥有部分数据，独立的构建局部直方图，合并后得到全局直方图，在全局直方图中寻找最优切分点进行分裂。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/LGB_data_parallelization.png" style="zoom: 67%;" />

LightGBM 采用一种称为 PV-Tree 的算法进行投票并行（Voting Parallel），其实本质上也是一种数据并行。PV-Tree 和普通的决策树差不多，只是在寻找最优切分点上有所不同。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/LGB_voting_based_parallel.png" style="zoom:67%;" />

## 投票方法

在训练好基学习器后，如何将这些基学习器的输出结合起来产生集成模型的最终输出，下面将介绍一些常用的结合策略。

对于回归问题，输出基回归模型的平均值。

**简单平均法**（simple averaging）：
$$
H(\mathbf x)=\frac{1}{M}\sum_{m=1}^Mh_m(\mathbf x)
$$
**加权平均法**（weighted averaging）：
$$
H(\mathbf x)=\sum_{m=1}^Mw_mh_m(\mathbf x)\\
\text{s.t.}\quad w_m>0,\sum_{m=1}^Mw_m=1
$$
对于分类问题，最常见的组合方法是硬投票和软投票。类别标签 $y\in\{c_1,c_2,\cdots,c_K\}$ 。

**硬投票**（hard voting）：即多数投票（ majority voting）。基学习器 $h_m$ 输出类别标签 $h_m(\mathbf x)\in\{c_1,c_2,\cdots,c_K\}$，预测结果中出现最多的类别。
$$
H(\mathbf x)=\arg\max_{c_k}\sum_{m=1}^Mh_m(\mathbf x|y=c_k)
$$
例如，给定样本的预测是

| classifier   | class 1 | class 2 | class 3 |
| ------------ | ------- | ------- | ------- |
| classifier 1 | 1       | 0       | 0       |
| classifier 2 | 1       | 0       | 0       |
| classifier 3 | 0       | 1       | 0       |
| sum          | 2       | 1       | 0       |

这里预测的类别为 class 1。

**软投票**（soft voting）：基学习器 $h_m$ 输出类别概率 $h_m(\mathbf x)\in[0,1]$，会选出基学习器的加权平均概率最大的类别。
$$
H(\mathbf x)=\arg\max_c\sum_{m=1}^Mw_mh_m(\mathbf x|y=c)\\
\text{s.t.}\quad w_m>0,\sum_{m=1}^Mw_m=1
$$
用一个简单的例子说明，其中3个分类器的权重相等 

| classifier       | class 1 | class 2 | class 3 |
| ---------------- | ------- | ------- | ------- |
| classifier 1     | 0.2     | 0.5     | 0.3     |
| classifier 2     | 0.6     | 0.3     | 0.1     |
| classifier 3     | 0.3     | 0.4     | 0.3     |
| weighted average | 0.37    | 0.4     | 0.23    |

这里预测的类别为 class 2，因为它具有最高的加权平均概率。

实际中，软投票和硬投票可以得出完全不同的结论。相对于硬投票，软投票考虑到了预测概率这一额外的信息，因此可以得出比硬投票法更加准确的结果。

## Stacking

stacking是指训练一个模型用于组合基学习器的方法，组合的学习器称为元学习器（meta learner）。
$$
H(\mathbf x)=H(h_1(\mathbf x),h_2(\mathbf x),\cdots,h_M(\mathbf x);\Theta)
$$

1. 首先，训练$M$个不同的基学习器，最好每个基学习器都基于不同的算法（KNN、SVM、RF等等），以产生足够的差异性。
2. 然后，每一个基学习器的输出作为组合学习器的特征来训练一个模型，以得到一个最终的结果。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/stacking_algorithm.png" style="zoom:80%;" />

若直接使用基学习器的训练集来生成元学习器的训练集，则过拟合风险会比较大；因此一般通过交叉验证，用基学习器未使用的样本来产生元学习器的训练样本。

以 k-folds 交叉验证为例

1. 初始训练集$D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}$被随机划分为 $k$ 个大小相似的集合 $\{D_1,D_2,\cdots,D_k\}$ 。令 $D_j$ 和 $D-D_j$ 分别表示第 $j$ 折的测试集和训练集。
2. 给定$M$个基学习算法，初级学习器 $h_m^{(j)}$ 通过在 $D-D_j$ 上使用第$m$个学习算法而得。
3. 对 $D_j$ 中每个样本 $\mathbf x_i$，令 $z_{im}=h_m^{(j)}(\mathbf x_i)$ ，则由 $\mathbf x_i$ 产生元学习器的训练集特征 $\mathbf z_i=(z_{i1},z_{i2},\cdots,z_{iM})$，目标值为 $y_i$。
4.  于是，在整个交叉验证过程结束后，从这$M$个基学习器产生的元学习器的训练集是 $D'=\{(\mathbf z_1,y_1),(\mathbf z_2,y_2),\cdots,(\mathbf z_N,y_N)\}$ 。

有研究表明，元学习器通常采用概率作为输入特征，用多响应线性回归（MLR）算法效果较好。







[^taylor]: 泰勒展开式 $f(x+\Delta x)=f(x)+f'(x)\Delta x+\dfrac{1}{2}f''(x)(\Delta x)^2+\cdots$
[^cdot]: 上述推导中用到一个公式 $\mathbf x^T\mathbf{Ay}=\mathbf y^T\mathbf{Ax}$，其中 $\mathbf x,\mathbf y$ 为向量，$\mathbf A$ 为对称阵，即 $\mathbf A^T=\mathbf A$。可通过转置的计算法则得到，过程如下：由于 $\mathbf x^T\mathbf{Ay}$ 为常数，所以 $\mathbf x^T\mathbf{Ay}=(\mathbf x^T\mathbf{Ay})^T=\mathbf y^T\mathbf A^T\mathbf x=\mathbf y^T\mathbf{Ax}$