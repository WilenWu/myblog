---
title: 机器学习中的优化
categories:
  - 'Artificial Intelligence'
  - Appendices
tags:
  - 机器学习
description: 优化是找出函数的最大值或最小值的方法
top_img: '#66CCFF'
katex: true
abbrlink: b0a72714
date: 2022-07-16 14:51:03
updated:
cover:
---

大多数机器学习算法都涉及某种形式的优化。优化指的是改变 $\mathbf x$ 以最小化或最大化 $f(\mathbf x)$ 的任务。我们通常以最小化 $f(\mathbf x)$ 指代大多数最优化问题，最大化可经由最小化 $-f(\mathbf x)$ 来实现。

我们通常使用一个上标 $*$ 表示函数的极值点
$$
\mathbf x^*=\arg\min\limits_{\mathbf x}f(\mathbf x)
$$
<!-- more -->

# 无约束优化

假设$f(x)$是一元函数，具有连续一阶导数和二阶导数。在无约束的优化问题中，任务是找出最小化或最大化$f(x)$的解$x^*$，而不对$x^*$施加任何约束条件。解$x^*$称作称为临界点（critical point）或驻点（stationary point）。，可以通过取$f$的一阶导数，并令它等于零找到:
$$
\frac{\mathrm{d}y}{\mathrm{d}x}\mid_{x=x^*}=0
$$

$f(x)$可以取极大或极小值，取决于该函数的二阶导数。

-   如果在$x=x^*$有$\cfrac{\mathrm{d}^2y}{\mathrm{d}x^2}<0$，则$x^*$是极大值。
-   如果在$x=x^*$有$\cfrac{\mathrm{d}^2y}{\mathrm{d}x^2}>0$，则$x^*$是极小值。
-   当在$x=x^*$有$\cfrac{\mathrm{d}^2y}{\mathrm{d}x^2}=0$，则$x^*$是拐点。

下图函数包含三个驻点（极大、极小和拐点）：

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/stationary-point.png)

该定义可以推广到多元函数$f(x_1,x_2,\cdots,x_d)$，这里找驻点$\mathbf x^*=(x_1^*,x_2^*,\cdots,x_d^*)^T$的条件为
$$
\frac{\mathrm{d}y}{\mathrm{d}x_i}\mid_{x_i=x_i^*}=0,\forall i=1,2,\cdots,d \tag{1.1}
$$
然而，不像一元函数，确定$\mathbf x^*$是极大还是极小值更困难。困难的原因在于我们需要对所有可能的一对$i,j$，考虑偏导数$\cfrac{\partial^2 f}{\partial x_i\partial x_j}$。二阶偏导数的完全集由 Hessian 矩阵给出：
$$
\mathbf H(\mathbf x)=\begin{pmatrix}
\cfrac{\partial^2 f}{\partial x_1^2}&\cfrac{\partial^2 f}{\partial x_1\partial x_2}&\cdots&\cfrac{\partial^2 f}{\partial x_1\partial x_d} \\
\cfrac{\partial^2 f}{\partial x_2\partial x_1}&\cfrac{\partial^2 f}{\partial x_2^2}&\cdots&\cfrac{\partial^2 f}{\partial x_2\partial x_d} \\
\vdots &\vdots &\ddots &\vdots \\ 
\cfrac{\partial^2 f}{\partial x_d\partial x_1}&\cfrac{\partial^2 f}{\partial x_d\partial x_2}&\cdots&\cfrac{\partial^2 f}{\partial x_d^2} \\
\end{pmatrix} \tag{1.2}
$$

- 如果$\mathbf H(\mathbf x^*)$是正定的，则$\mathbf x^*$是极小平稳点。Hessian 矩阵$\mathbf H$是正定的$\iff \forall \mathbf x\not=\mathbf 0,\mathbf x^T\mathbf H\mathbf x>0$
- 如果$\mathbf H(\mathbf x^*)$是正定的，则$\mathbf x^*$是极大平稳点。Hessian 矩阵$\mathbf H$是正定的$\iff \forall \mathbf x\not=\mathbf 0,\mathbf x^T\mathbf H\mathbf x<0$
- 具有不定 Hessian 矩阵的平稳点是鞍点(saddlepoint)，它在一个方向上具有极小值，在另一个方向上具有极大值。

在许多情况下，找解析解是一个很困难的问题，这就迫使我们使用数值方法找近似解。本文简略回顾用于求解优化问题的各种技术。

对于线性回归来说均方误差代价函数是一个凸函数（如下图），梯度下降可以一步一步找到全局最小值。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/convex-function.png" style="zoom: 33%;" />

# 拉格朗日乘子法

拉格朗日乘子法(Lagrange multipliers)是一种寻找多元函数在一组约束下的极值的方法。通过引入拉格朗日乘子，可将有 $d$ 个变量与 $k$ 个约束条件的最优化问题转化为具有 $d+k$ 个变量的无约束优化问题求解。

## 等式约束

考虑受限于如下形式的等式约束：
$$
g_i(\mathbf x)=0,i= 1,2,\cdots,p
$$
求$f(x_1,x_2,\cdots,x_d)$的极小值问题。

一种称作拉格朗日乘子的方法可以用来解约束优化问题。该方法涉及如下步骤。
(1) 定义拉格朗日函数
$$
L(\mathbf x,λ)=f(\mathbf x)+\sum_{i=1}^{p} λ_ig_i(\mathbf x)
$$
其中$λ_i$是哑变量，称作拉格朗日乘子(Lagrange multiplier)。
(2) 令拉格朗日函数关于$\mathbf x$和拉格朗日乘子的一阶导数等于0：
$$
\cfrac{\partial L}{\partial x_i}=0,\forall i=1,2,\cdots,d
$$
并且
$$
\cfrac{\partial L}{\partial λ_i}=0,\forall i=1,2,\cdots,p
$$
(3) 求解步骤(2)得到的$(d+p)$个方程，得到平稳点$\mathbf x^*$和对应的$λ_i$的值。

## 不等式约束

考虑受限于如下形式的等式约束：
$$
h_i(\mathbf x)=0,i= 1,2,\cdots,q
$$
求$f(x_1,x_2,\cdots,x_d)$的最小值问题。

求解这类问题的方法与上面介绍的拉格朗日方法非常相似。然而，不等式约束把一些附加条件施加到优化问题上。特殊地，上述优化问题导致如下拉格朗日函数：
$$
L(\mathbf x,λ)=f(\mathbf x)+\sum_{i=1}^{q} λ_ih_i(\mathbf x)
$$
和称作Karush-Kuhn-Tucker(KKT)条件的约束：
$$
\cfrac{\partial L}{\partial x_i}=0,\forall i=1,2,\cdots,d \\
h_i(\mathbf x)\leqslant 0,\forall i=1,2,\cdots,q \\
λ_i\geqslant 0,\forall i=1,2,\cdots,q \\
λ_ih_i(\mathbf x)=0,\forall i=1,2,\cdots,q \\
$$
注意，拉格朗日乘子在不等式约束中出现，不再是不受限的。
求解KKT条件可能是一项相当艰巨的任务，当约束不等式的数量较大时尤其如此。在这种情况下，求闭型解不再可行，而需要使用诸如线性和二次规划这样的数值优化技术。



# 梯度下降

梯度下降法（gradient descent）也称最速下降法（steepest descent），是一种常用的一阶（first-order）优化方法，是求解无约束优化问题最简单、最经典的迭代方法之一。它被广泛应用于机器学习，是许多算法的基础，比如线性回归、逻辑回归，以及神经网络的早期实现。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/gradient-descent.svg" style="zoom:50%;" />

假定函数$J(\mathbf w)$连续可微，要求解的无约束最优化问题是
$$
\min\limits_{\mathbf w} J(\mathbf w)
$$
梯度下降法希望通过不断执行迭代过程**收敛**到局部极小点。根据泰勒一阶展开式有
$$
J(\mathbf{w})\approx J(\mathbf w_0)+(\mathbf w-\mathbf w_0)^T\nabla J(\mathbf w_0)
$$
我们希望找到使 $J$ 下降得最快的方向，即 $J(\mathbf w)-J(\mathbf w_0)<0$  且取得最小值
$$
\min\limits_{\mathbf w-\mathbf w_0}(\mathbf w-\mathbf w_0)^T\nabla J(\mathbf w_0)=\min\limits_{\mathbf w-\mathbf w_0}\|\mathbf w-\mathbf w_0\|_2\|\nabla J(\mathbf w_0)\|_2\cos\theta
$$
当 $\mathbf w-\mathbf w_0$ 与梯度方向相反时（$\cos\theta=-1$）取得最小值。因此梯度下降迭代中建议下一轮的点为
$$
\mathbf w=\mathbf w_0-\lambda\nabla J(\mathbf w_0)
$$
数值 $\lambda>0$ 称为**学习率** (learning rate)，作用是控制下降的步幅。梯度向量 $\nabla J(\mathbf w)=\dfrac{\partial J(\mathbf{w})}{\partial\mathbf w}$ 控制下降的方向。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GradientDescentAlgorithm.png" style="zoom:80%;" />

**学习率**

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/LearningRateOfGradientDescent.svg)

- 如果 $\lambda$ 太小，梯度下降会起作用，但会很慢。
- 如果 $\lambda$ 太大，梯度下降可能不断跨过最小值，永不收敛。

使用梯度下降法时，通常建议尝试一系列 $\lambda$ 值，对于每一个学习率画出少量迭代的代价函数，在尝试了一系列 $\lambda$ 后，你可能会选择能快速且持续降低 $J$ 的 $\lambda$ 值。
$$
\text{Values of learning rate to try:} \\
\begin{matrix}
\cdots & 0.001 & 0.01 & 0.1 & 1 & \cdots \\
\cdots & 0.003 & 0.03 & 0.3 & 3 & \cdots \\
\cdots & 0.006 & 0.06 & 0.6 & 6 & \cdots \\
\end{matrix}
$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/fix-learning-rate.svg" style="zoom: 67%;" />

**梯度**：当我们接近局部最小值时，导数会自动变小。因此，即使学习率 $\lambda$ 保持在某个固定值，更新的步幅也会自动变小。

**检测梯度下降是否收敛**

- 学习曲线（LearningCurve）：横轴是梯度下降的迭代次数，纵轴代表代价函数 $J(\mathbf w)$。不同的应用场景中，梯度下降的收敛速度可能有很大差异。事实证明，我们很难事先知道梯度下降要经过多少次迭代才能收敛，所以可以先画个学习曲线后再训练模型。
- 另一种方法是自动收敛测试 (Automatic convergence test)：我们设置一个小容差 $\epsilon$ (=0.001)，如果代价函数在一次迭代中减少的量小于这个值，则可以认为它收敛了。

记住，收敛是指你找到了代价函数 $J$ 接近最小值的可能参数。选出正确的 $\epsilon$ 是相当困难的，所以更倾向于使用学习曲线。

**梯度下降法缺点**

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/LocalMinimumGradientDescent.png" alt="LocalMinimumGradientDescent" style="zoom: 36%;" /><img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GradientDescent-2.png" style="zoom: 67%;" />

- 初始位置的不同可能导致下降到不同的极值点。
- 多维情况下，单个点处每个方向上的导数可能差别很大。梯度下降不知道导数的这种变化，也会表现得很差。

# 随机梯度下降

机器学习算法的目标函数通常可以分解为训练样本上的求和。机器学习中的优化算法在计算参数的每一次更新时通
常仅使用整个代价函数中一部分项来估计代价函数的期望值。

准确计算这个期望的计算代价非常大，因为我们需要在整个数据集上的每个样本上评估模型。在实践中，我们可以从数据集中随机采样少量的样本，然后计算这些样本上的平均值。

使用整个训练集的优化算法被称为批量（batch）或确定性（deterministic）梯度算法，因为它们会在一个大批量中同时处理所有样本。

每次只使用单个样本的优化算法有时被称为随机（stochastic）或者在线（on-line）算法。

大多数用于深度学习的算法介于以上两者之间，使用一个以上，而又不是全部的训练样本。传统上，这些会被称为小批量（minibatch）或小批量随机（minibatch stochastic）方法，现在通常将它们简单地称为随机（stochastic）方法。

**随机梯度下降**

但是如果样本非常多的情况下，计算复杂度较高，但是，实际上我们并不需要绝对的损失函数下降的方向，我们只需要损失函数的期望值下降，但是计算期望需要知道真实的概率分布，我们实际只能根据训练数据抽样来估算这个概率分布（经验风险）

我们知道，样本量 $m$ 越大，样本近似真实分布越准确，但是对于一个数据，可以确定的标准差仅和 $\sqrt{m}$ 成反比，而计算速度却和 $m$ 成正比。因此可以每次使用较少样本，则在数学期望的意义上损失降低的同时，有可以提高计算速度，如果每次只使用一个错误样本，我们有下面的更新策略（根据泰勒公式，在负方向)：
$$
\mathbf w \gets \mathbf w-\lambda\nabla J(\mathbf w)
$$
是可以收敛的，同时使用单个观测更新也可以在一定程度上增加不确定度，从而减轻陷入局部最小的可能。在更大规模的数据上，常用的是小批量随机梯度下降法。

随机梯度下降（Stochastic Gradient Descent, SGD）是求解无约束优化问题的一种优化方法。与**批量梯度下降** (batch gradient descent)相比，SGD通过一次只考虑一个训练样本来逼近，的真实梯度。

随机梯度下降的优点是：

- 高效
- 易于实现 (有大量优化代码的机会)

随机梯度下降的缺点包括：

- SGD需要一些超参数，例如正则化参数和迭代次数
- SGD对特征缩放非常敏感

**小批量梯度下降法**

小批量梯度下降算法是FG和SG的折中方案，在一定程度上兼顾了以上两种方法的优点。每次从训练样本集上随机抽取一个小样本集，在抽出来的小样本集上采用FG迭代更新权重。

# 牛顿法和拟牛顿法

当目标函数 $J(\mathbf w)$ 二阶连续可微时，可使用更精确的二阶泰勒展开式，这样就得到了牛顿法(Newton's method)。牛顿法是典型的二阶方法，其迭代轮数远小于梯度下降法。

牛顿法是使用如下更新方程渐近地寻找函数的平稳点的多种增量方法之一：
$$
\mathbf x=\mathbf x+\lambda g(\mathbf x)\tag{1.7}
$$
函数$g(\mathbf x)$确定搜索方向，而$\lambda$确定步长。

牛顿法基于使用函数$f(x)$的二次近似。通过使用$f$在$x_0$的泰勒级数展开式，得到如下表达式：
$$
f(x)= f(x_0)+(x-x_0)f'(x_0)+\frac{(x-x_0)^2}{2}f''(x_0)\tag{1.5}
$$
取该函数关于$x$的导数，并令它等于零，得到如下等式：
$$
f'(x)= f'(x_0)+(x-x_0)f''(x_0)=0 \\
x=x_0-\frac{f'(x_0)}{f''(x_0)}
\tag{1.6}
$$
可以使用公式(1.6)更新$x$，直到它收敛于极小值。可以证明牛顿法是二次收敛的，尽管它可能不收敛，特别是当初始点$x_0$远离极小值时。

**牛顿法**：

$\begin{aligned}\hline
1:\ &\text{令}x_0\text{为初始点} \\
2:\ &\textbf{while}\ |f'(x_0)|>ϵ \mathbf{\ do} \\
3:\ &\quad x=x_0-\cfrac{f'(x_0)}{f''(x_0)} \\
4:\ &\quad x_0=x \\
5:\ &\textbf{end while} \\
6:\ &\textbf{return } x \\
\hline\end{aligned}$

用梯度算子$\nabla f(\mathbf x)$替换一阶导数$f'(x)$，用黑森矩阵$\mathbf H$替换二阶导数$f''(x)$，可以把牛顿法推广到多元数据：
$$
\mathbf x=\mathbf x-\mathbf H^{-1}\nabla f(\mathbf x)
$$
然而，更容易的办法不是计算黑森矩阵的逆，而是解如下方程：
$$
\mathbf{Hz}=-\nabla f(\mathbf x)
$$
来得到向量$\mathbf z$。找平稳点的迭代公式修改为$\mathbf{x=x+z}$。

牛顿法使用了二阶导数 $\nabla^2J(\mathbf x)$ ， 其每轮迭代中涉及到海森矩阵的求逆，计算复杂度相当高，尤其在高维问题中几乎不可行。若能以较低的计算代价寻找海森矩阵的近似逆矩阵，则可显著降低计算开销，这就是拟牛顿法(quasi-Newton method)。

# Adam

Adam 是 Adaptive Moment estimation (自适应矩估计) 的简称。它通常比梯度下降快得多，已经称为实践者训练神经网络的行业标准。根据梯度下降的过程，Adam 算法可以自动调整学习率，即对学习率容错性更强。

Adam 算法并不是全局都使用同一个 $\alpha$ ，模型的每个参数都会用不同的学习率。
$$
\begin{align} 
\text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& \mathbf w = \mathbf w -  \mathbf\Lambda\nabla J(\mathbf{w})
  \newline \rbrace
\end{align}
$$

where $\mathbf\Lambda=\text{diag}(\lambda_1,\lambda_2,\cdots,\lambda_n)$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/AdamAlgorithmIntuition.svg" style="zoom: 80%;" />

- 如果参数持续沿着大致相同的方向移动，我们将提高这个参数的学习率。
- 相反，如果一个参数来回振荡，我们将减小这个参数的学习率。

**优缺点**：

- 不需要手动指定学习率 
- 通常收敛速度远大于梯度下降
- 算法过于复杂

# 坐标下降法

# 黄金搜索

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Golden-section-search.png)

考虑图中所示的单峰分布，其极小值在区间a和b之间。黄金搜索方法迭代地找相继较小的、包含极小值的区间，直到区间的宽度足够小，可以近似平稳点。为了确定较小的区间，选择两个点c和d，使得区间 $(a,c,d)$ 和$(c,d,b)$具有相等的宽度。令$c-a=b-d=\alpha(b-a), d-c=\beta(b-a)$。因此
$$
1=\frac{(b-d)+(d-c)+(c-a)}{b-a}=\alpha+\beta+\alpha
$$
或等价地
$$
β=1-2α\tag{1.3}
$$
还要选择宽度，满足以下条件，使得我们可以使用递归过程：
$$
\frac{d-c}{b-c}=\frac{c-a}{b-a}
$$
或等价地
$$
\frac{β}{1-α}=α\tag{1.4}
$$
公式(1.3)和公式(1.4)中的方程可以一起求解，得到$\alpha,\beta$。通过比较$f(c)$和$f(d)$，可以确定极小值在区间$(a,c,d)$，还是在区间$(c,d,b)$。然后递归地划分包含最小值的区间，直到区间宽度足够小，可以近似极小值。

**黄金搜索算法**

$\begin{aligned}\hline
1:\ &c=a+α(b-a) \\
2:\ &\textbf{while } b -a >ϵ \mathbf{\ do} \\
3:\ &\quad d=b-α(b-a)  \\
4:\ &\quad \textbf{if } f(d)>f(c) \textbf{ then}  \\
5:\ &\qquad b=d \\
6:\ &\quad \textbf{else} \\
7:\ &\qquad a=c,c=d  \\
8:\ &\quad \textbf{end if} \\
9:\ &\textbf{end while} \\
10:\ &\textbf{return } c \\
\hline\end{aligned}$

除了假定函数在初始区间$[a,b]$中连续并且是单峰的之外，黄金搜索方法不对函数做其他假定。它线性收敛于极小值解。





# 高级优化算法

- Conjugate gradie
- BFGS  共轭梯度算法
- L-BFGS
