---
title: 机器学习(IV)--监督学习(四)支持向量机
date: 2022-11-27 21:40
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-supervised-learning.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 72ac77c8
description: 
---

# 支持向量机

**支持向量机**（support vector machine, SVM）是一种用于分类、回归和异常检测的有监督学习方法。按照数据集的特点，可分为3类：

1. 当数据集线性可分时，通过硬间隔最大化（不容许错误分类），学习线性可分支持向量机，又叫硬间隔支持向量机（hard-margin SVM）
2. 当数据集近似线性可分时，通过软间隔最大化（容许一定误差），学习线性支持向量机（linear SVM），又叫软间隔支持向量机（soft-margin SVM）
3. 当数据集线性不可分时，通过核技巧（kernel method）及软间隔最大化，学习非线性支持向量机（non-linear SVM）

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in\{-1,+1\}$ 。

## 间隔最大化

本节讨论线性可分的数据集，即存在无数超平面，可以将正样本和负样本正确分类。从几何角度，具有最大间隔的超平面有更好的泛化性能，因为该超平面对训练样本局部扰动的容忍性最好。SVM 采用**最大间隔超平面**（maximal margin hyperplane）作为决策边界，此时的超平面是存在且唯一的。

分离的超平面可以写为如下形式
$$
\mathbf{w}^T\mathbf{x}+b=0
$$
其中 $\mathbf w=(w_1,w_2,\cdots,w_p)^T$ 为法向量，决定了超平面的方向；$b$ 为位移项，决定了超平面与原点之间的距离。显然，超平面可被法向量 $\mathbf w$ 和位移 $b$ 确定，下面我们将其记为 $(\mathbf w,b)$ 。

**Model**：
$$
f_{\mathbf{w},b}(\mathbf{x})=\text{sign}(\mathbf{w}^T\mathbf{x}+b)
$$
符号函数 
$$
\text{sign}(z)=\begin{cases}+1 & \text{if }z\geqslant 0\\ -1 & \text{if }z<0\end{cases}
$$
**最大间隔**：

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/SVM-maximum-margin.svg" style="zoom:80%;" />

距离超平面 $(\mathbf w,b)$ 最近的样本，被称为**支持向量** （support vector）。如上图，设超平面两侧平行边界的方程分别为
$$
\begin{aligned}
B_1: & \mathbf{w}^T\mathbf{x}+b=\gamma \\
B_2: & \mathbf{w}^T\mathbf{x}+b=-\gamma
\end{aligned}
$$
超平面 $B_1,B_2$ 间的距离称为**间隔**（margin）
$$
\text{margin}=\frac{2\gamma}{\|\mathbf w\|}
$$
假设样本能被超平面正确分类，则
$$
\begin{cases}
\mathbf{w}^T\mathbf{x}_i+b\geqslant \gamma  & \text{if }y_i=+1\\
\mathbf{w}^T\mathbf{x}_i+b\leqslant- \gamma & \text{if }y_i=-1
\end{cases}
$$
可简写为
$$
y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant \gamma \quad i=1,2,\cdots,N
$$

**参数估计**：由于超平面的系数经过同比例缩放不会改变这个平面，我们不妨给出约束 $\gamma=1$，从而得到唯一系数。那么最大化间隔可表示为：
$$
\begin{aligned}
\max\limits_{\mathbf w,b}&\frac{2}{\|\mathbf w\|} \\
\text{s.t.}&\quad y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant 1, \quad i=1,2,\cdots,N
\end{aligned}
$$

s.t. 是 subject to (such that) 的缩写，表示约束条件。约束为分类任务的要求。

显然，为了最大化间隔，仅需最大化 $\|\mathbf w\|^{-1}$，这等价于最小化 $\|\mathbf w\|^2$ 。于是上式可重写为
$$
\begin{aligned}
\min\limits_{\mathbf w,b}&\frac{1}{2}\|\mathbf w\|^2 \\
\text{s.t.}&\quad y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant 1, \quad i=1,2,\cdots,N
\end{aligned}
$$
这就是 SVM 的基本形式，是一个包含 $N$ 个约束的凸优化问题。

## 对偶问题

支持向量机通常将原始问题（primal problem）转化成拉格朗日对偶问题（dual problem）来求解。首先引入 Lagrange 函数
$$
L(\mathbf w,b,\mathbf\alpha)=\frac{1}{2}\mathbf w^T\mathbf w+\sum_{i=1}^N\alpha_i(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))
$$
参数 $\alpha_i\geqslant 0$ 称为拉格朗日乘子（Lagrange multiplier）。根据拉格朗日对偶性，原始问题的对偶问题为
$$
\max_{\mathbf\alpha}\min_{\mathbf w,b}L(\mathbf w,b,\mathbf\alpha)
$$
令 $L(\mathbf w,b,\mathbf\alpha)$ 对 $\mathbf w$ 和 $b$ 的偏导数为 0 可以得到
$$
\sum_{i=1}^N\alpha_iy_i=0\\
\mathbf w=\sum_{i=1}^N\alpha_iy_i\mathbf x_i
$$
将上式带入拉格朗日函数，我们就可以消去 $\mathbf w$ 和 $b$ ，得到对偶最优化问题
$$
\begin{aligned}
\max\limits_{\mathbf\alpha}&\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\mathbf x_i^T\mathbf x_j \\
\text{s.t.}&\quad \sum_{i=1}^N\alpha_iy_i=0 \\
&\quad\alpha_i\geqslant 0, \quad i=1,2,\cdots,N
\end{aligned}
$$
原问题和对偶问题等价的充要条件为其满足 KKT（Karush-Kuhn-Tucker）条件
$$
\begin{cases}
\alpha_i\geqslant 0  \\
1-y_i(\mathbf{w}^T\mathbf{x}_i+b) \leqslant 0  \\
\alpha_i(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))=0 
\end{cases}
$$
解出 $\mathbf\alpha$ 后，求出 $\mathbf w$ 和 $b$ 即可得到模型
$$
\begin{aligned}
f_{\mathbf{w},b}(\mathbf{x})&=\text{sign}(\mathbf{w}^T\mathbf{x}+b) \\
&=\text{sign}(\sum_{i=1}^N\alpha_iy_i\mathbf x_i^T\mathbf x+b)
\end{aligned}
$$
由 KKT 互补条件可知，对任意训练样本 $(\mathbf x_i,y_i)$， 总有 $\alpha_i=0$ 或 $y_i(\mathbf{w}^T\mathbf{x}_i+b)=1$。

- 若 $\alpha_i=0$ ，则该样本将不会在式 $f_{\mathbf{w},b}(\mathbf{x})$ 的求和中出现，也就不会对模型有任何影响
- 若 $\alpha_i>0$，则必有 $y_i(\mathbf{w}^T\mathbf{x}_i+b)=1$ ，所对应的样本点位于最大间隔边界上，被称为支持向量。这显示出 SVM 的一个重要性质：训练完成后，大部分的训练样本都不需保留，最终模型仅与支持向量有关。

不难发现，求解 $\mathbf\alpha$ 是一个二次规划问题，可使用通用的二次规划算法来求解。然而，该问题的规模正比于训练样本数，这会在实际任务中造成很大的开销。为了避开这个障碍，人们通过利用问题本身的特性，提出了很多高效算法， SMO (Sequential Minimal Optimization) 是其中典型代表。

## 软间隔与正则化

在前面的讨论中，假定训练样本是线性可分的，即所有样本都必须严格满足约束
$$
y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant 1
$$
这称为**硬间隔**（hard margin）。然而，现实任务中的数据很难线性可分，这时，我们为每个样本引入**松弛变量**（slack variable） $\xi_i>0$ ，允许某些样本不满足约束，约束条件修改为
$$
y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant 1-\xi_i
$$
使用**软间隔**（soft margin）最大化求解。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/SVM-soft-margin.png" style="zoom:67%;" />

当然，在最大化间隔的同时，不满足约束的样本应尽可能少。于是，引入损失
$$
\xi_i=\max\{0,1-y_i(\mathbf{w}^T\mathbf{x}_i+b)\}
$$
每个样本都有一个对应的松弛变量， 用以表征该样本不满足严格约束的程度。上式称为hinge损失函数
$$
\text{hinge\_loss}(z)=\max\{0,1-z\}
$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/hinge-loss.svg" style="zoom: 80%;" />

因此，软间隔SVM的优化目标可写为
$$
\begin{aligned}
\min\limits_{\mathbf w,b,\mathbf\xi}&\frac{1}{2}\|\mathbf w\|^2+C\sum_{i=1}^N\xi_i \\
\text{s.t.}&\quad y_i(\mathbf{w}^T\mathbf{x}_i+b)\geqslant 1-\xi_i \\
&\quad \xi_i\geqslant 0, \qquad i=1,2,\cdots,N
\end{aligned}
$$
其中，$C$ 为惩罚参数，用于控制惩罚强度。这便是线性 SVM 的基本式。

原始问题的拉格朗日函数为
$$
L(\mathbf w,b,\mathbf\xi,\mathbf\alpha,\mathbf\eta)=\frac{1}{2}\mathbf w^T\mathbf w+C\sum_{i=1}^N\xi_i+\sum_{i=1}^N\alpha_i(1-\xi_i-y_i(\mathbf{w}^T\mathbf{x}_i+b))-\sum_{i=1}^N\eta_i\xi_i
$$
参数 $\alpha_i\geqslant 0,\eta_i$ 称为拉格朗日乘子。令 $L(\mathbf w,b,\mathbf\xi,\mathbf\alpha,\mathbf\eta)$ 对 $\mathbf w,b$ 和 $\mathbf \xi$ 的偏导数为 0 可以得到
$$
\sum_{i=1}^N\alpha_iy_i=0\\
\mathbf w=\sum_{i=1}^N\alpha_iy_i\mathbf x_i \\
\alpha_i+\eta_i=C
$$
将上式带入拉格朗日函数，得到**拉格朗日对偶问题**
$$
\begin{aligned}
\max\limits_{\mathbf\alpha}&\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\mathbf x_i^T\mathbf x_j \\
\text{s.t.}&\quad \sum_{i=1}^N\alpha_iy_i=0 \\
&\quad 0\leqslant\alpha_i\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
上式与线性可分对偶问题唯一不同的是对 $\alpha_i$ 的约束。对软间隔支持向量机， KKT 条件要求
$$
\begin{cases}
\alpha_i\geqslant 0 ,\quad \eta_i \geqslant 0 \\
1-\xi_i-y_i(\mathbf{w}^T\mathbf{x}_i+b)\leqslant 0  \\
\alpha_i(1-\xi_i-y_i(\mathbf{w}^T\mathbf{x}_i+b))=0 \\
\xi_i\geqslant 0 ,\quad \eta_i\xi_i=0
\end{cases}
$$
解出 $\mathbf\alpha,\mathbf\eta$ 后，求出 $\mathbf w$ 和 $b$ 即可得到模型
$$
\begin{aligned}
f_{\mathbf{w},b}(\mathbf{x})&=\text{sign}(\mathbf{w}^T\mathbf{x}+b) \\
&=\text{sign}(\sum_{i=1}^N\alpha_iy_i\mathbf x_i^T\mathbf x+b)
\end{aligned}
$$
由 KKT 条件可知，对任意训练样本 $(\mathbf x_i,y_i)$， 总有 $\alpha_i=0$ 或 $y_i(\mathbf{w}^T\mathbf{x}_i+b)=1-\xi_i$。

- 若 $\alpha_i=0$ ，则该样本将不会在式 $f_{\mathbf{w},b}(\mathbf{x})$ 的求和中出现，也就不会对模型有任何影响
- 若 $\alpha_i>0$，则必有 $y_i(\mathbf{w}^T\mathbf{x}_i+b)=1-\xi_i$ ，该样本为支持向量
- 若 $\alpha_i<C$，则 $\eta_i>0$，进而有 $\xi_i=0$ ，即该样本恰在最大间隔边界上
- 若 $\alpha_i=C$，则有 $\eta_i=0$，此时若  $\xi_i\leqslant 1$  则该样本落在最大间隔内部，若 $\xi_i>1$ 则该样本被错误分类

由此可看出，软间隔支持向量机的最终模型仅与支持向量有关，即通过采用hinge 损失函数仍保持了稀疏性。

## 核方法

现实任务中，原始样本空间内也许并不存在一个能正确划分两类样本的超平面。对这样的问题，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分。如下图，异或问题就不能线性可分

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/SVM-XOR.png" style="zoom:67%;" />

> 可证明，如果原始空间是有限维， 即属性数有限，那么一定存在一个高维特征空间使样本可分。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/feature_space_mapping.png" style="zoom: 50%;" />

令 $\mathbf{\phi(x)}$ 表示将 $\mathbf x$ 映射后的特征向量，于是，在特征空间中划分超平面所对应的模型可表示为
$$
f_{\mathbf{w},b}(\mathbf{x})=\text{sign}(\mathbf{w}^T\mathbf{\phi(x)}+b)
$$
那么最大化间隔可表示为
$$
\begin{aligned}
\min\limits_{\mathbf w,b}&\frac{1}{2}\|\mathbf w\|^2+C\sum_{i=1}^N\xi_i \\
\text{s.t.}&\quad y_i(\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b)\geqslant 1-\xi_i \\
&\quad \xi_i\geqslant 0, \qquad i=1,2,\cdots,N
\end{aligned}
$$
其对偶问题是
$$
\begin{aligned}
\max\limits_{\mathbf\alpha}&\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\mathbf\phi(\mathbf x_i)^T\mathbf\phi(\mathbf x_j) \\
\text{s.t.}&\quad \sum_{i=1}^N\alpha_iy_i=0 \\
&\quad 0\leqslant\alpha_i\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
求解上述问题涉及到计算 $\mathbf\phi(\mathbf x_i)^T\mathbf\phi(\mathbf x_j)$， 这是样本 $\mathbf x_i$ 与 $\mathbf x_j$ 映射到特征空间之后的内积。由于特征空间维数可能很高，甚至可能是无穷维，因此直接计算  $\mathbf\phi(\mathbf x_i)^T\mathbf\phi(\mathbf x_j)$ 通常是困难的。为了避开这个障碍，引入**核函数**（kernel function） 
$$
K(\mathbf x_1,\mathbf x_2)=\mathbf\phi(\mathbf x_1)^T\mathbf\phi(\mathbf x_2)
$$
即  $\mathbf x_i$ 与 $\mathbf x_j$  在特征空间的内积等于它们在原始样本空间中通过核函数计算的结果，这称为**核技巧**（kernel trick）。核函数 $K$ 的实现方法通常有比直接构建 $\mathbf\phi(\mathbf x)$ 再算点积高效很多。

于是，对偶问题可重写为
$$
\begin{aligned}
\max\limits_{\mathbf\alpha}&\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j K(\mathbf x_i,\mathbf x_j) \\
\text{s.t.}&\quad \sum_{i=1}^N\alpha_iy_i=0 \\
&\quad 0\leqslant\alpha_i\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
可写为矩阵形式
$$
\begin{aligned}
\min\limits_{\mathbf\alpha}&\frac{1}{2}\mathbf\alpha^T\mathbf Q\mathbf\alpha-\mathbf e^T\mathbf\alpha \\
\text{s.t.}&\quad \mathbf y^T\mathbf\alpha=0 \\
&\quad 0\leqslant\alpha_i\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
其中，$\mathbf e$ 是一个全1的 $N$ 维向量，$\mathbf Q$ 是一个 $N\times N$ 的半正定矩阵，$Q_{ij}=y_iy_j K(\mathbf x_i,\mathbf x_j)$。

求解后即可得到
$$
\begin{aligned}
f_{\mathbf{w},b}(\mathbf{x})&=\text{sign}(\mathbf{w}^T\mathbf{\phi(x)}+b) \\
&=\text{sign}(\sum_{i=1}^N\alpha_iy_iK(\mathbf x_i,\mathbf x)+b)
\end{aligned}
$$
通过前面的讨论可知，我们希望样本在特征空间内线性可分，因此特征空间的好坏对支持向量机的性能至关重要。需注意的是，在不知道特征映射的形式时，我们并不知道什么样的核函数是合适的，而核函数也仅是隐式地定义了
这个特征空间。于是，"核函数选择"成为支持向量机的最大变数，若核函数选择不合适，则意味着将样本映射到了一个不合适的特征空间，很可能导致性能不佳。

下面介绍几种常用的核函数

(1) 线性核函数（linear kernel function）
$$
K(\mathbf x_1,\mathbf x_2)=\mathbf x_1^T\mathbf x_2
$$
(2) 多项式核函数（polynomial kernel function）
$$
K(\mathbf x_1,\mathbf x_2)=(\mathbf x_1^T\mathbf x_2+1)^p
$$
(3) 高斯核函数（Gaussian kernel function）：也被称为径向基函数（radial basis function, RBF），是最常用的核函数。$\sigma>0$ 为高斯核的带宽（width）。
$$
K(\mathbf x_1,\mathbf x_2)=\exp(\frac{\|\mathbf x_1-\mathbf x_2\|^2}{2\sigma^2})
$$
(4) 拉普拉斯核函数（Laplace kernel function）
$$
K(\mathbf x_1,\mathbf x_2)=\exp(\frac{\|\mathbf x_1-\mathbf x_2\|}{2\sigma^2})
$$
(5) Sigmoid 核函数（Sigmoid kernel function）
$$
K(\mathbf x_1,\mathbf x_2)=\tanh(\beta\mathbf x_1^T\mathbf x_2+\theta)
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/svm-kernel-iris.png)

## 支持向量回归

支持向量分类方法可以推广到解决回归问题，这种方法称为支持向量回归（Support Vector Regression，SVR）。

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in\R$ 。

**Model**：相比于线性回归用一条线来拟合训练样本， SVR采用一个以 $f(\mathbf x)$ 为中心，宽度为 $2\epsilon$ 的间隔带，来拟合训练样本。预测函数仍为
$$
f_{\mathbf{w},b}(\mathbf{x})=\mathbf{w}^T\mathbf\phi(\mathbf x)+b
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/SVR.svg)

落在带子内的样本不计算损失，不在带子内的样本则以偏离的程度作为损失，然后以最小化损失的方式迫使间隔带从样本最密集的地方（中心地带）穿过，进而达到拟合训练样本的目的。

为适应每个样本，引入松弛变量
$$
\xi_i=\max\{0,|\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b-y_i|-\epsilon\}
$$
并将 $\xi_i\geqslant 0$ 作为惩罚项加入优化，惩罚那些偏离 $\epsilon$ 带的样本，$\xi_i$ 表示样本远离带的程度。

因此SVR的优化问题可以写为
$$
\begin{aligned}
\min\limits_{\mathbf w,b}&\frac{1}{2}\|\mathbf w\|^2+C\sum_{i=1}^N\xi_i \\
\text{s.t.}&\quad -\epsilon-\xi_i \leqslant \mathbf{w}^T\mathbf\phi(\mathbf x_i)+b\leqslant \epsilon+\xi_i \\
&\quad \xi_i\geqslant 0, \qquad i=1,2,\cdots,N
\end{aligned}
$$

> 注释：
>
> - 当样本在带内时，一定满足约束条件，因此代价函数中惩罚项取最小值 $\xi_i=0$
> - 当样本在带外时，为满足样本位置和惩罚项最小值，约束条件则变为 $|\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b|=\epsilon+\xi_i$

这里我们使用了**ϵ-不敏感损失函数**（epsilon-insensitive）, 即小于ϵ的误差被忽略了。
$$
\text{loss}_\epsilon(z)-=\max\{0,|z|-\epsilon\}
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/SVR-loss.svg)

> 在这里不用均方误差的目的是为了和软间隔支持向量机的优化目标保持形式上的一致，这样就可以导出对偶问题引入核函数

如果考虑两边采用不同的松弛程度，可重写为
$$
\begin{aligned}
\min\limits_{\mathbf w,b}&\frac{1}{2}\|\mathbf w\|^2+C\sum_{i=1}^N(\xi_i+\xi_i') \\
\text{s.t.}&\quad -\epsilon-\xi_i' \leqslant \mathbf{w}^T\mathbf\phi(\mathbf x_i)+b\leqslant \epsilon+\xi_i \\
&\quad \xi_i\geqslant 0,\xi_i'\geqslant 0, \qquad i=1,2,\cdots,N
\end{aligned}
$$
对偶问题为
$$
\begin{aligned}
\max\limits_{\mathbf\alpha,\mathbf\alpha'}&\sum_{i=1}^Ny_i(\alpha_i'-\alpha_i)-\epsilon(\alpha_i'+\alpha_i) \\
&-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N(\alpha_i'-\alpha_i)(\alpha_j'-\alpha_j)K(\mathbf x_i,\mathbf x_j) \\
\text{s.t.}&\quad \sum_{i=1}^N(\alpha_i'-\alpha_i)=0 \\
&\quad 0\leqslant\alpha_i,\alpha_i'\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
可写为矩阵形式
$$
\begin{aligned}
\min\limits_{\mathbf\alpha,\mathbf\alpha'}&\frac{1}{2}(\mathbf\alpha-\mathbf\alpha')^T\mathbf Q(\mathbf\alpha-\mathbf\alpha')+\epsilon\mathbf e^T(\mathbf\alpha+\mathbf\alpha')-\mathbf y^T(\mathbf\alpha-\mathbf\alpha') \\
\text{s.t.}&\quad \mathbf e^T(\mathbf\alpha-\mathbf\alpha')=0 \\
&\quad 0\leqslant\alpha_i,\alpha_i'\leqslant C, \quad i=1,2,\cdots,N
\end{aligned}
$$
其中，$\mathbf e$ 是一个全1的 $N$ 维向量，$\mathbf Q$ 是一个 $N\times N$ 的半正定矩阵，$Q_{ij}=K(\mathbf x_i,\mathbf x_j)$。

上述过程中需满足KKT 条件，即要求
$$
\begin{cases}
\alpha_i(\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b-y_i-\epsilon-\xi_i)=0 \\
\alpha_i'(y_i-\mathbf{w}^T\mathbf\phi(\mathbf x_i)-b-\epsilon-\xi_i')=0 \\
\alpha_i\alpha_i'=0 ,\quad \xi_i\xi_i'=0 \\
(C-\alpha_i)\xi_i=0 ,\quad (C-\alpha_i')\xi_i'=0
\end{cases}
$$
可以看出

- 当且仅当 $\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b-y_i-\epsilon-\xi_i=0$ 时，$\alpha_i$ 能取非零值
- 当且仅当 $y_i-\mathbf{w}^T\mathbf\phi(\mathbf x_i)-b-\epsilon-\xi_i'=0$ 时，$\alpha_i'$ 能取非零值

换言之，仅当样本不落入间隔带中，相应的 $\alpha_i$ 和 $\alpha_i'$ 才能取非零值。

- 此外，约束 $\mathbf{w}^T\mathbf\phi(\mathbf x_i)+b-y_i-\epsilon-\xi_i=0$ 和 $y_i-\mathbf{w}^T\mathbf\phi(\mathbf x_i)-b-\epsilon-\xi_i'=0$ 不能同时成立，因此 $\alpha_i$ 和 $\alpha_i'$ 中至少有一个为零。

预测函数为
$$
f_{\mathbf{w},b}(\mathbf{x})=\sum_{i=1}^N(\alpha_i-\alpha_i')K(\mathbf x_i,\mathbf x)+b
$$
能使上式中的 $\alpha_i-\alpha_i'\neq 0$ 的样本即为 SVR 的支持向量，它们必落在 $\epsilon$ 间隔带之外。显然，SVR 的支持向量仅是训练样本的一部分，即其解仍具有稀疏性。

# 附录

## 超平面几何

超平面方程为
$$
\mathbf{w}^T\mathbf{x}+b=0
$$
其中 $\mathbf x=(x_1,x_2,\cdots,x_p)^T$ ，$\mathbf w=(w_1,w_2,\cdots,w_p)^T$ 。

超平面的方程不唯一，同比例缩放 $\mathbf w,b$ ，仍是同一个超平面。若缩放倍数为负数，会改变法向量方向。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/hyperplane.svg)

**法向量**：取超平面上任意两点 $\mathbf x_1,\mathbf x_2$ ，则
$$
\mathbf{w}^T\mathbf{x}_1+b=0 \\
\mathbf{w}^T\mathbf{x}_2+b=0
$$
上面两式相减可得
$$
\mathbf{w}^T(\mathbf{x}_1-\mathbf{x}_2)=\mathbf{w}\cdot(\mathbf{x}_1-\mathbf{x}_2)=0
$$
即向量 $\mathbf w$ 与超平面垂直，所以超平面 $\mathbf{w}^T\mathbf{x}+b=0$ 的法向量为  $\mathbf w$ 。

**任意点到超平面的距离**：取超平面外任意一点 $\mathbf x$ ，在超平面上的投影为 $\mathbf x_1$ ，距离为 $d>0$ ，则
$$
\mathbf{w}\cdot\mathbf{x}_1+b=0 \\
\mathbf{w}\cdot(\mathbf{x}-\mathbf{x}_1)=\|\mathbf w\|\cdot d
$$
因此 $\mathbf x$ 到超平面距离
$$
d=\frac{|\mathbf{w}^T\mathbf{x}+b|}{\|\mathbf w\|}
$$

