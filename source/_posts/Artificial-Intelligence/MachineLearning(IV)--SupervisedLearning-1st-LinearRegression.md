---
title: 机器学习(IV)--监督学习(一)线性回归
date: 2022-11-27 21:40
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-supervised-learning.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 3c1747d9
description: 
---

每个样本都有标签的机器学习称为监督学习。根据标签数值类型的不同，监督学习又可以分为回归问题和分类问题。分类和回归是监督学习的核心问题。

- **回归**(regression)问题中的标签是连续值。
- **分类**(classification)问题中的标签是离散值。分类问题根据其类别数量又可分为二分类（binary classification）和多分类（multi-class classification）问题。

# 线性回归

## 最小二乘法

### 基本形式

给定的数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}
$$
包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T$ 。目标变量 $y_i\in \R$ 。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/linear-regression.svg" alt="线性回归" style="zoom:50%;" />

**Model**：线性模型假设目标变量是特征的线性组合。因此，我们试图拟合函数
$$
f_{\mathbf{w},b}(\mathbf{x})=w_1x_1+w_2x_2+\cdots+w_px_p+b=\sum_{j=1}^p w_jx_j+b
$$
称为多元线性回归 (multiple linear regression)。一般写作向量形式
$$
f_{\mathbf{w},b}(\mathbf{x})=\mathbf{w}^T\mathbf{x}+b
$$
特征向量 $\mathbf x=(x_1,x_2,\cdots,x_p)^T$，参数 $\mathbf{w}=(w_1,w_2,\cdots,w_p)^T$ 称为系数 (coefficients) 或权重 (weights)，标量 $b$ 称为偏置项(bias) 。求得参数 $\mathbf{w},b$ 后，模型就得以确定。$\mathbf w$ 可以直观表达了各特征在预测中的重要性，因此线性模型有很好的可解释性(comprehensibility) 。

为了计算方便，线性回归模型可简写为
$$
f_{\mathbf{w}}(\mathbf{x}) = \mathbf{w}^T\mathbf{x}
$$
其中，特征向量 $\mathbf x=(1,x_1,x_2,\cdots,x_p)^T$，权重向量 $\mathbf{w}=(b,w_1,w_2,\cdots,w_p)^T$

### 最小二乘法

**普通最小二乘法** (ordinary least squares, OLS) 使用残差平方和来估计参数，使得数据的实际观测值和模型预测值尽可能接近。

> Tip: 在古代汉语中平方称为二乘。

**loss function** ：衡量单个样本预测值 $f_{\mathbf{w}}(\mathbf{x})$ 和真实值 $y$ 之间差异的
$$
L(y,f_{\mathbf{w}}(\mathbf{x}))=(f_{\mathbf{w}}(\mathbf{x})-y)^2
$$
**cost function** ：衡量样本集的差异 
$$
J(\mathbf{w}) = \frac{1}{2N} \sum\limits_{i=1}^{N} \left(\mathbf{w}^T\mathbf{x}_i - y_i\right)^2
$$
为了建立一个不会因训练集变大而变大的代价函数，我们计算均方误差而不是平方误差。额外的 1/2 是为了让后面的计算更简洁些。矩阵形式为
$$
J(\mathbf{w})=\cfrac{1}{2N}\|\mathbf{Xw-y}\|_2^2=\cfrac{1}{2N}(\mathbf{Xw-y})^T(\mathbf{Xw-y})
$$
其中，$\mathbf X$ 称为**设计矩阵**（design matrix）
$$
\mathbf{X}=\begin{pmatrix}
1&x_{11}&x_{12}&\cdots&x_{1p} \\
1&x_{21}&x_{22}&\cdots&x_{2p} \\
\vdots&\vdots&\ddots&\vdots \\
1&x_{N1}&x_{N2}&\cdots&x_{Np} \\
\end{pmatrix},
\quad \mathbf{w}=\begin{pmatrix}w_0\\ w_1\\ \vdots\\w_p\end{pmatrix},
\quad \mathbf{y}=\begin{pmatrix}y_1\\ y_2\\ \vdots\\y_N\end{pmatrix}
$$
最后，模型参数估计等价于求解
$$
\arg\min\limits_{\mathbf w} J(\mathbf{w})
$$

**参数估计** ：(parameter estimation) 由于 $J(\mathbf w)$ 为凸函数，且 Hessian 矩阵 $\nabla^2 J(\mathbf w)$ 正定。可以使用凸优化方法求解
$$
\nabla J(\mathbf w)=\frac{\partial J}{\partial\mathbf w}=2\mathbf X^T(\mathbf{Xw-y})=\mathbf 0
$$
此方程被称为**正规方程**（normal equation）。如果 $\mathbf X^T \mathbf X$ 为满秩矩阵(full-rank matrix)或正定矩阵(positive definite matrix)，则可求得最优解
$$
\mathbf w^*=(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y
$$

**最小二乘法的特点** 

现实任务中 $\mathbf X^T\mathbf X$ 可能不可逆，原因如下

- 特征之间可能线性相关
- 特征数量大于样本总数 ($p>N$)

最小二乘法的优点

- 求解参数不需要迭代计算
- 不需要调试超参数

最小二乘法的缺点

- 仅适用于线性回归，无法推广到其他学习算法
- 假设 $N\geqslant p$ ，这个算法的复杂度为 $O(Np^2)$
- 如果样本特征的数量太大 (>10k)，模型将执行的非常慢

### 极大似然估计

概率学的角度来看，目标变量可看作随机变量。对于线性回归来说，通常假设其服从正态分布
$$
y \sim  N(\mathbf{w}^T\mathbf{x}, σ^2)
$$
即随机误差服从均值为0的正态分布 $e\sim N(0,σ^2)$ ，且样本间的误差相互独立 $\mathrm{cov}(e_i,e_j)=0$。

目标变量的概率密度函数为
$$
\mathbb P(y)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y-\mathbf{w}^T\mathbf{x})^2}{2\sigma^2})
$$
**极大似然估计**：(maximum likelihood estimate, MLE) 使得样本误差的联合概率（也称似然函数）取得最大值。为求解方便，对样本联合概率取对数似然函数
$$
\begin{aligned}
\ln L(\mathbf w) &=\ln\prod_{i=1}^N\mathbb P(y_i)=\sum_{i=1}^N\ln\mathbb P(y_i) \\
&=\sum_{i=1}^N\ln\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(\mathbf{w}^T\mathbf{x}_i-y_i)^2}{2\sigma^2})  \\
&=N\ln\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{2\sigma^2}\sum_{i=1}^N(\mathbf{w}^T\mathbf{x}_i-y_i)^2 \\
\end{aligned}
$$
最后，最大化对数似然函数等价于求解
$$
\arg\max\limits_{\mathbf w} \ln L(\mathbf{w})=\arg\min\limits_{\mathbf w} \sum_{i=1}^N(\mathbf{w}^T\mathbf{x}_i-y_i)^2
$$

上式与最小二乘法等价。

## 岭回归和LASSO

### 岭回归

**Ridge** (岭回归) 通过引入 $\ell_2$ 范数正则化(regularization) 项来解决普通最小二乘的过拟合问题

Cost function
$$
J(\mathbf{w})=\cfrac{1}{2N}\|\mathbf{Xw-y}\|_2^2+ \alpha \|\mathbf w\|^2
$$
其中，正则化参数 $\alpha>0$ 通过缩小特征权重来控制模型复杂度，值越大，收缩量越大，这样，系数对共线性的鲁棒性就更强了。

最小化代价函数可求得解析解
$$
\mathbf{w=(X^T X}+\alpha \mathbf{I)^{-1}X^T y}
$$
其中 $\mathbf I$ 是 $p+1$ 维单位阵。利用$\ell_2$ 范数进行正则化不仅可以抑制过拟合，同时叶避免了 $\mathbf{X^T X}$ 不可逆的问题。

### LASSO

**Lasso** (Least Absolute Shrinkage and Selection Operator) 是一个估计稀疏系数的线性模型。它在某些情况下是有用的，因为它倾向于给出非零系数较少的解，从而有效地减少了给定解所依赖的特征数。 它由一个带有 $\ell_1$ 范数正则项的线性模型组成。

Cost function
$$
J(\mathbf{w})=\cfrac{1}{2N}\left(\|\mathbf{Xw-y}\|_2^2+ \alpha \|\mathbf w\|_1\right)
$$
Lasso 中一般采用坐标下降法来实现参数估计。由于Lasso回归产生稀疏模型，因此也可以用来进行特征选择。

### 弹性网

**Elastic-Net** 是一个训练时同时用 $\ell_1$ 和  $\ell_2$ 范数进行正则化的线性回归模型。这种组合允许学习稀疏模型，其中很少有权重是非零类。当多个特征存在相关时，弹性网是很有用的。Lasso很可能随机挑选其中之一，而弹性网则可能兼而有之。在这种情况下，要最小化的目标函数

Cost function
$$
J(\mathbf{w})=\cfrac{1}{2N}\left(\|\mathbf{Xw-y}\|_2^2+ \alpha\rho \|\mathbf w\|_1+ \frac{\alpha(1-\rho)}{2} \|\mathbf w\|_2^2\right)
$$

Elastic-Net 使用坐标下降法来估计参数。

## 贝叶斯线性回归

贝叶斯线性回归（Bayesian linear regression）是使用统计学中贝叶斯推断（Bayesian inference）方法求解的线性回归模型。线性模型的权重系数视为随机变量，并通过先验分布计算其后验分布。

对于贝叶斯线性回归来说，目标变量 $y$ 看作随机变量，服从正态分布
$$
y =\mathbf{w}^T\mathbf{x}+e
$$
通常假设随机误差服从均值为0的正态分布 $e\sim N(0,σ^2)$ ，且样本间独立同分布。对于数据集 $D=(\mathbf X,\mathbf y)$ ，可记为服从多元正态分布
$$
\mathbf y \sim N(\mathbf{Xw},σ^2\mathbf I)
$$
其中 $\mathbf I$ 为单位阵。概率密度函数为
$$
p(\mathbf y|\mathbf X,\mathbf w)=(\frac{1}{\sqrt{2\pi\sigma^2}})^N\exp\left(-\frac{(\mathbf y-\mathbf{Xw})^T(\mathbf y-\mathbf{Xw})}{2\sigma^2}\right)
$$
由贝叶斯定理可推出，权重系数 $\mathbf w$ 的后验概率分布为
$$
p(\mathbf w|\mathbf X,\mathbf y)=\frac{p(\mathbf w)p(\mathbf y|\mathbf X,\mathbf w)}{p(\mathbf y|\mathbf X)}
$$
根据似然和先验的类型，可用于求解贝叶斯线性回归的方法有三类，即最大后验估计（MAP）、贝叶斯估计（共轭先验）和数值方法。

最大后验估计认为最优参数为后验概率最大的参数。由于$p(\mathbf y|\mathbf X)$与$\mathbf w$ 无关，等价于求解
$$
\mathbf w_{MAP}=\arg\max_{\mathbf w} p(\mathbf w)p(\mathbf y|\mathbf X,\mathbf w)
$$
贝叶斯估计使用共轭先验求解后验概率分布 $p(\mathbf w|\mathbf X,\mathbf y)$，通常使用期望$\mathbb E(\mathbf w)$ 作为估计值。

(1) 引入权重向量的正态先验分布 $\mathbf w\sim N(0,\sigma^2_w\mathbf I)$，即均值为零独立同分布。
$$
p(\mathbf w)=\frac{1}{\sqrt{2\pi\sigma_w^2}}\exp(-\frac{\mathbf w^T\mathbf w}{2\sigma_w^2})
$$
可使用最大后验估计。为求解方便，使用对数推导
$$
\begin{aligned}
\mathbf w_{MAP}&=\arg\max_{\mathbf w}\ln p(\mathbf y|\mathbf X,\mathbf w)+\ln p(\mathbf w) \\
&=\arg\max_{\mathbf w}-\frac{1}{2\sigma^2}\|\mathbf{Xw}-\mathbf y\|_2^2-\frac{\mathbf w^T\mathbf w}{2\sigma_w^2} \\
&=\arg\min_{\mathbf w} \|\mathbf{Xw-y}\|_2^2+ \frac{\sigma^2}{\sigma_w^2} \|\mathbf w\|_2^2
\end{aligned}
$$
在估计过程中引入了 $\ell_2$ 范数正则化项（先验分布），等价于岭回归。

(2) 引入权重向量的Laplace先验分布 $w_j\sim Laplace(0,b)$，即均值为零独立同分布。
$$
p(w_j)=\frac{1}{2b}\exp(-\frac{|w_j|}{b})
$$
使用最大后验估计。为求解方便，使用对数推导
$$
\begin{aligned}
\mathbf w_{MAP}&=\arg\max_{\mathbf w}\ln p(\mathbf y|\mathbf X,\mathbf w)+\ln p(\mathbf w) \\
&=\arg\max_{\mathbf w}-\frac{1}{2\sigma^2}\|\mathbf y-\mathbf{Xw}\|_2^2-\sum_{j=1}^p\frac{|w_j|}{b} \\
&=\arg\min_{\mathbf w} \|\mathbf{Xw-y}\|_2^2+ \frac{2\sigma^2}{b} \|\mathbf w\|_1
\end{aligned}
$$
在估计过程中引入了 $\ell_2$ 范数正则化项（先验分布），等价于LASSO。

(3) 引入权重向量的共轭先验 $\mathbf w\sim N(0,\Sigma_w)$ ，其中 $\Sigma_w^{-1}=\text{diag}(\lambda_1,\lambda_2,\cdots,\lambda_p)$ ，即权重系数独立分布，方差不同。此时称为**自关联判定**（Automatic Relevance Determination，ARD）回归。
$$
p(\mathbf w)=\frac{1}{\sqrt{(2\pi)^p\det \Sigma_w}}\exp\left(-\frac{1}{2}\mathbf w^T\Sigma_w^{-1}\mathbf w\right)
$$
则权重向量的联合概率
$$
\begin{aligned}
p(\mathbf w)p(\mathbf y|\mathbf X,\mathbf w) 
\propto & \exp(-\frac{1}{2}\mathbf w^T\Sigma_w^{-1}\mathbf w)\exp(-\frac{1}{2\sigma^2}(\mathbf{Xw}-\mathbf y)^T(\mathbf{Xw}-\mathbf y)) \\
\propto&\exp\left(-\frac{1}{2}(\mathbf w^T(\Sigma_w^{-1}+\sigma^{-2}\mathbf X^T\mathbf X)\mathbf w-2\sigma^{-2}\mathbf y^T\mathbf{Xw})\right) \\
\propto&\exp\left(-\frac{1}{2}(\mathbf w-\Lambda^{-1}\mathbf X^T\mathbf y)^T(\sigma^{-2}\Lambda)(\mathbf w-\Lambda^{-1}\mathbf X^T\mathbf y)\right)
\end{aligned}
$$

其中 $\Lambda=\mathbf X^T\mathbf X+\sigma^2\Sigma_w^{-1}$。于是得到$\mathbf w$的后验分布为正态分布
$$
\mathbf w|\mathbf X,\mathbf y\sim N(\Lambda^{-1}\mathbf X^T\mathbf y,\sigma^2\Lambda^{-1})
$$
权重系数的贝叶斯估计为
$$
\hat{\mathbf w}=(\mathbf X^T\mathbf X+\sigma^2\Sigma_w^{-1})^{-1}\mathbf X^T\mathbf y
$$
(4) 引入一般的共轭先验 $\mathbf w\sim N(\mu_w,\Sigma_w)$。
$$
p(\mathbf w)=\frac{1}{\sqrt{(2\pi)^p\det \Sigma_w}}\exp\left(-\frac{1}{2}(\mathbf w-\mu_w)^T\Sigma_w^{-1}(\mathbf w-\mu_w)\right)
$$
则权重向量的联合概率
$$
\begin{aligned}
p(\mathbf w)p(\mathbf y|\mathbf X,\mathbf w) 
\propto & \exp(-\frac{1}{2}(\mathbf w-\mu_w)^T\Sigma_w^{-1}(\mathbf w-\mu_w))\exp(-\frac{1}{2\sigma^2}(\mathbf{Xw}-\mathbf y)^T(\mathbf{Xw}-\mathbf y)) \\
\propto&\exp\left(-\frac{1}{2}(\mathbf w-\Lambda^{-1}\mathbf u)^T(\sigma^{-2}\Lambda)(\mathbf w-\Lambda^{-1}\mathbf u)\right)
\end{aligned}
$$

其中 $\Lambda=\mathbf X^T\mathbf X+\sigma^2\Sigma_w^{-1},\quad \mathbf u=\mathbf X^T\mathbf y+\sigma^2\Sigma_w^{-1}\mu_w$。于是得到$\mathbf w$的后验分布为正态分布
$$
\mathbf w|\mathbf X,\mathbf y\sim N(\Lambda^{-1}\mathbf u,\sigma^2\Lambda^{-1})
$$
权重系数的贝叶斯估计为
$$
\hat{\mathbf w}=(\mathbf X^T\mathbf X+\sigma^2\Sigma_w^{-1})^{-1}(\mathbf X^T\mathbf y+\sigma^2\Sigma_w^{-1}\mu_w)
$$

## 广义线性回归

线性模型往往不能很好地拟合数据，我们可以在线性方程后面引入一个非线性变换，拟合许多功能更为强大的非线性模型(non-linear model)。

例如，对数线性回归 (log-linear regression)
$$
\ln f_{\mathbf{w},b}(\mathbf{x})=\mathbf{w}^T\mathbf{x}+b
$$
这个非线性变换称为激活函数(activation function)。更一般地，考虑激活函数 $y=g(z)$，令 
$$
f_{\mathbf{w},b}(\mathbf{x})=g(\mathbf{w}^T\mathbf{x}+b)
$$
这样得到的模型称为广义线性模型 (Generalized Linear Models, GLM)，激活函数的反函数 $z=g^{-1}(y) $​ 称为联系函数 (link function)。广义线性模型的参数估计常通过加权最小二乘法或极大似然估计。

样条回归：分段式多项式回归。
https://zhuanlan.zhihu.com/p/34825299?utm_id=0
https://juejin.cn/post/7118600337660837925

## 多项式回归

为了更好的拟合数据，机器学习中一个常见模式是使用非线性函数对数据进行变换来创建新的特征。例如，可以通过构造**多项式特征**(**polynomial features**)来扩展简单的线性回归。在标准线性回归的情况下，您可能拟合一个二维特征的模型
$$
f_{\mathbf w}(\mathbf x)=w_0+w_1x_1+w_2x_2
$$
如果我们想用抛物面来拟合数据而不是平面，我们可以用二阶多项式组合特征，这样模型看起来就像这样
$$
f_{\mathbf w}(\mathbf x)=w_0+w_1x_1+w_2x_2+w_3x_1x_2+w_4x_1^2+w_5x_2^2
$$
其实，得到的多项式回归依旧是线性模型：只需引入新的特征向量进行转换
$$
\begin{matrix}
\text{from}&x_1&x_2&x_1x_2&x_1^2&x_2^2 \\
\to&z_1&z_2&z_3&z_4&z_5
\end{matrix}
$$
下面是一个应用于一维数据的例子，使用了不同程度的多项式特征：

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/polynomial-features.svg" style="zoom: 67%;" />

## 多标签回归

包含多个目标变量的回归任务称为 **Multi-output regression**

**Multi-task Lasso** 是一个估计多任务的稀疏系数的线性模型， $\mathbf Y$ 是一个  $N\times N_{tasks}$ 矩阵。约束条件是，对于所有回归问题（也叫任务），所选的特征是相同的。它混合使用 $\ell_1\ell_2$ 范数作为正则化项。

Cost function
$$
J(\mathbf{W})=\cfrac{1}{2N}\left(\|\mathbf{XW-Y}\|^2_{Fro}+ \alpha \|\mathbf W\|_{21}\right)
$$
其中 Fro 表示Frobenius范数
$$
\mathbf \|\mathbf A\|_{Fro}=\sqrt{\sum_{ij}a^2_{ij}}=\sqrt{\text{tr}(\mathbf A^T\mathbf A)}
$$
混合 $\ell_1\ell_2$ 范数
$$
\mathbf \|\mathbf A\|_{21}=\sum_i\sqrt{\sum_{j}a^2_{ij}}
$$
Multi-task Lasso 也采用坐标下降法来估计参数。

**Multi-task Elastic-Net** 是一个估计多任务的稀疏系数的线性模型， $\mathbf Y$ 是一个  $N\times N_{tasks}$ 矩阵。约束条件是，对于所有回归问题（也叫任务），所选的特征是相同的。它使用混合的 $\ell_1\ell_2$ 范数和$\ell_2$作为正则化项。

Cost function
$$
J(\mathbf{W})=\cfrac{1}{2N}\left(\|\mathbf{XW-Y}\|^2_{Fro}+ \alpha\rho \|\mathbf W\|_{21}+ \frac{\alpha(1-\rho)}{2} \|\mathbf W\|_{Fro}^2\right)
$$
Multi-task Elastic-Net 也采用坐标下降法来估计参数。

## 泊松回归

泊松回归：预测一个代表频数的响应变量

## Cox 回归

Cox回归的因变量就有些特殊，它不仅考虑结果而且考虑结果出现时间的回归模型。它用一个或多个自变量预测一个事件（死亡、失败或旧病复发）发生的时间。Cox回归的主要作用发现风险因素并用于探讨风险因素的强弱。但它的因变量必须同时有2个，一个代表状态，必须是分类变量，一个代表时间，应该是连续变量。只有同时具有这两个变量，才能用Cox回归分析。
