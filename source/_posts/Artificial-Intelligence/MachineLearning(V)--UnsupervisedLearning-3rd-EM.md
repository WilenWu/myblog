---
title: 机器学习(V)--无监督学习(三)EM算法
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
  - 无监督学习
cover: /img/ML-unsupervised-learning.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 4f81b9fa
date: 2024-07-09 20:47:30
description:
katex: true
---

# EM算法


## 极大似然估计

**极大似然估计**：(maximum likelihood estimate, MLE)  是一种常用的模型参数估计方法。它假设观测样本出现的概率最大，也即样本联合概率（也称似然函数）取得最大值。

为求解方便，对样本联合概率取对数似然函数
$$
\log L(\theta) =\log\mathbb P(X|\theta)=\sum_{i=1}^N\log \mathbb P(\mathbf x_i|\theta)
$$
优化目标是最大化对数似然函数
$$
\hat\theta=\arg\max_{\theta}\sum_{i=1}^N\log \mathbb P(\mathbf x_i|\theta)
$$

假设瓜田里有两种类型的西瓜🍉，瓜农随机抽取了10个西瓜，来了解西瓜的重量分布 $p(x|\theta)$，记录结果如下：

| 变量         | 样本                                              |
| :----------- | :------------------------------------------------ |
| 西瓜重量 $x$ | 5.3 , 5.7, 4.7, 4.3, 3.2, 4.9, 4.1, 3.5, 3.8, 1.7 |
| 西瓜品种 $z$ | 1, 1, 1, 1, 2, 2, 2, 2, 2, 2                      |

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GMM_example.png"  />

其中，西瓜的品种 $z$ 是离散分布 $\mathbb P(z=k)=\pi_k$，一般假设两种类型的西瓜服从均值和方差不同的高斯分布 $N(\mu_1,\sigma^2_1)$和 $N(\mu_2,\sigma^2_2)$。由全概率公式，西瓜重量的概率密度模型
$$
p(x;\theta)=\pi_1\mathcal N(x;\mu_1,\sigma^2_1)+\pi_2\mathcal N(x;\mu_2,\sigma^2_2)
$$

我们尝试用极大似然估计求解参数$\theta=(\pi_1,\pi_2,\mu_1,\sigma^2_1,\mu_2,\sigma^2_2)$。

优化目标函数
$$
\max_{\theta}\sum_{z_i=1}\log \pi_1\mathcal N(x_i;\mu_1,\sigma_1^2)+\sum_{z_i=2}\log \pi_2\mathcal N(x_i;\mu_2,\sigma_2^2) \\
\text{s.t. } \pi_1+\pi_2=1
$$
使用拉格朗日乘子法容易求得
$$
\pi_1=0.4,\quad \pi_2=0.6 \\
\mu_1=5,\quad \sigma_1^2=0.54^2 \\
\mu_2=3.53,\quad \sigma_2^2=0.98^2 \\
$$

最终得到

$$
p(x)=0.4\times\mathcal N(x;5,0.54^2)+0.6\times\mathcal N(x;3.53,0.98^2)
$$


但是，实际中如果瓜农无法辩识标记西瓜的品种，此时概率分布函数变为
$$
p(x;\theta)=\pi\mathcal N(x;\mu_1,\sigma^2_1)+(1-\pi)\mathcal N(x;\mu_2,\sigma^2_2)
$$

其中品种$z$ 成为隐藏变量。对数似然函数变为
$$
\log L(\theta)=\sum_{i}\log (\pi\mathcal N(x_i;\mu_1,\sigma^2_1)+(1-\pi)\mathcal N(x_i;\mu_2,\sigma^2_2))
$$
其中参数 $\theta=(\pi,\mu_1,\sigma^2_1,\mu_2,\sigma^2_2)$。上式中存在"和的对数"，若直接求导将会变得很麻烦。下节我们将会介绍EM算法来解决此类问题。

## 基本思想
概率模型有时既含有观测变量 (observable variable)，又含有隐变量 (latent variable)。EM（Expectation-Maximization，期望最大算法）是一种迭代算法，用于含有隐变量的概率模型的极大似然估计或极大后验估计，是数据挖掘的十大经典算法之一。

假设现有一批独立同分布的样本
$$
X=\{x_1,x_2,\cdots,x_N\}
$$
它们是由某个含有隐变量的概率分布 $p(x,z|\theta)$ 生成。设样本对应的隐变量数据
$$
Z=\{z_1,z_2,\cdots,z_N\}
$$

对于一个含有隐变量 $Z$ 的概率模型，一般将  $(X,Z)$ 称为完全数据 (complete-data)，而观测数据 $X$ 为不完全数据(incomplete-data)。

假设观测数据 $X$ 概率密度函数是$p(X|\theta)$，其中$\theta$是需要估计的模型参数，现尝试用极大似然估计法估计此概率分布的参数。为了便于讨论，此处假设 $z$ 为连续型随机变量，则对数似然函数为
$$
\log L(\theta)=\sum_{i=1}^N\log p(x_i|\theta)=\sum_{i=1}^N\log\int_{z_i}p(x_i,z_i|\theta)\mathrm dz_i
$$

  > Suppose you have a probability model with parameters $\theta$.
  > $p(x|\theta)$ has two names. It can be called the **probability of $x$** (given $\theta$), or the **likelihood of $\theta$** (given that $x$  was observed).

我们的目标是极大化观测数据 $X$ 关于参数  $\theta$ 的对数似然函数
$$
\hat\theta=\arg\max_{\theta}\log L(\theta)
$$

显然，此时 $\log L(\theta)$ 里含有未知的隐变量 $z$ 以及求和项的对数，相比于不含隐变量的对数似然函数，该似然函数的极大值点较难求解，而 EM 算法则给出了一种迭代的方法来完成对 $\log L(\theta)$ 的极大化。

注意：确定好含隐变量的模型后，即确定了联合概率密度函数 $p(x,z|\theta)$  ，其中$\theta$是需要估计的模型参数。为便于讨论，在此有必要说明下其他已知的概率函数。

联合概率密度函数
$$
p(x,z|\theta)=f(x,z;\theta)
$$
观测变量 $x$ 的概率密度函数
$$
p(x|\theta)=\int_z f(x,z;\theta)\mathrm dz
$$
隐变量 $z$ 的概率密度函数
$$
p(z|\theta)=\int_x f(x,z;\theta)\mathrm dx
$$
条件概率密度函数
$$
p(x|z,\theta)=\frac{p(x,z|\theta)}{p(z|\theta)}=\frac{f(x,z;\theta)}{\int_x f(x,z;\theta)\mathrm dx}
$$
和
$$
p(z|x,\theta)=\frac{p(x,z|\theta)}{p(x|\theta)}=\frac{f(x,z;\theta)}{\int_z f(x,z;\theta)\mathrm dz}
$$
下面给出两种推导方法：一种借助 Jensen 不等式；一种使用 KL 散度。

**首先使用 Jensen 不等式推导**：使用含有隐变量的全概率公式
$$
\begin{aligned}
\log p(x_i|\theta)&=\log\int_{z_i} p(x_i,z_i|\theta)\mathrm dz_i \\
&=\log\int_{z_i}q_i(z_i)\frac{p(x_i,z_i|\theta)}{q_i(z_i)}\mathrm dz_i \\
&=\log\mathbb E_z\left(\frac{p(x_i,z_i|\theta)}{q_i(z_i)}\right) \\
&\geqslant \mathbb E_z\left(\log\frac{p(x_i,z_i|\theta)}{q_i(z_i)}\right) \\
&= \int_{z_i}q_i(z_i) \log\frac{p(x_i,z_i|\theta)}{q_i(z_i)}\mathrm dz_i
\end{aligned}
$$
其中 $q_i(z_i)$ 是引入的第$i$个样本==隐变量$z_i$ 的任意概率密度函数（未知函数）==，其实 $q$ 不管是任意函数，上式都成立。从后续推导得知，当 $q_i(z_i)$ 是 $z_i$ 的概率密度时，方便计算。

所以
$$
\log L(\theta)=\sum_{i=1}^N\log p(x_i|\theta)\geqslant B(q,\theta)=\sum_{i=1}^N\int_{z_i}q_i(z_i) \log\frac{p(x_i,z_i|\theta)}{q_i(z_i)}\mathrm dz_i
$$

其中函数 $B$ 为对数似然的下界函数。下界比较好求，所以我们要优化这个下界来使得似然函数最大。

假设第 $t$ 次迭代时 $\theta$ 的估计值是 $\theta^{(t)}$，我们希望第 $t+1$ 次迭代时的 $\theta$ 能使 $\log L(\theta)$ 增大，即 
$$
\log L(\theta^{(t)}) \leqslant \log L(\theta^{(t+1)})
$$

可以分为两步实现：

- 首先，固定$\theta=\theta^{(t)}$ ，通过调整 $q$ 函数使得 $B(q^{(t)},\theta)$ 在 $\theta^{(t)}$ 处和 $\log L(\theta^{(t)})$ 相等；
  $$
  B(q^{(t)},\theta^{(t)})=\log L(\theta^{(t)})
  $$

- 然后，固定$q$，优化 $\theta^{(t+1)}$ 取到下界函数 $B(q^{(t)},\theta)$ 的最大值。
  $$
  \theta^{(t+1)}=\arg\max_{\theta} B(q^{(t)},\theta)
  $$

所以
$$
\log L(\theta^{(t+1)})\geqslant B(q^{(t)},\theta^{(t+1)})\geqslant B(q^{(t)},\theta^{(t)})=\log L(\theta^{(t)})
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/M-step.png"  />

因此，EM算法也可以看作一种坐标提升算法，首先固定一个值，对另外一个值求极值，不断重复直到收敛。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Coordinate_Descent.svg" style="zoom: 80%;" />

接下来，我们开始求解 $q^{(t)}$ 。Jensen不等式中等号成立的条件是自变量是常数，即
$$
\frac{p(x_i,z_i|\theta)}{q_i(z_i)}=c
$$
由于假设 $q_i(z_i)$是 $z_i$ 的概率密度函数，所以
$$
p(x_i|\theta)=\int_{z_i}p(x_i,z_i|\theta)\mathrm dz_i=\int_{z_i} cq_i(z_i)\mathrm dz_i=c
$$

于是
$$
q_i(z_i)=\frac{p(x_i,z_i|\theta)}{c}=\frac{p(x_i,z_i|\theta)}{p(x_i|\theta)}=p(z_i|x_i,\theta)
$$
可以看到，函数 $q_i(z_i)$ 代表第 $i$ 个数据是 $z_i$ 的概率密度，是可以直接计算的。

最终，我们只要初始化或使用上一步已经固定的 $\theta^{(t)}$，然后计算
$$
\begin{aligned}
\theta^{(t+1)}& = \arg\max_{\theta}\sum_{i=1}^N\int_{z_i}p(z_i|x_i,\theta^{(t)}) \log\frac{p(x_i,z_i|\theta)}{p(z_i|x_i,\theta^{(t)})}\mathrm dz_i \\
& = \arg\max_{\theta}\sum_{i=1}^N\int_{z_i}p(z_i|x_i,\theta^{(t)}) \log p(x_i,z_i|\theta)\mathrm dz_i \\
& = \arg\max_{\theta}\sum_{i=1}^N \mathbb E_{z_i|x_i,\theta^{(t)}}[\log p(x_i,z_i|\theta)] \\
& = \arg\max_{\theta} Q(\theta,\theta^{(t)})
\end{aligned}
$$

**接下来使用 KL 散度推导**：使用含有隐变量的条件概率
$$
\begin{aligned}
\log p(x_i|\theta)&=\log\frac{p(x_i,z_i|\theta)}{p(z_i|x_i,\theta)} \\
&=\int_{z_i}q_i(z_i)\log\frac{p(x_i,z_i|\theta)}{p(z_i|x_i,\theta)}\cdot\frac{q_i(z_i)}{q_i(z_i)}\mathrm dz_i \\
&= \int_{z_i}q_i(z_i) \log\frac{p(x_i,z_i|\theta)}{q_i(z_i)}\mathrm dz_i + \int_{z_i}q_i(z_i) \log\frac{q_i(z_i)}{p(z_i|x_i,\theta)}\mathrm dz_i \\
&=B(q_i,\theta)+KL(q_i\|p_i)
\end{aligned}
$$
同样 $q_i(z_i)$ 是引入的==关于 $z_i$ 的任意概率密度函数（未知函数）==，函数 $B(q_i,\theta)$ 表示对数似然的一个下界，散度 $KL(q_i\|p_i)$描述了下界与对数似然的差距。

同样为了保证
$$
\log L(\theta^{(t)}) \leqslant \log L(\theta^{(t+1)})
$$

分为两步实现：

- 首先，固定$\theta=\theta^{(t)}$ ，通过调整 $q$ 函数使得 $B(q^{(t)},\theta)$ 在 $\theta^{(t)}$ 处和 $\log L(\theta^{(t)})$ 相等，即 $KL(q_i\|p_i)=0$，于是
  $$
  q_i(z_i)=p(z_i|x_i,\theta^{(t)})
  $$

- 然后，固定$q$，优化 $\theta^{(t+1)}$ 取到下界函数 $B(q^{(t)},\theta)$ 的最大值。
  $$
  \theta^{(t+1)}=\arg\max_{\theta} B(q^{(t)},\theta)
  $$

## 算法流程

输入：观测数据 $X$，联合分布 $p(x,z;\theta)$，条件分布$P(z|x,\theta)$

输出：模型参数$\theta$

EM算法通过引入隐含变量，使用极大似然估计（MLE）进行迭代求解参数。每次迭代由两步组成：

- **E-step**：求期望 (expectation)。以参数的初始值或上一次迭代的模型参数 $\theta^{(t)}$ 来计算隐变量后验概率 $p(z_i|x_i,\theta^{(t)})$ ，并计算期望(expectation)
  $$
  Q(\theta,\theta^{(t)})=\sum_{i=1}^N\int_{z_i}p(z_i|x_i,\theta^{(t)}) \log p(x_i,z_i|\theta)\mathrm dz_i
  $$

- **M-step**: 求极大 (maximization)，极大化E步中的期望值，来确定 $t+1$ 次迭代的参数估计值
  $$
  \theta^{(t+1)}=\arg\max_{\theta} Q(\theta,\theta^{(t)})
  $$

依次迭代，直至收敛到局部最优解。

# 高斯混合模型

## 基础模型

高斯混合模型 (Gaussian Mixture Model, GMM) 数据可以看作是从$K$个高斯分布中生成出来的，每个高斯分布称为一个组件 (Component)。

引入隐变量 $z\in\{1,2,\cdots,K\}$，表示对应的样本 $x$ 属于哪一个高斯分布，这个变量是一个离散的随机变量：
$$
\mathbb P(z=k)=\pi_k \\
\text{s.t. } \sum_{k=1}^K\pi_k=1
$$
可将 $\pi_k$ 视为选择第 $k$ 高斯分布的先验概率，而对应的第$k$ 个高斯分布的样本概率
$$
p(x|z=k)=\mathcal N(x;\mu_k,\Sigma_k)
$$

于是高斯混合模型
$$
p_M(x)=\sum_{k=1}^K\pi_k\mathcal N(x;\mu_k,\Sigma_k)
$$

其中 $0\leqslant \pi_k\leqslant 1$为混合系数(mixing coefficients)。

高斯混合模型的参数估计是EM算法的一个重要应用，隐马尔科夫模型的非监督学习也是EM算法的一个重要应用。

## EM算法

高斯混合模型的极大似然估计
$$
\hat\theta=\arg\max_{\theta} \sum_{i=1}^N\log\sum_{k=1}^K\pi_k \mathcal N(x_i;\mu_k,\Sigma_k)
$$
其中参数 $\theta_k=(\pi_k,\mu_k,\Sigma_k)$，使用EM算法估计GMM的参数$\theta$。

**依照当前模型参数，计算隐变量后验概率**：由贝叶斯公式知道
$$
\begin{aligned}
P(z_i=k|x_i)&=\frac{P(z_i=k)p(x_i|z_i=k)}{p(x_i)} \\
&=\frac{\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)}{\sum_{k=1}^K\pi_k\mathcal N(x_i;\mu_k,\Sigma_k) } \\
&=\gamma_{ik}
\end{aligned}
$$

令 $\gamma_{ik}$表示第$i$个样本属于第$k$个高斯分布的概率。

**E-step：确定Q函数**

$$
\begin{aligned}
Q(\theta,\theta^{(t)})&=\sum_{i=1}^N\sum_{k=1}^Kp(z_i=k|x_i,\mu^{(t)},\Sigma^{(t)}) \log p(x_i,z_i=k|\mu,\Sigma)  \\
&=\sum_{i=1}^N\sum_{k=1}^K\gamma_{ik}\log\pi_k\mathcal N(x;\mu_k,\Sigma_k) \\
&=\sum_{i=1}^N\sum_{k=1}^K\gamma_{ik}(\log\pi_k+ \log\mathcal N(x;\mu_k,\Sigma_k) )
\end{aligned}
$$

**M-step：求Q函数的极大值**

上面已获得的$Q(\theta,\theta^{(t)})$分别对 $\mu_k,\Sigma_k$求导并设为0。得到
$$
\mu_k^{(t+1)}=\frac{\sum_{i=1}^N\gamma_{ik}x_i}{\sum_{i=1}^N\gamma_{ik}} \\
\Sigma_k^{(t+1)}=\frac{\sum_{i=1}^N\gamma_{ik}(x_i-\mu_k^{(t+1)}) (x_i-\mu_k^{(t+1)})^T }{\sum_{i=1}^N\gamma_{ik}}
$$

 可以看到第$k$个高斯分布的$\mu_k,\Sigma_k$ 是所有样本的加权平均，其中每个样本的权重为该样本属于第$k$个高斯分布的后验概率 $\gamma_{ik}$。

对于混合系数 $\pi_k$，因为有限制条件，使用拉格朗日乘子法可求得
$$
\pi_k^{(t+1)}=\frac{1}{N}\sum_{i=1}^N\gamma_{ik}
$$

即第$k$个高斯分布的混合系数是属于$k$的样本的平均后验概率，由此运用EM算法能大大简化高斯混合模型的参数估计过程，在中间步只需计算$\gamma_{ik}$就行了。

高斯混合模型的算法流程如下图所示：

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/GMM_algorithm.png" alt="GMM_algorithm" style="zoom:50%;" />

## 高斯混合聚类

高斯混合聚类假设每个类簇中的样本都服从一个多维高斯分布，那么空间中的样本可以看作由$K$个多维高斯分布混合而成。

引入隐变量$z$ 标记簇类别，这样就可以使用高斯混合模型
$$
p_M(x)=\sum_{k=1}^K\pi_k\mathcal N(x;\mu_k,\Sigma_k) 
$$

使用EM算法迭代求解。

相比于K-means更具一般性，能形成各种不同大小和形状的簇。K-means可视为高斯混合聚类中每个样本仅指派给一个混合成分的特例 
$$
\gamma_{ik}=\begin{cases}
1, & \text{if } k=\arg\min_k\|x_i-\mu_k\|^2\\
0, & \text{otherwise}
\end{cases}
$$
且各混合成分协方差相等，均为对角矩阵 $\sigma^2 I$。

# 附录

## Jensen 不等式

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Jensen_inequality_0.svg" style="zoom:80%;" />

若 $f$ 是凸函数(convex function)，对任意的  $\lambda\in [0,1]$，下式恒成立
$$
f(\lambda x_1+(1-\lambda)x_2)\leqslant \lambda f(x_1)+(1-\lambda)f(x_2)
$$
**Jensen's inequality**就是上式的推广，设 $f(x)$ 为凸函数，$\lambda_i\in[0,1],\ \sum_i\lambda_i=1$ ，则
$$
f(\sum_i\lambda_ix_i)\leqslant \sum_i\lambda_if(x_i)
$$
若将 $\lambda_i$ 视为一个概率分布，则可表示为期望值的形式
$$
f(\mathbb E[x])\leqslant\mathbb E[f(x)]
$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Jensen_inequality.png"  />

显然，如果 $f$ 是凹函数(concave function)，则将不等号反向。

