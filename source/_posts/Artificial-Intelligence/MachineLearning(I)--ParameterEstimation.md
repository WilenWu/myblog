---
title: 机器学习(I)--参数估计
date: 2023-05-02 14:36:00
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-overview.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 8006d3d0
description:
---

# 参数估计

事实上，概率模型的训练过程就是参数估计（parameter estimation）过程。对于参数估计，统计学界的两个学派分别提供了不同的解决方案：

- 频率主义学派（Frequentist）认为模型参数虽然未知，但却是客观存在的固定常数。因此，可通过优化似然函数等准则来确定参数值；
- 贝叶斯学派（Bayesian）则认为模型参数是未观察到的随机变量，并且服从某种先验分布。可基于观测到的数据来计算参数的后验分布。

对于一个未知参数的分布我们往往可以采用生成一批观测数据，通过这批观测数据做参数估计的做法来估计参数。最常用的有最大似然估计、矩估计、最大后验估计、贝叶斯估计等。

## 极大似然估计

### 基本形式

**极大似然估计**（Maximum Likelihood Estimate, MLE)  是根据数据采样来估计概率分布参数的经典方法。MLE认为当前发生的事件是概率最大的事件。因此就可以给定的数据集，使得该数据集发生的概率最大来求得模型中的参数。

给定包含 $N$ 个样本的数据集  
$$
D=\{\mathbf x_1,\mathbf x_2,\cdots,\mathbf x_N\}
$$
假设这些样本是独立同分布的，分布函数为 $\mathbb P(\mathbf x;\theta)$。样本的联合概率，即似然函数（likelihood）
$$
L(\theta;D)=\mathbb P(D|\theta)=\prod_{i=1}^N \mathbb P(\mathbf x_i|\theta)
$$
MLE认为参数是一个常数，希望在 $\theta$ 的所有可能的取值中，找出最大化产生观测数据的参数。似然函数中的连乘操作易造成下溢，通常使用对数似然（log-likelihood）
$$
\log L(\theta;D)=\sum_{i=1}^N \log\mathbb P(\mathbf x_i|\theta)
$$
此时参数 $\theta$ 的极大似然估计为
$$
\theta_{MLE}=\arg\max_{\theta}\sum_{i=1}^N \log\mathbb P(\mathbf x_i|\theta)
$$

### KL散度

**KL 散度**：极大似然估计也可看作最小化数据集上的经验分布 $\mathbb P(\mathbf x|\hat\theta)$和实际分布间 $\mathbb P(\mathbf x|\theta)$的差异。两者之间的差异程度可以通过 KL 散度度量
$$
\begin{aligned}
KL(P\|\hat P)&=\sum_{i=1}^N \mathbb P(\mathbf x_i|\theta)\log\frac{\mathbb P(\mathbf x_i|\theta)}{\mathbb P(\mathbf x_i|\hat\theta)} \\
&=\sum_{i=1}^N \mathbb P(\mathbf x_i|\theta)[\log\mathbb P(\mathbf x_i|\theta)-\log\mathbb P(\mathbf x_i|\hat\theta)]
\end{aligned}
$$
由于实际概率分布 $\mathbb P(\mathbf x|\theta)$ 是个确定值。于是最小化 KL 散度
$$
\begin{aligned}
\hat\theta &=\arg\min_{\theta} KL(P\|\hat P) \\
&=\arg\min_{\theta}\sum_{i=1}^N-\log\mathbb P(\mathbf x_i|\theta) \\
&=\arg\max_{\theta}\sum_{i=1}^N\log\mathbb P(\mathbf x_i|\theta)
\end{aligned}
$$
这等价于极大似然估计。

### 正态分布

假设 $\mathbf x$ 为连续特征，服从正态分布。概率密度函数
$$
\mathcal N(\mathbf x;\mu,\mathbf\Sigma)=\frac{1}{\sqrt{(2\pi)^p\det\mathbf\Sigma}}\exp\left(-\frac{1}{2}(\mathbf x-\mathbf\mu)^T\mathbf\Sigma^{-1}(\mathbf x-\mathbf\mu)\right)
$$
则参数的对数似然函数
$$
\begin{aligned}
\ln L(\mu,\mathbf\Sigma)&=\sum_{i=1}^N \ln\mathcal N(\mathbf x_i;\mu,\Sigma) \\
&=-\frac{1}{2}\sum_{i=1}^N(\mathbf x_i-\mathbf\mu)^T\mathbf\Sigma^{-1}(\mathbf x_i-\mathbf\mu)-\frac{1}{2}\ln(2\pi)^p\det\mathbf\Sigma
\end{aligned}
$$
首先对 $\mu$ 求导，并取值为零
$$
\frac{\partial \ln L(\mu,\mathbf\Sigma)}{\partial\mu}=\sum_{i=1}^N\mathbf\Sigma^{-1}(\mathbf x_i-\mathbf\mu)=\mathbf\Sigma^{-1}(\sum_{i=1}^N\mathbf x_i-N\mathbf\mu)=0
$$
再对 $\mathbf\Sigma$ 求导，并取值为零
$$
\frac{\partial \ln L(\mu,\mathbf\Sigma)}{\partial\mathbf\Sigma}=\frac{1}{2}\sum_{i=1}^N\mathbf\Sigma^{-1}(\mathbf x_i-\mathbf\mu)(\mathbf x_i-\mathbf\mu)^T\mathbf\Sigma^{-1}-\frac{1}{2}N\mathbf\Sigma^{-1}=0
$$
则参数 $\mu$ 和 $\mathbf\Sigma$ 的极大似然估计为
$$
\hat\mu=\bar{\mathbf x}=\frac{1}{N}\sum_{i=1}^N\mathbf x_i \\
\hat\Sigma=\frac{1}{N}\sum_{i=1}^N(\mathbf x_i-\bar{\mathbf x})(\mathbf x_i-\bar{\mathbf x})^T
$$
也就是说，通过极大似然法得到的正态分布均值和方差是一个符合直觉的结果。对于离散特征，也可通过类似的方式估计。

### 伯努利分布

假设二分类特征 $x\in\{0,1\}$，服从伯努利分布
$$
\mathbb P(x|\theta)=\theta^x(1-\theta)^{1-x}=\begin{cases}
\theta, &\text{if }x=1 \\
1-\theta, &\text{if }x=0
\end{cases}
$$
则参数 $\theta$ 的对数似然函数
$$
\begin{aligned}
\log L(\theta)&=\sum_{i=1}^N \log \theta^{x_i}(1-\theta)^{1-x_i} \\
&=\sum_{i=1}^N x_i\log \theta+\sum_{i=1}^N(1-x_i)\log(1-\theta)
\end{aligned}
$$
对数似然函数求导，并取值为零
$$
\frac{\partial \log L(\theta)}{\partial\theta}=\frac{1}{\theta}\sum_{i=1}^Nx_i-\frac{1}{1-\theta}\sum_{i=1}^N(1-x_i)=0
$$
则参数 $\theta$ 的极大似然估计为
$$
\hat\theta=\frac{1}{N}\sum_{i=1}^Nx_i=\bar x
$$
即为 $x=1$ 的频率。

### 离散特征分布率

假离散特征有 $K$ 个可能值 $x\in\{c_1,c_1,\cdots,c_K\}$，分布率为
$$
\mathbb P(x=c_k|\theta)=\theta_k,\quad k=1,2,\cdots,K \\
\text{s.t. }\sum_{k=1}^K\theta_k=1
$$
假设 $x=c_k$ 出现的次数为 $N_k$，即$\sum_{k=1}^KN_k=N$。则参数向量的对数似然函数
$$
\log L(\theta)=\log \prod_{k=1}^K\theta_k^{N_k} 
=\sum_{k=1}^KN_k\log\theta_k 
$$
考虑约束条件，拉格朗日函数为
$$
\mathcal L(\theta)=\sum_{k=1}^KN_k\log\theta_k+\alpha(1-\sum_{k=1}^K\theta_k)
$$
先对 $\theta_k$ 求导
$$
\frac{\partial\mathcal L(\theta)}{\partial\theta_k}=\frac{N_k}{\theta_k}-\alpha=0
$$
于是
$$
\theta_k=\frac{N_k}{\alpha}
$$
考虑
$$
\sum_{k=1}^K\theta_k=\sum_{k=1}^K\frac{N_k}{\alpha}=\frac{N}{\alpha}=1
$$
所以参数 $\theta_k$ 的极大似然估计为
$$
\hat\theta_k=\frac{N_k}{N}
$$
即为特征 $x=c_k$ 的频率。

## 贝叶斯估计

贝叶斯派认为被估计的参数是一个随机变量，服从某种分布。在获得观测数据之前，我们设定一个先验概率分布，在有观测数据之后，由贝叶斯公式计算出一个后验概率分布，这样估计出来的结果往往会更合理。


### 最大后验估计

**最大后验估计**（Maximum A Posteriori，MAP）认为最优参数为后验概率最大的参数。

给定包含 $N$ 个样本的数据集  
$$
D=\{\mathbf x_1,\mathbf x_2,\cdots,\mathbf x_N\}
$$
假设这些样本是独立同分布的，分布函数为 $\mathbb P(\mathbf x;\theta)$。引入贝叶斯定理：

(1) 若$\theta$ 为离散变量，分布率为
$$
\mathbb P(\theta_i|D)=\frac{\mathbb P(D|\theta_i)\mathbb P(\theta_i)}{\sum_j\mathbb P(D|\theta_j)\mathbb P(\theta_j)}
$$
(2) 若$\theta$ 为连续变量，概率密度函数为
$$
p(\theta|D)=\frac{p(D|\theta)p(\theta)}{\int_{\Theta}p(D|\theta)p(\theta)\mathrm d\theta}
$$
预估的参数为 $\theta$，条件概率 $p(\theta|D)$ 为参数 $\theta$ 的**后验概率**（posterior probability）密度，$p(\theta)$ 为引入的**先验概率**（prior probability）密度，在给定参数的前提下，观测数据的概率分布为$p(D|\theta)$，也就是似然函数（likelihood）。

后续统一考虑$\theta$ 为连续变量的情况。由于分母为边缘分布
$$
p(D)=\int_{\Theta}p(D|\theta)p(\theta)\mathrm d\theta
$$
该值不影响对 $\theta$的估计，在求最大后验概率时，可以忽略分母。
$$
p(\theta|D)\propto p(D|\theta)p(\theta)=\prod_{i=1}^N p(\mathbf x_i|\theta)p(\theta)
$$
于是参数 $\theta$ 的最大后验估计为
$$
\theta_{MAP}=\arg\max_{\theta} \prod_{i=1}^N p(\mathbf x_i|\theta)p(\theta)
$$
同样为了便于计算，对两边取对数
$$
\log p(\theta|D)\propto \sum_{i=1}^N \log p(\mathbf x_i|\theta)+\log p(\theta)
$$
于是参数 $\theta$ 的最大后验估计为
$$
\theta_{MAP}=\arg\max_{\theta} \left\{\sum_{i=1}^N \log p(\mathbf x_i|\theta)+\log p(\theta)\right\}
$$
与极大似然估计比较发现，当先验概率为均匀分布时，最大后验估计也就是极大似然估计。

### 贝叶斯估计

**贝叶斯估计**（Bayesian Estimation）是最大后验估计的进一步扩展，同样假定参数是一个随机变量，但贝叶斯估计并不是直接估计出参数的某个特定值，而是通过贝叶斯定理估计参数的后验概率分布。
$$
p(\theta|D)=\frac{p(D|\theta)p(\theta)}{\int_{\Theta}p(D|\theta)p(\theta)\mathrm d\theta}
$$
从上面的公式中可以看出，贝叶斯估计的求解非常复杂，因此选择合适的先验分布就非常重要。一般来说，计算积分 $\int_{\Theta}p(D|\theta)p(\theta)\mathrm d\theta$ 是不可能的，如果使用共轭先验分布，就可以更好的解决这个问题。

后验概率分布确定后，可以通过后验风险最小化获得点估计。一般常使用后验分布的期望作为最优估计，称为**后验期望估计**，它也被简称为**贝叶斯估计**。

假设 $L(\hat\theta,\theta)$ 是估计值为 $\hat\theta$ 的损失函数，则样本为 $x$ 下的条件风险（期望损失）为
$$
R(\hat\theta|x)=\int_\Theta L(\hat\theta,\theta)p(\theta|x)\mathrm d\theta
$$
则整个样本空间 $x\in\mathcal X$ 的风险为
$$
R=\int_{\mathcal X} R(\hat\theta|x)p(x)\mathrm dx
$$
由于 $R(\hat\theta|x)>0$，求 $R$ 最小即求 $R(\hat\theta|x)$ 最小。所以，最优估计
$$
\theta_{BE}=\arg\min_{\hat{\theta}}R(\hat\theta|x)
$$
通常采用平方误差损失函数
$$
L(\hat\theta,\theta)=\frac{1}{2}(\theta-\hat\theta)^2
$$
对条件风险求导，并置为0 [^int]
$$
\begin{aligned}
\frac{\mathrm dR(\hat\theta|x)}{\mathrm d\hat\theta}=&\frac{\mathrm d}{\mathrm d\hat\theta}\int_\Theta \frac{1}{2}(\theta-\hat\theta)^2p(\theta|x)\mathrm d\theta \\
=&\int_\Theta (\hat\theta-\theta)p(\theta|x)\mathrm d\theta \\
=&\hat\theta\int_\Theta p(\theta|x)\mathrm d\theta-\int_\Theta \theta p(\theta|x)\mathrm d\theta \\
=&\hat\theta|x-\mathbb E(\theta|x) \\
=&0
\end{aligned}
$$
可得到最优估计
$$
\hat\theta|x=\mathbb E(\theta|x)=\int_\Theta \theta p(\theta|x)\mathrm d\theta
$$
同理可得，在给定样本集 $D$ 下，$\theta$ 的贝叶斯估计
$$
\hat\theta|D=\mathbb E(\theta|D)=\int_\Theta \theta p(\theta|D)\mathrm d\theta
$$
**概率分布的核**：如果数据 $D$ 和参数 $\theta$ 的联合概率密度正比于概率密度$g(\theta;\tau)$ 的核 $\kappa(\theta;\tau)$
$$
p(D|\theta)p(\theta)\propto g(\theta;\tau)\propto k(\theta;\tau)
$$
则 $\theta$ 的后验概率密度
$$
p(\theta|D)=g(\theta;\tau)
$$

证明：假设联合概率密度 
$$
p(D|\theta)p(\theta)=h(D)g(\theta;\tau)=Ch(D)\kappa(\theta;\tau)
$$
其中 $Ch(D)$ 与 $\theta$ 无关， $g(\theta;\tau)$ 是由参数 $\tau$ 控制的概率密度函数，即
$$
\int_{\Theta}g(\theta;\tau)\mathrm d\theta=1
$$
由于边缘分布为
$$
\begin{aligned}
p(D)&=\int_{\Theta}p(D|\theta)p(\theta)\mathrm d\theta \\
&=\int_{\Theta}Ch(D)\kappa(\theta;\tau)\mathrm d\theta \\
&=h(D)\int_{\Theta}g(\theta;\tau)\mathrm d\theta \\
&=h(D)
\end{aligned}
$$
因此
$$
\begin{aligned}
p(\theta|D)&=\frac{p(D|\theta)p(\theta)}{p(D)} \\
&=\frac{h(D)g(\theta;\tau)}{h(D)} \\
&=g(\theta;\tau)
\end{aligned}
$$

### 共轭分布

先验分布的选择通常是需要有一些技巧性的。在贝叶斯统计中，如果后验分布与先验分布属于同类（分布形式相同），则先验分布与后验分布被称为**共轭分布**（conjugate distribution），而先验分布被称为似然函数的**共轭先验**（conjugate prior）。

共轭先验可以简化计算。因为后验分布和先验分布形式相近，只是参数有所不同，这意味着当我们获得新的观察数据时，我们就能直接通过参数更新，获得新的后验分布，此后验分布将会在下次新数据到来的时候成为新的先验分布。如此一来，我们更新后验分布就不需要通过大量的计算，十分方便。

常用的共轭先验分布如下：

(1) 当样本来自正态分布，方差已知时，估计均值的共轭先验是**正态分布**，记为 $X\sim N(\mu,\sigma^2)$。概率密度函数为
$$
f(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$
数学特征如下
$$
\mathbb E(X)=\mu;\quad \text{var}(X)=\sigma^2;\quad \text{mode}(X)=\mu
$$
(2) 当样本来自正态分布，均值已知时，估计方差的共轭先验是**逆Gamma分布**（Inverse Gamma），记为$X\sim IGamma(\alpha,\beta)$，定义域为 $x>0$。概率密度函数为
$$
f(x;\alpha,\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}(\frac{1}{x})^{\alpha+1}e^{-\beta/x}
$$
其中 
$$
\Gamma(\alpha)=\int_0^{+\infty} t^{\alpha-1}e^{-t}\mathrm dt
$$
数学特征如下
$$
\mathbb E(X)=\frac{\beta}{\alpha-1};\quad \text{var}(X)=\frac{\beta^2}{(\alpha-1)^2(\alpha-2)};\quad \text{mode}(X)=\frac{\beta}{\alpha+1}
$$
(3) 当样本来自正态分布，方差和均值均未知时，共轭先验分布为Normal-Inverse Gamma分布，形式过于复杂。

(4) 当样本来自伯努利分布$B(1,\theta)$，共轭先验是**Beta分布**，记为 $X\sim Beta(\alpha,\beta)$，定义域为 $[0,1]$。概率密度函数为
$$
f(x;\alpha,\beta)=\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}
$$
其中 $B(\alpha,\beta)$ 为Beta函数
$$
B(\alpha,\beta)=\int_0^1 t^{\alpha-1}(1-t)^{\beta-1}\mathrm dt=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}
$$
数学特征如下
$$
\mathbb E(X)=\frac{\alpha}{\alpha+\beta};\quad \text{var}(X)=\frac{\alpha\beta}{(\alpha+\beta+1)(\alpha+\beta)^2};\quad \text{mode}(X)=\frac{\alpha-1}{\alpha+\beta-2}
$$
(5) 当样本来自离散分布，共轭先验是**狄利克雷分布**（Dirichlet Distribution），是Beta分布的多元推广。表示为 $\mathbf X\sim \mathcal D(\alpha_1,\cdots,\alpha_K)$，随机变量 $\mathbf X=(X_1,\cdots,X_K)$，$x_k>0$且满足 $\sum_{k=1}^Kx_k=1$。概率密度函数为
$$
f(x_1,\cdots,x_K;\alpha_1,\cdots,\alpha_K)=\frac{1}{B(\alpha_1,\cdots,\alpha_K)}\prod_{k=1}^K x_k^{\alpha_k-1}
$$
其中
$$
B(\alpha_1,\cdots,\alpha_K)=\frac{\prod_{k=1}^K\Gamma(\alpha_k)}{\Gamma(\sum_{k=1}^K\alpha_k)}
$$
数学特征如下
$$
\mathbb E(X_k)=\frac{\alpha_k}{\alpha_0};\quad \text{var}(X_k)=\frac{\alpha_k(\alpha_0-\alpha_k)}{\alpha_0^2(\alpha_0+1)};\quad \text{mode}(X_k)=\frac{\alpha_k-1}{\alpha_0-K}
$$
其中$\alpha_0=\sum_{k=1}^K\alpha_k$

(6) 当样本来自Poisson分布 $P(\lambda)$，估计参数 $\lambda$ 的共轭先验是**Gamma分布**，记为$X\sim Gamma(\alpha,\beta)$，定义域为 $x>0$。概率密度函数为
$$
f(x;\alpha,\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}
$$
其中 
$$
\Gamma(\alpha)=\int_0^{+\infty} t^{\alpha-1}e^{-t}\mathrm dt
$$
数学特征如下
$$
\mathbb E(X)=\frac{\alpha}{\beta};\quad \text{var}(X)=\frac{\alpha}{\beta^2};\quad \text{mode}(X)=\frac{\alpha-1}{\beta}
$$
(7) 当样本来自指数分布 $Exp(\lambda)$，估计参数 $\lambda$ 的共轭先验是**Gamma分布**，记为$X\sim Gamma(\alpha,\beta)$，定义域为 $x>0$。

(8) 当样本来自均匀分布 $U(0,\theta)$，估计参数 $\theta$ 的共轭先验是**帕累托分布**（Pareto distribution），记为$X\sim Pareto(\alpha,\beta)$，定义域为 $x>\beta>0$​。概率密度函数为
$$
f(x;\alpha,\beta)=\frac{\alpha\beta^\alpha}{x^{\alpha+1}}
$$
数学特征如下
$$
\mathbb E(X)=\frac{\beta}{\alpha-1};\quad \text{var}(X)=\frac{\alpha\beta^2}{(\alpha-1)(\alpha-2)};\quad \text{mode}(X)=\beta
$$

### 正态分布

假设连续特征 $X$ 服从正态分布 $X\sim N(\mu,\sigma^2)$。概率密度函数
$$
p(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$
似然函数
$$
p(D|\mu,\sigma^2)=\left(\frac{1}{\sqrt{2\pi}\sigma}\right)^N\prod_{i=1}^N\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)
$$
(1) 若方差 $\sigma^2$ 已知，均值的共轭先验分布为正态分布$\mu\sim N(\mu_0,\tau_0^2)$，则
$$
p(\mu)\propto \exp\left(-\frac{(\mu-\mu_0)^2}{2\tau_0^2}\right)
$$
联合概率密度
$$
\begin{aligned}
p(\mu)p(D|\mu)&\propto \exp\left(-\frac{(\mu-\mu_0)^2}{2\tau_0^2}\right)\prod_{i=1}^N\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right) \\
&= \exp\left(-\frac{(\mu-\mu_0)^2}{2\tau_0^2}-\sum_{i=1}^N\frac{(x_i-\mu)^2}{2\sigma^2}\right) \\
&\propto\exp\left(-\frac{1}{2}\left((\frac{1}{\tau_0^2}+\frac{N}{\sigma^2})\mu^2-2(\frac{\mu_0}{\tau_0^2}+\frac{N\bar x}{\sigma^2})\mu\right)\right) \\
&\propto\exp\left(-\frac{(\mu-\mu_1)^2}{2\tau_1^2}\right) \\
\end{aligned}
$$
其中 $\mu_1=\dfrac{N\bar x\tau_0^2+\sigma^2\mu_0}{N\tau_0^2+\sigma^2},\quad\tau_1^2=\dfrac{\sigma^2\tau_0^2}{N\tau_0^2+\sigma^2}$

于是得到均值的后验分布服从正态分布
$$
\mu|D\sim N(\dfrac{N\bar x\tau_0^2+\sigma^2\mu_0}{N\tau_0^2+\sigma^2},\dfrac{\sigma^2\tau_0^2}{N\tau_0^2+\sigma^2})
$$
均值的最大后验估计和贝叶斯估计均为
$$
\hat\mu=\dfrac{N\bar x\tau_0^2+\sigma^2\mu_0}{N\tau_0^2+\sigma^2}
$$

注意到后验均值
$$
\hat\mu=\dfrac{\sigma^2}{N\tau_0^2+\sigma^2}\mu_0+\dfrac{N\tau_0^2}{N\tau_0^2+\sigma^2}\bar x
$$
是先验均值和样本均值的加权平均。后验精度(后验方差的倒数) $N\tau_0^2+\sigma^2$ 是先验精度与样本精度之和，因为精度大于0，后验整合了先验和样本的信息，提高了精度(降低了方差)。

(2) 若均值 $\mu$ 已知，方差的共轭先验分布为逆Gamma分布 $\sigma^2\sim IGamma(\alpha,\beta)$，则
$$
p(\sigma^2)\propto (\frac{1}{\sigma^2})^{\alpha+1}\exp(-\frac{\beta}{\sigma^2})
$$
联合概率密度
$$
\begin{aligned}
p(\mu)p(D|\mu)&\propto (\frac{1}{\sigma^2})^{\alpha+1}\exp(-\frac{\beta}{\sigma^2})(\frac{1}{\sigma^2})^{N/2}\prod_{i=1}^N\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right) \\
&=(\frac{1}{\sigma^2})^{\alpha+N/2+1}\exp\left(-\frac{1}{\sigma^2}(\beta+\frac{1}{2}\sum_{i=1}^N(x_i-\mu)^2)\right)
\end{aligned}
$$
于是得到方差的后验分布服从逆Gamma分布
$$
\sigma^2|D\sim IGamma(\alpha+\frac{N}{2},\beta+\frac{1}{2}\sum_{i=1}^N(x_i-\mu)^2)
$$
方差的最大后验估计为
$$
\sigma^2_{MAP}=\frac{2\beta+\sum_{i=1}^N(x_i-\mu)^2}{2\alpha+N+2}
$$
方差的贝叶斯估计为
$$
\sigma^2_{BE}=\sigma^2_{MAP}=\frac{2\beta+\sum_{i=1}^N(x_i-\mu)^2}{2\alpha+N-2}
$$

### 伯努利分布

假设二分类特征 $x\in\{0,1\}$，服从伯努利分布
$$
\mathbb P(x|\theta)=\theta^x(1-\theta)^{1-x}=\begin{cases}
\theta, &\text{if }x=1 \\
1-\theta, &\text{if }x=0
\end{cases}
$$
参数 $\theta$ 的似然函数为
$$
p(D|\theta)=\prod_{i=1}^N \theta^{x_i}(1-\theta)^{1-x_i}
$$
参数 $\theta$ 的共轭先验为Beta分布，$\theta\sim Beta(\alpha,\beta)$，则
$$
p(\theta)\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}
$$
联合概率密度
$$
\begin{aligned}
p(\theta)p(D|\theta)&\propto \theta^{\alpha-1}(1-\theta)^{\beta-1}\prod_{i=1}^N \theta^{x_i}(1-\theta)^{1-x_i} \\
&\propto \theta^{\alpha+N_1-1}(1-\theta)^{\beta+N_0-1}
\end{aligned}
$$
其中$N_1=\sum_{i=1}^Nx_i$为正类$x=1$的样本数，$N_0=N-N_1$为负类$x=0$的样本数。

于是得到参数 $\theta$ 的后验分布同样服从Beta分布
$$
\theta|D\sim Beta(\alpha+N_1,\beta+N_0)
$$

最大后验估计
$$
\theta_{MAP}=\frac{N_1+\alpha-1}{N+\alpha+\beta-2}
$$

贝叶斯估计
$$
\theta_{BE}=\mathbb E(\theta|D)=\frac{N_1+\alpha}{N+\alpha+\beta}
$$


### 离散分布

假设离散特征有 $K$ 个可能值 $x\in\{c_1,c_1,\cdots,c_K\}$，分布率为
$$
\mathbb P(x=c_k|\theta)=\theta_k,\quad k=1,2,\cdots,K \\
\text{s.t. }\sum_{k=1}^K\theta_k=1
$$
若 $x=c_k$ 出现的次数为 $N_k$，即$\sum_{k=1}^KN_k=N$。则参数向量的似然函数
$$
p(D|\theta)=\prod_{k=1}^K\theta_k^{N_k} 
$$
参数 $\theta$ 的共轭先验为狄利克雷分布，$\theta\sim \mathcal D(\alpha_1,\cdots,\alpha_K)$，则
$$
p(\theta)\propto \prod_{k=1}^K \theta_k^{\alpha_k-1}
$$
联合概率密度
$$
\begin{aligned}
p(\theta)p(D|\theta) &\propto \prod_{k=1}^K \theta_k^{\alpha_k-1}\prod_{k=1}^K\theta_k^{N_k} \\
&=\prod_{k=1}^K \theta_k^{\alpha_k+N_k-1}
\end{aligned}
$$
后验分布同样服从狄利克雷分布
$$
\theta|D\sim \mathcal D(\alpha_1+N_1,\cdots,\alpha_K+N_K)
$$
(1) 联合概率对数形式为
$$
\log p(\theta)p(D|\theta)=C+\sum_{k=1}^K(\alpha_k+N_k-1)\log\theta_k
$$
考虑约束条件，拉格朗日函数为
$$
\mathcal L(\theta)=C+\sum_{k=1}^K(\alpha_k+N_k-1)\log\theta_k+\lambda(1-\sum_{k=1}^K\theta_k)
$$
对上式求导，并置为0
$$
\frac{\partial\mathcal L(\theta)}{\partial\theta_k}=\frac{1}{\theta_k}(\alpha_k+N_k-1)-\lambda=0
$$
于是
$$
\theta_k=\frac{N_k+\alpha_k-1}{\lambda}
$$
考虑
$$
\sum_{k=1}^K\theta_k=\sum_{k=1}^K\frac{N_k+\alpha_k-1}{\lambda}=\frac{N-K+\sum_{k=1}^K\alpha_k}{\lambda}=1
$$
所以参数 $\theta_k$ 的最大后验估计
$$
\theta_k^{MAP}=\frac{N_k+\alpha_k-1}{N-K+\sum_{k=1}^K\alpha_k}
$$


(2) 对 $\theta$ 的后验分布求期望可获得贝叶斯估计
$$
\theta_k^{BE}=\frac{N_k+\alpha_k}{\sum_{k=1}^K(N_k+\alpha_k)}=\frac{N_k+\alpha_k}{N+\sum_{k=1}^K\alpha_k}
$$
如果先验分布中我们预先认为每个类别出现的概率是一致的，即 $\alpha_1=\alpha_2=\cdots=\alpha_K=\alpha$，此时有
$$
\hat\theta_k=\frac{N_k+\alpha}{N+K\alpha}
$$
称 $\alpha>0$ 为先验平滑因子。

- 当 $\alpha=0$ 时，就是极大似然估计；
- 当 $\alpha=1$ 时，称为拉普拉斯平滑（Laplaces moothing），也意味着参数服从的是均匀分布 $U(0,1)$，也是狄利克雷分布的一种情况。
- 当 $\alpha<1$ 时，称为Lidstone平滑。

在贝叶斯分类算法中，类条件概率常使用贝叶斯估计。假设特征 $x_j$ 有$S_j$个可能值 $x_j\in\{a_{j1},a_{j2},\cdots,a_{jS_j}\}$ ，则类条件概率的贝叶斯估计为
$$
\hat P_{\alpha}(x_j=a_{js}|c_k)=\frac{N_{ks}+\alpha}{N_k+\alpha S_j}
$$
其中 $N_{ks}=\sum_{i=1}^N\mathbb I(x_{ij}=a_{js},y_i=c_k)$ 是类别为$c_k$ 样本中特征值 $a_{js}$ 出现的次数。$N_k$为类别为$c_k$的样本个数。如果数据集中类别 $c_k$没有样本，即$N_k=0$，则 $\hat P(x_j=a_{js}|c_k)=1/S_j$ ，即假设类别 $c_k$中的样本均匀分布。

显然，先验平滑因子避免了因训练集样本不充分而导致概率估值为零的问题， 并且在训练集变大时，修正过程所引入的先验知识的影响也会逐渐变得可忽略，使得估值渐趋向于实际概率值。

### 泊松分布

假设特征 $x$ 服从泊松分布
$$
\mathbb P(x|\lambda)=\frac{\lambda^x}{x!} e^{-\lambda}
$$
参数 $\lambda$ 的似然函数为
$$
p(D|\lambda)=\exp(-N\lambda)\prod_{i=1}^N \frac{\lambda^{x_i}}{x_i!}
$$
参数 $\lambda$ 的共轭先验为Gamma分布，$\lambda\sim Gamma(\alpha,\beta)$，则
$$
p(\lambda)\propto\lambda^{\alpha-1}\exp(-\beta \lambda)
$$
联合概率密度
$$
\begin{aligned}
p(\lambda)p(D|\lambda)&= \lambda^{\alpha-1}\exp(-\beta \lambda)\exp(-N\lambda)\prod_{i=1}^N \frac{\lambda^{x_i}}{x_i!} \\
&\propto \lambda^{\alpha+N\bar x-1}\exp(-(\beta+N)\lambda)
\end{aligned}
$$
其中$\bar x=\frac{1}{N}\sum_{i=1}^Nx_i$为样本均值。于是得到参数 $\lambda$ 的后验分布同样服从Gamma分布

$$
\lambda|D\sim Gamma(\alpha+N\bar x,\beta+N)
$$

最大后验估计
$$
\lambda_{MAP}=\frac{\alpha+N\bar x-1}{\beta+N}
$$


贝叶斯估计
$$
\lambda_{BE}=\mathbb E(\lambda|D)=\frac{\alpha+N\bar x}{\beta+N}
$$

### 指数分布

假设特征 $x$ 服从指数分布
$$
\mathbb P(x|\lambda)=\lambda e^{-\lambda x}
$$
参数 $\lambda$ 的似然函数为
$$
p(D|\lambda)=\lambda^N\prod_{i=1}^N \exp(-\lambda x_i)
$$
参数 $\lambda$ 的共轭先验为Gamma分布，$\lambda\sim Gamma(\alpha,\beta)$，则
$$
p(\lambda)\propto\lambda^{\alpha-1}\exp(-\beta \lambda)
$$
联合概率密度
$$
\begin{aligned}
p(\lambda)p(D|\lambda)&= \lambda^{\alpha-1}\exp(-\beta \lambda)\lambda^N\prod_{i=1}^N \exp(-\lambda x_i) \\
&= \lambda^{\alpha+N-1}\exp(-(\beta+N\bar x)\lambda)
\end{aligned}
$$
其中$\bar x=\frac{1}{N}\sum_{i=1}^Nx_i$为样本均值。于是得到参数 $\lambda$ 的后验分布同样服从Gamma分布

$$
\lambda|D\sim Gamma(\alpha+N,\beta+N\bar x)
$$

最大后验估计
$$
\lambda_{MAP}=\frac{\alpha+N-1}{\beta+N\bar x}
$$


贝叶斯估计
$$
\lambda_{BE}=\mathbb E(\lambda|D)=\frac{\alpha+N}{\beta+N\bar x}
$$

### 均匀分布

假设特征 $x$ 服从均匀分布
$$
p(x|\theta)=\frac{1}{\theta},\quad x\in[0,\theta]
$$
参数 $\theta$ 的似然函数为
$$
p(D|\theta)=\theta^{-N}
$$
参数 $\theta$ 的共轭先验为Pareto分布，$\theta\sim Pareto(\alpha,\beta)$，则
$$
p(\theta)\propto\theta^{-(\alpha+1)}
$$
联合概率密度
$$
\begin{aligned}
p(\theta)p(D|\theta)&\propto \theta^{-(\alpha+1)}\theta^{-N} \\
&\propto \theta^{-(\alpha+N+1)}
\end{aligned}
$$
于是得到参数 $\theta$ 的后验分布同样服从Pareto分布
$$
\theta|D\sim Pareto(\alpha+N,\beta')
$$

注意 $\beta'=\max\{x_1,\cdots,x_N,\beta\}$

最大后验估计
$$
\theta_{MAP}=\beta'
$$


贝叶斯估计
$$
\theta_{BE}=\mathbb E(\theta|D)=\frac{\beta'}{N+\alpha-1}
$$

