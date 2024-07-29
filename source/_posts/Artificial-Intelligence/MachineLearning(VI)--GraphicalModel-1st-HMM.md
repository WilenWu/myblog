---
title: 机器学习(VI)--概率图模型(一)隐马尔可夫模型
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/data-analysis.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 69c08fe2
description:
date: 2024-07-29 14:30:00
---

# 贝叶斯网

**概率图模型** (Probabilistic Graphical Model，PGM) 是指一种用图来描述变量关系的概率模型。概率图模型的好处是提供了一种简单的可视化概率模型的方法，有利于设计和开发新模型，并且简化数学表示，提高推理运算能力。

概率图一般由结点和边（也叫连接）组成，一个结点表示一个或者一组随机变量， 边表示变量之间的概率关系，结点和边构成图结构。 根据边的性质不间，概率图模型可大致分为两类：

- 第一类是使用有向无环图 (Directed Acyclic Graph) 表示变量间的依赖关系，称为有向图模型 (Directed Graphical Model) 或贝叶斯网 (Bayesian Network, BN)；
- 第二类是使用无向图 (Undirected Graph) 表示变量间的相关关系，称为无向图模型 (Undirected Graphical Model) 或马尔可夫网 (Markov Random Field, MRF)。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/PGM.svg"  />

若变量间存在显式的因果关系，则常使用贝叶斯网；若变量问存在相关性，但难以获得显式的因果关系，则常使用马尔可夫网。

**概率图理论**共分为三个部分

- 表示 (representation)：对于一个概率模型，如何通过图结构来描述变量之间的依

  赖关系；

- 推断 (inference)：在已知部分变量时，计算其他变量的条件概率分布；

- 学习 (learning)：对图的结构和参数的学习。

**贝叶斯网**(Bayesian Network)也称信念网络(Belief Network)，它有效地表达了属性间的条件独立性。对于一个 $K$ 维随机向量 $\mathbf x$，假设$\pi_k$ 为 $x_k$ 的所有父节点集合，则联合概率分布为
$$
p(x_1,x_2,\cdots,x_K)=\prod_{k=1}^Kp(x_k|\pi_k)
$$
其中 $p(x_k|\pi_k)$ 表示每个随机变量的局部条件概率分布 (Local Conditional Probability Distribution)。很多经典的机器学习模型可以使用贝叶斯网来描述，比如朴素贝叶斯分类器、隐马尔可夫模型、深度信念网络等。

**条件独立性**：在贝叶斯网络中，如果两个节点是直接连接的，它们肯定是非条件独立的，是直接因果关系。如果两个节点不是直接连接的，但可以由一条经过其他节点的路径来连接， 那么这两个节点之间的条件独立性就比较复杂。以三个节点的贝叶斯网络为例

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/DAG.svg)

# 马尔可夫链

## 基本概念

为更好的理解马尔可夫链，我们先从一个实例开始。假设有一个小镇，只有三种天气：🌧️☀️☁️，而当前的天气只依赖于前一天的天气，而和之前的天气没有任何关系。图中每个剪头代表从一个状态到另一个状态的转移概率，其中每个状态的所有出箭头的权重之和都为1，这便是一个马尔可夫链。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Transition_matrix_example.png" style="zoom:80%;" />

19世纪，概率论的发展从对相对静态的随机变量的研究发展到对随机变量的时间序列，即**随机过程**(Stochastic Process)的研究。为了简化贝叶斯网的复杂度，马尔可夫提出了一种假设，在随机过程中，下一个状态的取值仅依赖于当前状态。而符合这个假设的随机过程称为**马尔可夫过程**，也称为**马尔可夫链** (Markov Chain)。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Markov_Chain.svg)

上图给出马尔可夫过程的状态序列 (state sequence) 
$$
X=\{x_1,x_2, \dots,x_T\}
$$
其中随机变量 $\forall x\in S=\{s_1,s_2,\cdots,s_N\}$。

由马尔科夫链的性质可知
$$
\mathbb P(x_{t+1}|x_1,x_2, \cdots,x_t)=\mathbb P(x_{t+1}|x_t)
$$
这会大大简化联合概率分布
$$
\mathbb P(X)=\mathbb P(x_1)\prod_{t=2}^T\mathbb P(x_t|x_{t-1})
$$
其中，条件概率 $\mathbb P(y_t|y_{t-1})$ 称为状态转移概率。不难看出马尔可夫链的核心是**状态转移矩阵** (Transition Matrix)，通常记为 $A=(a_{ij})_{N\times N}$ 。元素 $a_{ij}$ 表示从状态 $s_i$ 转移到状态 $s_j$ 的概率

$$
a_{ij}=\mathbb P(x_{t+1}=s_j|x_t=s_i)
$$
且转移矩阵每一行的元素之和都为1
$$
\sum_{j=1}^Na_{ij}=1
$$

## 平稳分布

对于一个系统来说， 考虑它的长期的性质是很必要的， 本节我们将研究马尔可夫链的极限状况。

继续来看上节的例子，假如小镇当前天气为 🌧️，记作行向量 $\pi^{(0)}=[1,0,0]$ ，现在你想预测未来的天气情况。根据马尔可夫链的性质 $\pi^{(t+1)}=\pi^{(t)}A$ ，我们可以依次求得之后每一天的天气的概率分布。

| State | 🌧️      | ☁️      | ☀️      |
| ----- | ------ | ------ | ------ |
| 0     | 1      | 0      | 0      |
| 1     | 0.5    | 0.3    | 0.2    |
| 2     | 0.37   | 0.27   | 0.36   |
| 3     | 0.293  | 0.273  | 0.434  |
| 4     | 0.2557 | 0.2727 | 0.4716 |
| 5     | 0.2369 | 0.2727 | 0.4903 |
| 6     | 0.2276 | 0.2727 | 0.4997 |
| 7     | 0.2229 | 0.2727 | 0.5044 |
| 8     | 0.2205 | 0.2727 | 0.5067 |
| 9     | 0.2194 | 0.2727 | 0.5079 |

可以看到从第 6 次开始，天气状态就开始收敛至 $[0.22,0.27,0.51]$ ，如果我们换一个初始状态同样会收敛到这个概率分布。这其实是马尔可夫链的收敛性质。

**平稳分布**：令行向量 $\pi^{(t)}$ 表示序列中 $t$ 时刻的**状态概率向量** $x_t\sim \pi^{(t)}$ 。
$$
\pi^{(t)}=\begin{bmatrix}
\pi^{(t)}_1&\pi^{(t)}_2 &\cdots& \pi^{(t)}_N
\end{bmatrix}\\ 
\text{s.t. }\sum_{j=1}^N\pi^{(t)}_j=1
$$
元素 $\pi^{(t)}_j$ 表示取状态 $s_j$ 的概率
$$
\pi^{(t)}_j=\mathbb P(x_t=s_j)
$$
$\pi^{(0)}$ 称为**初始状态概率分布**。

根据马尔可夫链的性质，我们知道
$$
\pi^{(t+1)}=\pi^{(t)}A
$$
如果这个概率分布在一段时间后不随时间变化  $\pi^{(n+1)}=\pi^{(n)}$ ，即存在一个概率分布 $\pi$ 满足
$$
\pi=\pi A\\ \text{s.t. }\pi\mathbf 1=1
$$
则称分布 $\pi$ 为该马尔可夫链的**平稳分布**(Stationary Distribution) 。

这其实是一个特征向量问题
$$
A^T\pi^T=\pi^T
$$
$\pi^T$ 为矩阵 $A^T$ 的对应特征值为 1 的特征向量(归一化)。

> 其实矩阵 $A^T$ 普遍存在$\lambda=1$ 的特征值：由于 $A^T$ 每列元素之和为1，则矩阵 $(A^T-I)$ 每列元素之和为 0 ，即矩阵 $(A^T-I)$ 行向量线性相关，所以行列式 $\det(A^T-I)=0$ 。因此 1 是矩阵 $A^T$ 的特征值。

上例中可以求得小镇天气的平稳分布
$$
\pi=\begin{bmatrix}0.2182&0.2727&0.5091\end{bmatrix}
$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/stationary_distribution.svg" style="zoom:80%;" />

下面从另一个角度讨论马尔可夫链的收敛。

**收敛性**：由马尔可夫链的性质可以递推得到
$$
\pi^{(n)}=\pi^{(0)}A^n
$$
如果一个马尔可夫链满足不可约性和周期性，那么对于任意一个初始状态 $\pi^{(0)}$，一段时间后都会收敛到平稳分布
$$
\pi=\lim\limits_{n\to\infty}\pi^{(0)}A^n
$$

**$n$ 步转移矩阵**：$n$ 步转移概率指的是系统从状态 $s_i$ 经过步 $n$ 后转移到状态 $s_j$ 的概率，它对中间的 $n-1$ 步转移经过的状态无要求。记作
$$
P_{ij}(n)=\mathbb P(x_{t+n}=s_j|x_t=s_i)
$$
通过递推我们可以得到
$$
P_{ij}(1)=A_{ij} \\
P_{ij}(2)=(A^2)_{ij} \\
\vdots \\
P_{ij}(n)=(A^n)_{ij}
$$
上面用到了 **Chapman-Kolmogorov 定理**：马尔可夫链满足
$$
P_{ij}(n)=\sum_kP_{ik}(r)P_{kj}(n-r)
$$
如果马尔可夫链的所有状态不可约、非周期、常返态，则称为遍历的，那么
$$
\lim\limits_{n\to\infty}P_{ij}(n)=\lim\limits_{n\to\infty}(A^n)_{ij}=\pi_j
$$
即
$$
\lim\limits_{n\to\infty} A^n=\begin{bmatrix}
\pi_1&\pi_2 &\cdots& \pi_N \\
\vdots &\vdots &\ddots& \vdots \\
\pi_1&\pi_2 &\cdots& \pi_N 
\end{bmatrix}
$$
表示从状态 $s_i$ 出发，经过无穷步数后处于状态 $s_j$ 的概率，对于固定的 $j$ 值，这个数是一样的。换句话说，它不依赖于开始的状态。

其中行向量
$$
\pi=\begin{bmatrix}\pi_1&\pi_2 &\cdots& \pi_N \end{bmatrix}
$$
是方程 
$$
\pi A=\pi \\ \text{s.t. }\sum_{j=1}^N\pi_j=1
$$
 的唯一非负解。

例如二阶转移矩阵
$$
A=\begin{bmatrix}1-p & p\\ q &1-q\end{bmatrix}\quad (0<p,q<1)
$$
特征值分解
$$
A=\begin{bmatrix}1 & -p\\ 1 &q\end{bmatrix}
\begin{bmatrix}1 & 0\\ 0 &1-p-q\end{bmatrix}
\begin{bmatrix}1 & -p\\ 1 &q\end{bmatrix}^{-1}
$$
这使得矩阵的乘方的计算化简
$$
A^n=\begin{bmatrix}1 & -p\\ 1 &q\end{bmatrix}
\begin{bmatrix}1 & 0\\ 0 & 1-p-q\end{bmatrix}^n
\begin{bmatrix}1 & -p\\ 1 &q\end{bmatrix}^{-1}
$$
由于 $|1-p-q|<1$ ，所以 $A^n$ 的极限
$$
\lim\limits_{n\to \infty} A^n=\begin{bmatrix}\frac{q}{p+q} & \frac{p}{p+q}\\ \frac{q}{p+q} & \frac{p}{p+q}\end{bmatrix}
$$
可见此马尔可夫链的多步转移概率有一个稳定的极限。

## 状态的性质

**常返性**：若状态 $s_i$ 和 $s_j$ 间存在转移 $P_{ij}(m)>0$ 和 $P_{ji}(n)>0$ ，则称状态 $s_i$ 和 $s_j$ 互通，记为 $s_i\lrarr s_j$ 。

记 $P_{ij}$ 是从状态 $s_i$ 出发经有限步到达 $s_j$ 的概率
$$
P_{ij}=\sum_{n=1}^{\infty} P_{ij}(n)
$$
若自身返回概率 $P_{ii}<1$ 的状态 $s_i$ 称为**暂态**（transient state），否则称为**常返状态**（recurrent state）。对于暂态，意味着有概率 $1-P_{ii}$ 从 $s_i$ 出发不能再返回 $s_i$ 。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/reducible_Markov_chain.svg)

**不可约性**：若马尔可夫链有些状态无法从其他状态返回，称这个马尔可夫链**可约**（reducible）。可以从任何一个状态到达其他任何状态的马尔可夫链称为**不可约**(irreducible)。

**周期性**：若状态 $s_i$ 可返回自身的最小步数 $n$ 大于 1，则称 $s_i$ 是周期的，否则称为非周期的。
$$
\arg\min_{n}P_{ii}(n)>1
$$

# 隐马尔可夫模型

## 基本概念

**隐马尔可夫模型** (Hidden Markov Model, HMM) 是用来表示一种含有隐变量的马尔可夫过程，主要用于时序数据建模，在语音识别、自然语言处理等领域有广泛应用。

假设 Jack 每天会在网络分享两种情绪 ☹️😄 ，他的情绪取决于小镇当天的天气 🌧️☁️☀️，小镇当天的天气仅与前一天的天气有关，而我们并不在 Jack的小镇，所以某一天的天气情况是未知的。这便是一个隐马尔可夫模型。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/hmm_example.png" style="zoom:80%;" />

在隐马尔可夫模型中，所有的隐变量构成一个马尔可夫链，称为**状态序列** (state sequence) 。每个状态生成一个观测，而由此产生**观测序列** (observation sequence)，观测变量是已知的输出值。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Hidden_Markov_Model.svg)

上图给出隐马尔可夫模型的图模型表示，其中观测序列和(隐藏)状态序列分别为
$$
X=\{x_1,x_2,\cdots,x_T\},\quad Y=\{y_1,y_2,\cdots,y_T\}
$$
任意时刻 $x_t\in\{o_1,o_2,\cdots,o_M\}$ 和 $y_t\in\{s_1,s_2,\cdots,s_N\}$ 。

> 状态变量通常是离散的，观测变量可以是离散型也可以是连续型，为便于讨论，我们仅考虑离散型观测变量。

变量之间的依赖关系遵循以下**两个准则**：

1. 观测变量的取值仅依赖于状态变量，即 $x_t$ 由 $y_t$ 确定；
2. 系统下一个状态的取值 $y_{t+1}$ 仅依赖于当前状态 $y_t$，不依赖于以往的任何状态（马尔可夫性质）。

图中我们使用箭头标注相应的概率

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/hmm_example_pg.png" style="zoom:80%;" />

基于这种依赖关系，隐马尔可夫模型的**联合概率分布**为
$$
\mathbb P(X,Y)=\mathbb P(y_1)\mathbb P(x_1|y_1)\prod_{t=2}^T\mathbb P(y_t|y_{t-1})\mathbb P(x_t|y_t)
$$
其中，条件概率 $\mathbb P(y_t|y_{t-1})$ 称为转移概率，条件概率 $\mathbb P(x_t|y_t)$ 称为观测概率。

易知，欲确定一个HMM模型需要以下**三组参数**：

- 状态转移概率矩阵：从当前状态转移到下一个状态的概率，通常记为 $A=(a_{ij})_{N\times N}$ 
  $$
  a_{ij}=\mathbb P(y_t=s_j|y_{t-1}=s_i)
  $$

- 观测概率矩阵：根据当前状态获得各个观测值的概率，通常记为 $B=(b_{ij})_{N\times M}$
  $$
  b_{ij}=\mathbb P(x_t=o_j|y_t=s_i)
  $$

- 初始状态概率向量：初始时刻各状态出现的概率，通常记为 $\pi=(\pi_1,\pi_2,\cdots,\pi_N)$ 
  $$
  \pi_i=\mathbb P(y_1=s_i)
  $$
  

> 初始状态概率向量 $\pi$ 一般使用马尔可夫链的平稳分布。

**观测序列的生成**：隐马尔可夫模型由初始状态概率向量$\pi$、状态转移概率矩阵 $A$ 和观测概率矩阵 $B$ 决定。$\pi$ 和 $A$ 决定了不可观测的状态序列，$B$ 决定如何从状态生成观测序列。因此，隐马尔可夫模型可以用三元符号 $\lambda=(A,B,\pi)$ 表示。它按如下过程产生观测序列 $X=\{x_1,x_2,\cdots,x_T\}$

1. 设置 $t=1$，并根据初始状态概率 $\pi$ 产生初始状态 $y_1$；
2. 根据状态 $y_t$ 和输出观测概率 $B$ 产生观测变量取值 $x_t$；
3. 根据状态 $y_t$ 和状态转移矩阵 $A$ 产生 $y_{t+1}$；
4. 若 $t<T$， 设置 $t=t+1$，并转到第 (2) 步，否则停止。

**三个基本问题**：在实际应用中，人们常关注隐马尔可夫模型的三个基本问题：

- 评估问题：给定模型$\lambda=(A,B,\pi)$ 和观测序列 $X=\{x_1,x_2,\cdots,x_T\}$ ，计算在模型 $\lambda$ 下观测序列 $X$ 出现的概率 $\mathbb P(X|\lambda)$。前向算法通过自底向上递推计算逐步增加序列的长度，直到获得目标概率值。
- 解码问题：给定模型 $\lambda=(A,B,\pi)$ 和观测序列 $X=(x_1,x_2,\cdots,x_T)$ ，找到与此观测序列最匹配的状态序列 $Y=\{y_1,y_2,\cdots,y_T\}$ ，即最大化 $\mathbb P(Y|X)$。Viterbi算法采用动态规划的方法，自底向上逐步求解。
- 学习问题：给定观测序列 $X=\{x_1,x_2,\cdots,x_T\}$，估计模型 $\lambda=(A,B,\pi)$ ，即最大化  $\mathbb P(X|\lambda)$ 。 EM 算法可以高效的对隐马尔可夫模型进行训练。

例如，在语音识别等任务中，观测值为语音信号，隐藏状态为文字，目标就是根据语音信号来推断最有可能的文字序列，，即上述第二个问题。

## 评估问题

评估问题指给定模型$\lambda=(A,B,\pi)$ 和观测序列 $X=\{x_1,x_2,\cdots,x_T\}$ ，计算在模型 $\lambda$ 下观测序列 $X$ 出现的概率 $\mathbb P(X|\lambda)$。

前向算法（forward algorithm）通过自底向上递推计算，逐步增加序列的长度，直到获得目标概率值。前向算法计算$P(O|\lambda)$的复杂度是$O(N^2T)$阶的，直接计算的复杂度是$O(TN^T)$阶，原因在于每一次计算直接引用前一时刻的计算结果，**避免重复计算**。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/hmm_forward_algorithm_example.svg)

在前向算法中，定义 $t$ 时刻部分观测序列为 $x_1,x_2,\cdots,x_t$，且状态为 $s_i$ 的概率为**前向概率**，记作
$$
\alpha_t(s_i)=\mathbb P(x_1,x_2,\cdots,x_t,y_t=s_i|\lambda)
$$
基于前向变量，很容易得到该问题的递推关系
$$
\alpha_{t+1}(s_i)=\sum_{j=1}^N\alpha_t(s_j)\mathbb P(s_i|s_j)\mathbb P(x_{t+1}|s_i)
$$
及目标概率
$$
\mathbb P(X|\lambda)=\sum_{i=1}^N\alpha_T(s_i)
$$
因此可使用动态规划法，自底向上逐步求解。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/hmm_forward_algorithm.svg)

## 解码问题

解码问题指给定模型 $\lambda=(A,B,\pi)$ 和观测序列 $X=(x_1,x_2,\cdots,x_T)$ ，找到与此观测序列最匹配的状态序列 $Y=\{y_1,y_2,\cdots,y_T\}$ ，即最大化 $\mathbb P(Y|X)$。
$$
\arg\max_{Y} \mathbb P(Y|X)
$$
如果仔细观察，会发现并没有直接的方法找到这个序列。

由贝叶斯定理和马尔可夫性质知道
$$
\mathbb P(Y|X)=\frac{\mathbb P(X|Y)\mathbb P(Y)}{\mathbb P(X)}=\frac{\prod_{t=1}^T\mathbb P(x_t|y_t)\mathbb P(y_t|y_{t-1})}{\mathbb P(X)}
$$
其中，定义 $\mathbb P(y_1|y_0)=\mathbb P(y_1)$ 。最大化目标可修改为
$$
\arg\max_{Y}\prod_{t=1}^T\mathbb P(x_t|y_t)\mathbb P(y_t|y_{t-1})
$$
Viterbi算法采用动态规划的方法，自底向上逐步求解。
