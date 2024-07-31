---
title: 机器学习(VI)--概率图模型(二)条件随机场
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/data-analysis.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 1ebcf865
description:
date: 2024-07-31 14:08:45
---

# 马尔可夫随机场

## 基本概念

有向图表达是变量之间的关系是单向关系，即一个影响另一个，比如因果关系。 但很多时候变量之间的关系是互相影响的，这时候有向图将不是那么方便了。 无向图模型 (Undirected Graphical Model) 也称**马尔可夫随机场** (Markov Random Field, MRF)，是一类用无向图来描述一组具有局部马尔可夫性质的模型。很多经典的机器学习模型可以使用无向图模型来描述，比如对数线性模型 (也叫最大熵模型)、条件随机场、玻尔兹曼机、受限玻尔兹曼机等。

下图中每个结点表示一个或一组变量，结点之间的边表示两个变量之间的依赖关系。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/Markov_Random_Field.svg)

> 图 (graph) 是由节点 (node) 和及连接节点的边 (edge) 组成的集合，记作 $G(V,E)$ ，其中 $V,E$ 分别表示节点和边的集合。无向图是指边没有方向的图。

**局部马尔可夫性** (local Markov property) ：对于一组观测序列 $X=\{x_1,x_2,\cdots,x_N\}$，变量 $x_i$ 在给定它的邻居的情况下独立于所有其他变量
$$
\mathbb P(x_i|X\setminus\{x_i\})=\mathbb P(x_i|\mathcal N(x_i))
$$
其中 $\mathcal N(x_i)$ 为 $x_i$ 的邻居集合(有连接)。

有向图中的有向边表示两个变量的依赖关系，这种关系可以看成是因果、影响、依赖等等，是一个影响着另一个，对应着条件概率。 而在无线图中，无向边表示两个变量的一种互相关系，互相影响，互相软限制等等，不再是条件概率，因此无法用链式法则对联合概率 $\mathbb P(X)$ 进行逐一分解。一般以全连通子图为单位进行分解。

**团与最大团**：无向图中的一个**全连通子图**，称为**团** (Clique)，即团内的所有节点之间都连边。若在一个团中加入另外任何一个结点都不再形成团，则称该团为**最大团** (maximal clique)。换言之，最大团就是不能被其他团所包含的团。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/maximal_clique_MRF.svg)

上图所示的无向图中共有 2 个最大团 $\{x_1,x_2,x_3\},\{x_2,x_4\}$ 。显然，每个结点至少出现在一个最大团中。

**条件独立**：与有向图一样，无向图也表示一个分解方式，同时也表示一组条件独立关系。如图所示，若从结点集 $A$ 中的结点到 $B$ 中的结点都必须经过结点集 $C$ 中的结点，则称结点集 $A$ 和 $B$ 被结点集 $C$ 分离， $C$ 称为**分离集** (separating set)。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/separating_set_MRF.svg" style="zoom:80%;" />

结点集 $A,B$ 和 $C$ 满足条件独立性
$$
P(x_A,x_B|x_C)=P(x_A|x_C)P(x_B|x_C)
$$
需要注意的一点，在无向图的一个团中，任意两个结点都是存在边的，换句话说一个团中的结点都是互为邻居的， 也就是说**同一个团中的结点间不存在（条件）独立性** 。

**全局马尔可夫性** (global Markov property)：给定两个变量子集的分离集，则这两个变量子集条件独立。

## 因子分解

根据概率论中的链式法则，联合概率写成若干子局部函数的乘积的形式，也就是将联合概率进行因子分解 (factorization)，这样便于模型的学习与计算。在有向图中，局部函数即为条件概率 $\mathbb P(x_i|\pi_i)$ 。考虑无向图中两个没有边的两个结点 $x_i,x_j$，根据条件独立性质，在给定其他所有结点 $X_{rest}$ 时，则有 
$$
\begin{aligned}
\mathbb P(X) &=\mathbb P(x_i,x_j|X_{rest})\mathbb P(X_{rest}) \\
&=\mathbb P(x_i|X_{rest})\mathbb P(x_j|X_{rest})\mathbb P(X_{rest})
\end{aligned}
$$
这意味着，我们可以将条件独立的 $x_i$ 和 $x_j$ 分别放到不同局部函数。而同一个团中的结点绝对不满足条件独立性的，不同团之间的结点是可以有条件独立性的，这就意味着我们可以将局部函数的定义在最大团上。
$$
\psi_C(X_C)\propto \mathbb P(X_C|X\setminus X_C)
$$
无向图表示的联合概率分布可以分解成定义在最大团上的局部函数的乘积。

**因子分解**：在马尔可夫随机场中，一般团中的变量关系也体现在它所对应的极大团中，因此常常基于最大团来定义变量的联合概率分布
$$
\mathbb P(X)=\frac{1}{Z}\prod_{C\in\mathcal C}\psi_C(X_C)
$$
其中，$\mathcal C$ 是图中最大团的集合， $X_C$ 是团 $C$ 中随机变量的集合，局部函数 $\psi_C(X_C)\ge 0$ 是从团 $C$ 到实数域的一个映射，称为**势函数** (potential Function) 或**因子** (factor)，主要用于定义概率分布函数。因为势函数不再是有效的概率形式，其连乘的结果同样不具有概率意义，所以需要一个归一化因子  $Z$ ，通常称为**分配函数**（partition function）
$$
Z=\sum_X\prod_{C\in\mathcal C}\psi_C(X_C)
$$
以确保联合概率 $\mathbb P(X)$ 构成一个概率分布。在实际应用中，精确计算 $Z$ 通常很困难，但许多任务往往并不需获得$Z$ 的精确值。

马尔可夫随机场中的势函数 $\psi_C(X_C)$ 的作用是来量化团 $X_C$ 中节点状态和相邻节点状态组合的联合概率。由于势能函数必须为正，因此我们一般定义为指数函数
$$
\psi_C=\exp(-E_C(X_C))
$$
其中 $E_C(X_C)$ 为**能量函数**(energy function)。

> 这里的负号是遵从物理上习惯，即能量越低意味着概率越高。

联合概率分布可以表示为
$$
\mathbb P(X)=\frac{1}{Z}\exp(-\sum_{C\in\mathcal C}E_C(X_C))
$$
这种形式的分布又称为**玻尔兹曼分布**(Boltzmann distribution)。

# 条件随机场

条件随机场 (Conditional Random Field, CRF) 是一种直接建模条件概率的无向图模型。即在给定随机序列  $X$ 条件下，随机变量 $Y$ 的马尔可夫随机场。具体来说，令随机序列 $X=\{x_1,x_2,\cdots,x_N\}$ 为观测序列，随机序列 $Y=\{y_1,y_2,\cdots,y_N\}$ 为与之相应的输出序列，称为标记序列或状态序列，条件随机场的目标是构建条件概率模型 $\mathbb P(Y|X)$ 。直观上看，条件随机场与HMM的解码问题十分类似，都是在给定观测值序列后，研究状态序列可能的取值。

**条件随机场**：若变量 $y_i$ 在给定它的邻居集 $\mathcal N(y_i)$ 的情况下独立于所有其他变量
$$
\mathbb P(y_i|X,Y\setminus\{y_i\})=\mathbb P(y_i|X,\mathcal N(y_i))
$$
则 $(Y,X)$ 构成一个条件随机场。

理论上来说，条件随机场可具有任意结构，只要能表示标记变量之间的条件独立性关系即可。但在现实应用中，一个最常用的条件随机场为下图所示的链式结构，称为**线性链条件随机场** (linear-chain CRF)

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/linear-chain_CRF.svg)

下面我们主要讨论线性链条件随机场，满足马尔可夫性
$$
\mathbb P(y_i|X,Y\setminus\{y_i\})=\mathbb P(y_i|X,y_{i-1},y_{i+1})
$$
在条件随机场中，对  $\mathbb P(Y|X)$  进行因子分解。如上图，线性链条件随机场主要包含两种关于标记变量的团，即单个标记变量 $\{y_i\}$ 以及相邻的标记变量 $\{y_{i-1},y_i\}$。通过选用指数势函数并引入两类**特征函数** (feature function)便可以定义出目标条件概率
$$
\mathbb P(Y|X)=\frac{1}{Z}\exp\left(\sum_k\sum_{i=1}^{N-1}\lambda_k t_k(X,y_{i+1},y_i,i)+\sum_j\sum_{i=1}^N\mu_j s_j(X,y_i,i)\right)
$$
上式中的第二项仅考虑单结点，第一项则考虑每一对结点的关系。

- $t_k$ 是定义在观测变量的两个相邻标记边上的**转移特征函数** (transition feature function)，用于刻画相邻标记变量之间的相关关系以及观测序列对它们的影响；
- $s_j$ 是定义在观测变量的标记位置上的**状态特征函数** (status feature function)，用于刻画观测变量对标记变量的影响；
-  $\lambda_k,\mu_j$ 为超参数，是对应 $t_k,s_j$ 的权重；
- $Z$ 为规范化因子，用于确保式构成概率分布。

以词性标注为例，如何判断给出的一个标注序列靠谱不靠谱呢？**转移特征函数主要判定两个相邻的标注是否合理**，例如：动词+动词显然语法不通；**状态特征函数则判定观测值与对应的标注是否合理**，例如： ly结尾的词—>副词较合理。因此我们可以定义一个特征函数集合，用这个特征函数集合来为一个标注序列打分，并据此选出最靠谱的标注序列。也就是说，每一个特征函数（对应一种规则）都可以用来为一个标注序列评分，把集合中所有特征函数对同一个标注序列的评分综合起来，就是这个标注序列最终的评分值。可以看出：**特征函数是一些经验的特性**。

**条件随机场的向量形式**：设有 $K$ 个转移特征，$J$​ 个状态特征，为简便起见，引入特征函数向量
$$
\mathbf t(X,Y)=\begin{bmatrix}
\sum_{i=1}^{N-1}t_k(X,y_{i+1},y_i,i)
\end{bmatrix}_{K\times 1} \\
\mathbf s(X,Y)=\begin{bmatrix}
\sum_{i=1}^Ns_j(X,y_i,i)
\end{bmatrix}_{J\times 1}
$$
和权重向量
$$
\mathbf w_{\lambda}=[\lambda_1,\lambda_2,\cdots,\lambda_K]^T \\
\mathbf w_{\mu}=[\mu_1,\mu_2,\cdots,\mu_J]^T
$$
于是，条件随机场可表示为
$$
\mathbb P(Y|X;\mathbf w)=\frac{1}{Z}\exp(\mathbf w_{\lambda}^T \mathbf t(X,Y)+\mathbf w_{\mu}^T \mathbf s(X,Y))
$$
其中
$$
Z(X;\mathbf w)=\sum_{Y}\exp(\mathbf w_{\lambda}^T \mathbf t(X,Y)+\mathbf w_{\mu}^T \mathbf s(X,Y))
$$

# 概率图转换

有向图和无向图可以相互转换，但将无向图转为有向图通常比较困难。在实际应用中，将有向图转为无向图更加重要，这样可以利用无向图上的精确推断算法，比如联合树算法 (Junction Tree Algorithm)。

无向图模型可以表示有向图模型无法表示的一些依赖关系，比如循环依赖；但它不能表示有向图模型能够表示的某些关系，比如因果关系。以下图中的有向图为例，其联合概率分布可以分解为
$$
\mathbb P(X)=\mathbb P(x_1)\mathbb P(x_2)\mathbb P(x_3)\mathbb P(x_4|x_1,x_2,x_3)
$$
其中 $\mathbb P(x_4|x_1,x_2,x_3)$ 和四个变量都相关。如果要转换为无向图，需要将这四个变量都归属于一个团中。因此，需要将 $x_4$ 的三个父节点之间都加上连边，如图所示。这个过程称为**道德化** (Moralization)，转换后的无向图称为**道德图** (Moral Graph)。在道德化的过程中，原来有向图的一些独立性会丢失，比如例子中 $x_1\perp x_2\perp x_3$在道德图中不再成立。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/moralization.svg)

> 道德化的名称来源是：有共同儿子的父节点都必须结婚(即有连边)。
