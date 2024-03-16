---
title: 机器学习(IV)--监督学习(六)贝叶斯分类
date: 2022-11-27 21:40
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-supervised-learning.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 8216e77c
description: 
---

#  贝叶斯分类

## 后验概率最大化

从概率学讲，特征变量可看作特征空间 $\mathcal X\sube \R^p$ 上的随机向量 $\mathbf X=(X_1,X_2,\cdots,X_p)^T$ ，目标变量可看作输出空间 $\mathcal Y=\{c_1,\cdots,c_K\}$ 上的随机变量 $Y$。数据集  
$$
D=\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\}
$$
是由 $\mathbf X$ 和 $Y$ 的联合概率 $\mathbb P(\mathbf X,Y)$ 独立同分布产生。数据集 $D$ 包含 $N$ 个样本，$p$ 个特征。其中，第 $i$ 个样本的特征向量为 $\mathbf x_i=(x_{i1},x_{i2},\cdots,x_{ip})^T\in\mathcal X$ 。目标变量 $y_i\in\mathcal Y$ 。

在概率框架下，分类任务是基于条件概率 $\mathbb P(Y=y|\mathbf X=\mathbf x)$ 的损失最小化，这个条件概率称为类 $y$ 的**后验概率**（posterior probability）。对于任意样本 $\mathbf x$ ，在每个类别的后验概率分别为 $\mathbb P(Y=c_k|\mathbf X=\mathbf x)$ 。假设模型$f$对样本 $\mathbf x$ 的预测类别为 $f(\mathbf x)=c\in \mathcal Y$ 。则期望损失为

$$
\mathbb E[L(y,c)]=\sum_{k=1}^K L(c_k,c)\mathbb P(Y=c_k|\mathbf X=\mathbf x)
$$
假设选择0-1损失函数 $L(y,f(\mathbf x))=\mathbb I(y\neq f(\mathbf x))$ ，则损失最小化的输出为
$$
\begin{aligned}
\hat y &=\arg\min_{c\in\mathcal Y}\sum_{k=1}^K \mathbb I(c_k\neq c)\mathbb P(Y=c_k|\mathbf X=\mathbf x) \\
&=\arg\min_{c\in\mathcal Y}\mathbb P(Y\neq c|\mathbf X=\mathbf x) \\
&=\arg\min_{c\in\mathcal Y} [1-\mathbb P(Y= c|\mathbf X=\mathbf x)] \\
&=\arg\max_{c\in\mathcal Y} \mathbb P(Y= c|\mathbf X=\mathbf x)
\end{aligned}
$$
即对每个样本$\mathbf x$ ，选择后验概率 $\mathbb P(Y=c|\mathbf X=\mathbf x)$ 最大的类别
$$
\hat y=\arg\max_{y\in\mathcal Y} \mathbb P(Y= y|\mathbf X=\mathbf x)
$$
期望损失最小化等价于后验概率最大化。

后验概率可由联合概率求得
$$
\mathbb P(Y=y|\mathbf X=\mathbf x)=\frac{\mathbb P(\mathbf X=\mathbf x,Y=y)}{\mathbb P(\mathbf X=\mathbf x)}
$$
后文中对概率事件 $\mathbf X=\mathbf x$ 及 $Y=y$ 统一用值 $\mathbf x$ 和 $y$ 简写。
$$
\mathbb P(y|\mathbf x)=\frac{\mathbb P(\mathbf x,y)}{\mathbb P(\mathbf x)}
$$
从概率角度来说，机器学习主要有两种策略：
- **判别式模型**（discriminative models）：直接通过最小化损失函数学习后验概率分布 $\mathbb P(y|\mathbf x)$ 来预测 $y$。例如，决策树、SVM等。
- **生成式模型**（generative models）：先学习联合概率分布 $\mathbb P(\mathbf x,y)$ 建模，然后求得后验概率分布 $\mathbb P(y|\mathbf x)$。例如，贝叶斯分类等。

## 贝叶斯定理
假设 $A,B$ 是一对随机变量，他们的联合概率和条件概率满足如下关系
$$
\mathbb P(A,B)=\mathbb P(A)\mathbb P(B|A)=\mathbb P(B)\mathbb P(A|B)
$$
由此可得到**贝叶斯定理**（Bayes’ theorem）
$$
\mathbb P(B|A)=\frac{\mathbb P(B)\mathbb P(A|B)}{\mathbb P(A)}
$$

以贝叶斯定理为基础的分类方法称为**贝叶斯分类**（Bayesian Classification），是一种典型的生成学习方法。基于贝叶斯定理，后验概率可写为
$$
\mathbb P(y|\mathbf x)=\frac{\mathbb P(y)\mathbb P(\mathbf x|y)}{\mathbb P(\mathbf x)}
$$
相对地， $\mathbb P(y)$ 称为类 $y$ 的**先验概率**（prior probability），$\mathbb P(\mathbf x|y)$ 称为类 $y$ 的**类条件概率**（class-conditional probability）。在比较不同 $y$ 值的后验概率时，分母$\mathbb P(\mathbf x)$是常数，可以忽略。因此我们可以使用以下分类规则：
$$
\hat y=\arg\max_{y}\mathbb P(y)\mathbb P(\mathbf x|y)
$$
因此，在训练阶段，我们需要基于训练数据集 $D$ 估计类先验概率 $\mathbb P(c_k)$和类条件概率$\mathbb P(\mathbf x|c_k)$。知道这些概率后，通过找出使后验概率 $\mathbb P(y'|\mathbf x')$ 最大的类 $y'$ 可以对测试记录 $\mathbf x'$ 进行分类。 

先验概率 $\mathbb P(c_k)$表达了各类样本所占的比例，根据大数定律，当训练集包含充足的独立同分布样本时，$\mathbb P(c_k)$ 可通过各类样本出现的频率来进行估计。

对类条件概率 $\mathbb P(\mathbf x|c_k)$ 来说，需要估计特征变量 $\mathbf x$ 所有取值和目标变量 $y$ 的所有组合。假设特征 $x_j$ 可取值有 $S_j$ 个，$y$ 可取值有 $K$ 个，那么所有组合的数量为 $K\prod\limits_{j=1}^p S_j$ 。可见类条件概率有指数级的数量，在现实应用中，这个值往往大于训练样本数，用频率来估计显然不可行。

## 朴素贝叶斯

### 条件独立性假设

朴素贝叶斯（Naive Bayes）在估计类条件概率时假设特征之间条件独立
$$
\mathbb P(\mathbf x|y)=\mathbb P(x_1,\cdots,x_p|y)=\prod_{j=1}^p\mathbb P(x_j|y)
$$
这是一个较强的假设。由于这一假设，模型包含的条件概率的数量大为减少，朴素贝叶斯法的学习与预测大为简化，因而朴素贝叶斯法高效，且易于实现，但同时也会牺牲一定的准确率。

基于条件独立性假设，后验概率可以重写为
$$
\mathbb P(y|\mathbf x)=\frac{\mathbb P(y)\prod\limits_{j=1}^p\mathbb P(x_j|y)}{\mathbb P(\mathbf x)}
$$
由于$\mathbb P(\mathbf x)$是常量，朴素贝叶斯分类规则变为 
$$
\hat y=\arg\max_{y}\mathbb P(y)\prod_{j=1}^p\mathbb P(x_j|y)
$$
因此，只需计算每一个特征的条件概率 $\mathbb P(x_j|y)$，总的组合数变为 $K\sum\limits_{j=1}^p S_j$，它不需要很大的训练集就能获得较好的概率估计。我们可以用极大似然估计（MLP）来估计先验概率 $\mathbb P(y)$和条件概率$\mathbb P(x_j|y)$ 。

朴素贝叶斯分类器可将涉及的所有概率估值事先计算好存储起来，这样在进行预测时只需"查表"即可进行判别。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/NaiveBayesAlgorithm.png" style="zoom: 50%;" />

尽管条件独立性假设过于简单化，但是经验表明，朴素贝叶斯分类器已经能很好的工作，最出名的就是文档分类和垃圾邮件过滤。它们需要少量的训练数据来估计必要的参数。但是，朴素贝叶斯分类器不能很好的输出模型概率。

不同的朴素贝叶斯分类器主要根据它们对特征分布所作的假设而不同。接下来，我们描述几种估计分类特征和连续特征的条件概率的方法。

### 高斯朴素贝叶斯

朴素贝叶斯分类法使用两种方法估计连续特征的类条件概率：

1. 可以把每一个连续特征离散化，然后用相应的离散区间的频率估计条件概率。
2. 可以假设连续特征服从某种概率分布，然后使用训练数据估计分布的参数。

**高斯朴素贝叶斯**（Gaussian Naive Bayes）假设连续特征服从高斯分布，对每个类别 $c_k$ ，特征 $x_j$ 的类条件概率等于
$$
\mathbb P(x_j|y)=\frac{1}{\sqrt{2\pi\sigma^2_{jk}}}\exp\left(-\frac{(x_j-\mathbf\mu_{jk})^2}{2\sigma^2_{jk}}\right)
$$

参数 $\mu_{jk}$ 和 $\sigma^2_{jk}$可以用对应的样本均值和方差估计。

注意，前面对类条件概率的解释有一定的误导性。上式子的右边对应于一个概率密度函数 $\mathcal N(x_j;\mu_{jk},\sigma^2_{jk})$。因为该函数是连续的，所以随机变量 $X_j$ 取某一特定值的概率为0。取而代之，我们应该计算 $X_j$ 落在区间 $x_j$ 到 $x_j+\epsilon$ 的条件概率，其中$\epsilon$是一个很小的常数：
$$
\mathbb P(x_j\leqslant X_j\leqslant x_j+\epsilon|Y=c_k)=\int_{x_j}^{x_j+\epsilon}\mathcal N(x_j;\mu_{jk},\sigma^2_{jk})\mathrm dx_j \approx \epsilon\cdot N(x_j;\mu_{jk},\sigma^2_{jk})
$$
由于$\epsilon$是每个类的一个常量乘法因子，在对后验概率 $\mathbb P(\mathbf x|y)$ 进行规范化的时候就抵消掉了。因此，我们仍可以使用概率密度公式来估计类条件概率 $\mathbb P(x_j|y)$ 。

考虑分类任务：预测 一个货款者是否会违约，训练集如下

| Tid  | Gender | Married | Income | Defaulted |
| :--: | :----: | :-----: | :----: | :-------: |
|  1   | Female |   No    |  125K  |    No     |
|  2   |  Male  |   Yes   |  100K  |    No     |
|  3   |  Male  |   No    |  70K   |    No     |
|  4   | Female |   Yes   |  120K  |    No     |
|  5   |  Male  |   No    |  95K   |    Yes    |
|  6   |  Male  |   Yes   |  60K   |    No     |
|  7   | Female |   No    |  220K  |    No     |
|  8   |  Male  |   No    |  85K   |    Yes    |
|  9   |  Male  |   Yes   |  75K   |    No     |
|  10  |  Male  |   Yes   |  90K   |    Yes    |

首先，估计类先验概率

P(Defaulted=No)=7/10, P(Defaulted=Yes)=3/10

然后，为每个特征估计条件概率

P(Gender=Female|Defaulted=No)=3/7,  P(Gender=Male|Defaulted=No)=4/7
P(Gender=Female|Defaulted=Yes)=0,  P(Gender=Male|Defaulted=Yes)=1
P(Married=No|Defaulted=No)=3/7,  P(Married=Yes|Defaulted=No)=4/7
P(Married=No|Defaulted=Yes)=2/3,  P(Married=Yes|Defaulted=Yes)=1/3
For Income: 
If Defaulted=No: sample mean=110,  sample variance=2975
If Defaulted=Yes: sample mean=90,  sample variance=25

对于测试样本 (Gender=Male, Married=No, Income=120K)，有

P(Defaulted=No)P(Gender=Male|Defaulted=No)P(Married=No|Defaulted=No)P(Income=120K|Defaulted=No)=7/10×4/7×3/7×0.0072=1.2×10^-3^
P(Defaulted=Yes)P(Gender=Male|Defaulted=Yes)P(Married=No|Defaulted=Yes)P(Income=120K|Defaulted=Yes)=3/10×1×2/3×1.2×10^-9^=2.4×10^-10^

由于 1.2×10^-3^ > 2.4×10^-10^，因此朴素贝叶斯将测试样本分类为No。


### 多项式朴素贝叶斯

**多项式朴素贝叶斯**（Multinomial Naive Bayes，MNB）是文本分类中使用的两个经典朴素贝叶斯变体之一。假设离散特征 $x_j$ 有$S_j$个可能值 $x_j\in\{a_{j1},a_{j2},\cdots,a_{jS_j}\}$ ，则类条件概率可以用极大似然估计（频率）
$$
\mathbb P_{\alpha}(x_j=a_{js}|c_k)=\frac{N_{ks}}{N_k}
$$
其中 $N_{ks}=\sum_{i=1}^N\mathbb I(x_{ij}=a_{js},y_i=c_k)$ 是类别为$c_k$ 样本中特征值 $a_{js}$ 出现的次数。$N_k$为类别为$c_k$的样本个数。

用极大似然估计可能会出现所要估计的概率值为0的情况。这时会影响到后验概率的计算结果，使分类产生偏差。这时可采用贝叶斯估计类条件概率
$$
\mathbb P_{\alpha}(x_j=a_{js}|c_k)=\frac{N_{ks}+\alpha}{N_k+\alpha S_j}
$$
其中 $\alpha>0$ 为先验平滑因子，为在学习样本中没有出现的类别而设计。如果数据集中类别 $c_k$没有样本，即$N_k=0$，则 $\hat P(x_j=a_{js}|c_k)=1/S_j$ ，即假设类别 $c_k$中的样本均匀分布。当训练集越大时，平滑修正引入的影响越来越小。

### 伯努利朴素贝叶斯

**伯努利朴素贝叶斯**（Bernoulli Naive Bayes）是多项式分布的特例，假设每个特征都是二值变量，服从伯努利分布。贝叶斯估计为
$$
\mathbb P_{\alpha}(x_j=1|c_k)=\frac{N_{k1}+\alpha}{N_k+2\alpha}
$$

### 补充朴素贝叶斯

**补充朴素贝叶斯**（Complement Naive Bayes，CNB）是标准多项式朴素贝叶斯算法的一种自适应算法，特别适用于不平衡的数据集。具体而言，CNB使用来自每个类的补充的统计数据来计算模型的权重。经验表明，CNB的参数估计比MNB的参数估计更稳定。此外，CNB在文本分类任务方面经常优于MNB(通常以相当大的幅度)。计算权重的程序如下：
$$
\hat\theta_{ci}=\frac{\alpha_i+\sum_{j:y_j\neq c} d_{ij}}{\alpha+\sum_{j:y_j\neq c}\sum_k d_{kj}}
$$

$$
w_{ci}=\log\hat\theta_{ci}
$$

$$
w_{ci}=\frac{w_{ci}}{\sum_j|w_{cj}|}
$$

其中对不在类$c$中的所有记录$j$求和, $d_{ij}$要么是记录$j$中的$i$的计数， 要么是tf-idf形式的值， $\alpha_i$就像MNB中的一个光滑的超参数， 同时$\alpha=\sum_i\alpha_i$ 。

第二个归一化解决了较长记录在MNB中支配参数估计的趋势。分类规则是：
$$
\hat c=\arg\max_c\sum_it_i w_{ci}
$$
也就是说， 记录被分配给最糟糕的匹配度的类。

## 贝叶斯网络

朴素贝叶斯分类器的条件独立假设似乎太严格了，特别是对那些特征之间有一定相关性的分类问题。贝叶斯网不要求给定的所有特征都条件独立，而是允许指定哪些特征条件独立。

**贝叶斯网**：令 $\mathbf x_{\pi_j}$ 表示变量 $x_j$ 所依赖的所有变量集合，则 $\mathbf x$ 的概率分布表示为
$$
\mathbb P(\mathbf x)=\prod_{j=1}^p\mathbb P(x_j|\mathbf x_{\pi_j})
$$

$\mathbf x_{\pi_j}$ 称为变量 $x_j$ 的父特征集。贝叶斯网用图形表示一组随机变量之间的概率关系，之后章节会具体介绍。

**独依赖估计**（One-Dependent Estimator，ODE）是最常用的一种策略。假设每个特征在类别之外最多仅依赖于一个其他特征，即
$$
\mathbb P(\mathbf x|y)=\prod_{j=1}^p\mathbb P(x_j|y,x_{\pi_j})
$$
其中，$x_{\pi_j}$为变量$x_j$的父特征。

**SPODE**（Super-Parent ODE）：假设所有特征都依赖于同一个特征，称为超父（Super-Parent）。然后通过交叉验证等模型选择方法来确定超父特征。

**TAN**（Tree Augmented naive Bayes）是在最大生成树（maximum weighted spanning tree）基础上，通过意两个特征间的条件互信息将特征间的依赖关系约简为如图所示的树形结构。
$$
I(x_i,x_j|y)=\sum_{i,j,k}\mathbb P(x_i,x_j|c_k)\log\frac{\mathbb P(x_i,x_j|c_k)}{\mathbb P(x_i|c_k)\mathbb P(x_j|c_k)}
$$
通过最大生成树算法，TAN实际上仅保留了强相关特征之间的依赖性。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML//TAN_DAG.svg" style="zoom:80%;" />

**OADE**（Averaged One-Dependent Estimator）是一种基于集成学习机制、更为强大的独依赖分器。与SPODE通过模型选择确定超父特征不同，AODE尝试将每个特征作为超父来构建 SPODE，然后将那些具有足够训练数据支撑的 SPODE 集成起来作为最终结果。