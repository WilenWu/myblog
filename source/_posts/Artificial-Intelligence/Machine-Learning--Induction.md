---
title: 机器学习--探索数据
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/data-analysis.png
top_img: /img/data-chart3.jpg
abbrlink: 8638abf9
description:
date:
---

# 引言

[经典：5种常见的数据分析方法](https://baijiahao.baidu.com/s?id=1707510739052343986&wfr=spider&for=pc)

**Machine learning algorithms**

- Supervised learning
- Unsupervised learning

Reinforcement learning 强化学习

**Supervised learning**

Learns from being given "right answers"

- **Regression**: Predict a number
- **Classification**: predict categories

**Unsupervised Learning**

Data only comes with inputs x, but not output labels y. Algorithm has to find structure in the data.

- **Clustering**: Group similar data points together.
- **Anomaly detection**: (异常检测) Find unusual data points.
- **Dimensionality reduction**: Compress data using fewer numbers.

回归分析主要解决以下几个方面的问题：

(1) 确定几个特定的变量之间是否存在相关关系，如果存在的话，找出它们之间合适的数学表达式。
(2) 根据一个或几个变量的值，预测或控制另一个变量的取值，并且可以知道这种预测能达到什么样的精确度。 
(3) 进行因表分析。例如在对于共同影响一个变量的许多变量(因素)之间，找出哪些是重要因素，哪些是次要因素，这些因素之间又有什么关系等等。  

超参数：数据标准化（归一化、标准化、无处理）、正则化参数、收敛容差、最大迭代次数、求解方法（normal最小二乘法、L-BFGS牛顿法）、惩罚函数（L1、L2）

分类：经典传统算法包括决策树分类，随机森林分类，线性判别分类，神经网络分类，基于样本的KNN分类，逻辑回归分类，支持向量机分类，朴素贝叶斯分类等。Xgboost

结果只有两种的分类问题称为二分类问题 (binary classification)，输出变量通常表示为正样本/负样本(positive/negative)


# 数据探索


## 数据类型

 **变量** variables  
 连续变量  continuous  定量变量(quantitative)  
 分类变量  discrete 定性变量(categories) 
 顺序变量  ordinal  定性变量(categories) 

## 汇总统计

Summary Statistics

调和平均值 (Harmonic Mean). 在组中的样本大小不相等的情况下用来估计平均组大小。调和平均值是样本总数除以样本大小的倒数总和。

峰度 (Kurtosis). 有离群值的程度的测量。对于正态分布，峰度统计的值为 0。正峰度指示数据表现出比正态分布更多的极值离群值。负峰度指示数据表现出比正态分布更少的极值离群值。

最后一个 (Last). 显示在数据文件中遇到的最后一个数据值。

最大值 (Maximum). 数值变量的最大值。

平均值 (Mean). 集中趋势的测量。算术平均，总和除以个案个数。

中位数 (Median). 第 50 个百分位，大于该值和小于该值的个案数各占一半。如果个案个数为偶数，那么中位数是个案在以升序或降序排列的情况下最中间的两个个案的平均。中位数是集中趋势的测量，但对于远离中心的值不敏感（这与平均值不同，平均值容易受到少数多个非常大或非常小的值的影响）。

最小值 (Minimum). 数值变量的最小值。

N . 个案（观察值或记录）的数目。

总个案数的百分比。每个类别中的个案总数的百分比。

总和的百分比。每个类别中的总和百分比。

范围 (Range). 数值变量最大值和最小值之间的差；最大值减去最小值。

偏度 (Skewness). 分布的不对称性测量。正态分布是对称的，偏度值为 0。具有显著的正偏度的分布有很长的右尾。具有显著的负偏度的分布有很长的左尾。作为一个指导，当偏度值超过标准误差的两倍时，那么认为不具有对称性。

标准差 (Standard Deviation). 对围绕平均值的离差的测量。在正态分布中，68% 的个案在平均值的一倍标准差范围内，95% 的个案在平均值的两倍标准差范围内。例如，在正态分布中，如果平均年龄为 45，标准差为 10，那么 95% 的个案将处于 25 到 65 之间。

峰度标准误差 (Standard Error of Kurtosis). 峰度与其标准误差的比可用作正态性检验（即，如果比值小于 -2 或大于 +2，就可以拒绝正态性）。大的正峰度值表示分布的尾部比正态分布的尾部要长一些；负峰度值表示比较短的尾部（变为像框状的均匀分布尾部）。

平均值的标准误差 (Standard Error of Mean). 取自同一分布的样本与样本之间的平均值之差的测量。它可以用来粗略地将观察到的平均值与假设值进行比较（即，如果差与标准误差的比值小于 -2 或大于 +2，那么可以断定两个值不同）。

偏度标准误差 (Standard Error of Skewness). 偏度与其标准误差的比可用作正态性检验（即，如果比值小于 -2 或大于 +2，就可以拒绝正态性）。大的正偏度值表示长右尾；极负值表示长左尾。

总和 (Sum). 所有带有非缺失值的个案的值的合计或总计。

方差 (Variance). 对围绕平均值的离差的测量，值等于与平均值的差的平方和除以个案数减一。度量方差的单位是变量本身的单位的平方。


连续变量

截尾平均数，即丢弃了最大5%和最小5%的数据和所有缺失值后的算术平均数。

mean(x, trim = 0.05, na.rm=TRUE)

 平均数 mean
 中位数 median
 和 sum
 最大值 max
 最小值 min
 四分位数 quartile
 四分卫间距 IQR   3/4分位数-1/4分位数 
 百分位数 percentile 
 分位数 quantile(x,probs) 
 极差 range max-min 
 偏度 skewness
 峰度 kurtosis
 方差 variance
 标准差   standard  deviation  std=sqrt(var)  
 变异系数  coefficient  of variation,CV  CV=sd/mean 


离散变量

 频数 Frequency
 众数 mode 
 列联表 
 联合分布  joint distribution  


## 相似性和相异性

Measures of Similarity and Dissimilarity

相关的类型

R可以计算多种相关系数，包括Pearson相关系数、Spearman相关系数、Kendall相关系数、偏
相关系数、多分格（polychoric）相关系数和多系列（polyserial）相关系数。下面让我们依次理
解这些相关系数。

1. Pearson、Spearman和Kendall相关
    Pearson积差相关系数衡量了两个定量变量之间的线性相关程度。Spearman等级相关系数则衡
    量分级定序变量之间的相关程度。Kendall’s Tau相关系数也是一种非参数的等级相关度量。
2. 偏相关
    偏相关是指在控制一个或多个定量变量时，另外两个定量变量之间的相互关系

在计算好相关系数以后，如何对它们进行统计显著性检验呢？常用的原假设为变量间不相关
（即总体的相关系数为0）

独立性检验

R提供了多种检验类别型变量独立性的方法。本节中描述的三种检验分别为卡方独立性检验 chisq、Fisher精确检验和Cochran-Mantel-Haenszel检验。

相关性的度量
上一节中的显著性检验评估了是否存在充分的证据以拒绝变量间相互独立的原假设。如果可
以拒绝原假设，那么你的兴趣就会自然而然地转向用以衡量相关性强弱的相关性度量。vcd包中
的assocstats()函数可以用来计算二维列联表的phi系数Phi-Coefficient、列联系数Contingency Coeff和Cramer’s V系数。

### 相关性

 **相关性**  
 二维列联表的phi系数、列联系数和Cramer’s  V系数。  分类变量相关性度量  
 协方差矩阵 Covariance  cov=mean((X-mean_X)*(Y-mean_Y)) 
 相关系数矩阵   CorrelationCoefficient  cov/sqrt(var_X*Var_Y)  
 *person*   线性相关（正态连续变量）  
 *spearman* 秩相关(分级定序变量之间的相关程度)  
 *kendall*  秩相关  

 相关系数可以用来描述定量变量之间的关系。相关系数的符号（±）表明关系的方向（正相
关或负相关），其值的大小表示关系的强弱程度（完全不相关时为0，完全相关时为1）。

1. Pearson、Spearman和Kendall相关
    Pearson积差相关系数衡量了两个定量变量之间的线性相关程度。Spearman等级相关系数则衡
    量分级定序变量之间的相关程度。Kendall’s Tau相关系数也是一种非参数的等级相关度量。
2. 偏相关
    偏相关是指在控制一个或多个定量变量时，另外两个定量变量之间的相互关系

### 距离

 距离 
 euclidean 欧几里德距离
 maximum   切比雪夫距离
 manhattan 绝对值距离 
 canberra  Lance 距离 
 minkowski 明科夫斯基距离  
 binary 二分类距离 

1. Euclidean，欧氏距离 
2. cosine，夹角余弦，机器学习中借用这一概念来衡量样本向量之间的差异。
3. jaccard，杰卡德相似系数，两个集合A和B的交集元素在A，B的并集中所占的比例，称为两个集合的杰卡德相似系数，用符号J(A,B)表示。 
4. Relaxed Word Mover's  Distance（RWMD）文本分析相似性距离 

## 可视化

条形图、箱线图和点图
- 饼图和扇形图
- 直方图与核密度图

箱线图（又称盒须图）通过绘制连续型变量的五数总括，即最小值、下四分位数（第25百分
位数）、中位数（第50百分位数）、上四分位数（第75百分位数）以及最大值，描述了连续型变量
的分布。箱线图能够显示出可能为离群点（范围±1.5*IQR以外的值，IQR表示四分位距，即上四
分位数与下四分位数的差值）的观测。

![](箱线图.png)

小提琴图是箱线图与核密度图的结合。

![](小提琴图.png)

# 参数估计

 **参数估计（正态分布**）  均值和标准差估计
 点估计   point estimation 
 矩估计   `μ^，σ^`
 极大似然估计   `μ*，σ2*`  
 稳健估计  （M估计，R估计）  
 Bootstrap估计 在原始数据的范围内做有放回抽样，预估参数的一些性质。
 **估计的优良性准则**
 平方和   sum of square  SS
 均方 mean  square   MS
 无偏估计E（mean）=μ ,<br>E（S2）=σ2（S2为样本方差） 
 均方误差准则MSE(估计量θ)=E（估计量θ-θ）
 **区间估计**  interval estimation 
 置信水平（1-α）   confidence level 
 置信区间  confidence  interval,CI

# 统计检验

最后，我们将转而通过参数检验（t检验）和非参数检验（Mann-Whitney U检验、Kruskal-Wallis检验）方法研究组间差异。

-   独立样本的t 检验：一个针对两组的独立样本t检验可以用于检验两个总体的均值相等的假设。这里假设两组数据是独立的，并且是从正态总体中抽得。

-   非独立样本的t 检验：

    再举个例子，你可能会问：较年轻（14\~24岁）男性的失业率是否比年长（35\~39岁）男性的
    失业率更高？在这种情况下，这两组数据并不独立。你不能说亚拉巴马州的年轻男性和年长男性
    的失业率之间没有关系。在两组的观测之间相关时，你获得的是一个非独立组设计（dependent
    groups design）。前后测设计（pre-post design）或重复测量设计（repeated measures design）同样
    也会产生非独立的组。
    非独立样本的t检验假定组间的差异呈正态分布。

-   多于两组的情况：如果想在多于两个的组之间进行比较，应该怎么做？如果能够假设数据是从正态总体中独
    立抽样而得的，那么你可以使用方差分析（ANOVA）。ANOVA是一套覆盖了许多实验设计和准
    实验设计的综合方法。就这一点来说，它的内容值得单列一章。

组间差异的非参数检验：如果数据无法满足t检验或ANOVA的参数假设，可以转而使用非参数方法。举例来说，若结果变量在本质上就严重偏倚或呈现有序关系，那么你可能会希望使用本节中的方法。

-   两组的比较
    若两组数据独立，可以使用Wilcoxon秩和检验（更广为人知的名字是Mann-Whitney U检验）
    来评估观测是否是从相同的概率分布中抽得的（即，在一个总体中获得更高得分的概率是否比另
    一个总体要大）。调用格式为：
    wilcox.test(y ~ x, data)
    其中的y是数值型变量，而x是一个二分变量

    Wilcoxon符号秩检验是非独立样本t检验的一种非参数替代方法。它适用于两组成对数据和
    无法保证正态性假设的情境。调用格式与Mann-Whitney U检验完全相同

-   多于两组的比较

-   在要比较的组数多于两个时，必须转而寻求其他方法。考虑7.4节中的state.x77数据集。
    它包含了美国各州的人口、收入、文盲率、预期寿命、谋杀率和高中毕业率数据。如果你想比较
    美国四个地区（东北部、南部、中北部和西部）的文盲率，应该怎么做呢？这称为单向设计（one-way
    design），我们可以使用参数或非参数的方法来解决这个问题。
    如果无法满足ANOVA设计的假设，那么可以使用非参数方法来评估组间的差异。如果各组
    独立，则Kruskal-Wallis检验将是一种实用的方法。如果各组不独立（如重复测量设计或随机区组
    设计），那么Friedman检验会更合适。
    Kruskal-Wallis检验的调用格式为：
    kruskal.test(y ~ A, data)
    其中的y是一个数值型结果变量，A是一个拥有两个或更多水平的分组变量（grouping variable）。
    （若有两个水平，则它与Mann-Whitney U检验等价。）而Friedman检验的调用格式为：
    friedman.test(y ~ A | B, data)
    其中的y是数值型结果变量，A是一个分组变量，而B是一个用以认定匹配观测的区组变量（blocking variable）。在以上两例中，data皆为可选参数，它指定了包含这些变量的矩阵或数据框。

10.1　预备知识：统计检验436
10.1.1　显著性检验436
10.1.2　假设检验440
10.1.3　多重假设检验443
10.1.4　统计检验中的陷阱448
10.2　对零分布和替代分布建模450
10.2.1　生成合成数据集450
10.2.2　随机化类标451
10.2.3　实例重采样451
10.2.4　对检验统计量的分布建模451
10.3　分类问题的统计检验452
10.3.1　评估分类性能452
10.3.2　以多重假设检验处理二分类问题453
10.3.3　模型选择中的多重假设检验453
10.4　关联分析的统计检验454
10.4.1　使用统计模型455
10.4.2　使用随机化方法457
10.5　聚类分析的统计检验458
10.5.1　为内部指标生成零分布459
10.5.2　为外部指标生成零分布459
10.5.3　富集460
10.6　异常检测的统计检验461

 **假设检验** 
 单侧检验  one-sided test
 双侧检验  two-sided test
 显著性水平（α） significant  level   犯第一类错误的概率  
 独立性检验
 游程检验  Runs  Test  对二分变量的随机检验 
 卡方独立性检验  chisq test   两分类变量
 Fisher精确检验  fisher test 两分类变量
 Cochran—Mantel—Haenszel卡方检验   mantelhaen test   两个名义变量在第三个变量的每一层中都是条件独立的  


 **方差齐性检验**  
 卡方检验  chisq.test  单个正态总体的方差检验：χ2检验（H0:  σ2=σ02）
 F检验   两个正态总体方差比：F检验（H0:  σ12=σ22）  
 bartlett.test 
 fligner.test  
 Brown-Forsythe
 **总体分布类型的拟合度检验**  
 正态分布检验   shapiro.test  
 F(n,m)分布 Kolmogorov-Smirnov（K-S）
 二项分布检验   Binomial  Test
 **总体均值检验**  
 两样本(连续变量)   t检验   独立或配对样本 
 两样本(有序分类)   *Mann-Whitney  U检验* 两样本独立
 *Wilcoxon秩和检验*   两样本独立
 *Wilcoxon秩和检验*   配对样本 
 *Walsh检验*  配对样本 
 两样本(分类变量)   Kolmogorov双样本单侧检验 两样本独立
 多样本(连续变量)   方差分析  独立或相关
 多样本(有序分类)   Kruskal-Walls检验   多样本独立
 推广的Mann-Whitney检验 多样本独立
 Jonckneere检验   多样本独立
 多样本(无序分类)   Friedman检验  多样本相关

# 方差分析


多样本均值比较（$H_0:  μ_1=μ_2=…=μ_n$)
方差分析主要用途：
①均数差别的显著性检验
②分离各有关因素并估计其对总变异的作用
③分析因素间的交互作用
④方差齐性检验