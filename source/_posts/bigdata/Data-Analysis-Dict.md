---
ID: 55168f393f3ebb573995f516e38947c0
title: 数据分析理论概览
date: 2018-05-09 21:31:41
katex: true
categories: [大数据]
tags: [大数据]
cover: /img/R-C.jpg
description: 数据分析术语的简单说明
---

数据分析术语的简单说明

<!-- more -->

# 概率论与数理统计

|抽样| sample  |      |
| :----- | :------ | :----- |
| 随机采样 | random  sampling | 受总体中个体间自相关关系的影响  |
| 系统采样 | systematic sampling | 受个体特点规律性的影响|
| 随机-系统采样 | systematic-random  sampling  | 不受上面的影响 |
| 多层次采样| strtifed  sampling  | 适用于可分层的总体  |
| **变量**| variables  ||
| 连续变量 | continuous | 定量变量(quantitative)  |
| 分类变量 | discrete| 定性变量(categories) |
| 顺序变量 | ordinal | 定性变量(categories) |
| **统计描述** | summary/describe ||
| ***定量变量*** |||
| 总体均数 | mean/median||
| 四分卫间距| IQR  | 3/4分位数-1/4分位数 |
| 极差| range| max-min |
| 偏度| skewness||
| 峰度| kurtosis||
| 方差| variance||
| 标准差  | standard  deviation | sd=sqrt(var)  |
| 变异系数 | coefficient  of variation,CV | CV=sd/mean |
| ***定性变量*** | |   |
| 频数| frequency/rate||
| 联合分布 | joint distribution  ||
| 众数| mode ||
| **数据预处理**|||
| 归一化/标准化 |||
| 连续变量离散化 |||
| **缺失值模式**|||
| 完全随机缺失  |MCAR||
| 随机缺失  |MAR||
| 非随机缺失|NMAR||
| **缺失值处理**|||
| 配对删除 |||
| 简单抽样填补|||
| 均值/中位数/众数填补|||
| 回归填补，多重插补  |||
| **异常检测** | outlierTest| 离群点，孤立点 |
| 正态分布 || 3倍标准差原则 |
| 多元正态 || 马氏距离 |
| **参数估计（正态分布**） || 均值和标准差估计|
| 点估计  | point estimation ||
| 矩估计  || `μ^，σ^`|
| 极大似然估计  || `μ*，σ2*`  |
| 稳健估计 || （M估计，R估计）  |
| Bootstrap估计|| 在原始数据的范围内做有放回抽样，预估参数的一些性质。|
| **估计的优良性准则**|||
| 平方和  | sum of square | SS|
| 均方| mean  square  | MS|
| 无偏估计||E（mean）=μ ,<br>E（S2）=σ2（S2为样本方差） |
| 均方误差准则||MSE(估计量θ)=E（估计量θ-θ）|
| **区间估计** | interval estimation ||
| 置信水平（1-α）  | confidence level ||
| 置信区间 | confidence  interval,CI||
| **假设检验** |||
| 单侧检验 | one-sided test||
| 双侧检验 | two-sided test||
| 显著性水平（α）| significant  level  | 犯第一类错误的概率  |
| 独立性检验|||
| 游程检验 | Runs  Test | 对二分变量的随机检验 |
| 卡方独立性检验 | chisq test  | 两分类变量|
| Fisher精确检验 | fisher test| 两分类变量|
| Cochran—Mantel—Haenszel卡方检验  | mantelhaen test  | 两个名义变量在第三个变量的每一层中都是条件独立的  |
| **相关性**  |||
| 二维列联表的phi系数、列联系数和Cramer’s  V系数。 || 分类变量相关性度量  |
| 协方差矩阵| Covariance | cov=mean((X-mean_X)*(Y-mean_Y)) |
| 相关系数矩阵  | CorrelationCoefficient | cov/sqrt(var_X*Var_Y)  |
|| *person*  | 线性相关（正态连续变量）  |
|| *spearman*| 秩相关(分级定序变量之间的相关程度)  |
|| *kendall* | 秩相关  |
| **方差齐性检验**  |||
| 卡方检验 | chisq.test | 单个正态总体的方差检验：χ2检验（H0:  σ2=σ02）|
| F检验  || 两个正态总体方差比：F检验（H0:  σ12=σ22）  |
|| bartlett.test ||
|| fligner.test  ||
|| Brown-Forsythe||
| **总体分布类型的拟合度检验**  |||
| 正态分布检验  | shapiro.test  ||
| F(n,m)分布| Kolmogorov-Smirnov（K-S）||
| 二项分布检验  | Binomial  Test||
| **总体均值检验**  |||
| 两样本(连续变量)  | t检验  | 独立或配对样本 |
| 两样本(有序分类)  | *Mann-Whitney  U检验*| 两样本独立|
|| *Wilcoxon秩和检验*  | 两样本独立|
|| *Wilcoxon秩和检验*  | 配对样本 |
|| *Walsh检验* | 配对样本 |
| 两样本(分类变量)  | Kolmogorov双样本单侧检验| 两样本独立|
| 多样本(连续变量)  | 方差分析 | 独立或相关|
| 多样本(有序分类)  | Kruskal-Walls检验  | 多样本独立|
|| 推广的Mann-Whitney检验| 多样本独立|
|| Jonckneere检验  | 多样本独立|
| 多样本(无序分类)  | Friedman检验 | 多样本相关|

# 方差分析 

多样本均值比较（$H_0:  μ_1=μ_2=…=μ_n$)
方差分析主要用途：
①均数差别的显著性检验
②分离各有关因素并估计其对总变异的作用
③分析因素间的交互作用
④方差齐性检验

# 回归(regression)

## 线性模型
$Y=βX+error$  （OLS最小二乘法，使ESS最小值）

**Gauss-Markov假设**：$e_i ~ N(0, σ^2)$
(1) $var(e)= σ^2$ , 误差方差=样本方差
(2) $cov( e_i ,e_j )=0 (i  ≠ j)$ , 误差独立性 

|回归方程的显著性检验| ($H_0:  β_i=0$)  | t检验和F检验 |
| :----- | :------ | :----- |
| 回归平方和| Residual Sum of Squares | $RSS=∑(y_i-mean(Y))^2$ |
| 残差平方和| Explained Sum of Squares| ESS|
| 总平方和 | Total Sum of Squares | TSS=var(Y)=RSS+ESS|
| 判定系数 | | R2= RSS/TSS |
| 自由度  | degree of freedom | df |
| 平方和  | sum of square  | SS |
| 均方| mean  square| MS=SS/df |
| F检验  | | F=MSR/MSE|


| 方差源  | SS | df |
| :----- | :------ | :----- |
| 回归| RSS| 1  |
| 误差| ESS| n-2|
| 总和| TSS| n-1|

## 广义线性模型

**广义加法模型**：扩展广义线性模型，以纳入任意平滑的函数。这意味着你可以自定义函数y  = f(x)。
**惩罚线性模型**：对惩罚复杂模型的距离添加惩罚项。这往往会使来自相同群体的新数据集预测的更好。
**稳健的线性模型**：对异常值的存在不那么敏感。

- **回归诊断**
正态检验(shapiro)：自变量多重共线性kappa系数(kappa)
线性模型假设的综合验证：若sqrt(vif)>2,存在多重共线性
- **异常值**
离群点(outlierTest)
高杠杆值点（帽子统计量）
强影响点
- **改进措施**
删除观测点
变量变换
正态变换
线性变换
增删变量
- **模型选择**
逐步回归(step/stepAIC)
全子集回归(regsubsets)
- **交叉验证**
通过交叉验证法，我们便可以评价模型的泛化能力。
在k重交叉验证中，样本被分为k个子样本，轮流将k–1个子样本组合作为训练集，另外1个子样本作为保留集。这样会获得k个预测方程，记录k个保留样本的预测表现结果，然后求其平均值。
- **相对权重**（relative  weight）

# 分类

| 分类算法  |  备注|
| -------- | ------- |
| 线性判别| Fisher |
| 距离判别 | mahalanobis |
| 贝叶斯分类器  |bayes|
| 逻辑回归  |  |
| 决策树|  |
|ID3<br />C4.5<br />C5.0|Information Gain<br />Information Gain Rate|
| CART  | Gini Index|
| 条件推断树 |   显著性检验|
| 随机森林  | random forest |
| 最近邻算法 | KNN |
|支持向量机 | SVM |
| 神经网络 | NN |
| 卷积神经网络 CNN| 空间数据|
| RNN|时序数据|

# 聚类

| 层次聚类法| |
| -------- | ------- |
| single| 最短距离法 |
| complete | 最长距离法 |
| median| 中间距离法 |
| mcquitty | 相似法|
| average  | 类平均法  |
| centroid | 重心法|
| ward  | 离差平方和法|
| **划分法**| |
| k-means  | 连续变量  |
| K-modes  | 分类变量  |
| k-prototype | 混合变量  |
| PAM| |
| clarans  | |
| **密度算法**  | |
| DBSCAN| |
| OPTICS| |
| DENCLUE  | |


| 距离| |
| ----------- | -------- |
| euclidean| 欧几里德距离|
| maximum  | 切比雪夫距离|
| manhattan| 绝对值距离 |
| canberra | Lance 距离 |
| minkowski| 明科夫斯基距离  |
| binary| 二分类距离 |
| one-hot  | |

# 文本分析

文本分析常见三大距离——cosine/jaccard/Euclidean  
1. Euclidean，欧氏距离 
2. cosine，夹角余弦，机器学习中借用这一概念来衡量样本向量之间的差异。
3. jaccard，杰卡德相似系数，两个集合A和B的交集元素在A，B的并集中所占的比例，称为两个集合的杰卡德相似系数，用符号J(A,B)表示。 
4. Relaxed Word Mover's  Distance（RWMD）文本分析相似性距离 

# 时间序列

1. 指数预测模型(Holt-Winters指数平滑)
$\text{Y(t)=Trend(t)+Seasonal(t)+Irregular(t)}$
2. ARIMA预测模型：由最近的真实值和最近的观测误差组成的线性函数
滞后阶数(lag)，自相关(ACF)，偏自相关(PACF)，差分(diff)，
平稳性：adf.test验证平稳性，通过diff或Box-Cox变换平稳 
残差的自相关检验：Box test  



