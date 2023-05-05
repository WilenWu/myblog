---
title: 机器学习（进阶）
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/data-analysis.png
top_img: /img/data-chart3.jpg
abbrlink: 205d08e0
description:
date:
---

## 回归模型

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

## 回归方程显著性检验

回归方程的显著性检验  $H_0: w_j=0$ 

残差平方和 Residual Sum of Squares  $RSS=∑(y_i-E(Y))^2$ 
解释平方和 Explained Sum of Squares ESS
总离差平方和  Total Sum of Squares  TSS=var(Y)=RSS+ESS
判定系数   R2= RSS/TSS 
自由度   degree of freedom  df 
平方和   sum of square   SS 
均方 mean  square MS=SS/df 
F检验    F=MSR/MSE


| 方差源 | SS   | df   |
| :----- | :--- | :--- |
| 回归   | RSS  | 1    |
| 误差   | ESS  | n-2  |
| 总和   | TSS  | n-1  |

## 非线性回归

### 泊松回归

泊松回归：预测一个代表频数的响应变量

### Cox

Cox ：预测一个事件（死亡、失败或旧病复发）发生的时间

## 逻辑回归

**最大期望算法**：（Expectation-Maximization algorithm, EM）与真实分布最接近的模拟分布即为最优分布，因此可以通过最小化交叉熵来求出最优分布。

真实分布可写为 
$$
\mathbb P(y|\mathbf x)=1
$$
模拟分布可写为 
$$
\mathbb Q(y|\mathbf x)=[f_{\mathbf{w}}(\mathbf{x})]^{y}[1-f_{\mathbf{w}}(\mathbf{x})]^{1-y}
$$
交叉熵为
$$
\begin{aligned}
H(\mathbb P,\mathbb Q) &=-\sum_{i=1}^m \mathbb P(y_i|\mathbf x_i)\log \mathbb Q(y_i|\mathbf x_i) \\
&=\sum_{i=1}^{m}(-y_i\mathbf{w}^T\mathbf{x}+\log(1+e^{\mathbf{w}^T\mathbf{x}}))
\end{aligned}
$$
**cost function**
$$
J(\mathbf w)=\frac{1}{m}\sum_{i=1}^{m}(-y_i\mathbf{w}^T\mathbf{x}+\log(1+e^{\mathbf{w}^T\mathbf{x}}))
$$
与极大似然估计相同。



## 参数估计

 **参数估计（正态分布**）  均值和标准差估计
 点估计   point estimation 
 矩估计   `μ^，σ^`
 极大似然估计   `μ*，σ2*`  
 稳健估计  （M估计，R估计）  
 Bootstrap估计 在原始数据的范围内做有放回抽样，预估参数的一些性质。
 **估计的优良性准则**
 平方和   sum of square  SS
 均方 mean  square   MS
 无偏估计E（mean）=μ 

E（S2）=σ2（S2为样本方差） 
 均方误差准则MSE(估计量θ)=E（估计量θ-θ）
 **区间估计**  interval estimation 
 置信水平（1-α）   confidence level 
 置信区间  confidence  interval,CI

## 统计检验

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

**独立性检验**

R提供了多种检验类别型变量独立性的方法。本节中描述的三种检验分别为卡方独立性检验 chisq、Fisher精确检验和Cochran-Mantel-Haenszel检验。

## 方差分析

多样本均值比较（$H_0:  μ_1=μ_2=…=μ_n$)
方差分析主要用途：
①均数差别的显著性检验
②分离各有关因素并估计其对总变异的作用
③分析因素间的交互作用
④方差齐性检验