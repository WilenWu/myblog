---
title: 机器学习--模型选择与评估
katex: true
categories:
  - 'Artificial Intelligence'
  - 'Machine Learning'
tags:
  - 机器学习
  - 模型评估
  - 混淆矩阵
  - ROC曲线
cover: /img/data-analysis.png
top_img: /img/data-chart3.jpg
description: false
abbrlink: cc734464
date: 2018-09-16 12:07:49
---

# 引言

机器学习是关于计算机基于数据构建概率统计模型并运用模型对数据进行分析与预测的一门学科。机器学习根据训练数据是否拥有标记信息，大致可分为监督学习、非监督学习、半监督学习。

- 监督学习：(supervised learning) 代表方法是分类和回归
- 非监督学习：(unsupervised learning) 代表方法是聚类
- 半监督学习：(semi-supervised learning) 

机器学习方法三要素：

- 模型：(model) 确定模型的假设空间。假设要学习的模型属于某个函数的集合，称为假设空间 (hypothesis space)。
- 策略：(strategy) 应用某个评价准则，从假设空间中确定选取最优模型的策略。
- 算法：(algorithm) 实现求解最优模型的算法。

3.模型选择与评估
 3.1 交叉验证：评估模型表现
 3.2 调整估计器的超参数
 3.3 指标和评分：量化预测的质量
 3.4 模型持久性
 3.5 验证曲线：绘制分数以评估模型


  4  .机器学习中，进行模型选择或者说提高学习的泛化能力是一个重要问题.如果只考虑减少训练误差，就可能产生过拟合现象.模型选择的方法有正则化与交叉验证.学习方法泛化能力的分析是机器学习理论研究的重要课题.


需注意的是，机器学习的目标是使学得的模型能很好地适用于"新样本"，.学得模型适用于
新样本的能力，称为"泛化" (generalization) 能力

常假设样本空间中全
体样本服从A个未知"分布" (distribution) Ð ， 我们获得的每个样本都是独立
地从这个分布上采样获得的，即"独立同分布" (independent and identically
distributed，简称i.i.d.).

"奥卡姆剃刀" (Occam's razor)是一种常用的、自然科学
研究中最基本的原则，即"若有多个假设与观察一致，则选最简单的那个


损失函数(loss function)或代价函数(cost function)

常用优化算法《机器学习方法-6.3》：直接求解法（最小二乘法）、迭代求解法（梯度下降法，牛顿法） 

正则化《深度学习-7》



版本空间

没有免费午餐定理 No Free Lunch Theorem，简称NFL

指示函数 $\mathbb I(\text{condition})=\begin{cases}1 & \text{if true}\\ 0 & \text{if false}\end{cases}$

符号函数 $\text{sign}(x)=\begin{cases}+1 & \text{if }x\geqslant 0\\ -1 & \text{if }x<0\end{cases}$


# 模型评估

## 经验误差与过拟合

我们把学习器的实际预测输出与样本的真实输出之间的差异称为"误差" (error) ,
学习器在训练集上的误差称为"训练误差" (training error)或"经验误
差" (empirical error) ，在新样本上的误差称为"泛化误差" (generalization
error). 显然，我们希望得到泛化误差小的学习器.然而，我们事先并不知道新
样本是什么样，实际能做的是努力使经验误差最小化.

这种现象在机器学习中称为
"过拟合" (overfitting). 与"过拟合"相对的是"欠拟合" (underfitting) ，

有多种因素可能导致过拟合，其中最常见的情况是由于学习能力过于强大，
以至于把训练样本所包含的不太一般的特性都学到了，而欠拟合则通常是由
于学习能力低下而造成的

经验误差vs泛化误差

过拟合vs欠拟合

## 评估方法

训练数据分层：

- 训练集：用来训练模型，模型的迭代优化
- 测试集：不参与训练流程，监测模型效果
- 验证集：调整超参数，优化模型



留出法

交叉验证法

自助法

## 性能度量

### 分类问题

**混淆矩阵**(Confusion Matrix)也称误差矩阵，是表示分类结果精度评价的一种标准格式。矩阵的每一行代表实例的预测类别，每一列代笔实例的真实类别。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/ConfusionMatrix.png)

- TP（实际为正预测为正）
- FP（实际为负但预测为正）
- TN（实际为负预测为负）
- FN（实际为正但预测为负）

**准确率**：是最常用的分类性能指标

$\text{Accuracy}=\Large \frac{TP+TN}{TP+FP+TN+FN}$

**精确率**（查准率）：

$\text{Precision}=\Large \frac{TP}{TP+FP}$

**召回率**（查全率）：正确预测的正例数/应际正例总数

$\text{Recall(Sensitivity)}=\Large \frac{TP}{TP+TN}$
$\text{Specificity}=\Large \frac{TN}{FP+TN}$

**F1-score**：查准率和查全率的**调和平均数**（harinonic mean）

$\cfrac{1}{F_1}=\cfrac{1}{2}(\cfrac{1}{\text{Precision}}+\cfrac{1}{\text{Recall}})$

$F_1=\cfrac{2\times\text{Recall}\times\text{Precision}}{\text{Recall}+\text{Precision}}$

**F~β~-score**：查准率和查全率的加权**调和平均数**。通常，对于不同的问题，查准率查全率的侧重不同。比如，在商品推荐系统中，为了尽可能少打扰用户，更希望推荐内容确是用户感兴趣的，此时查准率更重要；而在逃犯信息检索系统中，更希望尽可能少漏掉逃犯，此时查全率更重要。

$\cfrac{1}{F_β}=\cfrac{1}{1+β^2}(\cfrac{1}{\text{Precision}}+\cfrac{β^2}{\text{Recall}})$

$F_β=\cfrac{(1+β^2)\times\text{Recall}\times\text{Precision}}{\text{Recall}+β^2 \times\text{Precision}}$

其中 β>0 度量了查全率对查准率的相对重要性，β=1 时退化为标准的F1； β> 1 时查全率有更大影响；β < 1 时查准率有更大影响。

与算术平均数（$\cfrac{1}{2}(\text{Recall} + \text{Precision})$）和几何平均数 $\sqrt[]{\text{Recall} \times \text{Precision}}$ 相比，调和平均数更重视较小值，所以精确率和召回率接近时，F值最大。很多推荐系统的评测指标就是用F值的。

**ROC曲线** （受试者工作特征曲线，Receiver Operating Characteristic）

二分类问题，对于正负例的判定通常会有一个阈值，ROC曲线描绘的是不同的阈值时，TPR(True Positive Rate)随着FPR(False Positive Rate)的变化。
纵轴：$TPR=\Large \frac{TP}{TP+FN} \normalsize =\text{Recall}$
横轴：$FPR=\Large \frac{FP}{FP+TN}$
![roc](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/roc.png)

总之，ROC曲线越靠近左上角，该分类器的性能越好。而且一般来说，如果ROC是光滑的，那么基本可以判断没有太大的过拟合。

**AUC**（Area Under Curve）：被定义为ROC曲线下的面积，AUC越大的分类器，性能越好。

$AUC=\displaystyle \frac{1}{2}\sum_{i=1}^{m-1}(x_{i+1}-x_i)(y_i+y_{i+1})$

**PR曲线**：查准率和查全率（召回率）之间的关系。查准率和查全率是一对矛盾的度量，一般来说，查准率高时，查全率往往偏低，查全率高时，查准率往往偏低。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ML/PR-curve.png)

如果一个学习器的P-R曲线被另一个学习器的P-R曲线完全包住，则可断言后者的性能优于前者，当然我们可以根据曲线下方的面积大小来进行比较，但更常用的是**平衡点**（Break-Even Point, BEP）。平衡点是查准率=查全率时的取值，如果这个值较大，则说明学习器的性能较好。

**KS曲线**（洛伦兹曲线，Kolmogorov-Smirnov）
KS曲线和ROC曲线都用到了TPR，FPR。KS曲线是把TPR和FPR都作为纵坐标，而样本数作为横坐标。
TPR和FPR曲线分隔最开的位置就是最好的阈值，最大间隔距离就是KS值。KS值可以反应模型的最优区分效果，一般$KS>0.2$可认为模型有比较好的预测准确性。

$KS=\max\{TPR-FPR\}$

- $KS<0.2$ ：模型无鉴别能力
- $0.2 ⩽ KS<0.4$ ：模型勉强接受
- $0.4 ⩽ KS<0.6$ ：模型具有区别能力
- $0.6 ⩽ KS<0.75$ ：模型有非常好的区别能力
- $KS⩾0.75$ ：此模型异常

**Gain曲线**（增益图，Gain Chart）是描述整体精准度的指标。

$\text{Gain}=\Large \frac{TP}{TP+FP}$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/gain.png" alt="gain" style="zoom:50%;" />

**Lift曲线**（提升图，Lift Chart）衡量的是，与不利用模型相比，模型的预测能力“变好”了多少，lift(提升指数)越大，模型的运行效果越好。

$\text{Lift}=\Large\frac{\frac{TP}{TP+FP}}{\frac{P}{P+N}}=\frac{\text{Gain}}{PR}$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/lift.png" alt="lift" style="zoom: 50%;" />

**模型稳定度指标PSI**（Population Stability Index）反映了**验证样本**在各分数段的分布与**建模样本**分布的稳定性。可衡量测试样本及模型开发样本评分的的分布差异，为最常见的模型稳定度评估指标。

$PSI=\sum(f_{dev}^i-f_{valid}^i)*\ln(f_{dev}^i/f_{valid}^i)$

- 若 $PSI<0.1$ 样本分布有微小变化，模型基本可以不做调整
- 若 $0.1 ⩽ PSI ⩽ 0.2$ ，样本分布有变化，根据实际情况调整评分切点或调整模型
- 若 $PSI>0.2$ 样本分布有显著变化，必须调整模型

### 回归问题

**可解释方差**（Explained Variance）：衡量所有预测值$\hat y_i$和样本值$y_i$之间的差的方差与样本本身的方差的相近程度。最大值为1，数值越大代表模型预测结果越好。

$\text{Explained Var}=1-\cfrac{\text{var}(y-\hat y)}{\text{var}(y)}$

**平均绝对误差**（Mean Absolute Error, MAE）：

$\displaystyle MSE=\frac{1}{m}\sum_{i=1}^{m}|y_i-\hat y_i|$

**均方误差**（Mean Squared Error, MSE）：衡量的是样本与模型预测值偏离程度，数值越小代表模型拟合效果越好。

$\displaystyle MSE=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat y_i)^2$

**均方根误差**（Root Mean Squard Error, RMSE）：

$RMSE=\sqrt{MSE}$

**决定系数**（R-Square）：其取值范围为$[0,1]$，一般来说，R-Square 越大，表示模型拟合效果越好。R-Square 反映的是大概有多准，因为，随着样本数量的增加，R-Square必然增加，无法真正定量说明准确程度，只能大概定量。

$R^2=1-\cfrac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y_i)^2}=1-\cfrac{MSE(y,\hat y)}{\text{var}(y)/m}$

**调整R2**（Adjusted R-Square）：

$\text{Adjusted }R^2=1-\cfrac{(1-R^2)(n-1)}{n-p-1}$

其中，n 是样本数量，p 是特征数量。调整R2抵消样本数量对 R-Square的影响，做到了真正的 0~1，越大越好。

## 比较检验

## 偏差与方差

### 正则化

《统计-2》《花书-6,7》

正则化的实现方式通常是惩罚所有的特征。

$l_2$ 范数正则化项
$$
J(\mathbf{w}) = \frac{1}{2m} \left[\sum\limits_{i = 1}^{m}\text{loss}(f_{\mathbf{w}}(\mathbf{x}^{(i)}), y^{(i)})  + \lambda\sum_{j=0}^n w_j^2\right]
$$
 按照惯例，正则化项 (Regularization term) 也乘以 $\frac{1}{2m}$ ，这样正则化项不会随训练集规模变大而变大。

> 实际生活中，正则化 b 项影响很小，可以忽略。
