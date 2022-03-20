---
title: 二分类模型评价指标
katex: true
categories:
  - 机器学习
tags:
  - 机器学习
  - 二分类
  - 混淆矩阵
  - ROC曲线
cover: /img/data-analysis.png
top_img: /img/data-chart3.jpg
description: false
abbrlink: cc734464
date: 2018-09-16 12:07:49
---

- **混淆矩阵**
混淆矩阵(Confusion Matrix)也称误差矩阵，是表示精度评价的一种标准格式。

![](https://gitee.com/WilenWu/images/raw/master/common/ConfusionMatrix.png)
> TP（实际为正预测为正），FP（实际为负但预测为正），TN（实际为负预测为负），FN（实际为正但预测为负）

<!-- more -->

- **准确率**
$Accuracy=\Large \frac{TP+TN}{TP+FP+TN+FN}$

- **查全率（召回率）和查准率**
$Recall(Sensitivity)=\Large \frac{TP}{TP+TN}$
$Precision=\Large \frac{TP}{TP+FP}$
$Specificity=\Large \frac{TN}{FP+TN}$

- **F1度量**---查准率和查全率的加权**调和平均数**
$F_{1}Score=\Large \frac{1}{1/Recall+1/Precision}=\Large \frac{2\times{Recall}\times{Precision}}{Recall+Precision}$

- **G度量**--**几何平均数**
$G=\sqrt[]{Precision\times{Recall}}$

- **ROC曲线， AUC** 
ROC曲线描绘的是不同的截断点时，并以FPR(False Positive Rate)和TPR(True Positive Rate)为横纵坐标轴，描述随着截断点的变小，TPR随着FPR的变化。
纵轴：$TPR=\Large \frac{TP}{TP+FN} \normalsize =Recall$
横轴：$FPR=\Large \frac{FP}{FP+TN}$
![roc](https://gitee.com/WilenWu/images/raw/master/common/roc.png)

- **KS曲线，KS值**
KS曲线和ROC曲线都用到了TPR，FPR。KS曲线是把TPR和FPR都作为纵坐标，而样本数作为横坐标。
TPR和FPR曲线分隔最开的位置就是最好的”截断点“，最大间隔距离就是KS值，通常>0.2即可认为模型有比较好偶的预测准确性

- **Lift 和Gain图**
Lift图衡量的是，与不利用模型相比，模型的预测能力“变好”了多少，lift(提升指数)越大，模型的运行效果越好。
Gain图是描述整体精准度的指标。
$Gain=\Large \frac{TP}{TP+FP}$
$Lift=\Large\frac{\frac{TP}{TP+FP}}{\frac{P}{P+N}}=\frac{Gain}{PR}$

<img src="https://gitee.com/WilenWu/images/raw/master/common/lift.png" alt="lift" style="zoom: 50%;" />
<img src="https://gitee.com/WilenWu/images/raw/master/common/gain.png" alt="gain" style="zoom:50%;" />

参考链接：https://blog.csdn.net/shy19890510/article/details/79501582

