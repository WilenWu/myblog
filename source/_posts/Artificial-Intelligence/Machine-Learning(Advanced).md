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

回归平方和 Residual Sum of Squares  $RSS=∑(y_i-E(Y))^2$ 
残差平方和 Explained Sum of Squares ESS
总平方和  Total Sum of Squares  TSS=var(Y)=RSS+ESS
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