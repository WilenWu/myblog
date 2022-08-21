---
title: 机器学习--时间序列分析
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/data-analysis.png
top_img: /img/data-chart3.jpg
abbrlink: 52f10172
description:
date:
---



1. 指数预测模型(Holt-Winters指数平滑)
$\text{Y(t)=Trend(t)+Seasonal(t)+Irregular(t)}$
2. ARIMA预测模型：由最近的真实值和最近的观测误差组成的线性函数
滞后阶数(lag)，自相关(ACF)，偏自相关(PACF)，差分(diff)，
平稳性：adf.test验证平稳性，通过diff或Box-Cox变换平稳 
残差的自相关检验：Box test  