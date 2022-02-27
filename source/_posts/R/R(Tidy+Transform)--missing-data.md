---
title: R手册(Tidy+Transform)--naniar and simputation
tags:
  - R
  - 数据清洗
  - 缺失值
categories:
  - R
  - Tidy+Transform
cover: /img/naniar-cover.png
description: 缺失数据概述、可视化和插补
abbrlink: 71c316f2
date: 2018-05-01 23:37:57
top_img:
---

缺失值是指粗糙数据中由于缺少信息而造成的数据的聚类、分组、删失或截断。它指的是现有数据集中某个或某些属性的值是不完全的。

<!-- more -->

缺失值从缺失的分布来讲可以分为完全随机缺失，随机缺失和完全非随机缺失。

**完全随机缺失**（missing completely at random，MCAR）
指的是数据的缺失是随机的，数据的缺失不依赖于任何不完全变量或完全变量。

**随机缺失**(missing at random，MAR)
指的是数据的缺失不是完全随机的，即该类数据的缺失依赖于其他完全变量。

**完全非随机缺失**(missing not at random，MNAR)
指的是数据的缺失依赖于不完全变量自身。

# naniar

## 缺失数据摘要

函数|说明
:---|:---
vis_miss(x)|缺失数据可视化摘要
miss_var_summary(data)| 缺失数据数值摘要<br>`data %>%group_by(factor) %>%miss_var_summary()`
n_miss(x) , n_complete(x)| 缺失行/完整行数量
prop_miss(x) , prop_complete(x)| 缺失行/完整行比例
geom_miss_point()| ggplot2散点图扩展
gg_miss_var(x,show_pct = FALSE)| 绘制每个变量的缺失情况

## 阴影矩阵

函数|说明
:---|:---
as_shadow(data)|  shadow matrices (NA+!NA)
bind_shadow()|将阴影矩阵绑定到原始数据框
```
data %>%bind_shadow() %>%
ggplot(aes(x = v1, fill = v2_NA)) + geom_density(alpha = 0.5)
```
## 可视化缺失值变量分布关系

```r
可视化估算值分布:
data %>%bind_shadow() %>%
  simputation::impute_lm(Ozone ~ Temp + Solar.R) %>%
  ggplot(aes(x = Solar.R,y = Ozone,colour = Ozone_NA)) + 
  geom_point() 
```

----------

# simputation 

simputation: make imputation simpler for missing data

- 中位数插补：impute_median(dat, formula)
	 	`formula:  <imputed variables> ~ <model object>`

- 模型插补，函数调用约定如下：
`impute_<model>(data, formula, … )`

function|model|package
:---|:---|:---
impute_rlm|M-estimation|MASS
impute_en|ridge/elasticnet/lasso|glmnet
impute_cart|CART|rpart
impute_rf|random forest|randomForest
impute_rhd|random hot deck|VIM (optional)
impute_shd|sequential hot deck|VIM (optional)
impute_knn|k nearest neighbours|VIM (optional)
impute_mf|missForest|missForest
impute_em|mv-normal|norm

***for example***

```r
data %>% 
  impute_lm(v1 ~ v2 + v3) %>%impute_median(v1 ~ v4)%>%  #链接插补
  impute_rlm(data, v1 + v2 ~ v3 +v4)  #多变量插补

data %>% group_by(factor) %>% impute_lm(v1 ~ v2+v3) #分组插补
```
