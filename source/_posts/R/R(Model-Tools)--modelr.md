---
title: R手册(Model Tools)--modelr
tags:
  - R
  - 机器学习
  - tidyverse
categories:
  - R
  - 机器学习
cover: /img/modelr-cover.png
description: 辅助管道建模
abbrlink: 2e6782e
date: 2018-05-04 00:43:41
---

# modelr 

modelr软件包提供的功能可以帮助您在建模时创建精美的流水线。

<!-- more -->

## 分区和采样

`resample(data, idx)` 抽取idx向量指定的观测值
`resample_partition(data, p)` 按p向量给定的值分区

```r
rs <- resample(mtcars, 1:10)
as.data.frame(rs)

ex <- resample_partition(mtcars, c(test = 0.3, train = 0.7))
lapply(ex, dim)

bootstrap(mtcars, 100)%>%
 map( ~ lm(mpg ~ wt, data = .))%>%
 map_df( broom::tidy)
```

## 交叉验证抽样

`bootstrap(data, n, id = ".id")` 生成n个bootstrap副本
`crossv_kfold(data, k = 5, id = ".id")`将数据拆分为k独占分区，并将每个分区用于测试训练拆分
`crossv_mc(data, n, test = 0.2, id = ".id")`生成n组数据，交叉验证模型，test为测试集比例

```r
# bootstrap
boot <- bootstrap(mtcars, 100)
# k-fold cross-validation
cv1 <- crossv_kfold(mtcars, 5)
# Monte Carlo cross-validation
cv2 <- crossv_mc(mtcars, 100)

crossv_mc(mtcars, 100)%>%
  map(cv2$train, ~ lm(mpg ~ wt, data = .))%>%   #模型
  map2_dbl(cv2$test, rmse)%>%
  hist()
```


## 模型质量指标

`rmse(model, data)` 均方根误差
`mae(model, data)` 平均绝对误差
`rsquare(model, data)` R^2预测的方差除以响应的方差
`qae(model, data, probs)`误差的分位数


## 与模型交互

一组函数使您可以无缝地将预测和残差添加为现有数据框的附加列：
`add_predictions(data, model)`向数据框添加预测
`add_residuals(data, model)`向数据框添加残差

```r
df <- tibble::data_frame(
  x = sort(runif(100)),
  y = 5 * x + 0.5 * x ^ 2 + 3 + rnorm(length(x))
)

mod <- lm(y ~ x, data = df)
df %>% add_predictions(mod)
df %>% add_residuals(mod)
```

为了可视化目的，通常使用数据中均匀间隔的网格点
`data_grid(data, ..., .model = NULL)`展开数据框，包含所有值的组合，对可视化模型有用
`seq_range(x, n, by, trim = NULL, expand = NULL, pretty = FALSE)`在矢量范围内生成一个序列

```r
# 对连续性变量， seq_range函数很有用
mtcars_mod <- lm(mpg ~ wt + cyl + vs, data = mtcars)
data_grid(mtcars, wt = seq_range(wt, 10), cyl, vs) %>%   
  add_predictions(mtcars_mod)%>%
  head()
#> # A tibble: 60 x 4
#>       wt   cyl    vs  pred
#>    <dbl> <dbl> <dbl> <dbl>
#>  1  1.51     4     0  28.4
#>  2  1.51     4     1  28.9
#>  3  1.51     6     0  25.6
#>  4  1.51     6     1  26.2
#>  5  1.51     8     0  22.9
```

