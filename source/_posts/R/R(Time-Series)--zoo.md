---
title: R手册(Time Series)--zoo
tags:
  - R
  - 时间序列
  - 数据清洗
categories:
  - R
  - 'Time Series'
cover: /img/zoo-cover.png
top_img: /img/r-top-img.svg
description: zoo是对日期型数据进行清洗的R包
abbrlink: 3b11488b
date: 2018-05-28 16:55:01
---

<!-- more -->

# zoo

定义了规则和不规则时间序列S3类

## 基础对象

`zoo(x = NULL, order.by = index(x), frequency = NULL)`有序的时间序列对象
`zooreg(data, start = 1, end = numeric(), frequency = 1)`规则的的时间序列对象，继承zoo对象
`as.zoo(x)`把一个对象转型为zoo类型，泛型函数
`is.regular(x, strict = FALSE)`检查是否是规则的序列

## ggplot2扩展

`autoplot(object, geom = "line", facets, ...)`
`fortify(model, data, melt = FALSE, …)`数据操作

## 数据清洗

`read.zoo(file, format = "", index.column = 1, drop = TRUE)`
`write.zoo(x, file = "", index.name = "Index", row.names = FALSE, col.names = NULL)`
`coredata(x)`提取/替换zoo数据部分
`index(x)`提取/替换zoo索引部分
`window(x, index. = index(x), start = NULL, end = NULL)`按时间筛选数据
`merge()`合并多个zoo对象
`aggregate(x, by, FUN = sum)`分类计算
`lag(x, k = 1, na.pad = FALSE, ...)`计算步长
`diff(x, lag = 1, differences = 1)`计算分差
`rollapply(data, width, FUN)`对zoo数据的滚动处理

> rollmean, rollmax, rollmedian, rollsum,etc

`MATCH(x, table)`值匹配
`ORDER(x)`值排序，输出索引

## 缺失值处理

`na.fill(object, fill)`NA值的填充
`na.locf(object, na.rm = TRUE, fromLast=FALSE)`最近值替换NA
`na.aggregate(object,by = 1,FUN = mean,na.rm = FALSE)`计算统计值替换NA
`na.approx(object)`计算插值替换NA
`na.StructTS(object,na.rm=FALSE)`计算seasonalKalmanfilter替换NA
`na.trim(object)`过滤有NA的记录

## 显示控制

`yearqtr`以年季度显示时间
`yearmon`以年月显示时间

