---
ID: d5d0cbbdefbe44f680526a0de67493a1
title: R手册(Tidy+Transform)--forcats
tags: [R,tidyverse,数据清洗]
date: 2018-05-28 16:35:03
categories: [R,Tidy+Transform]
cover: /img/forcats.png
top_img: 
description: for factors
---

forcats: 分类变量数据处理

<!-- more -->

# forcats:  for factor

函数|说明
:---|:---
factor(x,levels,labels,ordered)|
as_factor(x)|
fct_expand(f, ...)|添加更多级别
fct_explicit_na(f, na_level = "(Missing)")|为缺失值提供因子水平
fct_c(...)|组合factor
fct_count(f, sort = FALSE)|统计各水平记录数
fct_unify(fs, levels = lvls_union(fs))|统一一系列factor中的水平
fct_drop(f, only)|删除未使用的水平
