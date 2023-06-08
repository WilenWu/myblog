---
title: R手册(Common)--tibble
tags:
  - R
  - tidyverse
categories:
  - R
  - General
cover: /img/tibble-cover.png
top_img: /img/tibble-cover.png
abbrlink: 2b3aa78e
description: tibble 重塑了data.frame，可存储任意类型，包括list，tbl_df 等
date: 2018-04-30 22:07:52
---


# tibble

`tibble` 重塑了data.frame，可存储任意类型，包括list，tbl_df 等。

`tibble()`永远不会改变输入的类型（例如它永远不会将字符串转换为因子），永远不会改变变量的名称，并且它永远不会创建row.names()

## 创建tibble

```r
`as_tibble(x)` #从现有对象创建
`tibble() `    #使用列向量创建
`tribble()`    #逐行布局生成

# 示例
as_tibble(iris)

tibble(x = runif(10), y = x * 2)
tibble(x =list(1,2), y = tibble("a","b"))

tribble(~colA,~colB, 
        "a",  1,
        "b",  2)
```

取子集|说明
:---|:---
`[ `|返回data.frame
`[[`, `$` |返回子向量

## 函数

函数|说明
:---|:---
is_tibble(x)|判断
**增加行/列**|
add_column(.data, ..., .before = NULL, .after = NULL)|将列添加到数据框
add_row(.data, ..., .before = NULL, .after = NULL)|将行添加到数据框
**用于处理行名的工具**|
has_rownames(df)|
remove_rownames(df)|
rownames_to_column(df, var = "rowname")|
rowid_to_column(df, var = "rowid")|
column_to_rownames(df, var = "rowname"|
