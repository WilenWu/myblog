---
ID: 186c177fbbb58b650cc4beb19587e69d
title: R手册(Common)--tidyverse+tibble
tags: [R,tidyverse]
date: 2018-04-30 22:07:52
categories: [R,Common]
cover: /img/tidyverse.png
top_img: 
---

[tidyverse](https://www.tidyverse.org/)是一系列包的组合，构建了一套完整的数据分析生态链，提供了一套整洁的数据导入，分析和建模方法，刷新了R语言原有的数据科学体系。


<!-- more -->

# tidyverse


## Usage

- small, in-memory data ( <2Gb):   **tidyverse**
- larger data (10-100 Gb):   **data.table**
- Parallel computing :   need a system (like Hadoop or Spark)

## core tidyverse packages

```R
tidyverse_packages() #查看tidyverse内含的包
tidyverse_update() #更新tidyverse packages
```
`library(tidyverse)` will load the core tidyverse packages:

- ggplot2, for data visualisation.
- dplyr, for data manipulation.
- tidyr, for data tidying.
- readr, for data import.
- purrr, for functional programming.(FP)
- tibble, for tibbles, a modern re-imagining of data frames.

## Import

As well as readr, for reading flat files, the tidyverse includes:

- readxl for .xls and .xlsx sheets.
- feather, for sharing with Python and other languages.
- haven for SPSS, Stata, and SAS data.
- jsonlite for JSON.
- xml2 for XML.
- httr for web APIs.
- rvest for web scraping.
- DBI for relational databases.

## Wrangle

As well as tidyr, and dplyr, there are five packages designed to work with specific types of data:

- stringr for strings.
- lubridate for dates and date-times.
- forcats for categorical variables (factors).
- hms for time-of-day values.
- blob for storing blob (binary) data.

## Program

除了purrr处理函数式编程外，还有三个解决常规编程的包

- rlang 提供了R的核心语言和tidyverse的工具
- magrittr 管道函数
- glue 提供了base::paste()的更加丰富的版本 

## Model

- modelr, for modelling within a pipeline
- broom, for turning models into tidy data

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






