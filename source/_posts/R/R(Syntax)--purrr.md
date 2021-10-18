---
ID: c7be577142716ed4cbac9d54f1022765
title: R手册(Syntax)--purrr
date: 2018-05-02 00:46:18
tags: [R,tidyverse]
categories: [R,语法]
cover: /img/purrr.png
top_img: 
description: A functional programming(FP) toolkit for R
---

**purrr** : A functional programming(FP) toolkit for R

<!-- more -->

# Apply Functions

`map(.x, .f, ...)` Apply a function to **each element** of **a list** or **vector**
`map2(.x,.y,.f,…)`  Apply a function to **pairs of elements** from **two lists**
`pmap(.x, .f, ...)` Apply a function to **groups of elements** from **list of lists**
`imap(.x, .f, ...)`  Apply a function to **each element** of a **vector, and its index**
`invoke_map(.f, .x = list(NULL) )`   调用不同的函数
`walk(.x, .f), walk2(), pwalk()`  并行处理

`map_if(.x, .p, .f, ...)`   `map_at(.x, .at, .f, ...)`  其他系列同map


**Parameters:** 
> `.f` : a function, formula, or atomic vector
>  `…` :  其他参数（`.x`为没有被`...`指定的第一个参数）

**`.f` 参数：**
> **function**：`as.character %>% map(iris)`
> 
> **formula**：`~ .x `becomes `function(x)`
> `map(l, ~ 2 +.x) `
> `~ .x .y` becomes `function(.x, .y) `, e.g.
> **character**：
> `map(x,"y")`becomes `x[["y"]]`
> `map(x,c("a","b"))` becomes `x[["a"]][["b"]]`
>  **integer**：
>  `map(x,2)` becomes `x[[2]]`

**Output**

function |returns
------|-------
map |list
map_chr |character vector
map_dbl |double (numeric) vector
map_df  |data frame(auto)
map_dfc |data frame (column bind)
map_dfr |data frame (row bind)
map_int |integer vector
map_lgl |logical vector
walk |返回指定的类型
> map2, pmap, imap, invoke_map 系列函数同map


Examples:
```r
1:10 %>%
 map(rnorm, n = 10) %>%
 map_dbl(mean)
 
tribble(mean=c(5,10,3),sd=c(1,5,3),n=c(2,3,8)%>%
  pmap(rnorm) #生成三组不同参数的正态分布

df <- tibble(
  f = c("runif", "rpois", "rnorm"),
  params = list(
    list(n = 10),
    list(n = 5, lambda = 10),
    list(n = 10, mean = -3, sd = 10))
)
invoke_map(df$f, df$params)

`list(1,"a",3)%>% walk(print)` #并行打印
```

# Work with Lists

filter lists|说明
:---|---
pluck(.x, ..., .default=NULL) |按名字或者索引筛选`x, pluck(x,"b")`
keep(.x, .p, …)<br>discard(.x, .p, …) |逻辑值筛选`keep(x, is.na)`<br>keep的否定结果`discard(x, is.na)`
compact(.x, .p = identity)|删除空元素
head_while(.x, .p, …)<br>tail_while|找到所有头部/尾部满足匹配的值

logic|说明
:---|---
every(.x, .p, …) <br>some(.x, .p, …)|列表中的每一个/某些元素是否满足要求(返回一个FALSE/TRUE)<br>`mtcars %>% some(is_numeric)`
has_element(.x, .y) |列表包含一个元素`has_element(x, "foo")`
detect(.x, .f, ..., .right=FALSE,.p)<br>detect_index(.x, .f, ..., .right= FALSE, .p) |返回第一个判断为TRUE的元素`detect(x, is.character)`<br>返回索引
vec_depth(x) |Return depth(number of levels of indexes)

reshape lists|说明
:---|---
flatten(.x)|将列表降低一个维度
flatten_lgl(), flatten_int(), flatten_dbl(), flatten_chr()<br>flatten_df,flatten_dfr(),flatten_dfc()|返回向量<br>返回data.frame
transpose(.list, .names = NULL)|转置一个列表
simplify(.x, .type = NULL)|列表强制成向量
simplify_all(.x, .type = NULL)|

combine|说明
:---|---
append(x, values, afer =length(x))|末尾追加`append(x, list(d = 1))`
prepend(x, values, before =1) |起始追加`prepend(x, list(d = 1))`
splice(…) |合并到一个列表


# Reduce Lists

递归|说明
:---|---
**reduce**(.x,.f,...init)|将.x的子元素递归，返回单个值<br>`.x`列表或原子矢量，<br>`init`开始积累的第一个值
reduce_right(.x,.f,...init)|从右往左递归
reduce2(.x, .y, .f, ..., .init)<br>reduce2_right(.x, .y, .f, ..., .init)|三参数递归
**accumulate**(.x, .f, ..., .init)|将.x的子元素递归，返回中间值
accumulate_right(.x, .f, ..., .init)|从右往左递归

```
list(data1,data2,data3) %>% 
   reduce(left_join,by = c("first","last"))
```

# Debug

调试|说明
:------|:------
`safely(.f, otherwise = NULL, quiet = TRUE)`|return list of results and errors
`quietly(.f)`|return list of results, output, messages, warnings. 
`possibly(.f, otherwise, quiet = TRUE)`|continue with default value (instead of error) and return list of results,error
`base::stop(message)` |stop and return message
`base::stopifnot(logit1,logit2,logit3…)`|检查每个参数为TRUE，否则停止执行当前表达式返回message


# Work with Tibbles

**Nested Data**

![](https://gitee.com/WilenWu/images/raw/master/common/nested_data.png)

`tidyr::nest(data, ..., .key = data)`
For grouped data, moves groups into cells as data frames.

```r
n_iris <- iris %>% group_by(Species) %>% nest()
```
`tidyr::unnest(data, ..., .drop = NA, .id=NULL, .sep=NULL)`
Unnests a nested data frame.

```r
n_iris %>% unnest()
```

**List Column Workflow**

```r
# step 1 Make a list column
n_iris <- iris %>%
 group_by(Species) %>%
 nest()

# step 2 Work with list columns
mod_fun <- function(df)
 lm(Sepal.Length ~ ., data = df)
m_iris <- n_iris %>%
 mutate(model = map(data, mod_fun))
 
# step 3 Simplify the list column
b_fun <- function(mod)
 coefficients(mod)[[1]]
m_iris %>% transmute(Species,
 beta = map_dbl(model, b_fun))
```
