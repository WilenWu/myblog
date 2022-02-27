---
title: R手册(Common)--R语言入门
tags:
  - R
categories:
  - R
  - Common
cover: /img/r-base-cover.png
description: 包括Rstudio介绍，R的数据结构和基础语法等
abbrlink: 81860df8
date: 2018-05-26 19:29:50
top_img: /img/r-top-img.svg
---

说明：本节中大部分内容摘自书籍《R语言实战（第2版）》

<!-- more -->

# RStudio: Take control of your R code

![](https://gitee.com/WilenWu/images/raw/master/common/rstudio.png)

RStudio是R的集成开发环境（IDE）。它包括一个控制台，支持直接代码执行的语法高亮编辑器，以及绘图，历史记录，调试和工作区管理工具。

RStudio可用于开源和商业版本，并可在桌面（Windows，Mac和Linux）或连接到RStudio Server或RStudio Server Pro（Debian / Ubuntu，RedHat / CentOS和SUSE Linux）的浏览器上运行。

[在 RStudio 下使用 R 的基本功能](https://blog.csdn.net/qingchongxinshuru/article/details/52003704)
[RStudio参考卡片](https://github.com/rstudio/cheatsheets/raw/master/rstudio-ide.pdf)

# 数据处理一般流程

![](https://gitee.com/WilenWu/images/raw/master/common/Rflow.png)

# R 数据结构

R拥有许多用于存储数据的对象类型，包括标量、向量、矩阵、数组、数据框和列表。
![](https://gitee.com/WilenWu/images/raw/master/common/r-dataframe.png)

**1. 向量**
向量是用于存储数值型、字符型或逻辑型数据的一维数组。
```r
a <- c(1, 2, 5, 3, 6, -2, 4) 
b <- c("one", "two", "three")
c <- c(TRUE, TRUE, TRUE, FALSE, TRUE, FALSE) 

d<-1:10 #生成1-10整数向量
```
**2. 矩阵**
矩阵是一个二维数组，只是每个元素都拥有相同的模式（数值型、字符型或逻辑型）。
```r
myymatrix <- matrix(vector, nrow=number_of_rows,ncol=number_of_columns, byrow=logical_value, 
  dimnames=list(char_vector_rownames, char_vector_colnames)) 
```
**3. 数组**
数组（array）与矩阵类似，但是维度可以大于2。
```r
myarray <- array(vector, dimensions, dimnames) 
```
**4. 数据框**
由于不同的列可以包含不同模式（数值型、字符型等）的数据，数据框的概念较矩阵来说更 为一般。
```r
mydata <- data.frame(col1, col2, col3,...) 
```
**5. 因子**
变量可归结为名义型、有序型或连续型变量。名义型变量是没有顺序之分的类别 变量。有序型变量表示一种顺序关系，而非数量关系。
类别（名义型）变量和有序类别（有序型）变量在R中称为因子（factor）。因子在R中非 常重要，因为它决定了数据的分析方式以及如何进行视觉呈现。
```r
factor(x = character(), levels, labels = levels,
       exclude = NA, ordered = is.ordered(x), nmax = NA)
```

**6. 列表**

列表（list）是R的数据类型中为复杂的一种。一般来说，列表就是一些对象（或成分， component）的有序集合。列表允许你整合若干（可能无关的）对象到单个对象名下。
```r
mylist <- list(object1, object2, ...)
```
**7. 日期值**
```r
as.Date(x, "input_format")
```
日期值的默认输入格式为`yyyy-mm-dd`。
```r
mydates <- as.Date(c("2007-06-22", "2004-02-13"),"%Y-%m-%d") 
```
format符号| 含义 |示例
:------|:------|:------
 %d |数字表示的日期（0~31） |01~31 
 %a |缩写的星期名| Mon 
 %A |非缩写星期名 |Monday 
 %m |月份（00~12）| 00~12 
 %b |缩写的月份 |Jan 
 %B |非缩写月份 |January 
 %y |两位数的年份 |07 
 %Y |四位数的年份 |2007 

**8. 索引和切片**
以data.frame为例
```r
# 以data.frame为例
df<-data.frame(x=1:10,y=rnorm(10, mean=0, sd=1))
```

用法|说明
:------|:------
`df[3,5]`|
`df['x']`|返回x列，data.frame类型
`df$x , df[[x]]`|返回向量类型
`df[c(1,3,5), ]`|
`df[-(1:5),]`|除了1-5列（R 中`-`是排除的意思）
`df[x>3,]`|逻辑索引
`df[x %in% c(3,5))]	`|


# R 运算符

数学运算|描述
:------|:------
`+, -, *, /` |
`^, **`| 求幂 
`x%%y `|求余（x mod y）。5%%2 的结果为 1 
`x%/%y` |整数除法。5%/%2 的结果为 2 

| 关系    | 逻辑运算 |
| :------ | :------- |
| `< ,<=` | `!x`     |
| `>,>=`  | `x|y`    |
| `==,!=` | `x & y`  |

# 概率函数 

 在R中，概率函数统一格式为：`[dpqr]distribution_abbreviation()`
其中第一个字母表示其所指分布的某一方面

首字母|说明|主参数|返回值|示例
:---|:---|:---|:---|:---|
p|分布函数(distribution function)|分位数x值|概率值F(x)=P{X<=x}|pnorm(q)
d|密度函数(density)|分位数x值|密度值f(x)=dF/dx|dnorm(x)
q|分位数函数(quantile function)|概率P值|分位数x|qnorm(p)
r|生成随机数|随机数n|随机向量密度值|rnorm(n)

**概率分布**

分布名称|缩写|分布名称|缩写|
:---|:---|:---|:---|
Beta 分布|beta|Logistic 分布|logis|
二项分布|binom|多项分布|multinom|
柯西分布|cauchy|负二项分布|nbinom|
(非中心)卡方分布|chsiq|正态分布|norm|
指数分布|exp|泊松分布|pois|
F分布|f|Wilcoxon 符号秩分布|signrank|
Gamma分布|gamma|t分布|t|
几何分布|geom|均匀分布|unif|
超几何分布|hyper|Weibull 分布|weibull|
对数正态分布|lnorm|Wilcoxon 秩和分布|wilcox|
```r
runif(n, min=0, max=1) #均匀分布
rnorm(n, mean=0, sd=1) #正态分布
```

# 控制语句与循环语句

**控制语句**
```r
if(cond) expr
if(cond) cons.expr else alt.expr
if...else if...else
ifelse(test, yes, no)

switch(expr, ...)
  expr：可以为一个数字或字符串
  ...:为一系列选择项列
  若expr为字符串，...列表项为name_i=expr_i
  若expr为数字，...列表项按顺序输出
```
**循环语句**
```
for(var in seq) expr
while(cond) expr
repeat expr
```
**循环语句关键字**

`break`：跳出整个循环语句
`continue`: 跳出本次循环
`next`：跳出本次loop


# 自定义函数

```r
function( arglist ) {
  expr
  return(object) }
```
>... (dot-dot-dot)：此特殊参数可以捕获任何数量的参数

example:
```r
>>>"%s*%" <- function(x,y) rep(x,y) #自定义二元符号
>>>"a"%s*%5
>>>"aaaaa"
```


# 调试

函数|用法
:-----|:-----
stop(message)|停止执行当前表达式返回message(常与if配合检查自定义函数参数)
stopifnot(logit1,logit2,logit3…)|检查每个参数为TRUE，否则停止执行当前表达式返回message


# 拟合线性模型formula

在R中，拟合线性模型基本的函数就是lm()，格式为： 
```r
myfit <- lm(formula, data) 
```
> formula指要拟合的模型形式，
> data是一个数据框，包含了用于拟合模型的数据。
> 结果对象（本例中是myfit）存储在一个列表中，包含了所拟合模型的大量信息。

表达式（formula） 形式如下： `Y ~ X1 + X2 + ... + Xk `
`~`左边为响应变量，右边为各个预测变量，预测变量之间用`+`符号分隔。

formula常用符号|说明|示例
:---|:---|:---|
~|分隔符，左边为响应变量，右边为解释变量
+|预测变量分隔符|y~x+y
:|预测变量交互项|y~x+y+x:y
\*|包含所有交互项的简洁方式|代码y~ x * z可展开为y ~ x + z + x:z
\^|交互项的最高次数|代码 y ~ (x + z + w)^2 可展开为 y ~ x + z + w + x:z + x:w + z:w
.(dot)|除因变量外的所有变量|y~.
-|移除一项|y~x*z*w–x:z:w可展开为 y ~ (x + z + w)^2
-1|移除截距|y~x-1
I()|算术|y~x+I(x^2)
function|数学函数|log(y) ~ x + z + w

# R语言可视化

除了基础图形，grid、lattice和ggplot2软件包也提供了图形系统，它们克服了R基础图 形系统的低效性，大大扩展了R的绘图能力。

- grid图形系统可以很容易地控制图形基础单元，给予编程者创作图形的极大灵活性。
- lattice包通过一维、二维或三维条件绘图，即所谓的网格图形（trellis graph）来对多元变量关 系进行直观展示。
- ggplot2包极大地扩展 了R绘图的范畴，提高了图形的质量。

系统 |基础安装中是否包含| 是否需要显式加载
:------|:------|:------
 base |是| 否 
 grid |是 |是 
 lattice |是| 是 
 ggplot2 |否 |是 


# data.table 和 tidyverse

**data.table**
R语言data.table包是自带包data.frame的升级版，用于数据框格式数据的处理，最大的特点快。
包括两个方面，一方面是写的快，代码简洁，只要一行命令就可以完成诸多任务。
另一方面是处理快，内部处理的步骤进行了程序上的优化，使用多线程，甚至很多函数是使用C写的，大大加快数据运行速度。因此，在对大数据处理上，使用data.table无疑具有极高的效率。

**tidyverse**
<img src="https://gitee.com/WilenWu/images/raw/master/common/Hadley.png" alt="hadley" style="zoom: 33%;" />

tidyverse是一系列包的组合，构建了一套完整的数据分析生态链，提供了一套整洁的数据导入，分析和建模方法，刷新了R语言原有的数据科学体系。

作者 [Hadley Wickham][hd] 在R语言数据科学领域是个鼎鼎有名的大人物，被称为**一个改变了R的人**。

[hd]: http://hadley.nz/



