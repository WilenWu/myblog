---
title: R手册(Common)--data.table
tags:
  - R
categories:
  - R
  - Common
cover: /img/rdatatable-cover.png
top_img: /img/r-top-img.svg
abbrlink: 4de35808
date: 2018-05-01 17:32:24
description: 简短的代码实现快速操作数据
---

R语言data.table包是自带包data.frame的升级版，用于数据框格式数据的处理，最大的特点快。
包括两个方面，一方面是写的快，代码简洁，只要一行命令就可以完成诸多任务。
另一方面是处理快，内部处理的步骤进行了程序上的优化，使用多线程，甚至很多函数是使用C写的，大大加快数据运行速度。因此，在对大数据处理上，使用data.table无疑具有极高的效率。

<!-- more -->

# 结构

```r
DT[ i,  j,  by  ,with=TRUE] + extra arguments
    |   |    --> grouped by what?
    |    ---> what to do?
     ----> on which rows?
```
> with=FALSE时，j为列名字符向量或者数字向量

**用法**

```R
DT[,c("a","b","c"),with=FALSE]
X[, sum(a), by=c:f][order(b)]
DT[, .(sa=sum(a)), by=.(x=x>0, y)]  # 列命名，表达式分组命名
X[Y, .(a, b), on="c"]               # 内连接Y$c==X$c,选择列'X$a'and‘X$b'
X[Y, .(a, i.a), on="c"]             # 内连接Y$c==X$c,选择列'X$a'and‘Y$a'
```

# 特殊符号

```R
DT[, .N, by=x]     # 分组计数
DT[, .GRP, by=x]   # 组计数器(1 for the 1st group, 2 for the 2nd, etc)
```

# 更新data.table

```R
# data.table更新符号 :=
`DT[i, LHS := RHS]`
`DT[i, c("LHS1", "LHS2") := list(RHS1, RHS2)] `# 批量新建列

# 示例
DT[b > 4, b := d * 2L]
DT[, c("price","off") := tstrsplit(Discount_rate, ":", fixed=TRUE,fill = NA)]
```

# 运算符

```R
x %chin% table   # base:: %in%
vector %like% pattern
x %between% y
```

# 其他函数

函数|说明
:---|:---
fread(input, sep="auto",colClasses=NULL, nrows=-1L,select=NULL) |输入
fwrite(x, file = "", append = FALSE)|输出
between(x, lower, upper, incbounds=TRUE)|incbounds是否包含边界
duplicated(x,by)|返回重复的逻辑值
unique(x,by)|返回唯一值
melt(data,id,measure) |id:要融合的变量矢量<br>measure:要测量的变量矢量
dcast(data,formula,<br>fun.aggregate,<br>fill = NULL,value.var)|rowvar1+rowvar2+...~colvar1+colvar2+..<br>聚合函数<br>填充,value变量
merge(x,y, by=c("id",...),all=FALSE)|by,by.x,by.y：连接键<br>all,all.x,all.y：inner(outer)/left/right join



