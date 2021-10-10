---
ID: 19e03f0c094e5f23c7b1938746efa7a7
title: R手册(Parallel Computing)--foreach
tags: [R,并行运算]
date: 2018-05-04 00:34:33
categories: [R,并行运算]
---

foreach为R语言并行运算包

<!-- more -->


- 后端支持

```r
library(doParallel) #为foreah包提供一个并行的后端
n_cores<-detectCores(logical=FALSE)  #获得物理内核数
registerDoParallel(cores=n_cores-1)  #注册集群
```
`stopImplicitCluster()`  关闭集群

-  foreach

```r
foreach(...,     #定义ex应执行的次数
  .combine,      #用于处理任务结果的函数（字符 c, cbind, rbind, +, *, 自定义函数），默认返回list
 .inorder=TRUE, 
 .multicombine=FALSE, #判断.combine 函数是否可以接受两个以上参数
 .packages=NULL)      #任务所依赖的包（字符向量）
```

-  `%dopar% ex` 执行语句集

```
example:并行1200棵树的随机森林
foreach(ntree=rep(200,6), .combine=combine, .packages="randomForest")
%dopar%  randomForest(x, y, ntree=ntree)
```




