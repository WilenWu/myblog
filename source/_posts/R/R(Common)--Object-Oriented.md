---
title: R手册(Common)--面向对象
tags:
  - R
  - 面向对象
categories:
  - R
  - Common
cover: /img/r6-cover.png
top_img: 
abbrlink: '515e9855'
date: 2018-05-01 17:37:07
description: 包括面向对象R6类和S4类
---

R 主要面向统计计算，似乎很少会用到面向对象的编程方法。但在统计计算中，在下列情形中使用面向对象的编程方法可以编程更有效率。

<!-- more -->

# 面向对象R6类

R 的面向对象 (OOP) 是基于泛型函数 (generic function) 的，而不是基于类层次结构

```r
library(R6)

# 创建R6类
R6Class(classname = NULL, public = list(), private = NULL,active = NULL, inherit = NULL, lock = TRUE, class = TRUE,portable = TRUE, parent_env = parent.frame())
```
参数列表：
> classname:定义类名。
public:定义公有成员，包括公有方法和属性。
privat:定义私有成员，包括私有方法和属性。
active :主动绑定的函数列表。
inherit:定义父类，继承关系。
lock:是否上锁，如果上锁则用于类变量存储的环境空间被锁定
class:是否把属性封装成对象，默认是封装，如果选择不封装，类中属性存存在一个环境空间中。
portable:是否为可移植类型，默认是可移植型类，类中成员访问需要用调用self和private对象。
parent_env:定义对象的父环境空间。

```r
`$new()` # 实例化函数
`$set()` # R6类的动态绑定
`print()` # 输出R6默认方法
```

# 面向对象S4类

## 自定义S4类

```r
myClass <-setClass(Class, slots=list(),prototype,contains=character(), validity) ) 
```
> 属性用@访问

参数列表
>Class: 定义类名
slots: 定义属性和属性类型，列表
prototype: 定义属性的默认值
contains: 定义父类，继承关系
validity: 定义属性的类型检查

## 实例化函数

```r
new(Class,...)
```

## S4的泛型函数

- 通过`setGeneric()`来定义接口
```r
setGeneric(name,def)
    # def指定了形式参数，并成为默认方法。
    # name通用函数的字符串名称
```

- 通过setMethod()来定义现实类
```r
setMethod(f,signature=character(),definition)
    # f：通用函数的字符串名称
    # signature：一些参数需要的类
    # definition：一个定义的函数
```



