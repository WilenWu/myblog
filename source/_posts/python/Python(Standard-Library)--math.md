---
title: Python手册(Standard Library)--数学模块
tags:
  - Python
  - 科学计算
categories:
  - Python
  - Python标准库
cover: /img/random.jpg
abbrlink: bb46152f
date: 2018-05-09 22:56:49
description: 基础包：math, random
---



<!-- more -->

# math

| math  | import math|
| :--------- | :--------- |
| math.truck(x)  | 取整|
| math.ceil(x)| 天花板|
| math.floor(x)  | 地板|
| math.exp(x) | |
| math.log(x,b=math.e) |计算以b为底的对数 |
| math.pow(x,y)  | x^y^  |
| math.sqrt(x)| |
| math.log10(x)  | |
|math.degrees|弧度转化为度||
|math.radians(x)|度转化为弧度|
|math.factorial(n)|n的阶乘
| math.nan | 缺失值|
| math.inf | |
| math.pi  | π|
| math.e| 自然常数  |

# random

| random| import random|
|:--------- | :--------- |
| random.choice(seq) | 随机挑选一个元素 |
| random.randrange(start,stop,step) | 从指定范围内，获取一个随机数 |
| random.random() | 随机生成一个实数[0,1)  |
| random.randint(a,b) | 随机生成一个整数[a,b]  |
| random.uniform(x, y)  | 随机生成一个实数[x,y]  |
| random.seed()| |
| random.shuffle(lst)| 随机排序  |
