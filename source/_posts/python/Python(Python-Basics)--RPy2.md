---
ID: 870f196d599cfaa4bde163b80704c519
title: Python手册(Python Basics)--RPy2
date: 2018-12-23 22:05:50
categories: [Python,Python基础]
tags: [Python]
cover: /img/rpy2-logo.png
top_img: /img/python-logo-wide.svg
description: Python 通过rpy2调用 R语言
---

# 概述

[Python](http://www.python.org/)是一种流行的通用脚本语言，而[R](http://www.r-project.org/)(S/Splus语言的开源实现)是一种主要用于数据分析，统计和图形的脚本语言。[rpy2](https://rpy2.github.io/doc/latest/html/overview.html) 是一个嵌入在Python进程中的R接口.

<!-- more -->

```python
pip install rpy2
```

子模块

- [rpy2.rinterface](https://rpy2.github.io/doc/latest/html/rinterface.html#module-rpy2.rinterface "rpy2.rinterface：与R的低级接口（Unix，Windows）"): Low-level interface to R，Close to R’s C-level API.当速度和灵活性最重要时
- [rpy2.robjects](https://rpy2.github.io/doc/latest/html/robjects.html#module-rpy2.robjects): High-level interface,Based on the previous one.当易用性最重要时
- [rpy2.interactive](https://rpy2.github.io/doc/latest/html/interactive.html#module-rpy2.interactive): High-level interface,Largely based on rpy2.robjects.当注重交互式工作时
- rpy2.rpy_classic: High-level interface，这是出于兼容性原因提供的，也是为了便于RPy-1.x迁移到RPy2。
- [rpy2.rlike](https://rpy2.github.io/doc/latest/html/rlike.html#module-rpy2.rlike): 数据结构和函数，以模仿纯Python中的一些R特性和特性（无嵌入式R过程）。


# IPython魔法接口(rmagic)

**Rmagic**: 用于在ipython中与R进行交互式工作的Magic命令。％R和%% R分别是单行和单元格(cell)的magic命令。
**Usage**: 要运行magics below,，先执行 `%load_ext rpy2.ipython`

- **%R**
   ```python
   %R [-i INPUT] [-o OUTPUT] [-n] [-w WIDTH] [-h HEIGHT] [-p POINTSIZE]
   
   [-b BG] [–noisolation] [-u {px,in,cm,mm}] [-r RES] [-c CONVERTER] [code [code …]]
   ```

- **%% R**: 
   ```python
   In [10]: %%R
      ....: Y = c(2,4,3,9)
      ....: summary(lm(Y~X))
   ```
   
   通过用分号连接来执行多个R表达式
   ```python
   In [9]: %R X=c(1,4,5,7); sd(X); mean(X)
   Out[9]: array([ 4.25])
   ```
   
   默认情况下，在 `%% R` 模式下，没有任何内容返回到python。对象可以通过行中的`-i -o`标志在rpy2和python之间来回传递：
   ```python
   In [14]: Z = np.array([1,4,5,10])
   
   In [15]: %R -i Z mean(Z)
   Out[15]: array([ 5.])
   
   In [16]: %R -o W W=Z*mean(Z)
   Out[16]: array([  5.,  20.,  25.,  50.])
   
   In [17]: W
   Out[17]: array([  5.,  20.,  25.,  50.])
   ```

- **%Rpush**  and **%Rpull**

   `%Rpush [input]` : pushes variables from python to rpy2
   `%Rpull [outputs outputs …]` : pulls variables from python to rpy2
   
   ```python
   In [8]: X = np.array([4.5,6.3,7.9])
   In [10]: %Rpush X
   In [11]: %R mean(X)
   Out[11]: array([ 6.23333333])
   
   In [18]: _ = %R x = c(3,4,6.7); y = c(4,6,7); z = c('a',3,4)
   In [19]: %Rpull x  y z
   In [20]: x
   Out[20]: array([ 3. ,  4. ,  6.7])
   ```

- **%Rget**

   `%Rget output` 从rpy2返回一个对象，可能是一个结构化数组（如果可能的话）。与Rpull类似，除了只接受一个参数并返回值而不是推送到self.shell.user_ns
   
   ```python
   In [3]: dtype=[('x', '<i4'), ('y', '<f8'), ('z', '|S1')]
   In [4]: datapy = np.array([(1, 2.9, 'a'), (2, 3.5, 'b'), (3, 2.1, 'c'), (4, 5, 'e')], dtype=dtype)
   
   In [5]: %R -i datapy
   In [6]: %Rget datapy
   Out[6]:
   array([['1', '2', '3', '4'],
          ['2', '3', '2', '5'],
          ['a', 'b', 'c', 'e']],
         dtype='|S1')
   ```



