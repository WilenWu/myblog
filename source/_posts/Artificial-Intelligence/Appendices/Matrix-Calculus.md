---
title: 矩阵求导
categories:
  - 'Artificial Intelligence'
  - 'Appendices
tags:
  - 数学
  - 矩阵
top_img: 
cover: /img/matrix-logo.png
katex: true
abbrlink: 99273c32
date: 2022-07-15 22:04:26
description:
---

在数学中， **矩阵微积分**是多元微积分的一种特殊表达，尤其是在矩阵空间上进行讨论的时候。它把单个函数对多个变量或者多元函数对单个变量的偏导数写成向量和矩阵的形式，使其可以被当成一个整体被处理。这使得要在多元函数寻找最大或最小值，又或是要为微分方程系统寻解的过程大幅简化。这里我们主要使用统计学和工程学中的惯用记法，而张量下标记法更常用于物理学中。

矩阵的导数类型：

| Types      | Scalar                                  | Vector                                         | Matrix                                  |
| ---------- | --------------------------------------- | ---------------------------------------------- | --------------------------------------- |
| **Scalar** | $\dfrac{\partial y}{\partial x}$        | $\dfrac{\partial\mathbf y}{\partial x}$        | $\dfrac{\partial\mathbf Y}{\partial x}$ |
| **Vector** | $\dfrac{\partial y}{\partial\mathbf x}$ | $\dfrac{\partial\mathbf y}{\partial\mathbf x}$ |                                         |
| **Matrix** | $\dfrac{\partial y}{\partial\mathbf X}$ |                                                |                                         |

# 向量求导

## scalar-by-vector

标量 $y=f(\mathbf x)$ 相对于向量 $\mathbf x=(x_1,x_2,\cdots,x_n)^T$ 的一阶导数称为==梯度==
$$
\nabla f(\mathbf x)=\frac{\partial f}{\partial\mathbf x}=\left(\frac{\partial f}{\partial x_i}\right)_{n\times 1}
$$
标量 $y=f(\mathbf x)$ 相对于向量 $\mathbf x=(x_1,x_2,\cdots,x_n)^T$ 的二阶导数称为 Hessian 矩阵
$$
\nabla^2f(\mathbf x)=\left(\frac{\partial^2 f}{\partial x_i\partial x_j}\right)_{n\times n}
$$

**求导法则**：

(1) 设 $a$ 为常数， 标量函数 $u=u(\mathbf x),v=v(\mathbf x)$ 

- $\dfrac{\partial au}{\partial\mathbf x}=a\dfrac{\partial u}{\partial\mathbf x}$ 
- $\dfrac{\partial (u+v)}{\partial\mathbf x}=\dfrac{\partial u}{\partial\mathbf x}+\dfrac{\partial v}{\partial\mathbf x}$ 
- $\dfrac{\partial (uv)}{\partial\mathbf x}=u\dfrac{\partial v}{\partial\mathbf x}+v\dfrac{\partial u}{\partial\mathbf x}$ 
- $\dfrac{\partial g(u)}{\partial\mathbf x}=\dfrac{\partial g}{\partial u}\dfrac{\partial u}{\partial\mathbf x}$

(2)  设 $\mathbf a$ 为常向量，$\mathbf A$ 为常矩阵，向量函数 $\mathbf u=\mathbf u(\mathbf x),\mathbf v=\mathbf v(\mathbf x)$ 

- $\dfrac{\partial \mathbf a^T\mathbf u}{\partial\mathbf x}=\dfrac{\partial \mathbf u^T\mathbf a}{\partial\mathbf x}=\dfrac{\partial\mathbf u}{\partial\mathbf x}\mathbf a$
- $\dfrac{\partial \mathbf u^T\mathbf v}{\partial\mathbf x}=\dfrac{\partial \mathbf v^T\mathbf u}{\partial\mathbf x}=\dfrac{\partial\mathbf u}{\partial\mathbf x}\mathbf v+\dfrac{\partial\mathbf v}{\partial\mathbf x}\mathbf u$
- $\dfrac{\partial \mathbf x^T\mathbf A\mathbf x}{\partial\mathbf x}=(\mathbf A+\mathbf A^T)\mathbf x$
- $\dfrac{\partial \mathbf x^T\mathbf x}{\partial\mathbf x}=\dfrac{\partial \|\mathbf x\|^2}{\partial\mathbf x}=2\mathbf x$
- $\dfrac{\partial \|\mathbf x-\mathbf a\|}{\partial\mathbf x}=\dfrac{\mathbf x-\mathbf a}{\|\mathbf x-\mathbf a\|}$

## vector-by-scalar

向量 $\mathbf y=(y_1,y_2,\cdots,y_n)^T$ 相对于标量 $x$ 的导数为向量
$$
\frac{\partial\mathbf y}{\partial x}=\left(\frac{\partial y_i}{\partial x}\right)_{1\times n}
$$

**求导法则**：设 $a$ 为常数，$\mathbf A$ 为常矩阵，向量函数 $\mathbf u=\mathbf u(\mathbf x),\mathbf v=\mathbf v(\mathbf x)$ 

- $\dfrac{\partial a\mathbf u}{\partial x}=a\dfrac{\partial \mathbf u}{\partial x}$
- $\dfrac{\partial \mathbf{Au}}{\partial x}=\dfrac{\partial \mathbf u}{\partial x}\mathbf A^T$
- $\dfrac{\partial (\mathbf{u+v})}{\partial x}=\dfrac{\partial\mathbf u}{\partial x}+\dfrac{\partial\mathbf v}{\partial x}$
- $\dfrac{\partial (\mathbf u^T\times \mathbf v)}{\partial x}=\dfrac{\partial\mathbf u}{\partial x}\times\mathbf v+\mathbf u^T\times(\dfrac{\partial\mathbf v}{\partial x})^T$
- $\dfrac{\partial\mathbf{g(u)}}{\partial x}=\dfrac{\partial\mathbf u}{\partial x}\dfrac{\partial\mathbf g}{\partial\mathbf u}$
- $\dfrac{\partial\mathbf{f(g(u))}}{\partial x}=\dfrac{\partial\mathbf u}{\partial x}\dfrac{\partial\mathbf g}{\partial\mathbf u}\dfrac{\partial\mathbf f}{\partial\mathbf g}$

## vector-by-vector

向量 $\mathbf y=(y_1,y_2,\cdots,y_n)^T$ 相对于向量 $\mathbf x=(x_1,x_2,\cdots,x_m)^T$ 的导数为 $m\times n$ 矩阵，第 $i$ 行 $j$ 列为
$$
\left(\frac{\partial\mathbf y}{\partial\mathbf x}\right)_{ij}=\frac{\partial y_j}{\partial x_i}
$$

**求导法则**：设 $a$ 为常数，$\mathbf A$ 为常矩阵，向量函数 $\mathbf u=\mathbf u(\mathbf x),\mathbf v=\mathbf v(\mathbf x)$ ，标量函数 $u=u(\mathbf x),v=v(\mathbf x)$ 

- $\dfrac{\partial \mathbf x}{\partial\mathbf x}=\mathbf I$
- $\dfrac{\partial a\mathbf u}{\partial\mathbf x}=a\dfrac{\partial \mathbf u}{\partial\mathbf x}$
- $\dfrac{\partial \mathbf{Au}}{\partial\mathbf x}=\dfrac{\partial\mathbf u}{\partial\mathbf x}\mathbf A^T$
- $\dfrac{\partial v\mathbf u}{\partial\mathbf x}=v\dfrac{\partial \mathbf u}{\partial\mathbf x}+\dfrac{\partial v}{\partial\mathbf x}\mathbf u^T$
- $\dfrac{\partial (\mathbf{u+v})}{\partial\mathbf x}=\dfrac{\partial\mathbf u}{\partial\mathbf x}+\dfrac{\partial\mathbf v}{\partial\mathbf x}$
- $\dfrac{\partial\mathbf{g(u)}}{\partial\mathbf x}=\dfrac{\partial \mathbf u}{\partial\mathbf x}\dfrac{\partial \mathbf g}{\partial\mathbf u}$

# 矩阵求导

## scalar-by-matrix

定义矩阵 $\mathbf X_{m\times n}$上的标量函数 $y$ 对矩阵的导数为矩阵

$$
\frac{\partial y}{\partial \mathbf X}=\left(\cfrac{\partial y}{\partial x_{ij}}\right)_{m\times n}
$$
定义矩阵上的重要的标量函数包括矩阵的==迹==和==行列式==。

**求导法则**：

(1) 设 $a$ 为常数，标量函数 $u=u(\mathbf X),v=v(\mathbf X)$ 

- $\dfrac{\partial au}{\partial\mathbf X}=a\dfrac{\partial u}{\partial\mathbf X}$ 
- $\dfrac{\partial (u+v)}{\partial\mathbf X}=\dfrac{\partial u}{\partial\mathbf X}+\dfrac{\partial v}{\partial\mathbf X}$ 
- $\dfrac{\partial (uv)}{\partial\mathbf X}=u\dfrac{\partial v}{\partial\mathbf X}+v\dfrac{\partial u}{\partial\mathbf X}$ 
- $\dfrac{\partial g(u)}{\partial\mathbf X}=\dfrac{\partial g}{\partial u}\dfrac{\partial u}{\partial\mathbf X}$

(2) 设 $\mathbf{a,b}$ 为常向量，$\mathbf A$ 为常矩阵

- $\dfrac{\partial\mathbf a^T\mathbf{Xb}}{\partial\mathbf X}=\mathbf{ab}^T$
- $\dfrac{\partial\text{tr}(\mathbf X)}{\partial\mathbf X}=\mathbf I$
- $\dfrac{\partial\text{tr}(\mathbf{AX)}}{\partial\mathbf X}=\dfrac{\partial\text{tr}(\mathbf{XA)}}{\partial\mathbf X}=\mathbf A^T$

## matrix-by-scalar

矩阵 $\mathbf Y=(y_{ij})_{m\times n}$ 对相对于标量 $x$ 的导数为矩阵 (仅分子布局)
$$
\frac{\partial\mathbf Y}{\partial x}=
\left(\cfrac{\partial y_{ij}}{\partial x}\right)_{m\times n}
$$





