---
title: 矩阵微积分
tags:
  - 数学
  - 矩阵
categories:
  - 数学
  - 高等代数
top_img: '#66CCFF'
katex: true
abbrlink: 99273c32
date: 2022-07-15 22:04:26
updated:
description:
cover:
---

在数学中， **矩阵微积分**是多元微积分的一种特殊表达，尤其是在矩阵空间上进行讨论的时候。它把单个函数对多个变量或者多元函数对单个变量的偏导数写成向量和矩阵的形式，使其可以被当成一个整体被处理。这使得要在多元函数寻找最大或最小值，又或是要为微分方程系统寻解的过程大幅简化。这里我们主要使用统计学和工程学中的惯用记法，而张量下标记法更常用于物理学中。

# 函数矩阵

我们引入下面的**函数矩阵** 
$$
\mathbf{Y}(\mathbf x)=\begin{pmatrix}
y_{11}(\mathbf x) & y_{12}(\mathbf x) & \cdots & y_{1n}(\mathbf x) \\ 
y_{21}(\mathbf x) & y_{22}(\mathbf x) & \cdots & y_{2n}(\mathbf x) \\ 
\vdots &\vdots &\ddots &\vdots \\ 
y_{m1}(\mathbf x) & y_{m2}(\mathbf x) & \cdots & y_{mn}(\mathbf x) \\ 
\end{pmatrix}
$$
$\mathbf{Y}(\mathbf x)$是$m\times n$函数矩阵，它的每一个元素 $y_{ij}(\mathbf x)=y_{ij}(x_1,x_2,\cdots,x_p)$是定义域 $D$上的函数。关于矩阵的代数运算，如相加、相乘与纯量相乘等性质对于以函数作为元素的矩阵同样成立。

函数矩阵的连续、微分的定义如下：如果函数矩阵$\mathbf{Y}(\mathbf x)$的每一个元素都是定义域 $D$上的连续函数，则称$\mathbf{Y}(\mathbf x)$在定义域 $D$上连续。如果函数矩阵$\mathbf{Y}(\mathbf x)$的每一个元素都是定义域 $D$上的可微函数，则称$\mathbf{Y}(\mathbf x)$在定义域 $D$上可微。

# 向量求导

由于向量可看成仅有一列的矩阵，最简单的矩阵求导为向量求导。

## 向量对标量求导

向量 $\mathbf y=\begin{pmatrix}y_1&y_2&\cdots&y_m\end{pmatrix}^T$ 关于标量$x$的导数可以（用分母记法）写成
$$
\frac{\partial\mathbf y}{\partial x}=\begin{pmatrix}
\cfrac{\partial y_1}{\partial x}
&\cfrac{\partial y_2}{\partial x}
&\cdots
&\cfrac{\partial y_m}{\partial x}
\end{pmatrix}
$$
在向量微积分中，向量 $\mathbf {y}$ 关于标量 $x$的导数也被称为向量$\mathbf {y}$ 的**切向量**。

**例子** 简单的样例包括欧式空间中的速度向量，它是位移向量（看作关于时间的函数）的切向量。更进一步而言， 加速度是速度的切向量。

## 标量对向量求导

**梯度**（多元函数的一阶导数）：标量函数 $f(\mathbf x)$ 对自变量 $\mathbf x=(x_1,x_2,\cdots,x_n)^T$ 各分量的偏导数（用分母记法）
$$
\mathrm{grad}f(\mathbf x)=\nabla f(\mathbf x)=
\frac{\partial f}{\partial\mathbf x}=
(\cfrac{\partial f}{\partial x_1},
\cfrac{\partial f}{\partial x_2},
\cdots,
\cfrac{\partial f}{\partial x_n})^T
$$

称为标量函数 $f(\mathbf x)$ 在 $\mathbf x$ 处的一阶导数或梯度。在物理学中，电场是电势的负梯度向量。

**Hessian（海塞）矩阵**（多元函数的二阶导数）：标量函数 $f(\mathbf x)$ 对自变量 $\mathbf x=(x_1,x_2,\cdots,x_n)^T$ 各分量的二阶偏导数（用分母记法）
$$
\mathrm H(\mathbf x)=\nabla^2f(\mathbf x)=
\begin{pmatrix}
\cfrac{\partial^2 f}{\partial x_1^2}&\cfrac{\partial^2 f}{\partial x_1\partial x_2}&\cdots&\cfrac{\partial^2 f}{\partial x_1\partial x_n} \\
\cfrac{\partial^2 f}{\partial x_2\partial x_1}&\cfrac{\partial^2 f}{\partial x_2^2}&\cdots&\cfrac{\partial^2 f}{\partial x_2\partial x_n} \\
\vdots &\vdots &\ddots &\vdots \\ 
\cfrac{\partial^2 f}{\partial x_n\partial x_1}&\cfrac{\partial^2 f}{\partial x_n\partial x_2}&\cdots&\cfrac{\partial^2 f}{\partial x_n^2} \\
\end{pmatrix}
$$

称为标量函数 $f(\mathbf x)$ 在 $\mathbf x$ 处的二阶导数或Hessian（海塞）矩阵。

## 向量对向量求导

前面两种情况可以看作是向量对向量求导在其中一个是一维向量情况下的特例。类似地我们将会发现有关矩阵的求导可被以一种类似的方式化归为向量求导。

向量函数 (分量为函数的向量) $\mathbf y=\begin{pmatrix}y_1&y_2&\cdots&y_m\end{pmatrix}^T$对输入向量$\mathbf x=\begin{pmatrix}x_1&x_2&\cdots&x_n\end{pmatrix}^T$的导数，可以（用分母记法) 写作
$$
\frac{\partial\mathbf y}{\partial\mathbf x}=\begin{pmatrix}
\cfrac{\partial y_1}{\partial x_1}&\cfrac{\partial y_2}{\partial x_1}&\cdots&\cfrac{\partial y_m}{\partial x_1} \\
\cfrac{\partial y_1}{\partial x_2}&\cfrac{\partial y_2}{\partial x_2}&\cdots&\cfrac{\partial y_m}{\partial x_2} \\
\vdots &\vdots &\ddots &\vdots \\ 
\cfrac{\partial y_1}{\partial x_n}&\cfrac{\partial y_2}{\partial x_n}&\cdots&\cfrac{\partial y_m}{\partial x_n} 
\end{pmatrix}
$$
## 运算性质

矩阵 $\mathbf A$ 不是向量 $\mathbf x$的函数，标量函数 $v=v(\mathbf x)$ ，向量函数 $\mathbf{u=u(x),v=v(x)}$

(1) $\cfrac{\partial\mathbf{Ax}}{\partial\mathbf x}=\mathbf A^T$  

(2) $\cfrac{\partial\mathbf x^T \mathbf A}{\partial\mathbf x}=\mathbf A$  

(3) $\cfrac{\partial v\mathbf u}{\partial\mathbf x}=v\cfrac{\partial \mathbf u}{\partial\mathbf x}+\cfrac{\partial \mathbf u}{\partial\mathbf x}\mathbf u^T$

(4) $\cfrac{\partial\mathbf{Au}}{\partial\mathbf x}=\cfrac{\partial \mathbf u}{\partial\mathbf x}\mathbf A^T$

(5) $\cfrac{\partial\mathbf{(u+v)}}{\partial\mathbf x}=\cfrac{\partial\mathbf u}{\partial\mathbf x}+\cfrac{\partial\mathbf v}{\partial\mathbf x}$

(6) $\cfrac{\partial\mathbf{g(u)}}{\partial\mathbf x}=\cfrac{\partial\mathbf u}{\partial\mathbf x}\cfrac{\partial\mathbf{g(u)}}{\partial\mathbf u}$

# 矩阵求导

有两种类型的矩阵求导可以被写成相同大小的矩阵：矩阵对标量求导和标量对矩阵求导。它们在解决应用数学的许多领域常见的最小化问题中十分有用。类比于向量求导，相应的概念有**切矩阵**和**梯度矩阵**。

## 矩阵对标量求导

矩阵函数$\mathbf Y$对标量$x$的导数被称为**切矩阵**，(用分母记法）可写成：
$$
\frac{\partial\mathbf Y}{\partial x}=\begin{pmatrix}
\cfrac{\partial y_{11}}{\partial x}&\cfrac{\partial y_{21}}{\partial x}&\cdots&\cfrac{\partial y_{m1}}{\partial x} \\
\cfrac{\partial y_{12}}{\partial x}&\cfrac{\partial y_{22}}{\partial x}&\cdots&\cfrac{\partial y_{m2}}{\partial x} \\
\vdots &\vdots &\ddots &\vdots \\ 
\cfrac{\partial y_{1n}}{\partial x}&\cfrac{\partial y_{2n}}{\partial x}&\cdots&\cfrac{\partial y_{nm}}{\partial x} 
\end{pmatrix}
$$

## 标量对矩阵求导

定义在元素是独立变量的矩阵$\mathbf X_{p\times q}$上的标量函数$y$对$\mathbf X$的导数可以（用分母记法）写作

$$
\frac{\partial y}{\partial \mathbf X}=\begin{pmatrix}
\cfrac{\partial y}{\partial x_{11}}&\cfrac{\partial y}{\partial x_{12}}&\cdots&\cfrac{\partial y}{\partial x_{1q}} \\
\cfrac{\partial y}{\partial x_{21}}&\cfrac{\partial y}{\partial x_{22}}&\cdots&\cfrac{\partial y}{\partial x_{2q}} \\
\vdots &\vdots &\ddots &\vdots \\ 
\cfrac{\partial y}{\partial x_{p1}}&\cfrac{\partial y}{\partial x_{p2}}&\cdots&\cfrac{\partial y}{\partial x_{pq}} \\
\end{pmatrix}
$$
定义矩阵上的重要的标量函数包括矩阵的==迹==和==行列式==。

# 矩阵指数

引入矩阵指数的目的是为了求解微分方程组基解矩阵。设 $A$ 是一个 $n\times n$ 常数矩阵，我们定义==矩阵指数==(Matrix Exponential)（或写作 $\exp A$）  
$$
e^A=\sum_{k=0}^{\infty}\cfrac{A^k}{k!}=E+A+\cfrac{A^2}{2!}+\cdots\cfrac{A^k}{k!}+\cdots
$$

不难证明，级数 $e^A$ 对于一切矩阵 $A$ 都是绝对收敛的。

矩阵指数$e^A$ 的性质：

(1) 如果矩阵$A,B$是可交换的，即 $AB=BA$，则 $e^{A+B}=e^Ae^B$
(2) 对于任何矩阵 $A$，矩阵指数都是可逆的 $(e^A)^{-1}=e^{-A}$
(3) 如果 $P$ 是非奇异矩阵，$e^{P^{-1}AP}=P^{-1}e^AP$

