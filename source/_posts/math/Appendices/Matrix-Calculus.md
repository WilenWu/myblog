---
title: 矩阵微积分
date: 2022-07-15 22:04:26
updated:
tags: 
  - 数学
  - 矩阵
categories:
  - 数学
  - 高等代数
description:
cover:
top_img: '#66CCFF'
katex: true
---

在数学中， **矩阵微积分**是多元微积分的一种特殊表达，尤其是在矩阵空间上进行讨论的时候。它把单个函数对多个变量或者多元函数对单个变量的偏导数写成向量和矩阵的形式，使其可以被当成一个整体被处理。这使得要在多元函数寻找最大或最小值，又或是要为微分方程系统寻解的过程大幅简化。这里我们主要使用统计学和工程学中的惯用记法，而张量下标记法更常用于物理学中。

# 函数矩阵

我们引入下面的**函数矩阵** 
$$
\mathbf{A}(\mathbf x)=\begin{pmatrix}
a_{11}(\mathbf x) & a_{12}(\mathbf x) & \cdots & a_{1n}(\mathbf x) \\ 
a_{21}(\mathbf x) & a_{22}(\mathbf x) & \cdots & a_{2n}(\mathbf x) \\ 
\vdots &\vdots &\ddots &\vdots \\ 
a_{m1}(\mathbf x) & a_{m2}(\mathbf x) & \cdots & a_{mn}(\mathbf x) \\ 
\end{pmatrix}
$$
$\mathbf{A}(\mathbf x)$是$m\times n$函数矩阵，它的每一个元素 $a_{ij}(\mathbf x)=a_{ij}(x_1,x_2,\cdots,x_p)$是定义域 $D$上的函数。关于矩阵的代数运算，如相加、相乘与纯量相乘等性质对于以函数作为元素的矩阵同样成立。

函数矩阵的连续、微分的定义如下：如果函数矩阵$\mathbf{A}(\mathbf x)$的每一个元素都是定义域 $D$上的连续函数，则称$\mathbf{A}(\mathbf x)$在定义域 $D$上连续。如果函数矩阵$\mathbf{A}(\mathbf x)$的每一个元素都是定义域 $D$上的可微函数，则称$\mathbf{A}(\mathbf x)$在定义域 $D$上可微。

# 向量求导

由于向量可看成仅有一列的矩阵，最简单的矩阵求导为向量求导。

## 向量对标量求导

向量 $\mathbf y=\begin{pmatrix}y_1&y_2&\cdots&y_m\end{pmatrix}^T$ 关于标量$x$的导数可以（用分子记法）写成
$$
\frac{\partial\mathbf y}{\partial x}=\begin{pmatrix}
\cfrac{\partial y_1}{\partial x}
&\cfrac{\partial y_2}{\partial x}
&\cdots
&\cfrac{\partial y_m}{\partial x}
\end{pmatrix}^T
$$
在向量微积分中，向量 $\mathbf {y}$ 关于标量 $x$的导数也被称为向量$\mathbf {y}$ 的**切向量**。

**例子** 简单的样例包括欧式空间中的速度向量，它是位移向量（看作关于时间的函数）的切向量。更进一步而言， 加速度是速度的切向量。

## 标量对向量求导

标量 $y$ 对向量$\mathbf x=\begin{pmatrix}x_1&x_2&\cdots&x_n\end{pmatrix}^T$ 导数可以（用分子记法）写成
$$
\frac{\partial y}{\partial\mathbf x}=\begin{pmatrix}
\cfrac{\partial y}{\partial x_1}
&\cfrac{\partial y}{\partial x_2}
&\cdots
&\cfrac{\partial y}{\partial x_n}
\end{pmatrix}
$$

在向量微积分中，标量$y$在的空间$\R^n$(其独立坐标是$\mathbf x$的分量)中的梯度是标量$y$对向量$\mathbf x$的导数的转置。在物理学中，电场是电势的负梯度向量。

标量函数$f(\mathbf x)$对空间向量$\mathbf x$在单位向量$\mathbf u$（在这里表示为列向量）方向上的方向导数可以用梯度定义：
$$
{\displaystyle \nabla _{\mathbf {u} }{f}(\mathbf {x} )=\nabla f(\mathbf {x} )\cdot \mathbf {u} }
$$
使用刚才定义的标量对向量的导数的记法，我们可以把方向导数写作 
$$
\displaystyle \nabla _{\mathbf {u} }f=\left({\frac {\partial f}{\partial \mathbf {x} }}\right)^T \mathbf {u} 
$$
这类记法在证明乘法法则和链式法则的时候非常直观，因为它们与我们熟悉的标量导数的形式较为相似。

## 向量对向量求导

前面两种情况可以看作是向量对向量求导在其中一个是一维向量情况下的特例。类似地我们将会发现有关矩阵的求导可被以一种类似的方式化归为向量求导。

向量函数 (分量为函数的向量) $\mathbf y=\begin{pmatrix}y_1&y_2&\cdots&y_m\end{pmatrix}^T$对输入向量$\mathbf x=\begin{pmatrix}x_1&x_2&\cdots&x_n\end{pmatrix}^T$的导数，可以（用分子记法) 写作
$$
\frac{\partial\mathbf y}{\partial\mathbf x}=\begin{pmatrix}
\cfrac{\partial y_1}{\partial x_1}&\cfrac{\partial y_1}{\partial x_2}&\cdots&\cfrac{\partial y_1}{\partial x_n} \\
\cfrac{\partial y_2}{\partial x_1}&\cfrac{\partial y_2}{\partial x_2}&\cdots&\cfrac{\partial y_2}{\partial x_n} \\
\vdots &\vdots &\ddots &\vdots \\ 
\cfrac{\partial y_m}{\partial x_1}&\cfrac{\partial y_m}{\partial x_2}&\cdots&\cfrac{\partial y_m}{\partial x_n} 
\end{pmatrix}
$$
在向量微积分中，向量函数$\mathbf y$对分量表示一个空间的向量$\mathbf x$的导数也被称为前推 (微分)，或雅可比矩阵。

向量函数$\mathbf f$对$\R ^n$空间中向量$\mathbf v$的前推为 ${\displaystyle d\,\mathbf {f} (\mathbf {v} )={\frac {\partial \mathbf {f} }{\partial \mathbf {v} }}d\,\mathbf {v} }$

# 矩阵求导

有两种类型的矩阵求导可以被写成相同大小的矩阵：矩阵对标量求导和标量对矩阵求导。它们在解决应用数学的许多领域常见的最小化问题中十分有用。类比于向量求导，相应的概念有**切矩阵**和**梯度矩阵**。

## 矩阵对标量求导

矩阵函数$\mathbf Y$对标量$x$的导数被称为**切矩阵**，(用分子记法）可写成：
$$
\frac{\partial\mathbf Y}{\partial x}=\begin{pmatrix}
\cfrac{\partial y_{11}}{\partial x}&\cfrac{\partial y_{12}}{\partial x}&\cdots&\cfrac{\partial y_{1n}}{\partial x} \\
\cfrac{\partial y_{21}}{\partial x}&\cfrac{\partial y_{21}}{\partial x}&\cdots&\cfrac{\partial y_{2n}}{\partial x} \\
\vdots &\vdots &\ddots &\vdots \\ 
\cfrac{\partial y_{m1}}{\partial x}&\cfrac{\partial y_{m2}}{\partial x}&\cdots&\cfrac{\partial y_{mn}}{\partial x} 
\end{pmatrix}
$$

## 标量对矩阵求导

定义在元素是独立变量的矩阵$\mathbf X_{p\times q}$上的标量函数$y$对$\mathbf X$的导数可以（用分子记法）写作

$$
\frac{\partial y}{\partial \mathbf X}=\begin{pmatrix}
\cfrac{\partial y}{\partial x_{11}}&\cfrac{\partial y}{\partial x_{21}}&\cdots&\cfrac{\partial y}{\partial x_{p1}} \\
\cfrac{\partial y}{\partial x_{12}}&\cfrac{\partial y}{\partial x_{22}}&\cdots&\cfrac{\partial y}{\partial x_{p2}} \\
\vdots &\vdots &\ddots &\vdots \\ 
\cfrac{\partial y}{\partial x_{1q}}&\cfrac{\partial y}{\partial x_{2q}}&\cdots&\cfrac{\partial y}{\partial x_{pq}} \\
\end{pmatrix}
$$
定义矩阵上的重要的标量函数包括矩阵的==迹==和==行列式==。

类比于向量微积分，这个导数常被写成如下形式：

$$
\nabla_{\mathbf X}y(\mathbf X)=\frac{\partial y(\mathbf X)}{\partial \mathbf X}
$$
类似地，标量函数$f(\mathbf X)$关于矩阵$\mathbf X$在方向$\mathbf Y$的**方向导数**可写成

$$
\nabla_{\mathbf Y}f=\text{tr}(\frac{\partial f}{\partial \mathbf X}\mathbf Y)
$$
梯度矩阵经常被应用在估计理论的最小化问题中，比如卡尔曼滤波算法的推导，因此在这些领域中有着重要的地位。

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

