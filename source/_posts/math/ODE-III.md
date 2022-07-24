---
title: 常微分方程(三)
categories:
  - 数学
  - 常微分方程
tags:
  - 数学
  - ODE
  - 微分方程
katex: true
description: 一阶线性常微分方程组
abbrlink: ccca011d
date: 2020-04-23 14:53:26
cover: /img/ode.png
top_img: /img/math-top-img.png
---


# 一阶线性微分方程组

## 基本概念

本章主要考虑的是如下形式的一阶线性微分方程组(First-order Linear Differential Equation System) 
$$
\begin{cases}
y'_1(x)=a_{11}(x)y_1+a_{12}(x)y_2+\cdots+a_{1n}y_n+f_1(x) \\
y'_2(x)=a_{21}(x)y_1+a_{22}(x)y_2+\cdots+a_{2n}y_n+f_2(x) \\
\cdots\quad\cdots \\
y'_n(x)=a_{n1}(x)y_1+a_{n2}(x)y_2+\cdots+a_{nn}y_n+f_n(x) \\
\end{cases}\tag{1}
$$
 其中已知函数 $a_{ij}(x)$和$f_i(x)(i,j=1,2,\cdots,n)$在区间 $[a,b]$ 上式连续的。

我们引入下面的**函数矩阵和向量** 
$$
\mathbf{A}(x)=\begin{pmatrix}
a_{11}(x) & a_{12}(x) & \cdots & a_{1n}(x) \\ 
a_{21}(x) & a_{22}(x) & \cdots & a_{2n}(x) \\ 
\vdots &\vdots &\ddots &\vdots \\ 
a_{n1}(x) & a_{n2}(x) & \cdots & a_{nn}(x) \\ 
\end{pmatrix}
$$
 $\mathbf{A}(x)$是$n\times n$函数矩阵，它的每一个元素 $a_{ij}(x)(i,j=1,2,\cdots,n)$是定义区间 $[a,b]$上的函数。
$$
\mathbf{f}(x)=\begin{pmatrix}
f_1(x) \\ f_2(x) \\ \vdots \\  f_n(x)
\end{pmatrix},
\mathbf{y}=\begin{pmatrix}
y_1 \\ y_2 \\ \vdots \\  y_n
\end{pmatrix}
$$
 这里$\mathbf{f}(x),\mathbf{y}$是$n\times 1$矩阵或$n$维列向量。
关于向量或矩阵的代数运算，如相加、相乘与纯量相乘等性质对于以函数作为元素的矩阵同样成立。
函数向量和函数矩阵的连续、微分和积分等概念的定义如下：如果函数向量$\mathbf{y}$或矩阵$\mathbf{A}(x)$的每一个元素都是区间 $a⩽x⩽b$上的连续函数，则称$\mathbf{y}$或 $\mathbf{A}(x)$在$a⩽x⩽b$上连续。
如果函数向量 $\mathbf{y}$或矩阵 $\mathbf{A}(x)$的每一个元素都是区间$a⩽x⩽b$上的可微函数，则称$\mathbf{y}$或$\mathbf{A}(x)$在$a⩽x⩽b$上可微，则定义它们的导数分别为
$$
\mathbf{A}'(x)=\begin{pmatrix}
a'_{11}(x) & a'_{12}(x) & \cdots & a'_{1n}(x) \\ 
a'_{21}(x) & a'_{22}(x) & \cdots & a'_{2n}(x) \\ 
\vdots &\vdots &\ddots &\vdots \\ 
a'_{n1}(x) & a'_{n2}(x) & \cdots & a'_{nn}(x) \\ 
\end{pmatrix} ,
\mathbf{y}'=\begin{pmatrix}
y'_1 \\ y'_2 \\ \vdots \\  y'_n
\end{pmatrix}
$$
不难证明 $n\times n$矩阵 $\mathbf{A}(x),\mathbf{B}(x)$及$n$维向量 $\mathbf{u}(x),\mathbf{v}(x)$ 是可微的，那么下式成立：
(1) $(\mathbf{A}+\mathbf{B})'=\mathbf{A}'+\mathbf{B}'$ 
$(\mathbf{u}+\mathbf{v})'=\mathbf{u}'+\mathbf{v}'$ 
(2) $(\mathbf{A}\cdot \mathbf{B})'=\mathbf{A}'\mathbf{B}+\mathbf{A}\mathbf{B}'$
(3) $(\mathbf{A}\mathbf{u})'=\mathbf{A}'\mathbf{u}+\mathbf{A}\mathbf{u}'$

   类似的，如果函数向量 $\mathbf{y}$或矩阵 $\mathbf{A}(x)$的每一个元素都是区间$a⩽x⩽b$上的可积函数，则称$\mathbf{y}$或$\mathbf{A}(x)$在$a⩽x⩽b$上可积，则定义它们的积分分别为
$$
\int_{a}^{b}\mathbf{A}'(x)dx=\begin{pmatrix}
\int_{a}^{b}a'_{11}(x)dx & \int_{a}^{b}a'_{12}(x)dx & \cdots & \int_{a}^{b}a'_{1n}(x)dx \\ 
\int_{a}^{b}a'_{21}(x)dx & \int_{a}^{b}a'_{22}(x)dx & \cdots & \int_{a}^{b}a'_{2n}(x)dx \\ 
\vdots &\vdots &\ddots &\vdots \\ 
\int_{a}^{b}a'_{n1}(x)dx & \int_{a}^{b}a'_{n2}(x)dx & \cdots & \int_{a}^{b}a'_{nn}(x)dx \\ 
\end{pmatrix}
$$
$$
\int_{a}^{b}\mathbf{y}'dx=
\begin{pmatrix}
\int_{a}^{b}y'_1dx \\ \int_{a}^{b}y'_2dx \\ \vdots \\ \int_{a}^{b} y'_ndx
\end{pmatrix}
$$

关于函数向量与函数矩阵的微分、积分运算法则和普通数值函数类似。

本章所讨论的**一阶线性微分方程组**可以写成以下的形式
$$
\mathbf{y}'=\mathbf{A}(x)\mathbf{y}+\mathbf{f}(x)\tag{2}
$$
若方程 (1) 的初始条件是 
$$
y_1(x_0)=η_1,y_2(x_0)=η_2,\cdots,y_n(x_0)=η_n
$$
 则初始问题可以写成 
$$
\begin{cases}
\mathbf{y}'=\mathbf{A}(x)\mathbf{y}+\mathbf{f}(x) \\
\mathbf{y}(x_0)=\mathbf{η}
\end{cases}
$$

**高阶线性微分方程和一阶线性微分方程组**

   对于 n阶线性微分方程初值问题
$$
\begin{cases}
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y'+a_n(x)y=f(x) \\
y(x_0)=η_1,y'(x_0)=η_2,\cdots,y^{(n-1)}(x_0)=η_n
\end{cases}
$$
引入变换 
$$
y_1=y,y_2=y',\cdots,y_n=y^{(n-1)}
$$
 可以得到下面的一阶线性微分方程组 
$$
\begin{cases}
y'_1=y_2 \\
y'_2=y_3 \\
\cdots \\
y'_{n-1}=y_n \\
y'_n=-a_n(x)y_1-\cdots-a_2(x)y_{n-1}-a_1(x)y_n+f(x)
\end{cases}
$$
初值问题可化为 
$$
\begin{cases}
\mathbf{y}'=\mathbf{A}(x)\mathbf{y+f}(x) \\
\mathbf{y}(x_0)=\mathbf{η}
\end{cases}
$$
 其中 $\mathbf{A}(x)=\begin{pmatrix}
0&1&0&\cdots&0 \\
0&0&1&\cdots&0 \\
\vdots&\vdots&\vdots&\ddots&\vdots\\
0&0&0&\cdots&1 \\
-a_n(x) & -a_{n-1}(x) & -a_{n-2}(x) &\cdots & -a_1(x)
\end{pmatrix} \\
\mathbf{y}=\begin{pmatrix}y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix},
\mathbf{f}(x)=\begin{pmatrix}0 \\0 \\ \vdots \\ f(x) \end{pmatrix},
\mathbf{η}=\begin{pmatrix}η_1 \\η_2 \\ \vdots \\ η_n \end{pmatrix}$

   一阶线性微分方程组的定理可推广到相应的n阶线性微分方程。
   n阶线性微分方程初值问题与对应一阶线性微分方程组初值问题在下列意义下是等价的：
若 $y=φ(x)$ 是n阶线性微分方程在区间 $[a,b]$ 上的解，则 $\mathbf{y}=\begin{pmatrix}φ(x) \\ φ'(x) \\ \vdots \\ φ^{(n-1)}(x) \end{pmatrix}$ 为相应一阶线性微分方程组在区间 $[a,b]$ 上的解。
反之，若向量函数 $\mathbf{y}=\begin{pmatrix}y_1(x) \\ y_2(x) \\ \vdots \\ y_n(x) \end{pmatrix}$ 是相应一阶线性微分方程组在区间 $[a,b]$ 上的解，则 $\mathbf{y}$ 的第一个分量 $y=y_1(x)$ 为n阶线性微分方程的解。


## 解的存在和唯一性定理

引入函数矩阵和向量的**范数**
$$
\|\mathbf{A}\|=\sum_{i,j=1}^{n}|a_{ij}|,\quad\|\mathbf{y}\|=\sum_{i=1}^{n}|y_i|
$$

设$\mathbf{A},\mathbf{B}$是$n\times n$矩阵，$\mathbf{u},\mathbf{v}$ 是$n$维向量，则容易证明下面的性质：
(1) $\|\mathbf{AB}\|⩽\|\mathbf{A}\|\cdot\|\mathbf{B}\|$
$\|\mathbf{Au}\|⩽\|\mathbf{A}\|\cdot\|\mathbf{u}\|$
(2) $\|\mathbf{A+B}\|⩽\|\mathbf{A}\|+\|\mathbf{B}\|$
$\|\mathbf{u+v}\|⩽\|\mathbf{u}\|+\|\mathbf{v}\|$

有了函数向量和函数矩阵的范数，我们就定义了一种函数向量和函数矩阵空间的距离，从而可研究向量序列和矩阵序列的**收敛性**问题。
(1) 向量序列 $\{\mathbf{x}_k\},
\mathbf{x}_k=\begin{pmatrix}
x_{1k} \\ x_{2k} \\ \vdots \\  x_{nk}
\end{pmatrix}$ 称为收敛的，如果对每一个 $i(i=1,2,\cdots,n)$，数列$\{\mathbf{x}_{ik}\}$都是收敛的。
(2) 函数向量序列 $\{\mathbf{x}_k(t)\},
\mathbf{x}_k(t)=\begin{pmatrix}
x_{1k}(t) \\ x_{2k}(t) \\ \vdots \\  x_{nk}(t)
\end{pmatrix}$ 称为区间$[a,b]$上收敛的（一致收敛的），如果对每一个 $i(i=1,2,\cdots,n)$，数列$\{\mathbf{x}_{ik}(x)\}$在区间$[a,b]$上都是收敛的（一致收敛的）。
(3) 设 $\displaystyle\sum_{k=1}^{∞}\mathbf{x}_k$ 是函数向量级数，如果其部分和所作成的函数向量序列在区间 $[a,b]$上收敛（一致收敛），则称 $\displaystyle\sum_{k=1}^{∞}\mathbf{x}_k$
在$[a,b]$上是收敛的（一致收敛的）。
由上面的定义，对函数向量序列和函数向量级数可得到与数学分析中关于函数序列和函数级数相类似的结论。
例如，判别通常的函数级数的一致收敛性的维尔斯特拉斯判别法对于函数向量级数也是成立的，即如果 
$$
\mathbf{x}_k⩽M_k,\quad a⩽t⩽b
$$
 而数值级数$\displaystyle\sum_{k=1}^{∞}M_k$是收敛的，则函数向量级数$\displaystyle\sum_{k=1}^{∞}\mathbf{A}_k$ 在区间 $[a,b]$上一致收敛的。
积分号下取极限的定理对于函数向量也成立，这就是说，如果连续函数向量序列 $\{\mathbf{x}_k(t)\}$ 在 $[a,b]$ 上是一致收敛的，则 
$$
\displaystyle\lim_{k\to\infty}\int_a^b\mathbf{x}_k(t)dt=\int_a^b\lim_{k\to\infty}\mathbf{x}_k(t)dt
$$

以上谈到是向量序列的有关定义和结果，对于一般的举证序列类似。总之，上述一切都是数学分析有关概念的自然推广，证明类似。

<kbd>定理 1 存在唯一性定理</kbd>设 $\mathbf{A}(x)$和$\mathbf{f}(x)$在区间 $[a,b]$ 内连续，则初值问题 
$$
\begin{cases}
\mathbf{y}'=\mathbf{A}(x)\mathbf{y}+\mathbf{f}(x) \\
\mathbf{y}(x_0)=\mathbf{η}
\end{cases}\tag{3}
$$
  在区间 $[a,b]$内存在唯一的解 $\mathbf{φ}(x)$。
和一阶微分方程一样，该定理的证明用到Picard 迭代法，共分五个小命题。
- **命题 1** 设$\mathbf{y}=\mathbf{φ}(x)$是初值问题(3)在$[a,b]$的解，则 $\mathbf{φ}(x)$是 
$$
\mathbf{y}=\mathbf{η}+\int_{x_0}^x[\mathbf{A}(x)\mathbf{y}+\mathbf{f}(x)]dx
$$
 在$[a,b]$上的连续解，反之亦然。

   现在任取 $\mathbf{φ}_0(x)=\mathbf{η}$ 构造皮卡逐步逼近向量函数序列  
$$
\begin{cases}
 \mathbf{φ}_0(x)=\mathbf{η} \\
 \mathbf{φ}_k(x)=\mathbf{η}+\int_{x_0}^x[\mathbf{A}(x)\mathbf{y}+\mathbf{f}(x)]dx
 \end{cases}
$$

 向量函数 $\mathbf{φ}_k(x)$ 成为第 $k$ 次近似解。

- **命题 2**  向量函数 $\mathbf{φ}_k(x)$  在区间 $[a,b]$ 上有定义且连续。

-  **命题 3** 向量函数序列 $\{\mathbf{φ}_k(x)\}$ 在$[a,b]$上是一致收敛的。
    现设
$$
\lim\limits_{n\to}\mathbf{φ}_k(x)=\mathbf{φ}(x)
$$
 则$\mathbf{φ}$也在$[a,b]$上连续。
    
-  **命题 4**   $\mathbf{φ}$ 是积分方程 (3) 在$[a,b]$上是的连续解。

-  **命题 5** 若$\mathbf{ψ}(x)$ 也是积分方程 (3) 在$[a,b]$上是的连续解，则 $\mathbf{ψ}(x)\equiv\mathbf{φ}(x)\quad(x\in[a,b])$。


## 齐次线性微分方程组

当 $\mathbf{f}(x)\equiv0$ 时，(2) 式变为
$$
\mathbf{y}'=\mathbf{A}(x)\mathbf{y}\tag{4}
$$
称为齐次线性微分方组。$\mathbf y\equiv0$ 是齐次方程组 (4) 的解，称为方程组的平凡解。

<kbd>定理 2 叠加原理</kbd>：若$\mathbf{y}_1(x),\mathbf{y}_2(x)$是方程组(4)的解，则他们的线性组合 $α\mathbf{y}_1(x)+β\mathbf{y}_2(x)$ 也是方程组(4)的解，其中$α,β$是任意常数。

令齐次线性方程组 (4) 在区间 $[a,b]$ 上的所有解组成的集合为 $S_n$ ，由叠加原理知，$S_n$ 是一个线性空间。

**线性相关和线性无关**：定义在区间 $[a,b]$ 上$k$个向量函数 $\mathbf{y}_1(x)=\begin{pmatrix}y_{11}(x) \\ y_{21}(x) \\ \vdots \\ y_{n1}(x)\end{pmatrix},
\mathbf{y}_2(x)=\begin{pmatrix}y_{12}(x) \\ y_{22}(x) \\ \vdots \\ y_{n2}(x)\end{pmatrix},
\cdots,
\mathbf{y}_k(x)=\begin{pmatrix}y_{1k}(x) \\ y_{2k}(x) \\ \vdots \\ y_{nk}(x)\end{pmatrix}$，如果存在不全为零的常数 $c_1,c_2,\cdots,c_k$ ，使得恒等式  
$$
c_1\mathbf{y}_1(x)+c_2\mathbf{y}_2(x)+\cdots+c_k\mathbf{y}_k(x)\equiv0
$$
 对所有的 $x\in[a,b]$ 都成立，称这些向量函数在所给区间是==线性相关==的，否则称这些向量函数在所给区间是==线性无关==的。

<kbd>定义</kbd>：有定义在区间 $[a,b]$上的 n 个向量函数$\mathbf{y}_1(x),\mathbf{y}_2(x),\cdots,\mathbf{y}_n(x)$所确定的行列式 
$$
W[\mathbf{y}_1(x),\mathbf{y}_2(x),\cdots,\mathbf{y}_n(x)]=W(x) \\
=\begin{vmatrix}
y_{11}(x) & y_{12}(x) & \cdots & y_{1n}(x) \\ 
y_{21}(x) & y_{22}(x) & \cdots & y_{2n}(x)\\ 
\vdots &\vdots &\ddots &\vdots \\ 
y_{n1}(x) & y_{n2}(x) & \cdots & y_{nn}(x) \\ 
\end{vmatrix}
$$
 称为由这些函数所确定的==伏朗斯基行列式==(Wronskian)。

 <kbd>定理 3</kbd>：若向量函数$\mathbf{y}_1(x),\mathbf{y}_2(x),\cdots,\mathbf{y}_n(x)$ 在区间$a⩽x⩽b$上线性相关，则在区间 $[a,b]$ 上它们的伏朗斯基行列式 $W(x)\equiv0$ 。
 证明：由假设，即知存在一组不全为零的常数 $c_1,c_2,\cdots,c_n$ 使得 
$c_1\mathbf{y}_1(x)+c_2\mathbf{y}_2(x)+\cdots+c_k\mathbf{y}_k(x)\equiv0\quad(a⩽x⩽b)$
上式可看成关于$c_1,c_2,\cdots,c_n$的齐次线性代数方程组，它的系数行列式就是伏朗斯基行列式 $W(x)$ ，于是由线性代数理论知道，要此方程组存在非零解，则它的系数行列式必须为零，即 $W(x)\equiv0$ 。

> 注意，定理 3的逆定理不一定成立。也就是说，由某些向量函数组构成的伏朗斯基行列式为零，但它们也可能是线性无关的。

 <kbd>定理 4</kbd>：齐次线性方程组(4)的解 $\mathbf{y}_1(x),\mathbf{y}_2(x),\cdots,\mathbf{y}_n(x)$ 在区间$a⩽x⩽b$上线性无关，等价于他们的伏朗斯基行列式 $W(x)\neq0\quad(a⩽x⩽b)$ 。
证明：用反证法即可。

<kbd>定理 5</kbd>：齐次线性方程组(4)一定存在n个线性无关的解。
根据解的存在唯一性定理，任取n组初始值 $(a⩽x_0⩽b)$
$\mathbf{y}_1(x_0)=\begin{pmatrix} 1 \\ 0 \\\vdots\\0\end{pmatrix},
\mathbf{y}_2(x_0)=\begin{pmatrix} 0 \\ 1 \\\vdots\\0\end{pmatrix},
\cdots,
\mathbf{y}_n(x_0)=\begin{pmatrix} 0 \\ 0 \\\vdots\\1\end{pmatrix}$
存在n个唯一解 $\mathbf{y}_1(x),\mathbf{y}_2(x),\cdots,\mathbf{y}_n(x)$
又因为 $W(x_0)=1\neq0$
所以在区间$[a,b]$上，$\mathbf{y}_1(x),\mathbf{y}_2(x),\cdots,\mathbf{y}_n(x)\quad(a⩽x_0⩽b)$ 线性无关。

<kbd>推论</kbd>：齐次线性方程组 (4) 的线性无关解的最大个数等于n。

<kbd>定理 6 通解结构定理</kbd> 若$\mathbf{y}_1(x),\mathbf{y}_2(x),\cdots,\mathbf{y}_n(x)$是齐次线性方程组(4)的n个线性无关的解，则方程组(4)的任一解可表示为
$$
\mathbf{y}(x)=c_1\mathbf{y}_1(x)+c_2\mathbf{y}_2(x)+\cdots+c_n\mathbf{y}_n(x)\tag{5}
$$
 其中 $c_1,c_2,\cdots,c_n$ 是任意常数。
证明：由叠加原理，$\mathbf{y}(x)$ 是方程组(4)的解，接下来证明 $\mathbf{y}(x)$ 包含了方程组 (4) 任一解。
由解的存在和唯一性定理，任取方程组 (4) 满足初始条件 $\mathbf{y}(x_0)=\mathbf{η}$ 的一个解 $\mathbf{y}(x)$ ，只需确定常数$c_1,c_2,\cdots,c_n$的值，使其满足(5)式，作非齐次线性代数方程组 
$$
\begin{pmatrix}
y_{11}(x_0) & y_{12}(x_0) & \cdots & y_{1n}(x_0) \\ 
y_{21}(x_0) & y_{22}(x_0) & \cdots & y_{2n}(x_0)\\ 
\vdots &\vdots &\ddots &\vdots \\ 
y_{n1}(x_0) & y_{n2}(x_0) & \cdots & y_{nn}(x_0) \\ 
\end{pmatrix}
\begin{pmatrix}
c_1 \\ c_2 \\ \vdots \\ c_n
\end{pmatrix}
=\begin{pmatrix}
η_1 \\ η_2 \\ \vdots \\ η_n
\end{pmatrix}
$$
 它的系数行列式即为 $W(x_0)\neq0$ ，根据线性代数方程组的理论，上述方程组有唯一解 ，记为$\bar c_1,\bar c_2,\cdots,\bar c_n$。
因此 $\mathbf{y}(x)=\bar c_1\mathbf{y}_1(x)+\bar c_2\mathbf{y}_2(x)+\cdots+\bar c_n\mathbf{y}_n(x)$，并且满足初始条件。
定理证毕。

**基解矩阵**：我们称n个线性无关的解为一个==基本解组==(fundamental system of solutions)。显然，基本解组不是惟一的。由n个解构成的 $n\times n$ 矩阵，称为==解矩阵== 。由基本解组构成的 $n\times n$ 矩阵，称为==基解矩阵== $\mathbfΦ(x)$。
(i) 基解矩阵的行列式就是这个解组的伏朗斯基行列式 $\det\mathbf{Φ}(x)=W(x)\neq0$。 
(ii) 由定理 6的证明可知，其它的解均可由基解矩阵表示 
$$
\mathbf{y} =\mathbf{Φ}(x)\mathbf{c}\tag{6}
$$
 其中 $\mathbf{c}$ 是确定的常数列向量。
(iii) 一个解矩阵 $\mathbf{Φ}(x)$ 是基解矩阵的充要条件是 $\det\mathbf{Φ}(x)\neq0$
<kbd>推论</kbd>
(i) 设 $\mathbf{Φ}(x)$ 是方程组 (4)的基解矩阵，则对任意非奇异（可逆）n 阶常数矩阵 $\mathbf{C}_{n\times n}$ ，矩阵 $\mathbf{Ψ=Φ}(x)\mathbf{C}$ 也是方程组 (4)的基解矩阵。
(ii) 设 $\mathbf{Φ}(x),\mathbf{Ψ}(x)$ 都是方程组 (4)的基解矩阵，则比存在非奇异（可逆）n 阶常数矩阵 $\mathbf{C}_{n\times n}$ 使得 $\mathbf{Ψ=Φ}(x)\mathbf{C}$ 成立。

## 非齐次线性微分方程组 

本节讨论非齐次线性方程组 
$$
\mathbf{y}'=\mathbf{A}(x)\mathbf{y}+\mathbf{f}(x)\tag{2}
$$

对应的齐次线性方程组为 
$$
\mathbf{y}'=\mathbf{A}(x)\mathbf{y}\tag{4}
$$

**解的性质**
(1) 如果 $\mathbf{y}(x)$ 是方程组 (2) 的解，而 $\mathbf{\bar y}(x)$ 是对应齐次线性方程组 (4) 的解，则 $\mathbf y(x)+\mathbf{\bar y}(x)$ 是方程组 (2) 的解。
(2) 如果 $\mathbf{\tilde y}(x), \mathbf{\bar y}(x)$ 是方程组 (2) 的两个解，则 $\mathbf{\tilde y}(x)-\mathbf{\bar y}(x)$ 是对应齐次线性方程组 (4) 的解。

<kbd>定理 7 通解结构定理</kbd>：设$\mathbfΦ(x)$是方程组 (4)的基解矩阵，而 $\mathbf{\bar φ}(x)$ 是方程组 (2) 的某一解，则方程组 (2) 的任一解都可表示为
$$
\mathbfφ(x) =\mathbfΦ(x)\mathbf c+\mathbf{\barφ}(x)\tag{7}
$$
 其中 $\mathbf c$ 是确定的常数列向量。
证明：由解的性质知，$\mathbfφ(x)-\mathbf{\barφ}(x)$ 是方程组(4)的解。
再由上节的结论得到 $\mathbfφ(x)-\mathbf{\barφ}(x)=\mathbfΦ(x)\mathbf c$
由此定理证毕。

**常数变易法**[^const]：定理 7告诉我们，要解非齐次线性方程组，只需知道它的一个特解和对应的齐次线性方程组的基解矩阵。其中，我们可以用常数变易法求得非齐次线性方程组的一个解。
设$\mathbfΦ(x)$是方程组 (4)的基解矩阵，因而方程组 (4) 的任一解为 $\mathbf y=\mathbfΦ(x)\mathbf c$ 。
用常数变易法，令 
$$
\mathbf y=\mathbfΦ(x)\mathbf c(x)
$$
 为非齐次方程组 (2) 的解。这里 $\mathbf c(x)$ 是待定的向量函数。
将它带入方程组 (2) 就得到方程 
$$
\mathbfΦ'(x)\mathbf c(x)+\mathbfΦ(x)\mathbf c'(x)=\mathbf{A}(x)\mathbfΦ(x)\mathbf c(x)+\mathbf{f}(x)
$$
因为$\mathbfΦ(x)$是方程组 (4) 基解矩阵，所以 $\mathbfΦ'(x)=\mathbf{A}(x)\mathbfΦ(x)$ ，由此，上式可化简为 
$$
\mathbfΦ(x)\mathbf c'(x)=\mathbf{f}(x)
$$

又因为基解矩阵的行列式在区间 $[a,b]$ 上恒不等于零，所以可逆，上式两边同左乘 $\mathbfΦ^{-1}(x)$，然后积分，便得到 
$$
\mathbf c(x)=\int_{x_0}^{x}\mathbfΦ^{-1}(s)\mathbf{f}(s)ds\quad x_0,x\in[a,b]
$$
 其中 $\mathbf c(x_0)=0$ 。

<kbd>定理 8</kbd>：如果$\mathbfΦ(x)$是方程组 (4) 的基解矩阵，则 
$$
\mathbf y=\mathbfΦ(x)\int_{x_0}^{x}\mathbfΦ^{-1}(s)\mathbf{f}(s)ds
$$
 是方程组 (2) 的解，且满足初始条件 $\mathbf y(x_0)=0$ 。

由定理 7和定理 8容易看出，方程组 (2) 满足初始条件 $\mathbf y(x_0)=\mathbfη$ 的解由下面给出 
$$
\mathbf y=\mathbfΦ(x)\mathbfΦ^{-1}(x_0)\mathbfη+\mathbfΦ(x)\int_{x_0}^{x}\mathbfΦ^{-1}(s)\mathbf{f}(s)ds \tag{8}
$$
上式称为非齐次方程组的==常数变易公式==。

## 常系数线性微分方程组

本节讨论常系数线性微分方程组 
$$
\mathbf{y'=Ay+f}(x)\tag{9}
$$

对应的常系数齐次线性方程组为 
$$
\mathbf{y'=Ay}\tag{10}
$$

这里系数矩阵 $A$ 为 $n\times n$ 常数矩阵。根据常数变易公式，我们只需要求出方程组 (10) 的一个基解矩阵即可。

**矩阵指数**：引入矩阵指数的目的是为了求解方程组 (10) 基解矩阵。
设 $A$ 是一个 $n\times n$ 常数矩阵，我们定义==矩阵指数==(Matrix Exponential)（或写作 $\exp A$）  
$$
e^A=\sum_{k=0}^{\infty}\cfrac{A^k}{k!}=E+A+\cfrac{A^2}{2!}+\cdots\cfrac{A^k}{k!}+\cdots
$$

不难证明，级数 $e^A$ 对于一切矩阵 $A$ 都是绝对收敛的。

矩阵指数$e^A$ 的性质：
(1) 如果矩阵$A,B$是可交换的，即 $AB=BA$，则 $e^{A+B}=e^Ae^B$
(2) 对于任何矩阵 $A$，矩阵指数都是可逆的 $(e^A)^{-1}=e^{-A}$
(3) 如果 $P$ 是非奇异矩阵，$e^{P^{-1}AP}=P^{-1}e^AP$

<kbd>定理 9</kbd>：矩阵指数 
$$
Φ(x)=e^{Ax}\tag{11}
$$
 是方程组 (10) 的基解矩阵，且 $Φ(0)=E$
证明：级数 $e^{Ax}=\displaystyle\sum_{k=0}^{\infty}\cfrac{A^kx^k}{k!}$ 在 $x$ 的任何有限区间上是一致收敛的。
事实上，对于一切正整数 $k$ ，当 $|x|⩽ c$（c为某一正数）时，有 $\|\cfrac{A^kx^k}{k!}\|⩽\cfrac{\|A\|^k|x|^k}{k!}⩽\cfrac{\|A\|^kc^k}{k!}$ ，而数值级数 $\displaystyle\sum_{k=0}^{\infty}\cfrac{(\|A\|c)^k}{k!}$ 是收敛的，所以 $e^{Ax}$ 是一致收敛的。
而且用逐项微分法，可以得到 $Φ'(x)=Ae^{Ax}=AΦ(x)$ ，这说明 $Φ(x)$ 是方程组 (10) 的解矩阵。又因为 $\det[Φ(0)]=\det E=1$，所以 $Φ(x)$为基解矩阵。

由定理 9我们可以得到方程组 (10) 的任一解可表示为 
$$
\mathbf y=e^{\mathbf Ax}\mathbf c\tag{12}
$$
 这里 $\mathbf c$ 是一个常数向量。方程组 (9) 满足初始条件 $\mathbf y(x_0)=\mathbfη$ 的解由下面给出 
$$
\mathbf y=e^{\mathbf A(x-x_0)}\mathbfη+\int_{x_0}^{x}e^{\mathbf A(x-s)}\mathbf{f}(s)ds \tag{13}
$$


**示例**：如果 $A$ 是一个对角阵 $A=\begin{pmatrix}
a_1&& \\
&a_2&& \\
&&\ddots\\
&&&a_n
\end{pmatrix}$，求 $\mathbf{y'=Ay}$ 的基解矩阵。
$e^{Ax}=E+A+\cfrac{A^2}{2!}+\cdots\cfrac{A^k}{k!}+\cdots=\begin{pmatrix}
e^{a_1x}&& \\
&e^{a_2x}&& \\
&&\ddots\\
&&&e^{a_nx}
\end{pmatrix}$


**基解矩阵的计算**：由于矩阵指数的计算量比较大，我们引入几种解法。
1. **特征值法**

   <kbd>定理 10</kbd>：设常数 $λ$ 是矩阵 $\mathbf A$ 的特征值， $\mathbf c$ 是对应于特征值 $λ$ 的特征向量（$\mathbf{Ac}=λ\mathbf c$），则 $e^{λx}\mathbf c$ 是齐次线性方程组 (10) 的解。
证明：直接代入方程组 $λe^{λx}\mathbf c=\mathbf Ae^{λx}\mathbf c$
因为 $e^{λx}\neq0$，上式简化为齐次线性代数方程组 
$$
(λ\mathbf{E-A)c}=0
$$
   根据线性代数理论，上述方程组获得非零解的充要条件是 $λ$ 满足方程 
$$
\det(λ\mathbf{E-A})=0 \tag{14}
$$
 $n$次多项式 
$$
p(λ)=\det(λ\mathbf{E-A})\tag{15}
$$
 称为==特征多项式==，$n$次代数方程 $p(λ)=0$ 称为==特征方程==。
$\mathbf A$ 的特征值就是特征方程的根，因为 $n$ 次代数方程有 $n$ 个根，所以$\mathbf A$有 $n$ 个特征值，当然不一定 $n$ 个互不相同。
如果 $λ=λ_0$ 是特征方程的单根，则称 $λ_0$ 为==简单特征根==；如果 $λ=λ_0$ 是特征方程的$k$ 重根，则称 $λ_0$ 为 ==$k$重特征根== 。

   <kbd>定理 11</kbd>：如果矩阵 $\mathbf A$ 具有 $n$ 个线性无关的特征向量 $\mathbf{c_1,c_2,\cdots,c_n}$ ，它们对应的特征值分别为 $λ_1,λ_2,\cdots,λ_n$ （不必各不相同），那么矩阵 
$$
\mathbfΦ(x)=(e^{λ_1x}\mathbf{c_1},e^{λ_2x}\mathbf{c_2},\cdots,e^{λ_nx}\mathbf{c_n})
$$
是常系数线性微分方程组 (10) 的一个基解矩阵。
证明：由定理 10知道，每一对特征向量和特征值组成的向量函数 $e^{λ_ix}\mathbf c_i\quad(i=1,2,\cdots,n)$ 都是方程组 (10) 的解。
因此矩阵 $\mathbfΦ(x)=(e^{λ_1x}\mathbf{c_1},e^{λ_2x}\mathbf{c_2},\cdots,e^{λ_nx}\mathbf{c_n})$ 是一个解矩阵
因为向量 $\mathbf{c_1,c_2,\cdots,c_n}$ 线性无关
所以 $W(0)=\det\mathbfΦ(0)=\det(\mathbf{c_1,c_2,\cdots,c_n})\neq0$
从而矩阵 $\mathbfΦ(x)$ 是一个基解矩阵。

   一般来说，定理 11 中的 $\mathbfΦ(x)$ 不一定等于矩阵指数 $e^{\mathbf Ax}$ ，然而根据基解矩阵的性质，存在一个非奇异的常数矩阵 $\mathbf C$ ，使得 $e^{\mathbf Ax}=\mathbfΦ(x)\mathbf C$
令 $x=0$ ，我们得到 $C=\mathbfΦ^{-1}(0)$，因此
$$
e^{\mathbf Ax}=\mathbfΦ(x)\mathbfΦ^{-1}(0)\tag{16}
$$
矩阵指数的计算问题变为方程组任意基解矩阵的计算问题。

   <kbd>结论</kbd>：假设 $λ_1,λ_2,\cdots,λ_k$ 分别是矩阵 $\mathbf A$ 的 $n_1,n_2,\cdots,n_k$ 重不同的特征根，这里 $n_1+n_2+\cdots+n_k=n$ ，$\mathbf{v_1,v_2,\cdots,v_n}$ 是$\mathbf A$ 的一组线性无关的特征向量。常系数线性微分方程组 (10) 满足条件 $\mathbf y(0)=\mathbf η$ 的解可以写成（需要用到线性代数空间分解知识）
$$
\displaystyle\mathbf y=\sum^{k}_{j=1}e^{λ_jx}[\sum_{i=0}^{n_i-1}\cfrac{x^i}{i!}(\mathbf{A}-λ_j\mathbf{E})^i]\mathbf{v}_j\tag{17}
$$
   作为公式 (16) 的应用，下面给出关于方程组的解的稳定性方面的重要定理。
<kbd>定理 12</kbd>：给定常系数线性微分方程组 (10) 
(i) 如果系数矩阵 $\mathbf A$ 的特征值的实部都是负数，则方程组 (10) 的任一解当 $x\to+\infty$ 时都趋于零。
(ii) 如果系数矩阵 $\mathbf A$ 的特征值的实部都是非正数，且实部为零的特征值都是简单特征值，则方程组 (10) 的任一解当 $x\to+\infty$ 时都保持有界。
(iii) 如果系数矩阵 $\mathbf A$ 的特征值至少有一个具有正实部，则方程组 (10) 至少有一解当 $x\to+\infty$ 时都趋于无穷。

2. **利用约当(Jordan)标准型计算**

   由线性代数理论知道，对于矩阵 $\mathbf A$ ，存在n阶非奇异矩阵 $\mathbf P$ ，使得 $\mathbf{PJP^{-1}=A}$，其中 $\mathbf J=\begin{pmatrix}
   \mathbf J_1 \\
   &\mathbf J_2 \\
   &&\ddots \\
   &&&\mathbf J_l \\
   \end{pmatrix}$ 为约当标准型，这里  $\mathbf J_j=\begin{pmatrix}
   λ_j & 1\\
   &λ_j &1\\
   &&\ddots &\ddots \\
   &&&\ddots &1 \\
   &&&&λ_j \\
   \end{pmatrix}\quad(j=1,2,\cdots,l)$ 为 $n_j$ 阶矩阵，并且 $n_1+n_2+\cdots+n_l=n$ ，而 $l$ 为矩阵 $\mathbf{A}-λ\mathbf{E}$ 的初级因子的个数；$λ_1,λ_2,\cdots,λ_k$ 是特征方程 $p(λ)=0$ 的根，可能有相同的；矩阵中空白的元素均为零。
   由于矩阵 $\mathbf J$ 及  $\mathbf J_j$ 的特殊形式，利用矩阵指数的定义容易得到
$$
e^{\mathbf Jx}=\begin{pmatrix}
   e^{\mathbf J_1x} \\
   &e^{\mathbf J_2x} \\
   &&\ddots \\
   &&&e^{\mathbf J_lx} \\
   \end{pmatrix}
$$
   其中  
$$
   e^{\mathbf J_jx}=e^{λ_jx}\begin{pmatrix}
   1 & x & \cfrac{x^2}{2!} &\cdots & \cfrac{x^{n_j-1}}{(n_j-1)!}\\
   &1 &x &\cdots &  \cfrac{x^{n_j-2}}{(n_j-2)!}\\
   &&\ddots &\ddots & \vdots \\
   &&&\ddots & x \\
   &&&&1 \\
   \end{pmatrix}
$$
   由矩阵指数的性质 (3) 知方程组 (10) 的 基解矩阵的计算公式：
$$
e^{\mathbf Ax}=e^{\mathbf{PJP^{-1}}}=\mathbf{P}e^{\mathbf{J}x}\mathbf{P}^{-1}\tag{18}
$$
   当然，根据基解矩阵的性质知道，矩阵 $\mathbf Ψ(x)=\mathbf{P}e^{\mathbf{J}x}$ 也是基解矩阵。
   问题是非奇异矩阵 $\mathbf P$ 的计算比较麻烦。


3. **利用 Hamiton-Cayley 定理计算**
用直接带入的方法应用Hamiton-Cayley 定理容易验证
$$
\displaystyle e^{\mathbf Ax}=\sum_{j=0}^{n-1}r_{j+1}(x)\mathbf P_j \tag{19}
$$
其中 $\displaystyle \mathbf{P_0=E},\mathbf{P_j}=\prod_{k=1}^{j}(\mathbf A-λ_k\mathbf E),(j=1,2,\cdots,n)$，而 $r_1(x),r_2(x),\cdots,r_n(x)$ 是初值问题
$$
\begin{cases}
r'_1=λ_1r_1 \\
r'_j=r_{j-1}+λ_jr_j \\
r_1(0)=1,r_j(0)=0
\end{cases}\quad(j=2,3,\cdots,n)
$$
的解。$λ_1,λ_2,\cdots,λ_k$ 是矩阵 $\mathbf A$ 的特征值（不必相异）。

最后我们看常系数非齐次线性微分方程组初值问题
$$
\begin{cases}
\mathbf{y'=Ay+f}(x) \\
\mathbf{y}(x_0)=\mathbf\eta
\end{cases}
$$
对应齐次方程组的基解矩阵 $\mathbfΦ(x)=e^{\mathbf Ax}$
常数变易公式变为
$$
\mathbf y=e^{(x-x_0)\mathbf A}\mathbfη+\int_{x_0}^{x}e^{(x-s)\mathbf A}\mathbf{f}(s)ds
$$

**拉普拉斯变换的应用**

首先将拉普拉斯变换推广到向量函数：定义
$$
\displaystyle \mathbf F(s)=\mathcal L[\mathbf f(t)]=\int^{\infty}_{0}\mathbf f(t)e^{-st}\text{d}t
$$
这里 $\mathbf f(x)$ 是 n 维向量函数，它的每一个分量都存在拉普拉斯变换。

<kbd>定理 13</kbd>：如果对向量函数常系数 $\mathbf f(x)$ ，存在常数 $M>0,σ>0$ 使不等式 
$$
\|\mathbf f(x)\|⩽Me^{σx}
$$
对所有充分大的 $x$ 成立，则常系数非齐次线性微分方程组初值问题 $\mathbf{y'=Ay+f}(x),\mathbf{y}(0)=\mathbf\eta$ 的解及其导数均像 $\mathbf f(x)$ 一样满足类似的不等式，从而它们的拉普拉斯变换都存在。

拉普拉斯变换可以提供另一种寻求微分方程组 $\mathbf{y'=Ay}$ 基解矩阵的方法。
设 $\mathbf φ(x)$ 是满足初始条件 $\mathbf{y}(0)=\mathbf\eta$ 的解，令 $\mathbf X(s)=\mathcal L[\mathbf φ(x)]$ ，对方程组两边取拉普拉斯变换并利用初始条件求得 $s\mathbf X(s)-\mathbf{\eta=AX}(s)$ ，因此
$$
(s\mathbf{E-A})\mathbf X(s)=\mathbf\eta\tag{20}
$$
这是以 $\mathbf X(s)$ 的 n个分量 $X_1(s),X_2(s),\cdots,X_n(s)$ 为未知量的 n 阶线性代数方程组。显然，若 $s$ 不是 $\mathbf A$ 的特征值，那么 $\det(s\mathbf{E-A})\neq0$，这时根据克莱姆法则，从代数方程组 (20) 可以唯一的解出 $\mathbf X(s)$。
因为  $\det(s\mathbf{E-A})$ 是 $s$ 的 n次多项式，所以  $\mathbf X(s)$ 的每一个分量都是 $s$ 的有理函数，而且关于 $\mathbf\eta$ 的分量 $\eta_1,\eta_2,\cdots,\eta_n$ 都是线性的。因此 $\mathbf X(s)$ 的每一个分量都可以展为部分分式（分母是 $s-λ_j$ 的整数幂，这里 $λ_j$ 是$\mathbf A$ 的特征值）。这样一来，取  $\mathbf X(s)$ 的反变换，就能取得对应任何初始向量 $\mathbf\eta$ 的解 $\mathbf y(x)$。
依次令 $\eta_1=\begin{pmatrix}1\\0\\0\\\vdots\\0\end{pmatrix},\eta_2=\begin{pmatrix}0\\1\\0\\\vdots\\0\end{pmatrix},\cdots,\eta_n=\begin{pmatrix}0\\0\\0\\\vdots\\1\end{pmatrix}$ ，可求得解 $y_1(x),y_2(x),\cdots,y_n(x)$ ，以这些解作为列向量就构成基解矩阵 $\mathbfΦ(x)$ ，且 $\mathbfΦ(0)=\mathbf E$ 。
应用拉普拉斯变换求解常系数线性微分方程组的初值问题是比较便捷的，遗憾的是，它对方程组的要求也比较高。


> **参考文献：**
> 丁同仁.《常微分方程教程》
> 王高雄.《常微分方程》
> 窦霁虹 付英《常微分方程》.西北大学(MOOC) 
> 《高等数学》.国防科技大学(MOOC)
