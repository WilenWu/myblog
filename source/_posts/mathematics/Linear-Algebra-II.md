---
title: 线性代数(下册)
categories:
  - Mathematics
  - 现代数学
tags:
  - 数学
  - 线性代数
  - 矩阵
  - 特征值
  - 特征向量
  - 线性变换
  - 二次型
cover: /img/Linear-Algebra.png
top_img: /img/matrix-logo.jpg
katex: true
description: 线性空间和内积、特征值与特征向量、二次型与合同、矩阵分解等
abbrlink: f92cc4e9
date: 2023-09-10 23:51:00
---

> [《线性代数的本质》 - 3blue1brown](https://www.bilibili.com/video/BV1ys411472E/)
> 高中数学A版选修4-2 矩阵与变换
> 《线性代数及其应用》(第五版)
> 《高等代数简明教程》- 蓝以中


# 线性空间

## 线性空间

> Grant: **普适的代价是抽象**。

仔细分析就会发现，关于向量空间的一切概念及有关定理都不依赖于向量的具体表现形式(有序数组)，也不依赖于向量加法、数乘的具体计算式，而只依赖于如下两点：

1. 向量的加法与数乘运算封闭；
2. 加法、数乘满足八条运算法则。

这一事实告诉我们：可以把向量的有序数组这一具体表达形式及加法、数乘的具体计算式这些非本质的东西拋弃 ，只把最根本的八条运算法则保留下来。这时它们就不能从理论上给予证明，而要当作公理加以承认。这样，我们就形成了本章的核心概念，也是线性代数这门学科的基本研究对象：数域上的抽象线性空间。

接下来，把向量空间的概念从理论上加以概括和抽象，就得到线性空间的一般性概念，它具有更大的普遍性，应用范围也更广。

<kbd>线性空间</kbd>：设 $V$ 是非空集合，$\mathbb F$ 是一个数域。对 $V$ 中的元素定义两种运算：加法 $\mathbf v+\mathbf w\quad (\mathbf v,\mathbf w\in V)$ 和数乘 $c\mathbf v\quad(c\in\mathbb F,\mathbf v\in V)$ 。若 $V$ 对于加法和数乘运算封闭：

1. $\forall\mathbf v,\mathbf w\in V,\ \mathbf v+\mathbf w\in V$ 
2. $\forall c\in\mathbb F,\mathbf v\in V,\ c\mathbf v\in V$

且  $\forall\mathbf u,\mathbf v,\mathbf w\in V$ and $\forall a,b\in\mathbb F$ 满足以下8条性质：

1. 加法交换律：$\mathbf v+\mathbf w=\mathbf w+\mathbf v$
2. 加法结合律：$\mathbf u+(\mathbf v+\mathbf w)=(\mathbf u+\mathbf v)+\mathbf w$
3. 加法单位元：$\exists 0\in V,\ 0+\mathbf v=\mathbf v$
4. 加法逆元：$\exists (-\mathbf v)\in V,\ \mathbf v+(-\mathbf v)=0$
5. 数乘结合律：$a(b\mathbf v)=(ab)\mathbf v$
6. 数乘分配律：$a(\mathbf v+\mathbf w)=a\mathbf v+a\mathbf w$
7. 数乘分配律：$(a+b)\mathbf v=a\mathbf v+b\mathbf v$
8. 数乘单位元：$\exists 1\in\mathbb F,\ 1\mathbf v=\mathbf v$

则称集合 $V$ 为数域 $\mathbb F$ 上的**线性空间**(或**向量空间**)。线性空间中的元素统称为向量，线性空间中的加法和数乘运算称为线性运算。

> **注意**：
>
> 1. 线性空间的概念是集合与运算二者的结合。同一个集合，若定义两种不同的线性运算，就构成不同的线性空间。
> 2. 线性空间中的向量不一定是有序数组。它已不再具有三维几何空间中向量的几何直观意义。
> 3. 线性运算不一定是有序数组的加法及数乘运算。

然后，之前向量空间的一切结论和性质都可同步到线性空间。

例 1：实数域上次数不大于 $m$ 的全体多项式构成线性空间，记为 $P_m(\R)=\{f(x)=a_0+a_1x+\cdots+a_mx^m\mid a_0,\cdots,a_m\in\R\}$。
例 2：全体 $m×n$ 实矩阵构成线性空间，记为 $\R^{m\times n}$。
例 3：全体函数的集合构成线性空间，也称函数空间。

<kbd>性质</kbd>：

1. 零元素是唯一的；
2. 任一元素的负元素是唯一的；
3. 如果 $c\mathbf v=0$，则 $\mathbf v=0$ 或 $c=0$ ；

## 子空间

<kbd>子空间</kbd>：设 $U$ 是向量空间 $V$ 的一个非空子集，如果$U$中的线性运算封闭，则 $U$ 也是向量空间，称为 $V$ 的**子空间**。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/subspace.svg)

<kbd>子空间的和</kbd>：设 $U_1,U_2$ 为线性空间 $V$ 的两个子空间，则
$$
U_1+U_2=\{\mathbf u_1+\mathbf u_2\mid \mathbf u_1\in U_1,\mathbf u_2\in U_2\}
$$
称为子空间 $U_1,U_2$ 的**和**(sum of subspaces) 。两个子空间的和是分别由两个子空间中各任取一个向量相加所组成的集合。注意 $U_1+U_2$ 和 $U_1\cup U_2$ 不同，后者只是把两个子空间的向量简单地聚拢在一起，成为一个新的集合而已，它们的向量之间并不相加，在一般情况下，$U_1\cup U_2\neq U_1+U_2$ 。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/sum_of_subspaces.svg)

$U_1+U_2$ 是包含子空间 $U_1,U_2$ 的最小子空间。

设 $U_1=\text{span}\{\mathbf a_1,\cdots,\mathbf a_p\},\quad U_2=\text{span}\{\mathbf b_1,\cdots,\mathbf b_q\}$ 则
$$
U_1+U_2=\text{span}\{\mathbf a_1,\cdots,\mathbf a_p,\mathbf b_1,\cdots,\mathbf b_q\}
$$
**维数公式**：
$$
\dim(U_1+U_2)=\dim U_1+\dim U_2-\dim(U_1\cap U_2)
$$

<kbd>直和</kbd>：若任意向量 $\mathbf u\in U_1+U_2$  能唯一的表示成
$$
\mathbf u=\mathbf u_1+\mathbf u_1\quad (\mathbf u_1\in U_1,\mathbf u_2\in U_2)
$$
则称子空间 $U_1+U_2$ 为**直和**(direct sum)，记作 $U_1\oplus U_2$ 。

$U_1+U_2$ 是直和 $\iff$ $U_1\cap U_2=\{O\}$


## 坐标与同构

类似之前向量空间讨论过的，确定线性空间 $V$ 的一组基后，对于任一向量 $\mathbf v\in V$ 可唯一线性表示为
$$
\mathbf v=x_1\mathbf a_1+x_2\mathbf a_2+\cdots+x_n\mathbf a_n
$$
向量的坐标为
$$
\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}\quad \text{or}\quad
(x_1,x_2,\cdots,x_n)
$$

建立了坐标之后，$n$维线性空间 $V_n$ 中的抽象元素与 $n$ 维向量空间 $\R^n$ 中的具体数组之间就有一一对应的关系，并且保持了线性组合(线性运算)的一一对应。

设 $\mathbf v\lrarr (v_1,v_2,\cdots,v_n)^T,\quad \mathbf w\lrarr (w_1,w_2,\cdots,w_n)^T$，则

1. $\mathbf v+\mathbf w\lrarr (v_1,v_2,\cdots,v_n)^T+(w_1,w_2,\cdots,w_n)^T$
2. $c\mathbf v \lrarr c(v_1,v_2,\cdots,v_n)^T$

因此可以说 $V_n$ 与 $\R^n$ 有相同的结构。

一般地，设 $V$ 与 $U$ 是两个线性空间，如果在它们的元素之间有一一对应关系，且这个对应关系保持线性组合的对应，那么就说线性空间  $V$ 与 $U$ **同构**(isomorphism)。

显然，任何实数域上的$n$维线性空间都与 $\R^n$ 同构，即维数相同的线性空间都同构，从而可知，**线性空间的结构完全被它的维数所决定**。

同构的概念除元素一一对应外，主要是保持线性运算的对应关系。因此， $V_n$ 中的抽象的线性运算就可转化为 $\R^n$ 中的线性运算，并且 $\R^n$ 中凡是涉及线性运算的性质就都适用于 $V_n$ 。 

## 线性变换与矩阵

**变换**(transformation)是线性空间的一种映射
$$
T:\quad \mathbf v\mapsto T(\mathbf v)
$$
称 $T(\mathbf v)$ 为向量 $\mathbf v$ 在映射 $T$ 下的**像**，而称 $\mathbf v$ 为 $T(\mathbf v)$ 在映射 $T$ 下的**原像**。

满足下列两条性质的变换称为**线性变换**(linear transformation)

1. 可加性(additivity)：$T(\mathbf v+\mathbf w)=T(\mathbf v)+T(\mathbf w)$
2. 伸缩性(scaling)：$T(c\mathbf v)=cT(\mathbf v)$

设$V$ 是数域 $\R$ 上的$n$ 维线性空间，$\mathbf e_1,\mathbf e_2,\cdots,\mathbf e_n$ 是 $V$ 的一组基。基向量$\mathbf e_j$ 是单位阵 $I_j$ 的第 $j$ 列。对于任一向量 $\mathbf v\in V$ ，设
$$
\mathbf v=\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}=x_1\mathbf e_1+x_2\mathbf e_2+\cdots+x_n\mathbf e_n
$$
对于线性变换 $T$，由线性变换的基本性质知
$$
\begin{aligned}
T(\mathbf v)&=T(x_1\mathbf e_1+x_2\mathbf e_2+\cdots+x_n\mathbf e_n)
=x_1T(\mathbf e_1)+x_2T(\mathbf e_2)+\cdots+x_nT(\mathbf e_n) \\
&=\begin{bmatrix}T(\mathbf e_1)&T(\mathbf e_2)&\cdots&T(\mathbf e_n)\end{bmatrix}\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}
=A\mathbf v
\end{aligned}
$$
矩阵 $A$ 称为线性变换 $T$ 在基 $\mathbf e_1,\mathbf e_2,\cdots,\mathbf e_n$ 下的矩阵。其中，矩阵 $A$ 的第 $j$ 列是基向量$\mathbf e_j$ 的像 $T(\mathbf e_j)$。==显然，矩阵 $A$ 由基的像唯一确定==。

示例：函数是一种特殊的线性空间，定义一个映射：
$$
D=\frac{\mathrm d}{\mathrm dx}:\quad f(x)\mapsto f'(x)
$$
由导数的性质可知，$D$ 是函数空间中的一个线性变换，称为**微分变换**。

在多项式空间 $\R[x]_n$ 内，对任一多项式
$$
f(x)=a_0+a_1x_1+a_2x^2+\cdots+a_nx^n
$$
在基 $1,x,x^2,\cdots,x^n$ 下的坐标表达式为
$$
f(x)=(1,x,x^2,\cdots,x^n)\begin{bmatrix}a_0\\a_1\\a_2\\\vdots\\a_n\end{bmatrix}
$$

基向量 $1,x,x^2,\cdots,x^n$ 的线性变换
$$
\begin{aligned}&D1=0,\\&D x=1,\\&Dx^2=2x,\\&\cdots\\&Dx^n=nx^{n-1}\end{aligned}
$$
故 $D$ 在基 $1,x,x^2,\cdots,x^n$ 下的矩阵为
$$
D=\begin{bmatrix}
0&1&0&\cdots&0\\ 
0&0&2&\cdots&0\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
0&0&0&\cdots&n\\ 
0&0&0&\cdots&0\end{bmatrix}
$$
 $Df(x)$ 在基 $1,x,x^2,\cdots,x^n$ 下的坐标为
$$
Df(x)=\begin{bmatrix}
0&1&0&\cdots&0\\ 
0&0&2&\cdots&0\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
0&0&0&\cdots&n\\ 
0&0&0&\cdots&0\end{bmatrix}
\begin{bmatrix}a_0\\a_1\\a_2\\\vdots\\a_n\end{bmatrix}=
\begin{bmatrix}a_1\\2a_2\\3a_3\\\vdots\\0\end{bmatrix}
$$
即 $Df(x)=a_1+2a_2x+3a_3x^2+\cdots+na_nx^{n-1}$，和直接求导的形式一致。

## 基变换与坐标变换

> Grant：坐标系的建立基于所选的基向量

以二维空间为例，Grant 选用标准坐标系下的基向量，坐标值为
$$
\mathbf i=\begin{bmatrix} 1 \\ 0 \end{bmatrix},\quad
\mathbf j=\begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

而 Jennifer 使用另外一组基向量 $\mathbf i',\mathbf j'$，在 Grant 的坐标系下的坐标表示为
$$
\mathbf i'=\begin{bmatrix} a \\ c \end{bmatrix},\quad
\mathbf j'=\begin{bmatrix} b \\ d \end{bmatrix}
$$

> 实际上在各自的坐标系统，基向量均为 $(1,0),(0,1)$ 。特别的，两个坐标系**原点的定义**是一致的。

同一个向量在不同基向量下表示不同。在 Jennifer 的坐标系中，向量 $\mathbf v=\begin{bmatrix} x' \\ y' \end{bmatrix}$，可以写成基向量的线性组合形式

$$
\mathbf v=x'\mathbf i'+y'\mathbf j'
$$
在 Grant 坐标系下的表示
$$
\mathbf v=x'\begin{bmatrix} a \\ c \end{bmatrix}+y'\begin{bmatrix} b \\ d \end{bmatrix}
$$
进一步，因为是线性变换，所以将其转化为矩阵乘法
$$
\mathbf v=\begin{bmatrix} a&b \\ c&d \end{bmatrix}\begin{bmatrix} x' \\ y' \end{bmatrix}=\begin{bmatrix} x \\ y \end{bmatrix}
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/change_of_basis.svg)

$\begin{bmatrix} x \\ y \end{bmatrix}$ 和 $\begin{bmatrix} x' \\ y’ \end{bmatrix}$ 实际是同一个向量，只不过是在不同基下的坐标。特别的，这里的 $\begin{bmatrix} a&b \\ c&d \end{bmatrix}$ 称为基变换矩阵，意味着同一个向量从 Jennifer 的坐标到 Grant 的坐标的映射，即以我们的视角描述 Jennifer 的向量。

进一步，我们将用基向量 $\mathbf i',\mathbf j'$ 描述的空间称为 “Jennifer's grid”，用基向量 $\mathbf i,\mathbf j$ 描述的空间称为 “Grant‘s grid”。在几何上，基变换矩阵表示的是将 Jennifer's grid 在数值上用 Grant 的语言来描述。而逆变换则是将 Grant 的语言变成 Jennifer 的语言。
$$
\begin{bmatrix} x' \\ y' \end{bmatrix}=\begin{bmatrix} a&b \\ c&d \end{bmatrix}^{-1}\begin{bmatrix} x \\ y \end{bmatrix}
$$
现讨论 $n$维线性空间 $V_n$ 中的情形。任取 $n$ 个线性无关的向量都可以作为 $V_n$ 的一组基，对于不同的基，同一个向量的坐标是不同的。接下来，寻找同一个向量在不同基下的坐标之间的关系。

<kbd>基变换公式</kbd>：设矩阵 $A=(\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n)$ 的列向量与 $B=(\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n)$ 的列向量是$n$维线性空间 $V_n$ 的两组基，则它们可以互相线性表示。若
$$
\begin{cases}
\mathbf b_1=p_{11}\mathbf a_1+p_{21}\mathbf a_2+\cdots+p_{n1}\mathbf a_n \\
\mathbf b_2=p_{12}\mathbf a_1+p_{22}\mathbf a_2+\cdots+p_{n2}\mathbf a_n \\
\cdots  \\
\mathbf b_n=p_{1n}\mathbf a_1+p_{2n}\mathbf a_2+\cdots+p_{nn}\mathbf a_n \\
\end{cases}
$$
利用分块矩阵的乘法形式，可将上式记为
$$
B=AP
$$
称为**基变换公式**。其中，矩阵
$$
P=\begin{bmatrix}
p_{11}&p_{12}&\cdots&p_{1n} \\
p_{21}&p_{22}&\cdots&p_{2n} \\
\vdots&\vdots&\ddots&\vdots \\
p_{n1}&p_{2n}&\cdots&p_{nn} \\
\end{bmatrix}
$$
称为由基 $A=\{\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n\}$ 到 $B=\{\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n\}$ 的**过渡矩阵**(transition matrix)。显然 $P^{-1}$ 为由基$B=\{\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n\}$到基$A=\{\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n\}$的过渡矩阵。

<kbd>坐标变换公式</kbd>：设线性空间 $V$ 中的元素 $\mathbf v$ 在基 $A=\{\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n\}$ 下的坐标为 $\mathbf v_A$ ，在基 $B=\{\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n\}$ 下的坐标为 $\mathbf v_B$ ，则有
$$
\mathbf v_A=P\mathbf v_B
$$
其中矩阵 $P$ 为由基 $A=\{\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n\}$ 到 $B=\{\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n\}$ 的过渡矩阵。

**计算过渡矩阵**：对于基变换公式 $B=AP$ ，可知过渡矩阵 $P=A^{-1}B$ 。写出增广矩阵 $(A\mid B) ，$用初等行变换把左边矩阵 $A$ 处化为单位矩阵 $I$ ，则右边出来的就是过渡矩阵$P$，示意如下：
$$
(A\mid B)\xrightarrow{}(I\mid A^{-1}B)
$$

例：设 $\mathbf b_1=\begin{bmatrix} -9 \\ 1 \end{bmatrix},\mathbf b_2=\begin{bmatrix} -5 \\ -1 \end{bmatrix},\mathbf c_1=\begin{bmatrix} 1 \\ -4 \end{bmatrix},\mathbf c_2=\begin{bmatrix} 3 \\ -5 \end{bmatrix}$ 考虑 $\R^2$ 中的基 $B=\{\mathbf b_1,\mathbf b_2\},C=\{\mathbf c_1,\mathbf c_2\}$ ，求 $B$ 到 $C$ 的过渡矩阵。

解：设基向量 $\mathbf c_1,\mathbf c_2$ 在基 $B$ 下的坐标分别为
$$
[\mathbf c_1]_B=\begin{bmatrix} x_1 \\ x_2 \end{bmatrix},\quad 
[\mathbf c_2]_B=\begin{bmatrix} y_1 \\ y_2 \end{bmatrix}
$$
由坐标的定义，可知
$$
(\mathbf b_1,\mathbf b_2)\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}=\mathbf c_1,\quad
(\mathbf b_1,\mathbf b_2)\begin{bmatrix} y_1 \\ y_2 \end{bmatrix}=\mathbf c_2
$$
为了同步解出这两个方程组，使用增广矩阵 $(B\mid C)$ 求解 
$$
(\mathbf b_1,\mathbf b_2\mid \mathbf c_1,\mathbf c_2)=
\begin{bmatrix}\begin{array}{cc:cc} -9&-5&1&3 \\ 1&-1&-4&-5 \end{array}\end{bmatrix}\to
\begin{bmatrix}\begin{array}{cc:cc} 1&0&-3/2&-2 \\0&1&5/2&3  \end{array}\end{bmatrix}
$$
因此， 由$B$ 到 $C$ 的过渡矩阵
$$
P=\begin{bmatrix} -3/2&-2 \\5/2&3 \end{bmatrix}
$$

# 特征值与特征向量

本章特征值和特征向量的概念只在方阵的范畴内探讨。

## 相似矩阵

> Grant：线性变换对应的矩阵依赖于所选择的基。

一般情况下，同一个线性变换在不同基下的矩阵不同。仍然以平面线性变换为例，Grant 选用标准坐标系下的基向量 $\mathbf i,\mathbf j$ ，线性变换 $T$ 对应的矩阵为 $A$ ，而 Jennifer 使用另外一组基向量 $\mathbf i',\mathbf j'$ 。

我们已经知道矩阵 $A$ 是追踪基向量$\mathbf i,\mathbf j$ 变换后的位置得到的，同样的线性变换在$\mathbf i',\mathbf j'$ 下的表示，也需要追踪基向量 $\mathbf i',\mathbf j'$ 变换后的位置。具体过程如下：

对于 Jennifer 视角下的向量 $\mathbf v=\begin{bmatrix} x' \\ y' \end{bmatrix}$

1. 同样的向量，用 Grant 的坐标系表示的坐标为 $P\begin{bmatrix} x' \\ y' \end{bmatrix}$ ，其中$P$ 为基变换矩阵；
2. 用 Grant 的语言描述变换后的向量 $AP\begin{bmatrix} x' \\ y' \end{bmatrix}$
3. 将变换后的结果变回 Jennifer 的坐标系 $P^{-1}AP\begin{bmatrix} x' \\ y' \end{bmatrix}$

于是，我们得到同一个线性变换 $T$ 在 Jennifer 的坐标系下对应的矩阵为 $P^{-1}AP$ 。

这个结果暗示着数学上的转移作用，中间的矩阵 $A$ 代表 Grant 坐标系下所见到的变换，$P$ 和$P^{-1}$ 两个矩阵代表着转移作用(基变换矩阵)，也就是在不同坐标系之间进行转换，实际上也是视角上的转化。$P^{-1}AP$ 仍然代表同一个变换，只不过是从别的坐标系的角度来看。

下面给出严格的数学证明。在线性空间 $V$ 中取两组基，基变换公式为
$(\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n)=(\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n)P$ 。

设线性变换 $T$ 在这两组基下的矩阵分别为 $A$ 和 $B$ 。那么
$$
T(\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n)=(\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n)A \\
T(\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n)=(\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n)B
$$
取向量 $\mathbf v\in V$ ，在两组基下的坐标向量分别为 $\mathbf x,\mathbf x'$，根据坐标变换公式有 $\mathbf x=P\mathbf x'$
$$
\begin{aligned}
T(\mathbf v)&=(\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n)B\mathbf x'\\
&=(\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n)A\mathbf x \\
&=(\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n)P^{-1}AP\mathbf x'
\end{aligned}
$$
因为 $\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n$ 线性无关，所以
$$
B=P^{-1}AP
$$

因此， $B$ 和 $P^{-1}AP$ 表示同一种线性变换在不同基向量下的表示。

<kbd>相似矩阵</kbd>：设 $A,B$ 都是 $n$ 阶矩阵，若有 $n$ 阶可逆矩阵 $P$ ，使
$$
B=P^{-1}AP
$$
则称矩阵 $A$ 与 $B$ **相似**(similar)，记作 $A\sim B$。

**用初等行变换计算相似矩阵**：计算相似矩阵 $P^{-1}AP$ 的一种有效方法是先计算 $AP$ ，然后用行变换将增广矩阵 $(P\mid AP)$ 化为 $(I\mid P^{-1}AP)$，这样就不需要单独计算$P^{-1}$了 。

## 特征值与特征向量

> Grant：行列式告诉你一个变换对面积的缩放比例，特征向量则是在变换中保留在他所张成的空间中的向量，这两者都是暗含于空间中的性质，坐标系的选择并不会改变他们最根本的值。

我们已经知道，对角阵对于矩阵运算来说最为简单。若线性变换 $T$ 在一组基下的矩阵为 $A$，为便于应用，自然考虑是否存在对角阵 $\Lambda$ 和矩阵 $A$ 相似，从而使用这种最简单的形式计算线性变换。

假设有对角阵 $\Lambda\sim A$，即存在可逆矩阵 $P$ ，使得
$$
P^{-1}AP=\Lambda=\text{diag}(\lambda_1,\lambda_2,\cdots,\lambda_n)
$$
 将矩阵 $P$ 按列分块 $P=(\mathbf x_1,\mathbf x_2,\cdots,\mathbf x_n)$ ，则上式等价于
$$
A(\mathbf x_1,\mathbf x_2,\cdots,\mathbf x_n)=(\mathbf x_1,\mathbf x_2,\cdots,\mathbf x_n)\Lambda
$$
按分块矩阵的乘法，上式可写成
$$
A\mathbf x_1=\lambda_1\mathbf x_1\\
A\mathbf x_2=\lambda_1\mathbf x_2\\
\cdots\\
A\mathbf x_n=\lambda_n\mathbf x_n
$$
根据假定 $P$ 可逆，其列向量非零，因此我们希望找到符合条件的 $\lambda_j,\mathbf x_j$。

<kbd>定义</kbd>：对于矩阵 $A$ ，如果存在数 $\lambda$ 和非零向量 $\mathbf u$，使得
$$
A\mathbf u=\lambda\mathbf u
$$

则称$\lambda$ 是矩阵 $A$ 的一个**特征值**（eigenvalue），$\mathbf u$ 是特征值 $\lambda$ 的一个**特征向量**（eigenvector）。

> (1) 特征向量必须是非零向量；
> (2) 特征值和特征向量是相伴出现的。

事实上，对于任意非零常数$c$， $c\mathbf u$ 都是特征值 $\lambda$ 的特征向量，这是因为
$$
\text{if }A\mathbf u=\lambda\mathbf u,\text{ then }A(c\mathbf u)=\lambda (c\mathbf u)
$$
由于矩阵和线性变换是一一对应的，我们可以借助几何直观理解这个定义。

- 特征向量在变换过程中只受到拉伸或者压缩
- 特征值描述对应特征向量经过线性变换后的缩放程度

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/eigenvectors_with_eigenvalue.svg" style="zoom:70%;" />

对于三维空间中的旋转，如果能够找到对应的特征向量，也即能够留在它所张成的空间中的向量，那么就意味着我们找到了旋转轴。特别地，这就意味着将一个三维旋转看成绕这个特征向量旋转一定角度，要比考虑相应的矩阵变换要直观。此时对应的特征值为1，因为旋转并不改变任何一个向量，所以向量的长度保持不变。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/axis_of_rotation.gif"  />

由定义知道，求解特征向量就是寻找非零向量 $\mathbf u$ 使得
$$
(A-\lambda I)\mathbf u=0
$$

显然，$\mathbf u=0$​ 时恒成立，但是我们要寻找的是非零解。 齐次矩阵方程有非零解的充分必要条件是系数矩阵的行列式为零，即
$$
\det(A-\lambda I)=0
$$
也就是系数矩阵所代表的线性变换将空间压缩到更低的维度。上式称为矩阵 $A$ 的**特征方程**(characteristic equation)。矩阵 $A$ 的特征值就是它的特征方程的根。

多项式
$$
f(\lambda)=\det(A-\lambda I)
$$
称为矩阵 $A$ 的**特征多项式**(characteristic polynomial)。

由上面的讨论可以得出求$n$阶矩阵$A$的特征值与特征向量的**简要步骤**：

1. 求出 $A$ 的特征多项式，即计算$n$阶行列式 $\det(A-\lambda I)$；
2. 求解特征方程 $\det(A-\lambda I)=0$ ，得到$n$个根，即为$A$的$n$ 个特征值；
3. 对求得的每个特征值 $\lambda_i$ 分别带入 $(A-\lambda I)\mathbf x=0$ 求其非零解，便是对应的特征向量。

示例：求矩阵 $A=\begin{bmatrix}1&2\\3&2\end{bmatrix}$ 的特征值和特征向量。

解： $A$ 的特征多项式为
$$
\begin{aligned}\det(A-\lambda I)&=\begin{vmatrix}1-\lambda&2\\3&2-\lambda\end{vmatrix} \\
&=\lambda^2-3\lambda-4=(\lambda-4)(\lambda+1)
\end{aligned}
$$
因此 $A$ 的特征值为 $\lambda_1=4,\lambda_2=-1$。

将  $\lambda_1=4$ 带入矩阵方程 $(A-\lambda I)\mathbf x=0$ ，有
$$
\begin{bmatrix}-3&2\\3&-2\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix}=0 \\
\begin{bmatrix}-3&2\\3&-2\end{bmatrix}\to\begin{bmatrix}3&-2\\0&0\end{bmatrix}
$$
求得特征值 $\lambda_1=4$ 对应的一个特征向量 $\mathbf u_1=c\begin{bmatrix}2\\3\end{bmatrix}$

将  $\lambda_1=-1$ 带入矩阵方程 $(A-\lambda I)\mathbf x=0$ ，有
$$
\begin{bmatrix}2&2\\3&3\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix}=0 \\
\begin{bmatrix}2&2\\3&3\end{bmatrix}\to\begin{bmatrix}1&1\\0&0\end{bmatrix}
$$
求得特征值 $\lambda_2=-1$ 对应的特征向量 $\mathbf u_2=c\begin{bmatrix}-1\\1\end{bmatrix}$

<kbd>性质</kbd>：

1. 相似矩阵(同样的线性变换)有相同的特征多项式，从而有相同的特征值；
2. 矩阵 $A$ 与其转置矩阵 $A^T$ 有相同的特征值；
3. 属于矩阵不同特征值的特征向量线性无关；
4. 矩阵的所有特征值之和等于其主对角线元素之和(矩阵的迹)；
5. 矩阵的所有特征值之积等于矩阵的行列式；
6. 三角阵的特征值是其主对角线元素；
7. 矩阵乘积 $AB$ 和 $BA$ 具有相同的非零特征值

证明：(性质1)设 $A\sim B$，即 $B=P^{-1}AP$ ，于是 
$$
\begin{aligned}
\det(B-\lambda I)&=\det(P^{-1}(A-\lambda I)P) \\
&=\det(P^{-1})\det(A-\lambda I)\det(P) \\
&=\det(A-\lambda I) \\
\end{aligned}
$$
故 $A$ 与 $B$ 有相同的特征多项式，从而有相同的特征值

(性质4)设$n$阶矩阵$A$ 的特征值为 $\lambda_1,\lambda_2,\cdots,\lambda_n$。由于矩阵的特征值就是其特征方程的根，从而
$$
f(\lambda)=\det(A-\lambda I)=(\lambda_1-\lambda)(\lambda_2-\lambda)\cdots(\lambda_n-\lambda)
$$
上式取 $\lambda=0$ ，有 $f(0)=\det A=\lambda_1\lambda_2\cdots\lambda_n$

(性质7)假设矩阵 $A$ 与 $B$ 分别是 $m\times n$ 与 $n\times m$ 矩阵。

证法1：设 $\lambda$ 是 $AB$ 的任一非零特征值，$\mathbf u$ 是这一特征值的特征向量，则 $(AB)\mathbf u=\lambda\mathbf u$ ，等式两边同时左乘 $B$ 有

$$
(BA)(B\mathbf u)=\lambda(B\mathbf u)
$$

又由于 $AB\mathbf u=\lambda\mathbf u\neq0$ 可知 $B\mathbf u\neq 0$ 。所以 $B\mathbf u$ 是 $BA$ 关于特征值 $\lambda$ 的特征向量。这也证明了$\lambda$ 也是$BA$ 的特征值。

同理可证 $BA$ 的非零特征值也是$AB$ 的特征值。这就证明了$AB$ 和 $BA$ 具有相同的非零特征值。

证法2：易知
$$
\begin{bmatrix}I_m&-A\\O&I_n\end{bmatrix}
\begin{bmatrix}AB&O\\B&O\end{bmatrix}
\begin{bmatrix}I_m&A\\O&I_n\end{bmatrix}=
\begin{bmatrix}O&O\\B&AB\end{bmatrix}
$$

又由于
$$
\begin{bmatrix}I_m&-A\\O&I_n\end{bmatrix}
\begin{bmatrix}I_m&A\\O&I_n\end{bmatrix}=
I_{m+n}
$$

可知
$$
\begin{bmatrix}AB&O\\B&O\end{bmatrix}\sim
\begin{bmatrix}O&O\\B&BA\end{bmatrix}
$$

它们有相同的特征多项式，即
$$
\lambda^n\det(\lambda I_m-AB)=\lambda^m\det(\lambda I_n-BA)
$$

上式称为**Sylvester降幂公式**。这里表明，$AB$ 和 $BA$ 的只相差了个 $m-n$ 个零特征值，其余非零特征值相同。

## 特征基与对角化

由上节知道，特征值和特征向量定义的初衷是为了线性变换的相似对角化，即
$$
P^{-1}AP=\Lambda
$$
由定义的推理知道，矩阵 $A$ 的每个特征向量就是 $P$ 的一个列向量，而 $P$ 是矩阵 $A$ 的基向量到对角阵 $\Lambda$ 基向量的过渡矩阵。过渡矩阵 $P$ 也可看作对角阵 $\Lambda$ 的基向量组在矩阵 $A$ 基向量下的坐标，所以对基向量的限制条件也适用于特征向量组。

<kbd>定理</kbd>：矩阵 $A_n$ 可以相似对角化的充要条件是 $A_n$ 有 $n$ 个线性无关的特征向量。此时，对角元素就是对应的特征值。

设矩阵$A$的特征值与特征向量对应关系 $A\mathbf u_1=\lambda_1\mathbf u_1,\quad A\mathbf u_2=\lambda_2\mathbf u_2$ ，令$P=[\mathbf u_1,\mathbf u_2]$ 
$$
AP=[\lambda_1\mathbf u_1,\lambda_2\mathbf u_2]=
[\mathbf u_1,\mathbf u_2]
\begin{bmatrix} \lambda_1&0 \\ 0&\lambda_2 \end{bmatrix}=
P\Lambda \\
$$

若 $P$ 可逆，即 $\mathbf u_1,\mathbf u_2$ 线性无关，则
$$
\Lambda=P^{-1}AP=\begin{bmatrix} \lambda_1&0 \\ 0&\lambda_2 \end{bmatrix}
$$

当特征向量的数量足够多时，这些特征向量就可以构成**特征基**(eigenbasis)。在特征基坐标系角度看，同一个线性变换只是伸缩变换(对角阵)。

> 特征基的坐标使用的是矩阵 $A$ 的基向量。

例：尝试将下列矩阵对角化
$$
A=\begin{bmatrix} 1&3&3 \\ -3&-5&-3 \\ 3&3&1 \end{bmatrix}
$$
解：对角化工作可分为4步来完成

step 1：求出特征值。矩阵 $A$ 的特征方程为
$$
\det(A-\lambda I)=-(\lambda-1)(\lambda+2)^2
$$
特征值是 $\lambda=1$ 和 $\lambda=-2$ 

step 2：求出线性无关的特征向量。对于 $\lambda=1$ 的特征向量 $\mathbf u_1=(1,-1,1)^T$

对于 $\lambda=-2$ 的特征向量 $\mathbf u_2=(-1,1,0)^T$ 和  $\mathbf u_3=(-1,0,1)^T$

可以验证 $\mathbf u_1,\mathbf u_2,\mathbf u_3$ 是线性无关的。

step 3：使用特征向量构造过渡矩阵(向量的次序不重要)
$$
P=\begin{bmatrix} 1&-1&-1 \\ -1&1&0 \\ 1&0&1 \end{bmatrix}
$$
step 4：使用对应的特征值构造对角阵(特征值的次序必须和矩阵$P$的列选择的特征向量的次序一致)
$$
\Lambda=\begin{bmatrix} 1&0&0 \\ 0&-2&0 \\ 0&0&-2 \end{bmatrix}
$$
可简单验证 $AP=P\Lambda$，这等价于验证当 $P$ 可逆时 $\Lambda=P^{-1}AP$ 。

**一些常见变换的特征值与特征向量列举如下**：

(1) 等比例缩放变换 $\begin{bmatrix}k &0\\0 &k\end{bmatrix}$ 的特征多项式为 $(\lambda-k)^2$ ，有两个相等的特征值 $\lambda=k$ ，但平面内任意非零向量都属于这个特征值的特征向量。

(2) 普通缩放变换 $\begin{bmatrix}k_1 &0\\0 &k_2\end{bmatrix}$ 的特征多项式为 $(\lambda-k_1)(\lambda-k_2)$ ，有两个特征值 $\lambda_1=k_1,\lambda_2=k_2$ ，特征向量分别为 $\mathbf u_1=\begin{bmatrix}1\\0\end{bmatrix},\mathbf u_2=\begin{bmatrix}0\\1\end{bmatrix}$。

(3) 旋转变换 $\begin{bmatrix}\cos\theta &-\sin\theta\\ \sin\theta &\cos\theta\end{bmatrix}$  的特征多项式为 $\lambda^2+2\lambda\cos\theta+1$ ，有两个复特征值 $\lambda_1=\cos\theta+i\sin\theta,\lambda_2=\cos\theta-i\sin\theta$ ，对应两个复特征向量 $\mathbf u_1=\begin{bmatrix}1\\-i\end{bmatrix},\mathbf u_2=\begin{bmatrix}1\\i\end{bmatrix}$。

值得注意的是，特征值出现虚数的情况一般对应于变换中的某一种旋转。

(4) 水平剪切变换 $\begin{bmatrix}1 &k\\0 &1\end{bmatrix}$ 的特征多项式为 $(\lambda-1)^2$ ，有两个相等的特征值 $\lambda=1$ ，只有一个特征向量 $\mathbf u_1=\begin{bmatrix}1\\0\end{bmatrix}$ ，不能张成整个平面。

## 特征向量的应用

许多实际问题都可归结为研究矩阵的方幂 $A^n\quad (n\in\N^*)$ 乘以向量 $\mathbf v$ ，不难想象，当方幂很大时，直接用矩阵的乘法、矩阵与向量的乘法进行计算会非常麻烦。而矩阵的特征值和特征向量矩阵对幂运算十分友好，因此在数学和实际问题中有着广泛的应用。

<kbd>性质</kbd>：

1. 设矩阵 $A$ 特征值 $\lambda$ 的特征向量为 $\mathbf u$，则用数学归纳法可以得到
   $$
   A^n\mathbf u=\lambda^n\mathbf u
   $$

2. 设矩阵 $A$ 特征值 $\lambda_1,\lambda_2$ 的特征向量分别为 $\mathbf u_1,\mathbf u_2$。对于任意向量 $\mathbf v$ ，可以用特征向量线性表示 $\mathbf v=v_1\mathbf u_1+v_2\mathbf u_2$ 。那么，用数学归纳法可以得到
   $$
   A^n\mathbf v=v_1\lambda_1^n\mathbf u_1+v_2\lambda_2^n\mathbf u_2
   $$

证明：从线性变换的角度理解，性质1中矩阵 $A$ 只是对特征向量做伸缩变换，因此矩阵幂的效果等价于特征值(缩放比例)的幂。性质2中矩阵的幂变换等同于切换到特征基中做了同等次数的伸缩变换。

性质1用数学归纳法证明：
(1) 当 $n=1$ 时
$$
A\mathbf u=\lambda\mathbf u
$$
(2) 假设当 $n=k-1$ 时成立，即
$$
A^{k-1}\mathbf u=\lambda^{k-1}\mathbf u
$$
当 $n=k$ 时，因为
$$
A^k\mathbf u=A(A^{k-1}\mathbf u)=A(\lambda^{k-1}\mathbf u)=\lambda^{k-1}(A\mathbf u)=\lambda^k\mathbf u
$$

所以，对 $n=k$ 时成立。由数学归纳法可知，对所有的 $n\in\N^*$ 都成立。

**实例**：在扩散理论中的应用。设某物质能以气态和液态的混合状态存在，假定在任意一段很短的时间内 
(1) 液体的 $5\%$ 蒸发成气态；
(2) 气体的 $1\%$ 凝结成液态。
假定该物质的总量一直保持不变，那么最终的情况如何？

为了研究的方便，用 $g_0,l_0$ 分别表示现在的气体和液体的比例 $(g_0+l_0=1)$， $g_n,l_n$ 分别表示 $n$ 段时间后液体和气体的比例。记物质总量为 $M$ ，一直保持不变。

(1) 先求 $g_1,l_1$ 

可以看出，在很短时间后，气体由现在气体的 $99\%$ 加上现在液体的 $5\%$ 组成，即
$$
g_1M=0.99g_0M+0.05l_0M
$$
同理，在很短时间后的液体
$$
l_1M=0.01g_0M+0.95l_0M
$$
因此
$$
\begin{cases}
g_1=0.99g_0+0.05l_0 \\
l_1=0.01g_0+0.95l_0
\end{cases}
$$
矩阵形式为
$$
\begin{bmatrix} g_1\\l_1 \end{bmatrix}=
\begin{bmatrix} 0.99&0.05\\0.01&0.95 \end{bmatrix}
\begin{bmatrix} g_0\\l_0 \end{bmatrix}
$$
记矩阵$P=\begin{bmatrix} 0.99&0.05\\0.01&0.95 \end{bmatrix}$ 则上式写为
$$
\begin{bmatrix} g_1\\l_1 \end{bmatrix}=P\begin{bmatrix} g_0\\l_0 \end{bmatrix}
$$
矩阵 $P$ 记录了很短时间内气液的转变情况。

(2) 类似与 $g_1,l_1$ 的推导过程，可以得到
$$
\begin{aligned}
& \begin{bmatrix} g_1\\l_1 \end{bmatrix}=P\begin{bmatrix} g_0\\l_0 \end{bmatrix}; \\
& \begin{bmatrix} g_2\\l_2 \end{bmatrix}=P\begin{bmatrix} g_1\\l_1 \end{bmatrix}=P^2\begin{bmatrix} g_0\\l_0 \end{bmatrix}; \\
& \cdots\cdots \\
& \begin{bmatrix} g_n\\l_n \end{bmatrix}=P\begin{bmatrix} g_{n-1}\\l_{n-1} \end{bmatrix}=P^n\begin{bmatrix} g_0\\l_0 \end{bmatrix}
\end{aligned}
$$
由于该问题已转化为矩阵指数的形式，我们可以用矩阵特征值和特征向量的性质求解。

(3) 可以证明矩阵
$$
A=\begin{bmatrix}1-p & q\\ p &1-q\end{bmatrix}\quad (0<p,q<1)
$$
的特征值是 $\lambda_1=1,\ \lambda_2=1-p-q$，对应的特征向量分别是 $\mathbf u_1=\begin{bmatrix} q\\ p\end{bmatrix},\ \mathbf u_2=\begin{bmatrix} 1\\ -1\end{bmatrix}$。

从而得到矩阵 $P$ 的特征值是 $\lambda_1=1,\ \lambda_2=0.94$，对应的特征向量分别是  $\mathbf u_1=\begin{bmatrix} 0.05\\ 0.01\end{bmatrix},\ \mathbf u_2=\begin{bmatrix} 1\\ -1\end{bmatrix}$。再把初始向量 $\begin{bmatrix} g_0\\l_0 \end{bmatrix}$ 用特征向量表示，设
$$
\begin{bmatrix} g_0\\l_0 \end{bmatrix}=k_1\begin{bmatrix} 0.05\\ 0.01\end{bmatrix}+k_2\begin{bmatrix} 1\\ -1\end{bmatrix}\quad\text{where }g_0+l_0=1
$$
解得 $k_1=\frac{50}{3},k_2=g_0-\frac{5}{6}$ ，所以由性质2得，对于任意的自然数 $n$ 有
$$
\begin{bmatrix} g_n\\l_n \end{bmatrix}=P^n\begin{bmatrix} g_0\\l_0 \end{bmatrix}=k_1\times1^n\begin{bmatrix} 0.05\\ 0.01\end{bmatrix}+k_2\times0.94^n\begin{bmatrix} 1\\ -1\end{bmatrix}
$$
从而 $g_n=0.05k_1+0.94^nk_2,\ l_n=0.01k_1-0.94^nk_2$，所以
$$
g_{\infty}=\lim\limits_{n\to\infty}(0.05k_1+0.94^nk_2)=0.05k_1=\frac{5}{6} \\
l_{\infty}=\lim\limits_{n\to\infty}(0.01k_1-0.94^nk_2)=0.01k_1=\frac{1}{6}
$$
那么，我们可以得到，不管该物质最初的气液比率如何，最终将达到一个平衡状态，此时该物质的 $5/6$ 是气态的，$1/6$ 是液体的。


# 内积空间

## 内积空间

三维几何空间是线性空间的一个重要例子，如果分析一下三维几何空间，我们就会发现它还具有一般线性空间不具备的重要性质：三维几何空间中向量有长度和夹角，这称为三维几何空间的度量性质。现在，我们在一般线性空间中引入度量有关的概念。

我们知道三维几何空间中向量的长度和夹角可由向量的内积来决定。内积就是一个函数，它把向量对$\mathbf u,\mathbf v$ 映射成一个数。在向量空间 $V$ 中，将内积运算记为 $\lang\mathbf u,\mathbf v\rang$，满足以下性质

1. $\lang\mathbf u,\mathbf v\rang=\lang\mathbf v,\mathbf u\rang$
2. $\lang\mathbf u,\mathbf v+\mathbf w\rang=\lang\mathbf u,\mathbf v\rang+\lang\mathbf u,\mathbf w\rang$
3. $c\lang\mathbf u,\mathbf v\rang=\lang c\mathbf u,\mathbf v\rang=\lang \mathbf u,c\mathbf v\rang$
4. $\lang\mathbf v,\mathbf v\rang\geqslant 0,\ \lang\mathbf v,\mathbf v\rang=0\text{ iff }\mathbf v=0$

定义了内积运算的向量空间称为**内积空间**(innerproductspace)。

> 注意，内积只给出了性质，而没给出具体的计算法则。

对于向量空间 $V$ 中的任意两向量
$$
\mathbf u=u_1\mathbf e_1+\cdots+u_n\mathbf e_n \\
\mathbf v=v_1\mathbf e_1+\cdots+v_n\mathbf e_n
$$
由内积的基本性质知道，其内积
$$
\lang\mathbf u,\mathbf v\rang =\lang u_1\mathbf e_1+\cdots+u_n\mathbf e_n,\ v_1\mathbf e_1+\cdots+v_n\mathbf e_n\rang 
=\sum_{i,j}u_iv_j\lang\mathbf e_i,\mathbf e_j\rang
$$
可见，只要知道基向量之间的内积，就可以求出任意两个向量的内积。上式用矩阵乘法表示为
$$
\lang\mathbf u,\mathbf v\rang=\mathbf u^TM\mathbf v
$$
其中，矩阵 $M=(\delta_{ij})$ 称为坐标基的**度量矩阵**，包含了基向量两两之间的内积
$$
\delta_{ij}=\lang\mathbf e_i,\mathbf e_j\rang
$$
<kbd>定义</kbd>：三维几何空间的度量概念也推广到向量空间中

1. $\|\mathbf v\|=\sqrt{\lang\mathbf v,\mathbf v\rang}$ 称为向量的**长度**或**范数**；
2. $\text{dist}(\mathbf u,\mathbf v)=\|\mathbf u-\mathbf v\|$ 称为向量 $\mathbf u,\mathbf v$ 间的**距离**；
3. 两向量的夹角余弦 $\cos\theta=\dfrac{\lang\mathbf u,\mathbf v\rang}{\|\mathbf u\|\cdot\|\mathbf v\|}$
4. 若 $\lang\mathbf u,\mathbf v\rang=0$ ，则称 $\mathbf u,\mathbf v$ **正交**(orthogonal)；
5. 长度为1的向量称为**单位向量**；
6. 如果向量空间的基向量都为单位向量且两两正交，则称为**标准正交基**(orthonormal basis)；

<kbd>性质</kbd>：

1. $\|\mathbf v\|\geqslant 0,\quad \|\mathbf v\|=0\text{ iff }\mathbf v=0$
2. $c\|\mathbf v\|=|c|\ \|\mathbf v\|$
3. <kbd>勾股定理</kbd>：若 $\mathbf u,\mathbf v$ 是 $V$ 中的正交向量，则 $\|\mathbf u+\mathbf v\|^2=\|\mathbf u\|^2+\|\mathbf v\|^2$
4. <kbd>柯西-施瓦茨不等式</kbd>：$|\lang\mathbf u,\mathbf v\rang|\leqslant\|\mathbf u\|\cdot\|\mathbf v\|$
5. <kbd>三角不等式</kbd>： $\|\mathbf u+\mathbf v\|\leqslant\|\mathbf u\|+\|\mathbf v\|$
6. 若向量组是一组两两正交的非零向量，则向量组线性无关

示例：向量空间的**欧几里得内积**定义为
$$
\lang\mathbf u,\mathbf v\rang=\mathbf u^T\mathbf v=u_1v_1+u_2v_2+\cdots+u_nv_n
$$

即采用的是标准正交基，度量矩阵为单位阵
$$
\delta_{ij}=\begin{cases}1, &i=j \\0, &i\neq j\end{cases}
$$
**以后，当我们讨论内积空间时，总默认采用欧几里得内积。**

<kbd>正交补</kbd>：设 $W$ 是 $V$ 的子空间，如果向量 $\mathbf z$ 与子空间 $W$ 中的任意向量都正交 ，则称 $\mathbf z$ **正交于** $W$。与子空间 $W$ 正交的全体向量的集合称为 $W$ 的**正交补**(orthogonal complement)，并记作 $W^{\perp}$ 。
$$
W^{\perp}=\{\mathbf z\in V\mid \forall\mathbf w\in W,\lang\mathbf z,\mathbf w\rang=0\}
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/orthogonal_complement.svg)

由其次方程 $A\mathbf x=0$ 的解空间易知：

1. $(\text{row }A)^{\perp}=\text{null } A$
2. $(\text{col }A)^{\perp}=\text{null } A^T$

<kbd>定理</kbd>：若 $\mathbf z$ 与$\mathbf u_1,\mathbf u_2,\cdots,\mathbf u_p$ 均正交，则 $\mathbf z$ 正交于 $W=\text{span }\{\mathbf u_1,\mathbf u_2,\cdots,\mathbf u_p\}$ 。

证：对于任意 $\mathbf v\in W$ ，可线性表示为
$$
\mathbf v=x_1\mathbf u_1+x_2\mathbf u_2+\cdots+x_p\mathbf u_p
$$
由内积的性质知
$$
\lang\mathbf z,\mathbf v\rang=x_1\lang\mathbf z,\mathbf u_1\rang+x_2\lang\mathbf z,\mathbf u_2\rang+\cdots+x_p\lang\mathbf z,\mathbf u_p\rang=0
$$
于是可知$\mathbf z$ 正交于 $W$ 。

## 正交矩阵与正交变换

<kbd>定义</kbd>：若矩阵 $A$ 满足 $A^TA=I$，即 $A^{-1}=A^T$，则称 $A$ 为**正交矩阵**。

上式用 $A$ 的列向量表示，即 
$$
\begin{bmatrix}\mathbf a_1^T\\ \mathbf a_2^T\\ \vdots\\\mathbf a_n^T\end{bmatrix}
(\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n)=I_n
$$
 于是得到
$$
\mathbf a_i\mathbf a_j=\begin{cases}1, &i=j\\ 0, &i\neq j\end{cases}
$$
<kbd>定理</kbd>：矩阵 $A$ 为正交矩阵的充要条件是$A$ 的列向量都是单位向量且两两正交。

考虑到 $A^TA=I$ 与 $AA^T=I$ 等价，所以上述结论对 $A$ 的行向量亦成立。

正交矩阵 $A$  对应的线性变换称为**正交变换**。设 $\mathbf u,\mathbf v\in V$ ，则变换后的内积
$$
\lang A\mathbf u,A\mathbf v\rang=(A\mathbf u)^T(A\mathbf v)=\mathbf u^T\mathbf v=\lang\mathbf u,\mathbf v\rang
$$
<kbd>定理</kbd>：正交变换后向量内积保持不变，从而向量的长度、距离和夹角均保持不变。

## 正交投影

<kbd>正交分解定理</kbd>：设 $W$ 是 $V$ 的子空间，那么对于任意 $\mathbf v\in V$ 可唯一表示为
$$
\mathbf v=\hat{\mathbf v}+\mathbf z
$$
其中 $\hat{\mathbf v}\in W,\mathbf z\in W^{\perp}$ 。$\hat{\mathbf v}$ 称为$\mathbf v$ 在 $W$ 上的**正交投影**(orthogonal projection)，记作 $\text{proj}_W\mathbf v$ 。若 $\mathbf u_1,\mathbf u_2,\cdots,\mathbf u_p$ 是 $W$ 的任意正交基，则
$$
\hat{\mathbf v}=\text{proj}_W\mathbf v=\frac{\lang\mathbf v,\mathbf u_1\rang}{\lang\mathbf u_1,\mathbf u_1\rang}\mathbf u_1+\frac{\lang\mathbf v,\mathbf u_2\rang}{\lang\mathbf u_2,\mathbf u_2\rang}\mathbf u_2+\cdots+\frac{\lang\mathbf v,\mathbf u_p\rang}{\lang\mathbf u_p,\mathbf u_p\rang}\mathbf u_p
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/orthogonal_projection.svg)

证：若$\mathbf u_1,\mathbf u_2,\cdots,\mathbf u_p$ 是 $W$ 的任意正交基，则任意 $\mathbf v\in V$ 的投影可线性表示
$$
\hat{\mathbf v}=x_1\mathbf u_1+x_2\mathbf u_2+\cdots+x_p\mathbf u_p
$$
令 $\mathbf z=\mathbf v-\hat{\mathbf v}$ ，由于任意基向量$\mathbf u_j$ 与其他基向量正交且 $\mathbf z\in W^{\perp}$，则
$$
\lang\mathbf z,\mathbf u_j\rang=\lang\mathbf v-\hat{\mathbf v},\mathbf u_j\rang=
\lang\mathbf v,\mathbf u_j\rang-x_j\lang\mathbf u_j,\mathbf u_j\rang=0
$$
于是便求得了投影的系数
$$
x_j=\frac{\lang\mathbf v,\mathbf u_j\rang}{\lang\mathbf u_j,\mathbf u_j\rang}
$$
<kbd>性质</kbd>：设 $W$ 是 $V$ 的子空间，$\mathbf v\in V,\hat{\mathbf v}=\text{proj}_W\mathbf v$

1. (最佳逼近定理) $\hat{\mathbf v}$ 是 $W$ 中最接近 $\mathbf v$ 的点，即对于 $\forall\mathbf w\in W,\ \|\mathbf v-\hat{\mathbf v}\|\leqslant \|\mathbf v-\mathbf w\|$
2. 若$U=(\mathbf u_1,\mathbf u_2,\cdots,\mathbf u_p)$ 的列向量是 $W$ 的单位正交基，则 $\text{proj}_W\mathbf v=UU^T\mathbf v$

证：(1) 取$W$ 中的任一向量 $\mathbf w$ ，由于 
$$
\mathbf v-\mathbf w=(\mathbf v-\hat{\mathbf v})+(\hat{\mathbf v}-\mathbf w)
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/orthogonal_projection-2.svg)

由勾股定理定理知道
$$
\|\mathbf v-\mathbf w\|^2=\|\mathbf v-\hat{\mathbf v}\|^2+\|\hat{\mathbf v}-\mathbf w\|^2
$$
 由于 $\|\hat{\mathbf v}-\mathbf w\|^2\geqslant 0$ 从而不等式得证。

(2) 由于$\mathbf u_1,\mathbf u_2,\cdots,\mathbf u_p$是 $W$ 的单位正交基，那么
$$
\text{proj}_W\mathbf v=\lang\mathbf v,\mathbf u_1\rang\mathbf u_1+\lang\mathbf v,\mathbf u_2\rang\mathbf u_2\cdots++\lang\mathbf v,\mathbf u_p\rang\mathbf u_p\\
=\mathbf u_1^T\mathbf v\mathbf u_1+\mathbf u_2^T\mathbf v\mathbf u_2+\cdots+\mathbf u_p^T\mathbf v\mathbf u_p=UU^T\mathbf v
$$

## 施密特正交化

**施密特(Schmidt)正交化**方法是将向量空间 $V$ 的任意一组基 $\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_r$ 构造成标准正交基 $\mathbf e_1,\mathbf e_2,\cdots,\mathbf e_r$  的简单算法。

取
$$
\begin{aligned}
&\mathbf b_1=\mathbf a_1 \\
&\mathbf b_2=\mathbf a_2-\frac{\mathbf b_1^T\mathbf a_2}{\mathbf b_1^T\mathbf b_1}\mathbf b_1 \\
&\mathbf b_3=\mathbf a_3-\frac{\mathbf b_1^T\mathbf a_3}{\mathbf b_1^T\mathbf b_1}\mathbf b_1-\frac{\mathbf b_2^T\mathbf a_3}{\mathbf b_2^T\mathbf b_2}\mathbf b_2 \\
&\cdots \\
&\mathbf b_r=\mathbf a_r-\frac{\mathbf b_1^T\mathbf a_r}{\mathbf b_1^T\mathbf b_1}\mathbf b_1-\frac{\mathbf b_2^T\mathbf a_r}{\mathbf b_2^T\mathbf b_2}\mathbf b_2-\cdots-\frac{\mathbf b_{r-1}^T\mathbf a_{r-1}}{\mathbf b_{r-1}^T\mathbf b_{r-1}}\mathbf b_{r-1} \\
\end{aligned}
$$
那么 $\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_r$ 是 $V$ 的一组正交基
$$
V=\text{span }\{\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_r\}=\text{span }\{\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_r\}
$$
再把它们单位化
$$
\mathbf e_1=\frac{1}{\|\mathbf b_1\|}\mathbf b_1,\quad\mathbf e_2=\frac{1}{\|\mathbf b_2\|}\mathbf b_2,\quad\cdots,\quad\mathbf e_r=\frac{1}{\|\mathbf b_r\|}\mathbf b_r
$$
最终获得 $V$ 的一组标准正交基。

例：设 $\mathbf a_1=\begin{bmatrix}1\\1\\1\\1\end{bmatrix},\mathbf a_2=\begin{bmatrix}0\\1\\1\\1\end{bmatrix},\mathbf a_3=\begin{bmatrix}0\\0\\1\\1\end{bmatrix}$ 是子空间$V$的一组基，试构造 $V$ 的一组正交基

解：step 1 取第一个基向量 $\mathbf b_1=\mathbf a_1,W_1=\text{span}\{\mathbf a_1\}=\text{span}\{\mathbf b_1\}$ 

step 2 取第二个基向量
$$
\mathbf b_2=\mathbf a_2-\text{proj}_{W_1}\mathbf a_2=
\mathbf a_2-\frac{\mathbf b_1^T\mathbf a_2}{\mathbf b_1^T\mathbf b_1}\mathbf b_1\\
=\begin{bmatrix}0\\1\\1\\1\end{bmatrix}-\frac{3}{4}\begin{bmatrix}1\\1\\1\\1\end{bmatrix}=
\begin{bmatrix}-3/4\\1/4\\1/4\\1/4\end{bmatrix}
$$

为计算方便，缩放 $\mathbf b_2=(-3,1,1,1)^T$ 。同样取 $W_2=\text{span}\{\mathbf b_1,\mathbf b_2\}$

step 3 取第三个基向量
$$
\mathbf b_3=\mathbf a_3-\text{proj}_{W_2}\mathbf a_3=
\mathbf a_3-\frac{\mathbf b_1^T\mathbf a_3}{\mathbf b_1^T\mathbf b_1}\mathbf b_1-\frac{\mathbf b_2^T\mathbf a_3}{\mathbf b_2^T\mathbf b_2}\mathbf b_2\\
=\begin{bmatrix}0\\0\\1\\1\end{bmatrix}-
\frac{2}{4}\begin{bmatrix}1\\1\\1\\1\end{bmatrix}-
\frac{2}{12}\begin{bmatrix}-3\\1\\1\\1\end{bmatrix}=
\begin{bmatrix}0\\-2/3\\1/3\\1/3\end{bmatrix}
$$
![Schmidt](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/Schmidt.svg)

## 实对称矩阵的对角化

<kbd>定理</kbd>：

1. 实对称矩阵对应于不同特征值的特征向量必正交。
2. 实对称矩阵可正交相似对角化。即对于对称矩阵 $A$ ，存在正交矩阵 $P$ ，使 $\Lambda=P^{-1}AP$ 。 $\Lambda$ 的对角元素为 $A$ 的特征值。

证明：(1) 设实对称矩阵 $A$ 对应不同特征值 $\lambda_1,\lambda_2$ 的特征向量分别为 $\mathbf u_1,\mathbf u_2$ 。则
$$
A^T=A,\quad A\mathbf u_1=\lambda_1\mathbf u_1,\quad A\mathbf u_2=\lambda_2\mathbf u_2
$$
对 $A\mathbf u_1=\lambda_1\mathbf u_1$两边求转置，再右乘向量 $\mathbf u_2$，有 
$$
\mathbf u_1^TA\mathbf u_2=\lambda_1\mathbf u_1^T\mathbf u_2
$$
 对 $A\mathbf u_2=\lambda_2\mathbf u_2$两边左乘向量 $\mathbf u_1^T$，有 
$$
\mathbf u_1^TA\mathbf u_2=\lambda_2\mathbf u_1^T\mathbf u_2
$$
两式相减，得到
$$
(\lambda_1-\lambda_2)\mathbf u_1^T\mathbf u_2=0
$$
由于 $\lambda_1\neq \lambda_2$ ，所以 $\mathbf u_1^T\mathbf u_2=0$ ，即特征向量 $\mathbf u_1,\mathbf u_2$ 正交。

例：将矩阵$A=\begin{bmatrix}3&-2&4\\-2&6&2\\4&2&3\end{bmatrix}$正交对角化

解：特征方程 $\det(A-\lambda I)=-(\lambda-7)^2(\lambda+2)=0$ ，特征值和特征向量分别为
$$
\lambda=7:\mathbf v_1=\begin{bmatrix}1\\0\\1\end{bmatrix},
\mathbf v_2=\begin{bmatrix}-1/2\\1\\0\end{bmatrix}; \quad
\lambda=-2:\mathbf v_1=\begin{bmatrix}-1\\-1/2\\1\end{bmatrix}
$$
尽管 $\mathbf v_1,\mathbf v_2$ 是线性无关的，但它们并不正交。我们可以用施密特正交化方法，计算与 $\mathbf v_1$ 正交的 $\mathbf v_2$ 分量
$$
\mathbf z_2=\mathbf v_2-\frac{\mathbf v_1^T\mathbf v_2}{\mathbf v_1^T\mathbf v_1}\mathbf v_1=\begin{bmatrix}-1/4\\1\\1/4\end{bmatrix}
$$
由于 $\mathbf z_2$ 是特征值$\lambda=7$ 的特征向量 $\mathbf v_1,\mathbf v_2$ 的线性组合，从而 $\mathbf z_2$ 是特征值$\lambda=7$ 的特征向量。

分别将 $\mathbf v_1,\mathbf v_2,\mathbf v_3$ 标准化
$$
\mathbf u_1=\begin{bmatrix}1/\sqrt{2}\\0\\1/\sqrt{2}\end{bmatrix},
\mathbf u_2=\begin{bmatrix}-1/\sqrt{18}\\4/\sqrt{18}\\1/\sqrt{18}\end{bmatrix},
\mathbf u_3=\begin{bmatrix}-2/3\\-1/3\\2/3\end{bmatrix}
$$
令
$$
P=(\mathbf u_1,\mathbf u_2,\mathbf u_3)=\begin{bmatrix}1/\sqrt{2}&-1/\sqrt{18}&-2/3\\0&4/\sqrt{18}&-1/3\\1/\sqrt{2}&1/\sqrt{18}&2/3\end{bmatrix},\quad 
\Lambda=\begin{bmatrix}7&0&0\\0&7&0\\0&0&-2\end{bmatrix}
$$
于是正交矩阵 $P$ 将 $A$ 正交对角化，即 $A=P\Lambda P^{-1}$

**对称矩阵的谱**：矩阵 $A$ 的特征值的集合称为 $A$ 的**谱**(spectrum)
$$
\text{spec }A=\{\lambda\in\Complex\mid\det(A-\lambda I)=0\}
$$
<kbd>性质</kbd> 设 $A$ 为 $n$ 阶对称阵

1. $A$ 有 $n$ 个实特征值(包含重复的特征值)；
2. 对于每一个特征值，对应的特征空间的维数等于特征方程的根的重数；
3. 不同特征值的特征空间相互正交的；
4. $A$ 可正交対角化;

**谱分解**：假设对称矩阵 $A=P\Lambda P^{-1}$ 。其中 $P$ 为正交矩阵，其列是 $A$ 的正交特征向量 $\mathbf u_1,\mathbf u_2,\cdots,\mathbf u_n$ ，对应的特征值 $\lambda_1,\lambda_2,\cdots,\lambda_n$是 $\Lambda$ 的对角线元素。由于 $P^T=P^{-1}$ ，故
$$
\begin{aligned}
A&=P\Lambda P^{-1}=(\mathbf u_1,\mathbf u_2,\cdots,\mathbf u_n)
\begin{bmatrix}\lambda_1\\&\lambda_2\\&&\ddots\\&&&\lambda_n\end{bmatrix}
\begin{bmatrix}\mathbf u_1^T\\\mathbf u_2^T\\\vdots\\\mathbf u_n^T\end{bmatrix} \\
&=(\lambda_1\mathbf u_1,\lambda_2\mathbf u_2,\cdots,\lambda_n\mathbf u_n)
\begin{bmatrix}\mathbf u_1^T\\\mathbf u_2^T\\\vdots\\\mathbf u_n^T\end{bmatrix} \\
&=\lambda_1\mathbf u_1\mathbf u_1^T+\lambda_2\mathbf u_2\mathbf u_2^T+\cdots+\lambda_n\mathbf u_n\mathbf u_n^T
\end{aligned}
$$
由于它将 $A$ 分解为由 $A$ 的特征值确定的小块，因此这个 $A$ 的表示就称为 $A$ 的**谱分解**。 上式中的每一项都是一个秩为1的 $n$ 阶方阵。例如，$\lambda_1\mathbf u_1\mathbf u_1^T$的每一列都是 $\mathbf u_1$ 的倍数。

# 二次型与合同

## 二次型与标准型

> Grant：二次型研究的是二次曲面在不同基下的坐标变换

由解析几何的知识，我们了解到二次函数的一次项和常数项只是对函数图像进行平移，并不会改变图形的形状和大小。以一元二次函数为例

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/quadratic_form_img01.svg" style="zoom:80%;" />

而二次函数的二次项控制函数图像的大小和形状。以二元二次函数为例，观察 $f(x,y)=1$ 的截面图形

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/quadratic_form_img02.svg" style="zoom:80%;" />

线性代数主要研究这些图形的二次项，通过线性变换使二次曲面变得规范简洁。

<kbd>定义</kbd>：$n$ 元二次齐次多项式
$$
\begin{aligned}
f(x_1,\cdots,x_n)=&a_{11}x_1^2+2a_{12}x_1x_2+\cdots+2a_{1n}x_1x_n \\
&+a_{22}x_2^2+2a_{23}x_2x_3+\cdots+2a_{2n}x_2x_n \\
&+a_{nn}x_n^2
\end{aligned}
$$
称为**二次型**(quadratic form)，这其实是二次曲面在一组坐标基下的解析表达式。

利用矩阵乘法，二次型可简记为
$$
f=\begin{bmatrix}x_1&x_2&\cdots&x_n\end{bmatrix}
\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1n} \\
a_{21}&a_{22}&\cdots&a_{2n} \\
\vdots&\vdots&\ddots&\vdots \\
a_{m1}&a_{m2}&\cdots&a_{mn} \\
\end{bmatrix}
\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}
=\mathbf x^TA\mathbf x
$$
其中 $A$ 是对称阵，其主对角线元素是平方项的系数，其余元素 $a_{ij}=a_{ji}$ 是二次项 $x_ix_j$ 系数 $2a_{ij}$ 的一半。显然，对称矩阵 $A$ 与二次型 $f$ 是相互唯一确定的。矩阵 $A$ 及其秩分别称为二次型的矩阵和秩。

在某些情况下，没有交叉乘积项的二次型会更容易使用，即通过线性变换 $\mathbf x=C\mathbf y$ 来消除交叉乘积项
$$
f=\mathbf x^TA\mathbf x\xlongequal{\mathbf x=C\mathbf y}\mathbf y^T(C^TAC)\mathbf y=\mathbf y^T\Lambda\mathbf y
$$
由于矩阵 $A$ 是对称阵，由上节对称矩阵的对角化知道，总有正交矩阵 $C$，使
$$
C^{-1}AC=C^TAC=\Lambda
$$
而 $\Lambda$ 的对角线元素是 $A$ 的特征值，于是二次型可简化为
$$
f=\lambda_1y_1^2+\lambda_2y_2^2+\cdots+\lambda_ny_n^2
$$
这种只含平方项的二次型称为**标准型**(standard form)。显然，标准形的矩阵是对角阵。**任何二次型都可通过正交变换化为标准型**。系数全为 +1,-1或 0 的标准型叫做**规范型**(gauge form)。

<kbd>定义</kbd>：设$A$和$B$是$n$阶矩阵，若有$n$阶可逆矩阵$C$，使
$$
B=C^TAC
$$
则称矩阵$A$和$B$**合同**，记为 $A\simeq B$ 。显然，合同矩阵即为二次型在不同基下的矩阵。

<kbd>性质</kbd>：设矩阵 $A\simeq B$

1. 若 $A$ 为对称阵，则 $B$ 也为对称阵；
2. 合同矩阵的秩相等 $\text{rank}(A)=\text{rank}(B)$；

**化二次型为标准型的三种方法：**

1. 求矩阵 $A$ 的特征值和特征向量化为标准型；

2. 使用多项式配方法化为标准型；

3. 使用初等变换法将上方的矩阵 $A$ 的位置变为对角阵(左乘为行变换，不影响下方单位阵变换)
   $$
   \begin{bmatrix}A\\I\end{bmatrix}\xrightarrow{}\begin{bmatrix}C^TAC\\C\end{bmatrix}
   $$

例：将椭圆方程 $5x_1^2-4x_1x_2+5x_2^2=48$ 标准化

解：二次型的矩阵 $A=\begin{bmatrix}5&-2\\-2&5\end{bmatrix}$ ，特征值分别为 3和 7，对应的单位特征向量为
$$
\mathbf u_1=\begin{bmatrix}1/\sqrt{2}\\1/\sqrt{2}\end{bmatrix},
\mathbf u_2=\begin{bmatrix}-1/\sqrt{2}\\1/\sqrt{2}\end{bmatrix}
$$
可使用特征向量 $\mathbf u_1,\mathbf u_2$ 作为二次型的标准正交基。正交变换矩阵和标准型矩阵分别为
$$
C=(\mathbf u_1,\mathbf u_2)=\begin{bmatrix}1/\sqrt{2}&-1/\sqrt{2}\\1/\sqrt{2}&1/\sqrt{2}\end{bmatrix},\quad \Lambda=\begin{bmatrix}3&0\\0&7\end{bmatrix}
$$
$C$ 可将 $A$ 正交对角化，$\Lambda=C^TAC$ 。所以正交变换 $\mathbf x=P\mathbf y$ 得到的标准型为
$$
\mathbf y^TC\mathbf y=3y_1^2+7y_2^2
$$
新的坐标轴如图

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/quadratic_form_img03.svg)

## 二次型的分类

<kbd>定义</kbd>：设二次型$f=\mathbf x^TA\mathbf x$ ，如果对于任何 $\mathbf x\neq 0$ 

1. 都有 $f(\mathbf x)>0$，则称 $f$ 为**正定二次型**，称 $A$ 为**正定矩阵**；
2. 都有 $f(\mathbf x)<0$，则称 $f$ 为**负定二次型**，称 $A$ 为**负定矩阵**；
3. 如果 $f(\mathbf x)$ 既有正值又有负值，则称为**不定二次型**；

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/quadratic_form_img04.png)

从上节可以看出二次型的标准型是不唯一的，但二次型的秩是唯一的，在化成标准型的过程中是不变的，即标准型中含有的非零平方项的个数是不变的。

<kbd>惯性定理</kbd>：二次型和标准型中系数为正的平方项的个数相同，称为**正惯性指数**；系数为负的平方项的个数也相同，称为**负惯性指数**；正负惯性指数之差称为**符号差**。

<kbd>定理</kbd>：

1. $n$元二次型为正定的充要条件是它的正惯性指数为 $n$；
2. 对称阵$A$正定 $\iff$ 特征值全为正 $\iff$ 与单位阵合同 $A\simeq I$ ；
3. 对称阵$A$ 正定 $\implies$ $A^{-1}$ 正定；


## 度量矩阵与合同

> Grant：合同矩阵为不同坐标系下的度量矩阵。

以二维空间为例，Grant 选用标准坐标系下的基向量 $\mathbf i,\mathbf j$，度量矩阵
$$
A=\begin{bmatrix} \lang\mathbf i,\mathbf i\rang&\lang\mathbf i,\mathbf j\rang \\ \lang\mathbf j,\mathbf i\rang&\lang\mathbf j,\mathbf j\rang \end{bmatrix}
$$

而 Jennifer 使用另外一组基向量 $\mathbf i',\mathbf j'$，过渡矩阵 $P=\begin{bmatrix} a&b \\ c&d \end{bmatrix}$。即基向量  $\mathbf i',\mathbf j'$ 在 Grant 的坐标系下的坐标表示为
$$
\mathbf p_1=\begin{bmatrix} a \\ c \end{bmatrix},\quad
\mathbf p_2=\begin{bmatrix} b \\ d \end{bmatrix}
$$
因此， Jennifer 的基向量间的内积
$$
\lang\mathbf i',\mathbf i'\rang=\mathbf p_1^TA\mathbf p_1\\
\lang\mathbf i',\mathbf j'\rang=\mathbf p_1^TA\mathbf p_2 \\
\lang\mathbf j',\mathbf i'\rang=\mathbf p_2^TA\mathbf p_1 \\
\lang\mathbf j',\mathbf j'\rang=\mathbf p_2^TA\mathbf p_2
$$
于是，Jennifer坐标系的度量矩阵
$$
B=\begin{bmatrix} \mathbf p_1^TA\mathbf p_1&\mathbf p_1^TA\mathbf p_2 \\ 
\mathbf p_2^TA\mathbf p_1&\mathbf p_2^TA\mathbf p_2 \end{bmatrix}=
\begin{bmatrix} \mathbf p_1^T \\ \mathbf p_2^T \end{bmatrix}A\begin{bmatrix} \mathbf p_1 & \mathbf p_2 \end{bmatrix}
=P^TAP
$$
由此可知，**合同矩阵刻画了两度量矩阵间的关系**。

当然，也可通过两个向量的内积在不同的坐标系中的计算公式获得两个度量矩阵间的关系。由过渡矩阵知道，同一个向量从 Jennifer 的坐标到 Grant 的坐标变换公式为
$$
\mathbf y=P\mathbf x
$$
在 Jennifer 的坐标系中，两向量 $\mathbf u,\mathbf v$ 的坐标为 $\mathbf x_1,\mathbf x_2$ ，度量矩阵为 $B$ 。内积计算公式
$$
\lang\mathbf u,\mathbf v\rang=\mathbf x_1^TB\mathbf x_2
$$
在 Grant 的坐标系中，两向量 $\mathbf u,\mathbf v$ 的的坐标为$\mathbf y_1,\mathbf y_2$，度量矩阵为 $A$ 。内积计算公式
$$
\lang\mathbf u,\mathbf v\rang=\mathbf y_1^TA\mathbf y_2
=(P\mathbf x_1)^TA(P\mathbf x_2)=\mathbf x_1^T(P^TAP)\mathbf x_2
$$
于是，我们得到了两坐标系中度量矩阵的关系
$$
B=P^TAP
$$


# 矩阵分解

矩阵的因式分解是把矩阵表示为多个矩阵的乘积，这种结构更便于理解和计算。

## LU分解

设 $A$ 是 $m\times n$ 矩阵，若 $A$ 可以写成乘积
$$
A=LU
$$
其中，$L$ 为 $m$ 阶下三角方阵，主对角线元素全是1。$U$ 为 $A$ 得到一个行阶梯形矩阵。这样一个分解称为**LU分解**。 $L$ 称为单位下三角方阵。

我们先来看看，LU分解的一个应用。当 $A=LU$ 时，方程 $A\mathbf x=\mathbf b$ 可写成 $L(U\mathbf x)=\mathbf b$，于是分解为下面两个方程
$$
L\mathbf y=\mathbf b \\
U\mathbf x=\mathbf y
$$
因为 $L$ 和 $U$ 都是三角矩阵，每个方程都比较容易解。

**LU 分解算法**：本节只讲述仅用行倍加变换求解。可以证明，单位下三角矩阵的乘积和逆也是单位下三角矩阵 。此时，可以用行倍加变换寻找 $L$ 和 $U$ 。假设存在单位下三角初等矩阵 $P_1,\cdots,P_s$ 使
$$
P_1\cdots P_sA=U
$$
于是便得到了 $U$ 和 $L$
$$
L=(P_1,\cdots,P_s)^{-1}
$$

## QR分解

如果 $m\times n$ 矩阵 $A$ 的列向量线性无关，那么 $A$ 可以分解为 $A=QR$，其中 $Q$ 是一个 $m\times n$ 正交矩阵，其列为 $\text{col }A$ 的一组标准正交基，$R$ 是一个上 $n\times n$ 三角可逆矩阵，且其对角线上的元素全为正数。

证：矩阵 $A=(\mathbf x_1,\mathbf x_2,\cdots,\mathbf x_n)$ 的列向量是 $\text{col }A$ 的一组基，使用施密特正交化方法可以构造一组标准正交基 $\mathbf u_1,\mathbf u_2,\cdots,\mathbf u_n$ ，取
$$
Q=(\mathbf u_1,\mathbf u_2,\cdots,\mathbf u_n)
$$
因为在正交化过程中 $\mathbf x_k\in\text{span}\{\mathbf x_1,\cdots,\mathbf x_k\}=\text{span}\{\mathbf u_1,\cdots,\mathbf u_k\},\quad k=1,2,\cdots,n$ 。所以 $\mathbf x_k$ 可线性表示为 
$$
\mathbf x_k=r_{1k}\mathbf u_1+\cdots+r_{kk}\mathbf u_k+0\cdot\mathbf u_{k+1}+\cdots+0\cdot\mathbf u_n
$$
于是
$$
\mathbf x_k=Q\mathbf r_k
$$
其中 $\mathbf r_k=(r_{1k},\cdots,r_{kk},0,\cdots,0)^T$ ，且 $r_{kk}\geqslant 0$ (在正交化过程中，若 $r_{kk}<0$ ，则$r_{kk}$ 和 $\mathbf u_k$ 同乘-1)。取 $R=(\mathbf r_1,\mathbf r_2,\cdots,\mathbf r_n)$ ，则
$$
A=(Q\mathbf r_1,Q\mathbf r_2,\cdots,Q\mathbf r_n)=QR
$$
例：求 $A=\begin{bmatrix}1&0&0\\1&1&0\\1&1&1\\1&1&1\end{bmatrix}$ 的一个 QR 分解

解：通过施密特正交化方法我们可以得到 $\text{col }A$ 的一组标准正交基，将这些向量组成矩阵
$$
Q=\begin{bmatrix}1/2&-3/\sqrt{12}&0\\1/2&1/\sqrt{12}&-2/\sqrt{6}\\1/2&1/\sqrt{12}&1/\sqrt{6}\\1/2&1/\sqrt{12}&1/\sqrt{6}\end{bmatrix}
$$
注意到 $Q$ 是正交矩阵，$Q^T=Q^{-1}$ 。所以 $R=Q^{-1}A=Q^TA$
$$
R=\begin{bmatrix}1/2&1/2&1/2&1/2\\
-3/\sqrt{12}&1/\sqrt{12}&1/\sqrt{12}&1/\sqrt{12} \\
0&-2/\sqrt{6}&1/\sqrt{6}&1/\sqrt{6} 
\end{bmatrix}
\begin{bmatrix}1&0&0\\1&1&0\\1&1&1\\1&1&1\end{bmatrix}=
\begin{bmatrix}2&3/2&1\\0&3/\sqrt{12}&2/\sqrt{12}\\0&0&2/\sqrt{6} \end{bmatrix}
$$

## 特征值分解

特征值分解是将矩阵分解成特征值和特征向量形式：
$$
A=Q\Sigma Q^{-1}
$$
其中，$\Sigma=\text{diag}(\lambda_1,\lambda_2,\cdots,\lambda_n)$ 是一个对角阵，其对角线元素是矩阵 $A$ 的特征值按降序排列 $\lambda_1\geqslant\lambda_2\geqslant\cdots\geqslant\lambda_n$，$Q=(\mathbf u_1,\mathbf u_2,\dots,\mathbf u_n)$ 是特征值对应的特征向量组成的矩阵。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/eigen_decomposition.svg" style="zoom:100%;" />

特征值分解后，方阵的幂变得更容易计算
$$
A^t=Q\Sigma^t Q^{-1}=Q\begin{bmatrix}\lambda_1^t\\&\ddots\\&&\lambda_n^t\end{bmatrix}Q^{-1}
$$
特征值分解可以理解为：先切换基向量，然后伸缩变换，最后再切换回原来的基向量。其中，$\Sigma$ 中的特征向量描述伸缩变换的程度，特征向量描述变换的方向。

特征值分解有一定的局限性，因为它只适用于满秩的方阵。

例：求矩阵 $A=\begin{bmatrix}-2&1&1\\0&2&0\\-4&1&3\end{bmatrix}$ 的特征值分解。

解：矩阵 $A$ 的特征多项式为 $\det(A-\lambda I)=-(\lambda-2)^2(\lambda+1)$ 。特征值和特征向量分别为
$$
\lambda_1=-1:\mathbf u_1=\begin{bmatrix}1\\0\\1\end{bmatrix};\quad
\lambda_2=2:\mathbf u_2=\begin{bmatrix}0\\1\\-1\end{bmatrix},
\mathbf u_3=\begin{bmatrix}1\\0\\4\end{bmatrix}
$$
可通过行变换计算逆矩阵
$$
(Q,I)=\begin{bmatrix}\begin{array}{ccc:ccc}
0&1&1&1&0&0\\1&0&0&0&1&0\\-1&4&1&0&0&1
\end{array}\end{bmatrix}\to
\begin{bmatrix}\begin{array}{ccc:ccc}
1&0&0&0&1&0\\0&1&0&-1/3&1/3&1/3\\0&0&1&4/3&-1/3&-1/3
\end{array}\end{bmatrix}=(I,Q^{-1})
$$
所以
$$
A=\begin{bmatrix}0&1&1\\1&0&0\\-1&4&1\end{bmatrix}
\begin{bmatrix}2&0&0\\0&2&0\\0&0&-1\end{bmatrix}
\begin{bmatrix}0&1&0\\-1/3&1/3&1/3\\4/3&-1/3&-1/3\end{bmatrix}
$$

## 奇异值分解

### 奇异值分解

奇异值分解(Singular Value Decomposition, SVD)是线性代数中一种重要的矩阵分解，在生物信息学、信号处理、金融学、统计学等领域有重要应用。

SVD 可以理解为同一线性变换 $T:\R^n\mapsto\R^m$ 在不同基下的矩阵表示。假设 Grant 选用标准基，对应的矩阵为 $A_{m\times n}$ 。类似于特征值分解， Jennifer 通过选择合适的基向量，对应的矩阵变为简单的长方形对角矩阵 $\Sigma_{m\times n}$，即只有伸缩变换。

假定 Jennifer 使用矩阵 $V_n=(\mathbf v_1,\cdots,\mathbf v_n)$ 的列向量作为 $R^n$ 的基，使用矩阵 $U_n=(\mathbf u_1,\cdots,\mathbf u_m)$的列向量作为 $R^m$ 的基 。那么，对于 Jennifer 视角下的向量 $\mathbf x\in R^n$ 

1. 同样的向量，用 Grant 的坐标系表示为 $V\mathbf x$ 
2. 用 Grant 的语言描述变换后的向量 $AV\mathbf x$
3. 将变换后的结果变回 Jennifer 的坐标系 $U^{-1}AV\mathbf x$

于是，我们得到同一个线性变换 $T$ 在 Jennifer 的坐标系下对应的矩阵 $\Sigma=U^{-1}AV$ ，也可理解为矩阵 $A$ 分解为 $A_{m\times n}=U_m\Sigma_{m\times n}V^{-1}_n$ 。

接下来，自然是探讨上述矩阵分解的适用条件。

注意到
$$
A^TA=(U\Sigma V^{-1})^T(U\Sigma V^{-1})=V^{-T}\Sigma^TU^TU\Sigma V^{-1} 
$$
不妨取 $U,V$ 为单位正交基，即$U,V$ 为正交矩阵 $U^TU=I,V^TV=I$ ，则
$$
A^TA=V\Sigma^T\Sigma V^T
$$
于是，可知 $V$ 的列向量为 $A^TA$ 的特征向量，$\Sigma^T\Sigma$ 为$n$ 阶对角阵，其对角元素为$A^TA$ 的特征值。事实上 $A^TA$ 为对称阵，必定存在正交矩阵 $V$ 相似对角化。

同理
$$
AA^T=U\Sigma\Sigma^T U^T
$$
可知 $U$ 的列向量为 $AA^T$ 的特征向量，$\Sigma\Sigma^T$ 为$m$ 阶对角阵，其对角元素为$AA^T$ 的特征值。矩阵 $A^TA$ 为对称阵，必定存在正交矩阵 $U$ 相似对角化。

目前 $U,V$ 我们都求出来了，只剩下求出长方形对角矩阵 $\Sigma$ 。根据 Sylvester降幂公式， $A^TA$ 和 $AA^T$ 有相同的非零特征值。

令 $\Sigma=\begin{bmatrix}\Lambda_r&O\\O&O\end{bmatrix}$ ，其中 $\Lambda_r=\text{diag}(\sigma_1,\cdots,\sigma_r)$ 。则
$$
\Sigma^T\Sigma=\begin{bmatrix}\Lambda_r^2&O\\O&O\end{bmatrix}_n,\quad
\Sigma\Sigma^T=\begin{bmatrix}\Lambda_r^2&O\\O&O\end{bmatrix}_m
$$
其中 $\Lambda_r^2=\text{diag}(\sigma_1^2,\cdots,\sigma_r^2)$ 。因此，矩阵 $\Sigma$ 的对角元素是 $A^TA$ 和 $AA^T$ 的特征值 $\lambda_j$ 的平方根
$$
\sigma_j=\sqrt{\lambda_j}
$$
综上，**任意矩阵均可奇异值分解**。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/SVD.svg" alt="SVD"  />

<kbd>定义</kbd>：SVD是指将秩为 $r$ 的 $m\times n$ 矩阵$A$分解为
$$
A=U\Sigma V^T
$$

其中 $U$ 为 $m$ 阶正交阵， $V$ 为 $n$ 阶正交阵，$\Sigma$ 为 $m\times n$ 维长方形对角矩阵，对角元素称为矩阵 $A$ 的**奇异值**，一般按降序排列 $\sigma_1\geqslant\sigma_2\geqslant\cdots\geqslant\sigma_r>0$ ，这样 $\Sigma$ 就唯一确定了。矩阵 $U$ 的列向量称为**左奇异向量**(left singular vector)，矩阵 $V$ 的列向量称为**右奇异向量**(right singular vector)。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/svd.png" style="zoom:80%;" />

例：这里我们用一个简单的矩阵来说明奇异值分解的步骤。求矩阵 $A=\begin{bmatrix}0&1\\1&1\\1&0\end{bmatrix}$ 的奇异值分解

解：首先求出对称阵 $A^TA$ 和 $AA^T$
$$
A^TA=\begin{bmatrix}0&1&1\\1&1&0\end{bmatrix}
\begin{bmatrix}0&1\\1&1\\1&0\end{bmatrix}=
\begin{bmatrix}2&1\\1&2\end{bmatrix} \\
AA^T=\begin{bmatrix}0&1\\1&1\\1&0\end{bmatrix}
\begin{bmatrix}0&1&1\\1&1&0\end{bmatrix}=
\begin{bmatrix}1&1&0\\1&2&1\\0&1&1\end{bmatrix}
$$
然后求出 $A^TA$ 的特征值和特征向量
$$
\lambda_1=3:\mathbf v_1=\begin{bmatrix}1/\sqrt{2}\\1/\sqrt{2}\end{bmatrix};\quad 
\lambda_2=1:\mathbf v_2=\begin{bmatrix}-1/\sqrt{2}\\1/\sqrt{2}\end{bmatrix}
$$
求出  $AA^T$ 的特征值和特征向量
$$
\lambda_1=3:\mathbf u_1=\begin{bmatrix}1/\sqrt{6}\\2/\sqrt{6}\\1/\sqrt{6}\end{bmatrix};\quad 
\lambda_2=1:\mathbf u_2=\begin{bmatrix}1/\sqrt{2}\\0\\-1/\sqrt{2}\end{bmatrix};\quad
\lambda_3=0:\mathbf u_3=\begin{bmatrix}1/\sqrt{3}\\-1/\sqrt{3}\\1/\sqrt{3}\end{bmatrix};
$$
其次可以利用 $\sigma_i=\sqrt{\lambda_i}$ 求出奇异值 $\sqrt{3},1$

最终得到$A$的奇异值分解
$$
A=U\Sigma V^T=\begin{bmatrix}1/\sqrt{6}&1/\sqrt{2}&1/\sqrt{3}\\2/\sqrt{6}&0&-1/\sqrt{3}\\1/\sqrt{6}&-1/\sqrt{2}&1/\sqrt{3}\end{bmatrix}
\begin{bmatrix}\sqrt{3}&0\\0&1\\0&0\end{bmatrix}
\begin{bmatrix}1/\sqrt{2}&1/\sqrt{2}\\-1/\sqrt{2}&1/\sqrt{2}\end{bmatrix}
$$

### 矩阵的基本子空间

设矩阵 $A=U\Sigma V^T$ ，有$r$ 个不为零的奇异值，则可以得到矩阵 $A$ 的四个基本子空间：

1. 正交阵 $U$ 的前 $r$ 列是 $\text{col }A$ 的一组单位正交基
2. 正交阵 $U$ 的后 $m-r$ 列是 $\text{null } A^T$ 的一组单位正交基
3. 正交阵 $V$ 的前 $r$ 列是 $\text{col }A^T$ 的一组单位正交基
4. 正交阵 $V$ 的后 $n-r$ 列是 $\text{null } A$ 的一组单位正交基


$$
A(\underbrace{\mathbf v_1,\cdots,\mathbf v_r}_{\text{col }A^T},\underbrace{\mathbf v_{r+1}\cdots\mathbf v_n}_{\text{null } A})=
(\underbrace{\mathbf u_1,\cdots,\mathbf u_r}_{\text{col }A},\underbrace{\mathbf u_{r+1}\cdots\mathbf u_m}_{\text{null } A^T})
\underbrace{\begin{bmatrix}\sigma_1\\&\ddots\\&&\sigma_r\\&&&O
\end{bmatrix}}_{\Sigma_{m\times n}}
$$

证：易知 $AV=U\Sigma$ ，即
$$
\begin{cases}
A\mathbf v_i=\sigma_i\mathbf u_i, &1\leqslant i\leqslant r \\
A\mathbf v_i=0, &r< i\leqslant n
\end{cases}
$$
取 $\mathbf v_1,\cdots,\mathbf v_n$ 为 $\R^n$ 的单位正交基，对于 $\forall\mathbf x\in \R^n$ ，可以写出 $\mathbf x=c_1\mathbf v_1+\cdots+c_n\mathbf v_n$，于是
$$
\begin{aligned}
A\mathbf x&=c_1A\mathbf v_1+\cdots+c_rA\mathbf v_r+c_{r+1}A\mathbf v_{r+1}+\cdots+c_n\mathbf v_n \\
&=c_1\sigma_1\mathbf u_1+\cdots+c_r\sigma_1\mathbf u_r+0+\cdots+0
\end{aligned}
$$
所以 $A\mathbf x\in\text{span}\{\mathbf u_1,\cdots,\mathbf u_r\}$ ，这说明矩阵  $U$ 的前 $r$ 列是 $\text{col }A$ 的一组单位正交基，因此 $\text{rank }A=r$ 。同时可知，对于任意的 $\mathbf x\in\text{span}\{\mathbf v_{r+1},\cdots,\mathbf v_n\}\iff A\mathbf x=0$ ，于是 $V$ 的后 $n-r$ 列是 $\text{null } A$ 的一组单位正交基。

同样通过 $A^TU=V\Sigma$  可说明 $V$ 的前 $r$ 列是 $\text{col }A^T$ 的一组单位正交基， $U$ 的后 $m-r$ 列是 $\text{null } A^T$ 的一组单位正交基。

### 奇异值分解的性质

设矩阵 $A=U\Sigma V^T$ ，秩 $\text{rank }A=r$ ，分别将 $U,\Sigma,V$ 进行分块
$$
U=(U_r,U_{m-r})  \\
V=(V_r,V_{n-r}) \\
\Sigma=\begin{bmatrix}\Lambda_r&O\\O&O\end{bmatrix}
$$
其中 $U_r=(\mathbf u_1,\cdots,\mathbf u_r)$ 为 $m\times r$维矩阵， $V_r=(\mathbf v_1,\cdots,\mathbf v_r)$ 为 $n\times r$维矩阵，$\Lambda_r=\text{diag}(\sigma_1,\cdots,\sigma_r)$ 为 $r$ 阶对角阵。应用矩阵乘法的性质，奇异值分解可以简化为
$$
A=U_r\Lambda_r V^T_r
$$
这个分解称为**简化奇异值分解**。

<kbd>性质</kbd>：

1. 奇异值分解可理解为将线性变换分解为三个简单的变换：正交变换 $V^T$，伸缩变换 $\Sigma$ 和正交变换 $U$ 。

2. 矩阵 $A$ 的奇异值分解中，奇异值是唯一的，但矩阵 $U,V$ 不是唯一的。

3. 令 $\lambda$ 为$A^TA$ 的一个特征值，$\mathbf v$ 是对应的特征向量，则
   $$
   \|A\mathbf v\|^2=\mathbf v^TA^TA\mathbf v=\lambda\mathbf v^T\mathbf v=\lambda\|\mathbf v\|
   $$

4. 易知 $AV=U\Sigma$  或 $A^TU=V\Sigma^T$，则左奇异向量和右奇异向量存在关系
   $$
   A\mathbf v_j=\sigma_j\mathbf u_j \\
   A^T\mathbf u_j=\sigma_j\mathbf v_j
   $$

### 矩阵的外积展开式

矩阵 $A=U\Sigma V^T$ 可展开为若干个秩为1的 $m\times n$矩阵之和
$$
A=\sigma_1\mathbf u_1\mathbf v_1^T+\sigma_2\mathbf u_2\mathbf v_2^T+\cdots+\sigma_r\mathbf u_r\mathbf v_r^T
$$

上式称为矩阵 $A$ 的外积展开式。

在长方形对角矩阵 $\Sigma$ 中奇异值按从大到小的顺序排列  $\sigma_1\geqslant\sigma_2\geqslant\cdots\geqslant\sigma_r>0$ 。在很多情况下，由于奇异值递减很快，前10%甚至1%的奇异值的和就占了全部的奇异值之和的99%以上。因此，我们可以用前面 $k$ 个大的奇异值来近似描述矩阵。

奇异值分解也是一种矩阵近似的方法，这个近似是在矩阵范数意义下的近似。矩阵范数是向量范数的直接推广。
$$
\|A\|_2=(\sum_{j=1}^{n}\sum_{i=1}^{m} |a_{ij}|^2)^{1/2}
$$
可以证明
$$
\|A\|_2^2=\text{tr}(A^TA)= \sum_{i=1}^{r} \sigma_i^2
$$
设矩阵
$$
A_k=\sum_{i=1}^k\sigma_i\mathbf u_i\mathbf v_i^T
$$
则 $A_k$ 的秩为 $k$ ，矩阵 $A_k$ 称为 $A$ 的**截断奇异值分解**。并且 $A_k$ 是秩为 $k$ 时的最优近似，即 $A_k$ 为以下最优问题的解
$$
\min\|A-X\|_2 \\
\text{s.t. rank }A=k
$$
上式称为低秩近似(low-rank approximation)。于是奇异值分解可近似为
$$
A\approx \sum_{i=1}^k\sigma_i\mathbf u_i\mathbf v_i^T=U_{m\times k}\Sigma_{k\times k}V_{n\times k}^T
$$

其中 $k$ 是一个远远小于$m$和$n$的数，从计算机内存的角度来说，矩阵左(右)奇异向量和奇异值的存储要远远小于矩阵$A$的。所以，截断奇异值分解就是在计算精度和时间空间之间做选择。如果$k$越大，右边的三个矩阵相乘的结果越接近于$A$。

截断奇异值分解常用于图像压缩，如下图

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/digit_SVD.svg"  />

# 复数矩阵


矩阵 $A$ 的元素 $a_{ij}\in\Complex$ ，称为复矩阵。现将实数矩阵的一些概念推广到复数矩阵，相应的一些性质在复数矩阵同样适用。


<kbd>定义</kbd>：设复矩阵 $A=(a_{ij})_{m\times n}$

1. 矩阵 $\bar A=(\overline{a_{ij}})$ 称为矩阵 $A$ 的共轭矩阵.
2. 矩阵 $A^H=\bar A^T$ 称为矩阵 $A$ 的共轭转置，又叫Hermite转置。
3. 若 $A^H=A$，则称 $A$ 为 Hermitian 矩阵，是实数域对称阵的推广。
4. 若 $A^HA=AA^H=I$，即 $A^{-1}=A^H$ ，则称 $A$ 为酉矩阵(unitary matrix)，是实数域正交阵的推广。
5. 复向量长度 $\|\mathbf z\|^2=|z_1|^2+|z_1|^2+\cdots+|z_n|^2$
6. 内积 $\mathbf u^H\mathbf v=\bar u_1v_1+\bar u_2v_2+\cdots+\bar u_nv_n$
7. 正交 $\mathbf u^H\mathbf v=0$


<kbd>性质</kbd>：
- $\overline{A+B}=\overline A+\overline B$
- $\overline{kA}=\bar k \bar A$
- $\overline{AB}=\bar A\bar B$
- $(AB)^H=B^HA^H$
- 内积满足共轭交换率 $\mathbf u^H\mathbf v=\overline{\mathbf v^H\mathbf u}$
- Hermitian 矩阵可正交对角化 $A=P\Lambda P^{-1}=P\Lambda P^H$
- Hermitian 矩阵的每个特征值都是实数

# 附录

## 极大线性无关组

由向量组线性相关的定义，容易得到以下结论：

(1) 向量组线性相关$\iff$向量组中存在向量能被其余向量线性表示。
(2) 向量组线性无关$\iff$向量组中任意一个向量都不能由其余向量线性表示。

<kbd>线性等价</kbd>：给定两个向量组
$$
\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_r \\
\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_s
$$
如果其中的每个向量都能被另一个向量组线性表示，则两个向量组**线性等价**。

例如，向量组 $\mathbf a,\mathbf b,\mathbf a+\mathbf b$ 与向量组 $\mathbf a,\mathbf b$ 线性等价。

<kbd>极大线性无关组</kbd>：从向量组 $A$ 中取$r$ 个向量组成部分向量组 $\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_r$ ，若满足

(1) 部分向量组 $\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_r$ 线性无关
(2) 从$A$ 中任取$r+1$个向量组成的向量组 都线性相关。

则称向量组 $\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_r$ 为**极大线性无关组**(maximum linearly independent group)。极大线性无关组包含的向量个数为向量组的秩。

<kbd>性质</kbd>：

(1) 一个向量组的极大线性无关组不一定是惟一的；
(2) 一个向量组与它的极大线性无关组是等价的；
(3) 一个向量组的任意两个极大线性无关组中包含的向量个数相同，称为向量组的**秩**(rank)。全由零向量组成的向量组的秩为零；
(4) 两个线性等价的向量组的秩相等；
(5) 两个等价的向量组生成的向量空间相同。

## 向量叉积

平面叉积
$$
\begin{bmatrix}v_1\\v_2\end{bmatrix}\times\begin{bmatrix}w_1\\w_2\end{bmatrix}=\det\begin{bmatrix}v_1 & w_1\\ v_2 & w_2 \end{bmatrix}
$$
大小等于 $v,w$ 围成的平行四边形的面积

三维叉积
$$
\begin{bmatrix}v_1\\v_2\\v_3\end{bmatrix}\times\begin{bmatrix}w_1\\w_2\\w_3\end{bmatrix}=\det\begin{bmatrix}\mathbf i & v_1 & w_1\\\mathbf j & v_2 & w_2 \\\mathbf k & v_3 & w_3 \end{bmatrix}
$$
大小等于 $v,w$ 围成的平行六面体的体积，方向遵循右手定则。

