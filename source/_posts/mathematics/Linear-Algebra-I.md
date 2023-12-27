---
title: 线性代数(上册)
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
description: 本文从线性变换出发理解线性代数的本质
abbrlink: '40113498'
date: 2023-09-10 23:51:00
---

> [《线性代数的本质》 - 3blue1brown](https://www.bilibili.com/video/BV1ys411472E/)
> 高中数学A版选修4-2 矩阵与变换
> 《线性代数及其应用》(第五版)
> 《高等代数简明教程》- 蓝以中

# 向量空间

> In the beginning Grant created the space. And Grant said, Let there be vector: and there was vector.

## 向量及其性质

三维几何空间中的一个有向线段称为**向量**(vector)。本文统一用 $a,b,c,k,\lambda$ 表示标量，小写黑体字母 $\mathbf u,\mathbf v,\mathbf w,\mathbf a,\mathbf b,\mathbf x$ 表示向量。

向量通常定义两种运算：加法和数乘。加法遵循三角形法则(平行四边形法则)，数乘被称为缩放(scaling)。运算法则如下图

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/linear_operation.svg" style="zoom:90%;" />

<kbd>性质</kbd>：根据向量的几何性质可证明向量的加法和数乘满足以下八条性质：

1. 加法交换律：$\mathbf v+\mathbf w=\mathbf w+\mathbf v$
2. 加法结合律：$\mathbf u+(\mathbf v+\mathbf w)=(\mathbf u+\mathbf v)+\mathbf w$
3. 加法单位元：$\exists 0\in V,\ 0+\mathbf v=\mathbf v$
4. 加法逆元：$\exists (-\mathbf v)\in V,\ \mathbf v+(-\mathbf v)=0$
5. 数乘结合律：$a(b\mathbf v)=(ab)\mathbf v$
6. 数乘分配律：$a(\mathbf v+\mathbf w)=a\mathbf v+a\mathbf w$
7. 数乘分配律：$(a+b)\mathbf v=a\mathbf v+b\mathbf v$
8. 数乘单位元：$\exists 1\in\mathbb F,\ 1\mathbf v=\mathbf v$

向量空间是三维几何空间向高维空间的推广。线性代数中，每个向量都以坐标原点为起点，那么任何一个向量就由其终点唯一确定。从而，向量和空间中的点一一对应。因此，空间也可以看成由所有向量组成的集合，并且集合中的元素可以进行加法和数乘运算。于是，便有了向量空间的抽象定义。

 <kbd>向量空间</kbd>： 设 $V$ 为 $n$ 维向量的**非空集合**，$\mathbb F$ 是一个数域，若 $V$ 对于向量的加法和数乘两种运算封闭，那么称集合 $V$ 为数域 $F$ 上的**向量空间**(vector space)。所谓封闭是指

1. $\forall\mathbf v,\mathbf w\in V,\ \mathbf v+\mathbf w\in V$
2. $\forall\mathbf v\in V, c\in F,\ c\mathbf v\in V$

> 线性代数中的数域通常取全体实数，即 $\mathbb F=\R$。

例如：$n$维向量的全体生成实数域上的向量空间

$$
\R^n=\{\mathbf x=(x_1,x_2,\cdots,x_n)\mid x_1,x_2,\cdots,x_n\in\R\}
$$

<kbd>子空间</kbd>：设 $U$ 是向量空间 $V$ 的一个非空子集，如果$U$中的线性运算封闭，则 $U$ 也是向量空间，称为 $V$ 的**子空间**。

## 基与维数

仿照解析几何的基本方法，建立一个坐标系，实现空间内的点与有序实数对一一对应，从而空间内的向量与有序实数对也一一对应，这样就可以用代数方法来研究向量的性质。

为方便建立空间的坐标系，先定义几个概念。

<kbd>定义</kbd>：取向量空间 $V$ 内一个向量组 $\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_r$ 

1. 向量 $x_1\mathbf a_1+x_2\mathbf a_2+\cdots+x_r\mathbf a_r$ 称为向量组的一个**线性组合**(linear combination)

2. 向量组的所有线性组合构成的向量集称为由该向量组张成的空间，记作
   $$
   \text{span}\{\mathbf a_1,\cdots,\mathbf a_n\}=\{x_1\mathbf a_1+\cdots+x_n\mathbf a_n\mid x_1,\cdots,x_n\in\R\}
   $$
   如下图，若 $\mathbf u,\mathbf v\in\R^3$ 不共线，则 $\text{span}\{\mathbf u,\mathbf v\}$ 是$\R^3$中包含 $\mathbf u,\mathbf v$ 和原点的平面，图示

   ![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/span.svg)

3. 当且仅当系数 $x_1=x_2=\cdots=x_r=0$ 时，线性组合为零
   $$
   x_1\mathbf a_1+x_2\mathbf a_2+\cdots+x_r\mathbf a_r=0
   $$
   则称向量组**线性无关**(linearly independence)。反之，如果存在不全为零的数使上式成立，则称向量组**线性相关**(linearly dependence)。
   ![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/linear_representations.svg)

<kbd>定理</kbd>：若向量 $\mathbf v$ 可由线性无关的向量组$\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_r$ 线性表示，则表示系数是唯一的。

证明：设向量$\mathbf v$ 有两组表示系数
$$
\mathbf b=k_1\mathbf a_1+k_2\mathbf a_2+\cdots+k_r\mathbf a_r \\
\mathbf b=l_1\mathbf a_1+l_2\mathbf a_2+\cdots+l_r\mathbf a_r
$$
则有
$$
(k_1-l_1)\mathbf a_1+(k_1-l_2)\mathbf a_2+\cdots+(k_1-l_r)\mathbf a_r=0
$$
因为 $\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_r$ 线性无关，故必有
$$
k_1-l_1=k_1-l_1=\cdots=k_1-l_1=0
$$
即表示系数是唯一的。

接下来，我们自然想用一组线性无关的向量来张成整个向量空间。

<kbd>向量空间的基</kbd>：张成向量空间$V$的一个线性无关的向量集合称为该空间的一组**基**(basis)。基向量组所含向量的个数，称为向量空间 $V$的**维数**(dimension)，记为 $\dim V$。

> 可以证明，向量空间的任意一组基的向量个数是相等的。
> 单由零向量组成的向量空间$\{0\}$称为**零空间**。零空间的维数定义为零。

<kbd>基定理</kbd>：$n$ 维向量空间的任意 $n$ 个线性无关的向量构成空间的一组基。

## 向量的坐标运算

向量空间选定了基向量后，空间中全体向量的集合与全体有序实数组的集合之间就建立了一一 对应的关系。

<kbd>坐标</kbd>：设向量组 $\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n$ 是线性空间 $V$ 的一组基，则空间内任一向量 $\mathbf v\in V$ 都可表示为基向量的唯一线性组合
$$
\mathbf v=x_1\mathbf a_1+x_2\mathbf a_2+\cdots+x_n\mathbf a_n
$$
有序数组 $x_1,x_2,\cdots,x_n$ 称为向量$\mathbf v$ 在基 $\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n$ 下的**坐标**，一般记作
$$
\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}\quad \text{or}\quad
(x_1,x_2,\cdots,x_n)
$$
类似于三维几何空间，由$n$个有序数构成的向量称为$n$维向量。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/coordinate.svg)

例：设 $\mathbf v_1=\begin{bmatrix}3\\6\\2\end{bmatrix},\mathbf v_2=\begin{bmatrix}-1\\0\\1\end{bmatrix},\mathbf x=\begin{bmatrix}3\\12\\7\end{bmatrix}$ 。判断 $\mathbf x$ 是否在 $H=\text{span }\{\mathbf v_1,\mathbf v_2\}$ 中，如果是，求 $\mathbf x$ 相对于基向量$B=\{\mathbf v_1,\mathbf v_2\}$ 的坐标。

解：如果 $\mathbf x$ 在 $H=\text{span }\{\mathbf v_1,\mathbf v_2\}$ 中，则下列方程是有解的
$$
c_1\begin{bmatrix}3\\6\\2\end{bmatrix}+c_2\begin{bmatrix}-1\\0\\1\end{bmatrix}=\begin{bmatrix}3\\12\\7\end{bmatrix}
$$
如果数 $c_1,c_2$存在，则它们是 $\mathbf x$ 相对于$B$ 的坐标。由初等行变换得
$$
\begin{bmatrix}\begin{array}{cc:c}
3&-1&3\\6&0&12\\2&1&7
\end{array}\end{bmatrix}\to
\begin{bmatrix}\begin{array}{cc:c}
1&0&2\\0&1&3\\0&0&0
\end{array}\end{bmatrix}
$$
于是， $\mathbf x$ 相对于$\mathbf v_1,\mathbf v_2$ 的坐标
$$
\mathbf v_B=\begin{bmatrix}3\\2\end{bmatrix}
$$

> 有时为了区分坐标的基向量，向量 $\mathbf v$ 在基 $B=\{\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n\}$ 下的坐标，记作 $\mathbf v_B$

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/coordinate2.svg)

建立了坐标之后，$V$中抽象的向量 $\mathbf v$ 和$\R^n$中具体的数组 $(x_1,x_2,\cdots,x_n)^T$ 实现了一一对应，并且向量的线性运算也可以表示为坐标的线性运算。

设 $\mathbf v,\mathbf w\in V$，有
$$
\mathbf v=v_1\mathbf a_1+v_2\mathbf a_2+\cdots+v_n\mathbf a_n\\
\mathbf w=w_1\mathbf a_1+w_2\mathbf a_2+\cdots+w_n\mathbf a_n
$$

向量加法运算
$$
\mathbf v+\mathbf w=(v_1+w_1)\mathbf a_1+(v_2+w_2)\mathbf a_2+\cdots+(v_n+w_n)\mathbf a_n 
$$
即对应的坐标运算为
$$
\begin{bmatrix}v_1\\ v_2\\ \vdots \\ v_n\end{bmatrix}+
\begin{bmatrix}w_1\\ w_2\\ \vdots \\ w_n\end{bmatrix}=
\begin{bmatrix}v_1+w_1\\ v_2+w_2\\ \vdots \\ v_n+w_n\end{bmatrix}
$$

向量数乘运算
$$
c\mathbf v=(cv_1)\mathbf a_1+(cv_2)\mathbf a_2+\cdots+(cv_n)\mathbf a_n
$$
即对应的坐标运算为
$$
c\begin{bmatrix}v_1\\ v_2\\ \vdots \\ v_n\end{bmatrix}=
\begin{bmatrix}cv_1\\ cv_2\\ \vdots \\ cv_n\end{bmatrix}
$$

向量的坐标取值依托于坐标系的基向量。选取的基向量不同，其所对应的坐标值就不同。当然，基向量自身的坐标总是：

$$
\mathbf e_1=\begin{bmatrix}1\\0\\\vdots\\0\end{bmatrix},\quad
\mathbf e_2=\begin{bmatrix}0\\1\\\vdots\\0\end{bmatrix},\quad
\cdots,\quad
\mathbf e_n=\begin{bmatrix}0\\0\\\vdots\\1\end{bmatrix},\quad
$$
这种坐标形式通常称为**标准向量组**(或**单位坐标向量组**)。

总之，在$n$维向量空间 $V_n$ 中任取一组基，则 $V_n$ 中的向量与 $\R^n$ 中的数组之间就有一一对应的关系，且这个对应关系保持线性组合(线性运算)的一一对应。接下来我们将默认使用标准坐标系：坐标原点为 $O$，基向量组为 $\mathbf e_1,\mathbf e_2,\cdots,\mathbf e_n$ 。**后续将对向量实体和坐标不做区分**。

# 线性变换与矩阵

## 线性变换与二阶方阵

本节从二维平面出发学习线性代数。通常选用平面坐标系 $Oxy$ ，基向量为 $\mathbf i,\ \mathbf j$，平面内的任意向量都可以写成基向量的线性组合
$$
\mathbf v=x\mathbf i+y\mathbf j
$$
这样，平面内的点和有序实数对 $(x,y)$ 一一对应。借助平面坐标系，我们可以从代数的角度来研究几何变换。

<img src='https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/linear_transformation_example.svg'/>

变换与函数类似，函数把数映射到数，变换把点(向量)映射到点(向量)。
$$
T:\quad \mathbf v\mapsto T(\mathbf v)
$$

例如，(1) 平面内任意一点 $P(x,y)$ 绕原点$O$ 逆时针方向旋转 $60\degree$ 角得到点 $P'(x',y')$，坐标变换公式为
$$
\begin{cases}
x'=\frac{1}{2}x-\frac{\sqrt 3}{2}y \\
y'=\frac{\sqrt 3}{2}x+\frac{1}{2}y
\end{cases}
$$
可写为向量形式
$$
\begin{bmatrix}x'\\y'\end{bmatrix}=
x\begin{bmatrix}\frac{1}{2}\\\frac{\sqrt 3}{2}\end{bmatrix}+
y\begin{bmatrix}-\frac{\sqrt 3}{2}\\\frac{1}{2}\end{bmatrix}
$$

(2) 平面内任意一点 $P(x,y)$ 关于 $y$ 轴的对称点 $P'(x',y')$的表达式为
$$
\begin{cases}
x'=-x \\
y'=y
\end{cases}
$$
可写为向量形式
$$
\begin{bmatrix}x'\\y'\end{bmatrix}=
x\begin{bmatrix}-1\\0\end{bmatrix}+
y\begin{bmatrix}0\\1\end{bmatrix}
$$

事实上，在平面坐标系 $Oxy$ 中，很多几何变换都具有如下坐标变换公式
$$
\begin{cases}
x'=ax+by \\
y'=cx+dy
\end{cases}
$$
向量形式为
$$
\begin{bmatrix}x'\\y'\end{bmatrix}=
x\begin{bmatrix}a\\c\end{bmatrix}+
y\begin{bmatrix}b\\d\end{bmatrix}
$$
其中 $(x',y')$为平面内任意一点 $(x,y)$ 变换后的点。我们把形如上式的几何变换叫做**平面线性变换**。

容易证明，线性变换满足下列两条性质

(1) 可加性：$T(\mathbf v+\mathbf w)=T(\mathbf v)+T(\mathbf w)$
(2) 伸缩性：$T(c\mathbf v)=cL(\mathbf v)$

事实上，这两条性质才是线性变换的严格定义。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/linear_transformation_additivity.svg" style="zoom:80%;" />

为了进一步了解线性变换的本质，取任意向量 $\mathbf v=x\mathbf i+y\mathbf j$ ，在线性变换 $T$ 的作用下
$$
T(\mathbf v)=T(x\mathbf i+y\mathbf j)=xT(\mathbf i)+yT(\mathbf j)
$$
可知，==变换后的向量 $T(\mathbf v)$ 由变换后的基向量以同样的系数完全确定==。设变换后的基向量分别为
$$
T(\mathbf i)=a\mathbf i+c\mathbf j=\begin{bmatrix}a\\c\end{bmatrix},\quad 
T(\mathbf j)=b\mathbf i+d\mathbf j=\begin{bmatrix}b\\d\end{bmatrix}
$$

> 注意：本章线性变换中的坐标始终使用最初的 $Oxy$ 坐标系。

于是，线性变换 $T:\mathbf v\mapsto T(\mathbf v)$ 对应的坐标运算为
$$
\begin{bmatrix}x'\\y'\end{bmatrix}=
x\begin{bmatrix}a\\c\end{bmatrix}+
y\begin{bmatrix}b\\d\end{bmatrix}
$$
由于上述变换由变换后的基向量唯一确定，我们可以按顺序写为数表的形式

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/define_matrix.svg"  />

我们把这个数表称为二阶矩阵，一般用大写英文字母表示。变换后的向量则定义为矩阵与向量的乘积
$$
\begin{bmatrix}a & b\\c & d\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=
x\begin{bmatrix} a \\ c \end{bmatrix}+
y\begin{bmatrix} b \\ d \end{bmatrix}=
\begin{bmatrix} ax+by \\ cx+dy \end{bmatrix}
$$
可知，矩阵代表一个特定的线性变换，==我们完全可以把矩阵的列看作变换后的基向量，矩阵向量乘法就是将线性变换作用于给定向量==。

> Grant：矩阵最初的定义就来自线性变换。

至此，任何一个线性变换都可以写为矩阵与向量乘积的形式。反之，确定了坐标系后，任何一个矩阵都唯一确定了一个线性变换。矩阵和向量的乘积与线性变换实现了一一对应。

**一般地，直线在线性变换后仍然保持直线**。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/vector_equation_of_line.svg" />

证明：如图 $l$ 为向量 $\mathbf w_1,\mathbf w_2$ 终点所确定的直线，$\mathbf v$ 为终点在直线 $l$ 上的任意向量。
$$
\mathbf v=\mathbf w_1+\lambda(\mathbf w_2-\mathbf w_1)=(1-\lambda)\mathbf w_1+\lambda \mathbf w_2 \quad (\lambda\in\R)
$$
令 $\lambda_1+\lambda_2=1$ 则
$$
\mathbf v=\lambda_1 \mathbf w_1+\lambda_2 \mathbf w_2
$$
这就是由向量  $\mathbf w_1,\mathbf w_2$ 的终点所确定的直线的向量形式。由线性变换的基本性质可知，直线 $l$ 在线性变换 $A$ 的作用下变成
$$
\mathbf v'=A(\lambda_1 \mathbf w_1+\lambda_2 \mathbf w_2)=\lambda_1 A\mathbf w_1+\lambda_2 A\mathbf w_2
$$
(1) 如果 $A\mathbf w_1\neq A\mathbf w_2$，那么 $\mathbf v'$ 表示由向量 $A\mathbf w_1,A\mathbf w_2$ 的终点确定的直线。此时矩阵 $A$ 对应的线性变换把直线变成直线；
(2) 如果 $A\mathbf w_1 = A\mathbf w_2$，那么 $\lambda_1 A\mathbf w_1+\lambda_2 A\mathbf w_2=A\mathbf w_1$ 。由于向量 $A\mathbf w_1$ 的终点是一个确定的点，因而，矩阵 $A$ 所对应的线性变换把直线 $l$ 映射成了一个点 $A\mathbf w_1$ 。

## 常见的线性变换

> Grant：我们可以使用无限网格刻画二维空间所有点的变换。==线性变换是操作空间的一种手段，它能够保持网格线平行且等距，并保持原点不动==。

我们已经知道，在线性变换的作用下，直线仍然保持直线(或一个点)。为了方便，我们只考虑在平面直角坐标系内，单位正方形区域的线性变换。

根据向量加法的平行四边形法则，单位正方形区域可用向量形式表示为
$$
\begin{bmatrix}x\\y\end{bmatrix}=x\mathbf i+y\mathbf j  \quad(0\leqslant x,y\leqslant 1)
$$
由线性变换基本性质知，变换后的区域为
$$
A\begin{bmatrix}x\\y\end{bmatrix}=x(A\mathbf i)+y(A\mathbf j) \quad(0\leqslant x,y\leqslant 1)
$$

表示以 $A\mathbf i,A\mathbf j$ 为邻边的平行四边形区域。因此，我们只需考虑单位向量 $\mathbf i,\mathbf j$ 在线性变换作用下的结果，就能得到单位正方形区域在线性变换作用下所变成的图形。

<img src='https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/linear_transformation_square.svg'/>

**恒等变换**：把平面内任意一点 $P(x,y)$ 变成它本身，记为 $I$ 。对应的矩阵称为单位阵
$$
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
$$

**旋转变换**：(rotations)平面内任意一点 $P(x,y)$ 绕原点$O$按逆时针方向旋转 $\theta$ 角，记为 $R_{\theta}$ 。对应的矩阵为
$$
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/rotations.svg" />

**切变变换**：(shears)平行于 $x$ 轴的切变变换对应的矩阵为
$$
\begin{bmatrix}
1 & k\\
0 & 1
\end{bmatrix}
$$
类似的，平行于 $y$ 轴的切变变换对应的矩阵为
$$
\begin{bmatrix}
1 & 0\\
k & 1
\end{bmatrix}
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/shears.svg" />

**反射变换**：(reflection)一般的我们把平面内任意一点 $P(x,y)$ 关于直线 $l$ 对称的线性变换叫做关于直线 $l$ 的反射变换。

(1) 关于 $y$ 轴的反射变换对应的矩阵为
$$
\begin{bmatrix}
-1 & 0\\
0 & 1
\end{bmatrix}
$$
(2) 关于直线 $y=x$ 的反射变换对应的矩阵为
$$
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
$$
(3) 关于直线 $y=kx$ 的反射变换对应的矩阵为
$$
\frac{1}{k^2+1}\begin{bmatrix}
1-k^2 & 2k\\
2k & k^2-1
\end{bmatrix}
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/reflection.svg"  />

**伸缩变换**：(stretching)将每个点的横坐标变为原来的 $k_1$ 倍，纵坐标变为原来的 $k_2$ 倍，其中 $k_1,k_2\neq0$ 。对应的矩阵为
$$
\begin{bmatrix}
k_1 & 0\\
0 & k_2
\end{bmatrix}
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/stretching.svg"  />

**投影变换**：(projection)平面内任意一点 $P(x,y)$ 在直线 $l$ 的投影称为关于直线 $l$ 的投影变换。

(1) 关于 $x$ 轴的投影变换对应的矩阵为
$$
\begin{bmatrix}
1 & 0\\
0 & 0
\end{bmatrix}
$$
(2) 关于 $y$ 轴的投影变换对应的矩阵为
$$
\begin{bmatrix}
0 & 0\\
0 & 1
\end{bmatrix}
$$
(3) 关于直线 $y=kx$ 的投影变换对应的矩阵为
$$
\frac{1}{\sqrt{k^2+1}}\begin{bmatrix}
1 & k\\
k & k^2
\end{bmatrix}
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/projection.svg"  />

**平移变换**：形如 $(x,y)\mapsto (x+h,y+k)$ 的平移变换并不是线性变换，我们无法直接使用矩阵向量乘法。对此可以引入**齐次坐标**：平面内的每个点 $(x,y)$ 都可以对应于空间中的点 $(x,y,1)$ 。平移变换可以用齐次坐标写成变换 $T:(x,y,1)\mapsto (x+h,y+k,1)$，对应的矩阵为
$$
\begin{bmatrix}
1 & 0 & h \\
0 & 1 & k \\
0 & 0 & 1
\end{bmatrix}
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/Translation.svg)

## 复合变换与矩阵乘法

平面内任意一向量，依次做旋转变换 $R_{\theta_1}:\begin{bmatrix}
\cos{\theta_1} & -\sin{\theta_1}\\
\sin{\theta_1} & \cos{\theta_1}
\end{bmatrix}$ 和 $R_{\theta_2}:\begin{bmatrix}
\cos{\theta_2} & -\sin{\theta_2}\\
\sin{\theta_2} & \cos{\theta_2}
\end{bmatrix}$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/composite_transformation.svg" />

很显然最终作用的效果可以用一个变换 $R_{\theta_1+\theta_2}$ 来表示，对应的矩阵为
$$
\begin{bmatrix}
\cos{(\theta_1+\theta_2)} & -\sin{(\theta_1+\theta_2)}\\
\sin{(\theta_1+\theta_2)} & \cos{(\theta_1+\theta_2)}
\end{bmatrix}
$$
旋转变换 $R_{\theta_1+\theta_2}$仍然是线性变换。

一般地，设矩阵 $A=\begin{bmatrix}a_1 & b_1\\ c_1 & d_1\end{bmatrix},B=\begin{bmatrix}a_2 & b_2\\ c_2 & d_2\end{bmatrix}$，他们对应的线性变换分别为 $f$ 和 $g$ 。

平面上任意一个向量 $\mathbf v=\begin{bmatrix} x \\ y \end{bmatrix}$ 依次做变换 $g$ 和 $f$ ，其作用效果为
$$
f(g(\mathbf v))=A(B\mathbf v)
$$

> Grant：线性变换的本质主要在于追踪基向量变换后的位置。

接下来，我们追踪变换过程中基向量的位置。由矩阵向量乘法的定义知道，基向量 $\mathbf i,\mathbf j$ 经过矩阵 $B$ 变换后(第一次变换)的位置为
$$
B\mathbf i=\begin{bmatrix}a_2\\c_2\end{bmatrix},\quad 
B\mathbf j=\begin{bmatrix}b_2\\d_2\end{bmatrix}
$$
基向量 $B\mathbf i,B\mathbf j$ 又经过矩阵 $A$ 变换后的最终位置为
$$
\mathbf i':\begin{bmatrix}a_1 & b_1\\ c_1 & d_1\end{bmatrix}
\begin{bmatrix}a_2\\ c_2\end{bmatrix}=
a_2\begin{bmatrix}a_1\\ c_1\end{bmatrix}+
c_2\begin{bmatrix}b_1\\d_1\end{bmatrix}=
\begin{bmatrix}a_1a_2+b_1c_2 \\ c_1a_2+d_1c_2\end{bmatrix} \\
\mathbf j':\begin{bmatrix}a_1 & b_1\\ c_1 & d_1\end{bmatrix}
\begin{bmatrix}b_2\\ d_2\end{bmatrix}=
b_2\begin{bmatrix}a_1\\ c_1\end{bmatrix}+
d_2\begin{bmatrix}b_1\\d_1\end{bmatrix}=
\begin{bmatrix}a_1b_2+b_1d_2\\c_1b_2+d_1d_2\end{bmatrix}
$$
从而，对任意向量 $\mathbf v=\begin{bmatrix} x \\ y \end{bmatrix}$ 依次做变换 $B$ 和 $A$ ，其总体作用效果为
$$
A(B\mathbf v)=x\mathbf i'+y\mathbf j'=\begin{bmatrix}a_1a_2+b_1c_2 & a_1b_2+b_1d_2\\ c_1a_2+d_1c_2 & c_1b_2+d_1d_2\end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}
$$
这也是一个线性变换，我们称为**复合变换**(composite transformation)，记为 $f\circ g$ 。

在此，我们定义复合变换 $f\circ g$ 为矩阵$A,B$ 的乘积，记为
$$
AB=\begin{bmatrix}a_1 & b_1\\ c_1 & d_1\end{bmatrix}
\begin{bmatrix}a_2 & b_2\\ c_2 & d_2\end{bmatrix}=
\begin{bmatrix}a_1a_2+b_1c_2 & a_1b_2+b_1d_2\\ c_1a_2+d_1c_2 & c_1b_2+d_1d_2\end{bmatrix}
$$

> 注意：矩阵乘积的次序与复合变换相同，从右向左相继作用。

由定义易知，对任意向量 $\mathbf v$ 有
$$
(AB)\mathbf v=A(B\mathbf v)
$$

## 矩阵的定义

接下来，我们将矩阵的概念推广到高维空间。高维线性空间中的变换与二维空间中的变换类似。

<kbd>矩阵</kbd>: $m\times n$ 个数按一定次序排成的数表称为**矩阵**
$$
\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1n} \\
a_{21}&a_{22}&\cdots&a_{2n} \\
\vdots&\vdots&\ddots&\vdots \\
a_{m1}&a_{m2}&\cdots&a_{mn} \\
\end{bmatrix}
$$
常用大写英文字母表示矩阵，如$A$或$A_{m× n}$。矩阵中的每个数 $a_{ij}$ 称为它的元素(entry)，有时矩阵也记作  $(a_{ij})$ 或 $(a_{ij})_{m× n}$ 。根据矩阵的元素所属的数域，可以将矩阵分为复矩阵和实矩阵。

**几种特殊的矩阵**：

1. 元素全为零的矩阵称为**零矩阵**(zero matrix)，记作$O$。
2. 只有一行的矩阵称为**行矩阵**(row matrix)或**行向量**；只有一列的矩阵称为**列矩阵**(column matrix)或**列向量**。行(列)矩阵通常用小写黑体字母表示，如 $\mathbf a,\mathbf x$。
3. 当行数和列数相等时的矩阵 $A_{n\times n}$ 称为**$n$ 阶方阵**(n-order square matrix)。
4. 不在主对角线上的元素全为零的方阵称为**对角阵**(diagonal matrix)，记作 $\mathrm{diag}(a_1,a_2,\cdots,a_n)$
5. 主对角线上的元素全为1的对角阵，称为**单位阵**(identity matrix)。记$n$ 阶单位阵记作$E_n$或$I_n$

**矩阵的线性运算**：因为矩阵 $A_{m\times n}$ 的各列是 $m$维向量，写作 $A=\begin{bmatrix}\mathbf a_1&\mathbf a_2&\cdots&\mathbf a_n\end{bmatrix}$ ，因此矩阵可看作向量集，向量的线性运算自然推广到矩阵。

设矩阵$A=(a_{ij})$ 与 $B=(b_{ij})$ 

1. 他们的对应元素完全相同 $a_{ij}=b_{ij}$，则称矩阵 $A$ 与 $B$ 相等，记作$A=B$；
2. 矩阵的加法定义为 $A+B=(a_{ij}+b_{ij})$ 
3. 矩阵的数乘定义为$kA=(ka_{ij})$

<kbd>性质</kbd>：线性运算满足以下性质

1. 加法交换律：$A+B=B+A$
2. 加法结合律：$A+(B+C)=(A+B)+C$
3. 零矩阵：$O+A=A$
4. 负矩阵：$A+(-A)=O$
5. 数乘结合律：$k(lA)=(kl)A$
6. 数乘分配律：$k(A+B)=kA+kB$
7. 数乘分配律：$(k+l)A=kA+lA$
8. 数乘单位元：$1A=A$

**矩阵向量的乘法**： 矩阵与向量的乘法来源于线性变换，它有着直观的、深刻的几何背景。设$m\times n$ 维矩阵$A=(a_{ij})$ 与 $n$维向量 $\mathbf v=(x_1,x_2,\cdots,x_n)^T$ 的乘积
$$
\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1n} \\
a_{21}&a_{22}&\cdots&a_{2n} \\
\vdots&\vdots&\ddots&\vdots \\
a_{m1}&a_{m2}&\cdots&a_{mn} \\
\end{bmatrix}
\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}=
x_1\begin{bmatrix}a_{11}\\a_{21}\\\vdots\\a_{m1}\end{bmatrix}+\cdots+
x_n\begin{bmatrix}a_{1n}\\a_{2n}\\\vdots\\a_{mn}\end{bmatrix}=
\begin{bmatrix}\sum_{j=1}^na_{1j}x_j\\\sum_{j=1}^na_{2j}x_j\\\vdots\\\sum_{j=1}^na_{mj}x_j\end{bmatrix}
$$
一般地，$m\times n$ 维的矩阵，表示将 $n$ 维空间中的向量映射到 $m$ 维空间中。矩阵的第$j$列表示第 $j$ 个基向量变换后的坐标。

**矩阵乘法**：矩阵与矩阵乘法来源于复合线性变换。设矩阵$A=(a_{ij})_{m\times n}$与$B=(b_{ij})_{n\times p}$，向量 $\mathbf v=(x_1,x_2,\cdots,x_p)$ ，用 $\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_p$表示矩阵 $B$ 的各列，则
$$
B\mathbf v=x_1\mathbf b_1+x_2\mathbf b_2+\cdots+x_p\mathbf b_p
$$
由线性变换的性质
$$
\begin{aligned}
A(B\mathbf v)&=A(x_1\mathbf b_1)+A(x_2\mathbf b_2)+\cdots+A(x_p\mathbf b_p) \\
&=x_1A\mathbf b_1+x_2A\mathbf b_2+\cdots+x_pA\mathbf b_p \\
&=\begin{bmatrix}A\mathbf b_1&A\mathbf b_2&\cdots&A\mathbf b_p\end{bmatrix}\mathbf v
\end{aligned}
$$
于是可定义矩阵的乘积 $AB$ 为$m\times p$ 矩阵
$$
AB=A\begin{bmatrix}\mathbf b_1&\mathbf b_2&\cdots&\mathbf b_p\end{bmatrix}=
\begin{bmatrix}A\mathbf b_1&A\mathbf b_2&\cdots&A\mathbf b_p\end{bmatrix}
$$
矩阵 $A$的列数必须和$B$ 的行数相等，乘积才有意义 。之前定义的矩阵向量乘法是矩阵乘法的特例。通常，更方便的方法是用元素定义矩阵乘法。设乘积 $AB=(c_{ij})_{m× p}$。则元素
$$
c_{ij}=a_{i1}b_{1j}+a_{i2}b_{2j}+\cdots+a_{ip}b_{pj}
$$
<kbd>性质</kbd>：矩阵乘法满足以下性质

1. 矩阵乘法满足结合率：$A(BC)=(AB)C$
2. 矩阵乘法满足左分配律：$A(B+C)=AB+AC$
3. 矩阵乘法满足右分配律：$(B+C)A=BA+CA$
4. 矩阵乘法满足数乘分配律：$k(AB)=(kA)B=A(kB)$
5. 矩阵乘法单位元：$IA=AI=A$

证明：(1) 可从矩阵乘法的定义证明满足结合率。从线性变换角度来看，对于复合变换 $A(BC)$ 和 $(AB)C$ 是同样的变换，且依次作用的顺序并不会发生改变，变换的最终结果自然不变。
$$
\mathbf v\xrightarrow{C}C\mathbf v\xrightarrow{B}BC\mathbf v\xrightarrow{A}ABC\mathbf v
$$

注意：

1. 矩阵乘法不满足交换率，即一般情况下 $AB\neq BA$
2. 矩阵乘法不满足消去率，即若 $AB=AC$，不能推出 $B=C$ ；同样由 $AB=O$，不能推出 $A=O$  或 $B=O$。

证明：(1) 一般地，复合变换 $f\circ g\neq g\circ f$ ，自然 $AB\neq BA$，矩阵乘法不满足交换率。
(2) 可举例证明矩阵乘法不满足消去率

设矩阵
$$
A=\begin{bmatrix}0&1&0\\ 0&0&1\\ 0&0&1\end{bmatrix},\quad
B=\begin{bmatrix}0&0&1\\ 0&0&0\\ 0&0&0\end{bmatrix}
$$
则有
$$
AB=\begin{bmatrix}0&1&0\\ 0&0&1\\ 0&0&1\end{bmatrix}
\begin{bmatrix}0&0&1\\ 0&0&0\\ 0&0&0\end{bmatrix}=
\begin{bmatrix}0&0&0\\ 0&0&0\\ 0&0&0\end{bmatrix}=O \\
BA=\begin{bmatrix}0&0&1\\ 0&0&0\\ 0&0&0\end{bmatrix}
\begin{bmatrix}0&1&0\\ 0&0&1\\ 0&0&1\end{bmatrix}=
\begin{bmatrix}0&0&1\\ 0&0&0\\ 0&0&0\end{bmatrix}\neq O
$$

## 列空间与基

<kbd>定义</kbd>：为方便使用，先介绍几个简单的定义

1. 线性变换是一种映射，称变换后的向量 $T(\mathbf v)$ 为向量 $\mathbf v$ 在映射 $T$ 下的**像**，而称 $\mathbf v$ 为 $T(\mathbf v)$ 在映射 $T$ 下的**原像**。

2. 线性变换 $T$ 的像集$T(V)$是一个线性空间，称为线性变换 $T$ 的**值域**，记作
   $$
   \text{range}(T)=\{T(\mathbf v)\mid\mathbf v\in V\}
   $$

3. 在前面几节的分析中，我们始终将矩阵的列看成是向量。而这些列向量所张成的空间，称为**列空间**，若 $A=(\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n)$
   $$
   \text{col }A=\text{span}\{\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n\}
   $$

我们已经知道，变换后的向量 $A\mathbf v$ 是变换后的基向量以同样的系数线性组合，而矩阵的列就是基向量变换之后的位置。因此，矩阵 $A$ 线性变换后的空间即是矩阵 $A$ 的列空间
$$
\text{col }A=\text{range }A=\{A\mathbf v\mid\mathbf v\in V\}
$$
<kbd>定理</kbd>：矩阵 $A$ 的主元列构成 $\text{col }A$ 的一组基。

下面两个例子给出对列空间求基的简单算法。

例1：求 $\text{Col }B$ 的一组基，其中
$$
B=(\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n)=\begin{bmatrix}1&4&0&2&0\\ 0&0&1&-1&0\\ 0&0&0&0&1\\0&0&0&0&0\end{bmatrix}
$$
事实上，$B$ 的每个非主元列都是主元列的线性组合 $\mathbf b_2=4\mathbf b_1,\mathbf b_4=2\mathbf b_1-\mathbf b_3$  且主元列时线性无关的，所以主元列构成列空间的一组基 $\text{col }B=\text{span }\{\mathbf b_1,\mathbf b_3,\mathbf b_5\}$ 。

当矩阵不是阶梯型矩阵时，回顾矩阵 $A=(\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n)$ 中列向量间的线性关系都可以用方程 $A\mathbf x=0$ 的形式刻画。当 $A$ 被行简化为阶梯型矩阵 $B=(\mathbf b_1,\mathbf b_2,\cdots,\mathbf b_n)$ 时，即存在可逆矩阵 $P$ 使 $B=PA$ 。若 $B$ 的列向量线性相关，即存在系数 $\mathbf x$ 使得 $B\mathbf x=0$ ，即
$$
x_1\mathbf b_1+x_2\mathbf b_2+\cdots+x_n\mathbf b_n=0
$$
同样的系数 $\mathbf x$ 也适用于矩阵 $A$ 的列向量，$A\mathbf x=P^{-1}B\mathbf x=0$，即
$$
x_1\mathbf a_1+x_2\mathbf a_2+\cdots+x_n\mathbf a_n=0
$$
==综上，即矩阵$A$的列与阶梯型矩阵 $B$ 的列具有完全相同的线性相关关系。==

例2：
$$
A=(\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n)=\begin{bmatrix}1&4&0&2&-1\\ 3&12&1&5&5\\ 2&8&1&3&2\\5&20&2&8&8\end{bmatrix}
$$
已知矩阵 $A$ 行等价于上例中的矩阵$B$ ，求  $\text{Col }A$ 的一组基。

由于上例中 $\mathbf b_2=4\mathbf b_1,\mathbf b_4=2\mathbf b_1-\mathbf b_3$ ，相关关系完全适用于矩阵 $A$ 的列向量 $\mathbf a_2=4\mathbf a_1,\mathbf a_4=2\mathbf a_1-\mathbf a_3$ 。于是线性无关集 $\mathbf a_1,\mathbf a_3,\mathbf a_5$ 是 $\text{Col }A$ 的一组基 $\text{col }A=\text{span }\{\mathbf a_1,\mathbf a_3,\mathbf a_5\}$。

> 注意：阶梯形矩阵的主元列通常不在原矩阵的列空间中。

## 矩阵的秩

**矩阵的秩**就是列空间的维度，记作 $\text{rank }A=\dim(\text{col }A)$。

前面介绍的都是方阵，表示向量空间到自身的映射。下面简单说下非方阵的映射关系。

一般地，$m\times n$ 维的矩阵，表示将 $n$ 维空间中的向量映射到 $m$ 维空间中。矩阵的第$j$列表示第 $j$ 个基向量变换后的坐标。例如：

$3\times 2$ 维矩阵是把二维空间映射到三维空间上，因为矩阵有两列，说明输入空间有两个基向量，三行表示每一个基向量在变换后用三个独立的坐标来描述。
$$
\begin{bmatrix}1&-1\\3&2\\0&3\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}=
\begin{bmatrix}1\\3\\0\end{bmatrix}x+
\begin{bmatrix}-1\\2\\3\end{bmatrix}y
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/matrix_3x2.svg)

$2\times 3$ 维矩阵是把三维空间映射到二维空间上，因为矩阵有三列，说明输入空间有三个基向量，二行表示每一个基向量在变换后用二个独立的坐标来描述。
$$
\begin{bmatrix}2&2&1\\1&0&-1\end{bmatrix}
\begin{bmatrix}x\\y\\z\end{bmatrix}=
\begin{bmatrix}2\\1\end{bmatrix}x+
\begin{bmatrix}2\\0\end{bmatrix}y+
\begin{bmatrix}1\\-1\end{bmatrix}z
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/matrix_2x3.svg)

若矩阵的秩等于列数，则称为**满秩矩阵**(full rank matrix)，零向量一定在列空间内，满秩变换中，唯一能落在原点的就是零向量自身。满秩矩阵的列即为列空间的基。

对于**非满秩矩阵**，意味着该线性变换会将空间压缩到一个更低维的空间，通俗来讲，就是会有一系列直线上不同方向的向量压缩为原点。

由此可得，**秩可以用来描述线性变换对空间的压缩程度**。


## 逆变换与逆矩阵

我们已经知道了矩阵与线性变换中的对应关系，试想一下，将变换后的向量还原到初始状态。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/inverse_transform.svg"  />

<kbd>逆矩阵</kbd>：对于 $n$ 阶方阵  $A$ ，如果存在 $n$ 阶方阵 $B$ ，使得
$$
AB=BA=I
$$
则称矩阵 $A$ **可逆**(invertible)，$B$ 是 $A$ 的**逆矩阵**。实际上， $A$ 的逆矩阵是唯一的，记为 $A^{-1}$。因为，若 $B,C$ 都是 $A$ 的逆矩阵，则

$$
B=(CA)B=C(AB)=C
$$

不可逆矩阵有时称为**奇异矩阵**，而可逆矩阵也称为**非奇异矩阵**。

<kbd>性质</kbd>：逆矩阵满足下列性质

1. $(A^{-1})^{-1}=A$
2. $(kA)^{-1}=\dfrac{1}{k}A^{-1},\quad(k\neq0)$
3. $(AB)^{-1}=B^{-1}A^{-1}$
4. $(A^T)^{-1}=(A^{-1})^T$

证明：(性质3)若方阵 $A,B$ 都可逆，则有
$$
(AB)(B^{-1}A^{-1})=(B^{-1}A^{-1})(AB)=I
$$
因此 $(AB)^{-1}=B^{-1}A^{-1}$ 。

从变换的角度考虑，复合变换的逆 $(f\circ g)^{-1}=g^{-1}\circ f^{-1}$ ，很容易理解。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/inverse_composite_transformation.svg"  />

(性质4)
$$
I=(AA^{-1})^T=(A^{-1})^TA^T,\quad I=(A^{-1}A)^T=A^T(A^{-1})^T
$$
因此 $(A^T)^{-1}=(A^{-1})^T$ 。

# 线性方程组

## 高斯消元法

客观世界最简单的数量关系是均匀变化的关系。在均匀变化问题中，列出的方程组是一次方程组，我们称之为线性方程组(Linear system of equations)。$n$元线性方程组的一般形式为
$$
\begin{cases} 
a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n=b_1 \\ 
a_{21}x_1+a_{22}x_2+\cdots+a_{2n}x_n=b_2 \\
\cdots\quad\cdots \\
a_{m1}x_1+a_{m2}x_2+\cdots+a_{mn}x_n=b_m
\end{cases}
$$
如果存在$n$个常数 $x_1=s_1,x_2=s_2,\cdots,x_n=s_n$ 满足线性方程组的所有方程，则称为线性方程组的一个**解**(solution)。方程组的所有解组成的集合称为这个方程组的**解集**。

解线性方程组的一般方法，是把方程组用一个更容易解的等价方程组 (即有相同解集的方程组)代替。用来化简线性方程组的三种基本变换是：

(1) 互换两个方程的位置；
(2) 把某一个方程的所有项乘以一个非零常数；
(3) 把某一个方程加上另一个方程的常数倍；

以上三种变换称为高斯消元法(Gaussian Elimination)。

例如，解方程组
$$
\begin{cases}
\begin{alignedat}{4} 
&\quad 2x_2&-\ \ x_3 &= 7 \\ 
x_1&+\ x_2&+2x_3& = 0 \\
x_1&+\ x_2&-\ \ x_3& = -6 \\
x_1&+3x_2&-2x_3&=1
\end{alignedat}
\end{cases}
$$
经过基本变换把线性方程组化成**阶梯形方程组**
$$
\begin{cases}
\begin{alignedat}{4} 
x_1&+x_2&-x_3& = -6 \\
&\quad 2x_2&-x_3 &= 7 \\ 
&\quad &\quad 3x_3& = 6 \\
&\quad &\quad 0& = 0
\end{alignedat}
\end{cases}
$$
 还可以进一步变换为**简化阶梯形方程组**
$$
\begin{cases} 
x_1 & & &=-9  \\ 
& x_2 & & = 5 \\
& & x_3& = 2 \\
& & 0& = 0
\end{cases}
$$
上面的简单例子代表了用消元法解线性方程组的一般方法和计算格式。

## 初等行变换

根据矩阵与向量的乘法定义，线性方程组可写为矩阵形式
$$
A\mathbf x=\mathbf b
$$
其中
$$
A=\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1n} \\
a_{21}&a_{22}&\cdots&a_{2n} \\
\vdots&\vdots&\ddots&\vdots \\
a_{m1}&a_{m2}&\cdots&a_{mn} \\
\end{bmatrix},\quad
\mathbf x=\begin{bmatrix}
x_1\\x_2\\\vdots\\x_n
\end{bmatrix},\quad
\mathbf b=\begin{bmatrix}
b_1\\b_2\\\vdots\\b_n
\end{bmatrix}
$$
矩阵 $A$ 称为**系数矩阵**， $\mathbf x$ 为**未知数向量**，$\mathbf b$ 为**常数向量**。

从上节求解线性方程组的过程中，不难发现，只是对线性方程组的系数和常数项进行了运算。因此，线性方程组可以用它的系数和常数项来求解。

为求解方便，把常数向量添加到系数矩阵最后一列，构成的矩阵
$$
\bar A=[A\mid b]=\begin{bmatrix}\begin{array}{ccc:c}
a_{11}&\cdots&a_{1n}&b_1 \\
a_{21}&\cdots&a_{2n}&b_2 \\
\vdots&\vdots&\ddots&\vdots \\
a_{m1}&\cdots&a_{mn}&b_m \\
\end{array}\end{bmatrix}
$$
称为方程组的**增广矩阵**(augmented matrix)。

**初等行变换**：上节所讲的三种基本变换对应于矩阵的下列变换：

(1) 行互换变换：对调矩阵的第$i$行和第$j$行 ，记为 $r_i\lrarr r_j$
(2) 行倍乘变换：矩阵的第$i$行乘以非零常数$k$，记为 $kr_i$
(3) 行倍加变换：将第$j$行的元素倍加到第$i$行，记作 $r_i+kr_j$

称为矩阵的**初等行变换**(elementary row transformation)。

**矩阵消元法**：在解线性方程组时，把它的增广矩阵经过初等行变换化成行阶梯形矩阵，写出相应的阶梯形方程组 ，进行求解；或者一直化成简化行阶梯形矩阵，写出它表示的简化阶梯形方程组，从而立即得出解。

上节例子中，增广矩阵经过初等行变换可简化为
$$
\bar A=\begin{bmatrix}\begin{array}{ccc:c}
0 & 2 & -1 & 7 \\
1 & 1 & 2  & 0\\
1 & 1 & -1 & -6 \\
1 & 3 & -2 & 1
\end{array}\end{bmatrix}\to
\begin{bmatrix}\begin{array}{ccc:c}
1 & 1 & -1 & -6 \\
0 & 2 & -1  & 7\\
0 & 0 & 3 & 6 \\
0 & 0 & 0 & 0
\end{array}\end{bmatrix}=B_1
$$
称形如 $B_1$ 的矩阵为**行阶梯形矩阵**(Row Echelon Form，REF)。其特点是：

(1) 若有零行(元素全为零的行)，零行均在非零行的下方；
(2) 非零行第一个非零元素(称为**主元**，pivot)以下的元素全为零。

使用初等行变换对行阶梯形矩阵进一步化简
$$
B_1=\begin{bmatrix}\begin{array}{ccc:c}
1 & 1 & -1 & -6 \\
0 & 2 & -1  & 7\\
0 & 0 & 3 & 6 \\
0 & 0 & 0 & 0
\end{array}\end{bmatrix}\to\begin{bmatrix}\begin{array}{ccc:c}
1 & 0 & 0 & -9 \\
0 & 1 & 0  & 5\\
0 & 0 & 1 & 2 \\
0 & 0 & 0 & 0
\end{array}\end{bmatrix}=B_2
$$
称形如 $B_2$ 的矩阵为**简化行阶梯形矩阵**(Reduced Row Echelon Form，RREF)。其特点是：

(1) 每个非零行主元都是1；
(2) 主元所在列的其他元素都是零。

通过简化行阶梯形矩阵，我们可以直接写出解 $x_1=-9,x_2=5,x_3=2$。

使用矩阵消元法，我们可以知道**任何矩阵都可以经过有限次初等行变换化成行阶梯形矩阵，任何矩阵也可进一步化成简化行阶梯形矩阵**。

从最后的简化行阶梯形矩阵可以直接写出一般解，但注意把自由变量的系数变号移到等式右边。

## 线性方程组的解

假设某方程组的增广矩阵行已变换为阶梯形矩阵
$$
\begin{bmatrix}\begin{array}{ccc:c}
1 & 0 & -5 & 1 \\
0 & 1 & 1 & 4\\
0 & 0 & 0 & 0
\end{array}\end{bmatrix}
$$
对应的线性方程组是
$$
\begin{cases}
\begin{alignedat}{4} 
x_1&&-5x_3& = 1 \\
&\quad\ x_2&+x_3 &= 4 \\ 
&\quad &\quad 0& =0
\end{alignedat}
\end{cases}
$$
方程组的解可显示表示为 $x_1=1+5x_3,\ x_2=4-x_3$ ，显然有无穷多组解。

把 $n$ 元线性方程组的增广矩阵化成行阶梯形矩阵后，若有 $r$ 个非零行，则行阶梯形矩阵有 $r$ 个主元。以主元为系数的末知量称为**主变量**，剩下的 $n-r$ 个未知量称为**自由变量**，其值可任取。

假设某方程组的增广矩阵行已变换为阶梯形矩阵
$$
\begin{bmatrix}\begin{array}{ccc:c}
2 & -3 & 2 & 1 \\
0 & 1 & -4 & 8\\
0 & 0 & 0 & 15 
\end{array}\end{bmatrix}
$$
对应的线性方程组是
$$
\begin{cases}
\begin{alignedat}{4} 
2x_1&-3x_2&+2x_3& = 1 \\
&\quad\ x_2&-4x_3 &= 8 \\ 
&\quad &\quad 0& = 15
\end{alignedat}
\end{cases}
$$
这个阶梯形方程组显然是矛盾的，故原方程组无解。

<kbd>解的情况</kbd>：线性方程组有解的充要条件是增广矩阵的增广列不是主元列，即行阶梯形方程组不包含矛盾方程。若线性方程组有解，则解有两种情况：(1) 当没有自由变量时，有唯 一解；(2) 当有自由变量是，有无穷多解。

## 向量方程

应用向量加法和数乘运算，线性方程组 $A\mathbf x=\mathbf b$ 可以写成**向量方程**
$$
x_1\mathbf a_1+x_2\mathbf a_2+\cdots+x_n\mathbf a_n=\mathbf b
$$
其中 $\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n$ 为系数矩阵 $A$ 的列向量组，$\mathbf b$ 为常数向量。它的一组解 $s=(x_1,x_2,\cdots,x_n)^T$ 称为方程组的**解向量**。

例如，方程组
$$
\begin{cases}
\begin{alignedat}{4} 
2x_1&-x_2&+x_3& = 4 \\
4x_1&+2x_2&-x_3& = -1
\end{alignedat}
\end{cases}
$$
可以表述为
$$
\begin{bmatrix}2\\4\end{bmatrix}x_1+
\begin{bmatrix}-1\\2\end{bmatrix}x_2+
\begin{bmatrix}1\\-1\end{bmatrix}x_3=
\begin{bmatrix}4\\-1\end{bmatrix}
$$
既然可表示为向量的形式，那么就可以从向量的角度分析。向量方程是否有解的问题等价于判断常数向量 $\mathbf b$ 能否由系数矩阵列向量组线性表示，即向量 $\mathbf b$ 是否属于系数矩阵的列空间 $\text{col }A=\text{span}\{\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n\}$。

<kbd>结论</kbd>：方程 $A\mathbf x=\mathbf b$有解的充要条件是 $\mathbf b$ 是 $A$ 的各列的线性组合。

以线性变换的角度理解，希望找出未知向量 $\mathbf x$ ，使得该向量在线性变换 $A$ 的作用下变成已知向量 $\mathbf b$。因此，我们可以从逆变换的角度获得未知向量。显然，如果变换后维度压缩，方程不一定有解。即列空间的维度低于未知向量维度。

## 齐次线性方程组的解

常数项都为零的线性方程组 $A\mathbf{x}=0$ 称为**齐次线性方程组**。向量方程为
$$
x_1\mathbf a_1+x_2\mathbf a_2+\cdots+x_n\mathbf a_n=0
$$

齐次线性方程组显然有一组解
$$
x_1=x_2=\cdots=x_n=0
$$
这组解称为**零解**或**平凡解**。除此之外的其他解称为**非零解**或**非平凡解**。

 方程 $A_{m\times n}\mathbf{x}=0$ 有非零解等价于 $A$ 的列向量组线性相关，即 $\text{rank}(A)<n$

齐次线性方程组的解有如下性质

1. 如果 $s_1,s_2$ 是齐次线性方程组的两个解向量，则 $s_1+s_2$ 也是方程组的解向量。
2. 如果 $s$ 是齐次线性方程组的解向量，则对任意常数$k$， $ks$ 也是方程组的解向量。

> 这两条性质只要直接代入向量方程进行验证就可以。 

显然，系数矩阵为 $A$ 的齐次线性方程组的解集 
$$
\text{null } A=\{\mathbf x|A\mathbf{x}=0\}
$$
满足向量空间的条件， 称为**零空间**(nullspace)或**核**(kernel)。解空间的一组基 $s_1,s_2,\cdots,s_{n-r}$ 称为该方程组的**基础解系**。==零空间的维数即为自由变量的个数==。

如果能找到基础解系，就能描述整个解空间。

<kbd>定理</kbd>：

1. 方程 $A_{m\times n}\mathbf{x}=0$ 有非零解的充要条件是 $\text{rank}(A)<n$。
2. 方程 $A_{m\times n}\mathbf{x}=0$ 基础解系中自由变量的个数等于 $n-\text{rank}(A)$。
3. 设 $A$ 是向量空间 $V$ 内的线性变换

$$
\dim V=\dim(\text{range }A)+\dim(\text{null } A)
$$


可以用系数矩阵的初等行变换来求基础解系。

示例：求下列齐次线性方程组的解集。
$$
\begin{cases}
x_2-x_3+x_4-x_5=0 \\
x_1+x_3+2x_4-x_5=0 \\
x_1+x_2+3x_4-2x_5=0 \\
2x_1+2x_2+6x_4-3x_5=0 
\end{cases}
$$
解：先做矩阵消元法获得阶梯形矩阵和简化阶梯形矩阵
$$
A=\begin{bmatrix}
0&1&-1&1&-1 \\
1&0&1&2&-1 \\
1&1&0&3&-2 \\
2&2&0&6&-3 
\end{bmatrix}\to
\begin{bmatrix}
1&0&1&2&-1 \\
0&1&-1&1&-1 \\
0&0&0&0&1 \\
0&0&0&0&0 
\end{bmatrix}\to
\begin{bmatrix}
1&0&1&2&0 \\
0&1&-1&1&0 \\
0&0&0&0&1 \\
0&0&0&0&0 
\end{bmatrix}
$$
因此 
$$
\begin{cases}
x_1=-x_3-2x_4 \\
x_2=x_3-x_4 \\
x_5=0
\end{cases}
$$
可写为解向量的形式
$$
\begin{bmatrix}x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5\end{bmatrix}=
x_3\begin{bmatrix}-1 \\ 1 \\ 1 \\ 0 \\ 0\end{bmatrix}
+x_4\begin{bmatrix}-2\\-1\\0\\1\\0\end{bmatrix}
$$

## 非齐次线性方程组的解

对于非齐次线性方程组 $A\mathbf{x}=0$ 。判断向量方程 $x_1\mathbf a_1+x_2\mathbf a_2+\cdots+x_n\mathbf a_n=\mathbf b$ 是否有解，等价于判断常数向量 $\mathbf b$ 是否属于 $\text{span}\{\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n\}$。

<kbd>判别定理</kbd>：线性方程组有解的充要条件是其系数矩阵$A$与增广矩阵$\bar A$的秩相等 $\text{rank}(A)=\text{rank}(\bar A)$。

> 通俗理解就是，变换后的阶梯形方程组不存在 $0=b$ 的矛盾方程。

解的结构：设 $n$ 元非齐次线性方程组 $\text{rank}(A)=\text{rank}(\bar A)$

(1) 若 $\text{rank}(A)=n$，方程组有唯一解；
(2) 若 $\text{rank}(A)<n$，方程组有无穷多解。

非齐次线性方程组 $A\mathbf x=\mathbf b$ 对应的齐次线性方程组 $A\mathbf x=0$ 称为**导出方程组**。解的关系：

1. $A\mathbf x=\mathbf b$ 的任意两个解向量之差是 $A\mathbf x=0$ 的一个解向量；
2. $A\mathbf x=\mathbf b$ 的通解是其任一解向量与 $A\mathbf x=\mathbf b$ 通解之和。

如下图

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/Nonhomogeneous_linear_equation.svg)

示例：求下列线性方程组的全部解

$$
\begin{cases}
\begin{alignedat}{4} 
x_1&+4x_2&-5x_3& = 0 \\
2x_1&-x_2&+8x_3& = 9
\end{alignedat}
\end{cases}
$$
解：对方程组的增广矩阵做初等行变换获得阶梯形矩阵和简化阶梯形矩阵
$$
\bar A=\begin{bmatrix}\begin{array}{ccc:c}
1&4&-5&0 \\
2&-1&8&9
\end{array}\end{bmatrix}\to
\begin{bmatrix}\begin{array}{ccc:c}
1&4&-5&0 \\
0&-9&18&9
\end{array}\end{bmatrix}\to
\begin{bmatrix}\begin{array}{ccc:c}
1&0&3&4 \\
0&1&-2&1
\end{array}\end{bmatrix}
$$
因此
$$
\begin{cases}
x_1=4-3x_3 \\
x_2=1+2x_3
\end{cases}
$$
解向量的形式为
$$
\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=
\begin{bmatrix}4\\1\\0\end{bmatrix}
+x_3\begin{bmatrix}-3 \\ 2 \\ 1 \end{bmatrix}
$$

# 行列式

## 二阶行列式

行列式引自对线性方程组的求解。考虑两个方程的二元线性方程组
$$
\begin{cases}
a_{11}x_1+a_{12}x_2=b_1 \\
a_{21}x_1+a_{22}x_2=b_2
\end{cases}
$$
可使用消元法，得
$$
(a_{11}a_{22}-a_{12}a_{21})x_1=b_1a_{22}-a_{12}b_2 \\
(a_{11}a_{22}-a_{12}a_{21})x_2=a_{11}b_2-b_1a_{21}
$$
当 $a_{11}a_{22}-a_{12}a_{21}\neq 0$ 时，得
$$
x_1=\frac{b_1a_{22}-a_{12}b_2}{a_{11}a_{22}-a_{12}a_{21}},\quad 
x_2=\frac{a_{11}b_2-b_1a_{21}}{a_{11}a_{22}-a_{12}a_{21}}
$$
从方程组解来看，分母 $a_{11}a_{22}-a_{12}a_{21}$ 是系数矩阵 $A=\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22}\end{bmatrix}$ 的元素计算得到，称这个值为矩阵 $A$ 的**二阶行列式**(determinant)，记为 $\det A$ 或 $|A|$ ，或记为数表形式
$$
\begin{vmatrix} 
a_{11} & a_{12} \\ 
a_{21} & a_{22} 
\end{vmatrix}=a_{11}a_{22}-a_{12}a_{21}
$$
利用二阶行列式的概念，分子也可写为二阶行列式
$$
\det A_1=\begin{vmatrix} b_1 & a_{12} \\ b_2 & a_{22}\end{vmatrix}=b_1a_{22}-a_{12}b_2 \\
\det A_2=\begin{vmatrix} a_{11} & b_1 \\ a_{21} & b_2\end{vmatrix}=a_{11}b_2-b_1a_{21}
$$
从上面对比可以看出，$x_j$ 的矩阵 $A_j$ 是系数矩阵 $A$的第 $j$ 列用常数项代替后的矩阵。这样，方程组的解可表示为
$$
x_1=\frac{\det A_1}{\det A},\quad
x_2=\frac{\det A_2}{\det A}
$$

## $n$ 阶行列式

考虑三个方程的三元线性方程组，系数矩阵为
$$
A=\begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\a_{31} & a_{32} & a_{33}\end{bmatrix}
$$
用消元法可知未知数的分母同样是系数矩阵$A$ 的元素运算得到，于是定义三阶行列式为
$$
\begin{vmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\a_{31} & a_{32} & a_{33}\end{vmatrix} 
=a_{11}a_{22}a_{33}+a_{12}a_{23}a_{31}+a_{13}a_{21}a_{32} 
-a_{11}a_{23}a_{32}-a_{12}a_{21}a_{33}-a_{13}a_{22}a_{31}
$$
由二阶行列式的定义，上式可变为
$$
\begin{vmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\a_{31} & a_{32} & a_{33}\end{vmatrix}=
a_{11}\begin{vmatrix}  a_{22} & a_{23} \\ a_{32} & a_{33}\end{vmatrix}-
a_{12}\begin{vmatrix}  a_{21} & a_{23} \\ a_{31} & a_{33}\end{vmatrix}+
a_{13}\begin{vmatrix}  a_{11} & a_{12} \\ a_{21} & a_{22}\end{vmatrix}
$$
进一步探索 $n$ 元线性方程组，可知高阶行列式定义。为书写方便，把元素 $a_{ij}$ 所在的行和列划掉后，剩下的元素组成的行列式称为 $a_{ij}$ 的**余子式**(cofactor)，记作 $M_{ij}$ ，并称
$$
A_{ij}=(-1)^{i+j}M_{ij}
$$
为 $a_{ij}$ 的**代数余子式**(algebraic cofactor)。

<kbd>定义</kbd>：方阵 $A$ 的行列式用第一行元素的代数余子式定义为
$$
\det A=\begin{vmatrix}
a_{11}&a_{12}&\cdots&a_{1n} \\
a_{21}&a_{22}&\cdots&a_{2n} \\
\vdots&\vdots&\ddots&\vdots \\
a_{n1}&a_{n2}&\cdots&a_{nn} \\
\end{vmatrix}=\sum_{j=1}^na_{1j}A_{1j}
$$
由定义易知，行列式可以按任意行(列)展开。
$$
\det A=\sum_{j=1}^na_{ij}A_{ij}, \quad \text{by row }i \\
\det A=\sum_{i=1}^na_{ij}A_{ij}, \quad \text{by col }j
$$

## 行列式的性质

<kbd>性质</kbd>：使用数学归纳法可知

1. 行列式与其转置行列式相等：$\det A^T=\det A$

2. 互换行列式两行(列)，行列式改变符号。
   $$
   \begin{vmatrix}a&b\\c&d\end{vmatrix}=-\begin{vmatrix}c&d\\a&b\end{vmatrix}
   $$

3. 行列式的某一行(列)所有元素同乘以数$k$，等于数$k$乘以该行列式。
   $$
   \begin{vmatrix}ka&b\\kc&d\end{vmatrix}=k\begin{vmatrix}a&b\\c&d\end{vmatrix}
   $$

4. 若行列式的某一行(列)的为两组数之和，则可表示为两行列式之和。
   $$
   \begin{vmatrix}a_1+a_2&b\\c_1+c_2&d\end{vmatrix}=\begin{vmatrix}a_1&b\\c_1&d\end{vmatrix}+\begin{vmatrix}a_2&b\\c_2&d\end{vmatrix}
   $$

5. 把行列式的某一行(列)所有元素同乘以数 $k$ 都加到另一行(列)对应的元素上去，行列式的值不变。
   $$
   \begin{vmatrix}a&b\\c&d\end{vmatrix}=\begin{vmatrix}a+kb&b\\c+kd&d\end{vmatrix}
   $$

6. 矩阵乘积的行列式等于行列式的乘积：$\det(AB)=(\det A)(\det B)=\det(BA)$ 

<kbd>推论</kbd>：

1. 行列式中若有两行(列)元素相同，该行列式的值为零。
2. 行列式中某一行(列)的公因子可以提取到行列式符号外面。
3. 行列式中若有两行(列)元素成比例，则此行列式等于零。
4. $\det(kA)=k^n\det A$

由上面的性质，我们很容易得到：

1. 出现零行和零列的行列式为零。
2. 对角阵 $A=\text{diag}(\lambda_1,\lambda_2,\cdots,\lambda_n)$ 的行列式 $\det A=\lambda_1\lambda_2\cdots\lambda_n$ 。
3. 如果 $A$ 是三角阵，行列式为主对角线元素的乘积。

**对于高阶行列式，一般利用行列式的性质，初等变换化为三角行列式求解。**

示例：可用数学归纳法证明**范德蒙行列式**(Vandermonde determinant)：
$$
\begin{vmatrix} 
1 & 1& \cdots &1 \\ 
 a_1 &a_2&\cdots  &a_n \\
 a_1^2 &a_2^2&\cdots  &a_n^2  \\
\vdots &\vdots&\vdots  &\vdots \\
  a_1^{n-1} &a_2^{n-1}&\cdots  &a_n^{n-1} 
\end{vmatrix}=\prod_{1⩽ i<j⩽n}(a_j-a_i)
$$

**行列式函数**：若 $A$ 为$n$阶矩 阵，可以将 $\det A$ 看作 $A$ 中 $n$ 个列向量的函数。若 $A$ 中除了一列之外都是固定的向量，则 $\det A$ 是线性函数。

假设第 $j$ 列是变量，定义映射 $\mathbf x\mapsto T(\mathbf x)$ 为
$$
T(\mathbf x)=\det A=\det\begin{bmatrix}\mathbf a_1\cdots\mathbf x\cdots\mathbf a_n\end{bmatrix}
$$
 则有
$$
T(c\mathbf x)=cT(\mathbf x) \\
T(\mathbf u+\mathbf v)=T(\mathbf u)+T(\mathbf v)
$$

## 克拉默法则

这里只讨论方程个数和未知数相等的$n$元线性方程组
$$
A\mathbf x=\mathbf b
$$
若 $\det A\neq0$，那么它有唯一解
$$
x_j=\frac{\det A_j(\mathbf b)}{\det A},\quad(j=1,2,\cdots,n)
$$

> 约定 $A_j(\mathbf b)$ 表示用向量 $\mathbf b$ 替换矩阵$A$的第$j$列。

证：用$\mathbf a_1,\mathbf a_2,\cdots,\mathbf a_n$ 表示矩阵$A$ 的各列，$\mathbf e_1,\mathbf e_2,\cdots,\mathbf e_n$ 表示单位阵$I_n$ 的各列。由分块矩阵乘法
$$
\begin{aligned}
AI_j(\mathbf x)&=A\begin{bmatrix}\mathbf e_1&\cdots&\mathbf x&\cdots&\mathbf e_n\end{bmatrix} \\
&=\begin{bmatrix}A\mathbf e_1&\cdots& A\mathbf x&\cdots& A\mathbf e_n\end{bmatrix} \\
&=\begin{bmatrix}\mathbf a_1&\cdots&\mathbf b&\cdots&\mathbf a_n\end{bmatrix} \\
&=A_j(\mathbf b)
\end{aligned}
$$
由行列式的乘法性质
$$
\det A\det I_j(\mathbf x)=\det A_j(\mathbf b)
$$
左边第二个行列式可沿第 $j$ 列余子式展开求得 $\det I_j(\mathbf x)=x_j$。从而
$$
x_j\det A=\det A_j(\mathbf b)
$$
若 $\det A\neq0$，则上式得证。


## 行列式的几何理解

> Grant：行列式告诉你一个线性变换对区域的缩放比例。

我们已经知道，线性变换保持网格线平行且等距。为了方便，我们只考虑在平面直角坐标系内，单位基向量 $\mathbf i,\mathbf j$ 所围成的单位正方形区域的线性变换。

根据向量加法的平行四边形法则和线性变换基本性质知，变换后的区域为矩阵 $A=\begin{bmatrix}a & b\\c & d\end{bmatrix}$ 的列向量 $\begin{bmatrix}a\\c\end{bmatrix}$ 和 $\begin{bmatrix}b\\d\end{bmatrix}$ 为邻边的平行四边形区域。

<kbd>结论</kbd>：二阶行列式的值表示由 $A$ 的列确定的有向平行四边形的面积。

(1) 若 $A$ 为对角阵，显然行列式 $\det\begin{bmatrix}a & b\\0 & d\end{bmatrix}$ 表示底为 $a$，高为 $d$ 的平行四边形面积

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/det_shears.svg" style="zoom:80%;" />

(2) 更一般的情况 $A=\begin{bmatrix}a & b\\c & d\end{bmatrix}$ ，可以看出，行列式的值与面积有着紧密的联系。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/determinant.svg" style="zoom:100%;" />

(3) 矩阵 $\begin{bmatrix}a^2 & a\\a & 1\end{bmatrix}$ 表示将单位正方形压缩成线段，面积自然为0，行列式的值为0

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/det_projection.svg" style="zoom:80%;" />

单位正方形区域缩放的比例，其实可以代表任意给定区域缩放的比例。这是因为，线性变换保持网格线平行且等距。对于空间中任意区域的面积，借助微积分的思想，我们可以采用足够的小方格来逼近区域的面积，对所有小方格等比例缩放，则整个区域也以同样的比例缩放。
$$
\text{volume }T(\Omega) = (\det T)(\text{volume }\Omega)
$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/det_area.svg"  />

通过行列式的几何意义，我们就建立了线性变换、矩阵、行列式之间的关系。不难得出

1. 复合线性变换缩放的比例相当于每次变换缩放比例的乘积，即
   $$
   \det AB=\det A\det B
   $$

2. 行列式的值为零，表示将空间压缩到更低的维度，矩阵的列向量线性相关

# 矩阵的运算

## 矩阵的转置

**转置**：矩阵$A$的行列互换得到的矩阵称为 $A$ 的转置(transpose)，记作 $A^T$。

<kbd>性质</kbd>：矩阵转置运算满足下列性质：

1. $(A+B)^T=A^T+B^T$
2. $(A^T)^T=A$
3. $(kA)^T=kA^T$
4. $(AB)^T=B^TA^T$
5. $(A^T)^{-1}=(A^{-1})^T$

## 方阵的运算

**三角矩阵**：(triangular matrix)主对角线的下方元素都是零的方阵，称为**上三角矩阵**。类似的，主对角线的上方元素都是零的方阵，称为**下三角矩阵**。
$$
\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1n} \\
&a_{22}&\cdots&a_{2n} \\
&&\ddots&\vdots \\
&&&a_{nn} \\
\end{bmatrix},\quad
\begin{bmatrix}
a_{11}&&& \\
a_{21}&a_{22}&& \\
\vdots&\vdots&\ddots& \\
a_{n1}&a_{n2}&\cdots&a_{nn} \\
\end{bmatrix}
$$

**上(下)三角阵的行列式为主对角线元素的乘积**
$$
\det A=a_{11}a_{22}\cdots a_{nn}
$$
**对角阵**：不在主对角线上的元素全为零的矩阵称为**对角阵**(diagonal matrix)，记作
$$
\mathrm{diag}(a_1,a_2,\cdots,a_n)=\begin{bmatrix}
a_1 \\
&a_2 \\
&&\ddots \\
&&&a_n \\
\end{bmatrix}
$$
对角阵有良好的性质：

1. 两对角阵的乘积仍为对角阵
   $$
   \begin{bmatrix}a_1 \\&a_2 \\&&\ddots \\&&&a_n \end{bmatrix}
   \begin{bmatrix}b_1 \\&b_2 \\&&\ddots \\&&&b_n \end{bmatrix}=
   \begin{bmatrix}a_1b_1 \\&a_2b_2 \\&&\ddots \\&&&a_nb_n \end{bmatrix}
   $$

2. 对角阵的幂仍为对角阵
   $$
   \begin{bmatrix}a_1 \\&a_2 \\&&\ddots \\&&&a_n \end{bmatrix}^k=
   \begin{bmatrix}a_1^k \\&a_2^k \\&&\ddots \\&&&a_n^k \end{bmatrix}
   $$

**数量阵**：主对角线上的元素都相等的对角阵，称为**数量阵**(scalar matrix)。
$$
\mathrm{diag}(a,a,\cdots,a)=\begin{bmatrix}
a \\
&a \\
&&\ddots \\
&&&a \\
\end{bmatrix}
$$
数量阵得名于它的乘法。如二阶数量阵
$$
\begin{bmatrix}k&0 \\ 0&k \end{bmatrix}A=k\begin{bmatrix}1&0 \\ 0&1 \end{bmatrix}A=kA
$$

**单位阵**：主对角线上的元素全为1的对角阵，称为**单位阵**(identity matrix)。$n$ 阶单位阵记作$E_n$或$I_n$。任何矩阵与单位阵的乘积都等于自身。
$$
I_3=\begin{bmatrix}1&0&0 \\0&1&0 \\0&0&1 \\ \end{bmatrix}
$$
**对称阵**与**反对称阵**：设 $A=(a_{ij})$ 为 $n$阶方阵，若$A^T=A$ ，即$a_{ij}=a_{ji}$，则称为**对称阵**(symmetric matrix)；若$A^T=-A$ ，即 $a_{ij}=-a_{ji}$，则称为**反对称阵**(skew-symmetric matrix)。

==易证明 $AA^T$ 和 $A^TA$ 是对称阵。==

**方阵的幂**：由于矩阵满足结合律，我们可以定义矩阵的幂运算
$$
A^0=I,\quad A^n=\overbrace{AA\cdots A}^n
$$
当矩阵 $A$ 可逆时，定义
$$
A^{-k}=(A^{-1})^k
$$
显然只有方阵的幂才有意义。幂运算满足如下性质：

1. $A^kA^l=A^{k+l}$
2. $(A^k)^l=A^{kl}$

注意：因为矩阵乘法无交换率，因此一般情况下 $(AB)^k\neq A^kB^k$


## 初等矩阵

**初等变换**：矩阵初等行变换的定义同样适用于列，相应的记法为 $c_i\lrarr c_j,kc_i,c_i+kc_j$ 。矩阵的初等行变换和初等列变换统称矩阵的**初等变换**。若矩阵 $A$ 经有限次初等变换变为$B$，则称$A$与$B$ **等价**(equivalent) 。

矩阵的初等变换是矩阵的一种最基本运算，其过程可以通过特殊矩阵的乘法来表示。

<kbd>初等矩阵</kbd>：由单位矩阵进行一次初等变换得到的矩阵称为**初等矩阵**(elementary matrix)。易知初等矩阵都是可逆的。

三种初等变换对应着三种初等矩阵。由矩阵的乘法运算可以验证：**对矩阵的初等行变换相当于左乘相应的初等矩阵；对矩阵的初等列变换相当于右乘相应的初等矩阵**。

1. 互换变换，如 $r_1\lrarr r_2$
   $$
   \begin{bmatrix}0&1&0 \\1&0&0\\0&0&1\end{bmatrix}
   \begin{bmatrix}a_1&b_1 \\a_2&b_2\\a_3&b_3\end{bmatrix}=
   \begin{bmatrix}a_2&b_2\\a_1&b_1 \\a_3&b_3\end{bmatrix}
   $$

2. 倍乘变换，如 $2r_1$
   $$
   \begin{bmatrix}2&0&0 \\0&1&0\\0&0&1\end{bmatrix}
   \begin{bmatrix}a_1&b_1 \\a_2&b_2\\a_3&b_3\end{bmatrix}=
   \begin{bmatrix}2a_1&b_1 \\a_2&b_2\\a_3&b_3\end{bmatrix}
   $$

3. 倍加变换，如 $r_1+2r_2$
   $$
   \begin{bmatrix}1&2&0 \\0&1&0\\0&0&1\end{bmatrix}
   \begin{bmatrix}a_1&b_1 \\a_2&b_2\\a_3&b_3\end{bmatrix}=
   \begin{bmatrix}a_1+2a_2&b_1+2b_2 \\a_2&b_2\\a_3&b_3\end{bmatrix}
   $$

<kbd>定理</kbd>：任意一个可逆矩阵都可以表示为有限个初等矩阵的乘积。

由于初等矩阵可逆，所以初等矩阵的乘积亦可逆。

所有矩阵都可通过初等变换化为**标准型**
$$
\begin{bmatrix}
\left.\begin{matrix}1&& \\ &\ddots&\\&&1\end{matrix}\right\}r &  \\
&\begin{matrix}0 \\ &\ddots&\\&&0\end{matrix}
\end{bmatrix}=
\begin{bmatrix}I_r&O \\O&O\end{bmatrix}
$$


## 分块矩阵

> 分块矩阵是矩阵运算的一种技巧。

在矩阵的运算和理论研究中，有时对矩阵进行分块处理，常常会简化矩阵的运算，或者使原矩阵显得结构简单而清晰。
$$
\begin{bmatrix}
\begin{array}{cc:cc} 
1&0 & 0 & 0 \\ 
0&1 & 0 &0 \\ 
\hdashline 
0&0 & 1 & 5
\end{array}\end{bmatrix}
=\begin{bmatrix}
   I_2 & O \\
   O & A
\end{bmatrix}
$$
像这样，结合矩阵本身的特点，把一个矩阵用横线和竖线划分为若干个子块，并以所分的子块为元素的矩阵称为**分块矩阵**(Block matrix)。一个矩阵可用不同的方法分块。

分块矩阵的运算形式上和普通矩阵相同，把子块当成元素计算即可。

**加法**：设分块 $A,B$ 是同型矩阵，且对它们的分法相同，则 $A+B=(A_{ij}+B_{ij})$
$$
\begin{bmatrix}A_1 & B_1 \\C_1 & D_1 \end{bmatrix}+
\begin{bmatrix}A_2 & B_2 \\C_2 & D_2 \end{bmatrix}=\begin{bmatrix}A_1+A_2 & B_1+B_2 \\C_1+C_2 & D_1+D_2 \end{bmatrix}
$$
**数乘**：分块矩阵 $A$ ，数乘作用于每个子块。
$$
k\begin{bmatrix}A & B \\C & D \end{bmatrix}=\begin{bmatrix}kA & kB \\kC & kD \end{bmatrix}
$$
**乘法**：分块矩阵的乘法按矩阵乘法的形式计算。
$$
AB=A\begin{bmatrix}\mathbf b_1&\mathbf b_2&\cdots&\mathbf b_p\end{bmatrix}=
\begin{bmatrix}A\mathbf b_1&A\mathbf b_2&\cdots&A\mathbf b_p\end{bmatrix}
$$
矩阵乘法的列行展开
$$
AB=\begin{bmatrix}\mathbf a_1&\mathbf a_2&\cdots&\mathbf a_n\end{bmatrix}
\begin{bmatrix}\mathbf b_1\\\mathbf b_2\\\vdots\\\mathbf b_n\end{bmatrix}
=\mathbf a_1\mathbf b_1+\mathbf a_2\mathbf b_2+\cdots+\mathbf a_n\mathbf b_n
$$
**转置**：分块矩阵 $A=(A_{ij})$ 的转置等于各子块的转置 $A^T=(A_{ij}^T)$

**分块上三角矩阵**：
$$
\begin{bmatrix}A&B\\O&D\end{bmatrix}^{-1}=
\begin{bmatrix}A^{-1}&-A^{-1}BD^{-1}\\O&D^{-1}\end{bmatrix}
$$
设分块矩阵 $\begin{bmatrix}X_1&X_2\\X_3&X_4\end{bmatrix}$ 是矩阵 $\begin{bmatrix}A&B\\O&D\end{bmatrix}$ 的逆，则
$$
\begin{bmatrix}A&B\\O&D\end{bmatrix}
\begin{bmatrix}X_1&X_2\\X_3&X_4\end{bmatrix}
=\begin{bmatrix}I_p&O\\O&I_q\end{bmatrix}
$$
这个矩阵方程包含了4个未知子块的方程
$$
AX_1+BX_3=I_p \\
AX_2+BX_4=O \\
DX_3=O \\
DX_4=I_q
$$
若 $D$ 可逆，从后两个方程可以得到 $X_3=O,X_4=D^{-1}$ ；若 $A$ 可逆，进一步可以得到$X_1=A^{-1},X_2=-A^{-1}BD^{-1}$ 。便可获得分块上三角矩阵的逆。

**分块对角矩阵**：分块对角矩阵拥有良好的性质。

(1) 分块对角矩阵乘积
$$
\begin{bmatrix}A_1 \\&A_2 \\&&\ddots \\&&&A_s \end{bmatrix}
\begin{bmatrix}B_1 \\&B_2 \\&&\ddots \\&&&B_s \end{bmatrix}
=\begin{bmatrix}A_1B_1 \\&A_2B_2 \\&&\ddots \\&&&A_sB_s \end{bmatrix}
$$
(2) 若分块对角矩阵的各个子块可逆，则该对角分块矩阵可逆
$$
\begin{bmatrix}A_1 \\&A_2 \\&&\ddots \\&&&A_s \end{bmatrix}^{-1}=
\begin{bmatrix}A_1^{-1} \\&A_2^{-1} \\&&\ddots \\&&&A_s^{-1} \end{bmatrix}
$$
(3) 分块对角矩阵的行列式为对角位置的行列式乘积
$$
\det\begin{bmatrix}A_1 \\&A_2 \\&&\ddots \\&&&A_s \end{bmatrix}
=\det A_1\det A_2\cdots\det A_s
$$


## 逆矩阵

利用克拉默法可以容易地导出一个求矩阵的逆的一般公式。设矩阵 $A=(a_{ij})_{n\times n}$ 的逆矩阵 $A^{-1}=(b_{ij})_{n\times n}$ ，利用分块矩阵的乘法
$$
AA^{-1}=A\begin{bmatrix}\mathbf b_1&\mathbf b_2&\cdots&\mathbf b_n\end{bmatrix}
=I_n=\begin{bmatrix}\mathbf e_1&\mathbf e_2&\cdots&\mathbf e_n\end{bmatrix}
$$
其中 $\mathbf b_j$ 是矩阵 $A^{-1}$ 的第 $j$ 列， $\mathbf e_j$ 是单位阵 $I_n$ 的第 $j$ 列。于是
$$
A\mathbf b_j=\mathbf e_j
$$
向量 $\mathbf b_j$ 的第 $i$ 个元素是 $A^{-1}$ 的元素 $b_{ij}$ 。由克拉默法则求得
$$
b_{ij}=\frac{\det A_i(\mathbf e_j)}{\det A}
$$
回顾代数余子式的定义，它是把矩阵 $A$ 中元素 $a_{ij}$ 所在的行和列划掉后得到的。$\det A_i(\mathbf e_j)$ 按第 $i$ 列的余子展开式为
$$
\det A_i(\mathbf e_j)=(-1)^{i+j}M_{ji}=A_{ji}
$$
于是可写出矩阵 $A$ 的逆
$$
A^{-1}=\dfrac{1}{\det A}\text{adj }A
$$
其中 $\text{adj }A$ 是矩阵 $A$ 的各个元素的代数余子式$A_{ji}$ 所构成的矩阵
$$
\text{adj }A=\begin{bmatrix}
A_{11}&A_{21}&\cdots&A_{n1} \\
A_{12}&A_{22}&\cdots&A_{n2} \\
\vdots&\vdots&\ddots&\vdots \\
A_{1n}&A_{2n}&\cdots&A_{nn} \\
\end{bmatrix}
$$
做矩阵$A$的**伴随矩阵**(Adjugate Matrix) 。

> 注意，伴随矩阵里代数余子式的排列顺序是颠倒的。

<kbd>定理</kbd>：方阵 $A$ 可逆的充要条件是 $\det A\neq0$ ，且 $A^{-1}=\dfrac{1}{\det A}\text{adj }A$

此定理仅适用于理论上的计算矩阵的逆，使我们不用实际计算出$A^{-1}$ 就可以推导出性质。

这里给出二阶方阵 $A=\begin{bmatrix}a&b\\c&d\end{bmatrix}$ 的逆，若 $\det A=ad-bc\neq0$ 则
$$
A^{-1}=\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}
$$


<kbd>推论</kbd>：

1. 若 $n$ 阶方阵  $A,B$ 满足 $AB=I$ 或 $BA=I$ ，则 $B=A^{-1}$ 。
2. $A(\text{adj }A)=(\text{adj }A)A=(\det A)I$

有了推论1，只需判断 $AB=I$ 或 $BA=I$ 中的一个条件就可判定逆矩阵，要比定义简单一些。

**利用初等变换计算逆矩阵**：写出增广矩阵 $(A\mid I)$， 用初等行变换把左边矩阵 $A$ 处化为单位矩阵 $I$ ，则右边出来的就是逆矩阵$A^{-1}$，示意如下：
$$
(A\mid I)\xrightarrow{}(I\mid A^{-1})
$$

同样，利用初等列变换计算逆矩阵的示意如下
$$
\begin{bmatrix}A\\I\end{bmatrix}\xrightarrow{}\begin{bmatrix}I\\A^{-1}\end{bmatrix}
$$

示例：解矩阵方程
$$
\begin{bmatrix}1&0&1\\-1&1&1\\2&-1&1\end{bmatrix}
\begin{bmatrix}x_1&y_1\\x_2&y_2\\x_3&y_3\\\end{bmatrix}=
\begin{bmatrix}1&1\\0&1\\-1&0\\\end{bmatrix}
$$
解：系数矩阵可逆的矩阵方程 $AX=B$ ，解为$X=A^{-1}B$ 。实际中，不必求逆矩阵，可使用一系列初等变换求解，即系数矩阵和常数项做同样的变换 $P=A^{-1}$。图示如下
$$
(A\mid B)\xrightarrow{}(I\mid X)
$$
本例计算过程如下
$$
\begin{bmatrix}
\begin{array}{ccc:cc}
1&0&1&1&1\\
-1&1&1&0&1\\
2&-1&1&-1&0
\end{array}
\end{bmatrix}\to
\begin{bmatrix}
\begin{array}{ccc:cc}
1&0&0&3&1\\
0&1&0&5&2\\
0&0&1&-2&0
\end{array}
\end{bmatrix}
$$
故
$$
\begin{bmatrix}x_1&y_1\\x_2&y_2\\x_3&y_3\end{bmatrix}=
\begin{bmatrix}3&1\\5&2\\-2&0\end{bmatrix}
$$

## 矩阵的秩

**行空间**：矩阵$A=(\mathbf r_1,\mathbf r_2,\cdots,\mathbf r_m)^T$ 的所有行向量张成的空间称为 $A$ 的行空间，记为
$$
\text{row }A=\text{span}\{\mathbf r_1,\mathbf r_2,\cdots,\mathbf r_m\}
$$
若两个矩阵 $A$ 和 $B$ 行等价，则它们的的行空间相同。若 $B$ 是阶梯型矩阵，则 $B$ 的非零行构成 $\text{row }B$ 的一组基，同时也是$\text{row }A$ 的一组基。

证明：若 $B$ 是由 $A$ 经行变换得到的，则 $B$ 的行是$A$ 的行的线性组合，于是 $B$ 的行的任意线性组合自然是 $A$ 的行的线性组合，从而 $B$ 的行空间包含于 $A$ 的行空间。因为行变换可逆，同理知 $A$ 的行空间是 $B$ 的行空间的子集，从而这两个空间相同。若 $B$ 是一个阶梯形矩阵，则其非零行是线性无关的，这是因为任何一个非零行均不为它下面的非零行的线性组合，于是 $B$ 的非零行构成 $B$ 的行空间的一组基，当然也是 $A$ 的行空间的一组基。

例：分别求矩阵 $A$ 的行空间、列空间和零空间的基
$$
A=\begin{bmatrix}-2&-5&8&0&-17\\1&3&-5&1&5\\3&11&-19&7&1\\1&7&-13&5&-3\end{bmatrix}
$$
解：为了求行空间和列空间的基，行化简$A$成阶梯形
$$
A\to \begin{bmatrix}1&3&-5&1&5\\0&1&-2&2&-7\\0&0&0&-4&20\\0&0&0&0&0\end{bmatrix}=B
$$
 矩阵 $B$ 的前 3 行构成$B$的行空间的一个基，也是$A$的行空间的一组基。

$\text{row }A$ 的基：$(1,3,-5,1,5),(0,1,-2,2,-7),(0,0,0,-4,20)$

对列空间，$B$ 的主元列在第1，2和4列，从而 $A$ 的第1，2和4列构成 $\text{col }A$ 的一组基。

 $\text{col }A$ 的基：$(-2,1,3,1)^T,(-5,3,11,7)^T,(0,1,7,5)^T$

对于核空间，需要进一步行变换得简化阶梯型矩阵
$$
B\to\begin{bmatrix}1&0&1&0&1\\0&1&-2&0&3\\0&0&0&1&-5\\0&0&0&0&0\end{bmatrix}=C
$$
方程 $A\mathbf x=0$ 的解空间等价于 $C\mathbf x=0$  的解空间，即
$$
\begin{cases}
x_1+x_3+x_5=0 \\
x_2-2x_3+3x_5=0  \\
x_4-5x_5=0
\end{cases}
$$
所以
$$
\begin{bmatrix}x_1\\x_2\\x_3\\x_4\\x_5\end{bmatrix}=
x_3\begin{bmatrix}-1\\2\\1\\0\\0\end{bmatrix}+
x_5\begin{bmatrix}-1\\-3\\0\\5\\1\end{bmatrix}
$$
$\text{null } A$ 的基：$(-1,2,1,0,0)^T,(-1,-3,0,5,1)^T$ 

通过观察可见，与 $\text{col }A$ 的基不同，$\text{row }A$ 和 $\text{null } A$ 的基与$A$ 中的元素没有直接的关系。

<kbd>定理</kbd>：对于 $m\times n$ 维矩阵 $A$ 

1. $\dim(\text{row }A)=\dim(\text{col }A)=\text{rank }A$
2. $\text{rank }A+\dim(\text{null } A)=n$

证明：$\text{rank }A$ 是$A$中主元列的个数，也是$A$的等价阶梯形矩阵$B$中主元列的个数。进一步，因为 $B$ 的每个主元都对应一个非零行，同时这些非零行构成 $A$ 的行空间的一组基，所以 $A$ 的秩等于 $\text{row }A$ 的维数。由于 $\text{null } A$ 的维数等于方程 $A\mathbf x=0$ 中自由变量的个数，换句话说， $\text{null } A$ 的维数是 $A$ 中非主元列的个数。上面的定理证闭。

<kbd>性质</kbd>：

1. 矩阵的秩在初等变换下保持不变
2. 矩阵的列向量组的秩等于行向量组的秩
3. $\text{rank}(A+B)\leqslant \text{rank}(A)+\text{rank}(B)$
4. $\text{rank}(kA)=\text{rank}(A)$
5. $\text{rank}(AB)\leqslant \min\{\text{rank}(A),\text{rank}(B)\}$

## 广义逆矩阵

对于非其次线性方程组 $A\mathbf x=\mathbf b$ ，当 $A$ 可逆时，则方程组存在唯一解 $\mathbf x=A^{-1}\mathbf b$，通常矩阵 $A$ 是任意的 $m\times n$ 矩阵，不可逆的，这就促使人们去推广逆矩阵的概念，引进某种具有普通逆矩阵类似性质的矩阵 $G$，使得方程组的解仍可表示为 $\mathbf x=G\mathbf b$ 这种简单的形式。

- 若 $AGA=A$，则 $A\mathbf x=AGA\mathbf x=A(G\mathbf b)=\mathbf b$，于是$G\mathbf b$ 是方程的解；
- 若 $GAG=G$，由于 $GA\mathbf x=G\mathbf b$，所以 $GA\mathbf x=GAGA\mathbf x=GA(G\mathbf b)=G\mathbf b$，于是$G\mathbf b$ 是方程的解；

对于$m\times n$ 维矩阵 $A$，若存在 $n\times m$ 维矩阵 $G$ 满足以下 M-P 方程
(1) $AGA=A$
(2) $GAG=G$
(3) $(AG)^T=AG$
(4) $(GA)^T=GA$

的全部或一部分，则称 $G$ 为 $A$ 的一个**广义逆矩阵**。若 $G$ 满足全部 M-P 方程，则称 $G$ 为 $A$ 的 Moore-Penrose 广义逆矩阵，简称M-P 广义逆矩阵，也称为伪逆矩阵，记为 $A^+$。事实上，只有伪逆矩阵存在且唯一，其他各类广义逆矩阵都不唯一。


<kbd>性质</kbd>：
1. $(A^+)^+=A$
2. $(A^T)^+=(A^+)^T$
3. $\text{rank }A^+=\text{rank }A$

若非其次线性方程组 $A\mathbf x=\mathbf b$ 有解，则解为
$$
\mathbf x=A^+\mathbf b+(I-A^+A)\mathbf c
$$
其中 $\mathbf c$ 是维数与 $\mathbf x$ 的维数相同的任意向量。显然，当 $A$ 可逆时，$\mathbf x=A^{-1}\mathbf b+(I-A^{-1}A)\mathbf c=A^{-1}\mathbf b$ 。


求伪逆矩阵的一个方法是利用奇异值分解 $A=U\Sigma V^T$ 。由于 $\Lambda_r$ 的对角线元素非零，所以 $\Lambda_r$ 可逆，可求得伪逆为
$$
A^+=V_r\Lambda_r^{-1} U^T_r
$$
