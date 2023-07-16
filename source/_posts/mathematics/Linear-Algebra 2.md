---
title: 线性代数引论
categories:
  - Mathematics
  - Advanced Algebra
tags:
  - 数学
  - 行列式
  - 矩阵
  - 特征值
  - 特征向量
  - 二次型
cover: /img/Linear-Algebra.png
top_img: /img/matrix-logo.jpg
katex: true
description: 矩阵，行列式，向量空间，二次型，线性变换
abbrlink: '40113498'
date: 2023-07-13 23:28:34
---

# 线性变换与矩阵

## 向量及其运算

线性代数中，每个向量都以坐标原点为起点，那么任何一个向量就由其终点唯一确定。从而，向量和空间中的点一一对应。因为选定坐标系后，空间内的点与有序实数对一一对应，从而空间内的向量与有序实数对也一一对应。因此，单个向量相当于空间中的点，多个向量则组成空间中的几何图形，后续将对向量和几何图形不做区分。

向量的运算法则如下图

<img src='https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/plane_vector.svg'/>

向量的坐标取值依托于坐标系的基向量。选取的基向量不同，其所对应的坐标值就不同。接下来我们将从二维平面出发学习线性代数。通常选用平面直角坐标系 $Oxy$ ，基向量的坐标值为
$$
\mathbf i=\begin{bmatrix} 1 \\ 0 \end{bmatrix},\quad
\mathbf j=\begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$
平面内的任意向量都可以写成基向量的线性组合
$$
\mathbf v=x\mathbf i+y\mathbf j
$$
通常记为列向量形式
$$
\mathbf v=\begin{bmatrix} x \\ y \end{bmatrix}
$$
**向量加法**的坐标运算为
$$
\mathbf v+\mathbf w=
\begin{bmatrix} v_1 \\ v_2 \end{bmatrix}+\begin{bmatrix} w_1 \\ w_2 \end{bmatrix}=\begin{bmatrix} v_1+w_1 \\ v_2+w_2 \end{bmatrix}
$$
**向量数乘**的坐标运算为
$$
\lambda\mathbf v=
\lambda\begin{bmatrix} v_1 \\ v_2 \end{bmatrix}=\begin{bmatrix} \lambda v_1 \\ \lambda v_2 \end{bmatrix}
$$

在数学中，我们将向量的数乘称为**缩放**（Scaling）。

## 线性变换

在平面直角坐标系 $Oxy$ 中，平面内的点和有序实数对 $(x,y)$ 一一对应。这样，借助直角坐标系，我们可以从代数的角度来研究几何变换。

> 变换与函数类似，函数把实数映射到实数，变换把点映射到点。

<img src='https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/linear_transformation_example.svg'/>

平面内任意一点 $P(x,y)$ 绕原点$O$按逆时针方向旋转 $60\degree$ 角得到点 $P'(x',y')$，坐标变换公式为
$$
\begin{cases}
x'=\frac{1}{2}x-\frac{\sqrt 3}{2}y \\
y'=\frac{\sqrt 3}{2}x+\frac{1}{2}y
\end{cases}
$$
平面内任意一点 $P(x,y)$ 关于 $y$ 轴的对称点 $P'(x',y')$的表达式为
$$
\begin{cases}
x'=-x \\
y'=y
\end{cases}
$$
事实上，在平面直角坐标系 $Oxy$中，很多几何变换都具有如下坐标变换公式
$$
\begin{cases}
x'=ax+by \\
y'=cx+dy
\end{cases}
$$
其中 $(x',y')$为平面内任意一点 $(x,y)$​ 变换后的点。我们把形如上式的几何变换叫做**线性变换**（linear transformation）。由于上述坐标变换公式中的系数唯一确定，我们可以按顺序简化为如下数表的形式
$$
\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
$$
这个数表完全刻画了上述线性变换，我们把这个数表称为二阶矩阵。然后，把变换后的向量定义为矩阵与向量的乘积
$$
\begin{bmatrix} x' \\ y' \end{bmatrix}=
\begin{bmatrix} a & b\\ c & d\end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}=
\begin{bmatrix} ax+by \\ cx+dy\end{bmatrix}
$$
至此，任何一个线性变换都可以写为矩阵与向量乘积的形式。反之，确定了坐标系后，任何一个矩阵 $A$ 都唯一确定了一个线性变换，这个变换把每一个向量$\mathbf v$变成了新向量 $A\mathbf v$。矩阵和向量的乘积与线性变换实现了一一对应。

<kbd>基本性质</kbd>：可用定义证明线性变换满足加法和数乘运算

(1) 加法： $A(\mathbf v+\mathbf w)=A\mathbf v+A\mathbf w$
(2) 数乘： $A(\lambda\mathbf v)=\lambda A\mathbf v$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/linear_transformation_additivity.svg" style="zoom:80%;" />

<kbd>性质2</kbd>>：**一般地，直线在线性变换后仍然保持直线**。<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/vector_equation_of_line.svg" align="right"/>

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
(2) 如果 $A\mathbf w_1 = A\mathbf w_2$，那么 $\lambda_1 A\mathbf w_1+\lambda_2 A\mathbf w_2=A\mathbf w_1$ 。由于向量 $A\mathbf w_1$ 的终点是一个确定的点，因而，矩阵 $A$ 所对应的线性变换把直线 $l$ 变成映射成了一个点 $A\mathbf w_1$ 。

## 基向量定位

本节参考 3Blue1Brown 教程，了解线性变换的本质。我们可以使用无限网格刻画二维空间所有点的变换。==线性变换是操作空间的一种手段，它能够保持网格线平行且等距，并保持原点不动==。

**严格定义**：满足下列两条性质的变换 $L$ 为线性变换

(1) 可加性：$L(\mathbf v+\mathbf w)=L(\mathbf v)+L(\mathbf w)$
(2) 伸缩性：$L(\lambda\mathbf v)=\lambda L(\mathbf v)$

对平面直角坐标系内的任意向量 $\mathbf v=x\mathbf i+y\mathbf j$ ，在线性变换 $L$ 的作用下
$$
L(\mathbf v)=L(x\mathbf i+y\mathbf j)=xL(\mathbf i)+yL(\mathbf j)
$$
可知，变换后的向量 $\mathbf v'=L(\mathbf v)$ 由变换后的基向量 $\mathbf i'=L(\mathbf i),\ \mathbf j'=L(\mathbf j)$ 以同样的系数完全确定
$$
\mathbf v'=x\mathbf i'+y\mathbf j'
$$
如果变换后的基向量坐标值分别为 $\mathbf i'=\begin{bmatrix} a \\ c \end{bmatrix}$ 和 $\mathbf j'=\begin{bmatrix} b \\ d \end{bmatrix}$ 。我们可以按顺序写为数表的形式

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/define_matrix.svg)

这样就定义了矩阵。变换后的向量则定义为矩阵与向量的乘积
$$
\begin{bmatrix}a & b\\c & d\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=
x\begin{bmatrix} a \\ c \end{bmatrix}+
y\begin{bmatrix} b \\ d \end{bmatrix}=
\begin{bmatrix} ax+by \\ cx+dy \end{bmatrix}
$$
可知，矩阵代表一个特定的线性变换，==我们完全可以把矩阵的列看作变换后的基向量，矩阵向量乘法就是将线性变换作用于给定向量==。

> 注意：线性变换中的坐标值始终使用最初的坐标系。

## 几种特殊的线性变换

我们已经知道，在线性变换的作用下，直线仍然保持直线（或一个点）。为了方便，我们只考虑在平面直角坐标系内，单位正方形区域的线性变换。

根据向量加法的平行四边形法则，单位正方形区域可用向量形式表示为
$$
x_1\mathbf i+x_2\mathbf j \quad(0\leqslant x_1,x_2\leqslant 1)
$$
由基本性质知，变换后的区域为
$$
A(x_1\mathbf i+x_2\mathbf j)=x_1(A\mathbf i)+x_2(A\mathbf j) \quad(0\leqslant x_1,x_2\leqslant 1)
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

**旋转变换**：（rotations）平面内任意一点 $P(x,y)$ 绕原点$O$按逆时针方向旋转 $\theta$ 角，记为 $R_{\theta}$ 。对应的矩阵为
$$
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/rotations.svg" />

**切变变换**：（shears）平行于 $x$ 轴的切变变换对应的矩阵为
$$
\begin{bmatrix}
1 & k\\
0 & 1
\end{bmatrix}
$$
类似的，平行于 $y$​ 轴的切变变换对应的矩阵为
$$
\begin{bmatrix}
1 & 0\\
k & 1
\end{bmatrix}
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/shears.svg" />

**反射变换**：（reflection）一般的我们把平面内任意一点 $P(x,y)$ 关于直线 $l$ 的对称点 $P'(x',y')$的线性变换叫做关于直线 $l$ 的反射变换。

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

**伸缩变换**：（stretching）将每个点的横坐标变为原来的 $k_1$ 倍，纵坐标变为原来的 $k_2$ 倍，其中 $k_1,k_2\neq0$ 。对应的矩阵为
$$
\begin{bmatrix}
k_1 & 0\\
0 & k_2
\end{bmatrix}
$$

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/stretching.svg"  />

**投影变换**：（projection）平面内任意一点 $P(x,y)$ 在直线 $l$ 的投影 $P'(x',y')$，称为关于直线 $l$ 的投影变换。

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

## 复合变换与矩阵乘法

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/composite_transformation.svg" align="right" />平面内任意一向量，依次做旋转变换

 $R_{\theta_1}:\begin{bmatrix}
\cos{\theta_1} & -\sin{\theta_1}\\
\sin{\theta_1} & \cos{\theta_1}
\end{bmatrix}$ 和 $R_{\theta_2}:\begin{bmatrix}
\cos{\theta_2} & -\sin{\theta_2}\\
\sin{\theta_2} & \cos{\theta_2}
\end{bmatrix}$

很显然作用的效果可以用一个变换 $R_{\theta_1+\theta_2}$ 来表示，对应的矩阵为
$$
\begin{bmatrix}
\cos{\theta_1+\theta_2} & -\sin{\theta_1+\theta_2}\\
\sin{\theta_1+\theta_2} & \cos{\theta_1+\theta_2}
\end{bmatrix}
$$
旋转变换 $R_{\theta_1+\theta_2}$仍然是线性变换。

一般地，设矩阵 $A=\begin{bmatrix}a_1 & b_1\\ c_1 & d_1\end{bmatrix},B=\begin{bmatrix}a_2 & b_2\\ c_2 & d_2\end{bmatrix}$，他们对应的线性变换分别为 $f$ 和 $g$ 。

平面上任意一个向量 $\mathbf v=\begin{bmatrix} x \\ y \end{bmatrix}$ 依次做变换 $g$ 和 $f$ ，其作用效果为
$$
\begin{bmatrix}x'\\ y'\end{bmatrix}=f(g(\mathbf v))=A(B\mathbf v)\\
=\begin{bmatrix}a_1a_2+b_1c_2 & a_1b_2+b_1d_2\\ c_1a_2+d_1c_2 & c_1b_2+d_1d_2\end{bmatrix}
\begin{bmatrix}x\\ y\end{bmatrix}
$$
这也是一个线性变换，我们称为**复合变换**（composite transformation），记为 $f\circ g$ 。从而，对任意向量 $\mathbf v$ 有
$$
(f\circ g)\mathbf v=f(g(\mathbf v))
$$
在此，我们定义复合变换 $f\circ g$ 为矩阵$A,B$ 的乘积，记为$AB$。
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

以下参考 3Blue1Brown 教程，进一步了解矩阵乘积的本质。求解矩阵乘积主要在于追踪基向量变换后的位置。基向量 $\mathbf i,\mathbf j$ 进过矩阵 $B$ 变换后的位置为
$$
B\mathbf i=\begin{bmatrix}a_2\\c_2\end{bmatrix},\quad B\mathbf j=\begin{bmatrix}b_2\\d_2\end{bmatrix}
$$
基向量 $B\mathbf i,B\mathbf j$ 进过矩阵 $A$ 变换后的最终位置为
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
上述复合变换可定义为矩阵 $A$ 和 $B$ 的乘积
$$
AB=\begin{bmatrix}a_1 & b_1\\ c_1 & d_1\end{bmatrix}
\begin{bmatrix}a_2 & b_2\\ c_2 & d_2\end{bmatrix}=
\begin{bmatrix}a_1a_2+b_1c_2 & a_1b_2+b_1d_2\\ c_1a_2+d_1c_2 & c_1b_2+d_1d_2\end{bmatrix}
$$
**矩阵乘法满足结合率**：
$$
A(BC)=(AB)C
$$
由矩阵乘法的定义可从数值角度证明上述性质。从线性变换角度来看，对于复合变换 $f\circ (g\circ h)$ 和 $(f\circ g)\circ h$ 是同样的变换相继作用，作用的顺序都是 $h\to g\to f$，变换的复合作用自然不变。

**一般地，矩阵乘法不满足交换率和消去率**

(1) 由于复合变换 $f\circ g\neq g\circ f$ ，自然 $AB\neq BA$，矩阵乘法不满足交换率
(2) 可举例证明 $AB=AC$，但 $B\neq C$ ， 矩阵乘法不满足消去率

任意矩阵与单位阵 $E=\begin{bmatrix}1 & 0\\ 0 & 1\end{bmatrix}$ 的乘积等于自身
$$
AE=EA=A
$$

## 高阶矩阵

**高维空间中的变换与二维空间中的变换类似。而三维变换在计算机图像处理、机器人学中有着重要的作用。**

一般的，设$\sigma,\rho$ 是同一坐标系下的两个线性变换，如果对于任意点 $P$，均有 $\sigma(P)=\rho(P)$ ，则称这两个线性变换相等，记作 $\sigma=\rho$。同时他们对应的系数也分别相等。

对于两个矩阵 $A,B$ ，如果他们的对应元素相等，则称矩阵 $A$ 与 $B$ 相等，记作$A=B$。

**矩阵的幂**：由于矩阵满足结合率，我们可以规定方阵的幂运算
$$
A^0=E,\quad A^n=AA^{n-1} 
$$
幂运算满足如下性质
$$
A^kA^l=A^{k+l} \\
(A^k)^l=A^{kl}
$$

称为矩阵元素

## 非方阵

 $m\times n$ 维的矩阵，表示将** n **维空间中的基向量映射到** m **维空间中，其中** n **列表示变换前基向量空间的维数；**m **行表示变换后基向量需要** m **个独立的坐标来描述**

# 逆矩阵

我们知道了矩阵与线性变换中的对应关系，试想一下，将变换后的向量还原到初始状态。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/inverse_transform.svg"  />

**逆变换**：设  $f$ 是一个线性变换，如果存在线性变换 $g$ ，使得 $f\circ g=g\circ f=I$ ，则称变换$f$ **可逆**（invertible），$g$ 是 $f$ 的逆变换。

**逆矩阵**：对于 $n$ 阶方阵  $A$ ，如果存在 $n$ 阶方阵 $B$ ，使得 $AB=BA=E$，则称矩阵 $A$ **可逆**（invertible），$B$ 是 $A$ 的逆矩阵。

<kbd>性质 1</kbd>：方阵 $A$ 的逆矩阵是唯一的，记为 $A^{-1}$ 。

证明：设 $B_1,B_2$ 都是 $A$ 的逆矩阵，则
$$
B_1=(B_2A)B_1=B_2(AB_1)=B_2
$$

> 相应地，线性变换 $\rho$ 的逆变换也是唯一的，记为 $\rho^{-1}$ 

$$
A^{-1}A=AA^{-1}=E
$$

<kbd>性质 2</kbd>：若方阵 $A,B$ 都可逆，则 $AB$ 也可逆，且
$$
(AB)^{-1}=B^{-1}A^{-1}
$$
证明：
$$
(AB)(B^{-1}A^{-1})=(B^{-1}A^{-1})(AB)=E
$$
从变换的角度思考，复合变换的逆 $(f\circ g)^{-1}=g^{-1}\circ f^{-1}$ ，很容易理解。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/inverse_composite_transformation.svg"  />

<kbd>定理 1</kbd>：二阶矩阵 $A=\begin{bmatrix}a & b\\c & d\end{bmatrix}$ 可逆 $\iff \det A\neq 0$ ，其逆矩阵
$$
A^{-1}=\frac{1}{\det A}\begin{bmatrix}d & -b\\-c & a\end{bmatrix}
$$


# 行列式

**行列式** (determinant)：引自对线性方程组的求解

给定区域面积缩放的比例

立方体体积缩放的比例

给定区域可以分解为若干小方格，对所有小方格等比例缩放，所以整个区域也同样比例的缩放。

行列式就代表这个特殊的缩放比例

单位正方形区域缩放的比例可以代表任意给定区域缩放的比例

三维空间中行列式的值代表着体积的缩放比例



为了方便，我们只考虑在平面直角坐标系内，单位正方形区域的线性变换。

二阶行列式代表平行四边形的面积

三阶行列式代表平行六面体的体积

三维空间中行列式的值代表着体积的缩放比例，我们关注的是单位立方体进行线性变换后的体积变化，**对应行列式的值表示对应平行六面体的体积**。



二阶行列式记作
$$
\begin{vmatrix}a & b\\c & d\end{vmatrix}
$$
也称为矩阵 $A=\begin{bmatrix}a & b\\c & d\end{bmatrix}$ 的行列式，记为 $\det A$ 或 $|A|$ 。

$ad-bc$ 是二阶行列式的展开式，它是位于两条对角线上元素的乘积之差。

1. 一个矩阵的行列式的绝对值为$k$说明将原来一个区域的面积变为$k$倍，变成0了说明降维了，平面压缩成了线，或者点。

   行列式为0说明降维了。矩阵的列线性相关

1. 行列式可以为负数，说明翻转了。这是二维空间的定向，三维空间的定向是“右手定则”

改变了空间的定向
$$
\begin{vmatrix}a&c\\b&d\end{vmatrix}=ad-bc
$$

(2)我们就建立了**线性变换、矩阵、行列式之间**的关系。



(4) **矩阵的列向量线性相关⇔行列式的值0**

我们只要追踪基向量构成的单位面积的变化，因为**其他区域面积变化的比例大小与单位面积变化的比例保持一致**，这样就可以知道空间中任意区域面积变化的比例，这是因为**线性变换保持“网格线平行且等距变换”**。

对于空间中任意区域的面积，借助**微积分**的思想，我们可以采用足够小方格来逼近区域的面积。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/det_area.svg"  />

## 行列式计算的直观理解

表示底为 a,高为 d 的平行四边形的面积

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/det_shears.svg"  />

(3)行列式值为0 表示将空间**压缩到更低的维度**

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/det_projection.svg"  />

可以看出，**行列式的值与面积有着紧密的联系**。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/math/determinant.svg"  />



性质
$$
\det AB=\det A\det B
$$
复合变换面积缩放的比例相当于每次变换面积缩放比例的乘积



# 线性方程组

## 高斯消元法

Linear system of equations

我们可以通过高斯消元法来计算

## 矩阵方程

关于 $x,y$ 的二元一次方程组
$$
\begin{cases}
ax+by=e \\
cx+dy=f
\end{cases}
$$
根据矩阵与向量的乘法定义，可写为矩阵形式
$$
\begin{bmatrix}a & b\\c & d\end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix}=\begin{bmatrix} e \\ f \end{bmatrix}
$$
可简写为
$$
A\mathbf x=\mathbf b
$$
其中，矩阵 $A=\begin{bmatrix}a & b\\c & d\end{bmatrix}$ 称为系数矩阵，$\mathbf b=\begin{bmatrix} e \\ f \end{bmatrix}$ 为常数向量。

以线性变换的角度来看，希望找出未知向量 $\mathbf x$ ，使得该向量在线性变换 $A$ 的作用下变成已知向量 $\mathbf b$。因此，我们可以从逆变换的角度获得未知向量。

<kbd>定理 1</kbd>：关于矩阵方程 $A\mathbf x=\mathbf b$ ，如果系数矩阵 $A$ 可逆，则有唯一解
$$
\mathbf x=A^{-1}\mathbf b
$$

# 基变换

坐标系（Coordinate system）的建立基于所选的基向量（Basis vectors）



通常选用平面直角坐标系 $Oxy$ ，基向量的坐标值为
$$
\mathbf i=\begin{bmatrix} 1 \\ 0 \end{bmatrix},\quad
\mathbf j=\begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$


图中 $\mathbf i,\mathbf j$ 为**标准坐标系下**的基向量。

假设二维空间中使用另外一个非标准基 $\mathbf i',\mathbf j'$，在标准坐标系下坐标表示为（基变换公式）
$$
\mathbf i'=\begin{bmatrix} a \\ c \end{bmatrix},\quad
\mathbf j'=\begin{bmatrix} b \\ d \end{bmatrix}
$$
特别的，两个坐标系**原点的定义**是一致的

同一个向量在不同基向量下表示不同

非标准基下的向量 $\mathbf v=\begin{bmatrix} x' \\ y' \end{bmatrix}$，可以写成基向量的线性组合形式
$$
\mathbf v=x'\mathbf i'+y'\mathbf j'
$$
在标准坐标系下的表示
$$
\mathbf v=x'\begin{bmatrix} a \\ c \end{bmatrix}+y'\begin{bmatrix} b \\ d \end{bmatrix}=\begin{bmatrix} a&b \\ c&d \end{bmatrix}\begin{bmatrix} x' \\ y' \end{bmatrix}=\begin{bmatrix} x \\ y \end{bmatrix}
$$
进一步，因为是线性变换，所以将其转化为矩阵乘法，矩阵的列是  $\mathbf i,\mathbf j$ 在变换到  $\mathbf i',\mathbf j'$ 后的位置

特别的，这里 $A=\begin{bmatrix} a&b \\ c&d \end{bmatrix}$ 就意味着是 $\mathbf i',\mathbf j'$ 在标准坐标系表示下的基坐标。

- 将 $\mathbf i',\mathbf j'$ 构成的空间网格称为 "詹妮弗的网格(Jennifer's grid)"
- 因为 $\mathbf i,\mathbf j$ 构成的是标准空间，因此构成的空间网格为“我们的网格(our grid)”

基变换矩阵（change of basis matrix）

![](Linear-Algebra%202.assets/change_of_basis.svg)

Jennifer使用以 $\mathbf i',\mathbf j'$ 为基向量的坐标系，我们使用以 $\mathbf i,\mathbf j$ 为基向量的标准坐标系

基变换矩阵为
$$
\begin{bmatrix} a&b \\ c&d \end{bmatrix}
$$
即以我们的视角描述Jennifer的基向量

实际上在各自的坐标系统，基向量均为 $(1,0),(0,1)$

特别的，:heart:**几何上表示的是这个矩阵是将我们的网络变换为Jennifer的网络，但是数值上是将Jennifer的语言用我们的语言来表示**

逆变换（Inverse）：**几何上表示的是这个矩阵是将Jennifer的网络变换为我们的网络，但是数值上是将我们的语言变成了Jennifer的语言**
$$
\begin{bmatrix} x' \\ y' \end{bmatrix}=\begin{bmatrix} a&b \\ c&d \end{bmatrix}^{-1}\begin{bmatrix} x \\ y \end{bmatrix}
$$
$\begin{bmatrix} x \\ y \end{bmatrix}$ 是在标准基向量的表示结果，$\begin{bmatrix} x' \\ y' \end{bmatrix}$ 是在不同基向量下的表示结果，**两者实际是同一个向量，只不过是在不同基坐标系的表示结果**。

## 相似矩阵

不同坐标系之间的线性变换

线性变换对应的矩阵依赖于所选择的基



同样的变换$\sigma$，矩阵表示依赖基向量的选择

线性变换、矩阵乘法、逆时针旋转90度坐标系，这种表示**与我们所选择的基向量有关**。

矩阵$\begin{bmatrix} a&b \\ c&d \end{bmatrix}$ 是追踪基向量$\mathbf i,\mathbf j$ 变换后的位置得到的

同样的线性变换在$\mathbf i',\mathbf j'$ 下的表示，也需要追踪基向量$\mathbf i',\mathbf j'$ 变换后的位置





$$
P^{-1}AP
$$

暗示着数学上的转移作用，中间的矩阵 $A$ 代表**标准坐标系下**所见到的变换，$P,P^{-1}$ 两个矩阵代表着转移作用（基变换矩阵），**也就是在不同于标准坐标系与标准坐标系之间进行转换, 实际上也是视角上的转化。**。矩阵乘积仍然代表同一个变换 $\sigma$，只不过是从别的坐标系的角度来看。

具体过程如下

对于Jennifer视角下的向量 $\mathbf v=\begin{bmatrix} x' \\ y' \end{bmatrix}$

(1) 同样的向量，用标准坐标系（我们的坐标系）表示 $P\begin{bmatrix} x' \\ y' \end{bmatrix}$

(2) 用我们的语言描述变换后的向量 $AP\begin{bmatrix} x' \\ y' \end{bmatrix}$

(3) 将变换后的结果变换回Jennifer的坐标系 $P^{-1}AP\begin{bmatrix} x' \\ y' \end{bmatrix}$

因此 $A$ 和 $P^{-1}AP$ 表示同一种变换在不同基向量下的表示，其中$P$ 为基变换矩阵

# 特征值和特征向量

**❤️线性变换的两种理解方式**

- 将基向量变化后的位置视为矩阵的列，也就是新的基向量【**依赖于坐标系**】
- 利用特征向量和特征值理解线性变换，**不依赖于坐标系的选择**

❤️特征值与特征向量

**特征向量：一个向量经过线性变换，仍留在它所张成的空间中**

**特征值：描述特征向量经过线性变换后的缩放程度**

**❤️求解特征向量就是寻找非零向量** $\mathbf u$ **使得** $(A-\lambda I)\mathbf u=0$

**❤️求解特征值就是寻找一个 $\lambda$ 使得 $\det(A-\lambda I)=0$**



<img src="Linear-Algebra%202.assets/eigenvectors_with_eigenvalue.svg" style="zoom:80%;" />

<kbd>定义</kbd>：对于矩阵 $A$ ，如果存在数 $\lambda$ 和非零向量 $\mathbf u$，使得
$$
A\mathbf u=\lambda\mathbf u
$$

则称$\lambda$ 是矩阵 $A$ 的一个**特征值**（eigenvalue），$\mathbf u$ 是特征值 $\lambda$ 的一个**特征向量**（eigenvector）。
$$
(A-\lambda I)\mathbf u=0
$$

$$
A-\lambda I=\begin{bmatrix}a-\lambda &b\\c &d-\lambda\end{bmatrix}
$$



显然，$\mathbf u=0$ 恒成立，但是我们要寻找的是非零解。 基于行列式的知识，**就是相当于求 $\det(A-\lambda I)=0$，当且仅当矩阵所代表的线性变换将空间压缩到更低的维度时，才会存在一个非零向量，使得矩阵与它的乘积为零向量**。





> (1) 特征向量必须是非零向量；
> (2) 特征值和特征向量是相伴出现的。

事实上，对于任意非零常数$k$， $k\mathbf u$ 都是特征值 $\lambda$ 的特征向量
$$
A(k\mathbf u)=\lambda (k\mathbf u)
$$
由于矩阵和线性变换是一一对应的，我们可以借助几何直观理解这个定义。

属于矩阵不同特征值的特征向量不共线。
$$
(A-\lambda I)\mathbf u=0
$$
有非零解的充分必要条件是系数矩阵行列式为零，记
$$
f(\lambda)=\det(A-\lambda I)
$$
通过求解一元 $n$ 次方程 $f(\lambda)=0$ 即可获得特征值和对应的特征向量。

在变换过程中只受到拉伸或者压缩。

特征向量在变换中拉伸或者压缩的比例因子叫做特征值。

如果没有实数解，说明没有特征向量。

**理解线性变换作用，如果采用矩阵的列可以看做变换后的基向量，这就依赖于特定的坐标系。为了减少对坐标系的依赖，更好的办法就是求出对应的特征向量与特征值。**

10.3.2 计算特征值与特征向量

**特征方程** (characteristic equation)。矩阵 A 的特征值就是它的特征方程的根.
矩阵Ａ的**特征多项式** (characteristic polynomial)



求解特征值与特征向量的简要步骤：
(1) 
(2) 
(3)

对于三维空间中的旋转，如果能够找到对应的特征向量，也即能够留在它所张成的空间中的向量，那么就意味着我们找到了旋转轴。特别地，**这就意味着将一个三维旋转看成绕这个特征向量旋转一定角度**，要比考虑相应的矩阵变换要直观。

此时对应的特征值为1，因为**旋转并不改变任何一个向量** ，所以向量的长度保持不变。

![](Linear-Algebra%202.assets/axis_of_rotation.gif)





同样的线性变换，特征向量和特征值不依赖于坐标系的选择（相似矩阵）
$$
B=P^{-1}AP
$$

$$
B\mathbf u=\lambda\mathbf u \\
P^{-1}AP\mathbf u=\lambda\mathbf u \\
A(P\mathbf u)=\lambda (P\mathbf u) \\
A\mathbf u'=\lambda \mathbf u'
$$

所以B和A的特征值相同。
$$
P(AB)P^{-1}=BA
$$
BA 和 AB 有相同的特征值，有了上面的结论，那么我们只用说明 BA 和 AB 相似就能解决问题

## 特征基与对角化

(1) 并非所有的矩阵都存在特征向量，例如二维旋转变换 $R_{90\degree}$
$$
\begin{bmatrix}0 &-1\\1 &0\end{bmatrix}
$$
$\lambda=i$ 或 $\lambda=-i$

**与 $i$ 相乘在复平面中表示为90度旋转，这和 $i$ 是这个二维实向量旋转变换的特征值有所关联。值得注意的一点就是，特征值出现虚数的情况一般对应于变换中的某一种旋转。**

(2) 特征向量不能张成空间，例如水平剪切变换
$$
\begin{bmatrix}1 &1\\0 &1\end{bmatrix}
$$
只有一个特征值$\lambda=1$ 

(3) 等比例缩放线性变换
$$
\begin{bmatrix}2 &0\\0 &2\end{bmatrix}
$$
它仅存在唯一的特征值 2，但平面内任意一个向量都属于这个特征值的特征向量。

❤️当特征向量的数量足够多时，这些特征向量就可以构成特征基。利用特征基可以简化矩阵的幂次计算。

对于对角矩阵，
$$
\begin{bmatrix}a &0\\0 &c\end{bmatrix}^n=\begin{bmatrix}a^n &0\\0 &c^n\end{bmatrix}
$$

$$
\begin{vmatrix}a-\lambda &0\\0 &c-\lambda\end{vmatrix}=0
$$

$$
f(\lambda)=(a-\lambda)(c-\lambda)=0
$$

特征值 $\lambda=a$ 或 $\lambda=c$

所有基向量都是特征向量，对角元素就是所属的特征值。
$$
\begin{bmatrix}a-a &0\\0 &c-a\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}0\\(c-a)y\end{bmatrix}=0
$$
解出 $y=0,x\in\R$，因此特征向量$\mathbf u_1=\begin{bmatrix}1\\0\end{bmatrix}$



利用特征向量可以将矩阵进行对角化。 用特征向量作为基向量，变换你的坐标系。

在特征基坐标系，矩阵只是伸缩变换

基变换矩阵（change of basis matrix）
$$
P=\begin{bmatrix} a&b \\ c&d \end{bmatrix}
$$

$$
A\begin{bmatrix} a \\ c \end{bmatrix}=\lambda_1\begin{bmatrix} a \\ c \end{bmatrix} \\
A\begin{bmatrix} b \\ d \end{bmatrix}=\lambda_2\begin{bmatrix} b \\ d \end{bmatrix}
$$



$\Lambda=P^{-1}AP$

这样所得的矩阵代表同一个变换，只不过是从新基向量所构成的坐标系的角度来看的。

矩阵$A$的特征值与特征向量对应关系 $A\mathbf u_1=\lambda_1\mathbf u_1,\quad A\mathbf u_2=\lambda_2\mathbf u_2$ ，令$P=[\mathbf u_1,\mathbf u_2]$ 
$$
AP=[\lambda_1\mathbf u_1,\lambda_2\mathbf u_2]=
[\mathbf u_1,\mathbf u_2]
\begin{bmatrix} \lambda_1&0 \\ 0&\lambda_2 \end{bmatrix}=
P\Lambda \\
$$

$$
\Lambda=P^{-1}AP=\begin{bmatrix} \lambda_1&0 \\ 0&\lambda_2 \end{bmatrix}
$$



**用线性无关的特征向量来完成这件事情的意义在于：最终变换的矩阵必然是对角阵，且对角元就是对应的特征向量。这是因为它处坐标系的基向量在变换中仅仅进行了缩放**

特征向量 $\mathbf u_1,\mathbf u_2$ 对应的特征值分别为 $\lambda_1,\lambda_2$
$$
\Lambda=P^{-1}AP=\begin{bmatrix} \lambda_1&0 \\ 0&\lambda_2 \end{bmatrix}
$$

5) 利用特征基计算矩阵的幂次

**特征值对应的所有的特征向量构成的集合被称为一个“特征基”**eigenbasis





## 特征向量的性质


$$
A^{-1}\mathbf u=\lambda^{-1}\mathbf u
$$


特征向量在数学和实际问题中有着广泛的应用，许多实际问题都可归结为研究矩阵的方幂 $A^n\quad (n\in\N^*)$ 乘以向量 $\mathbf v$ 。不难想象，当方幂很大时，直接用矩阵的乘法、矩阵与向量的乘法进行计算会非常麻烦。

性质：
$$
A^n\mathbf u=\lambda^n\mathbf u
$$

证明：数学归纳法证明
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
A^k\mathbf u=A(\lambda^{k-1}\mathbf u)=\lambda^k\mathbf u
$$

所以，对 $n=k$ 时成立。由数学归纳法可知，对所有的 $n\in\N^*$ 都成立。

性质2：对于任意向量 $\mathbf v$ ，可以找到实数 $k_1,k_2$，使得
$$
\mathbf v=k_1\mathbf u_1+k_2\mathbf u_2
$$
即以特征向量为基向量构建。那么，用数学归纳法可以得到
$$
A^n\mathbf v=k_1\lambda_1^n\mathbf u_1+k_2\lambda_2^n\mathbf u_2
$$
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

(3) 可以证明：矩阵
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







# 向量空间

行列式告诉你一个变换对面积的缩放比例，特征向量则是在变换中保留在他所张成的空间中的向量，这两者都是暗含与空间中的性质，坐标系的选择并不会改变他们最根本的值。



向量 $\mathbf e_1$ 和  $\mathbf e_2$ 全部线性组合构成的向量集合称为他们**张成的空间**（span）
$$
x_1\mathbf e_1+x_2\mathbf e_2 \quad (x_1,x_2\in\R)
$$
**线性相关**（Linear dependent）：意味着向量可以用其他向量的线性组合来表示
$$
\mathbf v=x_1\mathbf e_1+x_2\mathbf e_2 \quad (x_1,x_2\in\R)
$$
因为该向量已经落在了线性组合张成的空间中

**线性无关**（Linear independent）：所有向量都给张成的空间添加了新的维度
$$
\mathbf v\neq x_1\mathbf e_1+x_2\mathbf e_2 \quad (x_1,x_2\in\R)
$$
向量空间中的**基**（basis）是张成该空间中的一个线性无关的向量集合。

## 列空间和秩



秩Rank代表变换后空间的维数

矩阵的列告诉我们基向量变换之后的位置，列空间就是矩阵的列所张成的空间。

秩的定义是列空间的维数。

满秩，就是秩等于列数

零向量一定在列空间内，满秩变换中，唯一能落在原点的就是零向量自身。

变换后，落在零向量的点的集合是**零空间**，或者叫核，所有可能解的集合。

零空间/核（Null space/Kernel）

对于**非满秩矩阵**，意味着该线性变换会对空间进行压缩到一个更低维的空间，通俗来讲，就是会有一系列直线上不同方向的向量压缩为原点。

$3\times 2$，矩阵是把二维空间映射到三维空间上，因为矩阵有两列，说明输入空间有两个基向量，三行表示每一个基向量在变换后用三个独立的坐标来描述。

$2\times 3$，矩阵是把三维空间映射到二维空间上，因为矩阵有三列，说明输入空间有三个基向量，二行表示每一个基向量在变换后用二个独立的坐标来描述。

由此可得，**秩可以用来描述线性变换对空间的压缩程度**。

**秩** ⟺ **变换后空间的维数。** ⟺ **列空间的维数**。

## 点积

向量点积（dot product）等价于矩阵向量乘积（matrix-vector product）

即向量在一维空间的线性变换
$$
\begin{bmatrix} u_x\\ u_y\end{bmatrix}\cdot\begin{bmatrix} x\\ y\end{bmatrix}=u_x\cdot x+u_y\cdot y
$$

$$
\begin{bmatrix} u_x & u_y\end{bmatrix}\begin{bmatrix} x\\ y\end{bmatrix}=u_x\cdot x+u_y\cdot y
$$

## 叉积

平面叉积
$$
\begin{bmatrix}v_1\\v_2\end{bmatrix}\times\begin{bmatrix}w_1\\w_2\end{bmatrix}=\det\begin{bmatrix}v_1 & w_1\\ v_2 & w_2 \end{bmatrix}
$$
大小等于 $v,w$ 围成的平行四边形的面积

三维叉积
$$
\begin{bmatrix}v_1\\v_2\\v_3\end{bmatrix}\times\begin{bmatrix}w_1\\w_2\\w_3\end{bmatrix}=\det\begin{bmatrix}\mathbf i & v_1 & w_1\\\mathbf j & v_2 & w_2 \\\mathbf k & v_3 & w_3 \end{bmatrix}
$$
大小等于 $v,w$ 围成的平行四边形的面积，方向遵循右手定则。

## 线性空间

**普适的代价是抽象**

抽象，如“3”可以代表任何事物

满足**可加性与伸缩性**，则对应“线性变换”：

函数是一种特殊的向量空间

向量空间中的概念都可类比到函数

特别地，求导是一种特殊的线性运算
$$
\begin{aligned}
&\text{Additivity}: &\frac{\mathrm d}{\mathrm dx}(f+g)=\frac{\mathrm d}{\mathrm dx}f+\frac{\mathrm d}{\mathrm dx}g & \\
&\text{Scaling}: &\frac{\mathrm d}{\mathrm dx}(cf)=c\frac{\mathrm d}{\mathrm dx}f 
\end{aligned}
$$


**多项式空间**



基函数（Basis functions）
$$
b_0(x)=1 \\
b_1(x)=x \\
b_2(x)=x^2 \\
\vdots
$$
基函数起到的作用于类似于空间中的基向量。

多项式函数 $f(x)=a_0+a_1x_1+a_2x^2+\cdots+a_nx^n$ 对应的向量
$$
f=\begin{bmatrix}a_0\\a_1\\a_2\\\vdots\\a_n\\ \vdots\end{bmatrix}
$$
求导类似于线性变换（线性算子）
$$
\frac{\mathrm d}{\mathrm dx}=\begin{bmatrix}
0&1&0&0&\cdots\\ 0&0&2&0&\cdots\\
0&0&0&3&\cdots\\ 0&0&0&0&\cdots\\
\vdots&\vdots&\vdots&\vdots&\ddots
\end{bmatrix}
$$

$$
\frac{\mathrm d}{\mathrm dx}(1x^3+5x^2+4x+5)=3x^2+10x+4 \\
\begin{bmatrix}
0&1&0&0&\cdots\\ 0&0&2&0&\cdots\\
0&0&0&3&\cdots\\ 0&0&0&0&\cdots\\
\vdots&\vdots&\vdots&\vdots&\ddots
\end{bmatrix}\begin{bmatrix}
5\\4\\5\\1\\ \vdots
\end{bmatrix}=
\begin{bmatrix}
4\\10\\3\\0\\ \vdots
\end{bmatrix}
$$
<kbd>线性空间</kbd>：设 $V$ 是为非空集合，$\mathbb F$ 是一个数域，定义两种运算：加法 $\forall\mathbf v,\mathbf w\in V,\ \mathbf v+\mathbf w\in V$ 和数乘 $\forall c\in\mathbb F,\mathbf v\in V,\ c\mathbf v\in V$（满足封闭性）。若这两种运算满足以下8条性质，则称 $V$ 为 $\mathbb F$ 上的**线性空间**。

1. 加法交换律：$\forall\mathbf v,\mathbf w\in V,\ \mathbf v+\mathbf w=\mathbf w+\mathbf v$
2. 加法结合律：$\forall\mathbf u,\mathbf v,\mathbf w\in V,\ \mathbf u+(\mathbf v+\mathbf w)=(\mathbf u+\mathbf v)+\mathbf w$
3. 加法单位元：$\forall\mathbf v\in V,\exists 0\in V,\ 0+\mathbf v=\mathbf v$
4. 加法逆元：$\forall\mathbf v\in V,\exists \mathbf w\in V,\ \mathbf v+\mathbf w=0$
5. 数乘结合律：$\forall a,b\in\mathbb F,\forall\mathbf v\in V,\ a(b\mathbf v)=(ab)\mathbf v$
6. 数乘分配律：$\forall a\in\mathbb F,\forall\mathbf v,\mathbf w\in V,\ a(\mathbf v+\mathbf w)=a\mathbf v+a\mathbf w$
7. 数乘分配律：$\forall a,b\in\mathbb F,\forall\mathbf v\in V,\ (a+b)\mathbf v=a\mathbf v+b\mathbf v$
8. 数乘单位元：$\forall\mathbf v\in V,\exists 1\in\mathbb F,\ 1\mathbf v=\mathbf v$



# 奇异值分解

矩阵的奇异值分解是指将$m\times n$实矩阵$A$表示为以下三个实矩阵乘积形式的运算
$$
A=U\mit\Sigma V^\mathrm T
$$

 中间有一句，可以假设正交矩阵$V$的列的排列使得对应的特征值形成降序排列。这句怎么理解？

列是轴，实际上不同列的排列，对应的是坐标轴的顺序，不同坐标系顺序的选择，和实际上拿到的最后的向量是没有关系的。

### 几何解释

$A_{m\times n}$表示了一个从$n$维空间$\mathbf{R}^n$到$m$维空间$\mathbf{R}^m$的一个**线性变换**
$$
T:x\rightarrow Ax\\
x\in\mathbf{R}^n\\
Ax\in \mathbf{R}^m
$$
线性变换可以分解为三个简单的变换：

1. 坐标系的旋转或反射变换，$V^\mathrm{T}$
1. 坐标轴的缩放变换，$\Sigma$
1. 坐标系的旋转或反射变换，$U$



# 参考视频



[【官方双语/合集】线性代数的本质 - 系列合集](https://www.bilibili.com/video/BV1ys411472E/)



