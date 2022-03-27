---
title: 偏微分方程(四)
categories:
  - 数学
  - 偏微分方程
tags:
  - 数学
  - PDE
  - 微分方程
  - 变分法
katex: true
description: 变分法、非线性偏微分方程
abbrlink: 4fc7d09d
date: 2020-07-10 16:52:58
cover:
top_img:
---

# 变分法初步

## 泛函的概念

**泛函**： 泛函是将任意给定的函数映射为一个数，简单的说就是以整个函数为自变量的函数。这个概念可以看成是函数概念的推广。
对于一元函数，以函数集合 $M=\{y(x)|a⩽x⩽b\}$ 为定义域，几何上表示某一平面曲线的集合。定义域内的函数  $y(x)$ 与数值存在映射，称为泛函  $J[y]$ 。

**示例 1**：（极小曲线问题）设在 $Oxy$ 平面上有一簇曲线 $y(x)$ ，其长度为泛函
$$
J[y]=\int_a^b\sqrt{1+y'^2}\mathrm dx
$$
显然，$J[y]$ 的数值依赖于整个函数 $y(x)$ 的改变而改变。对于函数，给定一个 $x$ 值，有一个函数值与之对应，对于泛函，则必须给出某一区间上的函数 $y(x)$ ，才能得到一个泛函值 $J[y]$ 。
（定义在同一区间上的）函数不同，泛函值当然不同。为了强调泛函值 $J[y]$ 与函数 $y(x)$ 之间的依赖关系，常常又把函数 $y(x)$ 称为==自变函数==。

泛函的形式可以是多种多样的，本课程中只限于用积分形式定义的泛函。
对于自变函数为一元函数 $y(x)$ ，则泛函为
$$
J[y]=\int_a^bL(x,y,y')\mathrm dx\tag{1.1}
$$
其中 $L$ 是已知函数，具有连续的二阶偏导数。
如果自变函数是二元函数 $u(x,y)$ ，则泛函为
$$
J[u]=\iint\limits_SL(x,y,u,u_x,u_y)\mathrm dx\mathrm dy\tag{1.2}
$$
对于更多个自变量的多元函数，也有类似的定义。

**示例 2**：（最速下降问题）如图，在重力作用下，一质点从 $(x_0,y_0)$ 点沿平面曲线 $y(x)$ 无摩擦自由下滑到 $(x_1,y_1)$ ，则所需的时间为 $y(x)$ 的泛函
$$
\begin{aligned}
J[y]= & \int_{x_0}^{x_1}\cfrac{\mathrm ds}{\sqrt{2g(y_0-y)}} \\
= & \int_{x_0}^{x_1}\cfrac{\sqrt{1+y'^2}}{\sqrt{2g(y_0-y)}}\mathrm dx
\end{aligned}
$$
这里，要求 $y(x)$ 一定通过端点 $(x_0,y_0)$ 和 $(x_1,y_1)$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/brachistochrone-problem.png)

**示例 3**：（极小曲面问题）设在空间上的光滑曲面簇 $u(x,y)$ ，其曲面面积定义了泛函
$$
J[y]=\iint\limits_S\sqrt{1+u_x^2+u_y^2}\mathrm dx\mathrm dy
$$
## 泛函的极值

**变分法基本引理**：设 $f(x)$ 在 $[a,b]$ 上连续，若对于任意满足边界条件 $h(a)=h(b)=0$ 的函数 $h(x)$ 均有 
$$
\int_a^bf(x)h(x)\mathrm dx=0
$$
则必有 $f(x)=0$ 。由于 $h(x)$ 的随意性，此引理可用反证法证明。对于多元函数重积分的情况也有类似的引理。

**变分的概念**：对于函数 $f(x)$ 假设自变量 $x$ 不变，改变函数的形式得到一个与原函数稍有差别的新函数 $\bar f(x)=f(x)+δg(x)$ 。其中，$g(x)$ 是任意连续函数，$δ$ 是微小系数，即对于 $\forallϵ>0,|δy(x)|<ϵ$ 。
对于函数的任意自变量，函数 $f(x)$ 由于**形式上的微小改变**而得到的改变量称为该函数的==变分==。自变函数的变分其实是自变量微分的推广。如下图，函数 $y(x)$ 的变分 $δy(x)$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/variation.png)

以一元函数为例，总结变分的几条**运算性质**
(1) 由于变分独立于函数自变量，所以变分与微分可以交换
$$
δf'=(δf)'
$$
(2) 线性运算
$$
δ(\alpha f+\beta g)=\alpha\cdot δf+\beta\cdot δg
$$
(3) 乘积的变分运算
$$
δ(f\cdot g)=(δf)\cdot g+f\cdot (δg)
$$
(4) 积分的变分运算（只需把定积分看成级数和即可证明）
$$
δ\int_a^bf\mathrm dx=\int_a^b(δf)\mathrm dx
$$
(5) 复合函数的变分运算，其法则和微分运算完全相同，例如（注意变分和自变量无关）
$$
δF(x,y,y')=\cfrac{∂F}{∂y}δy+\cfrac{∂F}{∂y'}δy'
$$
**一元函数泛函的极值**：类似于函数的极值问题，若对自变函数为 $y_0(x)$ 及其附近的自变函数 $y_0(x)+δy(x)$ ，恒有 $J[y]⩽J[y_0+δy]$ ，则称泛函 $J[y]$ 在自变函数为 $y_0(x)$ 时取得极小值。类似的，可以定义泛函的极大值。极大值和极小值统称为==极值==。
可以仿照函数的方法，导出泛函取极值的必要条件。对于一元函数 $y(x)$ 的泛函
$$
J[y]=\int_a^bL(x,y,y')\mathrm dx
$$
泛函的差值为
$$
J[y+δy]-J[y]=\int_a^b[L(x,y+δy,y'+δy')-L(x,y,y')]\mathrm dx
$$
考虑到函数的变分 $δy(x)$ 足够小，可以将上式被积函数在极值函数附近作泰勒展开
$$
\begin{aligned}
J[y+δy]-J[y]= & \int_a^b[(δy\cfrac{∂}{∂y}+\cfrac{∂}{∂y'}δy')L
+\cfrac{1}{2!}(δy\cfrac{∂}{∂y}+\cfrac{∂}{∂y'}δy')^2L+\cdots]\mathrm dx \\
= & \int_a^b(δy\cfrac{∂L}{∂y}+\cfrac{∂L}{∂y'}δy')\mathrm dx + \cfrac{1}{2!}\int_a^b(δy\cfrac{∂}{∂y}+\cfrac{∂}{∂y'}δy')^2L\mathrm dx+\cdots
\end{aligned}
$$
其中定义
$$
δJ[y]=\int_a^b(δy\cfrac{∂L}{∂y}+δy'\cfrac{∂L}{∂y'})\mathrm dx\tag{2.1}
$$
是泛函 $J[y]$ 的==一级变分==。
$$
δ^2J[y]=\int_a^b(δy\cfrac{∂}{∂y}+\cfrac{∂}{∂y'}δy')^2L\mathrm dx\tag{2.2}
$$
是泛函 $J[y]$ 的==二级变分==，依次可以定义三级及以上高级变分。
函数 $f(x)$ 取极值的必要条件为 $f'(x)=0$ ，微分形式为 $\mathrm df(x)=f'(x)\mathrm dx=0$ 。
注意到 $δy(x)$ 和 $δy'(x)$ 永远是微量，因此舍弃掉二次项及以上高次项。和函数取极值类似，泛函取极值的必要条件为泛函的一级变分为零
$$
δJ[y]=\int_a^b(δy\cfrac{∂L}{∂y}+δy'\cfrac{∂L}{∂y'})\mathrm dx=0\tag{2.3}
$$
 将第二项分部积分
$$
δJ[y]=\cfrac{∂L}{∂y'}δy\Big|_a^b+\int_a^b(\cfrac{∂L}{∂y}
-\cfrac{\mathrm d}{\mathrm dx}\cfrac{∂L}{∂y'})δy\mathrm dx=0
$$
根据变分法基本引理，$y(x)$ 为极值时必须满足
$$
\cfrac{∂L}{∂y}-\cfrac{\mathrm d}{\mathrm dx}\cfrac{∂L}{∂y'}=0\tag{2.4}
$$
和边界条件
$$
\cfrac{∂L}{∂y'}\Big|_bδy(b)-\cfrac{∂L}{∂y'}\Big|_aδy(a)=0\tag{2.5}
$$
方程 (2.4) 称为 ==Euler-Lagrange 方程==。
如果 $L=L(x,y')$ ，则 Euler-Lagrange 方程化为 $\cfrac{\mathrm d}{\mathrm dx}\cfrac{∂L}{∂y'}=0$ ，它的首次积分 $F_{y'}=C_1$
如果 $L=L(y,y')$ ，则 Euler-Lagrange 方程的首次积分 $F-y'F_{y'}=C_1$
如果 $L=L(x,y)$ ， Euler-Lagrange 方程是一个隐函数方程 $F_y=0$
值得指出的是，E-L方程只是泛函有极值的必要条件，并不是充分条件。就是说，当泛函有极值时，E-L方程成立，解可能不止一个，它们只是极值函数的候选者。
(1) 如果边界值 $y(a)=\alpha,y(b)=\beta$，即两个端点固定的情形，此时
$$
δy(a)=δy(b)=0
$$
一定满足边界条件 (2.5)
(2) 如果边界值 $y(a),y(b)$ 可以任意取值，此时 $δy(a),δy(b)$ 也可以任意取值，所以极值满足
$$
\cfrac{∂L}{∂y'}\Big|_a=\cfrac{∂L}{∂y'}\Big|_b=0
$$
此边界条件称为自然边界条件。
(3) 如果一端固定，如 $y(a)=\alpha,y(b)$ 自由滑动，所以极值满足
$$
y(a)=\alpha,\cfrac{∂L}{∂y'}\Big|_b=0
$$
**二元函数泛函的极值**： 对于二元函数 $u(x,y)$ 的泛函
$$
J[u]=\iint\limits_SL(x,y,u,u_x,u_y)\mathrm dx\mathrm dy
$$
二元函数的泛函取极值的必要条件依然为泛函的一级变分为零。
$$
\begin{aligned}
δJ[u]= & \iint\limits_SδL(x,y,u,u_x,u_y)\mathrm dx\mathrm dy \\
=& \iint\limits_S[\cfrac{∂L}{∂u}δu+\cfrac{∂L}{∂u_x}δu_x+\cfrac{∂L}{∂u_y}δu_y]
\mathrm dx\mathrm dy \\
=& \iint\limits_S[\cfrac{∂L}{∂u}δu+\cfrac{∂L}{∂u_x}(δu)_x+\cfrac{∂L}{∂u_y}(δu)_y]
\mathrm dx\mathrm dy \\
=& \iint\limits_S\left[\cfrac{∂L}{∂u}
-\cfrac{∂}{∂x}\left(\cfrac{∂L}{∂u_x}\right)
-\cfrac{∂}{∂y}\left(\cfrac{∂L}{∂u_y}\right)\right]
δu\mathrm dx\mathrm dy \\
+& \iint\limits_S\left[\cfrac{∂}{∂x}\left(\cfrac{∂L}{∂u_x}δu\right)
+\cfrac{∂}{∂y}\left(\cfrac{∂L}{∂u_y}δu\right)\right]\mathrm dx\mathrm dy \\
= &0
\end{aligned}
$$
上式中使用格林公式
$$
\iint\limits_S(\cfrac{∂Q}{∂x}-\cfrac{∂P}{∂y})\mathrm dx\mathrm dy=
\int_{∂S} P\mathrm dx+Q\mathrm dy
$$
可得到
$$
\iint\limits_S\left[\cfrac{∂}{∂x}\left(\cfrac{∂L}{∂u_x}δu\right)
+\cfrac{∂}{∂y}\left(\cfrac{∂L}{∂u_y}δu\right)\right]\mathrm dx\mathrm dy
=\int_{∂S} \left[-\cfrac{∂L}{∂u_y}\mathrm dx+\cfrac{∂L}{∂u_x}\mathrm dy\right]δu
$$
因此，二元函数泛函取极值的必要条件（积分形式）为
$$
δJ[u]=\iint\limits_S\left[\cfrac{∂L}{∂u}
-\cfrac{∂}{∂x}\left(\cfrac{∂L}{∂u_x}\right)
-\cfrac{∂}{∂y}\left(\cfrac{∂L}{∂u_y}\right)\right]
δu\mathrm dx\mathrm dy \\
+\int_{∂S} \left[-\cfrac{∂L}{∂u_y}\mathrm dx+\cfrac{∂L}{∂u_x}\mathrm dy\right]δu=0
$$
根据变分法基本引理，极值满足 Euler-Lagrange 方程（微分形式）
$$
\cfrac{∂L}{∂u}
-\cfrac{∂}{∂x}\left(\cfrac{∂L}{∂u_x}\right)
-\cfrac{∂}{∂y}\left(\cfrac{∂L}{∂u_y}\right)=0\tag{2.6}
$$
和边界条件
$$
\int_{∂S} \left[-\cfrac{∂L}{∂u_y}\mathrm dx+\cfrac{∂L}{∂u_x}\mathrm dy\right]δu=0\tag{2.7}
$$
(1) 若函数满足第一类边界条件 $u\Big|_{∂S}=\phi(x,y)$，即边界数值固定，因此
$$
δu(x,y)\Big|_{∂S}=0
$$
则一定满足边界条件 (2.7)
(2) 如果允许函数在边界面上自由取值，则极值必须满足边界条件
$$
\left[\cfrac{∂L}{∂u_x}\mathrm dy-\cfrac{∂L}{∂u_y}\mathrm dx\right]\Big|_{∂S}=0
$$
类似的可以讨论三元及以上函数泛函和泛函极值的情况。

##  泛函的条件极值

**条件极值**：和函数求[条件极值][math2]类似，泛函
$$
J[y,z]=\int_a^bF(x,y,z,y',z')\mathrm dx
$$
在约束条件
$$
G(x,y,z)=0
$$
的条件极值也可用Lagrange 乘数法求解。
设约束条件 $G=0$ 确定的微分方程为 $z=\phi(x,y)$，带入泛函可得
$$
J[y]=\int_a^bF_1(x,y,y')\mathrm dx
$$
其中 $F_1(x,y,y')=F(x,y,\phi,y',\phi_x+\phi_yy')$ ，实际上用消元法把条件极值问题转化为无条件极值，所以极值满足的方程为
$$
\cfrac{∂F_1}{∂y}-\cfrac{\mathrm d}{\mathrm dx}\cfrac{∂F_1}{∂y'}=0
$$
于是
$$
(\cfrac{∂F}{∂y}-\cfrac{\mathrm d}{\mathrm dx}\cfrac{∂F}{∂y'})
+\phi_z(\cfrac{∂F}{∂z}-\cfrac{\mathrm d}{\mathrm dx}\cfrac{∂F}{∂z'})=0
$$
又因为 $G(x,y,z)=0,\phi_z=-\frac{G_y}{G_z}$ 所以
$$
\cfrac{1}{G_y}(\cfrac{∂F}{∂y}-\cfrac{\mathrm d}{\mathrm dx}\cfrac{∂F}{∂y'})
=\cfrac{1}{G_z}(\cfrac{∂F}{∂z}-\cfrac{\mathrm d}{\mathrm dx}\cfrac{∂F}{∂z'})
$$
等式两端都是关于 $x$ 的函数，记为 $-λ(x)$ ，则
$$
\begin{cases} 
F_y-\cfrac{\mathrm d}{\mathrm dx}F_{y'}+λ(x)G_y=0 \\
F_z-\cfrac{\mathrm d}{\mathrm dx}F_{z'}+λ(x)G_z=0
\end{cases}
$$
故得出了泛函条件极值的拉格朗日乘数法。

[math2]: /posts/Calculus-II/

**等周问题**：在约束条件
$$
J_1[y]=\int_a^bG(x,y,y')\mathrm dx=L
$$
下求泛函
$$
J[y]=\int_a^bF(x,y,y')\mathrm dx
$$
的极值问题，称为等周问题。其中，$L$ 为常数，边界值固定
$$
δy(a)=δy(b)=0
$$
可以证明等价于泛函
$$
J_0[y]=J[y]-λJ_1[y]
$$
的极值问题。因此必要条件是
$$
\left(\cfrac{∂}{∂y}-\cfrac{\mathrm d}{\mathrm dx}\cfrac{∂}{∂y'}\right)(F-λG)=0
$$

## 微分方程的变分法

泛函取极值的必要条件（Euler-Lagrange 方程）是常微分方程或偏微分方程，它和自变函数的定解条件结合起来，就构成一个定解问题。泛函的条件极值问题，其必要条件中出现待定参数，它和齐次边界条件结合起来，就构成本征值问题。

现在研究将微分方程的定解问题或本征值问题转化为泛函的无条件极值或条件极值问题，这称为微分方程的==变分法==。不难理解，本征函数正好泛函的极值函数，而本征值正好是泛函的极值。

**示例 1**：考虑二阶线性方程
$$
\cfrac{\mathrm d}{\mathrm dx}[k(x)y']
+q(x)y-f(x)=0\quad(a<x<b)
$$
根据变分法基本引理，方程可来源于积分
$$
\int_a^b\left[\cfrac{\mathrm d}{\mathrm dx}(ky')
+qy-f\right]δy\mathrm dx=0
$$
因为已知函数 $q(x),f(x)$ 与变分 $δy$ 无关，因此变分计算中可看做常数，上式中第二三项可直接化为
$$
\int_a^bqyδy\mathrm dx=\cfrac{1}{2}δ\int_a^bqy^2\mathrm dx \\
\int_a^bfδy\mathrm dx=δ\int_a^bfy\mathrm dx
$$
第一项可用分部积分法化为
$$
\int_a^b\left[\cfrac{\mathrm d}{\mathrm dx}(ky')\right]δy\mathrm dx
=k(x)y'δy(x)\Big|_a^b-\int_a^bk(x)y'(δy)'\mathrm dx
$$
若讨论第一边值问题，给定 $y(a)=\alpha,\quad y(b)=\beta$ ，则 $δy(a)=δy(b)=0$ ，于是
$$
\int_a^b\left[\cfrac{\mathrm d}{\mathrm dx}(ky')\right]δy\mathrm dx
=-\int_a^bk(x)y'(δy)'\mathrm dx
=-\cfrac{1}{2}δ\int_a^bk(x)y'^2\mathrm dx
$$
所以二阶线性方程可转化为泛函
$$
J[y]=\int_a^b[k(x)y'^2-q(x)y^2-2f(x)y]\mathrm dx
$$
取极值的必要条件。

几乎所有的物理和力学的基本规律都可陈述为规定某一泛函极值问题。于此，变分法使许多重要的物理问题及技术问题得以解决。 
费马原理：光线永远沿用时最短的路径传播。 
对于动力学系统，都遵循 Hamilton原理：在一个动力学系统中，质点系的真实运动满足积分
$$
J[u]=\int_{t_0}^{t_1}(T-V)\mathrm dt
$$
有极值的必要条件，即 $δJ[u]=0$ 。其中 $T,V$ 分别为 $t$ 时刻系统的总动能和总势能，函数 $L=T-V$ 称为 Lagrange 函数。

**示例 2**：有限长弦的强迫振动问题，$f(x,t)$表示横向力密度，$u(x,t)$ 表示横向位移。
弦的总动能和势能分别为
$$
T=\int_{a}^{b}\cfrac{1}{2}\rho(\cfrac{∂u}{∂t})^2\mathrm dx,\quad 
V=\int_{a}^{b}[\cfrac{1}{2}T(\cfrac{∂u}{∂x})^2-f(x,t)u]\mathrm dx
$$
其中 $\rho$ 是弦的线密度，$T$ 是张力，弦的端点为 $a,b$。由动力学理论知道， $u(x,t)$ 满足泛函
$$
J[u]=\int_{t_0}^{t_1}\mathrm dt\int_{a}^{b}
[\cfrac{\rho}{2}(\cfrac{∂u}{∂t})^2-\cfrac{T}{2}(\cfrac{∂u}{∂x})^2+f(x,t)u]\mathrm dx
$$
有极值的必要条件（Euler-Lagrange 方程）
$$
\rho\cfrac{∂^2u}{∂t^2}-T\cfrac{∂^2u}{∂x^2}=f(x,t)
$$

## Rayleigh-Ritz 方法

对于微分方程，在多数实际情况下，往往只能求得近似解。在变分法的基础上，建立了实用的近似解法。

**基本思路**：讨论一元函数 $y(x)$ 泛函
$$
J[y]=\int_a^bL(x,y,y')\mathrm dx
$$
的极值问题。可先选取一个合适的基函数序列 $\{\phi_i(x)\}$ 将函数 $y(x)$ 级数展开，设函数的 $n$ 级近似解为
$$
y_n(x)=\sum_{i=1}^nc_i\phi_i(x)
$$
由此得
$$
J[y_n]=\int_a^bL(x,y_n,y_n')\mathrm dx
$$
它是线性组合系数 $c_1,c_2,\cdots,c_n$ 的函数，记为 $J(c_1,c_2,\cdots,c_n)$ ，由多元函数取极值的必要条件知
$$
\cfrac{∂J(c_1,c_2,\cdots,c_n)}{∂c_i}=0\quad (i=1,2,\cdots,n)
$$
这时关于 $c_i$ 的方程组，解出组合系数，从而可确定泛函极值的 $n$ 级近似解。

# 非线性数学物理问题

**孤立子**：(soliton) 

1. KdV 方程

    $$
    \cfrac{∂u}{∂t}+\alpha u\cfrac{∂u}{∂x}+\cfrac{∂^3u}{∂x^3}=0
    $$
    主要描述浅水中的表面波、含气泡的水中的声波、磁流体及等离子体中的声波等。

2. Sin-Gordon 方程

    $$
    \cfrac{∂^2u}{∂t^2}-\cfrac{∂^2u}{∂x^2}+\sin u=0
    $$
    主要用于描述晶体中的位错运动、约瑟夫森结中的磁通运动等。

3. 非线性薛定谔方程

    $$
    \mathrm i\cfrac{∂u}{∂t}+\cfrac{∂^2u}{∂x^2}+\beta|u|^2u=0
    $$
    主要用于描述二维平面电磁波的自聚焦、一维单色波的自调制，光纤中超短光脉冲的传播等。

4. Toda 点阵方程

   $$
   \begin{aligned}
   & \cfrac{\mathrm dq_n}{\mathrm dt}=\cfrac{p_n}{m} \\
   & \cfrac{\mathrm dq_n}{\mathrm dt}=
   \exp(q_{n-1}-q_{n})-\exp(q_{n}-q_{n+1})
   \end{aligned}
   $$
   主要用于描述晶格点阵中的声传播。

------

> **参考文献：**
> 季孝达.《数学物理方程》.
> 吴崇试.《数学物理方法》.
> 梁昆淼.《数学物理方法》.
> 吴崇试 高春媛.《数学物理方法》.北京大学(MOOC)