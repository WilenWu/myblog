---
title: 偏微分方程(一)
categories:
  - Mathematics
  - 微分方程
tags:
  - 数学
  - PDE
  - 微分方程
  - 分离变量法
katex: true
description: （拟）线性偏微分方程、定解问题、分离变量法
abbrlink: 644684db
date: 2020-04-07 16:50:54
cover: /img/pde.png
top_img: /img/math-top-img.png
---

#  偏微分方程的定解问题

**偏微分方程** (Partial Differential Equations, PDE) ，是指含有多元未知函数及其若干阶偏导数的微分方程。
$$
F(u,x_1,x_2,\cdots,\cfrac{∂u}{∂x_1},\cfrac{∂u}{∂x_2},\cdots,\cfrac{∂^2u}{∂x_1^2},\cdots)=0\tag{1.1}
$$
其中最高阶偏导数的阶称为偏微分方程的阶。

在此引入一些重要的微分算子

Hamilton 算子： $∇=(\cfrac{∂}{∂x},\cfrac{∂}{∂y},\cfrac{∂}{∂z})$
Laplace算子 $Δ=\cfrac{∂^2}{∂x^2}+\cfrac{∂^2}{∂y^2}+\cfrac{∂^2}{∂z^2}=∇\cdot∇=∇^2$

## 定解问题及适定性

**数学物理方程** (Equation of Mathematical Physics) ，通常指从物理学及其他各门自然科学、技术科学中产生的偏微分方程。数学物理方程有时还包括常微分方程和积分方程。本课程将着重讨论三类基本的二阶线性偏微分方程。

(1) 弦的横振动方程：未知函数 $u(x,t)$ 表示坐标 $x$ 处弦的横向位移
$$
u_{tt}-a^2u_{xx}=f(x,t)
$$
其中 $a=\sqrt{\cfrac{T}{\rho}},f=\cfrac{F}{\rho}$， $\rho$ 为弦的线密度，$T$ 为弦的切应力，$F$ 表示单位长度受到的横向力。

杆的纵振动方程：未知函数 $u(x,t)$ 表示截面相对于平衡位置 $x$ 处的纵向位移
$$
u_{tt}-a^2u_{xx}=f(x,t)
$$
其中 $a=\sqrt{\cfrac{E}{\rho}},f=\cfrac{F}{\rho}$， $\rho$ 为杆的线密度，$E$ 为杆的Young模量，$F$ 表示单位长度受到的纵向力

更一般的，三维空间的**波动方程** (wave equation)是
$$
u_{tt}-a^2Δu=f(\mathbf r,t)\tag{1.2}
$$

(2) **热传导方程**：(heat equation)未知函数 $u(\mathbf r,t)$ 表示温度
$$
u_t-a^2 Δu=f(\mathbf r,t) \tag{1.3}
$$
其中 $\cfrac{\kappa}{\rho c},\cfrac{F}{\rho c}$，$\kappa$ 称为热传导系数， $\rho$ 是介质的密度，$c$ 是比热容，与介质的质料有关。 $F(x,y,z,t)$ 表示介质中有热源，单位时间内在单位体积介质中产生的热量

(3) 如果热传导方程热源不随时间变化，当温度达到稳定时 $(u_t=0)$ ，温度分布满足 **Poisson方程**
$$
Δu=-f/a^2\tag{1.4}
$$
如果没有热源 $(f=0)$ ，则有 **Laplace方程**（也称调和方程）
$$
Δu=0\tag{1.5}
$$
静电场的电势 $u(\mathbf r)$也满足 Poisson方程
$$
Δu=-\rho/\epsilon_0
$$
其中 $\rho$ 为电荷密度，$\epsilon_0$ 为真空介电常数。

(4) 如果波动方程 $u(\mathbf r,t)$ 随时间周期性变化，频率为 $\omega$，则 $u(\mathbf r,t)=v(\mathbf r)e^{-i\omega t}$，则 $v(\mathbf r)$ 满足 **Helmholtz 方程**
$$
Δv+k^2v=0\tag{1.6}
$$
其中 $k=\omega/a$ 称为波数。


**通解和特解**：如果多元函数 $u$ 具有偏微分方程中出现的各阶连续偏导数，并使方程 恒成立，则称此函数为方程的解，也称==古典解==。和常微分方程类似，称 $m$ 阶偏微分方程的含有 $m$ 个任意函数的解为==通解==。通解中的任意函数一旦确定便成为特解。

示例 1：求解偏微分方程 
$$
u_{xy}=0
$$
解：先关于 $y$ 积分，得 $u_x=f(x)$，再关于 $x$ 积分，就得到通解 
$$
u =\int f(ξ)\mathrm dξ+f_2(η)=f_1(ξ)+f_2(η)
$$
其中 $f_1,f_2$ 是任意函数。

示例 2：二维Laplace方程的通解
$$
u_{xx}+u_{yy}=0
$$
引入变换
$$
\begin{cases}
ξ=x+iy \\
η=x-iy
\end{cases}
$$
根据复合函数偏导法则，进一步求得
$$
\begin{cases}
u_{xx}=u_{ξξ}+2u_{ξη}+u_{ηη} \\
u_{yy}=-(u_{ξξ}-2u_{ξη}+u_{ηη})
\end{cases}
$$
原方程变为
$$
u_{ξη}=0
$$
于是可求得通解
$$
u(x,y)=f(x+iy)+g(x-iy)
$$

**定解条件**：通常把反应系统内部作用导出的偏微分方程称为==泛定方程==。为了完全描写一个实际物理问题的确定解，我们需要在一定的制约条件，称为==定解条件==。偏微分方程和相应的定解条件就构成一个==定解问题==。常见的定解条件有以下几类：

- **初始条件**：应该完全描写初始时刻（$t=0$）介质内部及边界上任意一点的状况。一般的讲，关于时间 $t$ 的 $m$ 阶偏微分方程，要给出 $m$ 个初始条件才能确定一个特解。
  对于波动方程来说，就是初始时刻的位移和速度
$$
u|_{t=0}=\phi(\mathbf r),\quad \cfrac{∂u}{∂t}|_{t=0}=\psi(\mathbf r)
$$
对于热传导方程，只用给出初始时刻物体温度的分布状态
$$
u|_{t=0}=\phi(\mathbf r)
$$
Laplace 方程和Possion 方程都是描述稳恒状态的，与时间无关，所以不需要初始条件。

- **边界条件**：边界条件形式比较多样化，要由具体问题中描述的具体状态决定。总的原则是：边界条件应该完全描写边界上各点在任意时刻的状况。
对于弦的横振动问题，约束条件通常有以下三种
1. 如果弦的两端固定，那么边界条件就是 
$$
u|_{x=0}=0,\quad u|_{x=l}=0
$$
2. 如果一端 $(x=0)$ 固定，另一端 $(x=l)$ 受位移方向的外力 $F(t)$ ，可以推导出
$$
u|_{x=0}=0,\quad \cfrac{∂u}{∂x}|_{x=l}=\cfrac{1}{T}F(t)
$$
如果一端固定，另一端外力为 0，即另一端自由，则 
$$
u|_{x=0}=0,\quad \cfrac{∂u}{∂x}|_{x=l}=0
$$
3. 如果外力是由弹簧提供的弹性力，即 $F(t)=-ku(l,t)$，其中 $k$ 为弹性系数，则 
$$
(\cfrac{∂u}{∂x}+\sigma u)|_{x=l}=0,\quad \sigma=\cfrac{k}{T}
$$

   对于热传导问题，也有类似的边界条件（以 $∂\Omega$ 表示区域 $\Omega$ 的边界）
1. 如果边界温度分布已知，则 
$$
u|_{\mathbf r\in∂\Omega}=f(\mathbf r,t)
$$
2.  如果物体边界和周围介质绝热，则 
$$
\cfrac{∂u}{∂n}|_{\mathbf r\in∂\Omega}=0
$$
3. 设周围介质的温度为 $u_1(\mathbf r,t)$ ，物体通过边界与周围有热量交换，则 
$$
(\cfrac{∂u}{∂n}+\sigma u)|_{\mathbf r\in∂\Omega}=\sigma u_1
$$
其中 $\sigma=h/k,h$表示两种物质间的热交换系数。

   常见的边界条件数学上分为三类（以 $∂\Omega$ 表示区域 $\Omega$ 的边界，$f$ 为已知函数）
1. 第一类边界条件 (Dircichlet条件)：直接给出边界上各点未知函数 $u$ 的值
$$
u|_{∂\Omega}=f(\mathbf r,t)
$$
2. 第二类边界条件 (Neu-mann条件)：给出边界外法线方向上方向导数的数值
$$
\cfrac{∂u}{∂n}|_{∂\Omega}=f(\mathbf r,t)
$$
3. 第三类边界条件 (Robin条件)：给出边界上各点的函数值与外法线方向上方向导数的某一线性组合值
$$
(\cfrac{∂u}{∂n}+\sigma u)|_{∂\Omega}=f(\mathbf r,t)
$$

- **连接条件**：从微分方程的推导知道，方程只在空间区域的内部成立，如果在区域内部出现结构上的跃变，那么在这些跃变点（线，面）上微分方程不再成立，因此需要补充上相应的条件，通常称为==连接条件==。
  例如，由两段不同材质组成的弦在 $x_0$ 处连接，波动方程需分段讨论，且在连接处位移和张力相等，连接条件为 $u_1(x_0,t)=u_2(x_0,t),\cfrac{∂u_1}{∂x}|_{x=x_0-0}=\cfrac{∂u_2}{∂x}|_{x=x_0+0}$

**定解问题的适定性**：定解问题来自于实际，它的解也应该回到实际中去。如果一个定解问题的解存在、唯一且稳定，则称该定解问题是==适定的== (well-posed) 。稳定性指的是，如果定解条件的数值有细微的改变，解的数值也作细微的改变。

## 线性叠加原理

**线性叠加原理**：考虑 $n$ 个自变量（包括时间 $t$） $x_1,x_2,\cdots,x_n$ 的二阶线性偏微分方程
$$
\displaystyle\sum_{j=1}^{n}\sum_{i=1}^n a_{ij}u_{x_ix_j}+\sum_{i=1}^{n}b_iu_{x_i}+cu=f
$$
其中 $a_{ij},b_i,c,f$ 是 $x_1,x_2,\cdots,x_n$ 的已知函数，$f$ 称为方程的非齐次项。

引入微分算子
$$
L=\sum_{j=1}^{n}\sum_{i=1}^n a_{ij}\cfrac{∂}{∂x_i∂x_j}
+\sum_{i=1}^{n}b_i\cfrac{∂}{∂x_i}+c
$$
则可简单的表示为
$$
L[u]=f
$$
一般的，对任意常数 $c_1,c_2$ 和任意函数 $u_1,u_2$ ，微分算子满足性质
$$
L[c_1u_1+c_2u_2]=c_1L[u_1]+c_2L[u_2]\tag{2.1}
$$
称为线性微分算子。显然，$L$ 为线性微分算子。
对于一般的线性边界条件（包括三类边界条件）也可以写成算子的形式
$$
L_0[u]=(\alpha\cfrac{∂u}{∂n}+\beta u)|_{∂\Omega}=\phi
$$
其中 $\alpha,\beta,\phi$ 是已知函数，容易证明 $L_0$ 也是线性微分算子。

**叠加原理**
(1) 有限叠加原理：若 $u_i$ 分别满足方程 $L[u_i]=f_i\quad(i=1,2,\cdots,m)$ ，则他们的线性组合 $u=\displaystyle\sum_{i=1}^mc_iu_i$ 满足方程 
$$
L[u]=\displaystyle\sum_{i=1}^mc_if_i\tag{2.2}
$$
(2) 级数叠加原理：若 $u_i$ 分别满足方程 $L[u_i]=f_i\quad(i=1,2,\cdots)$ ，则他们的级数 $u=\displaystyle\sum_{i=1}^\infty c_iu_i$ 满足方程 
$$
L[u]=\displaystyle\sum_{i=1}^\infty c_if_i\tag{2.3}
$$
(3) 积分叠加原理：若 $u(\mathbf{r;r_0})$ 满足方程 $L[u]=f(\mathbf{r;r_0})$ ，则积分 $U(\mathbf r)=\displaystyle\int_V c(\mathbf r_0)u(\mathbf{r;r_0})\mathrm d\mathbf r_0$ 满足方程
$$
L[U]=\int_Vc(\mathbf r_0)f(\mathbf{r;r_0})\mathrm d\mathbf r_0\tag{2.4}
$$

## 一阶(拟)线性偏微分方程

**引例**：求解右行单波方程
$$
u_{t}+au_{x}=0
$$
的通解，引入变换
$$
\begin{cases}
ξ=x-at \\
η=x
\end{cases}
$$
根据复合函数偏导法则，进一步求得
$$
\begin{cases}
u_{t}=-au_{ξ}\\
u_{x}=u_{ξ}+u_{η}
\end{cases}
$$
原方程变为
$$
u_{η}=0
$$
于是可求得通解
$$
u(x,t)=g(x-at)
$$

**一阶线性偏微分方程**

$n$ 个自变量的==一阶线性偏微分方程==的一般形式为
$$
\sum_{i=1}^{n}b_i\cfrac{∂u}{∂x_i}+cu=f
$$
其中 $b_i,c,f$ 是自变量 $x_1,x_2,\cdots,x_n$ 的函数。

先研究两自变量 $x,y$ 的一阶线性偏微分方程
$$
au_x+bu_y+cu+f=0\tag{2.1}
$$
其中 $a,b,c,f$ 都是自变量 $x,y$ 的函数。
(1) 若 $a\equiv0,b\neq0$ 方程改写为
$$
u_y+\cfrac{c}{b}u+\cfrac{f}{b}=0
$$
利用一阶线性常微分方程的求解方法可求得通解
$$
u=e^{-p(x,y)}[\int e^{p(x,y)}\cfrac{f(x,y)}{b(x,y)} \mathrm dy+g(x)]
$$
其中 $p(x,y)=\displaystyle\int\cfrac{c(x,y)}{b(x,y)} \mathrm dy,g(x)$ 是任意函数。
(2) 若 $a(x,y)b(x,y)\neq0$ 不能直接积分求解。我们希望通过适当的自变量变换和未知函数的变换，使方程简化，并在此基础上对方程求解。
首先作自变量非奇异变换（可逆）
$$
\begin{cases}
ξ=ξ(x,y) \\
η=η(x,y)
\end{cases}
$$
要求雅可比行列式 
$$
|J(ξ,η)|=|\cfrac{∂(ξ,η)}{∂(x,y)}|=\begin{vmatrix}ξ_x&ξ_y \\ η_x&η_y\end{vmatrix}\neq0
$$
以保证新变量 $ξ,η$ 相互独立，利用链式法则
$$
u_x=\cfrac{∂u}{∂ξ}\cfrac{∂ξ}{∂x}+\cfrac{∂u}{∂η}\cfrac{∂η}{∂x},\quad 
u_y=\cfrac{∂u}{∂ξ}\cfrac{∂ξ}{∂y}+\cfrac{∂u}{∂η}\cfrac{∂η}{∂y}
$$
可得到新的方程
$$
Au_ξ+Bu_η+Cu+F=0\tag{3.2} 
$$
其中
$$
\begin{cases}
A=aξ_x+bξ_y \\
B=aη_x+bη_y \\
C=c,F=f
\end{cases} \tag{3.3}
$$
新方程仍然是线性的。从上式看出，如果取 $ξ=ξ(x,y)$ 是一阶齐次线性偏微分方程
$$
a\cfrac{∂z}{∂x}+b\cfrac{∂z}{∂y}=0\tag{3.4}
$$
的解，则此时 $A=0$，这样方程 (3.2) 得以化简为
$$
Bu_η+Cu+F=0
$$
从而可以像第一种类型一样积分求解。
(3) 偏微分方程 (3.4) 可以做如下几何解释，方程可以改写为向量形式
$$
(a,b)\cdot(\cfrac{∂z}{∂x},\cfrac{∂z}{∂y})=0\tag{3.5}
$$
方程的解 $z=\phi(x,y)$ 表示空间 $xyz$ 中一张曲面 $S:z=\phi(x,y)$ ，任取平面 $z=c$ 截得的曲线方程为 $L:\phi(x,y)=c$ ，如图
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/characteristic-equation.png)
曲线 $L$ 上任意一点 $P(x,y)$ 的法线方向为 $(\cfrac{∂\phi}{∂x},\cfrac{∂\phi}{∂y})$ ，式 (3.5) 表明在 $P$ 点的向量 $(a,b)$ 与法线方向垂直，即切线方向。$P$ 点切向量微元可以表示为 $(\mathrm dx,\mathrm dy)$ ，所以存在对应关系
$$
\cfrac{\mathrm dx}{a(x,y)}=\cfrac{\mathrm dy}{b(x,y)}\tag{3.6}
$$
由于平面 $z=c$ 的任意性，所截得的 $L$ 形成的曲线簇覆盖整个曲面 $S$。所以，只要求出常微分方程 (3.6) 的积分曲线簇 $\phi(x,y)=c$ ，便可知道偏微分方程 (3.4) 对应的解 $z=\phi(x,y)$ 。反之，偏微分方程 (3.4) 的解也是常微分方程 (3.6) 的积分曲线簇。
证明：假设常微分方程 (3.6) 隐式通解为 $\phi(x,y)=c$ ，则
$$
\mathrm d\phi(x,y)=\cfrac{∂\phi}{∂x}\mathrm dx+\cfrac{∂\phi}{∂y}\mathrm dy=0
$$
将常微分方程改写成 $\cfrac{\mathrm dy}{\mathrm dx}=\cfrac{b(x,y)}{a(x,y)}$ 带入上式可得
$$
a\cfrac{∂\phi}{∂x}+b\cfrac{∂\phi}{∂y}=0
$$
可知 $z=\phi(x,y)$ 是偏微分方程 (3.4) 的一个解。反向推导亦成立。

(4) 常微分方程 (3.6) 叫做偏微分方程 (3.1) 的==特征方程==(characteristic equation)，特征方程的积分曲线叫做==特征线==。如果求出了积分曲线簇 $\phi(x,y)=c$ ，再任取常数 $c$ 使得 $|J(ξ,η)|\neq0$ ，以此 $ξ,η$ 作为变量代换，则一阶线性偏微分方程便可求得通解。


**一阶拟线性偏微分方程**

$n$ 个自变量的==一阶拟线性偏微分方程==的一般形式为
$$
\sum_{i=1}^{n}b_i(x_1,x_2,\cdots,x_n,u)\cfrac{∂u}{∂x_i}=c(x_1,x_2,\cdots,x_n,u)
$$

它的求解归结到 $n+1$ 维一阶线性偏微分方程
$$
\sum_{i=1}^{n}b_i(x_1,x_2,\cdots,x_n,u)\cfrac{∂\phi}{∂x_i}+c(x_1,x_2,\cdots,x_n,u)\cfrac{∂\phi}{∂u}=0
$$

以两自变量 $x,y$ 的一阶拟线性偏微分方程为例
$$
au_x+bu_y=c\tag{3.7}
$$
其中 $a,b,c$ 都是变量 $x,y,u$ 的函数，$u=u(x,y)$ ，且系数 $a,b$ 不同时为零。相应的一阶线性偏微分方程为
$$
a\cfrac{∂\phi}{∂x}+b\cfrac{∂\phi}{∂y}+c\cfrac{∂\phi}{∂u}=0\tag{3.8}
$$
若 $\phi(x,y,u)$ 是方程 (3.8) 的解，则 $\phi(x,y,u)=0$ 是方程 (3.7) 的隐式解。
证明：$\phi(x,y,u)=0$ 给出了隐函数 $\phi(x,y,u(x,y))=0$ 分别对 $x,y$ 求偏导
$$
\cfrac{∂\phi}{∂x}+\cfrac{∂\phi}{∂u}\cfrac{∂u}{∂x}=0,\quad \cfrac{∂\phi}{∂y}+\cfrac{∂\phi}{∂u}\cfrac{∂u}{∂y}=0
$$
从中解出 $\cfrac{∂\phi}{∂x}=-\cfrac{∂\phi}{∂u}\cfrac{∂u}{∂x},\quad  \cfrac{∂\phi}{∂y}=-\cfrac{∂\phi}{∂u}\cfrac{∂u}{∂y}$
带入 (3.8) 可得到方程 (3.7) ，即$\phi(x,y,u)=0$ 是方程 (3.7) 的隐式解。反之亦成立。
一阶线性偏微分方程 (3.8) 的通解可由常微分方程组
$$
\cfrac{\mathrm dx}{a(x,y,u)}=\cfrac{\mathrm dy}{b(x,y,u)}=\cfrac{\mathrm du}{c(x,y,u)}\tag{3.9}
$$
的首次积分确定。上式称为方程 (3.7) 的==完全特征方程组==，它的积分曲线称为==完全特征曲线==。

## 二阶线性偏微分方程的分类和标准式

以下研究方程的分类，并把各类方程分别化为标准型，这样以后就只需讨论标准型的解法。

**两个自变量方程的分类**

先研究两自变量 $x,y$ 的二阶线性偏微分方程
$$
a_{11}u_{xx}+a_{12}u_{xy}+a_{22}u_{yy}+b_1u_x+b_2u_y+cu+f=0\tag{4.1}
$$
其中 $a_{11},a_{12},a_{22},b_1,b_2,c,f$ 都是自变量 $x,y$ 的函数。我们希望通过适当的自变量变换和未知函数的变换，使方程简化，并在此基础上对方程进行分类。

作自变量非奇异变换（可逆）
$$
\begin{cases}
ξ=ξ(x,y) \\
η=η(x,y)
\end{cases}
$$
其中雅可比行列式 
$$
|J|=|\cfrac{∂(ξ,η)}{∂(x,y)}|=\begin{vmatrix}ξ_x&ξ_y \\ η_x&η_y\end{vmatrix}\neq0
$$
通过代换可得到新的方程
$$
A_{11}u_{ξξ}+2A_{12}u_{ξη}+A_{22}u_{ηη}+B_1u_ξ+B_2u_η+Cu+F=0\tag{4.2}
$$
其中
$$
\begin{cases}
A_{11}=a_{11}ξ_x^2+2a_{12}ξ_xξ_y+a_{22}ξ_y^2 \\
A_{12}=a_{11}ξ_xη_x+a_{12}(ξ_xη_y+ξ_yη_x)+a_{22}ξ_yη_y \\
A_{22}=a_{11}η_x^2+2a_{12}η_xη_y+a_{22}η_y^2 \\
B_1=a_{11}ξ_{xx}+2a_{12}ξ_{xy}+a_{22}ξ_{yy}+b_1ξ_x+b_2ξ_y \\
B_2=a_{11}η_{xx}+2a_{12}η_{xy}+a_{22}η_{yy}+b_1η_x+b_2η_y \\
C=c,F=f
\end{cases} \tag{4.3}
$$
新方程仍然是线性的。
系数 $A_{11},A_{12},A_{22}$ 可用矩阵表示为
$$
\mathbf D=
\begin{pmatrix}A_{11}&A_{12} \\ A_{21}&A_{22}\end{pmatrix}=
\begin{pmatrix}ξ_x&ξ_y \\ η_x&η_y\end{pmatrix}
\begin{pmatrix}a_{11}&a_{12} \\ a_{21}&a_{22}\end{pmatrix}
\begin{pmatrix}ξ_x&ξ_y \\ η_x&η_y\end{pmatrix}^{T} \tag{4.4}
$$
其中 $A_{12}=A_{21},a_{12}=a_{21}$

由 (4.4) 式看出，如果取一阶偏微分方程
$$
a_{11}z_x^2+2a_{12}z_xz_y+a_{22}z_y^2=0
$$
两个线性无关的特解作为新的自变量 $ξ,η$ ，此时 $A_{11}=A_{22}=0$，这样方程 (4.2) 得以化简。
上述一阶偏微分方程可变为常微分方程求解，首先修改为
$$
a_{11}(-\cfrac{z_x}{z_y})^2-2a_{12}(-\cfrac{z_x}{z_y})+a_{22}=0
$$
如果把 $z(x,y)=\text{const}$ 作为定义隐函数 $y=y(x)$ 的方程，则 $dy/dx=-z_x/z_y$，从而
$$
a_{11}(\cfrac{dy}{dx})^2-2a_{12}(\cfrac{dy}{dx})+a_{22}=0\tag{4.5}
$$
常微分方程 (4.5) 叫做偏微分方程 (4.1) 的==特征方程==，特征方程的积分曲线叫做==特征线==。
特征方程可分为两个方程
$$
\cfrac{dy}{dx}=\cfrac{a_{12} \pm \sqrt{a_{12}^2-a_{11}a_{22}}}{a_{11}}\tag{4.6}
$$
通常根据判别式 $Δ=a_{12}^2-a_{11}a_{22}$ 划分方程 (4.1) 的类型
(1) $Δ>0$ 方程是双曲型；
(2) $Δ=0$ 方程是抛物型；
(3) $Δ>0$ 方程是椭圆型

由 (4.4) 矩阵的性质得到
$$
A_{12}^2-A_{11}A_{22}=\det(\mathbf D)=|J|^2(a_{12}^2-a_{11}a_{22})
$$
这就是说，作自变量变换时，方程的类型不变。

下面就方程中 $Δ=a_{12}^2-a_{11}a_{22}$ 符号的不同情况，讨论如何把方程 (4.1) 化为标准型。

- **双曲型方程**：方程 (4.6) 给出两族实的特征线 $ξ(x,y)=c_1,η(x,y)=c_2$ 
  取 $ξ=ξ(x,y),η=η(x,y)$ 作为新的自变量，则 $A_{11}=A_{22}=0$，从而自变量代换后的新方程 (4.2) 成为

$$
u_{ξη}=-\cfrac{1}{2A_{12}}(B_1u_ξ+B_2u_η+Cu+F)
$$

  或者，再做自变量变换 $ξ=α+β,η=α-β$ ，则方程化为
$$
u_{αα}-u_{ββ}=-\cfrac{1}{A_{12}}[(B_1+B_2)u_α+(B_1-B_2)u_β+2Cu+2F]
$$
  上述两方程是双曲型方程的标准形式，弦振动方程即为标准型的双曲方程。

- **抛物型方程**：特征方程只能给出一族特征线 $ξ(x,y)=c$
  取 $ξ=ξ(x,y)$ 作为新的自变量，任取一函数 $η=η(x,y)$，与 $ξ$ 线性无关。由于 $a_{12}^2-a_{11}a_{22}=0$，带入 (4.3) 可得
  $A_{11}=a_{11}ξ_x^2+2a_{12}ξ_xξ_y+a_{22}ξ_y^2=(\sqrt{a_{11}}ξ_x+\sqrt{a_{22}}ξ_y)^2=0 \\
  A_{12}=a_{11}ξ_xη_x+a_{12}(ξ_xη_y+ξ_yη_x)+a_{22}ξ_yη_y=(\sqrt{a_{11}}ξ_x+\sqrt{a_{22}}ξ_y)(\sqrt{a_{11}}η_x+\sqrt{a_{22}}η_y)=0$
  从而自变量代换后的新方程成为

$$
u_{ηη}=-\cfrac{1}{2A_{22}}(B_1u_ξ+B_2u_η+Cu+F)
$$

  这是抛物型方程的标准形式，扩散方程、热传导方程都是标准形式的抛物型方程。

- **椭圆形方程**：特征方程给出两族复数的特征线 $ξ(x,y)=c_1,η(x,y)=c_2$ 
  而且 $ξ=ξ(x,y),η=η(x,y)$ 共轭，对 $ξ,η$ 再做自变量变换 $ξ=α+iβ,η=α-iβ$ ，则新方程化为

$$
u_{αα}+u_{ββ}=-\cfrac{1}{A_{12}}[(B_1+B_2)u_α+i(B_2-B_1)u_β+2Cu+2F]
$$

  这是椭圆型方程的标准形式，稳定温度分别、静电场方程都是椭圆型的双曲方程。

**多自变量方程的分类**

 考虑多自变量 $x_1,x_2,\cdots,x_n$ 的二阶线性偏微分方程
$$
\displaystyle \sum_{j=1}^{n}\sum_{i=1}^n a_{ij}u_{x_ix_j}+\sum_{i=1}^{n}b_iu_{x_i}+cu+f=0 \tag{4.7}
$$
做自变量的非奇异代换
$$
ξ_k=ξ_k(x_1,x_2,\cdots,x_n),\quad k=1,2,\cdots,n
$$
代换的雅克比行列式
$$
|J|=|\cfrac{∂(ξ_1,ξ_2,\cdots,ξ_n)}{∂(x_1,x_2,\cdots,x_n)}|\neq0
$$
通过代换可得到新的方程
$$
\displaystyle \sum_{j=1}^{n}\sum_{i=1}^n A_{ij}u_{ξ_iξ_j}+\sum_{i=1}^{n}B_iu_{ξ_i}+Cu+F=0 \tag{4.8}
$$
注意到变换后的方程仍然是线性的，其中系数
$$
\begin{cases}
A_{ij}=\displaystyle \sum_{l=1}^{n}\sum_{k=1}^na_{kl}\cfrac{∂ξ_i}{∂x_k}\cfrac{∂ξ_j}{∂x_l} \\
\displaystyle B_i=\sum_{k=1}^n\cfrac{∂ξ_i}{∂x_k}+\sum_{l=1}^{n}\sum_{k=1}^n\cfrac{∂^2ξ_i}{∂x_k∂x_l}
\end{cases}
$$
系数 $A_{ij}$ 可用矩阵表示为
$$
(A_{ij})_{n\times n}=JAJ^T
$$
其中 $A=(a_{ij})_{n\times n}$ 是由方程 (4.7) 二阶偏导数的系数组成。值得注意的是系数变换公式恰恰是二次齐次式 $\displaystyle \sum_{j=1}^{n}\sum_{i=1}^n a_{ij}λ_iλ_j$ 系数的非奇异线性变换，二次齐次式可以通过适当的变换而对角化，即
$$
(A_{ij})_{n\times n}=BAB^T=\mathrm{diag}(A_{11},A_{22},\cdots,A_{nn})
$$
其中 $A_{ii}\in\{-1,0,1\}$ ，根据二次齐次式对角化的惯性定律，$A_{ii}$ 为 $-1,0,1$ 的个数是一定的，设 $A_{ii}$ 中 1 的个数为 $p$ ，-1 的个数为 $q$ 。据此划分偏微分方程的类型：
(1) 当 $p=n,q=0$，椭圆型(elliptic)，标准形式是 
$$
\displaystyle\sum_{i=1}^n\cfrac{∂^2u}{∂ξ_i^2}+\sum_{i=1}^{n}B_i\cfrac{∂u}{∂ξ_i}+Cu+F=0
$$
(2) 当 $p=n-1,q=0$，抛物型(parabolic)，标准形式是 
$$
\displaystyle\sum_{i=1}^{n-1}\cfrac{∂^2u}{∂ξ_i^2}+\sum_{i=1}^{n}B_i\cfrac{∂u}{∂ξ_i}+Cu+F=0
$$
(3) 当 $p=1,q=n-1$，双曲型(hyperbolic)，标准形式是 
$$
\displaystyle\cfrac{∂^2u}{∂ξ_1^2}-\sum_{j=2}^n\cfrac{∂^2u}{∂ξ_j^2}+\sum_{i=1}^{n}B_i\cfrac{∂u}{∂ξ_i}+Cu+F=0
$$
(4) 当 $p>0,q>0$，超双曲型(ultrahyperbolic)，标准形式是 
$$
\displaystyle\sum_{i=1}^p\cfrac{∂^2u}{∂ξ_i^2}-\sum_{j=1}^{p+q}\cfrac{∂^2u}{∂ξ_i^2}+\sum_{i=1}^{n}B_i\cfrac{∂u}{∂ξ_i}+Cu+F=0
$$
**常系数二阶线性偏微分方程**：方程 (2.12) 中的系数 $a_{ij},b_i,c,f$ 都是常数，则方程还可以进一步化简。引入变换
$$
u(ξ,η)=V(ξ,η)e^{λξ+μη}
$$
其中 $λ,μ$ 是待定常数，根据方程的类型可选择 $λ,μ$，使一阶偏导数项或者函数项的系数为零。

## 波动方程的行波解

**无限长弦振动初值问题**：考虑两端为无限长的弦振动方程初值问题
$$
\begin{cases}
u_{tt}-a^2u_{xx}=0 \\
u|_{t=0}=\phi(x),\cfrac{∂u}{∂t}|_{t=0}=\psi(x)
\end{cases}\tag{5.1}
$$
其中 $\phi(x),\psi(x)$ 分别表示初始位移和初始速度。
(1) 弦振动方程的通解：当 $a>0$ 为常数时，上述方程可分解为
$$
(\cfrac{∂}{∂t}+a\cfrac{∂u}{∂x})(\cfrac{∂}{∂t}-a\cfrac{∂}{∂x})u=0
$$
等价于两个一阶线性偏微分方程组
$$
\begin{cases}
\cfrac{∂u}{∂t}+a\cfrac{∂u}{∂x}=0 \\
\cfrac{∂u}{∂t}-a\cfrac{∂u}{∂x}=v
\end{cases}
$$
这两个方程有一组独立的特征线 $x+at=c_1,x-at=c_2$，引入变换 $ξ=x+at,η=x-at$
利用复合函数的求导法则求得 $u_{tt},u_{xx}$ 带入弦振动方程，得到新的方程
$$
u_{ξη}=0
$$
上述方程容易求得通解 
$$
u=f_1(x+at)+f_2(x-at)\tag{5.2}
$$
其中 $f_1,f_2$ 是任意函数。

通解具有明确的物理意义，以 $f_2(x-at)$ 而论，改用以速度 $a$ 沿 $x$ 轴正方向移动的坐标轴 $X$，则新旧坐标和时间之间的关系为 $X=x-at,T=t$，而 $f_2(x-at)=f(X)$ 与时间 $T$ 无关。这就是说函数图像在动坐标系中保持不变，即是随着动坐标系以速度 $a$ 沿 $x$ 轴正方向移动的行波。同理， $f_1(x+at)$ 是以速度 $a$ 沿 $x$ 轴负方向移动的行波。这样，弦振动方程描述的是以速度 $a$ 沿 $x$ 轴正负两方向移动的行波。

(2) 函数 $f_1,f_2$ 由初始条件确定
将通解带入初始条件可以得到 $\begin{cases}
f_1(x)+f_2(x)=\phi(x) \\af'_1(x)-af'_2(x)=\psi(x)
\end{cases}$ ，由此可解得 
$$
f_1(x)=\frac{1}{2}\phi(x)+\frac{1}{2a}\int_{x_0}^x\psi(ξ)dξ+\frac{1}{2}[f_1(x_0)-f_2(x_0)] \\
f_2(x)=\frac{1}{2}\phi(x)-\frac{1}{2a}\int_{x_0}^x\psi(ξ)dξ-\frac{1}{2}[f_1(x_0)-f_2(x_0)]
$$
将此代回 (5.2) 可得，初始问题的解
$$
u(x,t)=\frac{1}{2}[\phi(x+at)+\phi(x-at)]+\frac{1}{2a}\int_{x-at}^{x+at}\psi(ξ)dξ\tag{5.3}
$$
这个公式称为==达朗贝尔(d’Alembert)公式==。

(3) 物理意义示例：设初速度为零 $\psi(x)=0$，初始位移 $\phi(x)$ 
如下图 $t=0$ 时刻函数图像实线所示，达朗贝尔公式给出 $u(x,t)=\frac{1}{2}\phi(x+at)+\frac{1}{2}\phi(x-at)$ ，即初始位移分为两半（下图虚线），分别以速度 $a$ 向左右两端移动，这两个行波的和给出各个时刻的波形（由下向上的实线）。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/dAlembert-demo.png" width="50%;" />

**半无限弦振动定解问题** ——延拓法
(1) 端点固定弦的自由振动：定解问题为
$$
\begin{cases}
u_{tt}-a^2u_{xx}=0 \quad (x>0)\\
u|_{t=0}=\phi(x),\cfrac{∂u}{∂t}|_{t=0}=\psi(x) \quad(x\geqslant 0)\\
u|_{x=0}=0
\end{cases}\tag{5.4}
$$
求解上述问题的基本思路是把半无限长当做无限长弦的 $x\geqslant0$ 的部分，然后用达朗贝尔公式求解即可。由定解条件知，振动过程 $x=0$ 处必须保持不动，由微积分知识可知，如果一个连续可微函数 $g(x)$ 在$(-\infty,+\infty)$ 上是奇函数，则必有 $g(0)=0$ 。因此对函数 $\phi,\psi$ 进行奇延拓，即
$$
\Phi(x)=\begin{cases}\phi(x)&(x\geqslant0)\\ -\phi(x)&(x<0)\end{cases};\quad
\Psi(x)=\begin{cases}\psi(x)&(x\geqslant0)\\ -\psi(x)&(x<0)\end{cases}
$$
将上式带入达朗贝尔公式，可求得
$$
u(x,t)=\begin{cases}\displaystyle
\frac{1}{2}[\phi(x+at)+\phi(x-at)]+\frac{1}{2a}\int_{x-at}^{x+at}\psi(ξ)dξ &(t⩽\cfrac{x}{a})\\
\displaystyle
\frac{1}{2}[\phi(x+at)-\phi(x-at)]+\frac{1}{2a}\int_{at-x}^{x+at}\psi(ξ)dξ &(t>\cfrac{x}{a})
\end{cases}\tag{5.5}
$$
物理意义示例：设初速度为零 $\psi(x)=0$，初始位移 $\phi(x)$ 
图中由下向上的实线描述了波形变化，右边的波分别向左右两端移动，左端奇延拓的波也分别向左右两端移动。端点固定的影响表现为反射波，反射波的相位和入射波相反，这就是==半波损失==。
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/half-wave-loss-demo.png" width="45%;" />

(2) 端点自由弦的自由振动：定解问题为
$$
\begin{cases}
u_{tt}-a^2u_{xx}=0 \\
u|_{t=0}=\phi(x),\cfrac{∂u}{∂t}|_{t=0}=\psi(x) \\
\cfrac{∂u}{∂x}|_{x=0}=0
\end{cases}\quad(0\leqslant x<\infty)\tag{5.6}
$$
同样对函数进行偶延拓
$$
\Phi(x)=\begin{cases}\phi(x)&(x\geqslant0)\\ \phi(-x)&(x<0)\end{cases};\quad
\Psi(x)=\begin{cases}\psi(x)&(x\geqslant0)\\ \psi(-x)&(x<0)\end{cases}
$$
将上式带入达朗贝尔公式，可求得
$$
u(x,t)=\begin{cases}\displaystyle
\frac{1}{2}[\phi(x+at)+\phi(x-at)]+\frac{1}{2a}\int_{x-at}^{x+at}\psi(ξ)dξ &(t⩽\cfrac{x}{a})\\
\displaystyle
\frac{1}{2}[\phi(x+at)+\phi(at-x)]+\frac{1}{2a}[\int_{0}^{x+at}\psi(ξ)dξ+\int_{0}^{at-x}\psi(ξ)dξ] &(t>\cfrac{x}{a})
\end{cases}\tag{5.7}
$$
自由端点的影响也是一种反射波，不同的是反射波的相位和入射波相同，没有半波损失。

**三维波动方程的球面波**：三维波动方程在求坐标系 $(r,θ,ϕ)$ 下可以表示为
$$
\cfrac{1}{a^2}\cfrac{∂^2u}{∂t^2}=\cfrac{1}{r^2}\cfrac{∂}{∂r}\left(r^2\cfrac{∂u}{∂r}\right)
+\cfrac{1}{r^2\sinθ}\cfrac{∂}{∂θ}\left(\sinθ\cfrac{∂u}{∂θ}\right)
+\cfrac{1}{r^2\sin^2θ}\cfrac{∂^2u}{∂ϕ^2}\tag{5.8}
$$
若函数 $u$ 是球对称函数，指 $u$ 与 $θ,ϕ$ 无关，即 $u=u(r,t)$，方程可化为
$$
\cfrac{∂^2u}{∂t^2}=a^2(\cfrac{∂^2u}{∂r^2}+\cfrac{2}{r}\cfrac{∂u}{∂r})
$$
进一步重写为
$$
\cfrac{∂^2(ru)}{∂t^2}=a^2\cfrac{∂^2(ru)}{∂r^2}
$$
这时关于 $ru$ 的一维波动方程，其通解为
$$
ru(r,t)=f_1(r+at)+f_2(r-at)
$$
从而
$$
u=\cfrac{f_1(r+at)+f_2(r-at)}{r}\tag{5.9}
$$
其中 $f_1,f_2$ 是任意两个二阶连续可微函数，可见，在球对称下，三维波的传播是以球心为中心，沿半径 $r$ 传播的球面波。


# 分离变量法

## 齐次边界条件下齐次方程的定解问题

**两端固定弦的自由振动**：（第一类边界条件）设长度为 $l$，两端固定弦的定解问题为
$$
\begin{cases}
u_{tt}-a^2u_{xx}=0 & x\in(0,l)\\
u|_{t=0}=\phi(x),\cfrac{∂u}{∂t}|_{t=0}=\psi(x) & x\in[0,l]\\
u|_{x=0}=0,u|_{x=l}=0
\end{cases}\tag{1.1}
$$
**(1) 分离变量**：上节半无限弦的振动问题启发我们是否可以用驻波研究两端固定弦的振动问题，驻波的表达式为
$$
u(x,t)=X(x)T(t)\tag{1.2}
$$
带入弦振动方程可得
$$
XT''-a^2X''T=0
$$
可化为
$$
\cfrac{T''}{a^2T}=\cfrac{X''}{X}
$$
上式中左端只是 $t$ 的函数，右端只是 $x$ 的函数，要想两端相等，则必等于同一常数，记作 $-\lambda$ 。于是上式化为两个常微分方程
$$
T''(t)+\lambda a^2T=0 \\
X''(x)+\lambda X=0
$$
同样，表达式 (1.2) 带入边界条件可得
$$
\begin{cases}
X(0)T(t)=0 \\
X(l)T(t)=0
\end{cases}
$$
这时必须有 $X(0)=X(l)=0$ ，因为 $T(t)=0$ ，则 $u(x,t)$ 恒为零，失去意义。
**(2) 本征值问题**：先求解关于 $X$ 的边值问题
$$
\begin{cases}
X''(x)+\lambda X(x)=0 \\
X(0)=0,X(l)=0
\end{cases}\tag{1.3}
$$
其中 $\lambda$ 为待定常数，这个问题称为==本征值问题==，使问题有非零解的 $\lambda$ 值称为==本征值==，对应的函数称为==本征函数==[^1]。分三种情形讨论

1. 若 $\lambda<0$ ，方程的通解为
$$
   X=c_1e^{\sqrt{-λ}x}+c_2e^{-\sqrt{-λ}x}
$$
   积分常数 $c_1,c_2$ 由边界条件解出 $c_1=c_2=0$ ，因此 $X(x)\equiv0$，所以驻波 $u(x,t)=X(x)T(t)\equiv0$，没有意义。

2. 若 $\lambda=0$ ，方程的通解为
$$
   X=c_1x+c_2
$$
   积分常数 $c_1,c_2$ 由边界条件解出 $c_1=c_2=0$ ，因此 $X(x)\equiv0$，所以驻波依然没有意义。

3. 若 $\lambda>0$ ，方程的通解为
$$
   X=c_1\cos\sqrt{λ}x+c_2\sin\sqrt{λ}x
$$
   积分常数 $c_1,c_2$ 由边界条件确定，$c_1=0,c_2\sin\sqrt{λ}l=0$ ，为了使 $X$ 不恒为零，所以 $\sin\sqrt{λ}l=0$ ，于是本征值
$$
   λ_n=(\cfrac{n\pi}{l})^2,\quad n=1,2,\cdots\tag{1.4}
$$
   对应的本征函数
$$
   X_n=C_n\sin\cfrac{n\pi x}{l}\tag{1.5}
$$
   其中 $C_n$ 为任意常数。 然后再用本征值求解对应的 $T(t)$ 
$$
   T''(t)+\lambda_n a^2T=0
$$
   其通解为
$$
   T_n=A_n\cos\cfrac{n\pi at}{l}+B_n\sin\cfrac{n\pi at}{l}\tag{1.6}
$$
从而得到一组分离变量形式的特解
$$
\begin{aligned}
u_n(x,t) & =X_n(x)T_n(t) \\
& =(a_n\cos\cfrac{n\pi at}{l}+b_n\sin\cfrac{n\pi at}{l})\sin\cfrac{n\pi x}{l} \\
&= A\sin(k_nx)\sin(ω_nt+θ_n)
\end{aligned}\tag{1.7}
$$
   其中 $a_n=C_nA_n,b_n=C_nB_n$ 为任意常数。这就是两端固定弦上可能的驻波，其中每一个 $n$ 对应一种驻波，这些驻波叫做==本征振动==。

   物理意义：本征振动中 $A\sin k_nx$ 表示弦上点的振幅分布，其中 $A=\sqrt{a_n^2+b_n^2},k_n=\cfrac{n\pi}{l}$ 
$\sin(ω_nt+θ_n)$ 表示相位因子，其中 $ω_n=\cfrac{n\pi a}{l},θ_n=\arcsin\cfrac{a_n}{A}$
$ω_n$ 表示驻波的圆频率，与初始条件无关
$k_n$ 称为波数，$θ_n$ 是初相位，由初始条件决定。
在 $x=kl/n\quad(k=0,1,2,\cdots,n)$ 共计 $n+1$ 个点上，$\sin(n\pi x/l)=\sin k\pi=0$ ，从而 $u_n(x,t)=0$ ，这些点就是驻波的节点。

**(3) 特解的叠加**：以上本征振动是满足弦振动方程和边界条件线性无关的特解，他们不一定满足初始条件。为此，根据叠加原理，对特解线性叠加
$$
\displaystyle u(x,t)=\sum_{n=1}^{\infty}u_n(x,t)=\sum_{n=1}^{\infty}
(a_n\cos\cfrac{n\pi at}{l}+b_n\sin\cfrac{n\pi at}{l})\sin\cfrac{n\pi x}{l}\tag{1.8}
$$
得到仍然满足弦振动方程和边界条件的==一般解==。
现在我们选定适当的 $a_n,b_n$ 使 $u(x,t)$ 满足初始条件
$$
\begin{cases}\displaystyle
 u|_{t=0}=\sum_{n=1}^{\infty}a_n\cos\cfrac{n\pi x}{l}=\phi(x) \\
\displaystyle
\cfrac{∂u}{∂t}|_{t=0}=\sum_{n=1}^{\infty}b_n\cfrac{n\pi a}{l}\sin\cfrac{n\pi x}{l}=\psi(x)
\end{cases}\quad(0<x<l)
$$
这两式正好是 $\phi(x),\psi(x)$ 在 $[0,l]$ 上的傅里叶正弦展开，因此我们可以正弦展开的系数公式得到
$$
\begin{cases}\displaystyle 
a_n=\cfrac{2}{l}\int_0^l\phi(\xi)\sin\cfrac{n\pi\xi}{l}d\xi \\
\displaystyle 
b_n=\cfrac{2}{n\pi a}\int_0^l\psi(\xi)\sin\cfrac{n\pi\xi}{l}d\xi
\end{cases}\tag{1.9}
$$


**两端自由弦的振动**：（第二类边界条件）设长度为 $l$，两端自由弦的定解问题为
$$
\begin{cases}
u_{tt}-a^2u_{xx}=0 & x\in(0,l)\\
u|_{t=0}=\phi(x),\cfrac{∂u}{∂t}|_{t=0}=\psi(x) & x\in[0,l]\\
\cfrac{∂u}{∂x}|_{x=0}=0,\cfrac{∂u}{∂x}|_{x=l}=0
\end{cases}\tag{1.10}
$$
同样用分离变量法我们可以求得
$$
u_n(x,t)=\begin{cases}
a_0+b_0t \quad (n=0)\\
(a_n\cos\cfrac{n\pi at}{l}+b_n\sin\cfrac{n\pi at} {l})\sin\cfrac{n\pi x}{l} \quad(n=1,2,\cdots)
\end{cases}\tag{1.11}
$$
他们的线性叠加为
$$
\displaystyle u(x,t)=a_0+b_0t+\sum_{n=1}^{\infty}
(a_n\cos\cfrac{n\pi at}{l}+b_n\sin\cfrac{n\pi at}{l})\sin\cfrac{n\pi x}{l}\tag{1.12}
$$
由初始条件可确定系数为
$$
\begin{cases}\displaystyle 
a_0=\cfrac{1}{l}\int_0^l\phi(\xi) \\
\displaystyle 
b_0=\cfrac{1}{l}\int_0^l\psi(\xi)
\end{cases}
,\quad
\begin{cases}\displaystyle 
a_n=\cfrac{2}{l}\int_0^l\phi(\xi)\sin\cfrac{n\pi\xi}{l}d\xi \\
\displaystyle 
b_n=\cfrac{2}{n\pi a}\int_0^l\psi(\xi)\sin\cfrac{n\pi\xi}{l}d\xi
\end{cases}\tag{1.13}
$$
**均匀杆的热传导问题**：（第三类边界条件）设长度为 $l$ 的杆，$x=0$ 处恒定为零度，$x=l$ 热量自由发散到温度为零的介质中，定解问题为
$$
\begin{cases}
u_{t}-a^2u_{xx}=0 &x\in(0,l)\\
u|_{t=0}=\phi(x) & x\in[0,l]\\
u|_{x=0}=0,u_x(l,t)+hu(l,t)=0
\end{cases}\tag{1.14}
$$
其中常数 $h>0$ ，同样用分离变量法我们可以取得本征值满足的方程
$$
\sqrt{λ}\cos\sqrt{λ} l+h\sin\sqrt{λ} l=0
$$
对应的函数族为
$$
X_n=B_n\sin\sqrt{λ_n}x,\quad T_n=A_ne^{-λ_na^2t}
$$
进一步求得
$$
u_n(x,t)=C_ne^{-λ_na^2t}\sin\sqrt{λ_n}x\quad (n=1,2,\cdots)\tag{1.15}
$$
线性叠加为
$$
\displaystyle u(x,t)=\sum_{n=1}^{\infty}C_ne^{-λ_na^2t}\sin\sqrt{λ_n}x\tag{1.16}
$$
由初始条件可确定系数为
$$
c_n=\cfrac{1}{L_n}\int_{0}^{l}\phi(\xi)\sin\sqrt{λ_n}\xi d\xi\tag{1.17}
$$
其中
$$
\displaystyle L_n=\int_{0}^{l}\sinλ_n\xi d\xi
$$
**矩形区域拉普拉斯方程的边界值问题**：
$$
\begin{cases}
u_{xx}+u_{yy}=0 & 0<x<a,0<y<b\\
u|_{y=0}=f(x),u|_{y=b}=g(x) & 0⩽y⩽b\\
\cfrac{∂u}{∂x}|_{x=0}=0,\cfrac{∂u}{∂x}|_{x=a}=0 &0⩽x⩽a
\end{cases}\tag{1.18}
$$
设解 $u=X(x)Y(y)$ ，同样运用分离变量法我们得到
$$
\cfrac{X''}{X}=-\cfrac{Y''}{Y}=-λ
$$
由此得到
$$
\begin{cases}
X''+λX=0 \\
Y''-λy=0 
\end{cases}
$$
同样带入关于 $x$ 的齐次边界条件，可得
$$
\begin{cases}X'(0)Y(y)=0 \\X'(a)Y(y)=0\end{cases}
$$
这时必须有 $X'(0)=X'(a)=0$ ，这样我们同样的得到一个本征值问题
$$
\begin{cases}
X''(x)+\lambda X(x)=0 \\
X'(0)=0,X'(a)=0
\end{cases}
$$
可解得特征值和对应的特征函数，进一步可求得 $u_n(x,y)$ ，最后叠加并利用关于 $y$ 的非齐次条件可求得
$$
\displaystyle u(x,y)=\frac{1}{2}(\pi-y)+
\frac{2}{\pi}\sum_{n=1}^{\infty}\cfrac{[(-1)^n-1]}{n^2}
\cfrac{\sinh n(\pi-y)}{\sinh n\pi}\cos nx
$$

**圆形区域内拉普拉斯方程边界值问题**
$$
\begin{cases}
Δ=u_{xx}+u_{yy}=0 & (x^2+y^2<a^2) \\
u|_{x^2+y^2=a^2}=f(x,y) 
\end{cases}\tag{4.1}
$$
由于区域为圆域，边界条件不能直接分离变量，考虑在平面极坐标系中，上述边界值问题化为
$$
\begin{cases}
u_{rr}+\cfrac{1}{r}u_r+\cfrac{1}{r^2}u_{ϕϕ} & (r<a,0⩽ϕ⩽2\pi) \\
u|_{r=a}=f(ϕ) & (0⩽ϕ⩽2\pi)
\end{cases}\tag{4.2}
$$
又因为在极坐标系中，$(r,ϕ)$ 和 $(r,ϕ+2\pi)$ 是同一点，需考虑==自然周期性条件==
$$
u(r,ϕ)=u(r,ϕ+2\pi)\tag{4.3}
$$
令 $u(r,ϕ)=R(r)Φ(ϕ)$ ，带入 (4.2) 中的方程分离变量得到
$$
\cfrac{r^2R''+rR'}{R}=-\cfrac{Φ''}{Φ}=λ
$$
于是有
$$
Φ''+λΦ=0 \tag{4.4}
$$

$$
r^2R''+rR'-λR=0 \tag{4.5}
$$

将 $u(r,ϕ)=R(r)Φ(ϕ)$ 带入周期条件可得到
$$
Φ(ϕ)=Φ(ϕ+2\pi),\quad Φ'(ϕ)=Φ'(ϕ+2\pi)\tag{4.6}
$$
常微分方程 (4.4) 和周期条件 (4.6) 构成本征值问题，容易求得方程 (4.4) 的通解为
$$
Φ=\begin{cases}
Ae^{\sqrt{-λ}ϕ}+Be^{-\sqrt{-λ}ϕ} & (λ<0)\\
A+Bϕ  & (λ=0)\\
A\cos\sqrt{λ}ϕ+B\sin\sqrt{λ}ϕ & (λ>0) 
\end{cases}
$$
将上式带入周期条件 (4.6) 从而求得本征值（当 $λ=0$ 时，$Φ\equiv0$ ，无意义）
$$
λ=m^2 \quad (m=0,1,2,\cdots)
$$
对应的本征函数为
$$
Φ=\begin{cases}
A_0  & (m=0)\\
A_m\cos mϕ+B_m\sin mϕ & (m\neq0) 
\end{cases}
$$
然后将本征值带入 (4.5) 得到
$$
r^2R''+rR'-m^2R=0
$$
此方程为欧拉方程，其通解为
$$
R=\begin{cases}
C_0+D_0\ln r  & (m=0)\\
C_mr^m+D_m\cfrac{1}{r^m} & (m\neq0) 
\end{cases}
$$
同时考虑在平面原点处 $u(x,y)$ 是有界的，所以补充条件 $u(0,ϕ)<\infty$
于是 $D_m=0,\quad(m=0,1,2\cdots)$ ，根据叠加原理可得到一般解
$$
\displaystyle u(r,ϕ)=\cfrac{a_0}{2}+\sum_{m=1}^\infty r^m(a_m\cos mϕ+b_m\sin mϕ)
$$
其中 $a_m,b_m\quad (m=0,1,2,\cdots)$ 为任意常数。将一般解带入 (4.2) 的边界值条件，根据本征函数的正交性可以得到系数
$$
\begin{cases}\displaystyle 
a_m=\cfrac{1}{a^m\pi}\int_0^{2\pi}f(t)\cos mtdt  & (m=0,1,2,\cdots) \\
\displaystyle 
b_m=\cfrac{1}{a^m\pi}\int_0^{2\pi}f(t)\sin mtdt  & (m=1,2,\cdots)
\end{cases}
$$
将系数带入上述一般解
$$
\displaystyle u(r,ϕ)=\cfrac{1}{2\pi}\int_0^{2\pi}f(t)
[1+2\sum_{m=1}^\infty(\cfrac{r}{a})^m\cos m(ϕ-t)]dt
$$
显然当 $r<a$ 时，级数收敛，将余弦函数改为复指数函数，利用等比级数的求和公式，最后可以得到
$$
\displaystyle u(r,ϕ)=\cfrac{a^2-r^2}{2\pi}\int_0^{2\pi}
\cfrac{f(t)}{r^2+a^2-2ar\cos(ϕ-t)}dt\tag{4.7}
$$
这个结果称为圆域内==泊松积分公式==。

## 齐次边界条件下非齐次方程的定解问题

上节只考虑了方程和边界条件都是齐次的情况，实际上还有方程和边界条件二者中至少有一个是非齐次的情况，称为非齐次问题。本节介绍齐次边界条件下非齐次方程的定解问题。

**两端固定弦的强迫振动**：定解问题为
$$
\begin{cases}
u_{tt}-a^2u_{xx}=f(x,t) & x\in(0,l)\\
u|_{t=0}=\phi(x),\cfrac{∂u}{∂t}|_{t=0}=\psi(x) & x\in[0,l]\\
u|_{x=0}=0,u|_{x=l}=0
\end{cases}\tag{2.1}
$$
分离变量法给出的结果提示：不妨把所求的解本身展开为傅里叶级数，即
$$
\displaystyle u(x,t)=\sum_{n=1}^{\infty}T_n(t)X_n(x)\tag{2.2}
$$
把 $\{X_n(x)\}$ 作为傅里叶级数基本函数族，而$T_n(t)$ 作为待定系数，只要基本函数族是完备的，那么所求的解 $u(x,t)$ 以及非齐次项 $f(x,t)$ 均可按基本函数族 $\{X_n(x)\}$ 展开，设
$$
\displaystyle f(x,t)=\sum_{n=1}^{\infty}g_n(t)X_n(x)\tag{2.3}
$$
原则上基本函数族的选取并无限制，但选择定解问题 (2.1) 对应的齐次定解问题的本征函数，会使后面求解待定系数变得简单可行。对应的齐次方程和齐次边界条件的本征值问题为
$$
\begin{cases}
X''(x)+\lambda X(x)=0 \\
X(0)=0,X(l)=0
\end{cases}\tag{1.3}
$$
根据上节求得本征函数，我们取 $X_n=\sin\cfrac{n\pi x}{l}$
利用本征函数的正交性可求得[^2]
$$
g_n(t)=\cfrac{2}{l}\int_0^lf(ξ,t)\sin\cfrac{n\piξ}{l}dξ
$$
将$u(x,t),f(x,t)$ 的展开式带入 (2.1) 偏微分方程，并逐项微商
$$
\displaystyle
\sum_{n=1}^{\infty}T''_n(t)X_n(x)-a^2\sum_{n=1}^{\infty}T_n(t)X''_n(x)
=\sum_{n=1}^{\infty}g_n(t)X_n(x)
$$
利用 $X_n$ 所满足 (1.3) 的常微分方程，又可以化成
$$
\displaystyle
\sum_{n=1}^{\infty}T''_n(t)X_n(x)+a^2\sum_{n=1}^{\infty}T_n(t)\lambda_n X_n(x)
=\sum_{n=1}^{\infty}g_n(t)X_n(x)
$$
根据傅里叶展开系数的唯一性，我们可以得到$T_n(t)$ 满足的常微分方程
$$
T''_n(t)+\lambda_na^2T_n(t)=g_n(t)\tag{2.4}
$$
同样将 $u(x,t)$ 的展开式带入初始条件，可得
$$
\displaystyle\sum_{n=1}^{\infty}T_n(0)X_n(x)=\phi(x),
\quad\sum_{n=1}^{\infty}T'_n(0)X_n(x)=\psi(x)
$$
根据傅里叶展开系数公式，可以导出
$$
\begin{cases}\displaystyle 
T_n(0)=\cfrac{2}{l}\int_0^l\phi(ξ)\sin\cfrac{n\piξ}{l}dξ \\
\displaystyle 
T'_n(0)=\cfrac{2}{l}\int_0^l\psi(ξ)\sin\cfrac{n\piξ}{l}dξ
\end{cases}\tag{2.5}
$$
可用常数变易法求得 $T_n(t)$ 满足方程 (2.4) 和初始条件 (2.5) 的解
$$
T_n=\cfrac{l}{n\pi a}\int_0^tg_n(τ)\sin\cfrac{2\pi a}{l}(t-τ)dτ
+T_n(0)\cos\cfrac{n\pi at}{l}+\cfrac{lT'_n(0)}{n\pi a}\sin\cfrac{n\pi at}{l}
$$
将 $X_n(x),T_n(t)$ 带入 (2.2) 便可得到定解问题 (2.1) 的一般解。

用同样的分离变量法也可解决热传导非齐次方程和齐次边界条件的定解问题、二维泊松方程的边界值问题。

[^2]: 式 (2.3) 两端同乘以 $X_m(x)$ ，并逐项积分得
    $$
    \displaystyle\sum_{n=1}^{\infty}\int_0^lg_n(t)X_n(x)X_m(x)dx=
    \int_0^lf(x,t)X_m(x)dx
    $$
    根据本征函数的正交性，当 $n\neq m$ 时，    $\displaystyle\int_0^lX_n(x)X_m(x)dx=0$ ，所以只留下 $n=m$ 项
    $$
    \displaystyle\int_0^lg_m(t)X^2_m(x)dx=
    \int_0^lf(x,t)X_m(x)dx
    $$
    将 $X_m$ 带入上式可求得
    $$
    \displaystyle g_n(t)=\cfrac{2}{l}\int_0^lf(x,t)\sin\cfrac    {n\pi x}{l}dx
    $$

## 非齐次边界条件的齐次化

非齐次边界条件处理的原则是利用叠加原理，把非齐次边界条件问题转化为另一未知函数的齐次边界条件。
考虑下面定解问题
$$
\begin{cases}
u_{tt}-a^2u_{xx}=f(x,t) & x\in(0,l)\\
u|_{t=0}=\phi(x),\cfrac{∂u}{∂t}|_{t=0}=\psi(x) & x\in[0,l]\\
u|_{x=0}=p(t),u|_{x=l}=q(t)
\end{cases}\tag{3.1}
$$
为了使边界条件齐次化，设
$$
u(x,t)=v(x,t)+w(x,t)\tag{3.2}
$$
使得 $v,w$ 分别满足边界条件
$$
\begin{cases}v|_{x=0}=p(t) \\ v|_{x=l}=q(t) \end{cases},\quad
\begin{cases}w|_{x=0}=0 \\ w|_{x=l}=0 \end{cases}
\tag{3.3}
$$
这样 $w(x,t)$ 满足的边界条件是齐次的，关于 $w(x,t)$ 的定解问题为
$$
\begin{cases}
w_{tt}-a^2w_{xx}=f_1(x,t) & x\in(0,l)\\
w|_{t=0}=\phi_1(x),\cfrac{∂u}{∂t}|_{t=0}=\psi_1(x) & x\in[0,l]\\
w|_{x=0}=0,w|_{x=l}=0
\end{cases}\tag{3.4}
$$
其中
$$
\begin{cases}
f_1(x,t)=f(x,t)-(v_{tt}-a^2v_{xx}) \\
\phi_1(x)=\phi(x)-v(x,0) \\
\psi_1(x)=\psi(x)-\cfrac{∂v}{∂t}|_{t=0}
\end{cases}
$$
关于函数 $v(x,t)$ 的选择是随意的，它只需满足 (3.3) 中的边界条件。不妨取直线
$$
v(x,t)=A(t)x+B(t)
$$
代入 (3.3) 中的边界条件，可求得 $A(t)=\frac{1}{l}[q(t)-p(t)],\quad B(t)=p(t)$
从而求出 $w(x,t)$ 再代回 (3.2) 就得到定解问题的解 $u(x,t)$ 。
选择不同的齐次化函数 $v(x,t)$ 导出的 $w(x,t)$ 定解问题也就不同，但是由于定解问题的解的存在唯一性，就保证了给出的 $u(x,t)$ 是相同的，尽管表达形式可能有所不同。

## 施图姆-刘维尔本征值问题

**一般的分类变量法**：对于一般的二阶线性齐次偏微分方程定解问题
$$
\begin{cases}
L_t[u]+L_x[u]=0 &x\in(a,b) \\
(α_1\cfrac{∂u}{∂x}+β_1u)|_{x=a}=0\\ 
(α_2\cfrac{∂u}{∂x}+β_2u)|_{x=b}=0 \\
关于 t 的定解条件
\end{cases}\tag{4.1}
$$
其中系数  $|α_1|+|β_1|\neq0,|α_2|+|β_2|\neq0$ ，方程中 $L_t,L_x$ 分别为线性偏微分算子
$$
L_t=a_0(t)\cfrac{∂^2}{∂t^2}+a_1(t)\cfrac{∂}{∂t}+a_2(t) \\
L_x=b_0(x)\cfrac{∂^2}{∂x^2}+b_1(x)\cfrac{∂}{∂x}+b_2(x)
$$
(1) 分离变量：令 $u(x,t)=X(x)T(t)$ 带入上述齐次方程和齐次边界条件，可得本征值问题
$$
\begin{cases}
L_x[X]+λX=0 \\
α_1X'(a)+β_1X(a)=0,\ α_2X'(b)+β_2X(b)=0
\end{cases}\tag{4.2}
$$
和常微分方程
$$
L_t[T]-λT=0
$$
(2) 解本征值问题，获得分离变量形式的特解
就 $λ$ 的不同情况，求本征值问题的通解，并由边界条件求出本征值 $\{λ_n\}$ 及对应的本征函数族 $\{X_n(x)\}$ ，再将本征值带入关于 $T(t)$ 的方程，求出相应的解，从而得到定解问题 (4.1) 分离变量形式的解
$$
u_n(x,t)=X_n(x)T_n(t)
$$
(3) 叠加定系数：令
$$
u(x,t)=\sum_{n=1}^{\infty}C_nX_n(x)T_n(t)
$$
带入关于 $t$ 的定解条件，定出系数 $C_n$ ，从而获得定解问题 (2.1) 的解。

以上三步中，分离变量是基础，求解本征值问题是核心。所谓本征值问题，就是在一定的边界条件下，求解含参数 $λ$ 的齐次线性常微分方程的非零解问题。本征函数不仅满足了齐次方程、齐次边界条件的分离变量形式的解族，更重要的是本征函数族正好是函数的傅里叶正弦展开或者傅里叶展开的完备正交函数系。从而可以通过适当叠加满足初始条件或者其他条件的解。分离变量法是否有效，完全取决于下列的基本理论问题：
(1) 本征值是否存在；
(2) 本征函数系是否存在，若存在，是否正交；
(3) 给定的函数是否可按照本征函数系展开。

**施图姆-刘维尔理论**：考虑带有参数 $λ$ 的二阶线性齐次常微分方程
$$
y''+a(x)y'+b(x)y+λc(x)y=0\quad(a<x<b)
$$
 如果方程两边同乘以 $\displaystyle k(x)=\exp[\int a(x)dx]$ 可化为
$$
\cfrac{d}{dx}[k(x)\cfrac{dy}{dx}]-q(x)y+λ\rho(x)y=0\quad(a<x<b)\tag{4.3}
$$
其中 $\displaystyle k(x)=\exp[\int a(x)dx],\quad q(x)=-b(x)k(x),\quad \rho(x)=c(x)k(x)$ 是实函数， $\rho(x)$ 称为==权重函数==，这个方程称为==施图姆-刘维尔方程== (Sturm-Liouville)，简称 ==S-L 方程==，我们在分离变量法中遇到的常微分方程都是式 (4.3) 的特例。S-L 方程附加上适当的边界条件就就构成==施图姆-刘维尔本征值问题==， $λ$ 称为本征值，满足 S-L 方程及相应的边界条件的非零解就是本征函数。
例如，当 $k(x)=1,q(x)=0,\rho(x)=1,a=0,b=l$ 时，方程 (6.1) 变为 $y''+λy=0,\quad(0<x<l)$


S-L 型方程，即 $k(x)y''+k'(x)y'-q(x)y+λ\rho(x)y=0$，亦可化为
$$
y''+\cfrac{k'(x)}{k(x)}y'+\cfrac{-q(x)+λ\rho(x)}{k(x)}y=0
$$
通常分为正则和奇异两种类型：

- **类型 1**  如果区间 $[a,b]$ 有界，系数 $k(x),q(x),\rho(x)$ 在 $[a,b]$ 上连续，且 $k(x)>0,\rho(x)>0$ ，则称 S-L 方程在  $(a,b)$ 上是正则的（非奇异的）
  如果 S-L 方程是正则的，那么 $k(a),k(b)>0$ 常见的边界条件是 $y(x),y'(x)$ 在端点 $a$ 和 $b$ 处的线性组合，第一、第二、第三齐次边界条件可以统一表示为

$$
  \begin{cases}
  α_1y'(a)+β_1y(a)=0 \\
  α_2y'(b)+β_2y(b)=0
  \end{cases}
$$

  其中 $|α_1|+|β_1|\neq0,|α_2|+|β_2|\neq0$ ，如果还有 $k(a)=k(b)$ 则给予周期边界条件
$$
y(a)=y(b)
$$

- **类型 2**   或者区间 $[a,b]$ 是无界的，或者当 $k(x)$ 或 $\rho(x)$ 在区间 $[a,b]$ 的一个端点或两个端点处等于零，则称 S-L 方程在  $(a,b)$ 上是奇异的。例如勒让德方程在 $(0,1)$ 上是奇异的。
  当边界点 $a,b$ 是函数 $k(x)$ 的一阶零点时，此时 S-L 方程 是奇异的，那么在该边界点上给予自然边界条件：
  若 $k(a)=0$，则 $y(a),y'(a)$ 有界；若 $k(b)=0$，则 $y(b),y'(b)$ 有界

**共同性质**：如果函数 $k(x),k'(x),q(x),\rho(x)$ 在 $[a,b]$ 上是实值连续函数
(1) 相应于不同本征值 $λ_m,λ_n$ 的本征函数 $y_m(x),y_n(x)$ 在区间 $[a,b]$ 上带权重函数 $\rho(x)$ 是==正交==的
$$
\displaystyle \int_a^by_m(x)y_n(x)\rho(x)dx=0\quad(m\neq n)\tag{4.4}
$$
例如：对于本征值问题
$$
X''+λX=0 \\
X(0)=X(l)=0
$$
设$X_n,X_m$分别是本征值问题对应于本征值 $λ_n\neqλ_m$ 的本征函数，则
$$
X_m(X_n''+λ_n X_n)=0 \\
X_n(X_m''+λ_m X_m)=0
$$
两式相减，并在 $[0,l]$ 上积分，利用本征值问题的边界条件，得到
$$
\begin{aligned}\displaystyle
&(λ_n-λ_m)\int_0^l X_nX_mdx \\
=&\int_0^l [X_nX_m''-X_mX_n'']dx \\
=&[X_nX_m'-X_mX_n']|_0^l \\
=&0
\end{aligned}
$$
因此得到本征函数的==正交性==
$$
\displaystyle\int_0^l X_n(x)X_m(x)dx=0\quad(n\neq m)
$$
(2) 若 $\rho(x)>0,x\in(a,b)$ 所有本征值是实数，且存在无穷多个本征值
$$
λ_0<λ_2<λ_3<\cdots<λ_n<\cdots
$$
其中当 $n\to\infty$ 时对应的无穷多个本征实函数 $y_n(x)$ 在区间 $(a,b)$ 内恰有 $n$ 个零点。此外，这些本征函数组成一个完备的正交系 $\{y_n(x)\}$
(3) 进一步地，若函数 $f(x)$ 在 $[a,b]$ 上满足狄利克雷条件和 S-L 问题的边界条件，那么 $f(x)$ 可以展开成  $\{y_n(x)\}$ 的傅里叶级数
$$
\displaystyle f(x)=\sum_{n=0}^{\infty}C_ny_n(x)
$$
如果函数 $f(x)$ 具有一阶连续偏导数和分段连续的二阶导数，则上述级数绝对且一致收敛于 $f(x)$

# 附录

## 热传导方程导出

由于温度分布不均匀，热量从温度高的地方向温度低的地方转移，这种现象叫热传导。
热传导研究的是温度在空间中的分布和在时间中的变化 $u(x,y,z,t)$ 。
热传导的起源是温度分布不均，温度分布不均匀的程度用温度梯度表示 $∇u$ ，热传导的强弱用热流强度表示 $\mathbf q$ ，即单位时间通过单位横截面的热量。根据实验，热传导遵循热传导定律，即傅里叶定律
$$
\mathbf q=-κ∇u
$$
其中比例系数 $κ$ 叫热传导系数， 与介质的质料有关。或写成分量的形式
$$
q_x=-\cfrac{∂u}{∂x},q_y=-\cfrac{∂u}{∂y},q_z=-\cfrac{∂u}{∂z}
$$
负号表示热量转移的方向和温度梯度相反。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/heat-conduction-equation.png)
取一封闭曲面 $S$ 包围的区域 $V$，根据傅里叶定律，在 $\mathrm dt$ 时间内穿过面元 $\mathrm dS$ 的热量
$$
\mathrm dQ=-κ∇u\cdot\mathrm d\mathbf S\mathrm dt
$$
若区域内无热源，从$t$ 到 $t+\mathrm dt$ 时刻根据能量守恒定律有
$$
\iiint\limits_Vc\rho\mathrm dV\mathrm du=\oiint\limits_Sκ∇u\cdot\mathrm d\mathbf S\mathrm dt
$$
其中$c$ 为比热容，$\rho$ 为密度。将等式右边用高斯公式变换
$$
\iiint\limits_Vc\rho\mathrm dV\mathrm du
=\iiint\limits_V∇\cdot(κ∇u)\mathrm dV\mathrm dt
$$
由于区域的随意性，取极限 $V\to M(x,y,z)$ 得到
$$
c\rho u_t=∇\cdot(κ∇u)
$$
如果物体是均匀的，此时 $c,\rho,κ$ 均为常数，则
$$
u_t-a^2Δu=0
$$
其中系数 $a^2=\cfrac{κ}{\rho c}$，它称为三维热传导方程。

如果物体内有热源，其热源密度函数为 $F(x,y,z,t)$ ，则热传导方程为
$$
u_t-a^2Δu=f
$$
其中 $f=\cfrac{F}{\rho c}$


------

> **参考文献：**
> 季孝达.《数学物理方程》.
> 吴崇试.《数学物理方法》.
> 梁昆淼.《数学物理方法》.
> 吴崇试 高春媛.《数学物理方法》.北京大学(MOOC)