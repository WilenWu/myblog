---
title: 常微分方程(一)
categories:
  - 数学
  - 常微分方程
tags:
  - 数学
  - ODE
  - 微分方程
katex: true
description: 一阶常微分方程
abbrlink: 1d08ba54
date: 2019-05-02 12:36:35
cover:
top_img:
keywords:
---

# 微分方程基本概念

## 微分方程

**微分方程**：包含未知函数及其导数的方程叫做==微分方程==(Differential Equation)。未知函数导数的最高阶数称为该==微分方程的阶==(order)。

1. **常微分方程(ODE)**：若未知函数是一元函数的微分方程称为常微分方程(Ordinary Differential Equation, ODE)  。
    一般的 ==n 阶常微分方程==的形式(也称隐式表达式)为
    $$
    F(x,y,y',y'',\dots,y^{(n)})=0 \tag{1}
    $$
     如果微分方程是关于未知函数及各阶导数的线性表达式 
    $$
    y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y'+a_n(x)y=f(x)\tag{2}
    $$
    则称为 ==n 阶线性(linearity)常微分方程==。

2. **偏微分方程(PDE)**：若未知函数是多元函数，方程中含有自变量的偏微分，称之为偏微分方程(Partial Differential Equations, PDE)。
如 $\cfrac{∂^2T}{∂x^2}+\cfrac{∂^2T}{∂y^2}+\cfrac{∂^2T}{∂z^2}=0$

## 微分方程的解

如果将一个函数$y=φ(x)$其各阶导数代入微分方程 (1) 得到恒等式 
$$
F(x,φ(x),φ'(x),φ''(x),\dots,φ^{(n)}(x))\equiv0
$$
 则称$y=φ(x)$为上述方程的一个==解==(solution)。

- **通解**：n 阶微分方程 (1) 的解 $y=φ(x,C_1,C_2,\cdots,C_n)$ 含有 n 个相互独立[^J]的任意常数 $C_1,C_2,\cdots,C_n$ ，则称为该微分方程的==通解==(general solution)。

[^J]: 其中任意常数相互独立是指每一个常数对解的影响是其他常数所不能代替的，即雅克比行列式(Jacobian)满足 
    $$
    \cfrac{∂(φ,φ',\cdots,φ^{(n-1)})}{∂(C_1,C_2,\cdots,C_n)}=
    \begin{vmatrix}
    \frac{∂φ}{∂C_1} &\frac{∂φ}{∂C_2} & \cdots &\frac{∂φ}{∂C_n} \\ 
    \frac{∂φ'}{∂C_1} &\frac{∂φ'}{∂C_2} &\cdots &\frac{∂φ'}{∂C_n} \\ 
    \vdots &\vdots &\ddots &\vdots \\ 
    \frac{∂φ^{(n)}}{∂C_1} &\frac{∂φ^{(n)}}{∂C_2} &\cdots &\frac{∂φ^{(n)}}{∂C_n} \\ 
    \end{vmatrix}\neq 0
    $$


- **特解**：我们称不包含任意常数的解为==特解==(particular solution)。

- **初值问题**：通常为了解决实际问题，确定常数的值，需要引入初值条件(initial conditions)。初值条件联合微分方程组成==初值问题==(Initial Value Problem, IVP)，或称柯西问题。
一阶常微分方程的初值问题通常记作 $\begin{cases}y'=f(x,y) \\ 
y(x_0)=y_0\end{cases}$

- **隐式解与隐式通解**：如果关系式 $\Phi(x,y)=0$ 所确定的隐函数 $y=φ(x)$ 为微分方程 (1) 的解，则称 $\Phi(x,y)=0$ 是方程的一个==隐式解==(implicit solution)。对于含有 n个相互独立常数的解  $\Phi(x,y,C_1,C_2,\cdots,C_n)=0$ 为隐式通解。

## 解的几何意义

1. **积分曲线**：微分方程的解对应的曲线称为==积分曲线==(integral curve)。
对于一阶微分方程初值问题 $\begin{cases}y'=f(x,y) \\ 
y(x_0)=y_0\end{cases}$ 的解对应过点$(x_0,y_0)$的一条积分曲线，该曲线在点$(x_0,y_0)$ 处的切线的斜率为 $f(x_0,y_0)$，切线方程为 $y=y_0+f(x_0,y_0)(x-x_0)$
若不给定初始条件，微分方程的通解在几何上对应着一族积分曲线，称为==积分曲线族==(family of integral curves)。

2. **线素场**：考虑微分方程 $y'=f(x,y)$，若 $f(x,y)$ 的定义域为平面区域G，在G 内每一点 $P(x,y)$ 作斜率为 $f(x,y)$ 的单位线段，则称该线段为点 $P(x,y)$的==线素==。G 内所有的线素构成由微分方程确定的==线素场==(line element field)或==方向场==(direction field)。
在构造微分方程 $y'=f(x,y)$ 的线素场时，通常利用斜率关系式 $f(x,y)=k$ 确定曲线 $L_k$，称它为线素场的==等斜线==(isocline)。显然，等斜线$L_k$上各点的斜率都等于 $k$，简化了线素场逐点构造的方法。

3. **奇异点**：设一阶微分方程为 $P(x,y)dx+Q(x,y)dy=0$，函数$P(x,y),Q(x,y)$在区域G是连续的，若 $P(x_0,y_0)=Q(x_0,y_0)=0,(x_0,y_0)\in G$，线素场在点$(x_0,y_0)$处便失去意义，我们称这样的点为==奇异点==(singular point)。

- **示例**：作微分方程 $y'=y/x$ 和 $y'=-x/y$的线素场。
(1) $y'=y/x$ 的等斜线为 $L_k:y=kx$，说明线素斜率为 k 的所有点都集中在直线 $y=kx$ 上，也可求得方程的积分曲线簇为射线 $\tanθ=y/x,θ$为任意常数，原点 O为奇异点。
(2) $y'=-x/y$的等斜线为 $L_k:y=-\frac{1}{k}x$，说明线素斜率为 k 的所有点都集中在直线 $y=-\frac{1}{k}x$ 上，且线素斜率和等斜线垂直相交，也可求得方程的积分曲线簇为同心圆 $x^2+y^2=C^2$，原点 O为奇异点。
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/direction-field-demo.png" height="160">  <img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/direction-field-demo2.png" height="160" width="160">

# 一阶常微分方程

## 可分离变量方程

对于形如
$$
\frac{\mathrm{d}y}{\mathrm{d}x}=g(x)h(y)\tag{1}
$$
的微分方程，称为==可分离一阶方程==(separable first-order equations)，方程可化为 $\cfrac{1}{h(y)}\mathrm{d}y=g(x)\mathrm{d}x$，对方程两边积分
$$
\displaystyle\int \cfrac{1}{h(y)}\mathrm{d}y=\int g(x)\mathrm{d}x+C
$$

设$H(y),G(x)$分别为$\cfrac{1}{h(y)},g(x)$的原函数，可知隐式通解为 $H(y)=G(x)+C$
这种通过分离变量求方程通解的方法叫做==分离变量法==(Separation of variables)。

## 变量替换法

**一阶齐次微分方程**：可化为
$$
\frac{\mathrm{d}y}{\mathrm{d}x}=φ(\frac{y}{x})\tag{1}
$$
 的方程称为==一阶齐次微分方程==。
引入中间函数 $u=\cfrac{y}{x}$，则 $y=ux$，由此 $\dfrac{\mathrm{d}y}{\mathrm{d}x}=x\cfrac{\mathrm{d}u}{\mathrm{d}x}+u$
带入原方程 $u+x\cfrac{\mathrm{d}u}{\mathrm{d}x}=φ(u) \implies \cfrac{\mathrm{d}u}{φ(u)-u}=\cfrac{\mathrm{d}x}{x}$
得到可分离变量的微分方程，求隐式通解即可。

**齐次方程等价形式**：微分方程 
$$
P(x,y)\mathrm{d}x+Q(x,y)\mathrm{d}y=0\tag{2}
$$
中的函数$P(x,y),Q(x,y)$都是 $x,y$ 的同次(例如 k 次)齐次函数，即
$P(tx,ty)=t^kP(x,y),Q(tx,ty)=t^kQ(x,y)$
比如，线性代数中出现的二次型 $f(x, y)=ax^2+bxy+cy^2$ 就是一个二次齐次函数，它满足 $f(tx, ty)=t^2f(x,y)$ 
根据齐次函数的定义，我们有 $\begin{cases}
P(x,y)=x^kP(1,y/x) \\
Q(x,y)=x^kQ(1,y/x) \\
\end{cases}$
从而方程 (2) 化为下面的形式
$x^kP(1,y/x)\mathrm{d}x+x^kQ(1,y/x)\mathrm{d}y=0$
整理可得
$\cfrac{\mathrm{d}y}{\mathrm{d}x}=-\cfrac{P(1,y/x)}{Q(1,y/x)}=φ(\cfrac{y}{x})$
方程化为了齐次方程标准形式，齐次二字来源于齐次函数的定义，所以与齐次线性方程中的齐次不同。

**可化为齐次的类型 I**
$$
\frac{\mathrm{d}y}{\mathrm{d}x}=f(ax+by+c)\tag{3}
$$
 引入新变量 $u=ax+by+c$ ，则 $\dfrac{\mathrm{d}u}{\mathrm{d}x}=a+b\dfrac{\mathrm{d}y}{\mathrm{d}x}$
带入原方程 $\dfrac{\mathrm{d}u}{\mathrm{d}x}=a+bf(u)$
这是可分离变量的微分方程，求隐式通解即可。

**可化为齐次的类型 II**
$$
\frac{\mathrm{d}y}{\mathrm{d}x}=f(\frac{ax+by+c}{a_1x+b_1y+c_1})\tag{4}
$$
(1) 当 $c=c_1=0$时，是齐次方程。
(2) 非齐次情形时，令 $x=X+h,y=Y+k$，于是$\mathrm{d}x=\mathrm{d}X,\mathrm{d}y=\mathrm{d}Y$，方程化为
$$
\frac{\mathrm{d}Y}{\mathrm{d}X}=f(\frac{aX+bY+ah+bk+c}{a_1X+b_1Y+a_1h+b_1k+c_1})
$$
如果方程组
$$
\begin{cases}ah+bk+c=0 \\ a_1h+b_1k+c_1=0\end{cases}
$$
 (i) 系数行列式$\begin{vmatrix}a&b\\a_1&b_1\end{vmatrix}\neq0$，即$\dfrac{a_1}{a}\neq\dfrac{b_1}{b}$，则可求出$h,k$，方程可化为
$$
\frac{\mathrm{d}Y}{\mathrm{d}X}=f(\frac{aX+bY}{a_1X+b_1Y})
$$
求出齐次方程的通解后，带入$X=x-h,Y=y-k$ 即可。
(ii) 当$\dfrac{a_1}{a}=\dfrac{b_1}{b}$时，令$\dfrac{a_1}{a}=\dfrac{b_1}{b}=λ$，从而
$$
\frac{\mathrm{d}y}{\mathrm{d}x}=f(\frac{ax+by+c}{λ(ax+by)+c_1})
$$
引入$v=ax+by$，则$\dfrac{\mathrm{d}v}{\mathrm{d}x}=a+b\dfrac{\mathrm{d}y}{\mathrm{d}x}$，方程化为可分离变量的方程
$$
\frac{\mathrm{d}v}{\mathrm{d}x}=a+bf(\frac{v+c}{λ v+c_1})
$$


## 一阶线性微分方程

形如
$$
\frac{\mathrm{d}y}{\mathrm{d}x}+P(x)y=Q(x)\tag{1}
$$
的方程称为==一阶线性微分方程==(First-order linear differential equation) ，其特点是未知函数 $y$ 和它的一阶导数都是一次的。
如果$Q(x)\equiv0$，称为==齐次线性方程==，$Q(x)\not\equiv0$，则称为==非齐次线性方程==。

**齐次线性方程**：
$$
\frac{\mathrm{d}y}{\mathrm{d}x}+P(x)y=0\tag{2}
$$
 是可分离变量的，对方程两边积分得 
$$
\ln|y|=-\int P(x)\mathrm{d}x+C_1
$$
 或 
$$
y=Ce^{-\int P(x)\mathrm{d}x}\quad(C=± e^{C_1})
$$
另外 $y=0$也是方程的特解，称为==平凡解==。

**非齐次线性方程**：解非齐次方程 (1) 常用的方法是==常数变易法==[^const]
(1) 解对应的齐次方程 (2) 通解
(2) 将齐次方程通解中的常数$C$ 换成未知函数$u(x)$，做变换 
$$
\displaystyle y=ue^{-\int P(x)\mathrm{d}x}\tag{3}
$$
 带入方程 (1) 便可求得  $\displaystyle u=\int Q(x)e^{-\int P(x)\mathrm{d}x}\mathrm{d}x+C$  
(3) 将 u 带入方程 (3)，便求得非齐次方程的通解
$$
\displaystyle y=e^{-\int P(x)\mathrm{d}x}[\int Q(x)e^{-\int P(x)\mathrm{d}x}\mathrm{d}x+C]
$$

[^const]: 常数变易法(method of variation of constant)：是解线性微分方程行之有效的一种方法。它是拉格朗日十一年的研究成果，我们所用仅是他的结论，并无过程。

**伯努利方程**：形如 
$$
\frac{\mathrm{d}y}{\mathrm{d}x}+P(x)y=Q(x)y^n
$$
 的微分方程称为==伯努利方程==(Bernoulli differential equation)
引入变量 $u=y^{1-n}$，方程可转化为一阶线性非齐次方程 
$$
\dfrac{\mathrm{d}u}{\mathrm{d}x}+(1-n)P(x)u=(1-n)Q(x)
$$
 可求得通解。

**里卡蒂方程**：形如 
$$
\cfrac{dy}{dx}=P(x)y^2+Q(x)y+R(x)
$$
的微分方程称为 ==里卡蒂方程== (Riccati equation)。
> 这是形式上最简单的非线性方程，由十七世纪意大利数学家黎卡提提出来的，在1841年法国数学家刘维尔证明了黎卡提方程一般没有初等解法，即其解不能用初等函数以及初等函数的积分来表示。
> 但在特殊情况下，是可以求解的。

若已知黎卡提方程的一个特解，则该方程可以求解。
设 $\tilde y(x)$ 是方程的一个特解，引入下列变换 $y=z+\tilde y$ 其中 $z$ 是新的未知函数，带入方程得到
$\begin{aligned} 
\cfrac{dy}{dx} &= \cfrac{dz}{dx}+\cfrac{d\tilde y}{dx}=P(x)(z+\tilde y)^2+Q(x)(z+\tilde y)+R(x)\\
&=P(x)z^2+2P(x)\tilde yz+Q(x)z+P(x)\tilde y^2+Q(x)\tilde y+R(x)
\end{aligned}$
由于 $\cfrac{d\tilde y}{dx}=P(x)\tilde y^2+Q(x)\tilde y+R(x)$
可得 $\cfrac{dz}{dx}=P(x)z^2+[2P(x)\tilde y+Q(x)]z$
这是以 $z$ 为未知函数， $n=2$ 的伯努利方程，是可以求解的。
从而原方程是可以求解的，它的解为：$y=z(x)+\tilde y(x)$
黎卡提方程告诉我们不是所有的方程都可以用初等积分法来求解的，从而发展起来了新的学科分支微分方程的定性理论。

## 恰当方程

**恰当方程**：对称形式的微分方程 
$$
P(x,y)\mathrm{d}x+Q(x,y)\mathrm{d}y=0\tag{1}
$$
 等价于其显示形式 $\cfrac{\mathrm{d}y}{\mathrm{d}x}=-\cfrac{P(x,y)}{Q(x,y)}=f(x,y)$
若存在可微函数 $F(x,y)$，使得它的全微分为 
$$
\mathrm{d}F(x,y)=P(x,y)\mathrm{d}x+Q(x,y)\mathrm{d}y
$$
 亦即偏导数为 
$$
\cfrac{∂F}{∂x}=P(x,y),\quad \cfrac{∂F}{∂y}=Q(x,y)
$$
 则称方程 (1) 为==恰当方程== (Exact equation) 或全微分方程。
那么原方程等价于 $\mathrm{d}F(x, y)=0$，即 
$$
F(x,y)=C
$$
 就是原方程隐式通解，$F(x,y)$称为==原函数==。

**判断恰当方程的充要条件**

<kbd>定理</kbd>：函数 $P(x,y)$ 和 $Q(x,y)$ 在区域 $R:a<x<b,c<y<d$ 上连续且有连续的一阶偏导数，则微分方程 (1) 是恰当方程的充要条件是 
$$
\cfrac{∂P}{∂y}=\cfrac{∂Q}{∂x}
$$
 必要性：若 (1) 式为恰当方程，则存在函数 $F(x,y)$，满足 $\cfrac{∂F}{∂x}=P(x,y),\cfrac{∂F}{∂y}=Q(x,y)$
则 $\cfrac{∂P}{∂y}=\cfrac{∂}{∂y}(\cfrac{∂F}{∂x})=\cfrac{∂^2F}{∂x∂y},\quad \cfrac{∂Q}{∂x}=\cfrac{∂}{∂x}(\cfrac{∂F}{∂y})=\cfrac{∂^2F}{∂y∂x}$
由P 和Q 具有连续一阶偏导数，可知上述混合偏导数连续且相等，因此必有 $\cfrac{∂P}{∂y}=\cfrac{∂Q}{∂y}$
充分性：设函数 $P(x,y)$ 和 $Q(x,y)$满足条件 $\cfrac{∂P}{∂y}=\cfrac{∂Q}{∂x}$
(1) 我们来构造函数 $F(x, y)$ 满足 $\cfrac{∂F}{∂x}=P(x,y),\quad \cfrac{∂F}{∂y}=Q(x,y)$，首先函数
$$
\displaystyle F(x, y)=\int P(x,y)dx+g(y)\tag{2}
$$
 函数$g(y)$待定，对任意 $g(y)$ 都满足 $\cfrac{∂F}{∂x}=P(x,y)$。
(2) 为确定 $g(y)$，计算第二个偏导数
$\displaystyle \cfrac{∂F}{∂y} =\cfrac{∂}{∂y}\int P(x,y)dx+g'(y)=Q$
所以 $\displaystyle g'(y)=Q-\cfrac{∂}{∂y}\int P(x,y)dx$
(3) 为了说明存在这样的函数 $g(y)$，仅与 $y$ 有关，与 $x$ 无关，只需证明 $g'(y)$ 关于 x 的导数恒为零。根据假设，有
$\begin{aligned}
\cfrac{∂}{∂x}\left[Q-\cfrac{∂}{∂y}\int P(x,y)dx\right] &= \cfrac{∂Q}{∂x}-\cfrac{∂}{∂x}\left[\cfrac{∂}{∂y}\int P(x,y)dx\right] \\
&= \cfrac{∂Q}{∂x}-\cfrac{∂}{∂y}\left[\cfrac{∂}{∂x}\int P(x,y)dx\right] \\
&= \cfrac{∂Q}{∂x}- \cfrac{∂P}{∂y} \equiv 0
\end{aligned}$
于是积分可得到 $\displaystyle g(y)=\int \left[Q-\cfrac{∂}{∂y}\int P(x,y)dx\right]dy$
带入可求得 
$$
\displaystyle F(x, y)=\int P(x,y)dx+\int \left[Q-\cfrac{∂}{∂y}\int P(x,y)dx\right]dy\tag{3}
$$
 因此满足条件的 (1) 式即为恰当方程，并且隐式通解为 
$$
\displaystyle \int P(x,y)dx+\int \left[Q-\cfrac{∂}{∂y}\int P(x,y)dx\right]dy=C\tag{4}
$$


**分项凑微分法求解**：往往在判断是恰当方程后，并不需要按照上述一般方法来求解，而是采用更简便的分项凑全微分的方法求解，这种方法要求熟记一些简单的全微分。
例如：求解方程 $(2x\sin y+3x^2y)dx+(x^3+x^2\cos y+3y^2)dy=0$
$\begin{aligned}
& (2x\sin y+3x^2y)dx+(x^3+x^2\cos y+3y^2)dy \\
=& (2x\sin y dx+x^2\cos y dy)+(3x^2ydx+x^3dy)+3y^2dy \\
=&(\sin y dx^2+x^2d\sin y)+(ydx^3+x^3dy)+3y^2dy \\
=&d(x^2\sin y)+d(x^3y)+d(y^3) \\
=&d(x^2\sin y+x^3y+y^3)
\end{aligned}$
所以通解为 $x^2\sin y+x^3y+y^3=C$

==部分全微分==
$ydx+xdy=d(xy)$
$\cfrac{ydx-xdy}{y^2}=d(\cfrac{x}{y})$
$\cfrac{ydx-xdy}{xy}=d(\ln\mid\cfrac{x}{y}\mid)$
$\cfrac{ydx-xdy}{x^2+y^2}=d(\arctan\cfrac{x}{y})$
$\cfrac{ydx-xdy}{x^2-y^2}=d(\ln\mid\cfrac{x-y}{x+y}\mid)$
$-\sin(x+y)(dx+dy)=d\cos(x+y)$
$\cos(x+y)(dx+dy)=d\sin(x+y)$

##  积分因子法

对于恰当方程，我们有多种方法求解，比如偏积分法、分项凑微分法等。因此，能否通过一些恒等变形，将非恰当方程化为恰当方程来求解呢？

首先，看一下前面讲的变量分离方程
$\cfrac{\mathrm{d}y}{\mathrm{d}x}=g(x)h(y) \iff g(x)h(y)\mathrm{d}x-\mathrm{d}y=0$
两端同乘非零函数 $μ(x,y)=\cfrac{1}{h(y)}$ 可得恰当方程 
$g(x)\mathrm{d}x-\cfrac{1}{h(y)}\mathrm{d}y=0$
然后再看下线性方程 
$\cfrac{\mathrm{d}y}{\mathrm{d}x}+P(x)y=Q(x) \iff [P(x)y-Q(x)]\mathrm{d}x+\mathrm{d}y=0$
两端同乘非零函数 $μ(x,y)=e^{\int P(x)dx}$ 可得恰当方程 
$M(x,y)\mathrm{d}x+N(x,y)\mathrm{d}y=0$
其中 
$M(x,y)=e^{\int P(x)dx}[P(x)y-Q(x)] ,N(x,y)=e^{\int P(x)dx}$
$\cfrac{∂M}{∂y}=P(x)e^{\int P(x)dx}=\cfrac{∂N}{∂x}$

现在我们尝试将这种方法一般化：**对一般的微分方程** 
$$
P(x,y)\mathrm{d}x+Q(x,y)\mathrm{d}y=0\tag{1}
$$
如果存在一个连续可微的非零函数 $μ=μ(x,y)$，使得 
$$
μP\mathrm{d}x+μQ\mathrm{d}y=0\tag{2}
$$
 为恰当方程，即 $\frac{∂(μP)}{∂y}=\frac{∂(μQ)}{∂x}$，则称 $μ(x,y)$ 为微分方程 (1) 的==积分因子==(integrating factor)。整理可得积分因子满足方程 
$$
Q\frac{∂μ}{∂x}-P\frac{∂μ}{∂y}=(\frac{∂P}{∂y}-\frac{∂Q}{∂x})μ\tag{3}
$$
 同一方程，可以有不同的积分因子，可以证明，只要方程有解存在，则必有积分因子存在，且不是唯一的。

一般情况下，求解方程 (3) 比求解微分方程 (1) 本身还要难！但是，在某些特殊情形，求解 (3) 还是可以实现的。

1. **如果存在一个只与 $x$ 有关的积分因子** $μ=μ(x)$ 则 (3) 式变为 
$$
Q\frac{dμ}{dx}=(\frac{∂P}{∂y}-\frac{∂Q}{∂x})μ
$$
 整理可得 
$$
\frac{1}{μ}\frac{dμ}{dx}=\frac{1}{Q}(\frac{∂P}{∂y}-\frac{∂Q}{∂x})\tag{4}
$$
它的左端只与 $x$ 有关，所以右端亦然，因此，方程 (1) 存在只与 $x$ 有关的积分因子的必要条件是：
$$
\frac{1}{Q}(\frac{∂P}{∂y}-\frac{∂Q}{∂x})=G(x)\tag{5}
$$
 只与 $x$ 有关。由此可得到积分因子 
$$
μ(x)=e^{\int G(x)dx}\tag{6}
$$


   <kbd>定理 1</kbd>：微分方程 (1) 存在仅依赖于 $x$ 的积分因子的充要条件是表达式 (5) 只与 $x$ 有关，而与 $y$ 无关。而且函数 (6) 就是一个积分因子。
类似的，我们得到下面的平行结果。
   <kbd>定理 2</kbd>：微分方程 (1) 存在仅依赖于 $y$ 的积分因子的充要条件是表达式 $\displaystyle\frac{1}{P}(\frac{∂Q}{∂x}-\frac{∂P}{∂y})=H(y)$ 只与 $y$ 有关，而与 $x$ 无关。而且函数 $μ(y)=e^{\int H(y)dy}$ 就是一个积分因子。

2. **分组求积分因子**

   <kbd>定理 3</kbd>：若 $μ=μ(x,y)$ 是微分方程 (1) 的一个积分因子，使得 
$$
μP(x,y)dx+μQ(x,y)dy=dΦ(x,y)
$$
 则 $μ(x,y)g[Φ(x,y)]$ 也是微分方程 (1)的一个积分因子，其中$g(\cdot)$是任意可微的（非零）函数。

   下面是对分组求积分因子的一般化说法。
   假设微分方程 (1) 的左端可以分成两组，即 
$$
(P_1dx+Q_1dy)+(P_2dx+Q_2dy)=0
$$
 其中第一组和第二组各有积分因子 $μ_1,μ_2$ 使得 
$$
μ_1(P_1dx+Q_1dy)=dΦ_1,\quad μ_2(P_2dx+Q_2dy)=dΦ_2
$$

由定理3可知，对任意可微函数$g_1,g_2$，函数 $μ_1g_1(Φ_1),μ_2g_2(Φ_2)$ 分别为第一、第二组的积分因子。因此，如果能适当选取 $g_1,g_2$，使得 $μ_1g_1(Φ_1)=μ_2g_2(Φ_2)$ ，则 $μ=μ_1g_1(Φ_1)$ 就是微分方程 (1)的积分因子。

例如，求解微分方程 $(x^3y-2y^2)dx+x^4dy=0$
将方程左端分组 $(x^3ydx+x^4dy)-2y^2dx=0$
前一组有积分因子 $x^{-3}$ 和通积分 $xy=C$；后一组有积分因子 $y^{-2}$ 和通积分 $y=C$
我们要寻找可微函数 $g_1,g_2$，使得 $x^{-3}g_1(x,y)=y^{-2}g_2(x)$
只要取$g_1(x,y)=\cfrac{1}{(xy)^2}, g_2(x)=\cfrac{1}{x^5}$
从而的到原方程得积分因子 $μ=\cfrac{1}{x^5y^2}$ 
即可化为全微分方程 $\cfrac{1}{(xy)^2}d(xy)-\cfrac{2}{x^5}dx=0$ 
积分此式，不难得到原方程通解 $y=\cfrac{2x^3}{2Cx^4+1}$，外加特解 $x=0,y=0$ 。他们实际上是用积分因子乘方程时丢失的解。

## 等角轨线族

假设在 $(x,y)$ 平面上由方程 
$$
Φ(x,y,C)=0\tag{1}
$$
 给出一个以C为参数的曲线族，我们设法求出另一个曲线族 
$$
Ψ(x,y,K)=0\tag{2}
$$
  其中K为参数，使得曲线族(2)中的任一条曲线与曲线族(1)中的每一条曲线相交成定角 $α (-\cfrac{π}{2}<α⩽\cfrac{π}{2})$，以逆时针方向为正，则称这样的曲线族(2)为已知曲线族(1)的 ==等角轨线族==(family of isogonal trajectories)，特别，当$α=\cfrac{π}{2}$时，称曲线族(2)为(1)的==正交轨线族==(family of orthogonal trajectories)。

方程(1)是一个单参数的曲线族，可以先求出它的每一条曲线满足的微分方程，再利用等角轨线的几何解释，得出等角轨线满足的微分方程，然后解此方程，即得所求的等角轨线族(2)。

具体来说，假设偏导 $Φ_C\neq0$，则可联立方程 
$$
\begin{cases}
Φ(x,y,C)=0 \\
Φ_x(x,y,C)dx+Φ_y(x,y,C)dy=0
\end{cases} \tag{3}
$$
 消去C，得到曲线族满足的微分方程 
$$
\cfrac{dy}{dx}=H(x,y)\tag{4}
$$
 其中 $H(x,y)=-\cfrac{Φ_x[x,y,C(x,y)]}{Φ_y[x,y,C(x,y)]}$，这里 $C=C(x,y)$ 是由 $Φ(x,y,C)=0$ 决定的函数。
如果我们把方程(4)在点 $(x,y)$ 的线素斜率记为$y'_1$，而把与它相交成角 $α$的线素斜率记为$y'$ ，则
(1) 当$α\neq\cfrac{π}{2}$时，有$\tan α=\cfrac{y'-y'_1}{1+y'y'_1}$ ，即 $y'_1=\cfrac{y'-\tan α}{1+y'\tan α}$
因为 $y'_1=H(x,y)$ ，所以等角轨线的微分方程为$H(x,y)=\cfrac{y'-\tan α}{1+y'\tan α}$ 即 
$$
\cfrac{dy}{dx}=\cfrac{H(x,y)+\tan α}{1-H(x,y)\tan α}
$$

(2) 而当$α=\cfrac{π}{2}$时，就有 $y'=-\cfrac{1}{y'_1}$ ，亦即可得正交轨线的微分方程为 
$$
\cfrac{dy}{dx}=-\cfrac{1}{H(x,y)}
$$

求解微分方程就可得到等角轨线族（正交轨线族）。

等角轨线族不仅在数学中有用，例如当 $0<α<π$ 时，可取为坐标系。而且在某些物理和力学中也有用，例如静电场中电场线和等势线互为正交轨线族。

## 一阶隐式微分方程

==一阶隐式微分方程==(First-order implicit differential equation) 的一般形式为 
$$
F(x,y,\frac{dy}{dx})=0 
$$
 所谓隐式的含义是指在方程中未知函数的导数$\frac{dy}{dx}$没有表示成 $(x, y)$ 的显函数。求解一阶隐式方程主要有两种方法：微分法和参数法，这两种方法的目的就是把隐式方程表示成显式方程来求解。

**微分法**：它主要针对 $y$ 被解出的方程（$x$ 被解出的方程同理）
$$
y=f(x,p)
$$
 其中 $p=\cfrac{dy}{dx}$，即 $dy=pdx$
假设函数 $f(x, p)$ 对 $(x, p)$ 是连续可微的， 那么对上式微分，便得到 
$$
dy=\cfrac{∂f}{∂x}(x,p)dx+\cfrac{∂f}{∂p}(x,p)dp
$$

从而得到关于 $x, p$ 的一阶显式微分方程 
$$
[\cfrac{∂f}{∂x}(x,p)-p]dx+\cfrac{∂f}{∂p}(x,p)dp=0
$$

求解方程可能出现以下情形：
(1) $p=φ(x,C)$ ，便得到通解 $y=f(x,φ(x,C))$，这是最理想的结果
(2) $x=φ(p,C)$ ，得参数方程表示的解 $\begin{cases} 
x=φ(p,C) \\
y=f(φ(x,C),p)
\end{cases}$
(3) 若得到关系式 $Φ(x,p,C)=0$ ，原方程的解只好表示为 $\begin{cases} 
Φ(x,p,C)=0 \\
y=f(x,p)
\end{cases}$

**参数法**：对于不明显包含自变量的方程
$$
F(y, p)=0
$$
 其中 $p=\cfrac{dy}{dx}$ 
由于函数 $F(y, p)=0$ 表示平面 $yOp$ 平面一条曲线或若干条曲线，而曲线都可以用参数方程表示，所以我们下面引入参数
$\begin{cases} 
y=g(t) \\
p=h(t)
\end{cases}$ ，带入 $p=\cfrac{dy}{dx}$ 可得 $dx=\cfrac{g'(t)}{h(t)}dt$
显然是可分离变量方程，可直接积分得到通解 
$$
\begin{cases} 
\displaystyle x=\int\cfrac{g'(t)}{h(t)}dt +C \\
y=g(t)
\end{cases}
$$


参数法对于方程 $F(x, p)=0$ 同样适用。

示例：求解微分方程 $y^2+(\cfrac{dy}{dx})^2=1$
显然，方程有参数表达式 $\begin{cases} 
y=\cos t \\
\cfrac{dy}{dx}=\sin t
\end{cases} (-∞<t<+∞)$
由此可得，$dx=\cfrac{(\cos t)'}{\sin t}-dt$
从而求得 $x=-t+C$，又因 $y=\cos t$
可得通解为 $y=\cos(C-x)$
除了上述参数形式的解外，还可设 $\begin{cases} 
y=\pm1 \\
\cfrac{dy}{dx}=0
\end{cases}$
可知 $y\pm1$也是方程的特解。积分曲线簇见下图：
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/parametric-method-demo.png)

**对于一般的一阶隐式微分方程** 
$$
F(x,y,p)=0\quad (p=\frac{dy}{dx})
$$
 在 $x,y,p$ 空间表示曲面，设它的参数方程为 $\begin{cases} 
x=f(u,v) \\
y=g(u,v) \\
p=h(u,v)
\end{cases}$ 
因为 $dy=pdx$ ，所以我们有 
$$
g_udu+g_vdv=h(u,v)(f_udu+f_vdv)
$$

可以写成如下形式 
$$
M(u,v)du+N(u,v)dv=0
$$
 其中 $\begin{cases} 
M(u,v)=g_u(u,v)-h(u,v)f_u(u,v) \\
M(u,v)=g_v(u,v)-h(u,v)f_v(u,v)
\end{cases}$
如果我们能求得上述一阶显式方程的通解 
$$
v=Q(u,C)
$$
则微分方程有通解 $\begin{cases} 
x=f[u,Q(u,C)] \\
y=g[u,Q(u,C)]
\end{cases}$
另外，如果显式方程除通解外，还有特解 $v=S(u)$
则微分方程有特解 $\begin{cases} 
x=f[u,S(u)] \\
y=g[u,S(u)]
\end{cases}$

# 解的存在和唯一性定理

里卡蒂方程的例子说明还有大量的方程的解不能用初等解法来求出通解的。而实际问题中所需要的往往是满足某种初始条件得解。因此对初值问题的研究被提到了重要的地位。自然要问，初值问题的解是否存在？若存在，是否唯一呢？

## 解的存在和唯一性定理

**(1) 首先考虑导数已求出的一阶显式微分方程初值问题** 
$$
\begin{cases}
\cfrac{dy}{dx}=f(x,y)\\
y(x_0)=y_0
\end{cases}\tag{E}
$$

<kbd>利普希茨条件</kbd>：存在常数 $L>0$ ，使得函数$f(x,y)$在区域D内满足不等式 
$$
|f(x,y_1)-f(x,y_2)|⩽L|y_1-y_2|
$$
则称函数 $f(x,y)$在区域D内对 $y$ 满足==利普希茨条件==(Lipschitz condition)，$L$ 称为利普希茨常数。

<kbd>定理 1</kbd>：==皮卡定理==(Picard theorem) ，如果函数 $f(x,y)$ 在矩形区域 
$$
R:|x-x_0|⩽a, |y-y_0|⩽b
$$
 内连续，且关于$y$ 满足利普希茨条件，则初值问题$(E)$在区间$|x-x_0|⩽h$存在唯一解。其中常数 
$$
h=\min\{a,\cfrac{b}{M}\},\displaystyle M=\max_{(x,y)\in R}|f(x,y)|
$$


我们采用皮卡(Picard)的逐步逼近法来证明这个定理。

   <img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/progressive-approaching-method.png" height="70%" width="70%">

- **命题 1**  设 $y=φ(x)$ 是初值问题 
$$
\begin{cases}
\cfrac{dy}{dx}=f(x,y)\quad &(1.1)\\
φ(x_0)=y_0\quad &(1.2)
\end{cases}\tag{E}
$$
 的解的充要条件是 $y=φ(x)$ 是积分方程 
$$
y=y_0+\int_{x_0}^xf(x,y)dx\quad(1.3)
$$
 的连续解。
   证明：因为$y=φ(x)$是方程 (1.1) 的解，故有 
$$
\cfrac{dφ(x)}{dx}\equiv f(x,φ(x))
$$

   两边从 $x_0$到$x$取定积分，有 
$$
φ(x)-φ(x_0)\equiv \int_{x_0}^xf(x,φ(x))dx
$$

   由于 $φ(x_0)=y_0$ ，带入可得 
$$
φ(x)=y_0+\int_{x_0}^xf(x,φ(x))dx
$$

   因此$y=φ(x)$是积分方程 (1.3) 的连续解。
   反之，只要设$y=φ(x)$是积分方程 (1.3) 的连续解，逆转上面的推导，就可得到$y=φ(x)$是初值问题(E)的解。

- 逐次迭代法构建==皮卡序列==(Picard sequence)
   **任取一个连续函数** $φ_0(x)$ 带入积分方程(1.3)右端的$y$，就得到函数 
$$
φ_1(x)=y_0+\int_{x_0}^xf(x,φ_0(x))dx
$$
 显然$φ_1(x)$也是连续函数，如果$φ_1(x)\equiv φ_0(x)$ ，那么$φ_0(x)$就是积分方程(1.3)的解（解的定义）；否则，我们又把$φ_1(x)$带入积分方程(1.3)的右端的$y$，得到 
$$
φ_2(x)=y_0+\int_{x_0}^xf(x,φ_1(x))dx
$$
 如果$φ_2(x)\equiv φ_1(x)$ ，那么$φ_1(x)$就是积分方程(1.3)的解；否则，我们继续这个步骤。一般的，我们作函数 
$$
φ_n(x)=y_0+\int_{x_0}^xf(x,φ_{n-1}(x))dx
$$
 这样，就得到连续函数序列 $φ_0(x),φ_1(x),\cdots,φ_n(x)$
如果$φ_{n+1}(x)\equiv φ_n(x)$ ，那么$φ_n(x)$就是积分方程(1.3)的解。

- **命题 2**  取$φ_0(x)=y_0$，构造皮卡序列 
$$
\begin{cases}
φ_0(x)=y_0 \\ 
\displaystyleφ_n(x)=y_0+\int_{x_0}^xf(x,φ_{n-1}(x))dx\quad(n=1,2,\cdots)
\end{cases}\tag{1.4}
$$
 序列中所有的函数$φ_n(x)$ 在区间 $|x-x_0|⩽h$ 上有定义、连续，且满足不等式 
$$
|φ_n(x)-y_0|⩽b
$$

证明：用数学归纳法
当 $n=1$ 时，$\displaystyleφ_1(x)=y_0+\int_{x_0}^xf(ξ,y_0)dξ$，显然 $φ_1(x)$在$|x-x_0|⩽h$ 上有定义、连续且有 
$$
\displaystyle|φ_1(x)-y_0|=|\int_{x_0}^xf(ξ,y_0)dξ|⩽\int_{x_0}^x|f(ξ,y_0)|dξ \\
⩽M|x-x_0|⩽Mh⩽b
$$
 即当 $n=1$ 时，命题2 成立。
现在用数学归纳法证明对于任何正整数 n ，命题2都成立。
假设当$n=k$时，$φ_k(x)$在$|x-x_0|⩽h$ 上有定义、连续且有
$$
|φ_k(x)-y_0|⩽b
$$

这时，$\displaystyleφ_{k+1}(x)=y_0+\int_{x_0}^xf(x,φ_k(x))dx$在$|x-x_0|⩽h$ 上有定义、连续，且
$$
\displaystyle|φ_{k+1}(x)-y_0|⩽\int_{x_0}^x|f(ξ,φ_k(x))|dξ ⩽M|x-x_0|⩽Mh⩽b
$$
 即命题２在 $n=k+1$时也成立。
由数学归纳法得知命题２对于所有 n 均成立。

-  **命题 3** 皮卡序列 $\{φ_n(x)\}$ 在$|x-x_0|⩽h$上是一致收敛的。
证明：我们考虑级数 
$$
\displaystyle φ_0(x)+\sum_{k=1}^{\infty}[φ_k(x)-φ_{k-1}(x)]\tag{1.5}
$$
 它的部分和为 
$$
\displaystyle φ_0(x)+\sum_{k=1}^{n}[φ_k(x)-φ_{k-1}(x)]=φ_n(x)
$$

因此，要证明$\{φ_n(x)\}$在$|x-x_0|⩽h$上一致收敛，只需证明级数(1.5)在$|x-x_0|⩽h$上一致收敛。
为此，我们进行如下估计：
$\displaystyle|φ_1(x)-φ_0(x)|⩽\int_{x_0}^x|f(ξ,φ_0(ξ))|dξ ⩽M|x-x_0|$
由(1.4)及李普希兹条件，我们有
$\begin{aligned}\displaystyle
|φ_2(x)-φ_1(x)|&⩽\int_{x_0}^x|f(ξ,φ_1(ξ))-f(ξ,φ_0(ξ))|dξ \\
&⩽L\int_{x_0}^x|φ_1(ξ)-φ_0(ξ)|dξ \\
&⩽L\int_{x_0}^xM|x-x_0|dξ=\cfrac{ML}{2!}|x-x_0|^2
\end{aligned}$
设对于正整数n，不等式 
$$
|φ_n(x)-φ_{n-1}(x)|⩽\cfrac{ML^{n-1}}{n!}|x-x_0|^n
$$
 成立。由李普希兹条件，在$|x-x_0|⩽h$上时
$$
\begin{aligned}\displaystyle
|φ_{n+1}(x)-φ_n(x)|&⩽\int_{x_0}^x|f(ξ,φ_n)(ξ)-f(ξ,φ_{n-1}(ξ)|dξ \\
&⩽L\int_{x_0}^x|φ_n(ξ)-φ_{n-1}(ξ)|dξ \\
&⩽\cfrac{ML^n}{n!}\int_{x_0}^x|x-x_0|^ndξ
=\cfrac{ML^n}{(n+1)!}|x-x_0|^{n+1}
\end{aligned}
$$

由数学归纳法：对于所有的正整数 k，在区间$|x-x_0|⩽h$有如下的估计
$|φ_k(x)-φ_{k-1}(x)|⩽\cfrac{ML^{k-1}}{k!}|x-x_0|^k$
因此，当$|x-x_0|⩽h$时，有 $|φ_k(x)-φ_{k-1}(x)|⩽\cfrac{ML^{k-1}}{k!}h^k$
上式右端是正项级数 $\displaystyle\sum_{\infty}^{k=1}\cfrac{ML^{k-1}}{k!}h^k$ 的一般项。
通过比值判别法
$[\cfrac{ML^k}{(k+1)!}h^{k+1}]/[\cfrac{ML^{k-1}}{k!}h^k]=\cfrac{Lh}{k+1}$
当$n\to\infty$时，$\cfrac{Lh}{k+1}\to0$
我们知道此级数时收敛的。
由维尔斯特拉斯(Weierstrass)判别法，我们知道级数(1.5)在$|x-x_0|⩽h$上一致收敛，因此皮卡序列 $\{φ_n(x)\}$ 在$|x-x_0|⩽h$上是一致收敛的。
    现设
$$
\lim\limits_{n\to}φ_n(x)=φ(x)
$$
 则$φ(x)$也在$|x-x_0|⩽h$上连续，且由命题2的结论可知，
$$
|φ(x)-y_0|⩽b
$$


-  **命题 4**   $φ(x)$ 是积分方程 (1.3) 在$|x-x_0|⩽h$上是的连续解。
证明：由利普希兹条件 $\displaystyle|f(x,φ_{n}(x))-f(x,φ(x))|⩽L|φ_n(x)-φ(x)|$ 
以及 $\{φ_n(x)\}$ 在$|x-x_0|⩽h$上是一致收敛于 $φ(x)$ 
即知序列 $\{f(x,φ_n(x))\}$ 在$|x-x_0|⩽h$上是一致收敛于 $f(x,φ(x))$ ，因而对皮卡序列(1.4)两边取极限，得到 
$$
\displaystyle\begin{aligned}
\lim\limits_{n\to\infty}φ_n(x)&=y_0+\lim\limits_{n\to\infty}\int_{x_0}^xf(ξ,φ_{n-1}(ξ))dξ \\
&=y_0+\int_{x_0}^x\lim\limits_{n\to\infty}f(ξ,φ_{n-1}(ξ))dξ
\end{aligned}
$$

即 $φ(x)=y_0+\int_{x_0}^xf(ξ,φ(ξ))dξ$，这就是说$φ(x)$是积分方程(1.3)在$|x-x_0|⩽h$上的解。

-  **命题 5** 若$ψ(x)$ 也是积分方程 (1.3) 在$|x-x_0|⩽h$上是的连续解，则 $ψ(x)\equivφ(x)\quad(|x-x_0|⩽h)$。
证明：我们首先证明$ψ(x)$也是序列$\{φ_n(x)\}$ 的一致收敛极限函数。
$$
ψ(x)\equiv y_0+\int_{x_0}^xf(ξ,ψ(ξ))dξ
$$

从 $φ_0(x)=y_0$ 开始进行如下的估计
$\displaystyle|φ_0(x)-ψ(x)|⩽\int_{x_0}^x|f(ξ,ψ(x))|dξ ⩽M|x-x_0|$
$\begin{aligned}\displaystyle
|φ_1(x)-ψ(x)|&⩽\int_{x_0}^x|f(ξ,φ_0(ξ))-f(ξ,ψ(ξ))|dξ \\
&⩽L\int_{x_0}^x|φ_0(ξ)-ψ(ξ)|dξ \\
&⩽L\int_{x_0}^xM|x-x_0|dξ=\cfrac{ML}{2!}|x-x_0|^2
\end{aligned}$
现设 $|φ_{n-1}(x)-ψ(x)|⩽\cfrac{ML^{n-1}}{n!}|x-x_0|^n$ ，则有
$$
\begin{aligned}\displaystyle
|φ_{n}(x)-ψ(x)|&⩽\int_{x_0}^x|f(ξ,φ_{n-1})(ξ)-f(ξ,ψ(ξ)|dξ \\
&⩽L\int_{x_0}^x|φ_{n-1}(ξ)-ψ(ξ)|dξ \\
&⩽\cfrac{ML^n}{n!}\int_{x_0}^x|x-x_0|^ndξ
=\cfrac{ML^n}{(n+1)!}|x-x_0|^{n+1}
\end{aligned}
$$

由数学归纳法：对于所有的正整数 k，在区间$|x-x_0|⩽h$有如下的估计
$$
|φ_k(x)-ψ(x)|⩽\cfrac{ML^{k}}{(k+1)!}|x-x_0|^{k+1}
$$

因此，当$|x-x_0|⩽h$时，有 $|φ_k(x)-ψ(x)|⩽\cfrac{ML^{k}}{(k+1)!}h^{k+1}$
上式右端是正项收敛级数 $\displaystyle\sum_{\infty}^{k=1}\cfrac{ML^{k}}{(k+1)!}h^{k+1}$ 的一般项。
因而，序列$\{φ_n(x)\}$在$|x-x_0|⩽h$上一致收敛于$ψ(x)$，根据极限的唯一性，即知在区间 $|x-x_0|⩽h$上 
$$
φ(x)\equiv ψ(x)
$$

**综合命题1-5，我们就证明了解的存在唯一性定理**(existence and uniqueness of solution) 。

- **近似计算和误差估计**：上述证明过程，我们用到的一步一步求出方程的解的方法，称为==逐步逼近法==。由方程(1.4)确定的函数 $φ_n(x)$称为初值问题 (E) 的==第n次近似解== ，同时还得到了误差估计公式 
$$
|φ_n(x)-φ(x)|⩽\cfrac{ML^{n}}{(n+1)!}h^{n+1}
$$


-  **附注 1**  存在唯一性定理中数 $h$ 的几何意义（见下图）
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/h-value.png)
图中 $h=\min\{a,\cfrac{b}{M}\}=\cfrac{b}{M}$，定理证明方程 (1.1) 过点$(x_0,y_0)$ 的积分曲线 $y=φ(x)$ 在区间$|x-x_0|⩽h$上确定。
因为积分曲线满足$Δ_h:|φ(x)-y_0|⩽M|x-x_0|$，积分曲线斜率介于$M$和$-M$之间。所以，当$|x-x_0|⩽h$时，$|φ(x)-φ(x_0)|=|φ(x)-y_0|⩽M|x-x_0|⩽b$。
也就是说积分曲线夹在R内的一个三角区域$Δ_h$之中，命题2中所有的函数 $y=φ_n(x)$都可在$|x-x_0|⩽h$上确定，它的图形都夹在三角区域$Δ_h$之中，自然，它的极限图形（积分曲线)也在其中。

- **附注 2** 由于利普希兹条件比较难于检验，常用 $f(x,y)$ 在$R$上的偏导数来代替。如果在闭矩形域 $R$ 上 $f_y(x,y)$存在且连续，则$f(x,y)$ 在$R$ 关于 $y$ 满足利普希兹条件。
证明：设$f_y(x,y)$ 在  $R$上连续，则在  $R$上有界，记为 $L$。
由中值定理 $|f(x,y_1)-f(x,y_2)|=|f_y(x,ξ)|\cdot|y_1-y_2|⩽L|y_1-y_2|$ ，其中 $ξ$ 介于 $y_1,y_2$ 之间。
但反过来，满足利普希兹条件的函数$f_y(x,y)$不一定有偏导数存在。

- **附注 3** 设方程 (1.1) 的是线性的，即方程为 
$$
\cfrac{dy}{dx}=P(x)y+Q(x)
$$
 那么容易知道，当$P(x), Q(x)$ 在区间 $[α,β]$ 上为连续时，则由任一初始值 $(x_0,y_0),x_0\in[α,β]$ 所确定的解在整个区间$[α,β]$内都存在且唯一。
对于一般方程 (1.1) ，由初始值确定的解只能定义$|x-x_0|⩽h$。而上述方程，右端函数对$y$没有任何限制，为了证明我们的结论，譬如取$\displaystyle M=\max_{x\in [α,β]}|P(x)y_0+Q(x)|$，而逐字重复定理 1的证明过程，即可证由 (1.4) 确定的函数序列$\{φ_n(x)\}$在整个区间$[α,β]$上都有定义和一致连续。


**(2) 再考虑一阶隐式微分方程** 
$$
F(x,y,y')=0\tag{2.1}
$$

<kbd>定理 2</kbd>如果在点 $(x_0,y_0,y'_0)$ 的某邻域中：
(i) $F(x,y,y')$对所有变元连续且具有一阶连续偏导数
(ii) $F(x_0,y_0,y'_0)=0$ 
(iii) $\cfrac{∂F}{∂y'}(x_0,y_0,y'_0)\neq 0$
则微分方程 (2.1) 存在唯一的解 
$$
y=y(x),\quad |x-x_0|⩽h
$$
 （其中$h$为足够小的正数）满足初始条件 $y(x_0)=y_0,\quad y'(x_0)=y'_0$ 
证明：根据隐函数存在定理[^a]，可把$y'$ 唯一的表示为$x,y$的函数 $y'=f(x,y)$ ，得到一个显示微分方程.
并且$f(x,y)$在点$(x_0,y_0)$的某一邻域内连续，且满足$y'_0=f(x_0,y_0)$
函数$f(x,y)$对$x,y$存在一阶连续偏导数 $\cfrac{∂f(x,y)}{∂y}=-\cfrac{∂F}{∂y}/\cfrac{∂F}{∂y'}$
我们可以在邻域中做一个闭的矩形域， 显然是有界的。根据定理 1，方程$y'=f(x,y)$ 满足初始条件 $y(x_0)=y_0$的解存在且唯一。即方程(2.1)过点 $(x_0,y_0)$且切线斜率为 $y'_0$ 的解或积分曲线存在且唯一。

[^a]: 设函数$F(x,y,z)$在点 $(x_0,y_0,z_0)$ 的某邻域中：
(i) $F(x,y,z)$对所有变元连续且具有一阶连续偏导数
(ii) $F(x_0,y_0,z_0)=0$ 
(iii) $\cfrac{∂F}{∂z}(x_0,y_0,z_0)\neq 0$
则方程 $F(x,y,z)=0$在点$(x_0,y_0,z_0)$的某邻域内恒能唯一确定一个单值连续且具有连续偏导数的函数 $z=f(x,y)$ ，它满足初始条件 $z_0=f(x_0,y_0)$ 并有 $\cfrac{∂z}{∂x}=-\cfrac{∂F}{∂x}/\cfrac{∂F}{∂z},\quad \cfrac{∂z}{∂y}=-\cfrac{∂F}{∂y}/\cfrac{∂F}{∂z}$

> 也可理解为，对于任意给定的一组值 $(x_0,y_0,y'_0)，F(x_0,y_0,y'_0)=0$，方程(2.1)沿给定方向$y'_0$通过点$(x_0,y_0)$的积分曲线有且只有一条。

**(3) Osgood条件**：一般而言，满足利普希茨条件只是解的唯一性的充分条件，而非充要条件。下面我们介绍一种比较弱的条件：
设函数$f(x,y)$在区域G内连续，而且满足不等式 
$$
|f(x,y_1)-f(x,y_2)|⩽ F(|y_1-y_2|)
$$
 其中 $F(r)>0$是$r>0$的连续函数，而且积分 
$$
\displaystyle\int_0^{r_1}\cfrac{dr}{F(r)}=\infty
$$
 （$r_1>0$为常数）。则称$f(x,y)$在G内对$y$满足==Osgood条件==。
    普希茨条件是Osgood条件的 特例，这是因为$F(r)=Lr$满足上述要求。
<kbd>定理 3</kbd> 设$f(x,y)$在G内对$y$满足Osgood条件，则微分方程 $\cfrac{dy}{dx}=f(x,y)$ 在G内经过每一点的解都是唯一的。
证明略。

## 解的延拓

对于定义在矩形域 $R:|x-x_0|⩽a, |y-y_0|⩽b$ 上的初值问题 
$$
\begin{cases}
\cfrac{dy}{dx}=f(x,y)\quad &(1)\\
y(x_0)=y_0\quad &(2)
\end{cases}\tag{E}
$$
解的存在唯一性定理是局部性的，当 $f(x,y)$满足一定的条件时，它只肯定了解至少在区间$|x-x_0|⩽h$ 存在唯一解，其中常数 $h=\min\{a,\cfrac{b}{M}\},\displaystyle M=\max_{(x,y)\in R}|f(x,y)|$ 。
本节准备把这种讨论扩大到整体。

<kbd>局部李普希兹条件</kbd>：函数 $f(x,y)$在某一区域G内连续，对于区域G内每一点P，都有以P为中心完全含于G内的闭矩形R存在，使得在R上 $f(x,y)$ 关于$y$满足李普希兹条件（对于不同的P，域R的大小和常数L可能不同），称 $f(x,y)$ 关于y满足==局部李普希兹条件== (Local Lipschitz condition)。

假设方程 (1) 右端函数 $f(x,y)$ 关于y满足局部李普希兹条件，初值问题$E$的解 $y=φ(x)$ 已定义在区间 $|x-x_0|⩽h$ 上，取 $x_1=x_0+h,y_1=φ(x_1)$，然后以 $(x_1,y_1)$ 为中心作一小矩形 $R_1\sub G$ ，则过点$(x_1,y_1)$的初值问题存在唯一解 $y=ψ(x)$，解的存在唯一区间为 $|x-x_1|⩽h_1$ 
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/local-Lipschitz-condition.png)
因为 $φ(x_1)=ψ(x_1)$ ，由唯一性定理，在解重叠的部分 $x_1-h_1⩽x⩽x_1$ 时有  $φ(x)\equiv ψ(x)$，但是在区间 $x_1⩽x⩽x_1+h_1$上，函数$y=ψ(x)$仍有定义，我们把它看成是定义在原来区间$|x-x_0|⩽h$上解$y=φ(x)$向右方的==延拓==(prolongement)。这样我们可以确定方程的解 $y=\begin{cases}
φ(x), & x_0-h_0⩽x⩽x_0+h_0\\
ψ(x), & x_0+h_0⩽x⩽x_1+h_1
\end{cases}$  。用同样的方法可把解 $y=φ(x)$ 向左方延拓，几何上就相当于在原来的积分曲线 $y=φ(x)$ 左右两端各接上一个积分曲线段。
以上这种把曲线向左右两方延拓的步骤可一次一次地进行下去，直到无法延拓为止，这样的解称为==饱和解==。任一饱和解 $y=\tilde φ(x)$的==最大存在区间==(maximum iinterval of existence)必定是一个开区间 $α<x<β$ 。因为如果这个区间的右端是封闭的，那么 β 便是有限数，且点 $(β,\tilde φ(β))\in G$ ，这样解就还能继续向右方延拓，从而是非饱和的。对左端 α 点可同样讨论。

<kbd>解的延拓定理</kbd>：如果方程(1)右侧函数$f(x,y)$在有界区域G中连续，且在G内关于$y$满足局部李普希兹条件，那么方程(1)通过G内任一点$(x_0,y_0)$的解$y=φ(x)$可以延拓，直到点$(x,φ(x))$任意接近G的边界。
以向 $x$ 增大一方的延拓来说， 如果 $y=φ(x)$ 只延拓到区间$x_0⩽x<m$上，则当 $x\to m$时， $(x,φ(x))$趋于区域 G 的边界。

<kbd>推论</kbd>：如果 G 是无界区域，在上面延拓定理条件下，方程(1)的通过点$(x_0,y_0)$的解$y=φ(x)$可以延拓，以向$x$增大一方的延拓来说，有下面的两种情况:
(1) 解$y=φ(x)$可以延拓到区间 $[x_0,+\infty)$
(2) 解$y=φ(x)$只可以延拓到区间 $[x_0,m)$，其中m为有限数，则当 $x\to m$时， 或者$y=φ(x)$无界，或者点$(x,φ(x))$趋于区域 G 的边界。

**示例**：讨论方程$\cfrac{dy}{dx}$通过点 $(\ln 2,-3)$ 的解存在区间。
解：该方程右侧函数定义在整个$xOy$平面上且满足解的存在唯一性定理及解的延拓定理条件，其通解为 $y=\cfrac{1+ce^x}{1-ce^x}$
故通过点 $(\ln 2,-3)$ 的解为 $y=\cfrac{1+e^x}{1-e^x}$
这个解的存在区间为 $(0,+\infty)$
如图，通过点$(\ln 2,-3)$ 的解向右可延拓到 $+∞$，但向左只能延拓到0，因为，当 $x\to0$时，$y\to+∞$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/life-span-of-solution-demo.png)

> 应用上述定理推论的结果不难证明：如果函数 $f(x,y)$ 在整个 $xOy$平面上有定义、连续和有界，同时存在关于 $y$ 的一阶连续偏导数，则方程 (1) 的解可以延拓到区间$(-∞,+∞)$

## 解对初值与参数的连续性和可微性

考虑初值问题 $y'=ay,y(x_0)=y_0$，易求出其解为 $y=y_0e^{a(x-x_0)}$ 。观察可知，这个解与 $x,x_0,y_0$ 及参数 $a$ 有关，因此可以把这个解记作 $\varphi(x;x_0,y_0;a)$，易见$\varphi$ 关于他的每个自变量都是连续且可微的，即每个一阶偏导数都存在且连续。
微分方程的解不仅决定于微分方程本身，而且也依赖于初值和参数。

考虑含有参数的一般微分方程
$$
\cfrac{dy}{dx}=f(x,y,λ)\tag{1}
$$
满足初始条件 $y(x_0)=y_0$ 的解记为 $y=φ(x;x_0,y_0;λ)$ 。

**解关于初值的对称性**：在表达式 $y=φ(x;x_0,y_0;λ)$ 中 $(x_0,y_0)$ 与 $(x,y)$ 可以调换其相对位置，即在解的存在范围内 $y_0=\varphi(x_0;x,y;λ)$ 。
在上述解的存在区间内任取一值 $x_1$，记$y_1=φ(x_1;x_0,y_0;λ)$ 。由解的唯一性知，过点 $(x_1,y_1)$ 与过点 $(x_0,y_0)$ 的积分曲线相同，因此此解也可写为 $y=φ(x;x_1,y_1;λ)$ ，显然有 $y_0=φ(x_0;x_1,y_1;λ)$ 。注意到点$(x_1,y_1)$的任意性，因此关系式 $y_0=φ(x_0;x,y;λ)$ 成立。

**解对初值和参数的连续依赖性**

<kbd>引理</kbd>：如果 $f(x,y)$ 在区域 D 内连续，且关于 $y$ 满足李普希兹条件（李普希兹常数为 $L$），则方程 $y'=f(x,y)$ 任意两个解 $\varphi(x),\psi(x)$ ，在对他们公共存在区间 $a ⩽x⩽ b$ 内的某一值$x_0$ 不等式成立 
$$
|\varphi(x)-\psi(x)|⩽|\varphi(x_0)-\psi(x_0)|e^{L|x-x_0|}
$$
证明：令 $V(x)=|\varphi(x)-\psi(x)|\quad x\in[a,b]$ 
则 $V'(x)=|f(x,\varphi)-f(x,\psi)|⩽L|\varphi(x)-\psi(x)|=LV(x)$
于是 $\cfrac{d(V(x)e^{-Lx})}{dx}⩽0$
 $V(x)e^{-Lx}⩽V(x_0)e^{-Lx_0}\quad x_0⩽x⩽b$
即 $V(x)⩽V(x_0)e^{L(x-x_0)}$
对于区间 $a⩽x⩽x_0$ ，令 $-x=t,-x_0=t_0$，类似上述过程
可得 $V(x)⩽V(x_0)e^{L(x_0-x)}$
因此 $V(x)⩽V(x_0)e^{L|x-x_0|}\quad x_0\in[a,b]$

<kbd>定理 1</kbd> （解对初值和参数的连续依赖定理）
（条件）：设 $f(x,y,λ)$ 在域 $G_λ:(x,y)\in G,α<λ<β$ 内连续，且在 $G_λ$ 内一致的关于 $y$ 满足局部李普希兹条件。
（前提）：$(x_0,y_0,λ_0)\in G_λ,y=φ(x;x_0,y_0;λ_0)$ 是方程(1) 通过点 $(x_0,y_0)$ 的解，在区间 $a ⩽ x ⩽ b$ 上有定义（$a ⩽x_0⩽ b$）。
（结论）：那么，对任意 $ϵ>0$ ，存在正数 $δ=δ(ϵ,a,b)$ ，使得当
$$
(\bar x_0-x_0)^2+(\bar y_0-y_0)^2+(λ-λ_0)^2⩽δ^2
$$
成立时，方程 (1) 通过点 $(\bar x_0,\bar y_0)$ 的解 $y=φ(x;\bar x_0,\bar y_0;λ)$ 在区间$a ⩽ x ⩽ b$也有定义，并且满足
$$
|φ(x;\bar x_0,\bar y_0;λ)-φ(x;x_0,y_0;λ_0)|<ϵ
$$
>  上述定理用极限的写法为
> $$
> \lim\limits_{(\bar x_0,\bar y_0,λ)\to(x_0,y_0,λ_0)}φ(x;\bar x_0,\bar y_0;λ)=φ(x;x_0,y_0;λ_0)\quad x\in[a,b]
> $$
> 即定义于闭区间上的积分曲线（一段闭弧）连续依赖于初值点和参数。

<kbd>定理 2</kbd> （解对初值和参数的连续性定理）：若函数 $f(x,y,λ)$ 在域 $G_λ$ 内连续，且在 $G_λ$ 内一致的关于 $y$ 满足局部李普希兹条件。则方程 (1) 的解 $y=φ(x;x_0,y_0;λ)$ 作为 $x,x_0,y_0,λ$ 的函数，在他们的存在区间内连续。

**解对初值和参数的可微性**：即解 $y=φ(x;x_0,y_0;λ)$ 关于初值和参数 $x_0,y_0,λ_0$ 的偏导数的存在性和连续性。

<kbd>定理 3</kbd> （解对初值和参数的可微性定理）：若函数 $f(x,y,λ)$ 及 $\cfrac{∂f}{∂y},\cfrac{∂f}{∂λ}$ 在域 $G_λ$ 内连续，则方程 (1) 的解 $y=φ(x;x_0,y_0;λ_0)$ 作为 $x,x_0,y_0,λ_0$ 的函数，在他的存在范围内连续可微。

> 微分方程解的光滑性不亚于微分方程（右端函数）的光滑性。

## 奇解和包络

我们已经看到某些一阶隐式微分方程可能存在不能包含于通解中的特解，即不能通过确定通解中的任意常数来得到这个特解，这个特解还具有特殊的几何意义。譬如，微分方程 $y^2+(\cfrac{dy}{dx})^2=1$ 通解为 $y=\cos(C-x)$，还有特解 $y=\pm1$，显然两个特解不能包含于通解之中。

<kbd>奇解</kbd>：设微分方程
$$
F(x,y,\cfrac{dy}{dx})=0\tag{1}
$$
 有一特解 
$$
Γ:y=φ(x)\quad (x\in J)
$$
 如果对于每一个点 $Q\in Γ$，在Q 点的任意邻域内都有一个不同于 $Γ$ 的解在Q 处与 $Γ$ 相切，则称 $Γ$ 是微分方程的==奇解==(singular solution)。

> 一般说来，求一阶微分方程式的奇解有两种方法，分别称为p-判别式法和C-判别式法，两种方法的区别在于p-判别式法不用求方程的通解而直接根据方程的表达式求出奇解，应用起来比较方便；C-判别式法需要先知道方程的通解，再通过寻找通解对应的曲线族的包络来获得奇解，相比于第一种方法，C-判别式法更复杂一点。

下面定理给出了奇解存在的必要条件。

<kbd>定理 1</kbd>：设函数 $F(x, y, p)$ 对 $(x, y, p)\in G$ 是连续的，而且对 $y,p$ 的偏导数， $F_y,F_p$ 也是连续的。那么，若 $y=φ(x)$ 是微分方程 (1) 在区间 $J$ 上的一个奇解，且对 $(x,φ(x),φ'(x))\in G\quad(x\in J)$，则奇解  $y=φ(x)$ 满足联立方程
$$
\begin{cases} 
F(x, y, p)=0\\
F_p(x, y, p)=0
\end{cases}\quad(p=\cfrac{dy}{dx})
$$
称为 ==p-判别式==。
若从p-判别式中消去p得到方程 $\Delta(x,y)=0$，则称由此所决定的曲线为微分方程(1)的==p-判别曲线==。因此，微分方程的奇解是一条p-判别曲线。

证明：设 $y=φ(x)$ 是微分方程 (1) 的奇解，它自然满足 p-判别式的第一式，即 $F(x, y, p)=0$，现证它也满足第二式。
假设不然，则必存在一点 $x_0\in J$使得 $F_p(x_0, y_0, p_0)\neq0$，其中$y_0=φ(x_0),p_0=φ'(x_0)$
对于 $(x_0, y_0, p_0)\in G$ 我们有 $F(x_0, y_0, p_0)=0$
于是，根据隐函数存在定理[^a]，可知 $F(x, y, p)=0$ 可以在 $(x_0,y_0)$ 的某邻域内确定一个隐函数 $p=f(x, y)$，其中 $f(x, y)$ 满足
(i) $f(x_0, y_0)=p_0$
(ii) $\cfrac{∂f(x,y)}{∂y}=-\cfrac{F_y(x,y,f(x,y))}{F_p(x,y,f(x,y))}$
一方面：函数 $p=f(x, y)$ 等价于$φ'(x)=f(x,φ(x))$，这说明奇解 $y=φ(x)$ 是初值问题 $\begin{cases} 
\cfrac{dy}{dx}=f(x, y)\\
y(x_0)=y_0
\end{cases}$ 的解。
另一方面，条件 $F(x, y, p)$ 在G 连续可推出 $f(x, y)$ 在 $(x_0,y_0)$ 的某邻域内连续；条件 $F_y,F_p$ 也是连续，再结合 (ii) 式，可知 $\cfrac{∂f}{∂y}(x,y)$ 也在 $(x_0,y_0)$ 的某邻域内连续。
最后利用一阶微分方程基本理论中解的存在唯一性定理可知上述初值问题有唯一解，也就是说微分方程(1)不可能有其它解在该点与$y=φ(x)$ 相切。这与 $y=φ(x)$ 是奇解的假设相矛盾，因此 $y=φ(x)$满足 p-判别式的第二式，证毕。
> 这里必须注意：定理 1只提供了奇解存在的必要条件，满足p-判别式的解不一定是奇解。
> 例如，微分方程 $(\cfrac{dy}{dx})^2-y^2=0$ 的 p-判别式为 $\begin{cases} 
> p^2-y^2=0\\
> 2p=0
> \end{cases}$ 消去 $p$ 得到 $y=0$，容易验证它是方程的解。但是容易求出该方程的通解是 $y=Ce^{\pm x}$，可见 $y=0$ 不是奇解。

为了确认满足 p-判别式的函数是否为奇解，我们奇解存在的充分条件。

<kbd>定理 2</kbd>：设函数 $F(x, y, p)$ 对 $(x, y, p)\in G$ 是二阶连续可微的。
再设微分方程(1)的 p-判别式 $\begin{cases} 
F(x, y, p)=0\\
F_p(x, y, p)=0
\end{cases}$ （消去 $p$ 后）得到的函数 $y=φ(x), x\in J$ 是微分方程(1)的解。
而且对所有的 $x\in J, y=φ(x)$ 满足下面三个条件 
$$
\begin{aligned} 
&F_y(x, φ(x), φ'(x))\neq 0\\
&F_{pp}(x, φ(x), φ'(x))\neq 0 \\
&F_p(x, φ(x), φ'(x))= 0
\end{aligned}
$$
 则 $y=φ(x)$ 是微分方程(1)的奇解。
这个定理的证明有一定难度，而且已经超出一般微分方程大纲的范围，有兴趣的同学可以参阅丁同仁、李承治编写的《常微分方程教程》

**例如**，考虑微分方程 $[(y-1)\cfrac{dy}{dx}]^2=ye^{xy}$ 的奇解。
这里的p-判别式为 $\begin{cases} 
(y-1)^2p^2-ye^{xy}=0\\
2p(y-1)^2=0
\end{cases}$ ，从而得到函数 $y=0$ 。
易知 $y=0$ 是原方程的解，而且满足
$F_y(x, 0, 0)=-1\neq 0\\
F_{pp}(x,0, 0)=2\neq 0 \\
F_p(x, 0, 0)= 0$
由定理2可知$y=0$是原方程的奇解。

<kbd>包络</kbd>对给定的单参数 $C$ 的曲线族： 
$$
\Phi(x,y,C)=0\tag{2}
$$
 其中函数$\Phi(x,y,C)$对$x,y,C$连续可微。设有一条连续可微的曲线$Γ$，它本身并不包含在曲线族(2)中，过曲线$Γ$的每一点都有曲线族中的一条曲线和它在此点相切，则称曲线$Γ$为此曲线族的==包络==(envelope)。
> 并非任何一个曲线族都有包络，例如同心圆族 $x^2+y^2=C^2$，其中 $C$ 为参数，就没有包络。

由奇解的定义可知：对于一阶隐式微分方程 $F(x,y,y')=0$，它的通解的包络一定是方程的奇解；反之亦成立。可通过求出方程的积分曲线族，再求出曲线族包络得到方程的奇解。

<kbd>定理 3</kbd>：若曲线$Γ$是曲线族 $\Phi(x,y,C)=0$的包络，且$\Phi(x,y,C)$对所有的变量具有一阶连续偏导数，则曲线 $Γ$ 满足如下==C-判别式== $\begin{cases} 
\Phi(x,y,C)=0\\
\Phi_C(x,y,C)=0
\end{cases}$ 
若从C-判别式中消去C可得到==C-判别曲线== $\Omega(x,y)=0$

证明：设曲线$Γ$关于参数 $C$ 的参数方程为 
$$
Γ: x=x(C),y=y(C),C\in[α,β]
$$
 曲线上的点都在曲线族 $\Phi(x,y,C)=0$上，故有 
$$
\Phi(x(C),y(C),C)=0,C\in[α,β]
$$
 对参数 $C$求导得到 
$$
\Phi_x(C)x'(C)+\Phi_y(C)y'(C)+\Phi_C(C)=0
$$
 另一方面，曲线$Γ$与曲线族$\Phi(x,y,C)=0$ 在点 $(x(C),y(C))$ 相切，故 
$$
\cfrac{y'(C)}{x'(C)}=-\cfrac{\Phi_x(C)}{\Phi_y(C)}
$$
由此可以得到 $\Phi_C=0$

> 曲线族 $\Phi(x,y,C)=0$ 的包络必定包含在C-判别曲线中，但是C-判别曲线未必是包络。需要验证C-判别曲线是否为包络。

<kbd>定理 4</kbd>：设曲线族 $\Phi(x,y,C)=0$ 对所有的变量具有一阶连续偏导数，若曲线族的一条C-判别曲线 
$$
Γ: x=x(C),y=y(C),C\in[α,β]
$$
 连续可微，且沿该曲线有 
$$
(\Phi_x(x,y,C))^2+(\Phi_y(x,y,C))^2\neq 0
$$
 则C-判别曲线 $Γ$ 是曲线族的包络。
证明：在曲线 $Γ$ 上任取一点$(x(C),y(C))$，有 
$$
\begin{cases} 
\Phi(x(C),y(C),C)=0\\
\Phi_C(x(C),y(C),C)=0
\end{cases}
$$
 由此通过对 $\Phi(x(C),y(C),C)=0$ 求导可以得到
$$
\Phi_x(x(C),y(C),C)x'(C)+\Phi_y(x(C),y(C),C)y'(C)=0
$$
 由条件知偏导数 $\Phi_x,\Phi_y$不同时为0，不妨设 $\Phi_y\neq 0$ ，故有 
$$
\cfrac{y'(C)}{x'(C)}=-\cfrac{\Phi_x(x(C),y(C),C)}{\Phi_y(x(C),y(C),C)}
$$
 从而曲线$Γ$与曲线族$\Phi(x,y,C)=0$ 在点 $(x(C),y(C))$ 相切
即曲线$Γ$是曲线族$\Phi(x,y,C)=0$的包络。

**示例**：求直线族 $x\cos a+y\sin a-p=0$ 的包络，这里 a 是参数 p 是常数。
(1) 上式对 a 求导得到 $-x\sin a+y\cos a=0$
为了消去 a，可将上面两式平方得到 $\begin{cases} 
x^2\cos^2 a+y^2\sin^2 a+2xy\cos a\sin a=p^2\\
x^2\sin^2 a+y^2\cos^2 a-2xy\cos a\sin a=0
\end{cases}$
相加可得到C-判别曲线 $x^2+y^2=p^2$
(2) C-判别曲线连续可微，且 $(\cos a)^2+(\sin a)^2=1\neq 0$
则此曲线 $x^2+y^2=p^2$ 即为曲线族的包络。
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/envelope-demo.png)

**克莱罗(Clairaut)方程**
形如 
$$
y=xp+f(p)
$$
 的方程称为可莱罗方程，其中  $p=\cfrac{dy}{dx}, f''(p)\neq0$
利用微分法，我们得到 
$$
p=p+x\cfrac{dp}{dx}+f'(p)\cfrac{dp}{dx}
$$
 即 $[x+f'(p)]\cfrac{dp}{dx}=0$
(1) 当 $\cfrac{dp}{dx}=0$时，可得 $p=C$，带入原方程得通解 $y=Cx+f(C)$ ，为一直线族。
(2) 当 $x+f'(p)=0$时，可得特解的参数方程 $\begin{cases} 
x+f'(p)=0 \\
y=xp+f(p)
\end{cases}$ （p为参数）
注意到，$f''(p)\neq0$，所以 $x=-f'(p)$有反函数 $p=ω(x)$，代入原方程便得到特解 $y=xω(x)+f[ω(x)]$
另外，由于$ω''(x)=-1/f''(ω(x))\neq0$，所以 $ω(x)$ 不是常数，因此，此特解不能由通解给出。

上述特解的参数形式正好是通解求包络的C-判别式。可以验证克莱罗方程特解正是通解的包络，也是克莱罗方程的奇解。
作为例子，当 $f(p)=-\frac{1}{4}p^2$时，克莱罗微分方程的积分曲线簇见下图：
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/Clairaut-demo.png)


> **参考文献：**
> 丁同仁.《常微分方程教程》
> 王高雄.《常微分方程》
> 窦霁虹 付英.《常微分方程》.西北大学(MOOC) 
> 《高等数学》.国防科技大学(MOOC)
