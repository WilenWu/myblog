---
title: 常微分方程(Ordinary Differential Equation II)
date: 2020-04-23 14:52:22
categories: [数学,微分方程]
tags: [数学,ODE,微分方程]
cover: 
top_img: 
keywords: 
katex: true
description: 高阶常微分方程
---



# 高阶微分方程

**高阶微分方程**的一般形式为 
$$
F(x,y,y',\cdots,y^{(n)})=0
$$
 一般的高阶微分方程没有普遍的解法，处理问题的基本原则是降阶。

**高阶线性微分方程** 
$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y'+a_n(x)y=f(x)\tag{1}
$$
 称为n阶==非齐次线性方程==，其中 $a_i(x)(i=1,2,\cdots,n)$及$f(x)$都是区间 $a⩽x⩽b$ 上的连续函数。
当 $f(x)\equiv0$，则方程 (1) 变为 
$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y'+a_n(x)y=0\tag{2}
$$
称为对应于方程(1)的n阶==齐次线性方程==。

**高阶线性微分方程的初值条件**

$$
y(x_0)=η_1,y'(x_0)=η_2,\cdots,y^{(n-1)}(x_0)=η_n\tag{3}
$$


<kbd>定理 1 解的存在和唯一性定理</kbd>：如果$a_i(x)(i=1,2,\cdots,n)$及$f(x)$都是区间 $[a,b]$ 上的连续函数，则对任一 $x_0\in[a,b]$ 及任意的 $η_1,η_2,\cdots,η_n$ ，方程 (1) 满足初始条件 (3) 的解在区间 $[a,b]$ 上存在且唯一解。
这个定理是一阶线性方程在高阶线性方程的推广，关于定理的证明在后边学习线性方程组时得出。

## 高阶线性齐次方程

**高阶线性齐次方程** 
$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y'+a_n(x)y=0\tag{2}
$$

**解的性质**
(1) $y\equiv0$ 是线性齐次方程的解，称为方程的平凡解。
(2) 任意两个解之和是方程的解。
(3) 任一解的常数倍也是方程的解。

<kbd>定理 2 叠加原理</kbd>：若$y_1(x),y_2(x),\cdots,y_k(x)$是方程(2)的$k$个解，则他们的线性组合 $c_1y_1(x)+c_2y_2(x)+\cdots+c_ky_k(x)$ 也是方程(2)的解，其中$c_1,c_2,\cdots,c_k$是任意常数。
特别的，当$k=n$时，即方程有解 $φ(x)=c_1y_1(x)+c_2y_2(x)+\cdots+c_ny_n(x)$ ，它含有 n 个任意常数 $c_1,c_2,\cdots,c_n$ ，考虑将此解作为方程(2)的通解。
根据通解的定义，n 个任意常数相互独立，即雅克比行列式(Jacobian)满足  
$$
\cfrac{∂(φ,φ',\cdots,φ^{(n-1)})}{∂(c_1,c_2,\cdots,c_n)}=
\begin{vmatrix}
\frac{∂φ}{∂c_1} &\frac{∂φ}{∂c_2} & \cdots &\frac{∂φ}{∂c_n} \\ 
\frac{∂φ'}{∂c_1} &\frac{∂φ'}{∂c_2} &\cdots &\frac{∂φ'}{∂c_n} \\ 
\vdots &\vdots &\ddots &\vdots \\ 
\frac{∂φ^{(n)}}{∂c_1} &\frac{∂φ^{(n)}}{∂c_2} &\cdots &\frac{∂φ^{(n)}}{∂c_n} \\ 
\end{vmatrix}\neq 0
$$
  导数求解如下  
$$
φ(x)=c_1y_1(x)+c_2y_2(x)+\cdots+c_ny_n(x) \\
φ'(x)=c_1y'_1(x)+c_2y'_2(x)+\cdots+c_ny'_n(x) \\
\cdots\quad\cdots \\
φ^{(n-1)}(x)=c_1y^{(n-1)}_1(x)+c_2y^{(n-1)}_2(x)+\cdots+c_ny^{(n-1)}_n(x)
$$
 因此  
$$
\cfrac{∂(φ,φ',\cdots,φ^{(n-1)})}{∂(c_1,c_2,\cdots,c_n)}=
\begin{vmatrix}
y_1(x) & y_2(x) & \cdots & y_n(x) \\ 
y'_1(x) & y'_2(x) & \cdots & y'_n(x) \\ 
\vdots &\vdots &\ddots &\vdots \\ 
y^{(n-1)}_1(x) & y^{(n-1)}_2(x) & \cdots & y^{(n-1)}_n(x) \\ 
\end{vmatrix}\neq 0
$$


<kbd>定义</kbd>：在区间 $[a,b]$上的 n 个函数$y_1(x),y_2(x),\cdots,y_n(x)$及导数所确定的行列式 
$$
W[y_1(x),y_2(x),\cdots,y_n(x)]=W(x) \\
=\begin{vmatrix}
y_1(x) & y_2(x) & \cdots & y_n(x) \\ 
y'_1(x) & y'_2(x) & \cdots & y'_n(x) \\ 
\vdots &\vdots &\ddots &\vdots \\ 
y^{(n-1)}_1(x) & y^{(n-1)}_2(x) & \cdots & y^{(n-1)}_n(x) \\ 
\end{vmatrix}
$$
 称为由这些函数所确定的==伏朗斯基行列式==(Wronskian)。

考虑定义在区间 $[a,b]$ 上的函数 $y_1(x),y_2(x),\cdots,y_k(x)$，如果存在不全为零的常数 $c_1,c_2,\cdots,c_k$ ，使得恒等式  
$$
c_1y_1(x)+c_2y_2(x)+\cdots+c_ky_k(x)\equiv0
$$
 对所有的 $x\in[a,b]$ 都成立，称这些函数在所给区间是==线性相关==的，否则称这些函数在所给区间是==线性无关==的。
即，在区间 $x\in[a,b]$ 上，要使得下式恒成立
$$
c_1y_1(x)+c_2y_2(x)+\cdots+c_ky_k(x)\equiv0
$$
 当且仅当 $c_1=c_2=\cdots=c_k=0$
例如函数 $\cos x$ 和 $\sin x$ 在任何区间都是线性无关的；但函数 $\cos^2x$ 和 $\sin^2x-1$ 在任何区间都是线性相关的。

 <kbd>定理 3</kbd>：若函数 $y_1(x),y_2(x),\cdots,y_n(x)$ 在区间$a⩽x⩽b$上线性相关，则在区间 $[a,b]$ 上它们的伏朗斯基行列式 $W(x)\equiv0$ 。
 证明：由假设，即知存在一组不全为零的常数 $c_1,c_2,\cdots,c_n$ 使得 
 $c_1y_1(x)+c_2y_2(x)+\cdots+c_ny_n(x)\equiv0\quad(a⩽x⩽b)$
 依次对 $x$ 求导，得到
 $\begin{cases}
 c_1y_1(x)+c_2y_2(x)+\cdots+c_ny_n(x)\equiv0 \\
c_1y'_1(x)+c_2y'_2(x)+\cdots+c_ny'_n(x)\equiv0 \\
\cdots\quad\cdots \\
c_1y^{(n-1)}_1(x)+c_2y^{(n-1)}_2(x)+\cdots+c_ny^{(n-1)}_n(x)\equiv0
\end{cases}$
上式可看出关于$c_1,c_2,\cdots,c_n$的齐次线性代数方程组，它的系数行列式就是 $W(x)$ ，于是由线性代数理论知道，要此方程组存在非零解，则它的系数行列式必须为零，即 $W(x)\equiv0$ 。

注意，定理 3的逆定理不一定成立。也就是说，由某些函数组构成的伏朗斯基行列式为零，但它们也可能是线性无关的。

 <kbd>定理 4</kbd>：齐次线性方程(2)的解 $y_1(x),y_2(x),\cdots,y_n(x)$ 在区间$a⩽x⩽b$上线性无关，等价于 $W[y_1(x),y_2(x),\cdots,y_n(x)]=W(x)$ 在这个区间的任何点上都不等于零，即 $W(x)\neq0\quad(a⩽x⩽b)$ 。
证明：用反证法即可。

根据定理 3和定理 4可以知道，由n阶齐次线性微分方程 (2) 的n个解构成的伏朗斯基行列式要么恒等于零，要么恒不为零。

现在考虑方程(2)是否存在n个线性无关的解。根据解的存在唯一性定理，取n组初始值 $(a⩽x_0⩽b)$
$\begin{cases}
y_1(x_0)=1,y'_1(x_0)=0,\cdots,y^{(n-1)}_1(x_0)=0 \\
y_2(x_0)=0,y'_2(x_0)=1,\cdots,y^{(n-1)}_2(x_0)=0 \\
\cdots\quad\cdots \\
y_n(x_0)=0,y'_n(x_0)=0,\cdots,y^{(n-1)}_n(x_0)=1
\end{cases}$
存在n个唯一解 $y_1(x),y_2(x),\cdots,y_n(x)$
又因为 $W(x_0)=\begin{vmatrix}
y_1(x_0) & y_2(x_0) & \cdots & y_n(x_0) \\ 
y'_1(x_0) & y'_2(x_0) & \cdots & y'_n(x_0) \\ 
\vdots &\vdots &\ddots &\vdots \\ 
y^{(n-1)}_1(x_0) & y^{(n-1)}_2(x_0) & \cdots & y^{(n-1)}_n(x_0) \\ 
\end{vmatrix}=1$
所以在区间$[a,b]$上$W(x)\neq0$，然后 $y_1(x),y_2(x),\cdots,y_n(x)\quad(a⩽x_0⩽b)$ 线性无关。
可以看出，n个线性无关的解组不是唯一的。

<kbd>定理 5</kbd>：n阶齐次线性方程(2)一定存在n个线性无关的解。

<kbd>定理 6 通解结构定理</kbd> 若$y_1(x),y_2(x),\cdots,y_n(x)$是n阶齐次线性方程(2)的n个线性无关的特解，则方程(2)的通解可表示为
$$
y^*(x)=c_1y_1(x)+c_2y_2(x)+\cdots+c_ny_n(x)\tag{5}
$$
 其中 $c_1,c_2,\cdots,c_n$ 是任意常数，且此通解包含了方程 (2) 所有的解。
证明：(1) 由叠加原理，$y^*(x)$ 是方程(2)的解。
(2) 证明 $y^*(x)$ 是方程(2)的通解。
任意常数$c_1,c_2,\cdots,c_n$的雅克比行列式(Jacobian)满足  
$\cfrac{∂(φ,φ',\cdots,φ^{(n-1)})}{∂(c_1,c_2,\cdots,c_n)}\equiv W(x)$
因为n个特解线性无关，$W(x)\neq0$，因此 $c_1,c_2,\cdots,c_n$相互独立
(3) 证明 $y^*(x)$ 包含了方程 (2) 所有的解。
由解的存在和唯一性定理，任取方程(2)满足初始条件 
$$
y(x_0)=η_1,y'(x_0)=η_2,\cdots,y^{(n-1)}(x_0)=η_n
$$
 的一个解 $y(x)$ ，只需确定常数$c_1,c_2,\cdots,c_n$的值，使其满足(5)式，作非齐次线性代数方程组 
$$
\begin{pmatrix}
y_1(x_0) & y_2(x_0) & \cdots & y_n(x_0) \\ 
y'_1(x_0) & y'_2(x_0) & \cdots & y'_n(x_0) \\ 
\vdots &\vdots &\ddots &\vdots \\ 
y^{(n-1)}_1(x_0) & y^{(n-1)}_2(x_0) & \cdots & y^{(n-1)}_n(x_0) \\ 
\end{pmatrix}
\begin{pmatrix}
c_1 \\ c_2 \\ \vdots \\ c_n
\end{pmatrix}
=\begin{pmatrix}
η_1 \\ η_2 \\ \vdots \\ η_n
\end{pmatrix}
$$
 它的系数行列式即为 $W(x_0)\neq0$ ，根据线性代数方程组的理论，上述方程组有唯一解 ，记为$\bar c_1,\bar c_2,\cdots,\bar c_n$。
因此 $y(x)=\bar c_1y_1(x)+\bar c_2y_2(x)+\cdots+\bar c_ny_n(x)$，并且满足初始条件。
定理证毕。

<kbd>推论</kbd>：n阶齐次线性方程 (2) 的线性无关解的最大个数等于n。
解的集合记为 $S(n)$ ，构成了一个n维的线性空间。方程 (2) 的一组n个线性无关的解就是解空间的一组基，称为==基本解组==(fundamental system of solutions)。其它的解可由基本解组线性表示即可。显然，基本解组不是惟一的。

## 高阶线性非齐次方程

**高阶线性非齐次方程** 
$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y'+a_n(x)y=f(x)\tag{1}
$$

**解的性质**
(1) 高阶非齐次线性方程的解与其对应齐次方程的解之和是非齐次方程的解。
如果 $\bar y(x)$ 是方程 (1) 的解，而 $y(x)$ 是方程 (2) 的解，则 $\bar y(x)+y(x)$ 是方程 (1) 的解。
(2) 高阶非齐次线性方程的任意两个解之差是其对应齐次方程的解。

<kbd>定理 7 通解结构定理</kbd>：设$y_1(x),y_2(x),\cdots,y_n(x)$是方程(2)的基本解组，而 $\bar y(x)$ 是方程 (1) 的某一特解，则方程 (1) 的通解可表示为
$$
y^*(x)=c_1y_1(x)+c_2y_2(x)+\cdots+c_ny_n(x)+\bar y(x) \tag{6}
$$
 其中 $c_1,c_2,\cdots,c_n$ 是任意常数，且此通解包含了方程 (1) 所有的解。
证明：(1) 由解的性质知，$y^*(x)$ 是方程(1)的解。
(2) 表达式(6)含有n个相互独立的任意常数，因此 $y^*(x)$ 是方程(1)的通解。
(3) 证明 $y^*(x)$ 包含了方程 (1) 所有的解。
现设 $\tilde y(x)$ 是方程 (1) 的任一解，则 $\tilde y(x)-\bar y(x)$ 是对应的齐次方程(2)的解，根据定理 6，并有一组确定的常数 $\tilde c_1,\tilde c_2,\cdots,\tilde c_n$ ，使得 $\tilde y(x)-\bar y(x)=\tilde c_1y_1(x)+\tilde c_2y_2(x)+\cdots+\tilde c_ny_n(x)$ ，即 $\tilde y(x)=\tilde c_1y_1(x)+\tilde c_2y_2(x)+\cdots+\tilde c_ny_n(x)+\bar y(x)$
定理证毕。

**常数变易法**[^const]：定理 7告诉我们，要解非齐次线性方程只需知道它的一个特解和对应的齐次线性方程的基本解组。我们可以用常数变易法求得非齐次线性方程的一个解。
设 $y_1(x),y_2(x),\cdots,y_n(x)$是齐次方程(2)的基本解组，因而方程 (2) 的通解为 $y=c_1y_1(x)+c_2y_2(x)+\cdots+c_ny_n(x)$ 。
用常数变易法，令 
$$
y=c_1(x)y_1(x)+c_2(x)y_2(x)+\cdots+c_n(x)y_n(x)
$$
 为非齐次方程 (1) 的解。这一证明类似一阶非齐次方程组的常数变易法，可以推导出系数满足的矩阵方程 
$$
\begin{pmatrix}
y_1(x) & y_2(x) & \cdots & y_n(x) \\ 
y'_1(x) & y'_2(x) & \cdots & y'_n(x) \\ 
\vdots &\vdots &\ddots &\vdots \\ 
y^{(n-1)}_1(x) & y^{(n-1)}_2(x) & \cdots & y^{(n-1)}_n(x) \\ 
\end{pmatrix}
\begin{pmatrix}
c'_1(x) \\ c'_2(x) \\ \vdots \\ c'_n(x)
\end{pmatrix}
=\begin{pmatrix}
0 \\ 0 \\ \vdots \\ f(x)
\end{pmatrix}
$$

可求得 
$$
c_k(x)=\int_{x_{0}}^{x}\cfrac{A_k(s)}{W(s)}f(s)ds
$$

得方程一个特解 
$$
\bar y=\sum_{k=1}^{n}y_k(x)c_k(x)
$$
 这里 $W(x)$ 为伏朗斯基行列式，$A_k(x)$为 $W(x)$ 中第 $n$ 行第 $k$ 列的代数余子式，即
$A_k(x)=(-1)^{n+k}\begin{vmatrix}
y_1(x)  & \cdots & y_{k-1}(x) & y_{k+1}(x) & \cdots &y_n(x) \\ 
y'_1(x)  & \cdots & y'_{k-1}(x) & y'_{k+1}(x)  &\cdots & y'_n(x) \\ 
\vdots  &\ddots &\vdots & \vdots  &\ddots &\vdots \\ 
y^{(n-2)}_1(x)  & \cdots & y^{(n-2)}_{k-1}(x) & y^{(n-2)}_{k+1}(x) & \cdots & y^{(n-2)}_n(x) \\ 
\end{vmatrix}$

**示例**：求 $y''+y=\cfrac{1}{\cos x}$ 的通解。
解：对应齐次方程的基本解组为 $\cos x,\sin x$
(1) 令方程特解为 $y=c_1(x)\cos x+c_2(x)\sin x$
(2) 解方程组 $\begin{cases} c'_1\cos x +c'_2\sin x =0 \\
-c'_1\sin x +c'_2\cos x =\frac{1}{\cos x}
\end{cases}$ 
解得 $c_1=\ln|\cos x|+γ_1,\quad c_2=x+γ_2$
(3) 原方程特解为 $\bar y=\cos x\ln|\cos x|+x\sin x$
(4) 原方程通解为 $y=γ_1(x)\cos x+γ_2\sin x+\cos x\ln|\cos x|+x\sin x$

## 常系数线性齐次微分方程

先引入高阶线性微分方程复值解的性质（请自行证明）
<kbd>定理 8</kbd> 如果方程(2)中所有系数 $a_i(x)(i=1,2,\cdots,n)$ 都是实值函数，而 $y=z(x)=φ(x)+\mathrm{i}ψ(x)$ 是方程的复值解，则 $z(x)$ 的共轭复值函数 $\bar z(x)=φ(x)-\mathrm{i}ψ(x)$ 也是方程(2)的解。

<kbd>定理 9</kbd> 如果方程 
$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y'+a_n(x)y=u(x)+\mathrm{i}v(x)
$$
 有复值解 $y=U(x)+\mathrm{i}V(x)$ ，这里 $a_i(x)(i=1,2,\cdots,n)$ 及 $u(x),v(x)$ 都是实值函数，那么这个解的实部$U(x)$和虚部$V(x)$分别是方程 
$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y'+a_n(x)y=u(x)
$$
和
$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y'+a_n(x)y=v(x)
$$
 的解。

**高阶常系数线性齐次微分方程** 
$$
L[y]=y^{(n)}+a_1y^{(n-1)}+\cdots+a_{n-1}y'+a_ny=0\tag{7}
$$
 其中系数 $a_i(i=1,2,\cdots,n)$ 为常数，这里 $L=\dfrac{\mathrm{d}^n}{\mathrm{d}x^n}+a_1\dfrac{\mathrm{d}^{n-1}}{\mathrm{d}x^{n-1}}+\cdots+\dfrac{\mathrm{d}}{\mathrm{d}x}+a_n$ 称为n阶线性微分算子(differential operator)。
按照定理，为了求方程的通解，只需要求出基本解组。实际上它的求解问题可以归结为代数方程求根问题，下面介绍基本解组的==欧拉待定指数函数法==。

回顾一阶常系数齐次线性方程 $y'+ay=0$ 通解为 $y=ce^{-ax}$，他有特解 $y=e^{-ax}$ ，这启示我们对于方程 (7) 也寻求类似形式的解 $y=e^{λx}$ ，其中 λ 是待定常数，可以是实数或复数。
带入方程(7) 可得 
$$
\begin{aligned}
L[e^{λx}] &=(e^{λx})^{(n)}+a_1(e^{λx})^{(n-1)}+\cdots+a_{n-1}(e^{λx})'+a_n(e^{λx}) \\
&=(λ^n+a_1λ^{n-1}+\cdots+a_n)e^{λx} \\
&=0
\end{aligned}
$$
 由 $e^{λx}\neq0$ 知 
$$
F(λ)=λ^n+a_1λ^{n-1}+\cdots+a_n=0\tag{8}
$$
  我们称方程(8)为==特征方程==，它的根称为==特征根==。
<kbd>结论</kbd> $y=e^{λx}$ 是方程 (7) 的解的充要条件是：λ 是代数方程 (8) 的根。

下面根据特征根的不同情形讨论
1. **特征根为单根**：如果特征方程 (8) 有 n 个互不相同的根 $λ_1,λ_2,\cdots,λ_n$，则齐次方程有 n 个解 $e^{λ_1x},e^{λ_2x},\cdots,e^{λ_nx}$， 这n个解的弗朗斯基行列式为 
$W(x)=\begin{vmatrix}
e^{λ_1x} & e^{λ_2x} & \cdots & e^{λ_nx} \\
λ_1e^{λ_1x} & λ_2e^{λ_2x} & \cdots & λ_ne^{λ_nx} \\
\vdots & \vdots &\ddots &\vdots\\
λ_1^{n-1}e^{λ_1x} & λ_2^{n-1}e^{λ_2x} & \cdots & λ_n^{n-1}e^{λ_nx}
\end{vmatrix}=e^{(λ_1+λ_2+\cdots+λ_n)x}
\begin{vmatrix}
1&1&\cdots &1 \\
λ_1 & λ_2 & \cdots & λ_n \\
\vdots & \vdots &\ddots &\vdots\\
λ_1^{n-1} & λ_2^{n-1} & \cdots & λ_n^{n-1}
\end{vmatrix}\neq 0$
这里需要用到线性代数里的范德蒙行列式， $W(x)\neq0$，因此解组 $e^{λ_1x},e^{λ_2x},\cdots,e^{λ_nx}$ 线性无关，为齐次方程的一个基本解组。
(1) 如果 $λ_1,λ_2,\cdots,λ_n$ 都是实数，方程通解为 
$$
y=c_1e^{λ_1x}+c_2e^{λ_2x}+\cdots+c_ne^{λ_nx}
$$
(2) 如果 $λ_1,λ_2,\cdots,λ_n$ 中有复数，那么由方程的系数是实数知，复根必然成对共轭出现。设 $λ_k=a+ib$ 为一特征根，则 $\bar λ_k=a-ib$ 也是特征根。这两个根对应的解为 
$$
e^{(a+ib)x}=e^{ax}(\cos bx+i\sin bx) \\e^{(a-ib)x}=e^{ax}(\cos bx-i\sin bx)
$$
 根据定理 8 他们的实部和虚部也是方程 (7) 的解，这样一来对于特征方程的一对共轭复根 $λ=a\pm ib$，对应一对线性无关的实值解 
$$
e^{ax}\cos bx,\quad e^{ax}\sin bx
$$


2. **特征根有重根**：设 $λ_1$ 为特征方程的 k 重根，则 $F(λ)$ 可表示为 $F(λ)=(λ-λ_1)^kP(λ),\quad P(λ_1)\neq0$
可知 $F(λ_1)=F'(λ_1)=\cdots=F^{(k-1)}(λ_1)=0,\quad F^{(k)}(λ_1)\neq0$
(i) 先设 $λ_1=0$，即 $F(λ)=λ^kP(λ)$，于是 $a_n=a_{n-1}=\cdots=a_{n-k+1}=0$
特征方程变为 $λ^n+a_1λ^{n-1}+\cdots+a_{n-k}λ^k=0$
对应的齐次方程变为 $y^{(n)}+a_1y^{(n-1)}+\cdots+a_{n-k}y^{(k)}=0$
易见他有 k 个线性无关的解 $1,x,x^2,\cdots,x^{k-1}$
(ii) 若 $λ_1\neq0$，做变量变换 $y=te^{λ_1x}$，注意到 
$y^{(m)}=(te^{λ_1x})^{(m)} 
=e^{λ_1x}[t^{(m)}+mλ_1t^{(m-1)}+\frac{m(m-1)}{2!}λ_1^2t^{(m-2)}+\cdots+λ_m^2t]$
所以 $L[te^{λ_1x}]=(t^{(n)}+b_1t^{(n-1)}+\cdots+b_nt)e^{λ_1x}=L_1[t]e^{λ_1x}$
于是齐次方程 (7) 化为 $L_1[t]=t^{(n)}+b_1t^{(n-1)}+\cdots+b_nt=0$
对应的特征方程为 $F_1(μ)=μ^n+b_1μ^{n-1}+\cdots+b_n=0$
直接计算可得 $F(μ+λ_1)e^{(μ+λ_1)x}=L[e^{(μ+λ_1)x}]=L_1[e^{μx}]e^{λ_1x}=F_1(μ)e^{(μ+λ_1)x}$
因此 $F(μ+λ_1)=F_1(μ)$
$F^{(j)}(μ+λ_1)=F_1^{(j)}(μ),j=1,2,\cdots,k$
可见特征方程 $F(λ)=0$ 的重根 $λ_1$ 对应于特征方程 $F_1(μ)=0$ 的重根 $μ_1=0$，且重数相同。这样问题转化为前面讨论过的情形。
重根 $μ_1=0$ 对应于方程 $L_1[t]=0$ 的 k个解 $t=1,x,x^2,\cdots,x^{k-1}$
因而重根 $λ_1$ 对应于方程 $L[y]=0$ 的 k个解 $y=e^{λ_1x},xe^{λ_1x},x^2e^{λ_1x},\cdots,x^{k-1}e^{λ_1x}$

于是我们下面的定理
<kbd>定理 10</kbd>：如果特征方程 $F(λ)=0$ 有 m 个互异的特征根 $λ_1,λ_2,\cdots,λ_m$，他们的重数依次为 $k_1,k_2,\cdots,k_m,k_i⩾1$，并且 $k_1+k_2+\cdots+k_m=n$，则下面的 n 个解：
$$
\begin{matrix}
e^{λ_1x},xe^{λ_1x},x^2e^{λ_1x},\cdots,x^{k_1-1}e^{λ_1x} \\
e^{λ_2x},xe^{λ_2x},x^2e^{λ_2x},\cdots,x^{k_2-1}e^{λ_2x} \\
\cdots \quad \cdots \quad \cdots \\
e^{λ_mx},xe^{λ_mx},x^2e^{λ_mx},\cdots,x^{k_m-1}e^{λ_mx}
\end{matrix}
$$
 构成齐次方程 $L[y]=0$ 的基本解组。

**示例**：求方程 $y^{(4)}-y=0$ 的通解。
(1) 特征方程 $λ^4-1=0$ 的根为 $\pm 1,\pm i$
(2) 两个共轭复根对应的实值解为 $\cos x,\sin x$
(3) 通解为 $c_1e^x+c_2e^{-x}+c_3\cos x+c_4\sin x$

**欧拉方程**：变系数微分方程，形如
$$
y^{(n)}+p_1x^{n-1}y^{(n-1)}+\cdots+p_{n-1}xy'+p_ny=0
$$
 的方程（其中$p_1,p_2,\cdots,p_n$为常数），叫做欧拉方程。此方程可通过变量变换化为常系数微分方程。
做变换[^5]$x=e^t$，将自变量 $x$ 换为 $t$，求得
$$
\begin{aligned}
&\dfrac{\mathrm{d}y}{\mathrm{d}x}=e^{-t}\dfrac{\mathrm{d}y}{\mathrm{d}t} \\ 
&\dfrac{\mathrm{d}^2y}{\mathrm{d}x^2}=e^{-2t}(\frac{\mathrm{d}^2y}{\mathrm{d}t^2}-\dfrac{\mathrm{d}y}{\mathrm{d}t}) \\
\end{aligned}
$$
  用数学归纳法可证明有 
$$
\dfrac{\mathrm{d}^ky}{\mathrm{d}x^k}=e^{-kt}(\frac{\mathrm{d}^ky}{\mathrm{d}t^k}+a_1\dfrac{\mathrm{d}^{k-1}y}{\mathrm{d}t^{k-1}}+\cdots+a_{k-1}\dfrac{\mathrm{d}y}{\mathrm{d}t})
$$
 把他带入欧拉方程，便得到一个以 $t$ 为自变量得常系数线性微分方程 
$$
\frac{\mathrm{d}^ny}{\mathrm{d}t^n}+b_1\dfrac{\mathrm{d}^{n-1}y}{\mathrm{d}t^{n-1}}+\cdots+b_{n-1}\dfrac{\mathrm{d}y}{\mathrm{d}t}+b_ny=0
$$
 其中$b_1,b_2,\cdots,b_n$为常数

[^5]: 这里仅在$x>0$范围内求解，若$x<0$，可变化为$x=-e^t$，所得结果类似。

## 常系数线性非齐次微分方程

**高阶常系数线性非齐次微分方程** 
$$
y^{(n)}+a_1y^{(n-1)}+\cdots+a_{n-1}y'+a_ny=f(x)\tag{9}
$$
 其中$a_1,a_2,\cdots,a_n$为常数。
对应齐次方程的通解上节已介绍，可利用常数变易法求得特解，但这一方法比较麻烦。下面介绍$f(x)$ 特殊形式时根据经验推测特解的方法。

**比较系数法**

- ==类型 I==：$f(x)=P_m(x)e^{μx}$
  其中 $P_m(x)=b_0x^{m}+b_1x^{m-1}+\cdots+b_{m-1}x+b_m$是m次多项式，$μ,b_0,b_1,\cdots,b_m$是常数。
  考虑到 $e^{μx}$ 和多项式乘积的导数仍然是 $e^{μx}$ 和多项式的乘积，设
  $$
  y=Q(x)e^{μx}
  $$
  是方程的特解，其中 $Q(x)$ 为待定多项式。由于
  $y'=e^{μx}[μQ(x)+Q'(x)] \\
  y''=e^{μx}[μ^2Q(x)+2μQ'(x)+Q''(x)]\\
  \cdots\quad\cdots\\
  y^{(n)}=e^{μx}[μ^nQ(x)+nμ^{n-1}Q'(x)+\frac{n(n-1)}{2!}μ^{n-2}Q''(x)+\cdots+Q^{(n)}(x)]$

  把关系式带入方程 (9) 并消去 $e^{μx}$ 得到
  $Q^{(n)}(x)+(nμ+a_1)Q^{(n-1)}(x)+\cdots+(μ^n+a_1μ^{n-1}+\cdots+a_n)Q(x)=P_m(x)$

  上式可利用方程 (9) 的特征多项式 $F(λ)$ 进一步化为
  $Q^{(n)}(x)+\frac{F^{(n-1)}(μ)}{(n-1)!}Q^{(n-1)}(x)+\cdots+F'(μ)Q'(x)+F(μ)Q(x)=P_m(x)$

  由此确定多项式 $Q(x)$ 的次数：
  (1) 当 $μ$ 不是特征方程 $F(λ)=0$ 的根时，$F(μ)\neq0$ ，上式中 $Q(x)$ 的系数不为零，左端多项式的最高次数由 $Q(x)$ 项确定，要想使等式两端恒成立，可设特解形式为 
$$
y=Q_m(x)e^{μx}
$$
其中$Q_m(x)=B_0x^{m}+B_1x^{m-1}+\cdots+B_{m-1}x+B_m$是m次多项式，比较等式两端 $x$ 的同次幂，就得到以$B_0,B_1,\cdots,B_m$为未知数的m+1方程组，从而求得 $B_i$，获得特解。
(2) 当 $μ$ 是特征方程 $F(λ)=0$ 的 k 重根时，易知 $F(λ_1)=F'(λ_1)=\cdots=F^{(k-1)}(λ_1)=0,\quad F^{(k)}(λ_1)\neq0$，所以$Q(x),Q'(x),\cdots,Q^{(k-1)}(x)$ 的系数为零，方程有形如 
$$
y=x^kQ_m(x)e^{μx}
$$
的特解，同样可以通过比较系数来确定待定常数。

- ==类型 II==：$f(x)=[A_l(x)\cosω x+B_n(x)\sinω x]e^{μx}$
其中$μ,ω$是常数，$A_l(x)$ 和 $B_n(x)$是多项式
(1) 应用欧拉公式把$f(x)$变成复指数形式 
$$
f(x)=f_1(x)+f_2(x)=\dfrac{A_l+iB_n}{2}e^{(μ-ω i)x}+\dfrac{A_l-iB_n}{2}e^{(μ+ω i)x}
$$
   根据叠加原理，方程 $L[y]=f_1(x)$ 与 方程 $L[y]=f_2(x)$ 的解之和必为方程 $L[y]=f(x)$ 的解。
(2) 由于 $\overline{f_1(x)}=f_2(x)$ ，易知若 $y_1(x)$ 为$L[y]=f_1(x)$的解，则共轭函数 $\overline{y_1(x)}$ 必为$L[y]=f_2(x)$的解。利用类型 I 的结果，可知方程的解形如 
$$
\begin{aligned} y&=x^kQ_m(x)e^{(μ-ω i)x}+x^k\bar{Q}_m(x)e^{(μ+ω i)x} \\ 
& =x^ke^{μ x}[R_m^{(1)}(x)\cosω x+R_m^{(2)}\sinω x]
\end{aligned}
$$
 其中$R_m^{(1)}(x),R_m^{(2)}(x)$是m次多项式，$m=\max\{l,n\}$
带入方程通过比较系数确定待定常数。

**拉普拉斯变换法**
<kbd>Laplace变换</kbd>：设函数$f(t)$ 在$t\geqslant 0$时有定义，且积分$\displaystyle\int_{0}^{∞}f(t)e^{-st}dt$在复数 s 的某一个区域内收敛，则此积分所确定的函数
$$
\displaystyle F(s)=\int^{\infty}_{0}f(t)e^{-st}\text{d}t
$$
称为函数$f(t)$的Laplace 变换，记为$F(s)=\mathcal L[f(t)]$，$f(t)$ 称为原函数， $F(s)$ 称为象函数。
给定微分方程 
$$
y^{(n)}+a_1y^{(n-1)}+\cdots+a_{n-1}y'+a_ny=f(x)\tag{9}
$$
 及初始条件
$$
y(0)=η_1,y'(0)=η_2,\cdots,y^{(n-1)}(0)=η_n
$$
 其中$a_1,a_2,\cdots,a_n$为常数，$f(x)$ 连续且满足原函数的条件。
可以证明，如果 $y(x)$ 是方程 (9) 的任意解，则 $y(x)$ 及其各阶导数 $y^{(k)}(x)$ 均是原函数。记
$$
F(s)=\mathcal L[f(x)]=\int_{0}^{∞}f(x)e^{-sx}dx \\
Y(s)=\mathcal L[y(x)]=\int_{0}^{∞}y(x)e^{-sx}dx
$$
 那么原函数的微分性质有  
$$
\mathcal L[y'(x)]=sY(s)-η_1 \\ \cdots \\
\mathcal L[y^{(n)}(x)]=s^nY(s)-s^{n-1}η_1-s^{n-2}η_2-\cdots-η_n
$$

于是对方程 (9) 两端进行拉普拉斯变换，并利用线性性质，就得到 
$s^nY(s)-s^{n-1}η_1-s^{n-2}η_2-\cdots-η_n \\
+a_1[s^{n-1}Y(s)-s^{n-2}η_1-s^{n-3}η_2-\cdots-η_{n-1}] \\
+\cdots \\
+a_{n-1}[sY(s)-η_1] \\
+a_nY(s)=F(s)$
即 $(s^n+a_1s^{n-1}+\cdots+a_n)Y(s) \\
=F(s)+(s^{n-1}+a_1s^{n-2}+\cdots+a_{n-1})η_1 \\
+(s^{n-2}+a_1s^{n-3}+\cdots+a_{n-2})η_2 \\
+\cdots+η_n$
或 
$$
A(s)Y(s)=F(s)+B(s)
$$
 其中 $A(s),B(s),F(s)$ 都是已知多项式，由此 
$$
Y(s)=\cfrac{F(s)+B(s)}{A(s)}
$$
 这便是方程 (9) 满足初始条件的解 $y(x)$ 的象函数，可直接查拉普拉斯变换表或逆变换求得 $y(x)$。


## 高阶微分方程的降阶

对于一般的高阶微分方程，我们没有普遍的求解方法，通常是通过变量替换把高阶方程的求解化为较低阶的方程来求解，下面我们介绍三种类型微分方程的降阶法(method of reduction of order)。

- ==方程不显含未知函数 $y$==  ，即 
$$
F(x,y^{(k)},y^{(k+1)},\cdots,y^{(n)})=0\quad(1⩽k⩽n)
$$
  设 $t=y^{(k)}$，则方程降阶为关于$t$的$n-k$阶方程 $F(x,t,t',\cdots,t^{(n-k)})$ 。如果可以求得此方程的通解，经过 $k$ 次积分就能求得原方程得通解。

- ==方程不显含自变量 $x$==  ，即 
$$
F(y,y',\cdots,y^{(n)})=0
$$
设 $t=y'$，利用复合函数的求导法则 
$y'=t \\
y''=\cfrac{dt}{dx}=\cfrac{dt}{dy}\cdot\cfrac{dy}{dx}=t\cfrac{dt}{dy}\\
y'''=t(\cfrac{dt}{dy})^2+t^2\cfrac{d^2t}{dy^2}\\
\cdots\quad\cdots$
利用数学归纳法，可得到以 $t$ 为未知函数 $y$ 为自变量的$n-1$阶微分方程。

- ==线性齐次方程== ：已知方程 $k$ 个线性无关的解，方程可降低 $k$ 阶。
**对于二阶线性齐次方程** 
$$
y''+p(x)y'+q(x)y=0
$$
设 $y_1(x)$ 是方程的特解，令 $y=y_1t$，则
$y'=y_1t'+y'_1t \\
y''=y_1t''+2y'_1t'+y''_1t$
把关系式带入方程得
$y_1t''+(2y'_1+py_1)t'+(y''_1+py'_1+qy_1)t=0$
因为 $y_1(x)$ 是方程的特解，所以上式关于 $t$ 的系数恒等于0
再引入未知函数 $z=t'$ 上式化为一阶方程 $y_1z'+(2y'_1+py_1)z=0$
可求得 $z=cy_1^{-2}e^{\int-pdx}$
因而 $y=y_1\int zdx=y_1(c_1+c\int y_1^{-2}e^{\int-pdx})$
显然 $y_1\int y_1^{-2}e^{\int-pdx}$ 与 $y_1$ 是线性无关的（因为他们之比不是常数），上式是二次方程的通解。

  **对于高阶线性齐次方程** 
$$
y^{(n)}+a_1(x)y^{(n-1)}+\cdots+a_{n-1}(x)y'+a_n(x)y=0\tag{2}
$$
  设 $y_1,y_2,\cdots,y_k$ 是方程的 k 个线性无关的特解，显然 $y_i\not\equiv0,i=1,2,\cdots,k$
**(1)** 令 $y=y_kt$，直接计算可得
$y'=y_kt'+y'_kt \\
y''=y_kt''+2y'_kt'+y''_kt \\
\cdots\quad\cdots\\
y^{(n)}=y_kt^{(n)}+ny'_kt^{(n-1)}+\frac{n(n-1)}{2!}y''_kt^{(n-2)}+\cdots+y^{(n)}_kt$
把关系式带入方程 (2) 得
$y_kt^{(n)}+[ny'_k+a_1(x)y_k]t^{(n-1)}+\cdots+ \\
[y^{(n)}_k+a_1(x)y^{(n-1)}_k+\cdots+a_{n-1}(x)y'_k+a_n(x)y_k]t\\
=0$
因为 $y_k(x)$ 是方程 (2) 的特解，所以上式关于 $t$ 的系数恒等于0
**(2)** 再引入未知函数 $z=t'$ ，在 $y_k\neq0$的区间上，上式可化为 $n-1$ 阶齐次方程
$z^{(n-1)}+b_1(x)z^{(n-2)}+\cdots+b_{n-2}(x)z'+b_{n-1}(x)z=0$
且知解之间的关系 $z=t'=(\cfrac{y}{y_k})'$ 或 $y=\displaystyle y_k\int zdx$
因此对于关于 z 的方程我们也知道它的 $k-1$ 个特解 $z_i=(\cfrac{y_i}{y_k})' ,i=1,2,\cdots,k-1$
关于 $z_1,z_2,\cdots,z_{k-1}$ 的线性无关，可用反证法证明。
假设 $α_1z_1+α_2z_2+\cdots+α_{k-1}z_{k-1}\equiv0 \\
α_1(\cfrac{y_1}{y_k})'+α_1(\cfrac{y_2}{y_k})'+\cdots+α_{k-1}(\cfrac{y_{k-1}}{y_k})'\equiv0$
两边关于 $x$ 积分可得
$α_1(\cfrac{y_1}{y_k})+α_1(\cfrac{y_2}{y_k})+\cdots+α_{k-1}(\cfrac{y_{k-1}}{y_k})\equiv-α_k$
 则 $α_1y_1+α_2y_2+\cdots+α_{k-1}y_{k-1}+α_ky_k\equiv0$
 $y_1,y_2,\cdots,y_k$线性相关，与已知条件矛盾，证毕。
 **(3)** 对关于 z 的方程仿以上做法，令 $z=\displaystyle z_{k-1}\int udx$
 则可将方程化为关于 u 的 $n-2$ 阶齐次线性方程
 $u^{(n-2)}+c_1(x)u^{(n-3)}+\cdots+c_{n-3}(x)u'+c_{n-2}(x)u=0$
 并且还知道它的 $k-2$ 个线性无关的特解 $u_i=(\cfrac{z_i}{z_{k-1}})' ,i=1,2,\cdots,k-2$
 依次类推，我们可以得到一个 $n-k$ 阶的齐次线性方程，相当于利用k 个线性无关的特解把方程 (2) 降低了 k 阶。

## 高阶微分方程幂级数求法

幂级数解法是求解常微分方程的一种方法，特别是当微分方程的解不能用初等函数或其积分式表达时，就要寻求其他求解方法，尤其是近似求解方法，幂级数解法就是常用的近似求解方法。用幂级数解法和广义幂级数解法可以解出许多数学物理中重要的常微分方程，例如: 贝塞尔方程、勒让德方程。

考虑二阶齐次线性微分方程 
$$
y''+p(x)y'+q(x)y=0\tag{10}
$$
 及初始条件 $y(x_0)=η_1,y'(x_0)=η_2$
> 为不失一般性，可设 $x_0=0$，否则我们引进新变量 $t=x-x_0$，此时方程形式不变，对应的 $t_0=0$

<kbd>定理 11</kbd>：若方程 (10) 中系数 $p(x)$ 和 $q(x)$ 都能展开成 $x$ 的幂级数，且收敛区间为 $|x|<R$，则方程有形如 $y=\displaystyle\sum_{n=0}^∞a_nx^n$ 的特解，也以 $|x|<R$ 为级数的收敛区间。

<kbd>定理 12</kbd>：若方程 (10) 中系数 $p(x)$ 和 $q(x)$ 具有性质：$xp(x)$ 和 $x^2q(x)$均能展开成 $x$ 的幂级数，且收敛区间为 $|x|<R$，则方程有形如 $y=\displaystyle x^α\sum_{n=0}^∞a_nx^n$ 的特解，也以 $|x|<R$ 为级数的收敛区间，这里 $a_0\neq0,α$是一个待定常数。

**求解贝塞尔方程**(Bessel equation) 
$$
x^2y''+xy'+(x^2-n^2)y=0\tag{11}
$$

 其中 n 为常数。
将方程改写成 $y''+\cfrac{1}{x}y'+\cfrac{x^2-n^2}{x^2}y=0$ 
它满足定理 12 的条件，且 $xp(x)=1,x^2q(x)=x^2-n^2$ 按 $x$ 展开成幂级数的收敛区间为 $(-∞,+∞)$。由定理 12 方程有形如 
$$
y=\displaystyle\sum_{k=0}^∞a_kx^{α+k}\tag{12}
$$

 的特解，$a_k,α$是待定常数。带入贝塞尔方程可得
$\displaystyle x^2\sum_{k=1}^∞(α+k)(α+k-1)a_kx^{α+k-2} \\
+x\sum_{k=1}^∞(α+k)a_kx^{α+k-1} \\
+(x^2-n^2)\sum_{k=0}^∞(α+k)a_kx^{α+k}=0$
进一步合并 $x$ 的同幂项
$\displaystyle\sum_{k=0}^∞[(α+k)(α+k-1)+(α+k)-n^2]a_kx^{α+k}+\sum_{k=0}^∞a_kx^{α+k+2}=0$
令各项的系数等于零，得代数方程组
$$
\begin{cases}
a_0[α^2-n^2]=0 \\
a_1[(α+1)^2-n^2]=0 \\
\cdots\quad\cdots \\
a_k[(α+k)^2-n^2]+a_{k-2}=0 \\
\cdots\quad\cdots
\end{cases}
$$
因为 $a_0\neq0$ ，故从方程组解得 $α=\pm n$
(i) 当 $α=n$ 时，带入代数方程组可得
$a_1=0,a_k=-\cfrac{a_{k-2}}{k(2n+k)},\quad k=2,3,\cdots$
或按下标是奇数或偶数，我们分别有
$\begin{cases}
a_{2k+1}=\cfrac{-a_{2k-1}}{(2k+1)(2n+2k+1)} \\
a_{2k}=\cfrac{-a_{2k-2}}{2k(2n+2k)}
\end{cases}\quad k=1,2,\cdots$
从而求得
$\begin{cases}
a_{2k-1}=0 \\
a_{2k}=(-1)^k\cfrac{a_0}{2^{2k}k!(n+1)(n+2)\cdots(n+k)} \\
\end{cases}\quad k=1,2,\cdots$
将各 $a_k$ 带入 (12) 得到贝塞尔方程得一个解
$\displaystyle y_1=a_0x^n+\sum_{k=1}^{∞}(-1)^k\cfrac{a_0}{2^{2k}k!(n+1)(n+2)\cdots(n+k)}x^{2k+n}$
既然是求特解，不妨设
$a_0=\cfrac{1}{2^n\Gamma(n+1)}$ ，其中[^gamma] $\displaystyle\Gamma(s)=\int_{0}^{∞}x^{s-1}e^{-x}dx$
从而上式特解变为 
$$
\displaystyle y_1=\sum_{k=0}^{∞}\cfrac{(-1)^k}{k!\Gamma(n+k+1)}(\cfrac{x}{2})^{2k+n}\equiv J_n(x)
$$

$J_n(x)$ 是由贝塞尔方程定义得特殊函数，称为==n 阶贝塞尔函数==。
因此，贝塞尔方程总有一个特解 $J_n(x)$，我们只需寻求另一个线性无关的，特解即可求得贝塞尔方程通解。

[^gamma]: $\Gamma$ 函数性质：$\Gamma(s+1)=s\Gamma(s+1)$

$\Gamma(n+1)=n!\quad n$为正整数。

 (ii) 当 $α=-n$ 时，带入 (12) 得到的特解形式为
$\displaystyle y_2=\sum_{k=0}^{∞}a_kx^{-n+k}$
注意到只要 n 不是非负整数，和 $α=n$ 的求解过程一样，我们可以求解代数方程组得到
$\begin{cases}
a_{2k-1}=0 \\
a_{2k}=(-1)^k\cfrac{a_0}{2^{2k}k!(-n+1)(-n+2)\cdots(-n+k)} \\
\end{cases}\quad (k=1,2,\cdots)$
因而可求得另一个特解
$\displaystyle y_2=a_0x^{-n}+\sum_{k=1}^{∞}(-1)^k\cfrac{a_0}{2^{2k}k!(-n+1)(-n+2)\cdots(-n+k)}x^{2k-n}$ 
此时，令
$a_0=\cfrac{1}{2^{-n}\Gamma(-n+1)}$
从而上式特解变为 
$$
\displaystyle y_2=\sum_{k=0}^{∞}\cfrac{(-1)^k}{k!\Gamma(-n+k+1)}(\cfrac{x}{2})^{2k-n}\equiv J_{-n}(x)
$$

$J_{-n}(x)$ 称为 ==-n 阶贝塞尔函数==。
由达朗贝尔判别法不难验证级数 $J_n(x)$ 在任何区间， $J_{-n}(x)$ 在 $x\neq0$ 时都是收敛的，并且  $J_{n}(x)$ 和  $J_{-n}(x)$ 线性无关。于是当 n 不是非负整数时，贝塞尔方程的通解为
$$
y=c_1J_n(x)+c_2J_{-n}(x)
$$
(iii) 当 $α=-n$ ，而 n 为自然数时，我们不能带入 (12) 求解 $a_{2k}$ 。这时可以采用降阶法求出与 $J_n(x)$ 线性无关的特解，由公式直接求得通解为
$$
\displaystyle y=J_n(x)[c_1+c_2\int\cfrac{dx}{xJ_n^2(x)}]
$$


> 参考文献：
> 《常微分方程教程》| 丁同仁
> 《常微分方程》| 王高雄
> MOOC《常微分方程》| 西北大学 | 窦霁虹、付英
> MOOC《高等数学》| 国防科技大学
