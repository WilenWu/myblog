---
title: 偏微分方程(三)
categories:
  - 数学
  - 偏微分方程
tags:
  - 数学
  - PDE
  - 微分方程
  - 格林函数
  - 积分变换
katex: true
description: 积分变换法、基本解和格林函数
abbrlink: b5843f92
date: 2020-06-21 18:55:54
cover:
top_img:
---

# 积分变换法

在积分变换中我们曾用拉普拉斯变换方法求解常微分方程。经过变换，常微分方程变为代数方程，解出代数方程，再进行反演就得常微分方程的解。积分变换在数学物理方程中亦有广泛的应用。

## 傅里叶变换法

傅里叶变换法常用于求解无界空间（含一维半无界空间）定解问题。本节通过几个例子给出几个重要的解的公式。

<kbd>Fourier 积分定理</kbd>：若 $f(x)$ 在 $(-\infty,+\infty)$上满足：
(1) 在任一有限区间上满足狄利克雷(Dirichlet)条件[^D]；
(2) 在无限区间 $(-∞,+∞)$上绝对可积，即 $\displaystyle\int_{-∞}^{+∞}|f(x)| dx$ 收敛
那么，对任意  $x\in(-\infty,+\infty)$
$$
\displaystyle f(x)=\dfrac{1}{2\pi}\int_{-∞}^{+∞}[\int^{+∞}_{-∞}f(τ)e^{-ik τ}\text{d}τ]e^{ikx}\text{d}k \tag{1.1}
$$
在间断点处，上式左端为 $\frac{1}{2}[f(x^-)+f(x^+)]$

<kbd>Fourier 变换</kbd>：如果函数 $f(x)$ 满足Fourier 积分定理，由式 (1.1) 知，令 
$$
\displaystyle F(k)=\int^{+∞}_{-∞}f(x)e^{-ikx}\text{d}x \tag{1.2}
$$
则有 
$$
\displaystyle f(x)=\dfrac{1}{2\pi}\int_{-∞}^{+∞}F(k)e^{ik x}\text{d}k \tag{1.3}
$$
从上面两式可以看出，$f(x)$ 和 $F(k)$ 通过确定的积分运算可以互相转换。 $F(k)$称为 $f(x)$ ==Fourier 变换==(Fourier transform)，或==象函数==(image function)，记为$F(k)=\mathcal{F}[f(x)]$ ；$f(x)$称为 $F(k)$ ==Fourier 逆变换==(inverse Fourier transform)，或==象原函数==(original image function)，记为$f(x)=\mathcal{F}^{-1}[F(k)]$ ；通常称$f(x)$与$F(k)$构成一个==Fourier 变换对==(transform pair)，记作 $f(x)\lrarr F(k)$ 。

**示例**：求函数 $f(x)=e^{-bx^2}$ 的傅里叶变换，其中常数 $b>0$
解：由定义和分部积分得
$$
\begin{aligned}
F(k)&=\int^{+∞}_{-∞}f(x)e^{-ikx}dx=\int^{+∞}_{-∞}e^{-ikx-bx^2}dx \\
&=-\frac{1}{\mathrm ik}e^{-ikx-bx^2}\Big|_{-∞}^{+∞}
-\frac{1}{\mathrm ik}\int^{+∞}_{-∞}2bxe^{-ikx-bx^2}dx \\
&=\frac{2b\mathrm i}{k}\mathcal F[xf(x)] =-\frac{2b}{k}\frac{\mathrm d}{\mathrm dk}F(k)
\end{aligned}
$$
取 $k=0$ 可得
$$
F(0)=\int^{+∞}_{-∞}e^{-bx^2}dx=\sqrt{\frac{\pi}{b}}
$$
转化为解常微分方程初值问题
$$
\begin{cases}
F'(k)+\cfrac{k}{2b}F(k)=0 \\
F(0)=\sqrt{\cfrac{\pi}{b}}
\end{cases}
$$
其解为
$$
F(k)=\sqrt{\cfrac{\pi}{b}}\exp(-\frac{k^2}{4b})
$$
即
$$
\mathcal F[e^{-bx^2}]=\sqrt{\cfrac{\pi}{b}}\exp(-\frac{k^2}{4b})\quad (b>0)\tag{1.4}
$$
特别的，取 $b=\cfrac{1}{4c^2t}$ 时，有
$$
\mathcal F[\exp(-\frac{x^2}{4c^2t})]=2c\sqrt{\pi t}e^{-c^2k^2t}\tag{1.5}
$$
逆变换为
$$
\mathcal F^{-1}[e^{-c^2k^2t}]=\frac{1}{2c\sqrt{\pi t}}\exp(-\frac{x^2}{4c^2t})\tag{1.6}
$$
由式 (1.4) 还可以得到
$$
\int^{+∞}_{0}e^{-bx^2}\cos kx\mathrm dx
=\frac{1}{2}\sqrt{\cfrac{\pi}{b}}\exp(-\frac{k^2}{4b})\tag{1.7}
$$
**多重傅里叶变换**：设 $\mathbf x=(x_1,x_2,\cdots,x_n)\in\R^n,\mathbf k=(k_1,k_2,\cdots,k_n),\mathrm d\mathbf x=dx_1dx_2\cdots dx_n$。若 $f(\mathbf x)$ 在 $\R^n$ 上连续，分片光滑且连续可积，令
$$
F(\mathbf k)=\int_{\R^n}f(\mathbf x)e^{-\mathrm i\mathbf{k\cdot x}}d\mathbf x\tag{1.8}
$$
则有
$$
f(\mathbf x)=\frac{1}{(2\pi)^n}\int_{\R^n}F(\mathbf k)e^{\mathrm i\mathbf{k\cdot x}}d\mathbf k\tag{1.9}
$$
其中 $F(\mathbf k)$ 称为 $f(\mathbf x)$ 的 ==多重傅里叶变换==，记为 $F(\mathbf k)=\mathcal{F}[f(\mathbf x)]$；$f(\mathbf x)$ 称为 $F(\mathbf k)$ 的 ==多重傅里叶逆变换==，记为 $f(\mathbf x)=\mathcal{F}^{-1}[F(\mathbf k)]$

[^D]: 若函数 $f(x)$ 在区间 $D$ 上满足：
(1) 连续或只有有限个第一类间断点；
(2) 只有有限个极值点
则称函数 $f(x)$ 在区间 $D$上满足狄利克雷(Dirichlet)条件

**求无限长弦的初值问题**
$$
\begin{cases}
u_{tt}-a^2u_{xx}=0 \\
u|_{t=0}=\phi(x),\cfrac{∂u}{∂t}|_{t=0}=\psi(x)
\end{cases}
$$
其中 $\phi(x),\psi(x)$ 分别表示初始位移和初始速度。
解：应用傅里叶变换，即方程及初始条件两边同乘以 $e^{-\mathrm ikx}$ ，并对空间变量 $x$ 进行积分（时间变量 $t$ 视作参数）。
 记 $U(k,t)=\mathcal{F}[u(x,t)]\displaystyle=\int^{+∞}_{-∞}u(x,t)e^{-\mathrm ikx}dx$ ，运用用[含参变量的积分][math]及傅里叶变换的[微分性质][F]

则定解问题变为关于 $t$ 的常微分方程及初值条件
$$
\begin{cases}
U''+k^2a^2U=0 \\
U(0)=\Phi(k),U'(0)=\Psi(k)
\end{cases}
$$
其中 $\Phi(k)=\mathcal{F}[\phi(x)],\quad\Psi(k)=\mathcal{F}[\psi(x)]$ 分别是 $\phi(x),\psi(x)$ 关于 $x$ 的傅里叶变换。其解为
$$
U(k,t)=\Phi(k)\cos kat+\cfrac{1}{ka}\Psi(k)\sin kat
$$

最后，对 $U(k,t)$ 做傅里叶逆变换，用[延迟性质和积分性质][F]，结果是
$$
u(x,t)=\frac{1}{2}[\phi(x+at)+\phi(x-at)]+\frac{1}{2a}\int_{x-at}^{x+at}\psi(ξ)dξ
$$
这个公式正是达朗贝尔(d’Alembert)公式。

[math]: /posts/math/Calculus-II/
[F]: /posts/math/Integral-Transform/

**求无限长杆的有源热传导问题**
$$
\begin{cases}
u_{t}-a^2u_{xx}=f(x,t) \\
u|_{t=0}=\phi(x)
\end{cases}
$$
解：做傅里叶变换，定解问题变为
$$
\begin{cases}
U'+k^2a^2U=F(k,t) \\
U(0)=\Phi(k)
\end{cases}
$$
其中 $U(k,t)=\mathcal{F}[u(x,t)],\quad F(k,t)=\mathcal{F}[f(x,t)],\quad\Phi(k)=\mathcal{F}[\phi(x)]$ 分别是 $u(x,t),f(x,t),\phi(x)$ 关于 $x$ 的傅里叶变换。这个常微分方程初值问题的解为
$$
U(k,t)=\Phi(k)e^{-k^2a^2t}+\int_0^tF(k,\tau)e^{-k^2a^2(t-\tau)}d\tau
$$

最后，对 $U(k,t)$ 做傅里叶逆变换，用[卷积定理][F]，结果是
$$
u(x,t)=\int_{-\infty}^{+\infty}\phi(\xi)
\frac{1}{2a\sqrt{\pi t}}\exp[-\frac{(x-\xi)^2}{4a^2t}]d\xi
+\int_0^t\int_{-\infty}^{+\infty}\frac{f(\xi,\tau)}{2a\sqrt{\pi (t-\tau)}}
\exp[-\frac{(x-\xi)^2}{4a^2(t-\tau)}]d\xi d\tau
$$

**傅里叶正弦变换或余弦变换**：如果 $f(x)$ 定义在半无界区间 $[0,+\infty)$ 上，满足狄利克雷(Dirichlet)条件[^D]且绝对可积，则有傅里叶正弦变换
$$
F(k)=\int^{+∞}_{0}f(x)\sin kx\text{d}x \\f(x)=\dfrac{1}{2\pi}\int_{0}^{+∞}F(k)\sin kx\text{d}k
$$
或余弦变换
$$
G(k)=\int^{+∞}_{0}g(x)\cos kx\text{d}x \\
g(x)=\dfrac{1}{2\pi}\int_{0}^{+∞}G(k)\cos kx\text{d}k
$$
对于半无界空间，存在自然边界条件
$$
\lim\limits_{x\to\infty}f(x)=0,\quad \lim\limits_{x\to\infty}f'(x)=0
$$
可以采用正弦变换或余弦变换，对于正弦变换
导数的正弦变换为
$$
\begin{aligned}
& \int^{+∞}_{0}f'(x)\sin kx\text{d}x \\
=& f(x)\sin kx\Big|_0^{+∞}-k\int^{+∞}_{0}f(x)\cos kx\text{d}x \\
=& -k\int^{+∞}_{0}f(x)\cos kx\text{d}x
\end{aligned}
$$
二阶导正弦变换为
$$
\begin{aligned}
& \int^{+∞}_{0}f''(x)\sin kx\text{d}x \\
=& -k\int^{+∞}_{0}f'(x)\cos kx\text{d}x \\
=& -k[f(x)\cos kx\Big|_0^{+∞}+k\int^{+∞}_{0}f(x)\sin kx\text{d}x] \\
=& kf(0)-k^2F(k)
\end{aligned}
$$

由此可见，对于二阶偏微分方程的定解问题，只有在半无界空间的 $x=0$ 端给出第一类边界条件时，才可以采用正弦变换。

同样对于余弦变换，也有
$$
\int^{+∞}_{0}g'(x)\cos kx\text{d}x=-g(0)+k\int^{+∞}_{0}g(x)\sin kx\text{d}x \\
\int^{+∞}_{0}g''(x)\cos kx\text{d}x=-g'(0)-k^2G(k)
$$
由此可见，对于二阶偏微分方程的定解问题，只有在半无界空间的 $x=0$ 端给出第二类边界条件时，才可以采用余弦变换。

**求半无界杆的热传导问题**
$$
\begin{cases}
u_{t}-a^2u_{xx}=0 &(x>0)\\
u|_{t=0}=0 \\
u|_{x=0}=u_0
\end{cases}
$$
解：采用傅里叶正弦变换，定解问题变为
$$
\begin{cases}
U'+k^2a^2U=ka^2u_0 \\
U(0)=0
\end{cases}
$$
其中 $U(k,t)$ 是 $u(x,t)$ 关于 $x$ 的傅里叶正弦变换。这个常微分方程初值问题的解为
$$
U(k,t)=\frac{u_0}{k}(1-e^{-k^2a^2t})
$$

最后，对 $U(k,t)$ 做傅里叶逆变换，结果是
$$
u(x,t)=\frac{2u_0}{\pi}\int_0^{+\infty}\frac{1}{k}(1-e^{-k^2a^2t})\sin kxdk
$$
通常记==误差函数== (error function)
$$
\mathrm{erf}(x)=\frac{2}{\sqrt{\pi}}\int_0^xe^{-s^2}ds
$$
和==余误差函数== (error function complement) 
$$
\mathrm{erfc}(x)=1-\mathrm{erf}(x)=\frac{2}{\sqrt{\pi}}\int_x^{\infty}e^{-s^2}ds
$$
故 $u(x,t)$ 可进一步变换为[^F2]
$$
u(x,t)=\frac{2u_0}{\pi}[\frac{\pi}{2}-\frac{\pi}{2}\mathrm{erf}(\frac{x}{2a\sqrt{t}})]
=u_0\mathrm{erfc}(\frac{x}{2a\sqrt{t}})
$$

[^F2]: 这里要用到拉普拉斯变换得到的两个公式
    $$
    \int_0^{+\infty}\frac{\sin kx}{k}dk=\frac{\pi}{2} \\
    \int_0^{+\infty}\frac{\sin kx}{k}e^{-k^2a^2t}dk=
    \sqrt{\pi}\int_0^{x/2a\sqrt{t}}e^{-\xi^2}d\xi
    $$

**求三维无界空间中的波动问题**
$$
\begin{cases}
u_{tt}-a^2Δu=0 \\
u|_{t=0}=\phi(x,y,z),\cfrac{∂u}{∂t}|_{t=0}=\psi(x,y,z)
\end{cases}
$$
解：作三重傅里叶变换，记 $\mathbf r=(x,y,z),\mathbf k=(k_1,k_2,k_3)$，定解问题变为
$$
\begin{cases}
U''+\mathbf k^2a^2U=0 \\
U(0)=\Phi(\mathbf k),\quad U'(0)=\Psi(\mathbf k)
\end{cases}
$$
其中 $U(\mathbf k,t),\Phi(\mathbf k),\Psi(\mathbf k)$ 分别是 $u(\mathbf r,t),\phi(\mathbf r),\psi(\mathbf r)$ 关于 $\mathbf r$ 的三维傅里叶变换。这个常微分方程初值问题的解为
$$
U(\mathbf k,t)=\Phi(\mathbf k)\cos kat+\Psi(\mathbf k)\frac{\sin kat}{ka}
$$

其中 $k=|\mathbf k|=\sqrt{k_1^2+k_2^2+k_3^2}$ ，再进行傅里叶逆变换
$$
\begin{aligned}
u(\mathbf r,t)&=\frac{1}{(2\pi)^3}\iiint\limits_{-\infty}^{+\infty}
[\Phi(\mathbf k)\cos kat+\Psi(\mathbf k)\frac{\sin kat}{ka}]
e^{\mathrm i\mathbf{k\cdot r}}d\mathbf k \\
&=\frac{1}{4\pi a}\frac{∂}{∂t}\iint\limits_{S_{at}^{\mathbf r}}\frac{\phi(\mathbf r)}{at}dS
+\frac{1}{4\pi a}\frac{∂}{∂t}\iint\limits_{S_{at}^{\mathbf r}}\frac{\psi(\mathbf r)}{at}dS
\end{aligned}
$$
上式称为==泊松公式==。式中 $S_{at}^{\mathbf r}$ 表示以 $\mathbf r$ 为圆心，以 $at$ 为半径的球面，$dS$ 表示 $S_{at}^{\mathbf r}$ 的面积元。

## 拉普拉斯变换法

拉普拉斯变换法适合求解初值问题，不管方程和边界条件是否为齐次的。

<kbd>Laplace变换</kbd>：设函数$f(t)$ 在 $t\geqslant 0$ 时有定义，且积分 $\displaystyle\int_{0}^{+∞}f(t)e^{-st}dt$ 收敛，则此积分所确定的函数
$$
\displaystyle F(s)=\int^{+\infty}_{0}f(t)e^{-st}\text{d}t\tag{2.1}
$$
称为函数 $f(t)$ 的 ==Laplace 变换==，记为 $F(s)=\mathcal L[f(t)]$，函数 $F(s)$ 也可称为 $f(t)$的象函数。
<kbd>Laplace逆变换</kbd>：令 $s=β+iω$ ，则有
$$
\displaystyle f(t)=\dfrac{1}{2\pi i}\int_{β-iω}^{β+iω}F(s)e^{st}\text{d}s \quad(t>0)
$$
称为 ==Laplace 逆变换==，记为 $f(t)=\mathcal L^{-1}[F(s)]$ 。在Laplace 变换中，只要求$f(t)$在 $[0,+∞)$ 内有定义即可。为了研究方便，以后总假定在$(−∞,0)$ 内，$f(t)≡0$
还可用留数就算拉普拉斯逆变换：设在复平面内只有有限个孤立奇点 $s_1,s_2,\cdots,s_n$ ，实数 $β$使这些奇点全在半平面 $\text{Re}(s)<β$ 内，且 $\lim\limits_{s\to∞}F(s)=0$ ，则有 
$$
\displaystyle f(t)=\sum_{k=1}^n\text{Res}[F(s)e^{st},s_k]\quad(t>0)
$$

**求半无界杆的热传导问题**
$$
\begin{cases}
u_{t}-a^2u_{xx}=0 &(x>0)\\
u|_{t=0}=0 &(x>0)\\
u|_{x=0}=u_0
\end{cases}
$$
解：对方程和边界条件关于 $t$ 进行拉普拉斯变换，采用[微分性质][F]，变换结果为
$$
\begin{cases}
sU-a^2U''=0 \\
U(0)=\cfrac{1}{s}u_0
\end{cases}
$$
其中 $U(s,x)$ 是 $u(x,t)$ 关于 $t$ 的傅里叶正弦变换。这个常微分方程通解为
$$
U(s,x)=A\exp(-\frac{\sqrt{sx}}{a})+B\exp(\frac{\sqrt{sx}}{a})
$$
考虑到自然边界条件 $\lim\limits_{x\to\infty}U$应为有限值，带入初值条件可求得
$$
U(s,x)=\cfrac{1}{s}u_0\exp(-\frac{\sqrt{sx}}{a})
$$
最后，对 $U(s,x)$ 进行拉普拉斯逆变换[^L1]
$$
u(x,t)=u_0\mathrm{erfc}(\frac{x}{2a\sqrt{t}})
$$

[^L]: 这里用到拉普拉斯反演公式
    $$
    \begin{aligned}
    & \mathcal F^{-1}[\cfrac{1}{\sqrt{s}}\exp(-a\sqrt{s})]=\cfrac{1}{\sqrt{\pi t}}\exp(-\cfrac{a^2}{4t}) \quad(a>0)\\
    & \mathcal F^{-1}[\cfrac{1}{s}\exp(-a\sqrt{s})]=\mathrm{erfc}(\cfrac{a}{2\sqrt{t}}) \quad(a>0)
    \end{aligned}
    $$

**求长 $l$ 均匀细杆的热传导问题**
$$
\begin{cases}
u_{t}-a^2u_{xx}=0 &(0<x<l)\\
u|_{t=0}=0 &(0<x<l) \\
u|_{x=0}=u_0, \quad \cfrac{∂u}{∂x}|_{x=l}=0
\end{cases}
$$
解：对方程和边界条件关于 $t$ 进行拉普拉斯变换，采用[微分性质][F]，变换结果为
$$
\begin{cases}
sU-a^2U''=0 \\
U(0)=\cfrac{1}{s}u_0,\quad U'(l)=0
\end{cases}
$$
其中 $U(s,x)$ 是 $u(x,t)$ 关于 $t$ 的傅里叶正弦变换。这个二阶常微分方程的解为
$$
U(s,x)=\cfrac{1}{s}u_0\cfrac{\cosh\frac{(l-x)\sqrt{s}}{a}}{\cosh\frac{l\sqrt{s}}{a}}
$$
最后，利用留数定理对 $U(s,x)$ 进行拉普拉斯逆变换
$$
u(x,t)=u_0-\cfrac{\pi u_0}{4}\sum_{n=0}^{\infty}\cfrac{1}{2n+1}
\sin\left(\cfrac{2n+1}{2l}\pi x\right)
\exp\left[-\left(\cfrac{2n+1}{2l}\pi\right)^2a^2t\right]
$$

# 基本解和格林函数

格林函数，又称点源影响函数，代表一个点源在一定的边界条件和（或）初始条件下所产生的场。均匀分布的函数可看做点源的叠加，该想法来源于静电场叠加原理。实际上，这种做法只不过是利用了偏微分方程的积分叠加原理。

## 泊松方程的基本解

**引例**：先举一个静电场的例子，设无界空间中电荷密度为 $\rho(\mathbf r)$ ，这样在坐标 $\mathbf r_0=(x_0,y_0,z_0)$ 的体积元 $dV_0$ 内的电荷量为 $\rho(\mathbf r_0)dV_0$ ，它在空间点 $\mathbf r=(x,y,z)$ 产生的电势为
$$
\cfrac{1}{4πε_0}\cfrac{\rho(\mathbf r_0)}{|\mathbf{r-r_0}|}
$$
根据电势叠加原理，可叠加求得任意密度分布引起的总电势分布
$$
φ(\mathbf r)=\cfrac{1}{4πε_0}\iiint\cfrac{\rho(\mathbf r_0)}{|\mathbf{r-r_0}|}dV_0
$$
**泊松方程的基本解**：对于无界空间的泊松方程
$$
Δu=f(\mathbf r)\tag{1.1}
$$
在物理上可看做电荷密度分布 $-ε_0f(\mathbf r)$ 在无界空间的电势方程。为了研究点源产生的场， $δ$ 函数恰是一个表示点源密度的函数，由 $δ$ 函数的性质知
$$
\iiint f(\mathbf r_0)δ(\mathbf{r-r_0})\mathrm dV_0=f(\mathbf r)
$$
这说明一般源 $f(\mathbf r)$ 可看成 $\mathbf r_0$ 点的点源 $f(\mathbf r_0)δ(\mathbf{r-r_0})$ 的积分叠加，由第一章积分叠加原理知道，只需求出方程
$$
ΔG(\mathbf{r,r_0})=δ(\mathbf{r-r_0})\tag{1.2}
$$
的解 $G(\mathbf{r,r}_0)$ ，便可得无界空间泊松方程的解
$$
u(\mathbf r)=\iiint G(\mathbf{r,r}_0)f(\mathbf r_0)\mathrm dV_0\tag{1.3}
$$
其中 $G(\mathbf{r,r}_0)$ 称为==格林函数==，又称==点源影响函数==。
由于是在无界空间，不妨先做平移变换，求方程（拉普拉斯算符平移不变性）
$$
ΔU(\mathbf{r})=δ(\mathbf{r})\tag{1.4}
$$
的解 $U(\mathbf r)$，称为==基本解==，代表置于原点的点源引起的场。则
$$
G(\mathbf{r,r_0})=U(\mathbf{r-r_0})\tag{1.5}
$$
进而有
$$
u(\mathbf r) =\iiint U(\mathbf{r-r_0})f(\mathbf r_0)\mathrm dV_0  
=U(\mathbf{r})*f(\mathbf r)\tag{1.6}
$$
**基本解的求法**：在物理上， 基本解 $U(\mathbf r)$ 描述了位于原点电荷量为 $-ε_0$ 的电荷在无界空间 $\mathbf r$ 处的电势。这里基本解没有定解条件限制，因此不是惟一的，通常根据问题的物理意义和数学的需要选定其中一个。
下面介绍数学上的一种求解方法。将方程用球坐标表示，当 $r\neq0$ 时，由于函数关于 $r$ 对称，所以方程化为
$$
\cfrac{1}{r^2}\cfrac{d}{dr}(r^2\cfrac{dU}{dr})=0
$$
其解为
$$
U=-\cfrac{c_1}{r}+c_2
$$
一般令无穷远处 $U=0$，则 $c_2=0$ 。为了求出 $c_1$ ，将方程包含原点的区域（不妨取以 $\epsilon$ 为半径的球 $V_ϵ$）进行体积分，利用 $δ$ 函数的性质
$$
\iiint\limits_{V_ϵ}ΔUdV=1
$$
利用格林公式，将上述体积分化为面积分
$$
\iiint\limits_{V_ϵ}ΔUdV=\iint\limits_{Σ_ϵ}\cfrac{∂U}{∂r}dS
=\int_0^{2\pi}\int_0^{\pi} r^2\sinθ dθ dϕ=4\pi c_1
$$
于是 $c_1=\cfrac{1}{4\pi}$ ，从而==三维泊松方程基本解==
$$
U(\mathbf{r})=-\cfrac{1}{4πr}\tag{1.7}
$$
类似的，用平面极坐标可求得==二维泊松方程的基本解==
$$
U(\mathbf{r})=-\cfrac{1}{2π}\ln\cfrac{1}{r}=\cfrac{1}{2π}\ln r \tag{1.8}
$$

## 泊松方程的格林函数

**泊松方程的边值问题**：泊松方程第一、第二、第三类边值问题可统一表示为
$$
\begin{cases}
Δu=f(\mathbf r) & (\mathbf r\in V) \\
(α\cfrac{∂u}{∂n}+βu)\Big|_{Σ}=\phi(\mathbf r) & (\mathbf r\in Σ)
\end{cases}\tag{2.1}
$$
若 $α=0,β\neq0$ 为第一类边值问题；若 $α\neq0,β=0$ 为第二类边值问题；若 $α\neq0,β\neq0$ 为第三类边值问题。由上节知道格林函数满足方程
$$
ΔG(\mathbf{r,r_0})=δ(\mathbf{r-r_0})\tag{2.2}
$$
从物理上看，格林函数就是位于 $\mathbf r_0$点电荷量为 $-ε_0$ 的电荷在区域 $V$ 内 $\mathbf r$ 点产生的电势。
现在，我们开始使用格林公式，叠加出泊松方程边值问题的解。为此，我们将 (2.1) 中泊松方程和方程 (2.2) 分别乘上 $G(\mathbf{r,r_0})$ 和 $u(\mathbf r)$ ，相减，然后在区域 $V$ 内积分，得到
$$
\iiint\limits_{V}(GΔu-uΔG)dV =\iiint\limits_{V}GfdV-\iiint\limits_{V}u(\mathbf r)δ(\mathbf{r-r_0})dV
$$
根据格林公式，可以将上式左端化为面积分
$$
\iiint\limits_{V}(GΔu-uΔG)dV=\iint\limits_{Σ}(G\cfrac{∂u}{∂n}-u\cfrac{∂G}{∂n})\mathrm{d}S
$$
右端第二项根据 $δ$ 函数的性质可以得到
$$
\iiint\limits_{V}u(\mathbf r)δ(\mathbf{r-r_0})dV=u(\mathbf r_0)
$$
于是，可以得到
$$
u(\mathbf r_0)=\iiint\limits_{V}G(\mathbf{r,r_0})f(\mathbf r)dV
-\iint\limits_{Σ}[G(\mathbf{r,r_0})\cfrac{∂u(\mathbf r)}{∂n}
-u(\mathbf r)\cfrac{∂G(\mathbf{r,r_0})}{∂n}]\mathrm{d}S \tag{2.3}
$$
上式称为==泊松方程的基本积分公式==。
需要注意的是，$G(\mathbf{r,r_0})$ 在 $\mathbf{r=r_0}$ 是不连续的，格林公式并不适用。严格的证明是，先在区域 $V$ 内奇点 $\mathbf r_0$ 处挖去半径为 $ε$ 的球形区域 $V_ε$ ，应用格林公式，再令 $ε\to 0$ 取极限求得。今后类似使用时将不再加以说明。
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/Poisson-equation.png)



基本积分公式将泊松方程的解 $u$ 用体积分和边界上的面积分表示了出来。对于面积分，$u$ 的边界条件是已知的，如果我们对 $G(\mathbf{r,r_0})$ 提出适当的边界条件，从而获得格林函数确切的解，就可以将 $u$ 确切的表示出来。

(1) 如果泊松方程满足第一类边界条件
$$
u\Big|_{Σ}=\phi(\mathbf r)\quad(\mathbf r\in Σ)
$$
同时要求 $G(\mathbf{r,r_0})$ 满足第一类齐次边界条件，即解决边值问题
$$
\begin{cases}
ΔG(\mathbf{r,r_0})=δ(\mathbf{r-r_0}) \\
G\Big|_{Σ}=0
\end{cases}
$$
则积分公式 (2.3) 含 $\cfrac{∂u}{∂n}$ 的一项为零，所以不需要知道$\cfrac{∂u}{∂n}$ 在边界上的值，上述边值问题的解称为==泊松方程第一边值问题的格林函数==。在物理上，可看做边界接地条件下，区域 $V$ 内 $\mathbf r_0$ 点（点源）电荷为 $-ε_0$ 的点源在 $V$ 内 $\mathbf r$ 点的电势。基本积分公式
$$
u(\mathbf r_0)=\iiint\limits_{V}G(\mathbf{r,r_0})f(\mathbf r)dV
+\iint\limits_{Σ}\phi(\mathbf r)\cfrac{∂G(\mathbf{r,r_0})}{∂n}\mathrm{d}S
$$
(2) 如果泊松方程满足第三类边界条件
$$
(α\cfrac{∂u}{∂n}+βu)\Big|_{Σ}=\phi(\mathbf r)\quad(\mathbf r\in Σ)
$$
令 $G(\mathbf{r,r_0})$ 满足第三类齐次边界条件，即解决边界问题
$$
\begin{cases}
ΔG(\mathbf{r,r_0})=δ(\mathbf{r-r_0}) \\
(α\cfrac{∂G}{∂n}+βG)\Big|_{Σ}=0
\end{cases}
$$
分别用 $G,u$ 或 $\cfrac{∂G}{∂n},\cfrac{∂u}{∂n}$ 交叉相乘上述两方程，并相减，可以得到
$$
(G\cfrac{∂u}{∂n}-u\cfrac{∂G}{∂n})\Big|_{Σ}
=\cfrac{1}{α}G\phi=-\cfrac{1}{β}\cfrac{∂G}{∂n}\phi
$$
上述边值问题的解称为==泊松方程第三边值问题的格林函数==，此时
$$
\begin{aligned}
u(\mathbf r_0) &=\iiint\limits_{V}G(\mathbf{r,r_0})f(\mathbf r)dV 
-\cfrac{1}{α}\iint\limits_{Σ}\phi(\mathbf r)G(\mathbf{r,r_0})\mathrm{d}S \\
&=\iiint\limits_{V}G(\mathbf{r,r_0})f(\mathbf r)dV 
+\cfrac{1}{β}\iint\limits_{Σ}\phi(\mathbf r)\cfrac{∂G(\mathbf{r,r_0})}{∂n}\mathrm{d}S
\end{aligned}
$$
(3) 至于第二类边界条件
$$
\cfrac{∂u}{∂n}\Big|_{Σ}=\phi(\mathbf r)\quad(\mathbf r\in Σ)
$$
似乎可以按照上面的方法，即解决边值问题
$$
\begin{cases}
ΔG(\mathbf{r,r_0})=δ(\mathbf{r-r_0}) \\
\cfrac{∂G}{∂n}\Big|_{Σ}=0
\end{cases}
$$
在格林公式中，令 $u(\mathbf r)=1,v(\mathbf r)=G(\mathbf{r,r_0})$ ，则有
$$
\iiint\limits_{V}ΔG(\mathbf{r,r_0})\mathrm{d}V
=\iint\limits_{Σ}∇G(\mathbf{r,r_0})\cdot\mathrm{d}\mathbf S
=\iint\limits_{Σ}\cfrac{∂G(\mathbf{r,r_0})}{∂n}\mathrm{d}S
$$
对边值问题中方程进行积分，根据 $δ$ 函数的性质又得到
$$
\iiint\limits_{V}ΔG(\mathbf{r,r_0})\mathrm{d}V=1
$$
于是格林函数在边界上的积分必须满足
$$
\iint\limits_{Σ}\cfrac{∂G(\mathbf{r,r_0})}{∂n}\mathrm{d}S=1\neq0
$$
这显然和上述边值问题中边界条件是矛盾的，第二类齐次边界问题一定无解。此时，需要引进广义的格林函数
$$
\begin{cases}
ΔG(\mathbf{r,r_0})=δ(\mathbf{r-r_0})-\cfrac{1}{v} \\
\cfrac{∂G}{∂n}\Big|_{Σ}=0
\end{cases}
$$
其中 $v$ 是区域 $V$ 的体积，并且当且仅当 
$$
\iiint\limits_{V}f(\mathbf r)dV=
-\iint\limits_{Σ}\phi(\mathbf r)\mathrm{d}S
$$
相应边值问题解的积分公式为
$$
u(\mathbf r_0)=\iiint\limits_{V}G(\mathbf{r,r_0})f(\mathbf r)dV
-\iint\limits_{Σ}\phi(\mathbf r)G(\mathbf{r,r_0})\mathrm{d}S
$$
**格林函数的对称性**：上述积分公式，似乎没有明确的物理意义。接下来我们先讨论格林函数的一个重要性质
$$
G(\mathbf{r_1,r_2})=G(\mathbf{r_2,r_1})\tag{2.4}
$$
引入两个格林函数 $G(\mathbf{r,r_1}),G(\mathbf{r,r_2})$，以第一类边界问题为例
$$
\begin{cases}
ΔG(\mathbf{r,r_1})=δ(\mathbf{r-r_1}) \\
G(\mathbf{r,r_1})\Big|_{Σ}=0
\end{cases},\quad
\begin{cases}
ΔG(\mathbf{r,r_2})=δ(\mathbf{r-r_2}) \\
G(\mathbf{r,r_2})\Big|_{Σ}=0
\end{cases}\quad(\mathbf{r,r_1,r_2}\in V)
$$
利用 $δ$ 函数的性质和格林公式，上述方程可以得到
$$
\begin{aligned}
& G(\mathbf{r_1,r_2})-G(\mathbf{r_2,r_1})  \\
= & \iiint\limits_{V}[G(\mathbf{r,r_2})δ(\mathbf{r-r_1})
-G(\mathbf{r,r_1})δ(\mathbf{r-r_2})]\mathrm{d}V \\
= & \iiint\limits_{V}[G(\mathbf{r,r_2})ΔG(\mathbf{r,r_1})
-G(\mathbf{r,r_1})ΔG(\mathbf{r,r_2})]\mathrm{d}V \\ 
=& \iint\limits_{Σ}[G(\mathbf{r,r_2})∇G(\mathbf{r,r_1})
-G(\mathbf{r,r_1})∇G(\mathbf{r,r_2})]\cdot\mathrm{d}\mathbf S
\end{aligned}
$$
带入边界条件，可得出面积分等于零，于是
$$
G(\mathbf{r_1,r_2})=G(\mathbf{r_2,r_1})
$$
对于第二类、第三类边界条件也可以得到同样的结果。

**综上所述**：对于泊松方程积分公式中的 $\mathbf r$ 和 $\mathbf r_0$ 互换下位置，并利用格林函数的对称性可得
第一边值问题解的积分表达式为
$$
u(\mathbf r)=\iiint\limits_{V}G(\mathbf{r,r_0})f(\mathbf r_0)\mathrm dV_0
+\iint\limits_{Σ}\phi(\mathbf r_0)\cfrac{∂G(\mathbf{r,r_0})}{∂n}\mathrm{d}S_0
$$
第二边值问题解的积分表达式为
$$
u(\mathbf r)=\iiint\limits_{V}G(\mathbf{r,r_0})f(\mathbf r_0)\mathrm dV_0
-\iint\limits_{Σ}\phi(\mathbf r_0)G(\mathbf{r,r_0})\mathrm{d}S_0
$$
第三边值问题解的积分表达式为
$$
\begin{aligned}
u(\mathbf r)&=\iiint\limits_{V}G(\mathbf{r,r_0})f(\mathbf r_0)\mathrm dV_0
-\cfrac{1}{α}\iint\limits_{Σ}\phi(\mathbf r_0)G(\mathbf{r,r_0})\mathrm{d}S_0 \\
&=\iiint\limits_{V}G(\mathbf{r,r_0})f(\mathbf r_0)dV_0 
+\cfrac{1}{β}\iint\limits_{Σ}\phi(\mathbf r_0)\cfrac{∂G(\mathbf{r,r_0})}{∂n}\mathrm{d}S_0
\end{aligned}
$$
此时，积分公式有明确的物理意义，右边第一个积分表示在区域 $V$ 内分布的点源在 $\mathbf r$ 处产生的场的总和，而第二项则代表边界面上感生场对 $\mathbf r$ 处场的影响的总和。

## 用镜像法求格林函数

泊松方程第一、第三类边值问题对应的格林函数满足
$$
\begin{cases}
ΔG=δ(\mathbf{r-r_0}) & (\mathbf r\in V) \\
(α\cfrac{∂G}{∂n}+βG)\Big|_{Σ}=0 & (\mathbf r\in Σ)
\end{cases}\tag{3.1}
$$
下面介绍格林函数的两种解法

- 第一种是按相应齐次问题及边界条件的本征函数展开，用分离变量法求得，但这样得到的解往往是无穷级数。
- 格林函数的物理意义启发我们，对于某些特殊区域，格林函数可以通过镜像法求得，可以取得有限形式的解。

**镜像法**：例如泊松方程第一边值问题的格林函数
$$
\begin{cases}
ΔG=δ(\mathbf{r-r_0}) &(\mathbf r\in V)\\
G|_Σ=0 &(\mathbf r\in Σ)
\end{cases}
$$
在物理上可理解为，一接地导体 $V$ 内 $\mathbf r_0$ 点电荷量为 $-ε_0$ 的点电荷在 $V$ 内 $\mathbf r$ 点的电势。
由叠加原理，通常将格林函数 $G$ 分成两部分
$$
G(\mathbf{r,r_0})=G_0(\mathbf{r,r_0})+G_1(\mathbf{r,r_0})
$$
其中 $U$ 满足
$$
ΔG_0=δ(\mathbf{r-r_0})
$$
是 $\mathbf r_0$ 点的点电荷产生的场，$G_1$ 满足
$$
\begin{cases}
ΔG_1=0 \\
G_1|_Σ=-G_0|_Σ 
\end{cases}
$$
是导体内 $\mathbf r_0$ 点的点电荷在边界上的感应电荷产生的场。
利用第一节中的基本解可知，在三维情形下
$$
G_0(\mathbf{r,r_0})=-\cfrac{1}{4π}\cfrac{1}{|\mathbf{r-r_0}|}\tag{3.2}
$$
类似的，二维情形下
$$
G_0(\mathbf{r,r_0})=-\cfrac{1}{2π}\ln\cfrac{1}{|\mathbf{r-r_0}|}=
\cfrac{1}{2π}\ln |\mathbf{r-r_0}|\tag{3.3}
$$
由于区域 $V$ 外的电源在 $V$ 内产生的场满足拉普拉斯方程，镜像法的中心思想是把边界上的感生电荷用一个等价的点电荷（像电荷）代替，困难在于 $V$ 内点电荷的电场在边界上必须和像电荷的电场相抵消，只有在某些特殊区域（例如，球形，半无界空间，等等）才能实现。

**求球内泊松方程第一边值问题格林函数**：
$$
\begin{cases}
ΔG=δ(\mathbf{r-r_0}) &(0<r,r_0<a)\\
G|_{r=a}=0 
\end{cases}
$$
(1) 像电荷如果存在的话，一定在球外。这是由于感应电荷的电势在球内是处处连续的，在球内的任何电荷都不能产生同样的效果。
(2) 考虑到对称性，这个像电荷一定存在于真实电荷所在半径的延长线上。
记球内电荷位于点 $Q_0(\mathbf r_0)$ ，像电荷位于点 $Q_1(\mathbf r_1)$ 电量为 $q$ ，如图

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/Poisson-equation2.png)
在球内任取一点 $P(\mathbf r)$ ，由叠加原理知道，总电势是球内电荷产生的电势和像电荷产生的电势叠加
$$
G=-\cfrac{1}{4π}\cfrac{1}{\mathbf{|r-r_0|}}
+\cfrac{q}{4πε_0}\cfrac{1}{\mathbf{|r-r_1|}}
$$
引入球坐标系（原点在球心），由于 $\mathbf{r_0,r_1}$ 共线，设 $\mathbf r_1=λ\mathbf r_0$ 则
$$
\mathbf{|r-r_0|}=PQ_0=\sqrt{r^2-2rr_0\cosγ+r_0^2} \\
\mathbf{|r-r_1|}=PQ_1=\sqrt{r^2-2λrr_0\cosγ+λ^2r_0^2}
$$
其中 $γ$ 是矢径 $\mathbf r$ 和 $\mathbf r_0$ $(\mathbf r_1)$ 之间的夹角
$$
\cosγ=\cosθ\cosθ_0+\sinθ\sinθ_0\cos(ϕ-ϕ_0)
$$
当观察点 $P$ 位于球面上时，考虑边界条件
$$
G|_{r=a}=0
$$
可得到球面上
$$
-\cfrac{1}{4π}\cfrac{1}{\sqrt{a^2-2ar_0\cosγ+r_0^2}}
+\cfrac{q}{4πε_0}\cfrac{1}{\sqrt{a^2-2λar_0\cosγ+λ^2r_0^2}}=0
$$
整理移项得
$$
\cfrac{q}{ε_0}\sqrt{a^2-2ar_0\cosγ+r_0^2}-\sqrt{a^2-2λar_0\cosγ+λ^2r_0^2}=0
$$
为使上式在球面上恒成立（与球坐标 $θ,ϕ$ 无关），可以得到
$$
\begin{cases}
\cfrac{q}{ε_0}>0 \\
λ=(\cfrac{q}{ε_0})^2 \\
a^2+λ^2r_0^2=(\cfrac{q}{ε_0})^2(a^2+r_0^2)
\end{cases}
$$
于是我们可得到
$$
\begin{cases}
q=\cfrac{ε_0a}{r_0} \\
λ=(\cfrac{a}{r_0})^2
\end{cases}
$$
这个设想的等效电荷 $q$  称为球内点电荷的==点像==。这样，球内任意一点总电势为
$$
G(\mathbf{r,r_0})=-\cfrac{1}{4π}\Big(\cfrac{1}{\mathbf{|r-r_0|}}
-\cfrac{a}{r_0}\cfrac{1}{|\mathbf{r}-\mathbf{r_1}|}\Big)
$$
其中 $a$ 为球半径，点像位置 $\mathbf r_1=(\cfrac{a}{r_0})^2\mathbf{r_0}$ 
球坐标表示式为
$$
G(r,θ,ϕ)=-\cfrac{1}{4π}\Big(\cfrac{1}{\sqrt{r^2-2rr_0\cosγ+r_0^2}}
-\cfrac{a}{\sqrt{r_0^2r^2-2a^2rr_0\cosγ+a^4}}\Big)
$$
类似的，圆内泊松方程第一边值问题的格林函数
$$
\begin{cases}
Δ_2G=δ(\mathbf{r-r_0}) \\
G|_{r=a}=0
\end{cases}
$$
用电像法求得其解为
$$
G(\mathbf{r,r_0}) =-\cfrac{1}{2π}\Big(\ln\cfrac{1}{\mathbf{|r-r_0|}}
-\ln\cfrac{1}{\mathbf{|r-r_1|}}-\ln\cfrac{a}{r_0}\Big) 
$$
其中 $a$ 为圆半径，电像位置 $\mathbf r_1=(\cfrac{a}{r_0})^2\mathbf{r_0}$ 
极坐标表示式为
$$
G(r,θ)=-\cfrac{1}{4π}\Big[-\ln(r^2-2rr_0\cos(θ-θ_0)+r_0^2)
+\ln(r^2-2r\cfrac{a^2}{r_0}\cos(θ-θ_0)+\cfrac{a^4}{r_0^2})
+2\ln\cfrac{a}{r_0}\Big]
$$
**示例 1**：在球内求解拉普拉斯方程的第一边值问题
$$
\begin{cases}
Δu=0 & (r<a)\\
u|_{r=a}=f
\end{cases}
$$
解：其用格林函数表示的解为
$$
u(\mathbf r)=\iint\limits_{Σ}f(\mathbf r_0)\cfrac{∂G(\mathbf{r,r_0})}{∂n}\mathrm{d}S_0
$$
其中球内泊松方程第一边值问题的格林函数前面已用电像法求得，球坐标表示式如下
$$
G(r,θ,ϕ)=-\cfrac{1}{4π}\Big(\cfrac{1}{\sqrt{r^2-2rr_0\cosγ+r_0^2}}
-\cfrac{a}{\sqrt{r_0^2r^2-2a^2rr_0\cosγ+a^4}}\Big)
$$
注意到，在球面上外法线方向与 $r_0$ 所在半径的方向一致，因此
$$
\cfrac{∂G}{∂n_0}|_{r_0=a}=\cfrac{∂G}{∂r_0}|_{r_0=a}
=\cfrac{1}{4\pi a}\cfrac{a^2-r^2}{(a^2-2ar\cosγ+r^2)^{3/2}}
$$
带入第一边值问题解的积分公式
$$
u(r,θ,ϕ)=\cfrac{a^2-r^2}{4\pi a}\int_{0}^{2\pi}\mathrm{d}ϕ_0\int_{0}^{\pi}
\cfrac{f(θ_0,ϕ_0)}{(a^2-2ar\cosγ+r^2)^{3/2}}
\sinθ_0\mathrm{d}θ_0
$$
上式称为==球域上的泊松公式==。

**示例 2**：在圆内解拉普拉斯方程的第一边值问题
$$
\begin{cases}u_{xx}+u_{yy}=0 & (r<a)\\u|_{r=a}=f\end{cases}
$$
解：和上例用类似的方法可求得
$$
u(r,θ)=\cfrac{a^2-r^2}{2\pi}\int_{0}^{2\pi}
\cfrac{f(θ_0)}{a^2-2ar\cos(θ-θ_0)+r^2}\mathrm{d}θ_0
$$
**示例 3**：在上半空间 $z>0$ 内求解拉普拉斯方程的第一边值问题
$$
\begin{cases}Δu=0 & (z>0)\\u|_{z=0}=f\end{cases}
$$
解：其用格林函数表示的解为
$$
u(\mathbf r)=\iint\limits_{Σ}f(\mathbf r_0)\cfrac{∂G(\mathbf{r,r_0})}{∂n}\mathrm{d}S_0
$$
格林函数 $G(\mathbf{r,r_0})$ 满足的方程为
$$
\begin{cases}
ΔG=δ(x-x_0)δ(y-y_0)δ(z-z_0) \\
G|_{z=0}=0\end{cases}
$$
这相当于接地导体平面 $z=0$ 上方的电势，如图，在点 $Q_0(x_0,y_0,z_0)$ 处放置电荷量为 $-ε_0$ 的点电荷。电势可用电像法求得，设想在 $Q_1(x_0,y_0,-z_0)$ 放置电量为 $+ε_0$ 的点电荷，不难验证，在 $z=0$ 上电势处处为零，$Q_1$ 即为 $Q_0$ 的电像。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/Laplace-equation-demo.png)
格林函数
$$
\begin{aligned}
G(\mathbf{r,r_0})&=-\cfrac{1}{4π}\cfrac{1}{\mathbf{|r-r_0|}}
+\cfrac{1}{4π}\cfrac{1}{|\mathbf{r}-\mathbf{r_1}|} \\
&=-\cfrac{1}{4π}\cfrac{1}{\sqrt{(x-x_0)^2+(y-y_0)^2+(z-z_0)^2}} \\
&+\cfrac{1}{4π}\cfrac{1}{\sqrt{(x-x_0)^2+(y-y_0)^2+(z+z_0)^2}} 
\end{aligned}
$$
外法向方向与 $z_0$ 正方向一致，因此
$$
\cfrac{∂G}{∂n_0}|_{z_0=0}=-\cfrac{∂G}{∂z_0}|_{z_0=0}=
\cfrac{1}{2π}\cfrac{z}{[(x-x_0)^2+(y-y_0)^2+z^2]^{3/2}}
$$
于是可得==上半空间的泊松积分==
$$
u(x,y,z)=\cfrac{z}{2π}\iint\limits_{-\infty}^{+\infty}
\cfrac{f(x_0,y_0)}{[(x-x_0)^2+(y-y_0)^2+z^2]^{3/2}}\mathrm dx_0\mathrm dy_0
$$
**示例 4**：在圆内解拉普拉斯方程的第一边值问题
$$
\begin{cases}u_{xx}+u_{yy}=0 & (y>0)\\u|_{y=0}=f\end{cases}
$$
解：和上例用类似的方法可求得
$$
u(x,y,z)=\cfrac{y}{π}\int_{-\infty}^{+\infty}
\cfrac{f(x_0)}{(x-x_0)^2+y^2}\mathrm dx_0
$$

## 演化问题的基本解

**冲量原理**：考虑一维受迫振动定解问题
$$
\begin{cases}
\cfrac{∂^2u}{∂t^2}-a^2\cfrac{∂^2u}{∂x^2}=f(x,t) & (t>0,0<x<l)\\
u|_{x=0}=u|_{x=l}=0 \\
u|_{t=0}=0,\cfrac{∂u}{∂t}|_{t=0}=0 \\
\end{cases}
$$

这里 $f(x,t)=\cfrac{F(x,t)}{\rho}$ 是作用在弦单位长度单位质量上的外力。
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/homogenizing.png)

考虑 $[τ,τ+Δτ)$ 时间段内的位移变化，令 $w|_{t=τ}=0,w|_{t=τ+Δτ}=Δu$
$f(x,τ)Δτ$ 表示 $Δτ$ 内冲量，这个冲量使得系统的速度有一定的增量。现在，我们把 $Δτ$ 时间内的速度增量看成是 $t=τ$ 瞬时得到的，而在 $Δτ$ 的其余时间内没有冲量的作用，即在这段时间内没有力的作用，故方程是齐次的。$t=τ$ 时的集中速度可置于“初始”条件中，得到的关于瞬时力引起的振动的定解问题是
$$
\begin{cases}
\cfrac{∂^2w}{∂t^2}-a^2\cfrac{∂^2w}{∂x^2}=0 \\
w|_{x=0}=w|_{x=l}=0 \\
w|_{t=τ}=0,\cfrac{∂u}{∂t}|_{t=τ}=f(x,τ)Δτ \\
\end{cases}\quad( τ<t<τ+Δτ ,0<x<l)
$$
令 $w(x,t;τ,Δτ)=v(x,t;τ)Δτ$ 则
$$
\begin{cases}
\cfrac{∂^2v}{∂t^2}-a^2\cfrac{∂^2v}{∂x^2}=0 \\
v|_{x=0}=v|_{x=l}=0 \\
v|_{t=τ}=0,\cfrac{∂v}{∂t}|_{t=τ}=f(x,τ) \\
\end{cases}\quad( t>τ ,0<x<l)
$$
于是
$$
u(x,t) = \lim_{Δτ\to0}\sum_{τ=0}^tw(x,t;τ)
=\int_0^tv(x,t;τ)\mathrm dτ
$$

<kbd>冲量原理</kbd>：（齐次化原理）设 $L$ 是关于 $\mathbf r=(x_1,x_2,\cdots,x_n)$ 的线性偏微分算子，若 $v(\mathbf r,t;τ)$ 满足齐次方程定解问题
$$
\begin{cases}
\cfrac{∂^mv}{∂t^m}-L[v]=0 \\
(α\cfrac{∂v}{∂t}+βv)\Big|_{Σ}=0 \\
v\Big|_{t=τ}=\cfrac{∂v}{∂t}\Big|_{t=τ}=\cdots
=\cfrac{∂^{m-2}v}{∂t^{m-2}}\Big|_{t=τ}=0  \\
\cfrac{∂^{m-1}v}{∂t^{m-1}}\Big|_{t=τ}=f(\mathbf r,τ)
\end{cases}\quad(t>τ)
$$
则
$$
u(\mathbf r,t)=\int_0^tv(\mathbf r,t;τ)\mathrm dτ
$$
是以下非齐次方程定解问题的解
$$
\begin{cases}
\cfrac{∂^mu}{∂t^m}-L[u]=f(\mathbf r,t) \\
(α\cfrac{∂u}{∂t}+βu)\Big|_{Σ}=0 \\
u\Big|_{t=0}=\cfrac{∂u}{∂t}\Big|_{t=0}=\cdots
=\cfrac{∂^{m-1}u}{∂t^{m-1}}\Big|_{t=0}=0  
\end{cases}\quad(t>0)
$$
以无界热传导初值问题为例，若 $v(\mathbf r,t;τ)$ 满足齐次方程初值问题
$$
\begin{cases}
\cfrac{∂v}{∂t}-a^2Δv=0 & (t>τ)  \\
v|_{t=τ}=f(\mathbf r,τ)
\end{cases}
$$
则积分 $\displaystyle u(\mathbf r,t)=\int_0^tv(\mathbf r,t;τ)\mathrm dτ$ 满足非齐次方程初值问题
$$
\begin{cases}
\cfrac{∂u}{∂t}-a^2Δu=f(\mathbf r,t) & (t>0)  \\
u|_{t=0}=0
\end{cases}
$$


**热传导方程初值问题**：对于无界空间的热传导方程初值问题
$$
\begin{cases}
\cfrac{∂u}{∂t}-a^2Δu=f(\mathbf r,t) & (t>0)  \\
u|_{t=0}=\phi(\mathbf r)
\end{cases}\tag{4.1}
$$
由叠加原理知道，$u=u_1+u_2$ ，其中 $u_1$ 满足初值问题
$$
\begin{cases}
\cfrac{∂u_1}{∂t}-a^2Δu_1=0\quad(t>0)    \\
u_1|_{t=0}=\phi(\mathbf r)
\end{cases}
$$
代表在初始时刻瞬时给予热量的传导问题，$u_2$ 满足初值问题
$$
\begin{cases}
\cfrac{∂u_2}{∂t}-a^2Δu_2=f(\mathbf r,t)\quad(t>0)   \\
u_2|_{t=0}=0
\end{cases},\quad
$$
代表持续热源下的热传导。由 $δ$ 函数的性质知道，空间持续热源可看做瞬时点源的叠加
$$
f(\mathbf r,t)=\iiint\mathrm dV_0 \int f(\mathbf{r_0},t_0)δ(\mathbf{r-r_0})δ(t-t_0)\mathrm dt_0
$$
将空间点 $\mathbf{r_0}$ 在 $t_0$ 时刻的热源在 $\mathbf r,t$ 引起的温度记作 $G(\mathbf{r},t;\mathbf{r_0},t_0)$ 称为热传导方程的格林函数。 $G$ 满足的初值问题
$$
\begin{cases}
G_{t}-a^2ΔG=δ(\mathbf{r-r_0})δ(t-t_0) \\
G|_{t=t_0}=0\end{cases}\tag{4.2}
$$
方程中 $t=t_0$ 时刻即代表初始时刻，之前无任何热源作用，$t<t_0$ 时刻 $G$ 均为零。
取 $\tau=t-t_0$ 由冲量原理，若 $v(\mathbf r,\tau;\tau_0)$ 满足齐次方程初值问题
$$
\begin{cases}
v_{\tau}-a^2Δv=0  \\
v|_{\tau=\tau_0}=δ(\mathbf{r-r_0})δ(\tau_0)
\end{cases}
$$

则格林函数 $\displaystyle G=\int_0^\tau v(\mathbf r,\tau;\tau_0)\mathrm d\tau_0$ 
对 $v(\mathbf r,\tau;\tau_0)$ 满足的齐次方程和初值条件积分可以得到
$$
\begin{cases}
G_\tau-a^2ΔG=0 \\
G|_{\tau=\tau_0}=δ(\mathbf{r-r_0})
\end{cases}
$$
上式取 $\tau_0=0$ ，并将 $\tau=t-t_0$ 代回，可知格林函数满足的定解问题 (4.2) 等价于以下定解问题
$$
\begin{cases}
G_{t}-a^2ΔG=0 \\
G|_{t=t_0}=δ(\mathbf{r-r_0})\end{cases}
$$
由冲量原理和积分叠加原理，进一步可求得
$$
u_2(\mathbf r,t)=\iiint\mathrm dV_0\int_0^t
G(\mathbf{r},t;\mathbf{r_0},t_0)f(\mathbf r_0,t_0)\mathrm dt_0
$$
为求得 $G(\mathbf{r},t;\mathbf{r_0},t_0)$ ，不妨先求满足定解问题
$$
\begin{cases}
U_{t}-a^2ΔU=0  \\
U|_{t=0}=δ(\mathbf r) \end{cases}
\tag{4.3}
$$
的==基本解==  $U(\mathbf r,t)$ ，代表初始瞬间原点给予热量 $Q=c\rho$ 后的温度分布。做变量变换有
$$
G(\mathbf{r},t;\mathbf{r_0},t_0)=U(\mathbf{r-r_0},t-t_0)\tag{4.4}
$$
进而有
$$
u_2(\mathbf r,t)=\iiint\mathrm dV_0\int_0^t
U(\mathbf{r-r_0},t)f(\mathbf r_0,t_0)\mathrm dt_0
=\int_0^tU(\mathbf{r},t-t_0)*f(\mathbf r,t_0)\mathrm dt_0
$$
由于 $\phi(\mathbf r)=δ(\mathbf r)*\phi(\mathbf r)$ ，根据叠加原理有
$$
u_1(\mathbf r,t)=\iiint U(\mathbf{r-r_0},t)\phi(\mathbf r_0) \mathrm dV_0
=U(\mathbf{r},t)*\phi(\mathbf r)
$$
所以
$$
u(\mathbf{r},t)=U(\mathbf{r},t)*\phi(\mathbf r)+
\int_0^tU(\mathbf{r},t-t_0)*f(\mathbf r,t_0)\mathrm dt_0\tag{4.5}
$$
可带入 (4.1) 可直接验证上述结论。

**波动方程初值问题**：对于无界空间的波动方程初值问题
$$
\begin{cases}
\cfrac{∂^2u}{∂t^2}-a^2Δu=f(\mathbf r,t) & (t>0)  \\
u|_{t=0}=\phi(\mathbf r),\quad \cfrac{∂u}{∂t}|_{t=0}=\psi(\mathbf r)
\end{cases}\tag{4.6}
$$
同样由叠加原理知道，$u=u_1+u_2+u_3$ ，分别满足初值问题
$$
\begin{cases}
\cfrac{∂^2u_1}{∂t^2}-a^2Δu_1=f(\mathbf r,t)\quad(t>0)   \\
u_1|_{t=0}=\cfrac{∂u_1}{∂t}|_{t=0}=0
\end{cases}
$$
代表持续作用的力引起的波动
$$
\begin{cases}
\cfrac{∂^2u_2}{∂t^2}-a^2Δu_2=0\quad(t>0)    \\
u_2|_{t=0}=0,\quad \cfrac{∂u_2}{∂t}|_{t=0}=\psi(\mathbf r)
\end{cases}
$$
代表初始时刻瞬时冲量引起的波动
$$
\begin{cases}
\cfrac{∂^2u_3}{∂t^2}-a^2Δu_3=0\quad(t>0)    \\
u_3|_{t=0}=\phi(\mathbf r),\quad \cfrac{∂u_3}{∂t}|_{t=0}=0
\end{cases}
$$
同样林函数 $G(\mathbf{r},t;\mathbf{r_0},t_0)$ 满足的定解问题为
$$
\begin{cases}
G_{tt}-a^2ΔG=δ(\mathbf{r-r_0})δ(t-t_0) \\
G|_{t=t_0}=0,\quad \cfrac{∂G}{∂t}|_{t=t_0}=0
\end{cases}\tag{4.7}
$$
方程中 $t=t_0$ 时刻即代表初始时刻，之前无任何力的作用，$t<t_0$ 时刻 $G$ 均为零。
根据冲量原理和积分叠加原理定解问题 (4.7) 等价于
$$
\begin{cases}
G_{tt}-a^2ΔG=0 \\
G|_{t=t_0}=0,\quad \cfrac{∂G}{∂t}|_{t=t_0}=δ(\mathbf{r-r_0})
\end{cases}
$$
不妨先求 ==基本解==  $U(\mathbf r,t)$ 满足的定解问题
$$
\begin{cases}
U_{tt}-a^2ΔU=0  \\
U|_{t=0}=0,\quad \cfrac{∂U}{∂t}|_{t=0}=δ(\mathbf{r}) 
\end{cases}\tag{4.8}
$$
代表初始时刻在原点处的瞬时冲量引起的波动。变量代换可求得
$$
G(\mathbf{r},t;\mathbf{r_0},t_0)=U(\mathbf{r-r_0},t-t_0)\tag{4.9}
$$
根据冲量原理和积分叠加原理，进而有
$$
\begin{aligned}
& u_1(\mathbf r,t)=\iiint\mathrm dV_0\int_0^t
U(\mathbf{r-r_0},t-t_0)f(\mathbf r_0,t_0)\mathrm dt_0
=\int_0^tU(\mathbf{r},t-t_0)*f(\mathbf r,t_0)\mathrm dt_0  \\
& u_2(\mathbf r,t)=\iiint U(\mathbf{r-r_0},t)\phi(\mathbf r_0) \mathrm dV_0
=U(\mathbf{r},t)*\psi(\mathbf r)
\end{aligned}
$$

设 $u_3=\cfrac{∂v}{∂t}$ ，则有
$$
\begin{cases}
\cfrac{∂^2v}{∂t^2}-a^2Δv=0\quad(t>0)    \\
v|_{t=0}=0,\quad \cfrac{∂v}{∂t}|_{t=0}=\phi(\mathbf r)
\end{cases}
$$
即 $v=U(\mathbf r,t)*\phi(\mathbf r)$ ， $u_3=\cfrac{∂v}{∂t}$ 满足 $u_3$ 的初值问题，所以
$$
u(\mathbf r,t)=U(\mathbf r,t)*\psi(\mathbf r)+
\cfrac{∂}{∂t}[U(\mathbf r,t)*\phi(\mathbf r)]+
\int_0^tU(\mathbf{r},t-t_0)*f(\mathbf r,t_0)\mathrm dt_0\tag{4.10}
$$
**基本解的求法**： 基本解 $U(\mathbf r,t)$ 没有边界条件限制，因此不是惟一的，适当选取即可。

三维无界热传导方程初值问题的基本解：（可用傅里叶变换法求得）
$$
U(\mathbf r,t)=(\cfrac{1}{2a\sqrt{\pi t}})^3\exp(-\cfrac{|\mathbf r|^2}{4a^2t})
$$

三维无界波动方程初值问题的基本解：（可用傅里叶变换法求得）
$$
U(\mathbf r,t)=(\cfrac{1}{2\pi})^3\iiint\limits_{-\infty}^{-\infty}
\cfrac{\sin(r_0at)}{r_0a}\exp(\mathrm i\mathbf{r_0\cdot r})\mathrm dV_0
$$

其中 $r_0=|\mathbf r_0|,r=|\mathbf r|$ 。以 $\mathbf r$ 为极轴方向取球坐标，则
$$
U(\mathbf r,t)=\cfrac{δ(r-at)}{4\pi ar}
$$

## 一般演化问题的格林函数

**波动方程定解问题**：一般强迫振动波动方程的定解问题
$$
\begin{cases}
u_{tt}-a^2Δu=f(\mathbf r,t) & (t>0,\mathbf r\in V)  \\
(α\cfrac{∂u}{∂n}+βu)\Big|_{Σ}=σ(\mathbf r,t) & (t>0,\mathbf r\in Σ) \\
u|_{t=0}=\phi(\mathbf r),\quad \cfrac{∂u}{∂t}|_{t=0}=\psi(\mathbf r,) &(t>0,\mathbf r\in V)
\end{cases}\tag{5.1}
$$
由 $δ$ 函数的性质知道，持续力 $f(\mathbf r,t)$ 可表示为
$$
f(\mathbf r,t)=\iiint\limits_V\mathrm dV_0 \int_t 
f(\mathbf{r_0},t_0)δ(\mathbf{r-r_0})δ(t-t_0)\mathrm dt_0
$$
波动方程的格林函数$G(\mathbf{r},t;\mathbf{r_0},t_0)$ 满足的定解问题是
$$
\begin{cases}
G_{tt}-a^2ΔG=δ(\mathbf{r-r_0})δ(t-t_0) & (t,t_0>0,\mathbf r\in V) \\
(α\cfrac{∂G}{∂n}+βG)\Big|_{Σ}=0  & (t,t_0>0,\mathbf r\in Σ) \\
G|_{t=t_0}=0,\quad \cfrac{∂G}{∂t}|_{t=t_0}=0 & (t,t_0>0,\mathbf r\in V)
\end{cases}\tag{5.2}
$$
方程中 $t=t_0$ 时刻即代表初始时刻，$G|_{t<t_0}\equiv0$ 。我们可以用和解泊松方程类似的方法求解波动方程解的积分表达式，首先讨论格林函数的对称性。

**格林函数的对称性**
$$
G(\mathbf{r_1},t_1;\mathbf{r_2},t_2)=G(\mathbf{r_2},-t_2;\mathbf{r_1},-t_1)\tag{5.3}
$$
引入两个格林函数 $G(\mathbf{r},t;\mathbf{r_1},t_1),G(\mathbf{r},-t;\mathbf{r_2},-t_2)$，简记为 $G_1,G_2$ ，分别是下面定解问题的解
$$
\begin{cases}
\cfrac{∂^2G_1}{∂t^2}-a^2ΔG_1=δ(\mathbf{r-r_1})δ(t-t_1) \\
(α\cfrac{∂G_1}{∂n}+βG_1)\Big|_{Σ}=0 \\
G_1|_{t=t_1}=0,\quad \cfrac{∂G_1}{∂t}|_{t=t_1}=0
\end{cases}, \quad
\begin{cases}
\cfrac{∂^2G_2}{∂(-t)^2}-a^2ΔG_2=δ(\mathbf{r-r_2})δ(t+t_2) \\
(α\cfrac{∂G_2}{∂n}+βG_2)\Big|_{Σ}=0 \\
G_2|_{-t=-t_2}=0,\quad \cfrac{∂G_2}{∂t}|_{-t=-t_2}=0
\end{cases}
$$
其中 $\mathbf{r_1,r_2}\in V,\quad t>t_1,t_2>0$ 。
利用 $δ$ 函数的性质和格林公式，在空间区域 $V$ 和 时间区间$[0,t]$ 上积分，上述方程可以得到
$$
\begin{aligned}
& G(\mathbf{r_1},t_1;\mathbf{r_2},t_2)-G(\mathbf{r_2},-t_2;\mathbf{r_1},-t_1) \\
=& \iiint\limits_{V}\mathrm dV \int_0^t
 [G_2δ(\mathbf{r-r_1})δ(t-t_1)-G_1δ(\mathbf{r-r_2})δ(t+t_2)]\mathrm dt \\
= & \iiint\limits_{V}\mathrm dV \int_0^t
[G_2(\cfrac{∂^2G_1}{∂t^2}-a^2ΔG_1)-G_1(\cfrac{∂^2G_2}{∂(-t)^2}-a^2ΔG_2)]\mathrm dt \\ 
=& \iiint\limits_{V}\mathrm dV \int_0^t
[(G_2\cfrac{∂^2G_1}{∂t^2}-G_1\cfrac{∂^2G_2}{∂t^2})+a^2(G_1ΔG_2-G_2ΔG_1)]\mathrm dt \\
=& \iiint\limits_{V}\mathrm dV
[G_2\cfrac{∂G_1}{∂t}-G_1\cfrac{∂G_2}{∂t}]\Big|_{0}^t
+a^2\int_0^t\mathrm dt\iint\limits_{Σ}(G_1\cfrac{∂G_2}{∂n}-G_2\cfrac{∂G_1}{∂n})\mathrm dS 
 \end{aligned}
$$
将上述两个边界条件分别乘以 $G_2,G_1$ ，相减可以得到
$$
(G_1\cfrac{∂G_2}{∂n}-G_2\cfrac{∂G_1}{∂n})\Big|_Σ=0
$$
带入初始条件我们又可以得到
$$
[G_2\cfrac{∂G_1}{∂t}-G_1\cfrac{∂G_2}{∂t}]\Big|_{0}^t=0
$$
于是有
$$
G(\mathbf{r_1},t_1;\mathbf{r_2},t_2)=G(\mathbf{r_2},-t_2;\mathbf{r_1},-t_1)
$$
> tips：
>
> 1. $\cfrac{∂G}{∂t}=\cfrac{∂G}{∂(-t)}\cfrac{\mathrm d(-t)}{\mathrm dt}=-\cfrac{∂G}{∂(-t)}$ 
> 将 $t_1,t_2$ 位置互换时出现的负号，正好保证了时间的先后次序不变，否则就会有悖于因果律的要求。
> 2. $\cfrac{∂^2G}{∂t^2}=\cfrac{∂^2G}{∂(-t)^2}$
> 3. 波动方程中重要的偏微分
>     $\begin{aligned} 
>     &\cfrac{∂}{∂t}(G_2\cfrac{∂G_1}{∂t}-G_1\cfrac{∂G_2}{∂t})  \\
>     = & (\cfrac{∂G_2}{∂t}\cfrac{∂G_1}{∂t}+G_2\cfrac{∂^2G_1}{∂t^2})
>     -(\cfrac{∂G_1}{∂t}\cfrac{∂G_2}{∂t}+G_1\cfrac{∂^2G_2}{∂t^2}) \\
>     = & G_2\cfrac{∂^2G_1}{∂t^2}-G_1\cfrac{∂^2G_2}{∂t^2}
>     \end{aligned}$
> 4. 热传导方程中重要的偏微分
>     $G_2\cfrac{∂G_1}{∂t}+G_1\cfrac{∂G_2}{∂t}=\cfrac{∂(G_1G_2)}{∂t}$

**解的积分表达式**：将波动方程定解问题中的 $\mathbf r,t$ 改写成 $\mathbf r_0,t_0$ 
$$
\begin{cases}
\cfrac{∂^2u(\mathbf r_0,t_0)}{∂t_0^2}-a^2Δ_0u(\mathbf r_0,t_0)=f(\mathbf r_0,t_0) \\
[α\cfrac{∂u(\mathbf r_0,t_0)}{∂n_0}+βu(\mathbf r_0,t_0)]\Big|_{Σ}=σ(\mathbf r_0,t_0)  \\
u(\mathbf r_0,t_0)|_{t_0=0}=\phi(\mathbf r_0),\quad 
\cfrac{∂u(\mathbf r_0,t_0)}{∂t_0}|_{t_0=0}=\psi(\mathbf r_0)
\end{cases}
$$
再将格林函数定解问题中的 $\mathbf r,t$ 改换成 $\mathbf r_0,-t_0$ ，将 $\mathbf r_0,t_0$ 改换成 $\mathbf r,-t$ 同时利用对称关系，得
$$
\begin{cases}
\cfrac{∂^2G(\mathbf r,t;\mathbf r_0,t_0)}{∂t_0^2}
-a^2Δ_0G(\mathbf r,t;\mathbf r_0,t_0)=δ(\mathbf{r-r_0})δ(t-t_0) \\ 
[α\cfrac{∂G(\mathbf r,t;\mathbf r_0,t_0)}{∂n_0}
+βG(\mathbf r,t;\mathbf r_0,t_0)]\Big|_{Σ}=0 \\ 
G(\mathbf r,t;\mathbf r_0,t_0)|_{-t_0=-t}=0,\quad 
\cfrac{∂G(\mathbf r,t;\mathbf r_0,t_0)}{∂t_0}|_{-t_0=-t}=0
\end{cases}
$$
两方程交叉相乘 $G(\mathbf r,t;\mathbf r_0,t_0),u(\mathbf r_0,t_0)$ 相减，再积分
$$
\begin{aligned}
& \iiint\limits_V\mathrm dV_0\int_0^{t+0}
[G\cfrac{∂^2u(\mathbf r_0,t_0)}{∂t_0^2}-u(\mathbf r_0,t_0)\cfrac{∂^2G}{∂t_0^2}]\mathrm dt_0 \\
& -a^2\iiint\limits_V\mathrm dV_0\int_0^{t+0}
[GΔ_0u(\mathbf r_0,t_0)-u(\mathbf r_0,t_0)Δ_0G]\mathrm dt_0 \\
=& \iiint\limits_V\mathrm dV_0\int_0^{t+0} Gf(\mathbf r_0,t_0)\mathrm dt_0 \\
& - \iiint\limits_V\mathrm dV_0\int_0^{t+0} u(\mathbf r_0,t_0)δ(\mathbf{r-r_0})δ(t-t_0)\mathrm dt_0
\end{aligned}
$$
利用 $δ$ 函数的性质和格林公式可得到
$$
\begin{aligned}
u(\mathbf r,t)= & \iiint\limits_V\mathrm dV_0\int_0^{t+0} Gf(\mathbf r_0,t_0)\mathrm dt_0 \\
& -\iiint\limits_V
[G\cfrac{∂u(\mathbf r_0,t_0)}{∂t_0}-u(\mathbf r_0,t_0)\cfrac{∂G}{∂t_0}]\Big|_{t_0=0}^{t_0=t+0} \mathrm dV_0\\
&+a^2\int_0^{t+0}\mathrm dt_0
\iint\limits_Σ[G\cfrac{∂u(\mathbf r_0,t_0)}{∂n_0}-u(\mathbf r_0,t_0)\cfrac{∂G}{∂n_0}]\mathrm dS_0
\end{aligned}
$$
带入初始条件和边界条件
$$
\begin{aligned}
u(\mathbf r,t)= & \iiint\limits_V\mathrm dV_0\int_0^{t} Gf(\mathbf r_0,t_0)\mathrm dt_0 \\
& +\iiint\limits_V
[G\psi(\mathbf r_0)-\phi(\mathbf r_0)\cfrac{∂G}{∂t_0}]\Big|_{t_0=0}\mathrm dV_0 \\
&+a^2\int_0^{t}\mathrm dt_0\iint\limits_Σ
[G\cfrac{∂u(\mathbf r_0,t_0)}{∂n_0}-u(\mathbf r_0,t_0)\cfrac{∂G}{∂n_0}]\mathrm dS_0
\end{aligned}\tag{5.4}
$$
对于不同类型的边界条件条件，可令 $G$ 满足相应的齐次边界条件，从而得到适用于不同边界条件的解以 $G$ 表示的解的积分表达式。

**热传导方程定解问题**
$$
\begin{cases}
u_{t}-a^2Δu=f(\mathbf r,t) & (t>0,\mathbf r\in V)  \\
(α\cfrac{∂u}{∂n}+βu)\Big|_{Σ}=σ(\mathbf r,t) & (t>0,\mathbf r\in Σ) \\
u|_{t=0}=\phi(\mathbf r) &(t>0,\mathbf r\in V)
\end{cases}\tag{5.5}
$$
格林函数满足的定解问题是
$$
\begin{cases}
G_{t}-a^2ΔG=δ(\mathbf{r-r_0})δ(t-t_0) & (t,t_0>0,\mathbf r\in V) \\
(α\cfrac{∂G}{∂n}+βG)\Big|_{Σ}=0  & (t,t_0>0,\mathbf r\in Σ) \\
G|_{t=t_0}=0 & (t,t_0>0,\mathbf r\in V)
\end{cases}\tag{5.6}
$$
类似上面的讨论，同样可以得到解的积分表达式
$$
\begin{aligned}
u(\mathbf r,t)= & \iiint\limits_V\mathrm dV_0\int_0^{t} Gf(\mathbf r_0,t_0)\mathrm dt_0 \\
& +\iiint\limits_V
[G\phi(\mathbf r_0)]\Big|_{t_0=0}\mathrm dV_0 \\
&+a^2\int_0^{t}\mathrm dt_0\iint\limits_Σ
[G\cfrac{∂u(\mathbf r_0,t_0)}{∂n_0}-u(\mathbf r_0,t_0)\cfrac{∂G}{∂n_0}]\mathrm dS_0
\end{aligned}\tag{5.7}
$$

**格林函数的求法**

示例 1：对于一维热传导问题的格林函数
$$
\begin{cases}
G_{t}-a^2G_{xx}=δ(x-x_0)δ(t-t_0) & (t,t_0>0,0<x<l) \\
G|_{x=0}=G|_{x=l}=0  \\
G|_{t=t_0}=0 
\end{cases}
$$
由特征函数展开，分离变量法可求得
$$
G(x,t;x_0,t_0)=\cfrac{l}{2}\sum_{n=1}^{\infty}\exp[-(\cfrac{n\pi a}{l})^2(t-t_0)]
\sin\cfrac{n\pi x_0}{l}\sin\cfrac{n\pi x}{l}\quad(t>t_0)
$$
示例 2：对于一维受迫振动的格林函数
$$
\begin{cases}
G_{t}-a^2G_{xx}=δ(x-x_0)δ(t-t_0) & (t,t_0>0,0<x<l) \\
G|_{x=0}=G|_{x=l}=0  \\
G|_{t=t_0}=0,\quad \cfrac{∂G}{∂t}|_{t=t_0}=0 
\end{cases}
$$
由特征函数展开，分离变量法可求得
$$
G(x,t;x_0,t_0)=\cfrac{2}{\pi a}\sum_{n=1}^{\infty}\cfrac{1}{n}
\sin\cfrac{n\pi x_0}{l}\sin\cfrac{n\pi a(t-t_0)}{l}\sin\cfrac{n\pi x}{l}\quad(t>t_0)
$$


# 附录
## 静电场理论

**库伦定律**
$$
\mathbf F=\cfrac{1}{4πε_0}\cfrac{Qq}{r^2}\mathbf e_r\tag{1.1}
$$
**电场强度**：是用来表示电场的强弱和方向的物理量
$$
\mathbf{E}=\cfrac{\mathbf F}{q}=\cfrac{1}{4πε_0}\cfrac{Q}{r^2}\mathbf e_r\tag{1.2}
$$
**高斯定理**：穿过闭合曲面 $Σ$ 向外的电场强度通量等于闭合曲面围成的空间 $V$ 内电量的 $1/ε_0$，其中 $ε_0$ 为真空介电常数。
$$
\oiint\limits_{Σ}\mathbf E\cdot\mathrm d\mathbf S
=\cfrac{1}{ε_0}\iiint\limits_{V}\rho\mathrm dV\tag{1.3}
$$
简要证明：先对点电荷的场证明，再推广到一半的电荷分布。
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/Gauss-theorem.png)  ![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/solid-angle.png)

(1) 取包围点电荷 $q$ 的任意闭合曲面 $Σ$，在闭合曲面上任取面元 $\mathrm d\mathbf S$ ，由库伦定律计算电通量
$$
\mathrm d\Phi=\mathbf E\cdot \mathrm d\mathbf S
=\cfrac{1}{4πε_0}\cfrac{q}{r^2}\mathrm dS\cos θ
$$
面元 $\mathrm d\mathbf S$ 在球面上（以点电荷为球心，$r$ 为半径）的投影 $\mathrm dS_0=\mathrm dS\cos θ$ ，在此，引入球面立体角[^sr] $\mathrm dΩ=\cfrac{\mathrm dS_0}{r^2}$ ，然后对上式积分
$$
\oiint\limits_{Σ}\mathbf E\cdot\mathrm d\mathbf S
=\cfrac{q}{4πε_0}\oiint\limits_Σ\mathrm dΩ
=\cfrac{q}{ε_0}
$$
(2) 根据场强叠加原理，上式可进一步扩展为
$$
\oiint\limits_{Σ}\mathbf E\cdot\mathrm d\mathbf S
=\cfrac{1}{ε_0}\sum_{q_i\text{ in }V}q_i
$$
若电荷为连续分布，电荷密度为 $\rho$ 则上式可改写为
$$
\oiint\limits_{Σ}\mathbf E\cdot\mathrm d\mathbf S
=\cfrac{1}{ε_0}\iiint\limits_{V}\rho\mathrm dV
$$
通过数学上的[高斯公式][math]，上式左端可化为体积分
$$
\iiint\limits_{V}∇\cdot \mathbf E\mathrm{d}V
=\cfrac{1}{ε_0}\iiint\limits_{V}\rho\mathrm dV
$$
由于闭合曲面的随意性，取极限 $V\to M(x,y,z)$ 上式可化为微分形式
$$
∇\cdot \mathbf E=\cfrac{\rho}{ε_0}\tag{1.4}
$$
**静电场环路定理**： 在静电场中，场强沿任意闭合路径的线积分等于 0 
$$
\oint_{L}\mathbf E\cdot\mathrm dl=0\tag{1.5}
$$
取点电荷 $q$ 的电场
$$
\mathbf E\cdot\mathrm dl=\cfrac{1}{4πε_0}\cfrac{q}{r^2}\mathrm dl\cos
=\cfrac{1}{4πε_0}\cfrac{q}{r^2}\mathrm dr
$$
环路积分
$$
\oint_{L}\mathbf E\cdot\mathrm dl=\cfrac{q}{4πε_0}\oint_L\cfrac{1}{r^2}\mathrm dr=0
$$
根据场强叠加原理，可推广至多个点电荷及电荷连续分布情形。
根据数学上的[斯托克斯公式][math]，上式左端可以化为面积分
$$
\iint\limits_{S}(∇\times\mathbf E)\cdot\mathrm d\mathbf S=0
$$
其中 $S$ 为以 $L$ 为环边界线的任意曲面。取极限 $S\to M(x,y,z)$ 上式可化为微分形式
$$
∇\times\mathbf E=0\tag{1.6}
$$

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/cycle-theorem.png)   ![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/cycle-theorem2.png)



**电势能和电势**：由静电场环路定理知道
$$
\oint_{L}\mathbf F\cdot\mathrm dl=\oint_{L}q\mathbf E\cdot\mathrm dl=0
$$
由此可见，电场力做功与路径无关，只和起始和终止位置有关，由此引入势能的概念。
定义：静电场力做功等于相应电势能的减小量
$$
\int_{A}^{B}\mathbf F\cdot\mathrm dl=W_A-W_B
$$
做功的大小还和电荷量有关，在此引入电势 $φ_P=\cfrac{W_P}{q}$ ，电势差与路径无关
$$
\int_{A}^{B}\mathbf E\cdot\mathrm dl=φ_A-φ_B
$$
取无穷远处为电势零点，可求得点电荷电势场方程为
$$
φ(\mathbf r)=\int_{r}^{\infty}\mathbf E\cdot\mathrm dl
=\int_{r}^{\infty}\cfrac{1}{4πε_0}\cfrac{Q}{r^2}\mathrm dr
=\cfrac{Q}{4πε_0r}
$$
根据电场叠加原理进一步推广到电荷连续分布
$$
φ(\mathbf r)=\cfrac{1}{4πε_0r}\iiint\limits_V\rho(\mathbf r)\mathrm dV\tag{1.7}
$$
由电势的定义，相距为 $\mathrm dl$ 的两点的电势差为
$$
\mathrm dφ=-\mathbf E\cdot\mathrm dl
$$
由于 
$$
\mathrm dφ=\dfrac{∂φ}{∂x}\mathrm dx+\dfrac{∂φ}{∂y}\mathrm dy+\dfrac{∂φ}{∂z}\mathrm dz
=∇φ\cdot\mathrm dl
$$
所以电场强度和电势的关系为
$$
\mathbf E=-∇φ\tag{1.8}
$$
带入 $(1.4)$ 可得
$$
Δφ=-\cfrac{\rho}{ε_0}\tag{1.9}
$$
这就是电势函数应当满足的静电场方程，$\mathbf E$ 是矢量，而 $φ$ 是标量，求解标量方程相对简单些。

[^sr]: 立体角：常用字母 $Ω$ 表示。以观测点为球心，构造一个单位球面；任意物体投影到该单位球面上的投影面积，即为该物体相对于该观测点的立体角。 因此，立体角是单位球面上的一块面积，这和“平面角是单位圆上的一段弧长”类似。
立体角的国际制单位是球面度 (steradian , sr) 
在球坐标系中，任意球面的极小面积为：
    $$
    \mathrm dA=(r\sinθ\mathrm dϕ)\cdot(r\mathrm dθ)=r^2\sinθ\mathrm dθ\mathrm dϕ
    $$
    因此，极小立体角（单位球面上的极小面积）为：
    $$
    \mathrm dΩ=\cfrac{\mathrm dA}{r^2}=\sinθ\mathrm dθ\mathrm dϕ
    $$
    所以，立体角是投影面积与球半径平方值的比，这和“平面角是圆的弧长与半径的比”类似。 对极小立体角做曲面积分即可得立体角
    $$
    Ω=\iint\limits_S\mathrm dΩ=\iint\limits_S\sinθ\mathrm dθ\mathrm dϕ
    $$
     一个完整的球面对于球内任意一点的立体角为 $4π$， 这个定理对所有封闭曲面皆成立。
    $$
    \oiint\limits_S\mathrm dΩ=\oiint\limits_S\sinθ\mathrm dθ\mathrm dϕ
    =\int_0^{\pi}\sinθ\mathrm dθ\int_0^{2\pi}\mathrm dϕ=4\pi
    $$

## 格林公式

<kbd>高斯公式</kbd>：(Gauss formula)  设空间闭区域 $V$ 由分片光滑的闭曲面 $Σ$ 所围成，函数$P(x,y,z), Q(x,y,z), R(x,y,z)$ 在 $V$ 上有连续的一阶偏导数，则有
$$
\begin{aligned} 
\iiint\limits_{V}(\dfrac{∂P}{∂x}+\dfrac{∂Q}{∂y}+\dfrac{∂R}{∂z})\mathrm{d}V
&=\oiint\limits_{Σ}P\mathrm{d}y\mathrm{d}z+Q\mathrm{d}x\mathrm{d}z+R\mathrm{d}x\mathrm{d}y \\
&=\oiint\limits_{Σ}(P\cosα+Q\cosβ+R\cosγ) \mathrm{d}S
\end{aligned}
$$
曲面 $Σ$ 的方向取外侧，$\cosα,\cosβ,\cosγ$ 为曲面 $Σ$ 在点 $(x,y,z)$ 处外法线的方向余弦[^cos]
高斯公式向量形式为
$$
\iiint\limits_{V}∇\cdot \mathbf A\mathrm{d}V
=\oiint\limits_{Σ}\mathbf A\cdot \mathrm{d}\mathbf S
=\oiint\limits_{Σ}\mathbf A\cdot \mathbf n \mathrm{d}S
$$
其中 $\mathbf A=(P,Q,R)$，$\mathbf n=(\cosα,\cosβ,\cosγ)$ 为曲面  $Σ$ 单位法向量，$\mathrm{d}\mathbf S=\mathbf n\mathrm{d}S=\mathrm{d}y\mathrm{d}z\mathbf i+\mathrm{d}x\mathrm{d}z\mathbf j+\mathrm{d}x\mathrm{d}y\mathbf k$ 为单位元。

[^cos]: 方向向量与坐标轴的夹角 $α,β,γ$ 称为方向角，$\cosα,\cosβ,\cosγ$  称为方向向量的方向余弦。

<kbd>格林公式</kbd> ：设函数 $u(x,y,z), v(x,y,z)$ 在空间闭区域 $V$ 及边界 $Σ$ 上有一阶连续偏导数，在边界 $Σ$ 上有二阶连续偏导数 。利用高斯公式可得到 
$$
\iiint\limits_{V}∇\cdot \mathbf (u∇v)\mathrm{d}V
=\iint\limits_{Σ} u∇v\cdot \mathrm{d}\mathbf S
$$
于是我们得到==第一格林公式==
$$
\iint\limits_{Σ} u∇v\cdot \mathrm{d}\mathbf S=
\iiint\limits_{V}uΔv\mathrm{d}V+\iiint\limits_{V}∇u\cdot ∇v\mathrm{d}V
$$
同理我们可以得到
$$
\iint\limits_{Σ} v∇u\cdot \mathrm{d}\mathbf S=
\iiint\limits_{V}vΔu\mathrm{d}V+\iiint\limits_{V}∇v\cdot ∇u\mathrm{d}V
$$
两式相减可得到
$$
\iiint\limits_{V}(uΔv-vΔu)\mathrm{d}V=\iint\limits_{Σ} (u∇v-v∇u)\cdot \mathrm{d}\mathbf S
=\iint\limits_{Σ} (u\cfrac{∂v}{∂n}-v\cfrac{∂u}{∂n})\mathrm{d}S
$$
其中 $\cfrac{∂}{∂n}$  表示沿边界 $Σ$ 外法线的[方向导数][math]。上式称为==第二格林公式==，简称格林公式。

**调和函数的边界性质**：设函数 $u(\mathbf r)$ 是区域 $V$ 内的调和函数，则有
$$
\iint\limits_{∂V}\cfrac{∂v}{∂n}\mathrm dS=0
$$
证明：在第二格林公式中取 $u(\mathbf r)$ 为调和函数，即满足 $Δu=0$ ，取 $v=1$ ，即得上式。

**调和函数的均值定理**：设区域 $V$ 是以 $\mathbf r_0$ 为球心 $a$ 为半径的球，函数 $u(\mathbf r)$ 是 $V$ 内的调和函数，则
$$
u(\mathbf r_0)=\cfrac{1}{4\pi a^2}\iint\limits_{∂V}u(\mathbf r)\mathrm dS
$$
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/mean-theorem-of-harmonic-function.png)
证明：已知调和函数 $u(\mathbf r)$ 满足 
$$
Δu(\mathbf r)=0
$$
取函数 $v(\mathbf r)=\cfrac{1}{|\mathbf{r-r_0}|}=\cfrac{1}{r}$ ，如图，利用多维 [$δ$ 函数][F]的性质得到
$$
Δv(\mathbf r)=-4\piδ(\mathbf{r-r_0})
$$
 将上述两方程分别乘以 $v(\mathbf r),u(\mathbf r)$ ，并相减，在体积 $V$ 内积分
$$
\iiint\limits_{V}(uΔv-vΔu)\mathrm{d}V
=-4\pi\iiint\limits_{V}u(\mathbf r)δ(\mathbf{r-r_0})
=-4\pi u(\mathbf r_0)
$$
利用第二格林公式得到
$$
u(\mathbf r_0)=\cfrac{1}{4\pi}
\iint\limits_{∂V}[v(\mathbf r)\cfrac{∂u(\mathbf r)}{∂n}
-u(\mathbf r)\cfrac{∂v(\mathbf r)}{∂n}]\mathrm{d}S
$$
因为在球面 $∂V$ 上，外法线 $\mathbf n$ 的方向与 $r$ 所在半径的方向一致，所以球面上
$$
\cfrac{∂v(\mathbf r)}{∂n}=\cfrac{∂}{∂n}(\cfrac{1}{r})
=\cfrac{∂}{∂r}(\cfrac{1}{r})=-\cfrac{1}{r^2}
$$
又因为调和函数带入格林公式
$$
\iiint\limits_{V}Δu\mathrm dV=\iint\limits_{∂V}\cfrac{∂u}{∂n}\mathrm dS=0
$$
于是最终得到
$$
u(\mathbf r_0)=\cfrac{1}{4\pi a^2}\iint\limits_{∂V}u(\mathbf r)\mathrm{d}S
$$
**调和函数的极值原理**：设函数 $u(\mathbf r)$ 是区域 $V$ 内的调和函数，则 $u(\mathbf r)$ 必在 $V$ 的边界面上取得最大值最小值。
证明：结合均值定理，可用反证法证明。

## δ 函数简介

$δ$ 函数起源于集中分布物理量的描述。
对于连续分布的物理量 $Q$ ，通常有两种描述方式，一种是局部性的，给出密度分布函数
$$
\rho(\mathbf r)=\cfrac{\mathrm dQ}{\mathrm d\mathbf r}
$$
另一种是整体性的
$$
Q=\int_V\rho(\mathbf r)\mathrm d\mathbf r
$$
对于集中分布的物理量同样有两种方式描述。

**$δ$ 函数**：(点电荷的线密度) 设在直线 $L$ 上，仅在 $x=0$ 处有一单位电荷，可以看成单位电荷分布在 $[-ε,ε]$ 上当 $ε\to0$ 的极限情况，后者密度可表示为
$$
\rho_ε(x)=\begin{cases}
\cfrac{1}{2ε}&(|x|⩽ε) \\
0&(|x|>ε)
\end{cases}
$$
且对任意 $ε>0$ 直线上的电荷总量为
$$
Q=\int_{-\infty}^{+\infty}\rho_ε(x)\mathrm dx=1
$$
令 $ε\to0$ 可由 $\rho_ε(x)$ 的极限推得单位点电荷的分布
$$
\rho(x)=\begin{cases}
∞ &(x=0) \\
0&(x\neq 0) 
\end{cases}
$$
且保持直线上的电荷总量为 1。
对于集中于 $x=0$ 点的单位物理量引起的密度函数叫做 $δ$ 函数，$δ(x)$ 满足条件
$$
δ(x)=\begin{cases}
∞ &(x=0) \\
0&(x\neq 0) 
\end{cases}
$$
和
$$
\int_{-∞}^{+∞}δ(x)\mathrm dx=1
$$
**注意**：
(1)  $δ$ 函数并不是经典意义下的函数，因此通常称其为广义函数(或者奇异函数)。
(2) 它不能用常规意义下的值的对应关系来理解和使用，而总是通过它的定义和性质来使用它。
(3)  $δ$ 函数还有其他多种定义方式。

 **$δ$ 函数的平移**：对于集中于 $x=x_0$ 点的单位物理量引起的密度函数， $δ$ 函数平移满足
$$
δ(x-x_0)=\begin{cases}∞ &(x=x_0) \\0&(x\neq x_0) \end{cases}
$$
和
$$
\int_{-∞}^{+∞}δ(x-x_0)\mathrm dx=1
$$
**筛选性质**
$$
\int_a^bδ(x)f(x)dx=\begin{cases}
f(0) & 0\in[a,b] \\
0 & 0\not\in[a,b]\end{cases}
$$
特别的
$$
\int_{-∞}^{+∞}δ(t)f(t)dt=f(0)
$$
也可以把上述性质作为 $δ$ 函数的另一种定义，此时我们对 $δ$ 函数有了全新的认识，它实际上是一种映射，把元素 $f(x)$ 映射成了一个数 $f(0)$ 。

**性质和运算**

(1) $δ(x)$ 和常数 $c$ 的乘积 $cδ(x)$
$$
\int_{-∞}^{+∞}[cδ(x)]f(x)dx=\int_{-∞}^{+∞}δ(x)[cf(x)]dx=cf(0)
$$
(2) 筛选性质
$$
\int_{-∞}^{+∞}δ(x-x_0)f(x)dx=f(x_0)
$$
(3) 对称性
$$
δ(x-x_0)=δ(x_0-x)
$$
特别的
$$
δ(x)=δ(-x)
$$
(4) 与连续分布函数 $f(x)$ 的乘积
$$
\int_{-∞}^{+∞}f(x)δ(x-x_0)=\int_{-∞}^{+∞}f(x_0)δ(x-x_0)
$$
即
$$
f(x)δ(x-x_0)=f(x_0)δ(x-x_0)
$$
(5)  $δ$ 函数的导数 $δ'(x)$，对于在 $x=0$ 点连续并有连续导数的任意函数 $f(x)$ ，应用分部积分
$$
\int_{-∞}^{+∞}δ'(x)f(x)dx=δ(x)f(x)\Big|_{-∞}^{+∞}-\int_{-∞}^{+∞}δ(x)f'(x)dx=-f'(0)
$$
 对于  $δ$ 函数的高阶导数 $δ^{(n)}(x)$ ，对于在 $t=0$ 点连续并有连续导数的任意函数 $f(x)$ ，有
$$
\int_{-∞}^{+∞}δ^{(n)}(x)f(x)dx=(-1)^{n}f^{(n)}(0)
$$
(6)  $δ$ 函数是单位阶跃函数的导数
$$
\dfrac{\mathrm dH(x)}{\mathrm dx}=δ(x)
$$
 $δ$ 函数的原函数为单位阶跃函数
$$
H(x)=\begin{cases}0 & (x<0) \\ 1 &(x>0) \end{cases}= \int_{-∞}^{x}δ(s)ds
$$
![单位阶跃函数](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/unit-step-fun.png)

(7) $δ$ 函数的卷积
$$
δ(x)*f(x)=f(x) \\
δ'(x)*f(x)=δ(x)*f'(x)=f'(x) \\
L[f*g]=L[f]*g=f*L[g]
$$

(8) 连续分布的质量、电荷或持续作用的力也可以用 $δ$ 函数表示。现在以从 $t=a$ 持续作用到 $t=b$ 的作用力 $f(t)$ 为例说明。将时间段 $[a,b]$ 分成许多小段，在某个 $\tau$ 到 $\tau+d\tau$ 的短时间上，力 $f(t)$ 的冲量为 $f(\tau)d\tau$ ，既然 $d\tau$ 很短，不妨将这段时间的力看成是瞬时力，记作 $f(\tau)δ(t-\tau)d\tau$  ，这样许许多多瞬时力的总和就是持续力 $f(t)$ ，即
$$
f(t)=\sum f(\tau)δ(t-\tau)d\tau=\int_a^b f(\tau)δ(t-\tau)d\tau
$$

**δ函数的Fourier 变换**
(1) 根据 $δ$ 函数筛选性质可得
$$
F(ω)=\mathcal{F}[δ(t)]=\int^{+∞}_{-∞}δ(t)e^{-iω t}\text{d}t=e^{-iω t}|_{t=0}=1 \\
δ(t)=\mathcal{F}^{-1}[1]=\dfrac{1}{2\pi}\int_{-∞}^{+∞}e^{iω t}\text{d}ω
$$
或写为
$$
δ(t)=\dfrac{1}{2\pi}\int_{-∞}^{+∞}\cosω t\text{d}ω
=\dfrac{1}{\pi}\int_{0}^{+∞}\cosω t\text{d}ω
$$
由此可见，δ函数包含所有频率成份，且它们具有相等的幅度，称此为均匀频谱或白色频谱。
我们可以得到 ：

|        原函数 |          | 像函数            |
| ------------: | :------: | :---------------- |
|        $δ(t)$ | $\lrarr$ | $1$               |
|    $δ(t-t_0)$ | $\lrarr$ | $e^{-iω t_0}$     |
|           $1$ | $\lrarr$ | $2\pi δ(ω)$       |
| $e^{-iω t_0}$ | $\lrarr$ | $2\pi δ(ω − ω_0)$ |
|           $t$ | $\lrarr$ | $2\pi iδ'(ω)$     |

$$
\cos(ω_0t)=\pi[δ(ω + ω_0)+δ(ω − ω_0)]  \\ 
\sin(ω_0t)=i\pi[δ(ω + ω_0)-δ(ω − ω_0)]
$$
(2) 有许多重要的函数不满足Fourier 积分定理条件（绝对可积），例如常数、符号函数、单位阶跃函数、正弦函数和余弦函数等，但它们的广义Fourier 变换[^gf]也是存在的，利用δ函数及其Fourier 变换可以求出它们的Fourier 变换。

[^gf]: 在δ函数的Fourier变换中，其广义积分是根据δ函数的性质直接给出的，而不是按通常的积分方式得到的，称这种方式的Fourier 变换为==广义Fourier 变换==。

**δ函数的Fourier 展开**：当 $x,x_0\in(-\pi,\pi)$ 时
$$
δ(x-x_0)=\cfrac{a_0}{2}+\sum_{n=1}^{\infty}(a_n\cos nx+b_n\sin nx)
$$
其中傅里叶系数
$$
\begin{cases}\displaystyle 
a_n=\cfrac{1}{\pi}\int_{-\pi}^{\pi}δ(x-x_0)\cos nxdx=\cfrac{1}{\pi}\cos nx_0 \\
\displaystyle  b_n=\cfrac{1}{\pi}\int_{-\pi}^{\pi}δ(x-x_0)\sin nxdx
=\cfrac{1}{\pi}\sin nx_0
\end{cases}
$$
**多维 $δ$ 函数**：例如位于三维空间的坐标原点质量为 $m$ 的质点，其密度函数可表示为 $mδ(\mathbf r)$。	在三维空间中的 $δ$ 函数定义如下：
$$
δ(\mathbf r)=
\begin{cases} 
0 &(\mathbf r\neq0) \\
\infty &(\mathbf r=0)
\end{cases}  \\
\iiint\limits_{-\infty}^{+\infty} δ(\mathbf r)\mathrm d\mathbf r=1
$$

(1) 三维 $δ$ 函数可表示为三个一维 $δ$ 函数乘积表示，在直角坐标系中
$$
δ(\mathbf r)=δ(x)δ(y)δ(z)
$$
三维空间点 $\mathbf r_0=(x_0,y_0,z_0)$ 处密度分布函数就是
$$
δ(\mathbf{r-r_0})=δ(x-x_0)δ(y-y_0)δ(z-z_0)
$$
(2) 变量代换：当
$$
\begin{cases}x=x(ξ,η,ζ) \\y=y(ξ,η,ζ)  \\z=z(ξ,η,ζ)  \\\end{cases}
$$
时有
$$
δ(x-x_0,y-y_0,z-z_0)=\cfrac{1}{|J|}δ(ξ-ξ_0,η-η_0,ζ-ζ_0)
$$
其中 $|J|\neq0$ 是 Jacobi 行列式的绝对值，$(x_0,y_0,z_0)$ 和 $(ξ_0,η_0,ζ_0)$ 相对应。
直角坐标系换算到柱坐标系 $\mathbf r=(r,θ,z)$
$$
δ(\mathbf{r-r_0})=\frac{1}{r_0}δ(r-r_0)δ(θ-θ_0)δ(z-z_0)
$$
直角坐标系换算到球坐标系 $\mathbf r=(r,θ,ϕ)$
$$
δ(\mathbf{r-r_0})=\frac{1}{r_0^2\sinθ_0}δ(r-r_0)δ(θ-θ_0)δ(ϕ-ϕ_0)
$$
(3) 筛选性质
$$
\iiint\limits_{-\infty}^{+\infty} f(\mathbf r)δ(\mathbf{r-r_0})\mathrm d\mathbf r=f(\mathbf r_0) \\\iiint\limits_{-\infty}^{+\infty} f(\mathbf r)[\nablaδ(\mathbf{r-r_0})]\mathrm d\mathbf r=-\nabla f(\mathbf r)|_{\mathbf{r=r_0}}
$$
位矢的微分：
$$
\Delta \frac{1}{r}=-4\piδ(\mathbf r)
$$
其中 $r=\sqrt{x^2+y^2+z^2}$ 

(4) 混合偏导：
$$
\dfrac{∂^3H(x,y,z)}{∂x∂y∂z}=δ(x,y,z)
$$
其中 $H(x,y,z)=H(x)H(y)H(z)$ 为单位阶跃函数

(5) 多重傅里叶变换
|原函数||像函数|
|---:|:--:|:---|
|$δ(x,y,z)$|$\lrarr$|$1$|
|$1$|$\lrarr$|$(2\pi)^3δ(λ,μ,ν)$|
|$x$|$\lrarr$|$(2\pi)^3i\cfrac{∂δ(λ,μ,ν)}{∂λ}$|
|$x^2+y^2+z^2$|$\lrarr$|$(2\pi)^3δ(λ,μ,ν)$|
|$e^{iax}$|$\lrarr$|$(2\pi)^3δ(λ-a,μ,ν)$|

(6) 多重卷积定义
$$
f*g=\iiint\limits_{-\infty}^{+\infty} f(\mathbf r)g(\mathbf{r-r_0})\mathrm d\mathbf r_0
$$
性质如下

| 等式                                                   |
| :----------------------------------------------------- |
| $δ*f=f$                                                |
| $\cfrac{∂δ}{∂x}*f=δ*\cfrac{∂f}{∂x}=\cfrac{∂f}{∂x}$     |
| $\cfrac{∂}{∂x}(f*g)=\cfrac{∂f}{∂x}*g=f*\cfrac{∂g}{∂x}$ |
| $L[f*g]=L[f]*g=f*L[g]$                                 |
| $\mathcal F(f*g)=\mathcal F(f)\cdot\mathcal F(g)$      |




------

> **参考文献：**
> 季孝达.《数学物理方程》.
> 吴崇试.《数学物理方法》.
> 梁昆淼.《数学物理方法》.
> 吴崇试 高春媛.《数学物理方法》.北京大学(MOOC)
