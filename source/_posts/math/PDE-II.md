---
title: 偏微分方程(Partial Differential Equation II)
date: 2020-05-15 17:54:41
categories: [数学,微分方程]
tags: [数学,PDE,微分方程,贝塞尔函数,勒让德函数]
cover: 
top_img: 
katex: true
description: false
---


> 参考文献：
>
> 《数学物理方程》| 季孝达
> 《数学物理方法》| 吴崇试
> 《数学物理方法》| 梁昆淼
> MOOC北京大学《数学物理方法》| 吴崇试 、高春媛

# 正交曲面坐标系下的分离变量

上章只是讨论了用分离变量法解决直角坐标系中的各种定解问题，但实际中的边界是多种多样的，坐标系参照问题中的边界形状来选择，可以方便的解决相应的本征值问题。

**平面极坐标系 $(r,ϕ)$**
$$
\begin{cases}
x=r\cosϕ \\
y=r\sinϕ
\end{cases}
$$
<img src="https://gitee.com/WilenWu/images/raw/master/DifferentialEquation/plane-polar-coordinate.png" style="zoom: 80%;" />
拉普拉斯算符 
$$
\begin{aligned}
Δ &=\cfrac{∂^2}{∂r^2}+\cfrac{1}{r}\cfrac{∂}{∂r}+\cfrac{1}{r^2}\cfrac{∂^2}{∂ϕ^2} \\
&=\cfrac{1}{r}\cfrac{∂}{∂r}\left(r\cfrac{∂}{∂r}\right)+\cfrac{1}{r^2}\cfrac{∂^2}{∂ϕ^2}
\end{aligned}
$$
**三维柱坐标系 $(r,ϕ,z)$**
$$
\begin{cases}x=r\cosϕ \\y=r\sinϕ \\z=z \end{cases}
$$
![](https://gitee.com/WilenWu/images/raw/master/DifferentialEquation/cylindrical-coordinates.png)
拉普拉斯算符 
$$
\begin{aligned}
Δ &=\cfrac{∂^2}{∂r^2}+\cfrac{1}{r}\cfrac{∂}{∂r}
+\cfrac{1}{r^2}\cfrac{∂^2}{∂ϕ^2}+\cfrac{∂^2}{∂z^2} \\
&=\cfrac{1}{r}\cfrac{∂}{∂r}\left(r\cfrac{∂}{∂r}\right)
+\cfrac{1}{r^2}\cfrac{∂^2}{∂ϕ^2}+\cfrac{∂^2}{∂z^2}
\end{aligned}
$$
**三维球坐标系 $(r,θ,ϕ)$**
$$
\begin{cases}
x=r\sinθ\cosϕ \\
y=r\sinθ\sinϕ \\
z=r\cosθ
\end{cases}
$$
<img src="https://gitee.com/WilenWu/images/raw/master/DifferentialEquation/spherical-coordinates.png" style="zoom:67%;" />
拉普拉斯算符 
$$
\begin{aligned}
Δ & =\cfrac{∂^2}{∂r^2}+\cfrac{2}{r}\cfrac{∂}{∂r}
+\cfrac{1}{r^2}\cfrac{∂^2}{∂θ^2}
+\cfrac{\cosθ}{r^2\sinθ}\cfrac{∂}{∂θ}
+\cfrac{1}{r^2\sin^2θ}\cfrac{∂^2}{∂ϕ^2} \\
&=\cfrac{1}{r^2}\cfrac{∂}{∂r}\left(r^2\cfrac{∂}{∂r}\right)
+\cfrac{1}{r^2\sinθ}\cfrac{∂}{∂θ}\left(\sinθ\cfrac{∂}{∂θ}\right)
+\cfrac{1}{r^2\sin^2θ}\cfrac{∂^2}{∂ϕ^2}
\end{aligned}
$$


**三维空间拉普拉斯方程**
$$
Δ=u_{xx}+u_{yy}+u_{zz}=0
$$
**(1) 球坐标系**
$$
\cfrac{1}{r^2}\cfrac{∂}{∂r}\left(r^2\cfrac{∂u}{∂r}\right)
+\cfrac{1}{r^2\sinθ}\cfrac{∂}{∂θ}\left(\sinθ\cfrac{∂u}{∂θ}\right)
+\cfrac{1}{r^2\sin^2θ}\cfrac{∂^2u}{∂ϕ^2}=0
$$
令 $u(r,θ,ϕ)=R(r)S(θ,ϕ)$ 带入方程分离变量，可得到
$$
\cfrac{1}{R}\cfrac{d}{dr}\left(r^2\cfrac{dR}{dr}\right)
=-\cfrac{1}{S\sinθ}\cfrac{∂}{∂θ}\left(\sinθ\cfrac{∂S}{∂θ}\right)
-\cfrac{1}{S\sin^2θ}\cfrac{∂^2S}{∂ϕ^2}=μ
$$
于是得到两个方程
$$
\cfrac{d}{dr}\left(r^2\cfrac{dR}{dr}\right)-μR=0
$$

$$
\cfrac{1}{\sinθ}\cfrac{∂}{∂θ}\left(\sinθ\cfrac{∂S}{∂θ}\right)
+\cfrac{1}{\sin^2θ}\cfrac{∂^2S}{∂ϕ^2}+μS=0
$$

第一个常微分方程为欧拉方程，此方程通解为 （取 $μ=m^2$ ）
$$
R=\begin{cases}C_0+D_0\ln r  & (m=0)\\C_mr^m+D_m\cfrac{1}{r^m} & (m\neq0) \end{cases}
$$
再令 $S(θ,ϕ)=Θ(θ)Φ(ϕ)$ 进一步分离变量，得到
$$
\cfrac{\sinθ}{Θ}\cfrac{d}{dθ}\left(\sinθ\cfrac{dΘ}{dθ}\right)
+μ\sin^2θ=-\cfrac{1}{Φ}\cfrac{d^2Φ}{dϕ^2}=λ
$$
同样分解为两个常微分方程
$$
Φ''+λΦ=0\tag{1.1}
$$

$$
\sinθ\cfrac{d}{dθ}\left(\sinθ\cfrac{dΘ}{dθ}\right)+(μ\sin^2θ-λ)Θ=0 \tag{1.2}
$$

常微分方程 (1.1) 与隐藏的自然周期条件构成本征值问题。易求得本征值是
$$
λ=m^2,\quad(m=0,1,2,\cdots)
$$
本征函数为
$$
Φ(ϕ)=A\cos mϕ+B\sin mϕ
$$
将本征值带入方程 1.2) ，并做转换 令 $x=\cosθ$ ，常数 $μ=l(l+1)$ 可得到
$$
(1-x^2)\cfrac{d^2Θ}{dx^2}-2x\cfrac{dΘ}{dx}+[l(l+1)-\cfrac{m^2}{1-x^2}]Θ=0
$$
这叫做 $l$ 阶==连带勒让德方程== (Legendre)。其 $m=0$ 的特例叫做==勒让德方程==。

**(2) 柱坐标系**
$$
\cfrac{1}{r}\cfrac{∂}{∂r}\left(r\cfrac{∂u}{∂r}\right)
+\cfrac{1}{r^2}\cfrac{∂^2u}{∂ϕ^2}+\cfrac{∂^2u}{∂z^2}=0
$$
令 $u(r,ϕ,z)=R(r)Φ(ϕ)Z(z)$ 带入方程分离变量，可得到
$$
\cfrac{r^2}{R}R''+\cfrac{r}{R}R'+r^2\cfrac{Z''}{Z}=-\cfrac{Φ''}{Φ}=λ
$$
于是分解为两个方程
$$
Φ''+λΦ=0\tag{1.3}
$$

$$
\cfrac{r^2}{R}R''+\cfrac{r}{R}R'+r^2\cfrac{Z''}{Z}=λ\tag{1.4}
$$

方程 (1.4) 同样分解为两个常微分方程
$$
Z''+μZ=0\tag{1.5}
$$

$$
R''+\cfrac{1}{r}R'-(μ+\cfrac{λ}{r^2})R=0\tag{1.6}
$$

常微分方程 (1.3) 与隐藏的自然周期条件构成本征值问题。易求得本征值是
$$
λ=m^2,\quad(m=0,1,2,\cdots)
$$
本征函数为
$$
Φ(ϕ)=A\cos mϕ+B\sin mϕ
$$
一般，圆柱区域上下底面齐次边界条件或圆柱侧面齐次边界条件分别与 (1.5) 和 (1.6) 构成本征值问题。
方程 (1.5) 的通解为
$$
Z=\begin{cases}
Ce^{\sqrt{-μ}z}+De^{-\sqrt{-μ}z} & (μ<0)\\
C+Dz  & (μ=0)\\
C\cos\sqrt{μ}z+D\sin\sqrt{μ}z & (μ>0) 
\end{cases}
$$
对于方程 (1.6) 分为三种情形
(1) 当 $μ=0$ ，方程为欧拉方程，通解为 
$$
R=\begin{cases}
E+F\ln r & (m=0)\\
Er^m+\cfrac{F}{r^m} &(m=1,2,\cdots)
\end{cases}
$$
(2) 当 $μ<0$ 取 $μ=−ν^2, x=νr$ 得到 $m$ 阶==贝塞尔方程== (Bessel)
$$
x^2\cfrac{d^2R}{dx^2}+x\cfrac{dR}{dx}+(x^2-m^2)R=0
$$
(3) 当 $μ>0$ 取 $x=\sqrt{μ}r$  得到 $m$ 阶==虚宗量贝塞尔方程==  
$$
x^2\cfrac{d^2R}{dx^2}+x\cfrac{dR}{dx}-(x^2+m^2)R=0
$$
**波动方程**
$$
u_{tt}-a^2Δu=0
$$
分离时间变量 $t$ 和空间变量 $\mathrm{r}$ ，令 $u(\mathrm{r},t)=T(t)v(\mathrm{r})$ 带入方程得到
$$
\cfrac{T''}{a^2T}=\cfrac{Δv}{v}=-k^2
$$
于是分解为两个方程
$$
T''+k^2a^2T=0\tag{1.7}
$$

$$
Δv+k^2v=0\tag{1.8}
$$

常微分方程 (1.7) 为已讨论过的欧拉方程，偏微分方程 (1.8) 叫做==亥姆霍兹方程==。

**热传导方程**
$$
u_t-a^2 Δu=0
$$
分离时间变量 $t$ 和空间变量 $\mathrm{r}$ ，令 $u(\mathrm{r},t)=T(t)v(\mathrm{r})$ 带入方程得到
$$
\cfrac{T'}{a^2T}=\cfrac{Δv}{v}=-k^2
$$
于是分解为两个方程
$$
T'+k^2a^2T=0\tag{1.9}
$$

$$
Δv+k^2v=0\tag{1.10}
$$

常微分方程 (1.9) 为已讨论过的欧拉方程，偏微分方程 (1.10) 也是==亥姆霍兹方程==。

**亥姆霍兹方程** (Helmholtz)
$$
Δv+k^2v=0
$$
**(1) 球坐标系**
$$
\cfrac{1}{r^2}\cfrac{∂}{∂r}\left(r^2\cfrac{∂v}{∂r}\right)
+\cfrac{1}{r^2\sinθ}\cfrac{∂}{∂θ}\left(\sinθ\cfrac{∂v}{∂θ}\right)
+\cfrac{1}{r^2\sin^2θ}\cfrac{∂^2v}{∂ϕ^2}+k^2v=0
$$
令 $v(r,θ,ϕ)=R(r)S(θ,ϕ)$ 带入方程分离变量，可得到
$$
\cfrac{d}{dr}\left(r^2\cfrac{dR}{dr}\right)+(k^2r^2-μ)R=0\tag{1.11}
$$

$$
\cfrac{1}{\sinθ}\cfrac{∂}{∂θ}\left(\sinθ\cfrac{∂S}{∂θ}\right)+\cfrac{1}{\sin^2θ}\cfrac{∂^2S}{∂ϕ^2}+μS=0\tag{1.12}
$$

再令 $S(θ,ϕ)=Θ(θ)Φ(ϕ)$ 进一步分离变量，可得到
$$
Φ''+λΦ=0
$$

$$
\sinθ\cfrac{d}{dθ}\left(\sinθ\cfrac{dΘ}{dθ}\right)+(μ\sin^2θ-λ)Θ=0
$$

可以像上节那样进一步得到
$$
Φ(ϕ)=A\cos mϕ+B\sin mϕ\quad(m=0,1,2,\cdots)
$$
和 $l$ 阶==连带勒让德方程==
$$
(1-x^2)\cfrac{d^2Θ}{dx^2}-2x\cfrac{dΘ}{dx}+[l(l+1)-\cfrac{m^2}{1-x^2}]Θ=0
$$
其中 $x=\cosθ$ ，常数 $μ=l(l+1)$ 。这时，方程 (1.11) 可成为
$$
\cfrac{d}{dr}\left(r^2\cfrac{dR}{dr}\right)+[k^2r^2-l(l+1)]R=0
$$
叫做 $l$ 阶==球贝塞尔方程==。

**(2) 柱坐标系**
$$
\cfrac{1}{r}\cfrac{∂}{∂r}\left(r\cfrac{∂v}{∂r}\right)+\cfrac{1}{r^2}\cfrac{∂^2v}{∂ϕ^2}+\cfrac{∂^2v}{∂z^2}+k^2v=0
$$
令 $v(r,ϕ,z)=R(r)Φ(ϕ)Z(z)$ 一步步分离变量，可得到
$$
Φ''+λΦ=0\tag{1.13}
$$

$$
Z''+μZ=0\tag{1.14}
$$

$$
R''+\cfrac{1}{r}R'+(k^2-μ-\cfrac{λ}{r^2})R=0\tag{1.15}
$$

圆柱区域上下底面齐次边界条件或圆柱侧面齐次边界条件分别与 (1.13) 和 (1.14) 构成本征值问题。
取 $x=\sqrt{k^2-μ}r$ ，方程 (1.15) 如上节那样变为 ==贝塞尔方程== 。

# 球函数

## 勒让德方程的解

求解勒让德方程(Legendre equation) 
$$
(1-x^2)y''-2xy'+l(l+1)y=0\tag{1.1}
$$

其中 $l$ 为实参数，该方程的任意非零解称为==勒让德函数==。由于方程是二阶变系数常微分方程，可采用[幂级数求解][ode]。

[ode]:  https://blog.csdn.net/qq_41518277/article/details/105707514#_559

易知 $x=0$ 是方程的常点[^point]，当 $|x|<1$ 时，方程有幂级数解
$$
y=\displaystyle\sum_{k=0}^∞c_kx^k\tag{1.2}
$$
[^point]: 对于齐次线性微分方程标准形式
$$
y''+p(x)y'+q(x)y=0
$$

如果系数 $p(x),q(x)$ 在点 $x_0$ 的邻域是解析的，则点 $x_0$ 叫做方程的常点；如果 $x_0$ 是 $p(x)$ 或 $q(x)$ 的奇点，则点 $x_0$ 叫做方程的奇点。

带入勒让德方程逐项微分整理合并，可以得到
$$
\displaystyle \sum_{k=0}^∞\{(k+2)(k+1)c_{k+2}-[k(k+1)-l(l+1)]c_k\}x^k=0
$$
根据泰勒展开的唯一性可以得到
$$
(k+2)(k+1)c_{k+2}-[k(k+1)-l(l+1)]c_k=0
$$
即获得递推公式 
$$
c_{k+2}=\cfrac{(k-l)(k+l+1)}{(k+2)(k+1)}c_k\tag{1.3}
$$
反复利用递推关系式就可以得到系数
$$
\begin{cases}
c_{2k}=\cfrac{c_0}{(2k)!}(2k-l-2)(2k-l-4)\cdots(-l)(2k+l-1)(2k+l-3)\cdots(l+1) \\
c_{2k+1}=\cfrac{c_1}{(2k+1)!}(2k-l-1)(2k-l-3)\cdots(-l+1)(2k+l)(2k+l-2)\cdots(l+2)
\end{cases}
$$
其中 $c_0,c_1$ 是任意常数。利用 $\Gamma$ 函数[^gamma]的性质，上式可化为
[^gamma]: $\Gamma$ 函数性质：$\Gamma(s+1)=s\Gamma(s+1)$
$\Gamma(ν+1)=ν!\quad ν$为正整数。

$$
\begin{cases}
c_{2k}=c_0\cfrac{2^{2k}}{(2k)!}\cfrac{Γ(k-\cfrac{l}{2})Γ(k+\cfrac{l+1}{2})}{Γ(-\cfrac{l}{2})Γ(\cfrac{l+1}{2})} \\
c_{2k+1}=c_1\cfrac{2^{2k}}{(2k+1)!}\cfrac{Γ(k-\cfrac{l-1}{2})Γ(k+1+\cfrac{l}{2})}{Γ(k-\cfrac{l-1}{2})Γ(1+\cfrac{l}{2})}
\end{cases}
$$
此时，分别取 $c_0=1,c_1=0$ 和 $c_0=0,c_1=1$ ，我们可以获得两个级数解
$$
y_1(x)=\sum_{k=0}^{∞}c_{2k}x^{2k} \tag{1.4}
$$

$$
y_2(x)=\sum_{k=0}^{∞}c_{2k+1}x^{2k+1}\tag{1.5}
$$

容易证明 $y_1(x),y_2(x)$ 线性无关，且在 $x\in(-1,1)$ 收敛。所以，勒让德方程的解就是
$$
y(x)=C_0y_1(x)+C_1y_2(x)
$$

其中 $C_0,C_1$ 为任意常数。

## 勒让德函数

**勒让德多项式**：观察上节级数 $y_1(x),y_2(x)$ ，容易发现，如果参数 $l$ 是某个偶数 ，$l=2n$（$n$是正整数），$y_1(x)$ 则直到 $x^{2n}$ 为止，因为从 $c_{2n+2}$ 开始都含有因子 $(2n-l)$ 从而都为零。 $y_1(x)$ 化为 $2n$ 次多项式，并且只含偶次幂，而 $y_2(x)$ 仍然是无穷级数。同理，当 $l$ 是奇数 ，$l=2n+1$（$n$是零或正整数）， $y_2(x)$ 化为 $2n+1$ 次多项式，并且只含奇次幂，而 $y_1(x)$ 仍然是无穷级数。
下面给出 $y_1(x)$ 或 $y_2(x)$ 为多项式时的表达式，为了简洁，通常取最高次项的系数（$l$为零或正整数）
$$
c_{l}=\cfrac{(2l)!}{2^l(l!)^2}
$$
反用系数递推公式 (1.3) 
$$
c_k=\cfrac{(k+2)(k+1)}{(k-l)(k+l+1)}c_{k+2}
$$
就可以把其他系数一一推算出来，一般的有
$$
c_{l-2n}=(-1)^n\cfrac{(2l-2n)!}{n!2^l(l-n)!(l-2n)!}
$$
这样求得勒让德方程 (1.1) 的解称为 $l$ 阶==勒让德多项式==，或==第一类勒让德函数==。
$$
P_l(x)=\sum_{n=0}^{[l/2]}(-1)^n
\cfrac{(2l-2n)!}{n!2^l(l-n)!(l-2n)!}x^{l-2n}\tag{2.1}
$$
其中 $l$为零或正整数，记号 $[l/2]$ 表示不超过 $l/2$ 的最大整数，即
$$
[l/2]=\begin{cases}
l/2 & (l 为偶数)\\
(l-1)/2 & (l 为奇数)
\end{cases}
$$
<img src="https://gitee.com/WilenWu/images/raw/master/DifferentialEquation/Legendre-polynomial.png"  width="50%" />

**勒让德多项式的微分表示**
$$
P_l(x)=\cfrac{1}{2^ll!}\cfrac{d^l}{dx^l}(x^2-1)^l\tag{2.2}
$$
此式称为==罗德里格斯表达式==(Rodrigues)。由表达式不难看出勒让德多项式的奇偶性
$$
P_l(-x)=(-1)^lP_l(x)
$$

**勒让德多项式的积分表示**：按照柯西公式，微分表示可写成路径积分
$$
P_l(x)=\cfrac{1}{2\pi\mathrm i}\cfrac{1}{2^l}\oint_C\cfrac{(z^2-1)^l}{(z-x)^{l+1}}dz\tag{2.3}
$$
其中 $C$ 为 $z$ 平面上围绕 $z=x$ 点任一闭合回路，这叫做==施列夫利积分== (SchlMli)。
还可以进一步表示为定积分，为此取 $C$ 为圆周，圆心在 $z=x$ ，半径为 $\sqrt{x^2-1}$ 。在圆周 $C$ 上 $z-x=\sqrt{x^2-1}e^{\mathrm iψ},dz=\mathrm i\sqrt{x^2-1}e^{\mathrm iψ}dψ$ ，所以 (2.3) 式成为
$$
P_l(x)=\cfrac{1}{\pi}\int_0^{\pi}[x+\mathrm i\sqrt{1-x^2}\cos\mathrm ψ]^ldψ
$$
这叫做==拉普拉斯积分==，如果从 $x$ 变换回变量 $θ,\ x=\cosθ$ ，则
$$
P_l(x)=\cfrac{1}{\pi}\int_0^{\pi}[\cosθ+\mathrm i\sinθ\cos\mathrm ψ]^ldψ\tag{2.4}
$$
从上式很容易看出 $P_l(1)=1,P_l(-1)=(-1)^l$ 从而得到
$$
|P_l(x)|⩽1,\quad(-1⩽x⩽1)
$$

**第二类勒让德函数**：以上讨论知道，当  $l$为零或正整数时，$y_1,y_2$ 中有一个是勒让德多项式，而另一个仍是无穷级数，此时勒让德方程的一般解为
$$
y=C_1P_l(x)+C_2Q_l(x)
$$
其中 $Q_l(x)$ 为由 $P_l(x)$ 导出具有统一形式的线性无关特解
$$
Q_l(x)=P_l(x)\int\cfrac{1}{(1-x^2)P_l^2(x)}dx\tag{2.5}
$$
可计算得 $Q_0(x)=\cfrac{1}{2}\ln\cfrac{1+x}{1-x},\quad Q_1(x)=\cfrac{x}{2}\ln\cfrac{1+x}{1-x}-1,\quad \cdots$
一般表达式为
$$
Q_l(x)=\cfrac{1}{2}P_l(x)\ln\cfrac{1+x}{1-x}
-\sum_{n=1}^{[l/2]}\cfrac{2l-4n+3}{(2n-1)(l-n+1)}P_{l-2n+1}(x)\tag{2.6}
$$

**勒让德多项式的正交性**：在区间 $(-1,1)$ 上正交
$$
\int_{-1}^{1}P_l(x)P_k(x)dx=0\quad(l\neq k)
$$
如果从 $x$ 变换回变量 $θ,\ x=\cosθ$ ，则
$$
\int_{0}^{\pi}P_l(\cosθ)P_k(\cosθ)\sinθdθ=0\quad(l\neq k)
$$

**勒让德多项式的模**
$$
\|P_l(x）\|^2=\int_{-1}^{1}P^2_l(x)dx
$$
可计算得
$$
\|P_l(x）\|=\sqrt{\cfrac{2}{2l+1}}\quad(l=0,1,2,\cdots)\tag{2.7}
$$

**傅里叶-勒让德级数**：设函数 $f(x)$ 在区间 $[-1,1]$ 上满足狄利克雷条件，则 $f(x)$ 在连续点处展开为
$$
f(x)=\sum_{k=0}^{\infty}c_kP_k(x)
$$
其中系数
$$
c_k=\cfrac{2k+1}{2}\int_{-1}^{1}f(x)P_k(x)dx
$$
在物理上常取 $x=\cosθ(0⩽θ⩽\pi)$ ，则
$$
f(θ)=\sum_{k=0}^{\infty}c_kP_k(\cosθ)
$$
其中系数
$$
c_k=\cfrac{2k+1}{2}\int_{0}^{\pi}f(θ)P_k(\cosθ)\sinθdθ
$$

**勒让德多项式的生成函数**：首先由电荷势理论引入
$$
\cfrac{1}{\sqrt{R^2-2Rr\cosθ+r^2}}=\begin{cases}
\displaystyle\sum_{k=0}^{\infty}\cfrac{r^k}{R^{k+1}}P_k(\cosθ) &(r<R) \\
\displaystyle\sum_{k=0}^{\infty}\cfrac{R^k}{r^{k+1}}P_k(\cosθ) &(r>R)
\end{cases} \tag{2.8}
$$

**勒让德多项式的递推关系**
(1) 递推公式
$$
(k+1)P_{k+1}(x)-(2k+1)xP_k(x)+kP_{k-1}(x)=0\tag{2.9}
$$
(2) 通过微分还可以获得许多其他类别的递推关系
$$
P'_k(x)-xP'_{k-1}(x)=kP_{k-1}(x) \\
P'_k(x)-P'_{k-1}(x)=kP_{k}(x) \\
(1-x^2)P'_k(x)=kP'_{k-1}(x)-kxP_{k}(x) \\
(1-x^2)P'_{k-1}(x)=kxP_{k-1}(x)-kP'_{k}(x) 
$$

**勒让德多项式的奇偶性**：当 $l$ 为偶数时，$P_l(x)$ 为偶函数；当 $l$ 为奇数时，$P_l(x)$ 为奇函数
$$
P_l(-x)=(-1)^lP_l(x)\tag{2.10}
$$

## 连带勒让德函数

**连带勒让德方程**
$$
(1-x^2)\cfrac{d^2Θ}{dx^2}-2x\cfrac{dΘ}{dx}+[l(l+1)-\cfrac{m^2}{1-x^2}]Θ=0\tag{3.1}
$$
为了寻找连带勒让德方程和勒让德方程之间的联系，通常作代换
$$
Θ=(1-x^2)^{m/2}y(x)
$$
则方程 (3.1) 可化为
$$
(1-x^2)y''-2(m+1)xy'+[l(l+1)-m(m+1)]y=0\tag{3.2}
$$
事实上，上述微分方程 (3.2) 就是勒让德方程求导 $m$ 次得到的方程，利用莱布尼茨求导公式
$$
(uv)^{(n)}=\displaystyle\sum_{k=0}^n ∁^k_n u^{(n-k)}v^{(k)}
$$
将勒让德方程
$$
(1-x^2)P''-2xP'+l(l+1)P=0
$$
对 $x$ 求导 $m$ 次得到
$$
(1-x^2)(P^{(m)})''-2x(P^{(m)})'+[l(l+1)-m(m+1)]P^{(m)}=0
$$
这正是方程 (3.2) 的形式，因此方程 (3.2) 的解 $y(x)$ 正是勒让德方程解 $P(x)$ 的 $m$ 阶导数。方程 (3.2) 与自然边界条件构成本征值问题，本征值是 $l(l+1)$ ，本征函数则是勒让德多项式 $P_l(x)$ 的 $m$ 阶导数，即
$$
y(x)=P_l^{(m)}(x)
$$
将此式代回可得到 $Θ=(1-x^2)^{m/2}P_l^{(m)}(x)$ ，通常记作
$$
P_l^m(x)=(1-x^2)^{m/2}P_l^{(m)}(x)\tag{3.3}
$$
这称为==连带勒让德多项式==。由于 $P_l(x)$ 是 $l$ 次多项式，最多只能求导 $l$ 次，超过后就得到零，因此必须有 $l⩾m$ 。

**连带勒让德多项式的微分表示**
$$
P^m_l(x)=\cfrac{(1-x^2)^{m/2}}{2^ll!}\cfrac{d^{l+m}}{dx^{l+m}}(x^2-1)^l\tag{3.4}
$$
此式称为==罗德里格斯表达式==(Rodrigues)。

**连带勒让德多项式的积分表示**：按照柯西公式，微分表示可写成路径积分
$$
P^m_l(x)=\cfrac{(1-x^2)^{m/2}}{2^l}
\cfrac{1}{2\pi\mathrm i}\cfrac{(l+m)!}{l!}
\oint_C\cfrac{(z^2-1)^l}{(z-x)^{l+m+1}}dz\tag{3.5}
$$
其中 $C$ 为 $z$ 平面上围绕 $z=x$ 点任一闭合回路，这叫做==施列夫利积分== (SchlMli)。
还可以进一步表示为
$$
P^m_l(x)=\cfrac{\mathrm i^m}{2\pi}\cfrac{(l+m)!}{l!}
\int_{-\pi}^{\pi}e^{-\mathrm imψ}
[x+\cfrac{1}{2}\sqrt{x^2-1}(e^{-\mathrm iψ}+e^{\mathrm iψ})]^ldψ
$$
或变为==拉普拉斯积分== $(x=\cosθ)$ 
$$
P^m_l(x)=\cfrac{\mathrm i^m}{2\pi}\cfrac{(l+m)!}{l!}
\int_{-\pi}^{\pi}e^{-\mathrm imψ}
[\cosθ+\mathrm i\sinθ\cos\mathrm ψ]^ldψ\tag{3.6}
$$
**连带勒让德多项式的正交性**：在区间 $(-1,1)$ 上正交
$$
\int_{-1}^{1}P^m_l(x)P^m_k(x)dx=0\quad(l\neq k)
$$
如果从 $x$ 变换回变量 $θ,\ x=\cosθ$ ，则
$$
\int_{0}^{\pi}P^m_l(\cosθ)P^m_k(\cosθ)\sinθdθ=0\quad(l\neq k)
$$

**连带勒让德多项式的模**
$$
\|P^m_l(x）\|^2=\int_{-1}^{1}[P^m_l(x)]^2dx
$$
可计算得
$$
\|P^m_l(x）\|=\sqrt{\cfrac{2(l+m)!}{(2l+1)(l-m)!}}\quad(l=0,1,2,\cdots)\tag{3.7}
$$

**连带傅里叶-勒让德级数**：设函数 $f(x)$ 在区间 $[-1,1]$ 上满足狄利克雷条件，则 $f(x)$ 在连续点处展开为
$$
f(x)=\sum_{k=0}^{\infty}c_kP^m_k(x)
$$
其中系数
$$
c_k=\cfrac{(2l+1)(l-m)!}{2(l+m)!}\int_{-1}^{1}f(x)P^m_k(x)dx
$$
在物理上常取 $x=\cosθ(0⩽θ⩽\pi)$ ，则
$$
f(θ)=\sum_{k=0}^{\infty}c_kP^m_k(\cosθ)
$$
其中系数
$$
c_k=\cfrac{(2l+1)(l-m)!}{2(l+m)!}\int_{0}^{\pi}f(θ)P^m_k(\cosθ)\sinθdθ
$$

**连带勒让德多项式的递推关系**
$$
(k-m+1)P^m_{k+1}(x)-(2k+1)xP^m_k(x)+(k+m)P^m_{k-1}(x)=0\tag{3.8}
$$

## 球谐函数

**球函数**：我们回到拉普拉斯变换在球坐标下的分离变量，我们曾得到方程
$$
\cfrac{1}{\sinθ}\cfrac{∂}{∂θ}\left(\sinθ\cfrac{∂S}{∂θ}\right)
+\cfrac{1}{\sin^2θ}\cfrac{∂^2S}{∂ϕ^2}+μS=0\tag{4.1}
$$
称为==球函数方程==。令 $S(θ,ϕ)=Θ(θ)Φ(ϕ)$ 进一步分离变量，可获得其解
$$
\begin{aligned}
S_l^m(θ,ϕ) &=P_l^m(\cosθ)(A_l^m\cos mϕ+B_l^m\sin mϕ) \\
&=P_l^m(\cosθ)\begin{Bmatrix}
\sin mϕ \\
\cos mϕ \\
\end{Bmatrix}, 
\end{aligned}
\quad (m=0,1,2,\cdots,l)\tag{4.2}
$$
称为 $l$ 阶==球谐函数== (Spherical harmonics) 。其中常数 $μ=l(l+1)$ ，符号 $\{\}$ 表示其中列举的函数是线性独立的，可任取其一。

线性独立的 $l$ 阶球函数共有 $2l+1$ 个，这是因为对应于 $m=0$ ，只有一个球函数 $P_l(\cosθ)$ ，对应于 $m=1,2,\cdots,l$ ，则各有两个 $P^m_l(\cosθ)\sin mϕ$ 和 $P^m_l(\cosθ)\cos mϕ$。

**复数形式的球函数**：根据欧拉公式 (4.2) 可以完全写为
$$
S_l^m(θ,ϕ)=P_l^{|m|}(\cosθ)e^{\mathrm imϕ}\quad (m=0,\pm1,\pm2,\cdots,\pm l) \tag{4.3}
$$
**球函数正交关系**：任意两个球函数 (4.2) 在球面 $S\ (0⩽θ⩽\pi,0⩽ϕ⩽2\pi)$ 上正交
$$
\int_0^{\pi}\int_0^{2\pi}S_l^m(θ,ϕ)S_k^n(θ,ϕ)\sinθdθdϕ=0\quad(m\neq n\text{ or }l\neq k)
$$
**球函数的模**：
$$
\|S_l^m(θ,ϕ)\|^2=\int_0^{\pi}\int_0^{2\pi}[S_l^m(θ,ϕ)]^2\sinθdθdϕ
$$
计算得
$$
\|S_l^m(θ,ϕ)\|=\sqrt{\cfrac{2\piδ_m(l+m)!}{(2l+1)(l-m)!}}\tag{4.4}
$$
其中  $δ_m=\begin{cases}2&(m=0) \\ 1 & (m=1,2,\cdots)\end{cases}$

复数形式的模可写成
$$
\|S_l^m(θ,ϕ)\|=\sqrt{\cfrac{4\pi(l+|m|)!}{(2l+1)(l-|m|)!}}\tag{4.5}
$$
**广义傅里叶级数**：定义在球面 $S$ 上的函数 $f(θ,ϕ)$ 以球函数为基的二重傅里叶展开为
$$
f(θ,ϕ)=\sum_{m=0}^{\infty}\sum_{l=m}^{\infty}[A_l^m\cos mϕ+B_l^m\sin mϕ]P_l^m(\cosθ)\tag{4.6}
$$
其中系数为
$$
A_l^m=\cfrac{(2l+1)(l-m)!}{2\piδ_m(l+m)!}
\int_0^{\pi}\int_0^{2\pi}f(θ,ϕ)P_l^m(\cosθ)\cos mϕ\sinθdθdϕ
$$

$$
B_l^m=\cfrac{(2l+1)(l-m)!}{2\pi(l+m)!}
\int_0^{\pi}\int_0^{2\pi}f(θ,ϕ)P_l^m(\cosθ)\sin mϕ\sinθdθdϕ
$$

复数形式的傅里叶展开为
$$
f(θ,ϕ)=\sum_{l=0}^{\infty}\sum_{m=-l}^{l}C_l^mP_l^{|m|}(\cosθ)e^{\mathrm imϕ}\tag{4.7}
$$
其中系数为
$$
C_l^m=\cfrac{(2l+1)(l-|m|)!}{4\pi(l+|m|)!}
\int_0^{\pi}\int_0^{2\pi}f(θ,ϕ)P_l^{|m|}(\cosθ)[e^{\mathrm imϕ}]^*\sinθdθdϕ
$$
其中 $[e^{\mathrm imϕ}]^*$ 是 $e^{\mathrm imϕ}$ 的共轭复数。

**正交归一化**：物理中常常用正交归一化的球函数
$$
Y_l^m=\sqrt{\cfrac{(2l+1)(l-|m|)!}{4\pi(l+|m|)!}}
P_l^{|m|}(\cosθ)e^{\mathrm imϕ}\quad (m=0,\pm1,\pm2,\cdots,\pm l) \tag{4.8}
$$
这时就有正交归一关系
$$
\int_0^{\pi}\int_0^{2\pi}Y_l^m(θ,ϕ)Y_k^n(θ,ϕ)\sinθdθdϕ=δ_{l,k}δ_{m,n}
$$

# 柱函数

## 贝塞尔方程的解

**贝塞尔方程**(Bessel equation) 
$$
x^2y''+xy'+(x^2-ν^2)y=0\tag{1.1}
$$

其中 $ν$ 为实参数。由于方程是二阶变系数常微分方程，采用[幂级数求解][ode]，设其解的形式为
$$
y=\displaystyle\sum_{k=0}^∞c_kx^{k+r}\tag{1.2}
$$

其中 $c_0\neq0,\quad c_k,r$ 是待定常数。带入贝塞尔方程可得
$$
\displaystyle x^2\sum_{k=1}^∞(k+r)(k+r-1)c_kx^{k+r-2} +x\sum_{k=1}^∞(k+r)c_kx^{k+r-1} 
+(x^2-ν^2)\sum_{k=0}^∞(k+r)c_kx^{k+r}=0
$$
进一步合并 $x$ 的同幂项
$$
\displaystyle\sum_{k=0}^∞[(k+r)(k+r-1)+(k+r)-ν^2]c_kx^{k+r}+\sum_{k=0}^∞c_kx^{k+r+2}=0
$$
令各项的系数等于零，得代数方程组
$$
\begin{cases}
c_0[r^2-ν^2]=0 \\
c_1[(r+1)^2-ν^2]=0 \\
\cdots\quad\cdots \\
c_k[(r+k)^2-ν^2]+c_{k-2}=0 \\
\cdots\quad\cdots
\end{cases}
$$
因为 $c_0\neq0$ ，故从方程组解得 $r=\pm ν$
(1) 当 $r= ν$ 时，带入代数方程组可得
$$
c_1=0,\quad c_k=-\cfrac{c_{k-2}}{k(2ν+k)}\quad (k=2,3,\cdots)
$$
或按下标是奇数或偶数，我们分别有
$$
\begin{cases}
c_{2k+1}=\cfrac{-c_{2k-1}}{(2k+1)(2ν+2k+1)} \\
c_{2k}=\cfrac{-c_{2k-2}}{2k(2ν+2k)}
\end{cases}\quad (k=1,2,\cdots)
$$
从而求得
$$
\begin{cases}
c_{2k-1}=0 \\
c_{2k}=(-1)^k\cfrac{c_0}{2^{2k}k!(ν+1)(ν+2)\cdots(ν+k)} \\
\end{cases}\quad (k=1,2,\cdots)
$$
将各 $c_k$ 带入 (1.2) 得到贝塞尔方程得一个解
$$
\displaystyle y_1=c_0x^ν+\sum_{k=1}^{∞}(-1)^k\cfrac{c_0}{2^{2k}k!(ν+1)(ν+2)\cdots(ν+k)}x^{2k+ν}
$$
此时 $c_0$ 仍是任意常数，通常为求特解取
$c_0=\cfrac{1}{2^ν\Gamma(ν+1)}$ ，其中[^gamma] $\displaystyle\Gamma(s)=\int_{0}^{∞}x^{s-1}e^{-x}dx$
从而上式特解变为 
$$
\displaystyle J_ν(x)=\sum_{k=0}^{∞}\cfrac{(-1)^k}{k!\Gamma(ν+k+1)}(\cfrac{x}{2})^{2k+ν}\tag{1.3}
$$

$J_ν(x)$ 是由贝塞尔方程定义得特殊函数，称为 $ν$ 阶==第一类贝塞尔函数==。
由达朗贝尔判别法不难验证级数 $J_ν(x)$ 在 $(-\infty,+\infty)$ 收敛，因此，贝塞尔方程总有一个特解 $J_ν(x)$，我们只需寻求另一个线性无关的特解即可求得贝塞尔方程通解。

 (2) 当 $r=-ν$ 时，和 $r=ν$ 的求解过程一样，我们可以求得另一个特解
$$
\displaystyle y_2=c_0x^{-ν}+\sum_{k=1}^{∞}(-1)^k\cfrac{c_0}{2^{2k}k!(-ν+1)(-ν+2)\cdots(-ν+k)}x^{2k-ν}
$$
此时，令 $c_0=\cfrac{1}{2^{-ν}\Gamma(-ν+1)}$ ，从而上式特解变为 
$$
\displaystyle J_{-ν}(x)=\sum_{k=0}^{∞}\cfrac{(-1)^k}{k!\Gamma(-ν+k+1)}(\cfrac{x}{2})^{2k-ν}\tag{1.4}
$$

级数 $J_{-ν}(x)$ 在 $x>0$ 时收敛。由于当 $ν$ 不为整数时  $J_{ν}(x)$ 和  $J_{-ν}(x)$ 线性无关，贝塞尔方程的通解为
$$
y=C_1J_ν(x)+C_2J_{-ν}(x) \tag{1.5}
$$
其中 $C_1,C_2$ 为任意常数。
有时取 $C_1=\cot ν\pi,\quad C_2=-\csc ν\pi$ 带入 (1.5) 得到一个特解，作为 $J_ν(x)$ 另一个线性无关的特解
$$
N_ν(x)=\cfrac{J_ν(x)\cos ν\pi-J_{-ν}(x)}{\sin ν\pi}\tag{1.6}
$$
叫做 $ν$ 阶==诺依曼函数== (Neumann)或 $ν$ 阶==第二类贝塞尔函数==。因此贝塞尔方程的通解也可取为
$$
y=C_1J_ν(x)+C_2N_ν(x)\tag{1.7}
$$
(3) 当 $ν=n$ 为正整数时，$J_{-n}(x)=(-1)^nJ_n(x)$ 与 $J_n(x)$ 线性相关。我们可以考虑诺依曼函数，定义
$$
Y_n(x)=\lim\limits_{α\to n}\cfrac{J_α(x)\cos ν\pi-J_{-α}(x)}{\sin α\pi}
$$
上式为 $\frac{0}{0}$ 不定式，根据洛必达法则得
$$
Y_n(x)=\lim\limits_{α\to n}(\cfrac{∂J_α(x)}{∂α}-\cfrac{1}{\cosα\pi}\cfrac{∂J_{-α}(x)}{∂α})
$$
可以证明 $J_n(x),Y_n(x)$ 线性无关，因此对于任意实数 $ν$ ，贝塞尔方程的通解为
$$
y=C_1J_ν(x)+C_2Y_ν(x)\tag{1.8}
$$
**虚宗量贝塞尔方程**
$$
x^2y''+xy'-(x^2+ν^2)y=0\tag{1.9}
$$
做变量变换 $t=\mathrm{i}x$ ，方程变为 $t^2y''+ty'+(t^2-ν^2)y=0$ 已求得其解。
(1) 当 $ν$ 非整数时，通常取一般解为
$$
y=C_1I_ν(x)+C_2I_{-ν}(x)\tag{1.10}
$$
其中 $I_ν(x)=\mathrm{i}^{-ν}J_ν(\mathrm{i}x),I_{-ν}(x)=\mathrm{i}^{ν}J_{-ν}(\mathrm{i}x)$ 为实值函数，称为==虚宗量贝塞尔函数==。
(2) 关于第二类虚宗量贝塞尔函数的处理，通常又取线性独立的特解
$$
\begin{cases}
H_ν^{(1)}(x)=J_ν(x)+\mathrm{i}N_ν(x) \\
H_ν^{(2)}(x)=J_ν(x)-\mathrm{i}N_ν(x)
\end{cases}
$$
并称为==第一种和第二种汉克儿函数== (Hankel)，或==第三类贝塞尔函数==。于是虚宗量贝塞尔方程的一般解又可表示为
$$
y=C_1H_ν^{(1)}(x)+C_2H_ν^{(2)}(x)
$$
为了获得两个线性独立的实数特解通常：
当 $ν$ 非整数时，取 
$$
K_ν(x)=\cfrac{\pi}{2}\mathrm{i}\exp(\cfrac{\mathrm{i}\pi ν}{2})H_ν^{(1)}(\mathrm{i}x)
=\cfrac{\pi}{2\sinν\pi}[I_{-ν}(x)-I_ν(x)]
$$
当 $ν= n$ 是整数时，取极限 
$$
K_ν(x)=\lim\limits_{α\to ν}\cfrac{\pi}{2\sinα\pi}[I_{-α}(x)-I_α(x)]
$$
函数 $K_ν(x)$ 称为==虚宗量汉克尔函数==  。通常取虚宗量贝塞尔方程的解为
$$
y=C_1I_ν(x)+C_2K_{ν}(x)\tag{1.11}
$$

## 贝塞尔函数

**整数阶贝塞尔函数的性质** ：第二、三类贝塞尔函数都是第一类贝塞尔函数的线性组合，因此第一类贝塞尔函数的性质都适用。

(1) $J_{n}(x)$ 与 $J_{-n}(x)$ 线性相关
$$
J_{-n}(x)=(-1)^nJ_n(x)
$$
(2) $J_n(x)$ 的奇偶性
$$
J_n(-x)=(-1)^nJ_n(x)
$$
(3) $J_n(x)$ 的生成函数
$$
\displaystyle\exp[\cfrac{x}{2}(t-\cfrac{1}{t})]
=\sum_{n=-\infty}^{+\infty}J_n(x)t^n\quad (n\neq0)
$$
 (4) $J_n(x)$ 的积分表示：生成函数中令 $t=e^{\mathrm iθ}$ 得到
$$
J_n(x)=\cfrac{1}{\pi}\int_0^\pi\cos(x\sinθ-nθ)dθ
$$
(5) 如果生成函数中令 $t=\mathrm ie^{\mathrm iθ}$ 得到
$$
e^{\mathrm ikr\cosθ}=J_0(kr)+2\displaystyle\sum_{n=1}^{\infty}\mathrm i^nJ_n(kr)\cos nθ
$$
**$ν$ 阶贝塞尔函数的性质**

(1) 递推关系式
$$
\cfrac{d}{dx}[x^νJ_ν(x)]=x^νJ_{ν-1}(x) \\
\cfrac{d}{dx}[x^{-ν}J_ν(x)]=-x^{-ν}J_{ν+1}(x) \tag{2.1}
$$
从递推关系式中还可以得到两个新的关系式
$$
J_{ν-1}(x)-J_{ν+1}(x)=2J'_{ν}(x) \\
J_{ν-1}(x)+J_{ν+1}(x)=\cfrac{2ν}{x}J_{ν}(x)
$$
(2) 贝塞尔函数 $J_{ν}(x)$ 与 $J_{-ν}(x)$ 的 Wronski 行列式
$$
W[J_{ν}(x),J_{-ν}(x)]=\begin{vmatrix}
J_{ν}(x) & J_{-ν}(x) \\
J'_{ν}(x) & J'_{-ν}(x)
\end{vmatrix}=-\cfrac{2}{\pi x}\sin\piν\tag{2.2}
$$
**半奇数阶贝塞尔函数**：第一类贝塞尔函数和诺依曼函数一般不是初等函数，但半奇数阶第一类贝塞尔函数 $J_{n+\frac{1}{2}}(x)\quad (n=0,1,2,\cdots)$  可以用初等函数表示，由递推关系可得
$$
J_{\frac{1}{2}}(x)=\sqrt{\cfrac{2}{\pi x}}\sin x,\quad 
J_{-\frac{1}{2}}(x)=\sqrt{\cfrac{2}{\pi x}}\cos x
$$

$$
J_{n+\frac{1}{2}}(x)=(-1)^n\sqrt{\cfrac{2}{\pi x}}x^n
(\cfrac{d}{xdx})^n(\cfrac{\sin x}{x}) \\
J_{-(n+\frac{1}{2})}(x)=\sqrt{\cfrac{2}{\pi x}}x^n
(\cfrac{d}{xdx})^n(\cfrac{\cos x}{x}) \tag{2.3}
$$
**贝塞尔函数的零点**：即方程 $J_{ν}(x)=0$ 的根，在求解数学物理方程定解问题时，具有重要意义。 由级数表达式 (1.3) 知 $J_{ν}(x)$ 为偶函数，故实数零点存在的话，必然成对出现，而 $J_0(0)=1,J_ν(0)=0(ν>0)$ 。下面给出一些结论：
(1) $J_{ν}(x)$ 由无穷多个单重零点，且在实轴上关于原点对称分布，因而必有无穷多个正零点。当 $ν>-1$ 或为整数时，只有实数零点。
(2) $J_{ν}(x)$ 的零点和$J_{ν+1}(x)$  的零点彼此相间，且没有非零的公共零点。
(3) 设 $μ_1<μ_2<\cdots<μ_m<μ_{m+1}<\cdots$ 表示 $J_{ν}(x)$的正实零点，则当 $m\to\infty$ 时，$μ_{m+1}-μ_{m}\to\pi$ ，即 $J_{ν}(x)$ 几乎是以 $2\pi$ 为周期的周期函数。
(4) 第二类贝塞尔函数 $Y_ν(x)$ 的零点分布在 $(0,+\infty)$ 上，他与第一类贝塞尔函数零点有相似的结论。 
(5) 虚宗量贝塞尔函数 $I_ν(x)$ 和虚宗量贝塞尔函数 $K_ν(x)$ 不存在实数零点。

<img src="https://gitee.com/WilenWu/images/raw/master/DifferentialEquation/Bessel-fun1.png"  width="70%" /> <img src="https://gitee.com/WilenWu/images/raw/master/DifferentialEquation/Bessel-fun2.png"  width="60%" />

**贝塞尔函数的渐进展开**：一般用于判断自然边界条件进行取舍
(1) 当 $x\to0$ 时，$J_0(x)\to1,\quad J_ν(x)\to0,\quad J_{-ν}(x)\to\infty \\
N_0(x)\to-\infty,\quad N_ν(x)\to\pm\infty\quad(ν\neq0)$
(2) 当 $x\to\infty$ 时，$H_ν^{(1)}(x)\sim\sqrt{\cfrac{2}{\pi x}}\exp[\mathrm i(x-\cfrac{ν\pi}{2}-\cfrac{\pi}{4})] \\
H_ν^{(2)}(x)\sim\sqrt{\cfrac{2}{\pi x}}\exp[-\mathrm i(x-\cfrac{ν\pi}{2}-\cfrac{\pi}{4})] \\
J_ν(x)\sim\sqrt{\cfrac{2}{\pi x}}\cos(x-\cfrac{ν\pi}{2}-\cfrac{\pi}{4}) \\
N_ν(x)\sim\sqrt{\cfrac{2}{\pi x}}\sin(x-\cfrac{ν\pi}{2}-\cfrac{\pi}{4})$

**贝塞尔函数与本征值问题**：以三维空间拉普拉斯方程圆柱坐标系分离变量法为例
在圆柱内 $(0⩽r⩽r_0)$ 关于半径 $r$ 的微分方程
$$
R''+\cfrac{1}{r}R'-(μ+\cfrac{m^2}{r^2})R=0\quad(m=0,1,2,\cdots)
$$
分三种情况讨论，如果有柱侧边界条件的限制，当 $μ>0$ 时，得到虚宗量贝塞尔方程，不存在实数零点，应予排除。
当 $μ⩽0$ 取 $μ=−ν^2, x=νr$ 得到整数 $m$ 阶==贝塞尔方程== (Bessel)
$$
x^2\cfrac{d^2R}{dx^2}+x\cfrac{dR}{dx}+(x^2-m^2)R=0
$$
在这个方程的线性独立解中，由于自然边界条件的限制 $m ⩾ 0$ ，我们只要非负阶贝塞尔函数
$$
R(r)=J_m(x)=J_m(\sqrt{-μ}r)
$$
再由圆柱侧面的齐次边界条件决定本征值 $μ$ ，和相应的本征函数 $R(r)$ 。
对于第一类齐次边界条件 $R(r_0)=0$ ，即 $J_m(\sqrt{-μ}r_0)=0$ 
对于第二类齐次边界条件 $R'(r_0)=0$ ，即 $\sqrt{-μ}J'_m(\sqrt{-μ}r_0)=0$ ，若 $\mu\neq0$ ，则 $J'_m(\sqrt{-μ}r_0)=0$
对于第三类齐次边界条件 $R(r_0)+HR'(r_0)=0$ ，即 $J_m(\sqrt{-μ}r_0)+H\sqrt{-μ}J'_m(\sqrt{-μ}r_0)=0$ 。

**贝塞尔函数的正交性**：作为施图姆-刘维尔本征值问题正交关系的特例，用 $μ_k$ 表示 $J_m(\sqrt{μ}r)$ 在圆柱侧面常见的三类齐次边界条件的第 $k$ 个正根或本征值，在区间 $[0,r_0]$ 上带权重函数 $r$ 正交
$$
\displaystyle\int_{0}^{r_0}J_m(\sqrt{μ_k}r)J_m(\sqrt{μ_l}r)rdr=0\quad(k\neq l)
$$
**贝塞尔函数的模**：为了用于计算基于贝塞尔函数的广义傅里叶展开，定义模
$$
\begin{aligned}
\|J_m(\sqrt{μ_k}r)\|^2 &=\int_{0}^{r_0}J^2_m(\sqrt{μ_k}r)rdr \\
&=\cfrac{1}{2}(r_0^2-\cfrac{m^2}{μ_k})J^2_{m}(\sqrt{μ_k}r_0)
+\cfrac{1}{2}r_0^2[J'_{m}(\sqrt{μ_k}r_0)]^2
\end{aligned}\tag{2.4}
$$
**傅里叶-贝塞尔级数**：设函数 $f(r)$ 在区间 $[0,r_0]$ 上满足狄利克雷条件，且 $f(0)$ 有界，$f(r_0)=0$ 则函数 $f(r)$ 的傅里叶-贝塞尔级数是
$$
\displaystyle f(r)=\sum_{k=1}^{\infty}c_kJ_m(\sqrt{μ_k}r)\tag{2.5}
$$
其中系数
$$
c_k=\cfrac{1}{\|J_m(\sqrt{μ_k}r)\|^2}\int_0^{r_0}f(r)J_m(\sqrt{μ_k}r)rdr\tag{2.6}
$$
当 $r_0\to\infty$ 时，则有**傅里叶-贝塞尔积分**
$$
f(r)=\int_0^{\infty}F(ω)J_m(ωr)ωdω \\
F(ω)=\int_0^{\infty}f(r)J_m(ωr)rdr
$$


## 球贝塞尔方程

$$
\cfrac{d}{dr}\left(r^2\cfrac{dR}{dr}\right)+[k^2r^2-l(l+1)]R=0\tag{3.1}
$$

做变量变换 $x=kr,\quad R(r)=\sqrt{\cfrac{\pi}{2x}}y(x)$ 带入上式，则方程化为 $l+1/2$ 阶贝塞尔方程
$$
x^2y''+xy'+[x^2-(l+\cfrac{1}{2})^2]y=0\tag{3.2}
$$
若 $k=0$ ，方程 (3.1) 退化为
$$
r^2R''+2rR'-l(l+1)R=0
$$
其两个线性独立的解为 $r^l,1/r^{l+1}$ 较为简单，下面着重讨论 $k\neq0$ 的情形

**线性独立解**

 $l+1/2$ 阶贝塞尔方程有如下几种解
$$
J_{l+1/2}(x),\ J_{-(l+1/2)}(x),\ N_{l+1/2}(x),\ H^{(1)}_{l+1/2}(x),\ H^{(2)}_{l+1/2}(x)
$$
其中任取两个就组成方程 (3.2) 的线性独立解。这样求贝塞尔方程 (3.1) 的线性独立解就是下列五种任取两种
球贝塞尔函数
$$
j_l(x)=\sqrt{\frac{\pi}{2x}}J_{l+1/2}(x),\\ j_{-l}(x)=\sqrt{\frac{\pi}{2x}}J_{-l+1/2}(x)
$$
球诺依曼函数
$$
n_l=\sqrt{\frac{\pi}{2x}}N_{l+1/2}(x)
$$
球汉克儿函数
$$
h^{(1)}_l(x)=\sqrt{\frac{\pi}{2x}}H^{(1)}_{l+1/2}(x),\\ h^{(2)}_l(x)=\sqrt{\frac{\pi}{2x}}H^{(2)}_{l+1/2}(x)
$$
球汉克儿函数由定义知  $h^{(1)}_l(x)=j_l(x)+\mathrm in_l(x),\quad h^{(2)}_l(x)=j_l(x)-\mathrm in_l(x)$

**初等函数表示式**
$$
j_0(x)=\cfrac{\sin x}{x},\quad j_{-1}(x)=\cfrac{\cos x}{x}
$$

$$
n_l(x)=(-1)^{l+1}j_{-(l+1)}(x) \\n_0(x)=\cfrac{\cos x}{x},\quad n_{-1}(x)=\cfrac{\sin x}{x}
$$



