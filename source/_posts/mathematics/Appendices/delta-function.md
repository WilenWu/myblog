---
title: δ 函数简介
categories:
  - Mathematics
  - 附录
tags:
  - 数学
katex: true
top_img: '#66CCFF'
abbrlink: 3dc94f84
date: 2022-07-15 19:56:34
updated:
keywords:
description:
cover:
---

# δ 函数

在物理学中，常有集中于一点或一瞬时的量，如脉冲力、脉冲电压、点电荷、质点的质量。只有引入一个特殊函数来表示它们的**分布密度**，才有可能把这种集中的量与连续分布的量来统一处理。

- **单位脉冲函数**(Unit Impulse Function)
<引例>：假设在原来电流为零的电路中，在 $t=0$ 时瞬时进入一电量为 $q_0$的脉冲。现在确定电流强度分布 $i(t)=\cfrac{\mathrm dq}{\mathrm dt}$，分析可知 $i(t)=\begin{cases}
0&(t\neq 0) \\
∞&(t=0)
\end{cases}$
同时需要引入积分值表示电量大小 $\displaystyle\int_{-∞}^{+∞}i(t)dt=q_0$
为此我们引入==单位脉冲函数==，又称为==Dirac函数或者δ函数==。

   <kbd>定义</kbd>：单位脉冲函数 $δ(t)$ 满足
(1) 当 $t\neq 0$ 时，$δ(t)=0$
(2) $\displaystyle\int_{-∞}^{+∞}δ(t)dt=1$
由此，引例可表示为 $i(t)=q_0δ(t)$
![delta函数](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/delta-fun.png)
**注意**：
(1) 单位脉冲函数 $δ(t)$ 并不是经典意义下的函数，因此通常称其为广义函数(或者奇异函数)。
(2) 它不能用常规意义下的值的对应关系来理解和使用，而总是通过它的定义和性质来使用它。
(3) 单位脉冲函数 $δ(t)$ 有多种定义方式，前面所给出的定义方式是由Dirac(狄拉克)给出的。

- **单位脉冲函数其他定义方式**
构造一个在 $ε$ 时间内激发的矩形脉冲 $δ_ε(t)$，定义为
$δ_ε(t)=\begin{cases}
0&(t< 0) \\
1/ε&(0⩽t⩽ε) \\
0&(t>ε)
\end{cases}$  
对于任何一个在 $(-∞,+∞)$ 上无穷次可微的函数 $f(t)$ 如果满足
$$
\displaystyle\lim\limits_{ε\to 0}\int_{-∞}^{+∞}δ_ε(t)f(t)dt=\int_{-∞}^{+∞}δ(t)f(t)dt
$$
则称$δ_ε(t)$的极限为$δ(t)$，记为
$$
\lim\limits_{ε\to 0}δ_ε(t)=δ(t)
$$
![delat函数](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/delta-fun2.png)

  <kbd>筛选性质</kbd>：(sifting property)设函数 $f(t)$ 是定义在 $\R$上的有界函数，且在 $t = 0$ 处连续，则有
$$
\displaystyle\int_{-∞}^{+∞}δ(t)f(t)dt=f(0)
$$
证明：取 $f(t)\equiv1$，则有 $\displaystyle\int_{-∞}^{+∞}δ(t)dt=\lim\limits_{ε\to 0}\int_{0}^{ε}\frac{1}{ε}dt=1$
事实上 $\displaystyle\int_{-∞}^{+∞}δ(t)f(t)dt=\lim\limits_{ε\to 0}\int_{-∞}^{+∞}δ_ε(t)f(t)dt=\lim\limits_{ε\to 0}\frac{1}{ε}\int_{0}^{ε}f(t)dt$
由微分中值定理有 $\displaystyle\frac{1}{ε}\int_{0}^{ε}f(t)dt=f(θε)\quad(0<θ<1)$
从而 $\displaystyle\int_{-∞}^{+∞}δ(t)f(t)dt=\lim\limits_{ε\to 0}f(θε)=f(0)$

 正是因为 $δ$ 函数并不是给出普通数值间的对应关系，因此，$δ$ 函数也不像普通函数那样具有唯一确定的表达式，事实上凡是具有
$$
   \lim\limits_{ε\to 0}\int_{-∞}^{+∞}δ_ε(t)f(t)dt=f(0)
$$
 性质的函数序列 $δ_ε(t)$ ，或是具有
$$
 \lim\limits_{n\to \infty}\int_{-∞}^{+∞}δ_n(t)f(t)dt=f(0)
$$
 性质的函数序列 $δ_n(t)$，他们的极限都是 $δ$ 函数，例如

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/delta-series-demo.png" style="zoom: 8%;" /> <img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/delta-series-demo2.png" style="zoom:8%;" />
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/delta-series-demo3.png" style="zoom:8%;" /> <img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/delta-series-demo4.png" style="zoom:8%;" />

对于连续分布的物理量 $Q$ ，通常有两种描述方式，一种是局部性的，给出密度分布函数
$$
\rho(\mathbf r)=\cfrac{\mathrm dQ}{\mathrm d\mathbf r}
$$
另一种是整体性的
$$
Q=\int_V\rho(\mathbf r)\mathrm d\mathbf r
$$
# 基本性质

这些性质的严格证明可参阅广义函数

(1) $δ(t)$ 和常数 $c$ 的乘积 $cδ(t)$
$$
\int_{-∞}^{+∞}[cδ(t)]f(t)dt=\int_{-∞}^{+∞}δ(t)[cf(t)]dt=cf(0)
$$
(2) 平移变换， $t\to t-t_0$ 
$$
\int_{-∞}^{+∞}δ(t-t_0)f(t)dt=\int_{-∞}^{+∞}δ(x)f(x+t_0)dx=f(t_0)
$$
(3) 放大（或缩小）变换， $t\to at  \quad(a\neq 0)$
$$
\int_{-∞}^{+∞}δ(at)f(t)dt=δ(x)f(\frac{x}{a})\frac{dx}{|a|}=\frac{1}{|a|}f(0)
$$
由此可以得到
$$
δ(at)=\cfrac{1}{|a|}δ(t)\quad(a\neq 0)
$$

   特别的，当 $a=-1$ 时，$δ(t)=δ(-t)$ ，说明 $δ(t)$为==偶函数==。

(4)  $δ$ 函数的导数 $δ'(t)$ ，对于在 $t=0$ 点连续并有连续导数的任意函数 $f(t)$ ，应用分部积分
$$
\int_{-∞}^{+∞}δ'(t)f(t)dt=δ(t)f(t)\Big|_{-∞}^{+∞}-\int_{-∞}^{+∞}δ(t)f'(t)dt=-f'(0)
$$
 (5)  $δ$ 函数的高阶导数 $δ^{(n)}(t)$ ，对于在 $t=0$ 点连续并有连续导数的任意函数 $f(t)$ ，有
$$
\int_{-∞}^{+∞}δ^{(n)}(t)f(t)dt=(-1)^{n}f^{(n)}(0)
$$
(6) $δ$ 函数与普通函数的乘积 $g(t)δ(t)$
$$
\int_{-∞}^{+∞}[g(t)δ(t)]f(t)dt=\int_{-∞}^{+∞}[f(t)g(t)]δ(t)dt=f(0)g(0)
$$
即 
$$
f(t)δ(t)=f(0)δ(t)
$$
例如：  $tδ(t)=0$

(7) 单位阶跃函数[^unit]等于 $δ$ 函数的积分
$$
\displaystyle u(t)=\int_{-∞}^{t}δ(s)ds
$$
由高数知识知，$δ$ 函数是单位阶跃函数的导数，即
$$
\dfrac{\mathrm du(t)}{\mathrm dt}=δ(t)
$$

   (8) $δ$ 函数的卷积
$$
   f(t)*δ(t)=f(t)
$$
一般的有 $f(t)*δ(t-t_0)=f(t-t_0)$

[^unit]: 单位阶跃函数(unit step function)，也称Heaviside单位函数

$$
u(t)=\begin{cases}
0 & t<0 \\ 1 &t>0 \end{cases}
$$

![单位阶跃函数](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/unit-step-fun.png)

按广义函数理论，定义为

$$
\displaystyle\int_{-∞}^{+∞}u(t)f(t)dt=\int_{0}^{+∞}f(t)dt
$$
单位阶跃函数的积分为：
$$
\int_{-\infty}^{t}u(\tau)\mathrm d\tau=tu(t)
$$

# Fourier 变换

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

# Fourier 展开

当 $x,x_0\in(-\pi,\pi)$ 时
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
#  Laplace 变换

狄利克雷函数 
$$
δ_τ(t)=\begin{cases}
 \frac{1}{τ} &0⩽ t<τ \\
 0  &\text{others}
\end{cases}
$$
的Laplace 变换为
$$
\displaystyle \mathcal L[δ_τ(t)]=\int^{τ}_{0}\frac{1}{τ}e^{-st}\text{d}t=\frac{1}{τs}(1-e^{-τs})
$$
所以
$$
\displaystyle \mathcal L[δ(t)]=\lim\limits_{τ\to0}\mathcal L[δ_τ(t)]=\lim\limits_{τ\to0}\frac{1}{τs}(1-e^{-τs})
$$
用洛必达法则计算此极限 
$$
\displaystyle\lim\limits_{τ\to0}\frac{1}{τs}(1-e^{-τs})=\lim\limits_{τ\to0}\frac{se^{-τs}}{s}=1
$$
所以
$$
\mathcal L[δ(t)]=1
$$

# 多维 $δ$ 函数

例如位于三维空间的坐标原点质量为 $m$ 的质点，其密度函数可表示为 $mδ(\mathbf r)$。	在三维空间中的 $δ$ 函数定义如下：
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
|        原函数 |          | 像函数                           |
| ------------: | :------: | :------------------------------- |
|    $δ(x,y,z)$ | $\lrarr$ | $1$                              |
|           $1$ | $\lrarr$ | $(2\pi)^3δ(λ,μ,ν)$               |
|           $x$ | $\lrarr$ | $(2\pi)^3i\cfrac{∂δ(λ,μ,ν)}{∂λ}$ |
| $x^2+y^2+z^2$ | $\lrarr$ | $(2\pi)^3δ(λ,μ,ν)$               |
|     $e^{iax}$ | $\lrarr$ | $(2\pi)^3δ(λ-a,μ,ν)$             |

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

# 非齐次项为 δ 函数的常微分方程

在传统意义下，非齐次项为 $δ$ 函数的常微分方程没有意义。

- 正当  $δ$ 函数应当理解为连续函数序列 $\{δ_n(x)\}$ 的极限一样，这类常微分方程也应当理解为非齐次项为 $δ_n(x)$ 的常微分方程的极限。
- 这类常微分方程的解也应当理解为非齐次项为 $δ_n(x)$ 的常微分方程的解的极限（先解微分方程再取极限）。
- 引进  $δ$ 函数的好处就在于可以直接处理这类极限情形的微分方程求解问题，而不必考虑具体的函数序列以及它的极限过程。
- 正因为  $\delta$  函数不是传统意义下的函数，使得这类常微分方程的解具有独特的连续性质。就二阶常微分方程而言，我们将要看到，它的解是连续的，但是解的一阶导数不连续。正是由于一阶导数的不连续，才使得它正好是非齐次项为 $δ$ 函数的常微分方程。

非齐次项为 $δ$ 函数的常微分方程，这是一种特殊的非齐次方程，除了使用 $\delta$  函数的个别点外，方程是齐次的，使得这种非齐次常微分方程又很容易求解，特殊情形下甚至可以直接积分求解。

示例 1：求解初值问题（初位移和初速度为 0 的物体，在 $t_0$ 时刻受到瞬时冲量）
$$
\begin{cases}
\cfrac{d^2s}{dt^2}=\delta(t-t_0) & t>0,t_0>0 \\
s|_{t=0}=0,\quad \cfrac{ds}{dt}|_{t=0}=0
\end{cases}
$$
解：直接积分
$$
\cfrac{ds}{dt}=u(t-t_0)+c_1
$$
其中函数 $u(t)$ 为单位阶跃函数[^unit]，再次积分
$$
s=(t-t_0)u(t-t_0)+c_1t+c_2
$$
带入初始条件可得
$$
c_1=c_2=0
$$
于是
$$
s=(t-t_0)u(t-t_0)
$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/ODE-delta.png" style="zoom:67%;" /> <img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/ODE-delta2.png" style="zoom:67%;" />

示例 2：求解边值问题（物体在 $t=a,b$ 时刻的位移为 0，在 $t_0$ 时刻受到瞬时冲量）
$$
\begin{cases}
\cfrac{d^2s}{dt^2}=\delta(t-t_0) & 0<a<t_0<b \\
s|_{t=a}=0,\quad s|_{t=b}=0
\end{cases}
$$
解：直接积分可求得
$$
s=(t-t_0)u(t-t_0)+v_1t+v_2
$$
带入初始条件可解得
$$
\begin{cases}
v_1=-\cfrac{b-t_0}{b-a} \\
v_2=-v_1a
\end{cases}
$$
于是
$$
s=(t-t_0)u(t-t_0)-\frac{b-t_0}{b-a}(t-a)
$$
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/ODE-delta3.png" style="zoom:67%;" /><img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ComplexFunction/ODE-delta4.png" style="zoom:67%;" />
