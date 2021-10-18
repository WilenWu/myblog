---
title: 复变函数和积分变换(Integral Transform)
date: 2019-07-24 13:09:37
categories: [数学,复变函数]
tags: [数学,复傅里叶变换,拉普拉斯变换]
cover: 
top_img: 
katex: true
description: false
---


> 参考文献：
> mooc国防科技大学《复变函数》
> 王忠仁、张静《工程数学：复变函数和积分变换》
> 焦红伟、尹景本《复变函数与积分变换》
> 梁昆淼《数学物理方法》

# Fourier 变换

所谓积分变换，就是把某函数类 A 中的函数 $f(t)$ 乘上一个确定的二元函数$k(t, p)$，然后计算积分 $\displaystyle F(p)=\int k(t, p)f(t)dt$，这样变成另一个函数类 B 中的函数 $F(p)$ 。这里二元函数$k(t, p)$是一个确定的二元函数，通常称为该积分变换的==核函数(kernel function)==， $f(t)$ 称为象原函数(original image function)，$F(p)$ 称为 $f(t)$的象函数(image function)。如果取积分核 $k(ω,t)=e^{-iωt}$，就是著名的Fourier 变换。

## Fourier 变换

- **周期函数的Fourier 级数**：设 $f_T(t)$ 是以T为周期的实值函数，在区间 $[-\frac{T}{2},\frac{T}{2}]$上满足狄利克雷(Dirichlet)条件：
(1)连续或只有有限个第一类间断点；
(2)只有有限个极值点
则$f_T(t)$在连续点处可以展开成Fourier 级数：
$$
\displaystyle f_T(t)=\dfrac{a_0}{2}+\sum_{n=1}^{∞}(a_n\cos nω_0 t+b_n\sin nω_0 t) \tag{F0}
$$
在间断点处，上式左端为 $\frac{1}{2}[f_T(t^-)+f_T(t^+)]$
其中 $\displaystyle ω_0=2\pi/T \\
a_n=\frac 2T \int_{-T/2}^{T/2}f_T(t)\cos nω_0 t\text{d}t \quad(n=0,1,2,\cdots) \\
b_n=\frac 2T \int_{-T/2}^{T/2}f_T(t)\sin nω_0 t\text{d}t \quad(n=1,2,3,\cdots)$
式 (F0) 称为Fourier 级数的三角形式。

- **奇函数和偶函数的傅里叶展开**
若周期函数 $f_T(t)$ 是奇函数，由展开式知 $a_0$ 及 $a_n$ 均为零，展开式称为
$$
\displaystyle f_T(t)=\sum_{n=1}^{∞}b_n\sin nω_0 t
$$
称为==傅里叶正弦级数==。
若周期函数 $f_T(t)$ 是偶函数，由展开式知 $b_n$ 均为零，展开式称为
$$
\displaystyle f_T(t)=\dfrac{a_0}{2}+\sum_{n=1}^{∞}a_n\cos nω_0 t
$$
称为==傅里叶余弦级数==。

- **Fourier 级数的指数形式**：利用欧拉公式 $\cosθ=\dfrac{e^{iθ}+e^{-iθ}}{2},\sinθ=\dfrac{e^{iθ}-e^{-iθ}}{2i}$ 将Fourier 级数转化为复指数形式， 
$$
\displaystyle f_T(t)=c_0+\sum_{n=1}^{∞}(c_ne^{inω_0 t}+c_{-n}e^{-inω_0 t})=\sum_{n=-∞}^{∞}c_ne^{inω_0 t} \tag{F1}
$$
 其中 
$$
 \displaystyle c_n=\dfrac1T\int^{T/2}_{-T/2}f_T(t)e^{-inω_0 t}\text{d}t\quad(n=0,\pm1,\pm2,\cdots)\tag{F2}
$$
  由 $c_n$与$a_n,b_n$的关系可知
$\begin{cases}
c_n=c_{-n}=\frac{1}{2}\sqrt{a_n^2+b_n^2}=\frac{1}{2}A_n \\
\arg c_n=-\arg c_{-n}=θ_n \\
|c_0|=A_0
\end{cases}$


- **Fourier 级数的物理含义**：
针对Fourier 级数的三角形式 (F0) ，取$A_0=a_0/2$，令$A_n=\sqrt{a_n^2+b_n^2},\cosθ_n=a_n/A_n,\sinθ_n=-b_n/A_n$，则(F0)化为
$\begin{aligned}
\displaystyle f_T(t)&=A_0+\sum_{n=1}^{∞}A_n(\cosθ_n\cos nω_0 t+\sinθ_n\sin nω_0 t) \\
&=A_0+\sum_{n=1}^{∞}A_n\cos(nω_0 t+θ_n)
\end{aligned}$
![关系图](https://gitee.com/WilenWu/images/raw/master/ComplexFunction/Fourier-series-physics.png)
(1) 上式表明，周期信号可以分解为一系列固定频率的简谐波之和，这些简谐波的(角) ==频率(frequency)== 为一个==基频(fundamental frequency)== $ω_0$的倍数。
==振幅(amplitude)== $A_n$ 反映了在信号 $f_T(t)$ 中频率为 $nω_0$的简谐波所占有的份额；
==相位(phase)==  $nω_0 t+θ_n$反映了在信号 $f_T(t)$ 中频率为 $nω_0$的简谐波沿时间轴移动的大小，==初相位(Initial Phase)== 为$θ_n$。
$A_0$表示周期信号在一个周期内的平均值，也叫 ==直流分量(DC component)==，$|A_0|$称为直流分量的振幅。
(2) 对于Fourier 级数的复指数形式，我们不难看出 $c_n$作为复数，其模和辐角恰好反应了第 n次谐波的振幅和初相位，$c_n$是离散频率 $nω_0$的函数，描述了各次谐波的振幅和初相位随离散频率变化的分布情况。称 $c_n$为 $f_T(t)$的离散==频谱(spectrum)==，$|c_n|$为离散==振幅谱(amplitude spectrum)==，$\arg c_n$为离散==相位谱(phase spectrum)==。


- **非周期函数的Fourier 变换**
上面研究的是周期函数，事实上对于一个非周期函数$f(t)$ 可以看成是一个周期为 T的函数$f_T(t)$ 当$T\to +∞$时转化而来。
由Fourier 级数式(F1)和式(F2)有 $\displaystyle f(t)=\lim\limits_{T\to +∞}\sum_{n=-∞}^{∞}[\dfrac1T\int^{T/2}_{-T/2}f_T(τ)e^{-inω_0 τ}\text{d}τ]e^{inω_0 t}$
记 $ω_n=nω_0$，间隔 $ω_0=Δω$，当n 取一切整数时， $ω_n$ 所对应的点便均匀地分布在整个数轴上，并由 $T=\dfrac{2\pi}{ω_0}=\dfrac{2\pi}{Δω}$ 得
$\displaystyle f(t)=\dfrac{1}{2\pi}\lim\limits_{Δω\to0}\sum_{n=-∞}^{∞}[\int^{π​/Δω}_{-π​/Δω}f_T(τ)e^{-iω_n τ}\text{d}τ]e^{iω_n t}Δω$
这是一个和式得极限，按照积分的定义，在一定条件下，上式可写成
$$
\displaystyle f(t)=\dfrac{1}{2\pi}\int_{-∞}^{+∞}[\int^{+∞}_{-∞}f(τ)e^{-iω τ}\text{d}τ]e^{iω t}\text{d}ω \tag{F3}
$$
 这个公式称为函数 $f(t)$的==Fourier 积分公式==。应该指出，上式只是由式(F1)的右端从形式上推出来的，是不严格的.。至于一个非周期函数 $f(t)$在什么条件下，可以用Fourier 积分公式表示，有下面的定理。
<kbd>Fourier 积分定理</kbd>：若 $f(t)$在 $\R$上满足：
(1) 在任一有限区间上满足狄利克雷(Dirichlet)条件；
(2) 在无限区间$(-∞,+∞)$上绝对可积 ( 即 $\int_{-∞}^{+∞}|f (t)| dt$ 收敛)
则有(F3)式成立
在间断点处，(F3)式左端为 $\frac{1}{2}[f(t^-)+f(t^+)]$
<kbd>Fourier 变换</kbd>：如果函数 $f(t)$满足Fourier 积分定理，由式(F3)，令 
$$
\displaystyle F(ω)=\int^{+∞}_{-∞}f(τ)e^{-iω τ}\text{d}τ \tag{F4}
$$
 则有 
$$
 \displaystyle f(t)=\dfrac{1}{2\pi}\int_{-∞}^{+∞}F(ω)e^{iω t}\text{d}ω \tag{F5}
$$
从上面两式可以看出，$f(t)$和 $F(ω)$通过确定的积分运算可以互相转换。  $F(ω)$称为 $f(t)$ ==Fourier 变换==(Fourier transform)，或==象函数==(image function)，记为$F(ω)=\mathcal{F}[f(t)]$ ；$f(t)$称为 $F(ω)$ ==Fourier 逆变换==(inverse Fourier transform)，或==象原函数==(original image function)，记为$f(t)=\mathcal{F}^{-1}[F(ω)]$ ；通常称$f(t)$与$F(ω)$构成一个==Fourier 变换对==(transform pair)，记作 $f(t)\lrarr F(ω)$

- **傅里叶正弦变换和余弦变换**：和傅里叶级数的情形类似，奇函数 $f(x)$ 的傅里叶变换是傅里叶正弦变换
$$
B(ω)=\int^{+∞}_{0}f(t)\sin ωt\text{d}t \\
f(t)=\dfrac{1}{2\pi}\int_{0}^{+∞}F(ω)\sin ωt\text{d}ω
$$
偶函数 $f(t)$ 的傅里叶变换是傅里叶余弦变换
$$
A(ω)=\int^{+∞}_{0}f(t)\cos ωt\text{d}t \\
f(t)=\dfrac{1}{2\pi}\int_{0}^{+∞}F(ω)\cos ωt\text{d}ω
$$


- **Fourier 变换的物理意义**
Fourier 积分公式表明非周期函数的频谱是连续取值的。
像函数$F(ω)$反映的是函数 $f(t)$中各频率分量的分布密度，它为复值函数，故可表示为 $F(ω)=|F(ω)|e^{i\arg F(ω)}$
称 $F(ω)$为 $f(t)$的==频谱(spectrum)==，$|F(ω)|$为==振幅谱(amplitude spectrum)==，$\arg F(ω)$为==相位谱(phase spectrum)==。
不难证明当$f(t)$为实函数时，$|F(ω)|$为偶函数，$\arg F(ω)$为奇函数。

**Fourier 变换的性质**
1. ==线性性质==：$\mathcal{F}[αf_1(t)+βf_2(t)]=α\mathcal{F}[f_1(t)]+β\mathcal{F}[f_2(t)]$
2. ==延迟性质==：设 $F(ω)=\mathcal{F}[f(t)]$，则
$\mathcal{F}[f(t-t_0)]=e^{-iω t_0}F(ω)$
3. ==位移性质==：设 $F(ω)=\mathcal{F}[f(t)]$，则
$\mathcal{F}[e^{-iω t_0}f(t)]=F(ω-ω_0)$
4. ==伸缩性质==(相似性质)：设 $F(ω)=\mathcal{F}[f(t)],a\neq 0$，则
$\mathcal{F}[f(at)]=\dfrac{1}{|a|}F(\dfrac{ω}{a})$
5. ==微分性质==：若 $\lim\limits_{|t|\to +\infty}f^{(k)}(t)=0(k=0,1,2,\cdots,n-1)$，则
$\mathcal{F}[f^{(n)}(t)]=(iω)^n\mathcal{F}[f(t)]$
6. ==积分性质==：设 $\displaystyle g(t)=\int_{-∞}^{t}f(t)dt$，若 $\lim\limits_{t\to +\infty}g(t)=0$则
$\mathcal{F}[g(t)]=\dfrac{1}{iω}\mathcal{F}[f(t)]$
7. ==帕赛瓦尔(Parseval)等式==：设 $f_1(t),f_2(t)$均为平方可积函数，即$\displaystyle \int_{-∞}^{+∞}|f_k(t)|^2dt<+\infty(k=1,2)$
设  $F_1(ω)=\mathcal{F}[f_1(t)],F_2(ω)=\mathcal{F}[f_2(t)]$，则
$\displaystyle \int_{-∞}^{+∞}f_1(t)\overline{f_2(t)}dt=\dfrac{1}{2\pi}\int_{-∞}^{+∞}F_1(ω)\overline{F_2(ω)}dω$
特别的当 $f_1(t)=f_2(t)=f(t),F(ω)=\mathcal{F}[f(t)]$时
$\displaystyle \int_{-∞}^{+∞}|f(t)|^2dt=\dfrac{1}{2\pi}\int_{-∞}^{+∞}|F(ω)|^2dω$
平方可积函数在物理上就是能量有限的信号，上式也叫==能量积分(energy integral)==，$|F(ω)|^2$ 也叫==能量谱密度(energy spectrum density)==。

8. <kbd>卷积定理</kbd>[^1]：设  $F_1(ω)=\mathcal{F}[f_1(t)],F_2(ω)=\mathcal{F}[f_2(t)]$，则有
$\mathcal F[f_1*f_2]=F_1(ω)\cdot F_2(ω)$
$\mathcal F^{-1}[F_1(ω)\cdot F_2(ω)]=f_1*f_2$
$\mathcal F[f_1\cdot f_2]=\frac{1}{2\pi}[F_1(ω)*F_2(ω)]$
$\mathcal F^{-1}[F_1(ω)*F_2(ω)]=2\pi f_1f_2$

[^1]: 卷积(Convolution)：设函数$f_1(t),f_2(t)$在$(-\infty,\infty)$上绝对可积，则积分$\displaystyle\int^{+\infty}_{-\infty}f_1(τ)f_2(t-τ)dτ$ 称为$f_1(t),f_2(t)$的卷积。记为
$$
\displaystyle f_1(t)*f_2(t)=\int^{+\infty}_{-\infty}f_1(τ)f_2(t-τ)dτ
$$
 [如何通俗易懂地解释卷积？——知乎](https://www.zhihu.com/question/22298352)
根据定义，卷积满足如下性质：
(1) 交换律：$f_1(t)*f_2(t)=f_2(t)*f_1(t)$
(2) 结合律：$f_1*[f_2*f_3]=[f_1*f_2]*f_3$
(3) 分配律：$f_1*[f_2+f_3]=f_1*f_2+f_1*f_3$

## 多重傅里叶积分
以三重傅里叶积分说明，首先将三维空间的非周期函数 $f(x,y,z)$ 按自变量 $x$ 展开为傅里叶积分，其傅里叶变换为 $F_1(k_1;y,z)$ ，其中 $y,z$ 作为参数出现。再将 $F_1(k_1;y,z)$ 按 $y$ 展开为傅里叶积分，得到 $F_2(k_1,k_2;z)$ ，最后将$F_2(k_1,k_2;z)$ 按 $z$ 展开为傅里叶积分。综合三次展开，得到 $f(x,y,z)$ 的三重傅里叶积分。
$$
f(\mathbf x)=\frac{1}{(2\pi)^3}\iiint\limits_{-\infty}^{+\infty}F(k_1,k_2,k_3)e^{\mathrm i(k_1x+k_2y+k_3z)}dk_1dk_2dk_3
$$
$$
F(k_1,k_2,k_3)=\iiint\limits_{-\infty}^{+\infty}f(x,y,z)e^{-\mathrm i(k_1x+k_2y+k_3z)}dxdydz
$$
引入矢量 $\mathbf r=(x,y,z)\in\R^n,\mathbf k=(k_1,k_2,k_3)$，可写为较简介的形式
$$
F(\mathbf k)=\iiint\limits_{-\infty}^{+\infty}f(\mathbf r)e^{-\mathrm i\mathbf{k\cdot x}}d\mathbf x
$$
则有
$$
f(\mathbf r)=\frac{1}{(2\pi)^3}\iiint\limits_{-\infty}^{+\infty}F(\mathbf k)e^{\mathrm i\mathbf{k\cdot x}}d\mathbf k
$$
其中 $F(\mathbf k)$ 称为 $f(\mathbf r)$ 的 ==多重傅里叶变换==，记为 $F(\mathbf k)=\mathcal{F}[f(\mathbf r)]$；$f(\mathbf r)$ 称为 $F(\mathbf k)$ 的 ==多重傅里叶逆变换==，记为 $f(\mathbf r)=\mathcal{F}^{-1}[F(\mathbf k)]$

## δ 函数
在物理学中，常有集中于一点或一瞬时的量，如脉冲力、脉冲电压、点电荷、质点的质量。只有引入一个特殊函数来表示它们的**分布密度**，才有可能把这种集中的量与连续分布的量来统一处理。

- **单位脉冲函数**(Unit Impulse Function)
<引例>：假设在原来电流为零的电路中，在 $t=0$ 时瞬时进入一电量为 $q_0$的脉冲。现在确定电流强度分布 $i(t)=\cfrac{\mathrm dq}{\mathrm dt}$，分析可知 $i(t)=\begin{cases}
0&(t\neq 0) \\
∞​&(t=0)
\end{cases}$
同时需要引入积分值表示电量大小 $\displaystyle\int_{-∞}^{+∞}i(t)dt=q_0$
为此我们引入==单位脉冲函数==，又称为==Dirac函数或者δ函数==。

   <kbd>定义</kbd>：单位脉冲函数 $δ(t)$ 满足
(1) 当 $t\neq 0$ 时，$δ(t)=0$
(2) $\displaystyle\int_{-∞}^{+∞}δ(t)dt=1$
由此，引例可表示为 $i(t)=q_0δ(t)$
![delta函数](https://gitee.com/WilenWu/images/raw/master/ComplexFunction/delta-fun.png)
**注意**：
(1) 单位脉冲函数 $δ(t)$ 并不是经典意义下的函数，因此通常称其为广义函数(或者奇异函数)。
(2) 它不能用常规意义下的值的对应关系来理解和使用，而总是通过它的定义和性质来使用它。
(3) 单位脉冲函数 $δ(t)$ 有多种定义方式，前面所给出的定义方式是由Dirac(狄拉克)给出的。

- **单位脉冲函数其他定义方式**
构造一个在 $ε$ 时间内激发的矩形脉冲 $δ_ε(t)$，定义为
$δ_ε(t)=\begin{cases}
0&(t< 0) \\
1/ε​&(0⩽t⩽ε) \\
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
![delat函数](https://gitee.com/WilenWu/images/raw/master/ComplexFunction/delta-fun2.png)
   <kbd>筛选性质(sifting property)</kbd>： 设函数 $f(t)$ 是定义在 $\R$上的有界函数，且在 $t = 0$ 处连续，则有
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
<img src="https://gitee.com/WilenWu/images/raw/master/ComplexFunction/delta-series-demo.png" style="zoom: 8%;" /> <img src="https://gitee.com/WilenWu/images/raw/master/ComplexFunction/delta-series-demo2.png" style="zoom:8%;" />
<img src="https://gitee.com/WilenWu/images/raw/master/ComplexFunction/delta-series-demo3.png" style="zoom:8%;" /> <img src="https://gitee.com/WilenWu/images/raw/master/ComplexFunction/delta-series-demo4.png" style="zoom:8%;" />

- **δ函数的基本性质**：（这些性质的严格证明可参阅广义函数）
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
0 & t<0 \\ 1 &t>​0 \end{cases}
$$
![单位阶跃函数](https://gitee.com/WilenWu/images/raw/master/ComplexFunction/unit-step-fun.png)
按广义函数理论，定义为
$$
\displaystyle\int_{-∞}^{+∞}u(t)f(t)dt=\int_{0}^{+∞}f(t)dt
$$
单位阶跃函数的积分为：
$$
\int_{-\infty}^{t}u(\tau)\mathrm d\tau=tu(t)
$$

- **δ函数的Fourier 变换**：
(1) 根据 $δ$ 函数筛选性质可得
$\displaystyle F(ω)=\mathcal{F}[δ(t)]=\int^{+∞}_{-∞}δ(t)e^{-iω t}\text{d}t=e^{-iω t}|_{t=0}=1$ 
$\displaystyleδ(t)=\mathcal{F}^{-1}[1]=\dfrac{1}{2\pi}\int_{-∞}^{+∞}e^{iω t}\text{d}ω$
或写为
$\displaystyleδ(t)=\dfrac{1}{2\pi}\int_{-∞}^{+∞}\cosω t\text{d}ω
=\dfrac{1}{\pi}\int_{0}^{+∞}\cosω t\text{d}ω$
由此可见，单位冲激函数包含所有频率成份，且它们具有相等的幅度，称此为均匀频谱或白色频谱。
我们可以得到 ：
$$
\begin{aligned}
δ(t) & \lrarr 1 \\
δ(t-t_0) & \lrarr e^{-iω t_0} \\
1 & \lrarr 2\pi δ(ω) \\
e^{-iω_0 t} & \lrarr 2\pi δ (ω − ω_0 )
\end{aligned}
$$
(2) 有许多重要的函数不满足Fourier 积分定理条件（绝对可积），例如常数、符号函数、单位阶跃函数、正弦函数和余弦函数等，但它们的广义Fourier 变换[^gf]也是存在的，利用单位脉冲函数及其Fourier 变换可以求出它们的Fourier 变换。

[^gf]: 在δ函数的Fourier变换中，其广义积分是根据δ函数的性质直接给出的，而不是按通常的积分方式得到的，称这种方式的Fourier 变换为==广义Fourier 变换==。

- **周期函数的Fourier 变换**
<kbd>定理</kbd>：设 $f(t)$ 以T 为周期，在 $[0,T]$ 上满足 Dirichlet 条件，则 $f(t)$的Fourier 变换为：
$$
\displaystyle F(ω)=2\pi\sum_{n=-∞}^{+∞}F(nω_0)δ (ω − nω_0)
$$
 其中 $ω_0=2\pi/T,F(nω_0)$是 $f(t)$ 的离散频谱。

- **多维 $δ$ 函数**：例如位于三维空间的坐标原点质量为 $m$ 的质点，其密度函数可表示为 $mδ(\mathbf r)$。	在三维空间中的 $δ$ 函数定义如下：
$$
δ(\mathbf r)=
\begin{cases} 
0 &(\mathbf r\neq0) \\
\infty &(\mathbf r=0)
\end{cases}  \\
\iiint\limits_{-\infty}^{+\infty} δ(\mathbf r)\mathrm d\mathbf r=1
$$
三维 $δ$ 函数可表示为三个一维 $δ$ 函数乘积表示，在直角坐标系中
$$
δ(\mathbf r)=δ(x)δ(y)δ(z)
$$
   三维空间点 $\mathbf r_0=(x_0,y_0,z_0)$ 处密度分布函数就是
$$
δ(\mathbf{r-r_0})=δ(x-x_0)δ(y-y_0)δ(z-z_0)
$$
换算到柱坐标系 $\mathbf r_0=(r_0,θ_0,z_0)$
$$
δ(\mathbf{r-r_0})=\frac{1}{r_0}δ(r-r_0)δ(θ-θ_0)δ(z-z_0)
$$
换算到球坐标系 $\mathbf r_0=(r_0,θ_0,ϕ_0)$
$$
δ(\mathbf{r-r_0})=\frac{1}{r_0^2\sinθ_0}δ(r-r_0)δ(θ-θ_0)δ(ϕ-ϕ_0)
$$
多维 $δ$ 函数主要性质：
$$
\iiint\limits_{-\infty}^{+\infty} f(\mathbf r)δ(\mathbf{r-r_0})\mathrm d\mathbf r=f(\mathbf r_0) \\
\iiint\limits_{-\infty}^{+\infty} f(\mathbf r)[\nablaδ(\mathbf{r-r_0})]\mathrm d\mathbf r=-\nabla f(\mathbf r)|_{\mathbf{r=r_0}}
$$
位矢的微分：
$$
\Delta \frac{1}{r}=-4\piδ(\mathbf r)
$$
其中 $r=\sqrt{x^2+y^2+z^2}$ 



## Fourier 变换的应用
1. 求==矩形脉冲函数(rectangular pulse function)== $f(t)=\begin{cases}1&|t|<a \\ 0 &|t|>a \end{cases}$ 的Fourier 变换及其Fourier 积分表达式。
![矩形脉冲](https://gitee.com/WilenWu/images/raw/master/ComplexFunction/rectangular-pulse-fun.png)
(1) Fourier 变换为
$\begin{aligned}
\displaystyle F(ω) &=\int^{+∞}_{-∞}f(t)e^{-iω t}\text{d}t=\int^{a}_{-a}e^{-iω t}\text{d}t \\
&=\int^{a}_{-a}\cos(ωt)\text{d}t-\text{i}\int^{a}_{-a}\sin(ωt)\text{d}t \\
&=2\int^{a}_{0}\cos(ωt)\text{d}t \\
&=\frac{2\sin(aω)}{ω} =2a\frac{\sin(aω)}{aω}
\end{aligned}$ 
(2) 振幅谱 $\displaystyle |F(ω)| =2a\left|\frac{\sin(aω)}{aω}\right|$
相位谱 $\arg F(ω)=\begin{cases}
0 &  \frac{2n\pi}{a}⩽|ω|⩽ \frac{2n\pi}{a}  \\
\pi &\text{others}
\end{cases}$
![频谱](https://gitee.com/WilenWu/images/raw/master/ComplexFunction/spectrum.png)
(3) Fourier 积分表达式为
$\begin{aligned}
\displaystyle f(t) &=\mathcal{F}^{-1}[F(ω)] \\
&=\dfrac{1}{2\pi}\int_{-∞}^{+∞}F(ω)e^{iω t}\text{d}ω=\dfrac{1}{2\pi}\int_{-∞}^{+∞}\frac{2\sin(aω)}{ω}e^{iω t}\text{d}ω \\
&=\dfrac{1}{\pi}\int_{-∞}^{+∞}\frac{\sin(aω)}{ω}\cosωt\text{d}ω \\
&=\begin{cases}
1 &  |t|<a  \\
\frac{1}{2} &  |t|=a  \\
0 &  |t|>a  \\
\end{cases}
\end{aligned}$
在上式中令 $t = 0$，可得重要公式：
$\displaystyle\boxed{\int_{-∞}^{+∞}\frac{\sin(ax)}{x}\text{d}x=
\begin{cases}
-\pi &a<0 \\ 
0 &a=0 \\
\pi  &a>0
\end{cases}}$
特别的 $\displaystyle\int_{0}^{+∞}\frac{\sin x}{x}\text{d}x=\frac{\pi}{2}$


2. 求==指数衰减函数(exponential decay function)== $f(t)=\begin{cases}
0 & t<0 \\ e^{-a t} &t⩾0 \end{cases}\quad(a>0)$ 的Fourier 变换及Fourier 积分表达式。
(1) Fourier 变换为
$\begin{aligned}
\displaystyle F(ω) &=\int^{+∞}_{-∞}f(t)e^{-iω t}\text{d}t=\int^{+∞}_{0}e^{-a t}e^{-iω t}\text{d}t \\
&=\frac{1}{-(a+iω)}e^{-(a+iω)t}\Big|^{t\to+∞}_{t=0} \\
&=\frac{1}{a+iω}=\frac{a-iω}{a^2+ω^2}
\end{aligned}$
(2) 振幅谱 $\displaystyle |F(ω)| =\frac{1}{\sqrt{a^2+ω^2}}$
相位谱 $\arg F(ω)=-\arctan\dfrac{ω}{a}$

![频谱图](https://gitee.com/WilenWu/images/raw/master/ComplexFunction/spectrum2.png)
(3) Fourier 积分表达式为
$\begin{aligned}
\displaystyle f(t) &=\mathcal{F}^{-1}[F(ω)] \\
&=\dfrac{1}{2\pi}\int_{-∞}^{+∞}F(ω)e^{iω t}\text{d}ω=\dfrac{1}{2\pi}\int_{-∞}^{+∞}\frac{β-iω}{β^2+ω^2}e^{iω t}\text{d}ω \\
&=\dfrac{1}{2\pi}\int_{-∞}^{+∞}\frac{1}{β^2+ω^2}(β-iω)(\cosωt+i\sinωt)\text{d}ω
\end{aligned}$
利用奇偶函数的积分性质，可得
$\displaystyle f(t)=\dfrac{1}{\pi}\int_{0}^{+∞}\frac{β\cosωt+ω\sinωt}{β^2+ω^2}\text{d}ω$
由此顺便得到一个含参变量广义积分的结果
$\displaystyle \boxed{\int_{0}^{+∞}\frac{β\cosωt+ω\sinωt}{β^2+ω^2}\text{d}ω=
\begin{cases}
0 &t<0\\ 
\dfrac{\pi}{2} &t=0 \\
\pi e^{-βt} &t>0
\end{cases}}$

3. 求单位阶跃函数[^unit] $u(t)=\begin{cases}
0 & (t<0) \\ 1 & (t>​0) \end{cases}$ 的Fourier 变换及其积分表达式。
(1) 现将 $u(t)$看作是指数衰减函数$f(t;β)=\begin{cases}
0 & t<0 \\ e^{-β t} &t>0 \end{cases}$ 在 $β\to0^+$时的极限，即 $u(t)=\lim\limits_{β\to0^+}f(t;β)$
$\begin{aligned}
\displaystyle F(ω) & =\lim\limits_{β\to0^+}\mathcal{F}[f(t;β)]\\
&=\lim\limits_{β\to0^+}\frac{1}{β+iω} =\lim\limits_{β\to0^+}(\frac{β}{β^2+ω^2}-i\frac{ω}{β^2+ω^2})\\
&=\pi δ(ω)+\frac{1}{iω}
\end{aligned}$
又因 $\displaystyle\lim\limits_{β\to0^+}\int_{-∞}^{+∞}\frac{β}{β^2+ω^2}dω=\lim\limits_{β\to0^+}[\arctan \frac{ω}{β}]\Big|^{-∞}_{+∞}=\pi$
所以$\lim\limits_{β\to0^+}\dfrac{β}{β^2+ω^2}=πδ(ω)$
$\mathcal{F}[u(t)]=\pi δ(ω)+\dfrac{1}{iω}$

   (2) Fourier 积分表达式
$\begin{aligned}
\displaystyle u(t) &=\mathcal{F}^{-1}[F(ω)] \\
&=\dfrac{1}{2\pi}\int_{-∞}^{+∞}F(ω)e^{iω t}\text{d}ω=\dfrac{1}{2\pi}\int_{-∞}^{+∞}[\pi δ(ω)+\dfrac{1}{iω}]e^{iω t}\text{d}ω \\
&=\dfrac{1}{2}\int_{-∞}^{+∞}δ(ω)e^{iω t}\text{d}ω+\dfrac{1}{2\pi}\int_{-∞}^{+∞}\dfrac{1}{iω}e^{iω t}\text{d}ω \\
&=\dfrac{1}{2}+\dfrac{1}{\pi}\int_{0}^{+∞}\dfrac{\sinω t}{ω}\text{d}ω
\end{aligned}$
在上式中令 t=1，可得狄利克雷积分 $\displaystyle\int_{0}^{+∞}\dfrac{\sin t}{t}\text{d}t=\dfrac{\pi}{2}$

4. 求余弦函数 $f (t) = \cosω_0t$ 的Fourier 积分
由欧拉公式 $\cosω_0t=\frac{1}{2}(e^{iω_0t}+e^{-iω_0t})$ 有
$\begin{aligned}
\displaystyle 
\mathcal{F}[\cosω_0t] &=\int^{+∞}_{-∞}\cosω_0te^{-iω t}\text{d}t\\
&=\int^{+∞}_{0}\frac{1}{2}(e^{iω_0t}+e^{-iω_0t})e^{-iω t}\text{d}t \\
&=\frac{1}{2}[\int^{+∞}_{0}e^{-i(ω-ω_0)t}\text{d}t+\int^{+∞}_{0}e^{-i(ω+ω_0)t}\text{d}t] \\
&=\pi [δ(ω-ω_0)+δ(ω+ω_0)]
\end{aligned}$
同理可证 $\mathcal{F}[\sinω_0t]=i\pi [δ(ω+ω_0)-δ(ω-ω_0)]$


# Laplace 变换

## Laplace 变换
- **Fourier 变换的局限性**
当函数满足Dirichlet条件，且在 $(-∞,+∞)$ 上绝对可积时，则可以进行古典Fourier 变换。
引入广义函数和广义Fourier 变换是扩大Fourier 变换使用范围的一种方法，却要求有一系列更深刻的数学理论支持。对于以指数级增长的函数，如 $e^{at} (a > 0)$ 等，广义Fourier 变换仍无能为力。
如何对Fourier 变换进行改造？
(1) 由于单位阶跃函数 $u(t)\equiv 0(t<0)$，因此 $f(t)u(t)$ 可使积分区间从 $(−∞,+∞)$ 变成 $[0,+∞)$；
(2) 另外，函数 $e^{-βt} (β > 0)$ 具有衰减性质，对于许多非绝对可积的函数 $f(t)$，总可选择适当大的 β，使 $f(t)u(t)e^{-βt}$ 满足绝对可积的条件。
通过上述处理，就有希望使得函数 $f(t)u(t)e^{-βt}$ 满足Fourier 变换的条件，从而可以进行Fourier 变换。
$\displaystyle \mathcal{F}[f(t)u(t)e^{-βt}] =\int^{+∞}_{-∞}f(t)u(t)e^{-βt}e^{-iω t}\text{d}t=\int^{+∞}_{0}f(t)e^{-(β+iω) t}\text{d}t$
令 $s=β+iω$ 可得 $\displaystyle\mathcal{F}[f(t)u(t)e^{-βt}]=\int^{+∞}_{0}f(t)e^{-s t}\text{d}t$

   > [用幂级数推导出 “Laplace 变换”](https://baijiahao.baidu.com/s?id=1626062298972618369&wfr=spider&for=pc)

- **Laplace变换**
<kbd>Laplace变换</kbd>：设函数$f(t)$ 在$t\geqslant 0$时有定义，且积分$\displaystyle\int_{0}^{+∞}f(t)e^{-st}dt$在复数 s 的某一个区域内收敛，则此积分所确定的函数
$$
\displaystyle F(s)=\int^{+\infty}_{0}f(t)e^{-st}\text{d}t
$$
称为函数$f(t)$的Laplace 变换，记为$F(s)=\mathcal L[f(t)]$，函数 $F(s)$ 也可称为 $f(t)$的象函数。$f(t)=\mathcal L^{-1}[F(s)]$称为Laplace 逆变换。
在Laplace 变换中，只要求$f(t)$在 $[0,+∞)$ 内有定义即可。为了研究方便，以后总假定在$(−∞,0)$ 内，$f(t)≡0$

   <kbd>Laplace变换存在定理</kbd>：设函数 $f(t)$满足
(1) 在$t⩾0$的任何有限区间分段连续；
(2) 当 $t\to +∞$时，$f(t)$的增长速度不超过某指数函数，即 $\exists M>0,C⩾0$，使得 $|f(t)|⩽Me^{Ct}(t⩾0)$ 成立。
则$f(t)$的Laplace 变换$F(s)$在半平面 $\text{Re }(s)>C$上一定存在，且是解析的。

   **周期函数的Laplace变换**：设 $f(t)$是 $[0, +\infty)$ 内以T 为周期的函数，且逐段光滑，则
$$
   \displaystyle\mathcal L[f(t)]=\frac{1}{1-e^{-sT}}\int^{T}_{0}f(t)e^{-st}\text{d}t
$$

**Laplace变换的性质**
1. ==线性性质==：设$F_1(s)=\mathcal L[f_1(t)],F_2(s)=\mathcal L[f_2(t)]$
$\mathcal L[\alpha f_1(t)+\beta f_2(t)]=\alpha F_1(s)+\beta F_2(s)$
​ $\mathcal L^{-1}[\alpha F_1(s)+\beta F_2(s)]=\alpha f_1(t)+\beta f_2(t)$

2. ==位移性质==：$\mathcal L  [e^{s_0t}f(t)]=F(s-s_0)$

3. ==微分性质==：设$F(s)=\mathcal L[f(t)]$
$\mathcal L[f'(t)]=sF(s)-f(0)$
​ $\displaystyle \mathcal L[f^{(n)}(t)]=s^nF(s)-\sum_{k=1}^{n} s^{n-k}f^{(k-1)}(0)$
​ $F'(s)=-\mathcal L[tf(t)]$
​ $F^{(n)}(s)=(-1)^n\mathcal L[t^nf(t)]$

4. ==积分性质==：设$F(s)=\mathcal L[f(t)]$
$\displaystyle\mathcal L[\int^t_0f(t)dt]=\frac 1sF(s)$
$\displaystyle\mathcal L[\underbrace{\int^t_0dt\int^t_0dt\cdots\int^t_0}_{\text{n times}}f(t)dt]=\frac{1}{s^n}F(s)$
$\displaystyle\mathcal L[\frac{f(t)}{t}]=\int^{\infty}_sF(s)ds$
$\displaystyle\mathcal L[\frac{f(t)}{t^n}]= \mathcal L[\underbrace{\int^∞_sdt\int^∞_sdt\cdots\int^∞_s}_{\text{n times}}F(s)ds]$

5. ==延迟性质==：$\text{if } t>0,f(t)=0, \text{then }\forall t_0>0$
 $\mathcal L[f(t-t_0)]=e^{-st_0}F(s)$
    ​ $\mathcal L^{-1}[e^{-st_0}F(s)]=f(t-t_0)u(t-t_0)$

6. <kbd>卷积定理</kbd>[^1]：设  $F_1(s)=\mathcal{L}[f_1(t)],F_2(s)=\mathcal{L}[f_2(t)]$，则有
$\mathcal L[f_1*f_2]=F_1(s)\cdot F_2(s) \\
\mathcal L^{-1}[F_1(s)\cdot F_2(s)]=f_1*f_2$

## Laplace 逆变换

- **反演积分公式**(inverse integral formula)：由于 $f(t)$ 的Laplace 变换 $F(s)=F(β+iω)$就是 $f(t)u(t)e^{-βt}$ 的Fourier 变换，即
$\displaystyle\mathcal L[f(t)]=\mathcal F[f(t)u(t)e^{-βt}]=\int_{−∞}^{+∞} f(t)u(t)e^{-βt}e^{-iωt}dt$
因此，在 $f(t)(t>0)$的连续点处有
$\displaystyle f(t)u(t)e^{-βt}=\dfrac{1}{2\pi}\int_{-∞}^{+∞}F(β+iω)e^{iω t}\text{d}ω$
等式两边同乘 $e^{βt}$，并令 $s=β+iω$ 则有
$\displaystyle f(t)u(t)=\dfrac{1}{2\pi i}\int_{β-iω}^{β+iω}F(s)e^{st}\text{d}s$
因此 
$$
\displaystyle f(t)=\dfrac{1}{2\pi i}\int_{β-iω}^{β+iω}F(s)e^{st}\text{d}s \quad(t>0)
$$

- **利用留数计算反演积分**
<kbd>定理</kbd>设 $F(s)$ 在复平面内只有有限个孤立奇点 $s_1,s_2,\cdots,s_n$ ，实数 β使这些奇点全在半平面 $\text{Re}(s)<β$ 内，且 $\lim\limits_{s\to∞}F(s)=0$ ，则有 
$$
\displaystyle f(t)=\sum_{k=1}^n\text{Res}[F(s)e^{st},s_k]\quad(t>0)
$$
![Laplace 逆变换](https://gitee.com/WilenWu/images/raw/master/ComplexFunction/inverse-Laplace-transform.png)
证明：作半圆将所有奇点包围，设 $C=C_R+L$，由于 $e^{st}$在全平面解析，所以$F(s)e^{st}$的奇点就是 $F(s)$的奇点，由留数定理可得
$\displaystyle 2\pi i\sum_{k=1}^n\text{Res}[F(s)e^{st},s_k]=\oint_{C}F(s)e^{st}ds=\int_{β-iR}^{β+iR}F(s)e^{st}ds+\int_{C_R}F(s)e^{st}ds$
由若尔当引理，当 t>0 时，有 $\displaystyle\lim\limits_{R\to+\infty}\int_{C_R}F(s)e^{st}ds=0$
再根据反演积分公式可得定理公式。
实际中经常遇到有理函数类 $F(s)=\dfrac{A(s)}{B(s)}$，其中 $A(s),B(s)$是不可约的多项式，当分子$A(s)$ 的次数小于分母$B(s)$的次数时，满足定理对 $F(s)$ 的要求，可用留数计算Laplace 逆变换。

## Laplace 变换的应用

**常用函数的Laplace变换**
1. 求指数函数 $f(t) = e^{at} ( a⩾0)$的Laplace 变换
$\displaystyle \mathcal L[e^{at}]=\int^{+\infty}_{0}e^{at}e^{-st}\text{d}t=\int^{+\infty}_{0}e^{-(s-a)t}\text{d}t$
 当 $\text{Re }s>a$ 时，设 $s=β+iω$ ，此时
 $\lim\limits_{t\to+\infty}e^{-(s-a)t}=\lim\limits_{t\to+\infty}e^{-(β-a)t}e^{-iω}=0 (β>0)$ 
所以有 $\displaystyle \mathcal L[e^{at}]=\frac{1}{s-a}\quad(\text{Re }s>a)$

2. 求函数 $f(t) = 1$ 的Laplace 变换
$\displaystyle \mathcal L[1]=\int^{+\infty}_{0}e^{-st}\text{d}t=\frac{1}{s}\quad(\text{Re }s>0)$

3.  单位阶跃函数[^unit] $u(t)=\begin{cases}
0 & t<0 \\ 1 &t>​0 \end{cases}$ 的Laplace 变换
$\displaystyle \mathcal L[u(t)]=\frac{1}{s}\quad(\text{Re }s>0)$

4. 正弦函数 $\displaystyle \mathcal L[ \sinωt]=\frac{ω}{s^2+ω^2}\quad(\text{Re }s>0)$ 
余弦函数 $\displaystyle \mathcal L[ \cosωt]=\frac{s}{s^2+ω^2}\quad(\text{Re }s>0)$ 

5. 幂函数 $f(t)=t^m(m\in\Z^+)$ 的Laplace 变换
$\displaystyle\begin{aligned} 
 \mathcal L[t^m] &=\int^{+\infty}_{0}t^me^{-st}\text{d}t=-\frac{1}{s}\int^{+\infty}_{0}t^m\text{d}e^{-st} \\ 
& =-\frac{1}{s}t^me^{-st}\Big|_{0}^{+\infty}+\frac{m}{s}\int^{+\infty}_{0}t^{m-1}e^{-st}\text{d}t \\
&=\frac{m}{s}\mathcal L[t^{m-1}]\quad(\text{Re }s>0)
\end{aligned}$
又由 $\displaystyle\mathcal L[1]=1/s$，故递推可得
$\displaystyle\mathcal L[t^m]=\frac{m!}{s^{m+1}}\quad(\text{Re }s>0)$

6. 求 δ 函数的Laplace 变换。
狄利克雷函数 $δ_τ(t)=\begin{cases}
 \frac{1}{τ} &0⩽ t<τ \\
 0  &\text{others}
\end{cases}$ 的Laplace 变换为
$\displaystyle \mathcal L[δ_τ(t)]=\int^{τ}_{0}\frac{1}{τ}e^{-st}\text{d}t=\frac{1}{τs}(1-e^{-τs})$
$\displaystyle \mathcal L[δ(t)]=\lim\limits_{τ\to0}\mathcal L[δ_τ(t)]=\lim\limits_{τ\to0}\frac{1}{τs}(1-e^{-τs})$
用洛必达法则计算此极限 $\displaystyle\lim\limits_{τ\to0}\frac{1}{τs}(1-e^{-τs})=\lim\limits_{τ\to0}\frac{se^{-τs}}{s}=1$
所以 $\mathcal L[δ(t)]=1$

**微分方程的Laplace变换解法**：主要借助于Laplace变换的微分性质
​ $\displaystyle \mathcal L[f^{(n)}(t)]=s^nF(s)-\sum_{k=1}^{n} s^{n-k}f^{(k-1)}(0)$
(1) 将微分方程(组)化为象函数的代数方程(组)；
(2) 求解代数方程得到象函数；
(3) 求Laplace 逆变换得到微分方程(组)的解。
![](https://gitee.com/WilenWu/images/raw/master/ComplexFunction/solving-DE.png)

1. 求解微分方程 $y''+ω^2y=0$ 满足初始条件 $y(0)=0,y'(0)=ω$ 
(1) 令$Y(s)=\mathcal L[y(t)]$ ，对方程两边取Laplace 变换
$s^2Y(s)-sy(0)-y'(0)+ω^2Y(s)=0$，带入初始条件可得
$s^2Y(s)-ω+ω^2Y(s)=0$
(2) 求解此方程，得 $Y(s)=\dfrac{ω}{s^2+ω^2}$
(3) 求Laplace 逆变换，得 $y=\mathcal L^{-1}[Y(s)]=\sin ωt$

2. 求解微分方程初值问题 $\begin{cases}
ax'(t)+bx(t)=f(t),&t>0 \\
x(0)=c \\
\end{cases}$
令$X(s)=\mathcal L[x(t)],F(s)=\mathcal L[f(t)]$ ，对方程两边取Laplace 变换，带入初始条件可得
$a(sX(s)-c)+bX(s)=F(s)$
解得 $X(s)=\cfrac{F(s)+ac}{as+b}=c\cfrac{1}{s+b/a}+\cfrac{1}{a}\cfrac{1}{s+b/a}F(s)$
由于 $\mathcal L^{-1}[\cfrac{1}{s+b/a}]=e^{-\frac{b}{a}t}$，故上式Laplace 逆变换为
$\displaystyle x(t)=ce^{-\frac{b}{a}t}+\frac{1}{a}\int_{0}^{t}f(τ)e^{-\frac{b}{a}(t-τ)}\text{d}τ$

3. 求微分方程组：$\begin{cases} x'+y+z'=1\\ x+y'+z=0\\ y+4z'=0 \end{cases}$ 满足初始条件 $x(0)=0,y(0)=0,z(0)=0$ 
令$\mathcal L[x(t)]=X(s),\mathcal L[y(t)]=Y(s),\mathcal L[z(t)]=Z(s)$
对方程组两边取Laplace 变换，并带入初始条件可得
$\begin{cases} 
sX(s)+Y(s)+sZ(s)=\dfrac 1s\\ 
X(s)+sY(s)+Z(s)=0\\ 
Y(s)+4sZ(s)=0
\end{cases}$
解代数方程组得：
$\begin{cases}
X(s)=\dfrac{4s^2-1}{4s^2(s^2-1)} \\
Y(s)=\dfrac {-1}{s(s^2-1)} \\
Z(s)=\dfrac {1}{4s^2(s^2-1)} \\
\end{cases}$
对每一像函数取Laplace 逆变换可得：
$\begin{cases}
x(t)=\mathcal L^{-1}[X(s)]=\dfrac 14\mathcal L^{-1}[\dfrac {3}{s^2-1}+\dfrac{1}{s^2}]=\dfrac 14(3\sinh t+t) \\
y(t)=\mathcal L^{-1}[Y(s)]=\mathcal L^{-1}[\dfrac 1s-\dfrac {s}{s^2-1}]=1-\cosh t \\
z(t)=\mathcal L^{-1}[Z(s)]=\dfrac 14\mathcal L^{-1}[\dfrac {1}{s^2-1}-\dfrac {1}{s^2}]=\dfrac 14(\sinh t-t)
\end{cases}$

4. 求解积分方程：$\displaystyle f(t)=at-\int_{0}^{t}\sin(x-t)f(x)dt\quad(a\neq0)$
原方程化为$\displaystyle f(t)=at+f(t)*\sin t$
令$F(s)=\mathcal L[f(t)]$ ，对方程两边取Laplace 变换
$F(s)=\dfrac{a}{s^2}+\dfrac{1}{s^2+1}F(s)$
解得 $F(s)=a(\dfrac{a}{s^2}+\dfrac{a}{s^4})$
求Laplace 逆变换 $f(t)=a(t+\dfrac{t^3}{6})$

**物理学问题**
1. 设质量为m 的物体静止在原点，在 t = 0 时受到 x 方向的冲击力 $F_0δ(t)$的作用，求物体的运动方程。
设物体的运动方程为 $x = x(t)$ ，根据Newton 定律
$mx''(t)=F_0δ(t),x(0)=x'(0)=0$
令$X(s)=\mathcal L[x(t)]$ ，对方程两边取Laplace 变换，并带入初始条件得
$ms^2X(s)=F_0\implies X(s)=\frac{F_0}{ms^2}$
求Laplace 逆变换即得物体的运动方程为：$x(t)=\frac{F_0}{m}t$


2. 质量为m的物体挂在弹簧系数为k 的弹簧一端(如图)，作用在物体上的外力为 $f(t)$。若物体自静止平衡位置 x = 0 处开始运动，求该物体的运动规律 $x(t)$ 。
![](https://gitee.com/WilenWu/images/raw/master/ComplexFunction/Laplace-transform-demo.png)
(1) 根据 Newton 定律及 Hooke 定律，物体的运动规律 $x(t)$ 满足如下的微分方程：
$mx''(t)+kx(t)=f(t);\quad x(0)=x'(0)$
(2) 令$X(s)=\mathcal L[x(t)],F(s)=\mathcal L[f(t)]$ ，对方程两边取Laplace 变换，带入初始条件可得
$ms^2X(s)+kX(s)=F(s)$
令 $ω_0^2=k/m$，有 $X(s)=\dfrac{1}{mω_0}\cdot\dfrac{ω_0}{s^2+ω_0^2}\cdot F(s)$
(3) 利用卷积定理，求Laplace 逆变换得：
$x(t)=\mathcal L^{-1}[X(s)]=\dfrac{1}{mω_0}[\sinω_0t*f(t)]$
当 $f(t)$具体给出时，即可以求得运动规律 $x(t)$ 
设物体在 t = 0时受到的外力为 $f(t ) = Aδ(t)$ 
此时，物体的运动规律为：
$x(t)=\dfrac{1}{mω_0}[\sinω_0t*f(t)]=\dfrac{A}{mω_0}\sinω_0t$

# 附录
## 积分变换表

|$f(t)$|Fourier Transform|Laplace Transform|
:---|:---|:---|
Conditions|若 $f(t)$在 $\R$上满足：<br>(1) 在任一有限区间上满足Dirichlet条件；<br>(2) 在无限区间 $(-∞,+∞)$上绝对可积 ，即 $\displaystyle\int_{-∞}^{+∞}\mid f(t)\mid \text{d}t$ 收敛<br>Dirichlet 条件：<br>(1)连续或只有有限个第一类间断点；<br>(2)只有有限个极值点|若 $f(t)$满足<br>(1) 在$t⩾0$的任何有限区间分段连续；<br>(2) 当 $t\to +∞$时，$f(t)$的增长速度不超过某指数函数，即<br> $\exists M>0,C⩾0$，使得 $\mid f(t)\mid ⩽Me^{Ct}(t⩾0)$ 成立。<br>则$f(t)$的Laplace 变换$F(s)$在半平面 $\text{Re }(s)>C$上一定存在，且是解析的。
Kernel Function|$e^{-\text{i}ωt}$|$e^{-st}$|
Interval|$(-∞,+∞)$|$(0,+∞)$|
Symbols|$F(ω)=\mathcal{F}[f(t)]$<br>$f(t)=\mathcal{F}^{-1}[F(ω)]$|$F(s)=\mathcal{L}[f(t)]$<br>$f(t)=\mathcal{L}^{-1}[F(s)]$
Transform<br>(image)|$\displaystyle F(ω)=\int^{+∞}_{-∞}f(τ)e^{-\text{i}ωt}\text{d}t$|$\displaystyle F(s)=\int^{+\infty}_{0}f(t)e^{-st}\text{d}t$
Inverse Transform<br>(original image)|$\displaystyle f(t)=\dfrac{1}{2\pi}\int_{-∞}^{+∞}F(ω)e^{\text{i}ω t}\text{d}ω$|$\displaystyle f(t)=\dfrac{1}{2\pi i}\int_{β-iω}^{β+iω}F(s)e^{st}\text{d}s \quad(t>0)$<br>$\displaystyle f(t)=\sum_{k=1}^n\text{Res}[F(s)e^{st},s_k]\quad(t>0)$


|Functions<br>(original image)|Fourier Transform<br>(image)|Laplace Transform|
|:---|:---|:---|
 |$δ(t)$|1|1
|$δ(t−t_0)$|$e^{-\text{i}ω t_0}$|$e^{-st_0}\quad(t_0>0)$
|1|$2\pi δ(ω)$|$\dfrac{1}{s}\quad(\text{Re }s>0)$
|$e^{-\text{i}ω_0 t}$|$2\pi δ(ω−ω_0)$
|$t$||$\dfrac{1}{s^2}\quad(\text{Re }s>0)$
|$t^m\quad(m\in\Z)$||$\dfrac{m!}{s^{m+1}}\quad(\text{Re }s>0)$
|$t^a\quad(a>-1)$||$\dfrac{\Gamma(a+1)}{s^{a+1}}\quad(\text{Re }s>0)$
|$e^{-at}\quad(a⩾0)$|$\dfrac{1}{a+\text{i}ω}$|$\dfrac{1}{s+a}\quad(\text{Re }s+a>0)$
|$te^{-at}\quad(a⩾0)$||$\dfrac{1}{(s+a)^2}\quad(\text{Re }s+a>0)$
|$u(t)=\begin{cases} 0 & t<0 \\ 1 &t>​0 \end{cases}$|$\pi δ(ω)+\dfrac{1}{\text{i}ω}$|$\dfrac{1}{s}\quad(\text{Re }s>0)$
$\text{sgn}(t)=\begin{cases}-1&t<0 \\ 1 &t >0 \end{cases}$|$\dfrac{2}{\text{i}ω}$|
|$\text{rect}(t)=\begin{cases}1&\mid t\mid <a \\ 0 &\mid t\mid >a \end{cases}$|$\dfrac{2\sin(aω)}{ω}$
|$\cosω_0t$|$\pi [δ(ω-ω_0)+δ(ω+ω_0)]$|$\dfrac{s}{s^2+ω_0^2}\quad(\text{Re }s>0)$
|$\sinω_0t$|$\text{i}\pi [δ(ω+ω_0)-δ(ω-ω_0)]$|$\dfrac{ω_0}{s^2+ω_0^2}\quad(\text{Re }s>0)$ 
|$e^{-a^2t^2}$|$\cfrac{\sqrt{\pi}}{a}\exp(-\cfrac{ω^2}{4a^2})$|
|$\cfrac{\sinω_0t}{t}$|$\begin{cases} \pi & \vert ω \vert<ω_0 \\ 0 & \vert ω \vert>ω_0  \end{cases}$

## 非齐次项为 $δ$ 函数的常微分方程

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
<img src="https://gitee.com/WilenWu/images/raw/master/ComplexFunction/ODE-delta.png" style="zoom:67%;" /> <img src="https://gitee.com/WilenWu/images/raw/master/ComplexFunction/ODE-delta2.png" style="zoom:67%;" />

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
<img src="https://gitee.com/WilenWu/images/raw/master/ComplexFunction/ODE-delta3.png" style="zoom:67%;" /><img src="https://gitee.com/WilenWu/images/raw/master/ComplexFunction/ODE-delta4.png" style="zoom:67%;" />