---
title: 常微分方程(四)
categories:
  - 数学
  - 常微分方程
tags:
  - 数学
  - ODE
  - 微分方程
katex: true
description: 非线性微分方程、定性理论、边值问题
abbrlink: 3689ee12
date: 2021-10-30 21:58:00
cover:
top_img:
keywords:
---

# 非线性微分方程基本理论

对于一般的 n 阶微分方程 
$$
z^{(n)}=g(x;z,z',\cdots,z^{(n-1)})
$$
做变换
$$
y_1=z,y_2=z',\cdots,y_n=z^{(n-1)}
$$
则 n 阶微分方程变换为一阶微分方程组
$$
\begin{cases}y_1'=y_2 \\y_2'=y_3 \\\cdots \\y_n'=g(x,y_1,y_2,\cdots,y_n)\end{cases}
$$
向量形式为
$$
\mathbf y'=\mathbf g(x;\mathbf y)\tag{1}
$$
其中
$$
\mathbf y=\begin{pmatrix}y_1 \\ y_2 \\ \vdots \\ y_n\end{pmatrix},\mathbf g(x;\mathbf y)=\begin{pmatrix}g_1(x,y_1,y_2,\cdots,y_n) \\g_2(x,y_1,y_2,\cdots,y_n) \\\cdots \quad \cdots \\g_n(x,y_1,y_2,\cdots,y_n) \\\end{pmatrix}
$$

在讨论方程组解的性态之前，先引入一些定理，由于定理的证明同一阶方程类似，就不在重复了。
设方程组 (1) 的初始条件为
$$
\mathbf{y}(x_0)=\mathbf{y_0}
$$
<kbd>利普希茨条件</kbd>：存在常数 $L>0$ ，使得向量函数 $\mathbf g(x;\mathbf y)$在区域D内满足不等式 
$$
\|\mathbf g(x;\mathbf{\tilde y})-\mathbf g(x;\mathbf{\bar y})\|⩽L\|\mathbf{\tilde y}-\mathbf{\bar y}\|
$$
本章关于范数的定义为 $\|\mathbf y\|=\displaystyle\sqrt{\sum_{i=1}^ny_i^2}$

<kbd>局部李普希兹条件</kbd>：向量函数 $\mathbf g(x;\mathbf y)$ 在某一区域 $G$ 内连续，对于区域 $G$ 内每一点 $(x_0,\mathbf y_0)$，存在矩形闭区域 $R\sub G$ ，而 $\mathbf g(x;\mathbf y)$ 在 $R$ 上关于 $\mathbf y$ 满足李普希兹条件，其中
$$
R:|x-x_0|⩽a,\|\mathbf{y-y_0}\|⩽b
$$
则称 $\mathbf g(x;\mathbf y)$ 在域 $G$ 上关于 $\mathbf y$ 满足局部李普希兹条件。

<kbd>解的存在唯一性定理</kbd>：如果向量函数 $\mathbf g(x;\mathbf y)$  在矩形区域 
$$
R:|x-x_0|⩽a,\|\mathbf{y-y_0}\|⩽b
$$
 内连续，且关于$\mathbf y$ 满足利普希茨条件，则方程组 (1) 在区间 $|x-x_0|⩽h$ 存在唯一解 $\mathbf{y=\varphi}(x;x_0,\mathbf y_0)$，而且 $\mathbf{y_0=\varphi}(x_0;x_0,\mathbf y_0)$。其中常数 
$$
h=\min\{a,\cfrac{b}{M}\},\displaystyle M=\max_{(x,\mathbf y)\in R}\|\mathbf g(x;\mathbf y)\|
$$

<kbd>解的延拓与连续性定理</kbd>：如果向量函数 $\mathbf g(x;\mathbf y)$  在某域 $G$ 中连续，且关于 $y$ 满足局部李普希兹条件，那么方程组 (1) 满足初始条件的解 $\mathbf{y_0=\varphi}(x;x_0,\mathbf y_0) \quad (x_0,\mathbf y_0)\in G$ 可以延拓。或者延拓到 $+\infty$ 或 $-\infty$ ，或者使点点$(x,\mathbf{\varphi}(x;x_0,\mathbf y_0))$任意接近 $G$ 的边界。
并且解 $\mathbf{y_0=\varphi}(x;x_0,\mathbf y_0) \quad (x_0,\mathbf y_0)\in G$ 作为 $x,x_0,\mathbf y_0$ 的函数在存在范围内是连续性的。

<kbd>解的可微性定理</kbd>：如果向量函数 $\mathbf g(x;\mathbf y)$ 及 $\cfrac{∂g_i}{∂y_j}\quad(i,j=1,2,\cdots,n)$ 在域 $G$ 内连续，则方程组 (1) 的解 $\mathbf{y_0=\varphi}(x;x_0,\mathbf y_0) \quad (x_0,\mathbf y_0)\in G$ 作为 $x,x_0,\mathbf y_0$ 的函数在存在范围内连续可微。

方程组 $\mathbf y'=\mathbf g(x;\mathbf y)$ 称为非自治系统，右端函数不显含 $x$ 时，方程组 $\mathbf y'=\mathbf g(\mathbf y)$ 称为自治系统。

# 定性理论初步

对于二阶微分方程组自治系统：
$$
\begin{cases}
\cfrac{dx}{dt}=f(x,y) \\
\cfrac{dy}{dt}=g(x,y)
\end{cases}\tag{2}
$$
对于自治系统，若$f(x_0,y_0)=g(x_0,y_0)=0$，则称$(x_0,y_0)$ 为系统的==平衡点==或者==奇点==。
若视 $t$ 为时间， $(x,y)$ 为二维空间的动点， $f(x,y)$与$g(x,y)$为==速度分量==，称$(x,y)$所在的平面为==相平面==， 相平面上的点称为为相点。
设 $x=x(t),y=y(t)$ 是自治系统 (2) 的解，它在以 $t,x,y$ 为坐标的（欧几里得）空间中决定了一条曲线，称为==积分曲线==（解曲线）。积分曲线在相平面上的投影称为==轨线==。

一阶常系数齐次线性自治系统
$$
\begin{cases}
\cfrac{dx}{dt}=a_{11}x+a_{12}y \\
\cfrac{dy}{dt}=a_{21}x+a_{22}y
\end{cases}\tag{3}
$$
系数矩阵 $\mathbf A=\begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} 
\end{pmatrix}$ 的特征方程为 $|\lambda\mathbf{E-A}|=\begin{vmatrix}
a_{11}-\lambda & a_{12} \\
a_{21} & a_{22}-\lambda 
\end{vmatrix}=0$
即 $\lambda^2+(a_{11}+a_{22})\lambda+a_{11}a_{22}-a_{12}a_{21}$

(1) $\lambda_1,\lambda_2$为不相等的负实根，平衡点是稳定的。
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/20200326095533878.PNG" style="zoom: 67%;" />

(2)  $\lambda_1,\lambda_2$为不相等的正实根，平衡点是不稳定的。
(3)  $\lambda_1,\lambda_2$为异号实根，平衡点是不稳定的。
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/20200326095936647.PNG" style="zoom:67%;" />

(4) $\lambda_1=\lambda_2>0$平衡点是不稳定的；$\lambda_1=\lambda_2<0$平衡点是稳定的。
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/20200326100240897.PNG" style="zoom:67%;" />

(5) $\lambda_1,\lambda_2=\alpha\pm\beta i\quad(\beta\neq0)$为共轭复根，$\alpha<0$时，平衡点是稳定的；$\alpha>0$时，平衡点是不稳定的。
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/20200326100409344.PNG" style="zoom:67%;" />
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/DifferentialEquation/2020032610155280.PNG" style="zoom:67%;" />

> [常微分方程的“动力系统”（相空间）分析简介](http://bbs.21ic.com/icview-2599270-1-1.html)
> 百度百科：[相空间](https://baike.baidu.com/item/%E7%9B%B8%E7%A9%BA%E9%97%B4/8172498#viewPageContent) 、 [相空间表述](https://baike.baidu.com/item/%E7%9B%B8%E7%A9%BA%E9%97%B4%E8%A1%A8%E8%BF%B0/22687295) (量子力学)
> 知乎：[什么是相空间？](https://www.zhihu.com/question/264986355?sort=created)


------
<center>*待更新*</center>

Nonlinear systems

微分方程解析理论analytic theory of differential equation
代数微分方程algebraic differential equation
潘勒韦理论Painleve&1& theory

定性理论qualitative theory
Equilibria and limit cycles(平衡与极限环)
Hyperbolic equilibria and stable limit cycle (双曲平衡与稳定极限环)
Chaos and the Lorenz attractor(混沌与洛伦兹吸引子)

流动奇点movable singular point
Perturbation Method(摄动法; 微扰方法)
Phase Portraits(相图; 相轨迹;)

相空间(Phase Space)
相即是状态(state) 


# 边值问题

(boundary value problem)
边值条件boundary value condition
自伴边值问题self-adjoint boundary value problem


> **参考文献：**
> 丁同仁.《常微分方程教程》
> 王高雄.《常微分方程》
> 窦霁虹 付英《常微分方程》.西北大学(MOOC) 
> 《高等数学》.国防科技大学(MOOC)