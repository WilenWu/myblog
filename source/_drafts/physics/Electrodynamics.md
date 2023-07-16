---
title: 电动力学
categories:
  - Physics
  - Basic Physics
tags:
cover: 
top_img: 
katex: true
abbrlink: 
date: 
description: 
---



电动力学(Electrodynamics)



引入 Maxwell 张量
$$
F_{\mu\nu}=\partial_{\mu}A_{\nu}-\partial_{\nu}A_{\mu}
$$
得到 Maxwell 方程组相对论形式
$$
\partial_{\mu}\vec{F}^{\mu\nu}=j^{\nu}
$$

Electrodynamics of Continuous Media, 连续介质电动力学



时空图（Spacetime Diagrams）

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/physics/light_cone.jpg" alt="light_cone" style="zoom:67%;" />





<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/physics/UPhysics_Lorentz.jpg" alt="UPhysics_Lorentz" style="zoom: 67%;" />



<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/physics/UPhysics_SpaceTwins.jpg" alt="UPhysics_SpaceTwins" style="zoom:67%;" />



<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/physics/UPhysics_STtrain.jpg" alt="UPhysics_STtrain" style="zoom:67%;" />

# 狭义相对论

## 洛伦兹变换

狭义相对论（Special Relativity）
时空悖论（Spacetime Paradoxes）
参考系（Reference Frames）



时空间隔（Spacetime Intervals）
$$
\mathrm ds^2=c^2\mathrm dt^2-(\mathrm dx^2+\mathrm dy^2+\mathrm dz^2)
$$
洛伦兹变换（Lorentz Transformations）

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/physics/Lorentz_transformation.jpg" alt="Lorentz_transformation" style="zoom:80%;" />
$$
\begin{cases}
t'=\gamma(t-\dfrac{v}{c^2}x) \\
x'=\gamma(x-vt) \\
y'=y \\
z'=z
\end{cases}
$$
其中
$$
\gamma=\frac{1}{\sqrt{1-\beta^2}},\quad \beta=\frac{v}{c}
$$
将变换写为矩阵形式
$$
x'_{\mu}=a_{\mu\nu}x_{\nu}
$$
其中 $\mu,\nu=0,1,2,3$ 代表矩阵的行列标号
$$
A(\nu)=(a_{\mu\nu})=\begin{bmatrix}
\gamma&-\gamma v/c^2 &0&0 \\
-\gamma v&\gamma&0&0 \\
0&0&1&0 \\
0&0&0&1 \\
\end{bmatrix}
$$
洛伦兹速度变换
$$
\begin{cases}
u'_x=\dfrac{u_x-v}{1-u_xv/c^2} \\
u'_y=\dfrac{u_y}{\gamma(1-u_xv/c^2)} \\
u'_z=\dfrac{u_z}{\gamma(1-u_xv/c^2)}
\end{cases}
$$
逆变换
$$
\begin{cases}
u_x=\dfrac{u'_x+v}{1+u'_xv/c^2} \\
u_y=\dfrac{u'_y}{\gamma(1+u'_xv/c^2)} \\
u_z=\dfrac{u'_z}{\gamma(1+u'_xv/c^2)}
\end{cases}
$$
同时单位相对性
$$
\Delta t'=\gamma(\Delta t-\frac{v}{c^2}\Delta x) \\
\Delta x'=\gamma(\Delta x-v\Delta t)
$$
洛伦兹收缩
$$
l=l_0\sqrt{1-\beta^2}=l_0/\gamma
$$
其中，$l_0$ 为物体相对于坐标系静止时的长度

时间膨胀（Time Dilation）
$$
\Delta t=\frac{\Delta\tau}{\sqrt{1-\beta^2}}=\gamma\Delta\tau
$$
其中，$\Delta\tau$ 为固有时间

## 相对论力学

四维速度
$$
U_\mu=\frac{\mathrm dx_\mu}{\mathrm d\tau}=(\gamma\vec{u},\mathrm i\gamma c)
$$
四维动量
$$
P_\mu=m_0U_\mu=(\gamma m_0\vec{u},\mathrm i\gamma m_0c)
$$
令 
$$
m=\frac{m_0}{\sqrt{1-u^2/c^2}}
$$
则动量
$$
\vec p=\gamma m_0\vec u=m\vec u
$$
质能方程（Mass-energy Equivalence）
$$
E=mc^2
$$
使用泰勒展开
$$
E\approx m_0c^2+\frac{1}{2}m_0u^2
$$
于是相对论动能
$$
T=E-m_0c^2
$$
则
$$
P_\mu=(\vec p,\mathrm i\frac{1}{c} E)
$$
得到守恒量
$$
P_\mu P_\mu=P'_\mu P'_\mu=\text{const}
$$
取 $S'$ 为物体静止的参考系，可得相对论的能量与动量关系是
$$
E=\sqrt{p^2c^2+m_0^2c^2}
$$

## 牛顿力学的协变式

牛顿第二定律
$$
\vec F=\frac{\mathrm d}{\mathrm dt}\vec p
$$
由能量守恒定律得
$$
\vec F\cdot\vec u=\frac{\mathrm dE}{\mathrm dt}
$$
引进固有时间 $\mathrm d\tau=\mathrm dt/\gamma$
$$
\gamma\vec F=\frac{\mathrm d}{\mathrm d\tau}\vec p \\
$$
于是
$$
\mathrm i\frac{1}{c}\gamma\vec F\cdot\vec u=\frac{\mathrm d}{\mathrm d\tau}(\mathrm i\frac{1}{c} E)
$$
令$F_\mu=(\gamma\vec F,\mathrm i\frac{1}{c}\gamma\vec F\cdot\vec u)$，称为四维力，则
$$
F_\mu=\frac{\mathrm dP_\mu}{\mathrm d\tau}
$$
