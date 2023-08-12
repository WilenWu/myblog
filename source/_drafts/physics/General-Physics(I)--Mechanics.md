---
title: 普通物理学(I)-力学
date: 
categories:
  - Physics
  - 基础物理
tags:
cover: 
top_img: 
katex: true
abbrlink: 
description: 力学和热学
---

传统的《普通物理学》包括：牛顿力学、热学、电磁学、光学、原子物理学，但不包括相对论和量子力学以及物理学的前沿内容。随着科学的发展，相对论和量子力学以及物理学的前沿内容渐渐地进入了《普通物理学》。为了区分一下，后来有了《大学物理》的提法。普通物理是物理学专业学生的必修课，大学物理是非物理专业的理工科学生的必修课。普通物理在难度和广度上都要大于大学物理，他们是两门不同的课程。

# 质点运动学

Motion, 运动

习题：

追击线




# 质点动力学

Newtonian Mechanics
Force

Newton's First Law
Newton's Second Law
Newton's Third Law
摩擦(Friction)

# 机械能守恒

Energy and Work, 功和能
动能(Kinetic Energy)
势能(Potential Energy)
机械能(Mechanical Energy)
能量守恒(Conservation of Energy)
冲量与动量Impulse and Momentum

# 刚体力学

The Center of Mass 质心

Rotation and Angular Momentum 旋转与角动量
Kinetic Energy of Rotation
the Rotational Inertia 转动惯量
Torque 力矩
6-14 Angular Momentum
6-15 Newton's Second Law in Angular orm
6-16 The Angular Momentum of a System of Particles
6-17 The Angular Momentum of a Rigid Body Rotating About a Fixed Axis
6-18 Precession of a Gyroscope

# 引力

Gravitation 引力

7-2 Newton's Law of Gravitation
7-3 Gravitation and the Principle of Superposition
7-4 Gravitation Near Earth's Surface
7-5 Gravitation Inside Earth
7-6 Gravitational Potential Energy
7-7 Planets and Satellites: Kepler's Laws
7-8 Satellites: Orbits and Energy
7-9 Einstein and Gravitation

# 连续体力学

## 固体弹性力学

应力和应变

## 流体静力学

帕斯卡原理

阿基米德原理

表面张力

毛细现象

## 理想流体力学

伯努力方程

## 黏性流体

层流和湍流



# 狭义相对论

Relativity 相对论

22-2 The Postulates
22-3 Measuring an Event
22-4 The Relativity of Simultaneity
22-5 The Relativity of Time
22-6 The Relativity of Length
22-7 The Lorentz Transformation
22-8 Some Consequences of the Lorentz Equations
22-9 The Relativity of Velocities
22-10 Doppler Effect for Light
22-11 A New Look at Momentum
22-12 A New Look at Energy

时空图（Spacetime Diagrams）

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/physics/light_cone.jpg" alt="light_cone" style="zoom:67%;" />



<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/physics/UPhysics_Lorentz.jpg" alt="UPhysics_Lorentz" style="zoom: 67%;" />



<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/physics/UPhysics_SpaceTwins.jpg" alt="UPhysics_SpaceTwins" style="zoom:67%;" />



<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/physics/UPhysics_STtrain.jpg" alt="UPhysics_STtrain" style="zoom:67%;" />


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


# 振动与波

Oscillations 振动

8-2 Simple Harmonic Motion
8-3 The Force Law for Simple Harmonic Motion
8-4 Energy in Simple Harmonic Motion
8-5 An Angular Simple Harmonic Oscillator
8-6 Pendulums
8-7 Simple Harmonic Motion and Uniform Circular Motion
8-8 Damped Simple Harmonic Motion
8-9 Forced Oscillations and Resonance

## 简谐运动

简谐运动(Simple harmonic motion, SHM)是最基本也最简单的机械振动。当某物体进行简谐运动时，物体所受的力跟位移成正比，并且总是指向平衡位置。它是一种由自身系统性质决定的周期性运动（如单摆运动和弹簧振子运动）。实际上简谐振动就是正弦振动。

**匀速圆周运动在直径上的投影**

![匀速圆周投影](physics.assets/%E5%8C%80%E9%80%9F%E5%9C%86%E5%91%A8%E6%8A%95%E5%BD%B1.svg)

$P$ 在直径 $AB$ 上的投影 $P'$ 做往复运动， $P'$ 的加速度 $a=\omega^2r\cos\theta$

其中 $r\cos\theta$ 是$P'$ 到圆心的距离 $x$ ，若把$x$ 看作相对平衡位置的位移，则
$$
a=-\omega^2x
$$
则任意简谐运动总有与之对应的匀速圆周运动。

**简谐振动**：设参考圆半径 $r=A$ ，取向右为正方向，则
$$
x=A\cos\theta=A\cos\omega t
$$
振动速度
$$
v=-\omega A\sin\theta=-\omega A\sin\omega t
$$
振动加速度
$$
a=-\omega^2x=-\omega^2A\cos\omega t
$$
其中 $A$ 称为振幅， $\omega$ 称为简谐振动的圆频率 $\omega=2\pi f=\dfrac{2\pi}{T}$

**周期**

动力学观点 $a=-\dfrac{k}{m}x$ ，参考圆观点 $a=-\omega^2x$，则 $\omega^2=\dfrac{k}{m}$
$(\dfrac{2\pi}{T})^2=\dfrac{k}{m}$ ，于是
$$
T=2\pi\sqrt{\frac{m}{k}}
$$
可知周期只与振动系统本身有关，和振幅大小无关，所以此周期称为固有周期。

**能量**

任意时刻
$$
E=\frac{1}{2}mv^2+\frac{1}{2}kx^2=\frac{1}{2}m(\omega A\sin\omega t)^2+\frac{1}{2}k(A\cos\omega t)^2
$$
又因为 $\omega^2=\dfrac{k}{m}$，所以
$$
E=\frac{1}{2}kA^2
$$


## 单摆

![单摆](physics.assets/%E5%8D%95%E6%91%86.svg)
$$
\begin{cases}
T-mg\cos\theta=ma_n \\
mg\sin\theta=ma_\tau
\end{cases}
$$
回复力为切向分力 $F=mg\sin\theta$
当 $\theta$ 很小时，$\sin\theta\approx\theta$
相对平衡位置的位移 $x$ 与力 $F$ 间的夹角近似为 $2\pi$ ，即方向相反。所以
$$
F=-\frac{mg}{L}x
$$
即单摆在$\theta$ 很小时为简谐振动。




## 波动

Waves 

9-2 Types of Waves
9-3 Transverse and Longitudinal Waves
9-4 Wavelength and Frequency
9-5 The Speed of Wave
9-6 Energy and Power of a Wave Traveling Along a String
9-7 The Wave Equation
9-8 Standing Waves
9-9 Sound Waves
9-10 Traveling Sound Waves
9-11 Interference
9-12 Intensity and Sound Level
9-13 Sources of Musical Sound
9-14 Beats
9-15 The Doppler Effect
9-16 Supersonic Speeds, Shock Waves
