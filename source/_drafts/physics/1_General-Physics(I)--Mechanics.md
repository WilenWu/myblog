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
description: 
---

传统的《普通物理学》包括：牛顿力学、热学、电磁学、光学、原子物理学，但不包括相对论和量子力学以及物理学的前沿内容。随着科学的发展，相对论和量子力学以及物理学的前沿内容渐渐地进入了《普通物理学》。为了区分一下，后来有了《大学物理》的提法。普通物理是物理学专业学生的必修课，大学物理是非物理专业的理工科学生的必修课。普通物理在难度和广度上都要大于大学物理，他们是两门不同的课程。

# 质点运动学

物体的大小和形状可以忽略不计时，我们就可以把该物体看作是一个具有质量的几何点，称为质点(mass point，particle)。

运动 Motion

参考系(reference frame)

坐标系(coordinate system)

时间(time)和空间(space)

位矢(position vector)
$$
\mathbf r=x\mathbf i+y\mathbf j+z\mathbf k
$$
运动方程 
$$
\mathbf r(t)=x(t)\mathbf i+y(t)\mathbf j+z(t)\mathbf k
$$


位移(displacement)
$$
\Delta\mathbf r=\mathbf r(t+\Delta t)-\mathbf r(t)
$$
速度(velocity)
$$
\mathbf v=\lim_{\Delta t\to 0}\frac{\Delta\mathbf r}{\Delta t}=\frac{\mathrm d\mathbf r}{\mathrm dt}
$$
速率
$$
v=\lim_{\Delta t\to 0}\frac{\Delta s}{\Delta t}=\frac{\mathrm ds}{\mathrm dt}=|\mathbf v|
$$
加速度(acceleration)
$$
\mathbf a=\lim_{\Delta t\to 0}\frac{\Delta\mathbf v}{\Delta t}=\frac{\mathrm d\mathbf v}{\mathrm dt}=\frac{\mathrm d^2\mathbf r}{\mathrm dt^2}
$$


匀加速运动
$$
\mathbf r=\mathbf r_0+\mathbf v_0t+\frac{1}{2}\mathbf at^2
$$
抛体运动
$$
\begin{cases}
x=v_0t\cos\theta \\
y=v_0t\sin\theta-\cfrac{1}{2}gt^2
\end{cases}
$$
圆周运动

由 $s=R\theta$ 可得
$$
v=\frac{\mathrm ds}{\mathrm dt}=R\frac{\mathrm d\theta}{\mathrm dt}
$$
定义角速度(angular velocity)
$$
\omega=\frac{\mathrm d\theta}{\mathrm dt}=\frac{v}{R}
$$
角加速度(angular acceleration)
$$
\alpha=\frac{\mathrm d\omega}{\mathrm dt}=\frac{\mathrm d^2\theta}{\mathrm dt^2}
$$
加速度
$$
\mathbf a=\mathbf a_t+\mathbf a_n
$$
切向加速度(tangential acceleration)沿切向方向
$$
a_t=\frac{\mathrm dv}{\mathrm dt}=R\alpha
$$
法向加速度(normal acceleration)指向圆心
$$
a_n=\frac{v^2}{R}=R\omega^2
$$



# 质点动力学

## 牛顿运动定律

**Newton's 1st Law**：
$$
\frac{\mathrm d\mathbf v}{\mathrm dt}=0\quad\text{s.t.}\quad \mathbf F=0
$$
**Newton's 2nd Law**：运动变化与所加的力成正比
$$
\mathbf F=\frac{\mathrm d\mathbf p}{\mathrm dt}
$$
其中
$$
\mathbf p=m\mathbf v
$$
称为**动量**(momentum)，所以
$$
\mathbf F=m\mathbf a
$$
**Newton's 3rd Law**：作用力总存在反作用力，且大小相等方向相反
$$
\mathbf F_{12}=-\mathbf F_{21}
$$

## 常见的几种力

重力
$$
f=mg
$$
弹性力
$$
f=-kx
$$
摩擦力(friction)
$$
f_k=\mu_kN
$$
流体曳力
$$
f_d=kv
$$
对于物体在空气中运动的情况，曳力的大小可以表示
$$
f_d=\frac{1}{2}C\rho Av^2
$$
终极速率
$$
v_t=\sqrt{\frac{2mg}{C\rho A}}
$$
基本的自然力

引力 Gravitation
$$
f=-\frac{Gm_1m_2}{r^2}
$$
电磁力
$$
f=\frac{kq_1q_2}{r^2}
$$

强力

弱力

## 伽利略相对性原理

$$
\mathbf v=\mathbf v'+\mathbf u \\
\mathbf a=\mathbf a'
$$



## 非惯性系和惯性力

平动惯性力
$$
\mathbf F_i=-m\mathbf a_0
$$
转动惯性力
$$
\mathbf F_i=m\omega^2\mathbf r
$$

科里奥利力
$$
\mathbf F_c=2m\mathbf v'\times\omega
$$
潮汐：地球上观察到的一种惯性力作用的表现

# 动量守恒和机械能守恒

动量定理：合外力的冲量等于质点(或质点系)动量的增量，即
$$
\mathbf F\mathrm dt=\mathrm d\mathbf p=m\mathrm d\mathbf v
$$
冲量(impulse)
$$
\mathbf I=\int_t^{t+\Delta t}\mathbf F\mathrm dt=\int_{\mathbf p}^{\mathbf p+\Delta\mathbf p}\mathrm d\mathbf p=\Delta\mathbf p
$$
质心 The Center of Mass
$$
\mathbf r_c=\frac{1}{m}\sum_im_i\mathbf r_i \quad\text{or}\quad
\mathbf r_c=\frac{1}{m}\int\mathbf r\mathrm dm
$$
质心速度
$$
\mathbf v_c=\frac{\mathrm d\mathbf r_c}{\mathrm dt}=\frac{1}{m}\sum_i\mathbf p_i
$$
质心运动定理
$$
\mathbf F=\frac{\mathrm d\mathbf p}{\mathrm dt}=\frac{m\mathrm d\mathbf v_c}{\mathrm dt}=m\mathbf a_c
$$
动量守恒定律
$$
\sum\mathbf p_i=\text{const}\quad\text{s.t.}\quad \sum\mathbf F_i=0
$$
力矩(torque)
$$
\mathbf M=\mathbf r\times \mathbf F
$$
角动量 Angular Momentum
$$
\mathbf L=\mathbf r\times \mathbf p=\mathbf r\times m\mathbf v
$$
由于
$$
\frac{\mathrm d\mathbf L}{\mathrm dt}
=\mathbf r\times\frac{\mathrm d\mathbf p}{\mathrm dt}+\frac{\mathrm d\mathbf r}{\mathrm dt}\times \mathbf p
$$
其中
$$
\frac{\mathrm d\mathbf p}{\mathrm dt}=\mathbf F\quad\text{and}\quad
\frac{\mathrm d\mathbf r}{\mathrm dt}\times \mathbf p=\mathbf v\times m\mathbf v=0
$$
所以角动量定理
$$
\mathbf M=\frac{\mathrm d\mathbf L}{\mathrm dt}
$$
角动量守恒定律
$$
\sum\mathbf L_i=\text{const}\quad\text{s.t.}\quad \sum\mathbf M_i=0
$$


功和能(Work and Energy)
$$
\mathrm dE=\mathbf F\cdot\mathrm d\mathbf r
$$

$$
W_{AB}=\int_{A}^{B}\mathbf F\cdot\mathrm d\mathbf r
$$
速度变化
$$
\mathrm dE=\mathbf F\cdot\mathrm d\mathbf r
=m\frac{\mathrm d\mathbf v}{\mathrm dt}\cdot \mathbf v\mathrm dt
=m\mathbf v\cdot\mathrm d\mathbf v
=\mathrm d(\frac{1}{2}mv^2)
$$
定义动能(Kinetic Energy)
$$
E_k=\frac{1}{2}mv^2=\frac{p^2}{2m}
$$


势能(Potential Energy)

重力势能
$$
E_p=mgh
$$
弹性势能
$$
E_p=\frac{1}{2}kx^2
$$
引力势能
$$
E_p=-\frac{Gm_1m_2}{r}
$$
势能和保守力
$$
\mathbf F=-\nabla E_p
$$
机械能(Mechanical Energy)
$$
E=E_k+E_p
$$

$$
W_{AB}=E_B-E_A
$$



能量守恒(Conservation of Energy)
$$
E_{in}=\text{const}\quad\text{s.t.}\quad W_{ex}=0
$$


# 刚体力学

转动惯量 Rotational Inertia 
$$
J=\sum_im_ir_i^2 \quad\text{or}\quad
J=\int r^2\mathrm dm
$$
定轴转动
$$
\mathbf L=J\omega
$$

$$
\mathbf M=\frac{\mathrm d\mathbf L}{\mathrm dt}=J\alpha
$$

平行轴定理
$$
J=J_c+md^2
$$
转动动能 Kinetic Energy of Rotation
$$
\mathrm dE=\mathbf F\cdot\mathrm d\mathbf r
=M\mathrm d\theta
=J\frac{\mathrm d\omega}{\mathrm dt}\mathrm d\theta
=J\omega\mathrm d\omega
=\mathrm d(\frac{1}{2}J\omega^2)
$$
定义
$$
E_k=\frac{1}{2}J\omega^2
$$
力矩的功
$$
W=\int_{\theta_1}^{\theta_2}M\mathrm d\theta
$$
陀螺仪进动 Precession of a Gyroscope

角速度
$$
\Omega=\frac{M}{L\sin\theta}
$$



# 连续体力学

## 固体弹性力学

物理学把固态和液态统称为凝聚态物理

晶体具有规则的几何形状。具有各向异性（如热膨胀系数、力学、机械强度、导电性、力学、光学等）

- 单晶：一个物体就是一个完整晶体（雪花、食盐颗粒、单晶硅）
- 多晶：整个物体由许多杂乱无章的小晶体组成，无固定几何形状，也不显示各向异性（如金属）

空间点阵、离子点阵、原子点阵、分子点阵、金属点阵

微粒的热运动表现为在平衡位置不停地作微小的振动



应力和应变

## 流体静力学

液体分子只在很小的区域内作有规则的排列，这种区域是暂时的，边界和大小随时改变，有时瓦解，有时又重建。液体由大量这种小区域形成，小区域杂乱无章的分布，因此液体表现各向同性。液体分子的热运动，主要在平衡位置作微小振动，仅能保持在一个短暂的时间（约10^-10^s），右转到另一个平衡位置依次下去，因此液体具有流动性。

帕斯卡原理

阿基米德原理

表面张力：液体表面层有收缩的趋势，表面各部分之间的相互吸引力叫表面张力

边界线上作用的表面张力
$$
F=\gamma l
$$
表面张力系数

毛细现象

## 理想流体力学

伯努力方程

## 黏性流体

层流和湍流




# 振动与波

Oscillations 振动
匀速圆周运动 Uniform Circular Motion

## 简谐运动

简谐运动(Simple harmonic motion, SHM)是最基本也最简单的机械振动。当某物体进行简谐运动时，物体所受的力跟位移成正比，并且总是指向平衡位置。它是一种由自身系统性质决定的周期性运动（如单摆运动和弹簧振子运动）。实际上简谐振动就是正弦振动。
$$
x=A\cos(\omega t+\varphi)
$$
其中
$$
\omega=\frac{2\pi}{T}=2\pi\nu
$$
简谐运动的速度和加速度分别为
$$
v=\frac{\mathrm dx}{\mathrm dt}=\omega A\cos(\omega t+\varphi+\frac{\pi}{2})\\
a=\frac{\mathrm d^2x}{\mathrm dt^2}=\omega^2 A\cos(\omega t+\varphi+\pi)
$$
比较两式可知
$$
a=-\omega^2 x
$$
这一关系式说明，简谐运动的加速度和位移成正比而反向。
$$
F=ma=-m\omega^2x=-kx
$$
于是固有角频率
$$
\omega=\sqrt{\frac{k}{m}}
$$
固有周期
$$
T=\frac{2\pi}{\omega}=2\pi\sqrt{\frac{m}{k}}
$$


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

Pendulums

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

阻尼振动
$$
A=A_0e^{-\beta t}
$$
时间常数
$$
\tau=\frac{1}{2\beta}
$$
Q值
$$
Q=2\pi\frac{\tau}{T}=\omega\tau
$$
受迫振动 Forced Oscillations 



共振 Resonance


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

# 狭义相对论基础

牛顿相对性原理和伽利略变换
$$
\mathbf v'=\mathbf v-\mathbf u
$$

$$
\mathbf a'=\mathbf a
$$

$$
\mathbf F=m\mathbf a'
$$



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
U_\mu=\frac{\mathrm dx_\mu}{\mathrm d\tau}=(\gamma\mathbf{u},\mathrm i\gamma c)
$$
四维动量
$$
P_\mu=m_0U_\mu=(\gamma m_0\mathbf{u},\mathrm i\gamma m_0c)
$$
令 
$$
m=\frac{m_0}{\sqrt{1-u^2/c^2}}
$$
则动量
$$
\mathbf p=\gamma m_0\mathbf u=m\mathbf u
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
P_\mu=(\mathbf p,\mathrm i\frac{1}{c} E)
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
\mathbf F=\frac{\mathrm d}{\mathrm dt}\mathbf p
$$
由能量守恒定律得
$$
\mathbf F\cdot\mathbf u=\frac{\mathrm dE}{\mathrm dt}
$$
引进固有时间 $\mathrm d\tau=\mathrm dt/\gamma$
$$
\gamma\mathbf F=\frac{\mathrm d}{\mathrm d\tau}\mathbf p \\
$$
于是
$$
\mathrm i\frac{1}{c}\gamma\mathbf F\cdot\mathbf u=\frac{\mathrm d}{\mathrm d\tau}(\mathrm i\frac{1}{c} E)
$$
令$F_\mu=(\gamma\mathbf F,\mathrm i\frac{1}{c}\gamma\mathbf F\cdot\mathbf u)$，称为四维力，则
$$
F_\mu=\frac{\mathrm dP_\mu}{\mathrm d\tau}
$$

# 附录

## 常用的数学极限

(1) 在极限情况下，弧长$\Delta s$ 和弦长$\Delta l$ 相等
$$
\lim\limits_{\theta\to0}\frac{\Delta l}{\Delta s}=1
$$

证：设曲率半径为 $\rho$。则
$$
\frac{\Delta l}{\Delta s}=\frac{2\rho\sin(\theta/2)}{\theta\rho}=\frac{\sin(\theta/2)}{\theta/2}
$$
由夹逼定理知道
$$
\lim\limits_{\theta\to0}\frac{\sin(\theta/2)}{\theta/2}=1
$$
得证。

## 天体运动方程

椭圆方程
$$
\frac{x^2}{a^2}+\frac{y^2}{b^2}=1
$$

$$
\frac{1}{2}mv_1^2-\frac{GMm}{a-c}=\frac{1}{2}mv_2^2-\frac{GMm}{a+c} \\
m(a-c)v_1=m(a+c)v_2
$$

求出
$$
v_2=\frac{a-c}{a+c}v_1 \\
v_1=\sqrt{\frac{(a+c)GM}{a(a-c)}}
$$

$$
E_{tot}=\frac{1}{2}mv_1^2-\frac{GMm}{a-c}=\frac{1}{2}mv_2^2-\frac{GMm}{a+c} \\
=\frac{GMm}{a-c}(\frac{a+c}{2a}-1)=-\frac{GMm}{2a}<0
$$



抛物线 $y=2px$
$$
E_{tot}=\frac{1}{2}mv_O^2-\frac{GMm}{p/2}
$$
由于 $\rho=p$
$$
m\frac{v_O^2}{\rho}=\frac{GMm}{(p/2)^2}=\frac{GMm}{(\rho/2)^2}
$$
则 $E_{tot}=0$

双曲线
$$
E_{tot}=\frac{1}{2}mv_O^2-\frac{GMm}{c-a}=\frac{1}{2}mv_\infty^2
$$

$$
\frac{1}{2}v_O(c-a)=\frac{1}{2}v_\infty r_\infty\sin\theta
=\frac{1}{2}v_\infty c\sin\alpha =\frac{1}{2}v_\infty b
$$

所以
$$
v_\infty=\frac{c-a}{b}v_O
$$
因此
$$
E_{tot}=\frac{1}{2}m(\frac{c-a}{b})^2v_O^2 \implies
\frac{1}{2}mv_O^2=\frac{GMm(c+a)}{2a(c-a)} \\
E_{tot}>0
$$

