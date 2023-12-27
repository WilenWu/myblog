---
title: 量子力学
categories:
  - Physics
  - 基础物理
tags:
cover: 
top_img: 
katex: true
abbrlink: 
date: 
description: 
---



1.新概念量子物理，第二版，赵凯华，高等教育出版社. 本书体系新颖，内容详实，但不太适合初学者；

2.量子力学(卷I)，第四版，曾谨言著. 国内量子力学通用教材，内容庞杂；

3.Modern Quantum Mechanics (1995)，J. J. Sakurai.量子力学现代讲法的开创者；

4.Introduction to quantum mechanics, D.J. Griffiths.国内目前流行的美国教材。

Braket Notation
Hilbert Spaces
Interference
Observables
Operators
Photons
Quantum Measurement
Schrodinger Equation
Spin
Uncertainty Principles
Vector Spaces

# 23 Quantum Physics 量子物理

23-2 The Photon, the Quantum of Light
23-3 Electrons and Matter Waves
23-4 Schrrdinger's Equation and Heisenberg's Uncertainty Principle
23-5 Energies of a Trapped Electron One-Dimensional Traps
23-6 The Bohr Model of the Hydrogen tom
23-7 Some Properties of Atoms
23-8 Angular Momenta and Magnetic Dipole Moments
23-9 The Stern-Gerlach Experiment
23-10 Magnetic Resonance
23-11 The Pauli Exclusion Principle
23-12 Building the Periodic Table
23-13 X Rays and the Ordering of the Elements
23-14 Lasers and Laser Light



## 量子物理

氢原子光谱的谱线频率满足公式
$$
\nu=Rc(\frac{1}{\tau_2}-\frac{1}{\tau_1})
$$
其中 $R$ 为Rydberg 常量。$\tau_1,\tau_2\in\Z^+$ 且 $\tau_1>\tau_2$ 。当 $\tau_2=2$ 时，$\tau_1$ 取不同值时就得到氢原子光谱的正常巴尔末线系。

(1) 引入徳布罗意关系
$$
E=h\nu,\quad \lambda=h/p
$$
假设 $2\pi r=n\lambda$ ，则
$$
p=\frac{nh}{2\pi r}=\frac{n\hbar}{r}
$$
由经典力学
$$
\frac{e^2}{4\pi\varepsilon_0r^2}=m_e\frac{v^2}{r}=\frac{2E_k}{r}=\frac{p^2}{m_er}
=\frac{n^2\hbar^2}{m_er^3}
$$
解得电子定态轨道半径
$$
r=\frac{4\pi\varepsilon_0n^2\hbar^2}{m_ee^2}
$$
(2) 当 $n=1$ 时，$r=a$ 为玻尔半径
$$
E_k=\frac{1}{2}m_ev^2=\frac{e^2}{8\pi\varepsilon_0r} \\
E_{tot}=E_k-\frac{e^2}{4\pi\varepsilon_0r}=-\frac{e^2}{8\pi\varepsilon_0r}
$$
将轨道半径带入上式，得到处于定态的电子能量
$$
E_n=-\frac{2\pi^2m_ee^4}{(4\pi\varepsilon_0)^2n^2h^2}
$$
能级差
$$
E_{n2}-E_{n1}=-\frac{2\pi^2m_ee^4}{(4\pi\varepsilon_0)^2h^2}(\frac{1}{n_2^2}-\frac{1}{n_1^2})=h\nu
$$
所以
$$
\nu=-\frac{2\pi^2m_ee^4}{(4\pi\varepsilon_0)^2h^3}(\frac{1}{n_2^2}-\frac{1}{n_1^2})
$$
符合以上的频率谱线经验公式。

(3) 假定定态电子轨道为圆形，电子必须满足角动量量子化条件 $J=rp=n\hbar$ 。索末菲把此条件推广为
$$
\oint p\mathrm dq=n\hbar
$$
其中 $q$ 为电子广义坐标，$p$ 是对应的广义动量，积分沿电子轨道进行一周，称为玻尔-索末菲量子化条件。不仅适用于圆轨道，也适用于椭圆轨道。

(4) 
$$
E=\hbar\omega,\quad p=\hbar k,\quad(\lambda=2\pi/k)
$$
自由粒子的德布罗意波通常取平面波形式
$$
\varphi_p(x,t)=A\exp(\frac{i}{\hbar}(px-Et))=A\exp(i(kx-\omega t))
$$

## 态空间

对于狄拉克符号，所有右矢构成一个矢量空间，所有左矢也构成一个矢量空间，后者称为前者的对偶空间。

量子力学函数空间任意两个矢量 $\psi$ 和 $\varphi$ 内积定义为：
$$
\lang\psi,\varphi\rang=\int_{-\infty}^{+\infty}\psi^*\varphi\mathrm d\tau
$$
右矢空间和左矢空间都称为量子力学态空间，经常称为希尔伯特空间。

自旋算符 $\hat S_z$ 表象及本征失

(1)  $\hat S_z$ 的本征失

设自旋的$z$ 分量算符 $\hat S_z$ 有两个本征右失 $|+\rang_z$ 和 $|-\rang_z$ ，分别对应本征值 $+\hbar/2$ 和 $-\hbar/2$ 即
$$
\hat S_z|+\rang_z=\frac{\hbar}{2}|+\rang_z,\quad 
\hat S_z|-\rang_z=-\frac{\hbar}{2}|-\rang_z
$$
 $|+\rang_z$ 和 $|-\rang_z$满足正交归一条件
$$
_z\lang+|+\rang_z=1,\quad _z\lang-|-\rang_z=1 \\
_z\lang+|-\rang_z= _z\lang-|+\rang_z^*=0
$$
选取 $\hat S_z$ 的这两个本征右失 $|+\rang_z$ 和 $|-\rang_z$作为二维希尔伯特空间的基失。 $\hat S_z$ 在自身表象中是对角矩阵，对角线是本征值：
$$
\hat S_z\to\begin{bmatrix}\hbar/2&0\\0&-\hbar/2\end{bmatrix}=
\frac{\hbar}{2}\begin{bmatrix}1&0\\0&-1\end{bmatrix}
$$
 $\hat S_z$ 的基失在自身表象分别是
$$
|+\rang_z\to\begin{bmatrix}1\\0\end{bmatrix},\quad 
|-\rang_z\to\begin{bmatrix}0\\1\end{bmatrix}
$$
电子的一般自旋态矢量
$$
|x\rang_z=\alpha|+\rang_z+\beta|-\rang_z
$$
归一化要求上式中的复数 $\alpha$ 和 $\beta$ 满足 $|\alpha|^2+|\beta|^2=1$ 。在 $\hat S_z$ 表象
$$
|x\rang_z\to\begin{bmatrix}\alpha\\\beta\end{bmatrix}
$$
(2)  $\hat S_x$ 和 $\hat S_y$ 的本征失
$$
|+\rang_x=\frac{1}{\sqrt{2}}(|+\rang_z+|-\rang_z) \\
|-\rang_x=\frac{1}{\sqrt{2}}(|+\rang_z-|-\rang_z) \\
|+\rang_y=\frac{1}{\sqrt{2}}(|+\rang_z+i|-\rang_z) \\
|-\rang_y=\frac{1}{\sqrt{2}}(|+\rang_z-i|-\rang_z)
$$

## 光子极化

<img src="physics.assets/%E5%85%89%E5%AD%90%E6%9E%81%E5%8C%96.svg" style="zoom:80%;" />

马吕斯定律
$$
I=I_0\cos^2\theta
$$
选择沿 $x,y$ 方向的极化态作为该二维希尔伯特空间的基失，分别为 $|x\rang$ 和 $|y\rang$ ，定义光子极化态为
$$
|\theta\rang=\cos\theta|x\rang+\sin\theta|y\rang \quad (0\leqslant\theta\leqslant\pi)
$$
(1) 从 $P_1$ 出来的光子通过 $P_2$ 的概率幅由右到左写成
$$
\lang P_2|P_1\rang=\lang P_2|x\rang\lang x|P_1\rang+\lang P_2|y\rang\lang y|P_1\rang
$$
此例中
$$
\lang P_2|x\rang=\lang x|x\rang=1 \\
\lang P_2|y\rang=\lang x|y\rang=0
$$
而 $\lang x|P_1\rang=e^{i\alpha}\cos\theta,\quad \lang y|P_1\rang=e^{i\alpha}\sin\theta$ ，共同因子 $e^{i\alpha}$ 略去不写，带入得
$$
\lang P_2|P_1\rang=\cos\theta
$$
通过概率
$$
\mathbb P(\theta)=|\lang P_2|P_1\rang|^2=\cos^2\theta
$$
(2) 去掉 $P_1$ 只留 $P_2$ ，把它叫做 $P$ ，将入射到 $P$ 上的光换为圆偏振光
$$
\lang P|\odot\rang=\lang P|x\rang\lang x|\odot\rang+\lang P|y\rang\lang y|\odot\rang
$$
按照经典光学，圆偏振光可看作振幅分别为 $A_x=A\cos45\degree,A_y=A\sin45\degree$ ，相位差为 $\pm\frac{\pi}{2}$ 的一对垂直线偏振光合成的，因此
$$
\lang x|\odot\rang=e^{i\alpha}\cos45\degree=\frac{1}{\sqrt{2}}e^{i\alpha} \\
\lang y|\odot\rang=e^{i(\alpha\pm\pi/2)}\sin45\degree=\pm\frac{i}{\sqrt{2}}e^{i\alpha}
$$
略去共同因子 $e^{i\alpha}$ 不写，带入得
$$
\lang P|\odot\rang=\frac{1}{\sqrt{2}}(\lang P|x\rang\pm i\lang P|y\rang)
$$
$+,-$ 号分别对应右旋和左旋圆偏振态。

(3) 取 $|x\rang$ 和 $|y\rang$ 为基失，左、右圆偏振态展开为
$$
|R\rang=\frac{1}{\sqrt 2}(|x\rang+i|y\rang) \\
|L\rang=\frac{1}{\sqrt 2}(|x\rang-i|y\rang)
$$
和
$$
\lang R|=\frac{1}{\sqrt 2}(\lang x|-i\lang y|) \\
\lang L|=\frac{1}{\sqrt 2}(\lang x|+i\lang y|)
$$
(4) 光子自旋角动量

<img src="physics.assets/%E5%85%89%E5%AD%90%E8%87%AA%E6%97%8B%E8%A7%92%E5%8A%A8%E9%87%8F.svg" style="zoom:100%;" />

空间旋转操作
$$
|\bar x\rang=\cos\varphi|x\rang+\sin\varphi|y\rang \\
|\bar y\rang=-\sin\varphi|x\rang+\cos\varphi|y\rang \\
|\bar R\rang=\frac{1}{\sqrt 2}(|\bar x\rang+i|\bar y\rang)=|R\rang=\frac{1}{\sqrt 2}e^{-i\varphi}(|x\rang+i|y\rang)
$$
即
$$
|\bar R\rang=e^{-i\varphi}|R\rang
$$
同理
$$
|\bar L\rang=e^{-i\varphi}|L\rang
$$
光子具有动量 $\vec P=\hbar\vec k$ ，其中 $\vec k$ 为波矢。对于某个坐标原点，光子具有轨道角动量
$$
\hat l=\vec r\times \hat P=\hbar \vec r\times \vec k
$$
光子还具有自旋角动量 $\hat S$ ，总的角动量 $\hat j=\hat l+\hat S$ 。$z$ 方向上的分量 $\hat j_z=\hat l_z+\hat S_z$ 。

沿 $\vec k$ 方向轨道角动量份量恒为0，即 $\hat l_k\equiv 0$ ，则 $\hat j_k=\hat S_k$。上述旋转操作是以 $\vec k$ 方向为轴的，其微分算符是与 $\hat j_k$ 或 $\hat S_k$ 对应的。
$$
\hat j_k=\hat S_k=-i\hbar\frac{\partial}{\partial\varphi}
$$
可以看出左右圆偏振态都是它们的本征态，本征值分别为 $\pm\hbar$ 
$$
-i\hbar\frac{\partial}{\partial\varphi}|\bar L\rang=
-i\hbar\frac{\partial}{\partial\varphi}e^{i\varphi}|L\rang=
+\hbar|\bar L\rang
$$
同理
$$
-i\hbar\frac{\partial}{\partial\varphi}|\bar R\rang=-\hbar|\bar R\rang
$$
既然光子的自旋角动量在 $\vec k$ 方向的本征值有 $\pm\hbar$ ，则可推论光子的自旋角量子数 $S=1$ (光子自旋为 1)，即 $\hat S^2/\hbar^2$ 的本征值为 $S(S+1)=2$ 。如果是这样，$\hat S_k$ 还应有一个本征值为 0的本征态，但这个本征态代表纵波，在物理上不能实现。

## 量子共振
量子系统中，固有频率相当于两定态能级 $a,b$ 的间隔
$$
\omega=\frac{E_a-E_b}{\hbar} 
$$
驱动力往往是外加交变电磁场，作用在系统电矩过磁矩上

(1) 经典物理中，一个磁矩在外磁场 $\vec B$ 中作用能
$$
H=-\vec\mu\cdot\vec B
$$
电子处在 $\vec B$ 中，必须考虑它的自旋磁矩与 $\vec B$ 的作用 $H_s$ 。量子力学哈密顿算符应当写成
$$
\hat H=\hat H_0+\hat H_s=\hat H_0-\hat\mu\cdot\hat B
$$
$\hat H_s$ 作用于空间波函数时，总含有对位置坐标的微分，若 $\vec B$ 为空间均匀场，则上式右边两项可以对易，体系状态可以表示为
$$
|\Psi\rang=|\psi\rang|x\rang
$$
其中 $|\psi\rang,|x\rang$ 分别是外部空间状态矢量和自旋空间状态矢量，二者之间省去了一个直积记号 $\otimes$ 。自旋可用 $\sigma_z$ 的基矢展开
$$
|x\rang=\alpha|+\rang_z+|-\beta\rang_z
$$
对应的矩阵形式为 $\begin{bmatrix}\alpha\\\beta\end{bmatrix}$ 。其中复数 $\alpha,\beta$ 满足归一化条件 $|\alpha|^2+|\beta|^2=1$ 

当外部采用坐标表象时，这种表象称为混合表象，波函数为
$$
\Psi=\psi(\vec r,t)\begin{bmatrix}\alpha\\\beta\end{bmatrix}
=\begin{bmatrix}\alpha\psi(\vec r,t)\\\beta\psi(\vec r,t)\end{bmatrix}
$$
归一化可表示为
$$
\lang\Psi|\Psi\rang=\int\mathrm d\tau\begin{bmatrix}\alpha^*\psi^*&\beta^*\psi^*\end{bmatrix}\begin{bmatrix}\alpha\psi\\\beta\psi\end{bmatrix}=(|\alpha|^2+|\beta|^2)\int\mathrm d\tau|\psi|^2=1
$$
当自旋归一化$|\alpha|^2+|\beta|^2=1$ 和空间归一化$\int|\psi|^2\mathrm d\tau=1$分别满足时，上式自动成立。

$|\alpha\psi(\vec r,t)|^2$ 代表$t$时刻，空间$\vec r$ 处，电子自旋 $S_z=\hbar/2(\sigma_z=1)$ 的概率密度；而$|\psi(\vec r,t)|^2$ 代表$t$时刻，空间$\vec r$ 处(不管自旋如何)的概率密度。

混合表象的哈密顿写成矩阵的形式
$$
H=\begin{bmatrix}\hat H&0\\0&\hat H\end{bmatrix}
$$
其中 $\hat H=\hat H_0-\hat\mu\cdot\hat B$

(2) 对于磁场沿$z$轴方向不是均匀场，则哈密顿算符两项不对易，态矢量不能写成上节那样简单的形式。应为
$$
|\Psi\rang=|\psi_+\rang|+\rang_z+|\psi_-\rang|-\rang_z
$$
其中 $|+\rang_z,|-\rang_z$ 分别是电子在斯特恩-格拉赫磁场中，向上运动和向下运动的状态。

取$|\psi_+\rang,|\psi_-\rang_z$ 的坐标表象，自旋取 $\sigma_z$ 表象。混合表象有
$$
\Psi=\psi_+(\vec r,t)\begin{bmatrix}1\\0\end{bmatrix}+
\psi_-(\vec r,t)\begin{bmatrix}0\\1\end{bmatrix}
=\begin{bmatrix}\psi_+(\vec r,t)\\\psi_-(\vec r,t)\end{bmatrix}
$$
归一化要求
$$
\lang\Psi|\Psi\rang=\int\mathrm d\tau(|\psi_+|^2+|\psi_-|^2)=1
$$
薛定谔方程为
$$
i\hbar\frac{\partial}{\partial t}\begin{bmatrix}\psi_+\\\psi_-\end{bmatrix}
=\begin{bmatrix}\hat H&0\\0&\hat H\end{bmatrix}
\begin{bmatrix}\psi_+\\\psi_-\end{bmatrix}
$$

## 拉莫尔进动

设均匀磁场方向为$z$轴方向 $\vec B=\vec e_zB_0$。哈密顿算符和自旋有关的部分
$$
\hat H_s=-\vec\mu_s\cdot\vec B=\frac{e}{m_e}\hat S_zB_0=\frac{e\hbar}{2m_e}\hat\sigma_zB_0
$$
是自旋磁矩与外加均匀磁场作用能。混合表象
$$
H=\frac{e\hbar}{2m_e}\begin{bmatrix}B_0&0\\0&-B_0\end{bmatrix}
$$
的对角元即是相互作用的能量本征值。定义
$$
E_{\pm}=\pm\frac{e\hbar}{2m_e}B_0=\pm\mu_BB_0\equiv\pm\frac{\hbar}{2}\omega_0
$$
其中 $\omega_0\equiv\dfrac{e}{m_e}B_0$ 称为Larmor频率。

若电子自旋初态为
$$
|\psi(0)\rang=\alpha|+\rang+\beta|-\rang
$$
在外磁场作用下，$t$ 时刻状态为
$$
|\psi(t)\rang=\alpha\exp(-\frac{1}{2}\omega_0t)|+\rang+\beta\exp(\frac{i}{2}\omega_0t)|-\rang
$$
为了理解它的物理意义，在 $S_z$ 表象考察自旋磁矩的平均值
$$
\begin{aligned}\lang\mu_x\rang&=\lang\psi(t)|-\frac{e}{m_e}S_x|\psi(t)\rang \\
&=-\frac{e}{m_e}\begin{bmatrix}\alpha^*e^{i\omega_0t/2}&\beta^*e^{-i\omega_0t/2}\end{bmatrix}
\frac{\hbar}{2}\begin{bmatrix}0&1\\1&0\end{bmatrix}
\begin{bmatrix}\alpha e^{-i\omega_0t/2}\\\beta e^{i\omega_0t/2}\end{bmatrix} \\
&=-\frac{e\hbar}{2m_e}(\alpha^*\beta e^{i\omega_0t}+\alpha\beta^* e^{-i\omega_0t}) \\
&=-\frac{e\hbar}{2m_e}(|\alpha^*\beta|e^{i\phi}e^{i\omega_0t}+|\alpha^*\beta| e^{-i\phi}e^{-i\omega_0t}) \\
&=-\frac{e\hbar}{2m_e}\cdot 2A\cos(\omega_0t+\phi) \\
&=-\mu_B\cdot 2A\cos(\omega_0t+\phi)
\end{aligned}
$$
类似

$$
\lang\mu_x\rang=-\mu_B\cdot 2A\cos(\omega_0t+\phi) \\
\lang\mu_y\rang=-\mu_B\cdot 2A\sin(\omega_0t+\phi) \\
\lang\mu_z\rang=-\mu_B\cdot (|\alpha|^2-|\beta|^2)
$$
以上三式中用到复数的指数表示：$\alpha^*\beta=|\alpha^*\beta|e^{i\phi}=Ae^{i\phi}$

电子自旋算符的平均值与以上三式形式完全类似，只要 $\mu_B$ 换成 $\hbar/2$ 就行了
$$
\lang S_x\rang=-A\hbar\cos(\omega_0t+\phi) \\
\lang S_y\rang=-A\hbar\sin(\omega_0t+\phi) \\
\lang S_z\rang=-\frac{\hbar}{2}(|\alpha|^2-|\beta|^2)
$$
这些结果表明，电子自旋在均匀磁场中经 $z$ 轴(磁场方向)进动，进动频率 $\omega_0=\dfrac{e}{m_e}B_0$  ，磁场越强，进动频率越高。

## 磁共振

除了$z$轴方向的均匀磁场 $\vec B_0=\vec e_zB_0$外，在 x-y 平面内施加随时间变化的磁场
$$
\vec B_1=\vec e_xB_1\cos\omega t+\vec e_yB_1\sin\omega t
$$
它是在 x-y 平面内的旋转磁场，旋转频率为 $\omega$ ，但是不随空间位置变化。与自旋有关的哈密顿
$$
\hat H_s=-\vec\mu\cdot\vec B=-\vec\mu_B B_0\hat\sigma_z-\vec\mu_BB_1\hat\sigma_x\cos\omega t-\vec\mu_BB_1\hat\sigma_y\sin\omega t
$$
定义
$$
\vec\mu_B B_0=\frac{1}{2}\hbar\omega_0,\quad \vec\mu_B B_1=\frac{1}{2}\hbar\omega_1
$$
则在 $\sigma_z$ 表象
$$
H=\frac{\hbar}{2}\begin{bmatrix}\omega_0&\omega_1e^{-i\omega t}\\
\omega_1e^{i\omega t}&-\omega_0\end{bmatrix}
$$
对于含时哈密顿，我们通过解薛定谔方程，确定电子自旋状态。设
$$
|\psi(t)\rang=a_+(t)|+\rang_z+a_-(t)|-\rang_z
$$
在 $\sigma_z$ 表象的薛定谔波方程
$$
i\hbar\frac{\partial}{\partial t}\begin{bmatrix}a_+\\a_-\end{bmatrix}=
\frac{\hbar}{2}\begin{bmatrix}\omega_0&\omega_1e^{-i\omega t}\\
\omega_1e^{i\omega t}&-\omega_0\end{bmatrix}
\begin{bmatrix}a_+\\a_-\end{bmatrix}
$$
引入变换
$$
a_+(t)=b_+(t)e^{-i\omega t/2} \\
a_-(t)=b_-(t)e^{+i\omega t/2}
$$
对时间求导得到
$$
\frac{\mathrm d}{\mathrm dt}ib_+=-\frac{\omega-\omega_0}{2}b_++\frac{\omega_1}{2}b_- \\
\frac{\mathrm d}{\mathrm dt}ib_-=\frac{\omega_1}{2}b_++\frac{\omega-\omega_0}{2}b_-
$$
对上两式再对时间求导，得
$$
\frac{\mathrm d^2}{\mathrm dt^2}b_++(\frac{\Omega}{2})^2b_+=0 \\
\frac{\mathrm d^2}{\mathrm dt^2}b_-+(\frac{\Omega}{2})^2b_-=0
$$
其中 $\Omega^2=(\omega-\omega_0)^2+\omega_1^2$ 。

当初始条件 $a_-(0)=b_-(0)=0$ 时，即 $t=0$ 时，自旋处在 $|+\rang_z$ 态，则
$$
b_-=-i\frac{\omega_1}{\Omega}\sin(\frac{\Omega}{2}t) \\
b_+=\cos(\frac{\Omega}{2}t)+i\frac{\omega-\omega_0}{\Omega}\sin(\frac{\Omega}{2}t)
$$
在 $t$ 时刻，测量自旋 $z$ 分量，得到 $S_z=-\hbar/2$ 的概率为
$$
\mathbb P_{+\to -}=|_z\lang-|\psi(t)\rang|^2=|a_-(t)|^2=|b_-(t)|^2=
(\frac{\omega_1}{\Omega})^2\sin^2(\frac{\Omega}{2}t)
$$
问题涉及三种圆频率：$\omega_0,\omega_1,\omega$ 。前两者分别由磁场 $\vec B_0$ 和 $\vec B_1$ 大小决定，以下分三种情况讨论磁场强弱带来的物理后果。

1. 当 $|\omega-\omega_0|\gg\omega_1$ ，即 $\Omega\gg\omega_1$ ，任何时刻 $\mathbb P_{+\to -}\ll 1$ ，即任何时刻测得 $S_z=-\hbar/2$ 的概率很小；
2. 旋转磁场的频率接近拉莫尔频率 $\omega\sim\omega_0$ ，那么 $\Omega\sim\omega_1$， $\mathbb P_{+\to -}$ 在0和1之间振荡。
   当 $t=(2n+1)\pi/\omega_1$ 时，$\mathbb P_{+\to -}=1$ ，自旋翻转，$\omega_1$ 可以很小，即引起自旋翻转的场可以很弱；
3. 当 $|\omega-\omega_0|\sim\omega_1$ ，则 $\mathbb P_{+\to -}\sim\frac{1}{2}\sin^2(\frac{\Omega}{2}t)$；

## 拉比实验

测量质子磁矩

<img src="physics.assets/%E6%8B%89%E6%AF%94%E5%AE%9E%E9%AA%8C.svg" style="zoom:100%;" />

在磁场存在区，改变恒定均匀磁场  $\vec B_0$ 的大小(即改变拉莫尔频率$\omega_0$) ，当 $\omega_0\sim\omega$ 时，在某时刻，磁矩会突然翻转，经过第二个 Stern-Gerlach 偏转器后，粒子将向背离探测器方向偏转，这是探测器 D 得计数突然减少，这就给出了 $\dfrac{|\mu|}{j}=\dfrac{\hbar\omega_0}{B_0}$ 的值，角动量量子数 $j$ 可借助光学或核谱实验确定。

<img src="physics.assets/%E6%8B%89%E6%AF%94%E5%AE%9E%E9%AA%8C-2.png" style="zoom:30%;" />

Stern-Gerlach 实验

<img src="physics.assets/Stern-Gerlach%E5%AE%9E%E9%AA%8C.svg" style="zoom:80%;" />

银原子的磁矩矢量 $\vec\mu$ 将绕磁场方向进动。实验银原子 $\vec\mu$ 大小相同，当方向是随机分布的。平均来看，由于进动的存在，$\mu_x$ 和 $\mu_y$ 的时间平均值为0，只考虑沿磁场方向的 $\mu_z:\ W=-\mu_zB$ 

## 纠缠态

贝尔基失
$$
|\varphi^+\rang=\frac{1}{\sqrt{2}}(|++\rang_z+|--\rang_z) \\
|\varphi^-\rang=\frac{1}{\sqrt{2}}(|++\rang_z-|--\rang_z) \\
|\psi^+\rang=\frac{1}{\sqrt{2}}(|+-\rang_z+|-+\rang_z) \\
|\psi^-\rang=\frac{1}{\sqrt{2}}(|+-\rang_z-|-+\rang_z) 
$$
这四个纠缠态构成四维希尔伯特空间的一组基失。

贝尔不等式：贝尔采用玻姆和阿哈罗诺夫关于隐变量讨论的粒子：一对自旋1/2的粒子处于自旋单态：
$$
\psi_0=\frac{1}{\sqrt 2}(|+\rang_1|-\rang_2-|-\rang_1|+\rang_2)
$$
这两个粒子向相反方向飞去，可用 Stern-Gerlach 磁场测量每个粒子的自旋 $\hat\sigma_1$ 和 $\hat\sigma_2$ 的任何分量，如果测得 $\hat\sigma_1\cdot\vec a$  ($\vec a$ 为空间某单位矢量)得到本征值+1 ，根据量子力学，测得 $\hat\sigma_2\cdot\vec a$ 必然得到-1。

当两粒子相距很远时，它们之间无任何相互作用，这符合EPR定域相互作用思想。为了构造EPR完全性理论，假设存在隐变量 $\lambda$ ，测量 $\hat\sigma_1\cdot\vec a$ 的结果 $A$ 是由 $\vec a$ 和 $\lambda$ 确定，而同一时刻 $\hat\sigma_2\cdot\vec b$ 的结果 $B$ 是由 $\vec b$ 和 $\lambda$ 确定：
$$
A(\vec a,\lambda)=\pm 1,\quad B(\vec b,\lambda)=\pm 1
$$
上两式取正号或负号分别由 $(\vec a,\lambda)$ 和 $(\vec b,\lambda)$ 决定。

这里所做的重要假设为：粒子2的结果不依赖于 $\vec a$ ，即不依赖于测量粒子1用的 Stern-Gerlach磁场方向，反之也一样。隐变量 $\lambda$ 是怎样的数学对象我们不清楚，不妨把它当成单值连续参数，处理它的分布 $\rho(\lambda)$ 满足归一化条件：
$$
\int\rho(\lambda)\mathrm d\lambda=1
$$
两个分量 $\hat\sigma_1\cdot\vec a$  和  $\hat\sigma_2\cdot\vec b$  的乘积的期望值是
$$
\mathbb P(\vec a,\vec b)=\int\rho(\lambda)A(\vec a,\lambda)B(\vec b,\lambda)\mathrm d\lambda
$$
量子力学的期望值为 $\lang\hat\sigma_1\cdot\vec a,\hat\sigma_2\cdot\vec b\rang=-\vec a\cdot\vec b$ 。证明从略。

对于假设的系统测量 $A(\vec a,\lambda)=-B(\vec a,\lambda)$ ，所以
$$
\mathbb P(\vec a,\vec b)=-\int\rho(\lambda)A(\vec a,\lambda)A(\vec b,\lambda)\mathrm d\lambda
$$
如果 $\vec c$ 为空间另一单位向量
$$
\mathbb P(\vec a,\vec c)=-\int\rho(\lambda)A(\vec a,\lambda)A(\vec c,\lambda)\mathrm d\lambda
$$
所以
$$
\mathbb P(\vec a,\vec b)-\mathbb P(\vec a,\vec c)=\int\rho(\lambda)A(\vec a,\lambda)A(\vec b,\lambda)[A(\vec b,\lambda)A(\vec c,\lambda)-1]\mathrm d\lambda
$$
上式用到 $A(\vec b,\lambda)A(\vec b,\lambda)=1$ ，注意到 $|A(\vec a,\lambda)A(\vec a,\lambda)|=1$ 则
$$
\begin{aligned}
|\mathbb P(\vec a,\vec b)-\mathbb P(\vec a,\vec c)|&=
|\int\rho(\lambda)A(\vec a,\lambda)A(\vec b,\lambda)[A(\vec b,\lambda)A(\vec c,\lambda)-1]\mathrm d\lambda| \\
&\leqslant \int\mathrm d\lambda\rho(\lambda)|A(\vec a,\lambda)A(\vec b,\lambda)[A(\vec b,\lambda)A(\vec c,\lambda)-1]| \\
&=\int\mathrm d\lambda\rho(\lambda)|1-A(\vec b,\lambda)A(\vec c,\lambda)|\\
&=1+\mathbb P(\vec b,\vec c)
\end{aligned}
$$
即
$$
|\mathbb P(\vec a,\vec b)-\mathbb P(\vec a,\vec c)|\leqslant1+\mathbb P(\vec b,\vec c)
$$
上式即为贝尔不等式。量子力学的理论结果违反贝尔不等式。

