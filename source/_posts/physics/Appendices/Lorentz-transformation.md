---
title: 固有洛伦兹变换的严格推导
categories:
  - Physics
  - Appendices
tags:
  - 物理
  - 相对论
  - 洛伦兹变换
description: 根据狭义相对论的两条基本原理来严格推导固有洛伦兹变换
katex: true
abbrlink: 55b8eadb
date: 2022-07-24 18:41:28
top_img: /img/韦伯太空望远镜.webp
cover: /img/Superstar/Lorentz-cover.png
---

在本节中采用四维实坐标。设有两个惯性系$S$和$S'$，同一事件$P$在其中的坐标分别用$(x_0,x_1,x_2,x_3)$和$(x_0',x_1',x_2',x_3')$表示。规定第0坐标是时间分量，即
$$
x_0=ct_0,\quad x_0'=ct_0'
$$
规定希腊字母 $\mu,\nu,\cdots\in\{0,1,2,3\}$，拉了字母 $i,j,\cdots\in\{1,2,3\}$，并采用爱因斯坦求和约定[^Einstein_notation]。

[^Einstein_notation]: 所谓Einstein求和约定就是略去求和式中的求和号。在此规则中两个相同指标就表示求和，而不管指标是什么字母，有时亦称求和的指标为哑指标。例如：$\displaystyle a_ib_i=\sum_{i=1}^3a_ib_i,\quad a_\mu b_\mu=\sum_{i=0}^3a_\mu b_\mu$

# 狭义相对论基本原理

狭义相对论原理之一是一切惯性系均等效。这就是说，变换不改变惯性运动的性质，在$S$系中以速度$u_i$做匀速直线运动的粒子，在$S'$中观测必定是速度为$u_i'$的匀速直线运动，即匀速直线运动形式不变。
$$
\begin{cases}
x_i=x_{i0}+u_i(t-t_0) \\
x_i'=x_{i0}'+u_i'(t'-t_0')
\end{cases}\tag{1}
$$
原理之二是真空中光速恒为 $c$
$$
\begin{cases}
u_iu_i=c^2 \\
u_i'u_i'=c^2
\end{cases}\tag{2}
$$
引入中间变量
$$
\begin{cases}
\beta_\mu=\beta_0\cfrac{u_\mu}{c} &(u_0=c) \\
S=\cfrac{c}{\beta_0}(t-t_0)
\end{cases}\tag{3}
$$
则条件(1)式变为
$$
\begin{cases}
x_\mu=\xi_\mu+\beta_\mu S \\
x_\mu'=\xi'_\mu+\beta_\mu' S'
\end{cases}\tag{4}
$$
当$\mu=0$时为恒等式，其中$\xi_\mu,\xi'_\mu$是$x_\mu,x_\mu'$的初始值。
$$
\xi_\mu=(ct_0,x_{i0}),\quad \xi'_\mu=(ct_0',x_{i0}')
$$
故 $x_\mu,\xi_\mu;x_\mu',\xi'_\mu$是相互独立无关的，条件(2)式变为
$$
\begin{cases}
\eta_{\mu\nu}\beta_\mu\beta_\nu=0  \\
\eta_{\mu\nu}'\beta_\mu'\beta_\nu'=0
\end{cases}\tag{5}
$$
其中$\eta_{\mu\nu}$是由A式定义的闵可夫斯基度规。

我们假设所求惯性系的变换是
$$
x'_\mu=f_\mu(x_0,x_1,x_2,x_3)\tag{6}
$$
$f_\mu$是待求的函数，它要符合条件(1)(2)或(4)(5)的要求。下面分四步求解。

# 考虑惯性系条件对变换的限制

由惯性系的等价性可知变换(6)的逆变换应当唯一确定，其充要条件是雅可比行列成不等于零
$$
\det(\cfrac{\partial f_\nu}{\partial x_\mu})\neq 0 \tag{7}
$$
由(4)式的等一式得
$$
\tag{8}\cfrac{\mathrm df_\mu}{\mathrm dS}=\cfrac{\mathrm dx_\nu}{\mathrm dS}\cfrac{\partial f_\mu}{\partial x_\nu}=\beta_\nu\cfrac{\partial f_\mu}{\partial x_\nu}
$$
由(4)式的等二式得
$$
\mathrm df_\mu=\mathrm dx'_\mu=\beta'_\mu\mathrm dS'=\cfrac{\beta'_\mu}{\beta'_0}\mathrm df_0\\
\cfrac{\mathrm df_\mu}{\mathrm df_0}=\cfrac{\beta'_\mu}{\beta'_0}=\cfrac{u_\mu'}{c}=\text{const}
$$
利用(8)式将上式变形为
$$
\tag{9}\cfrac{\beta'_\mu}{\beta'_0}=\cfrac{\mathrm df_\mu/\mathrm dS}{\mathrm df_0/\mathrm dS}=\cfrac{\beta_\nu\cfrac{\partial f_\mu}{\partial x_\nu}}{\beta_\sigma\cfrac{\partial f_0}{\partial x_\sigma}}=\text{const}
$$
求对数后再求导得
$$
\cfrac{\mathrm d}{\mathrm dS}[\ln({\beta_\nu\cfrac{\partial f_\mu}{\partial x_\nu}})-\ln(\beta_\sigma\cfrac{\partial f_0}{\partial x_\sigma})]=0
$$
最会得到
$$
\tag{10}\cfrac{\beta_\nu\beta_\sigma\cfrac{\partial^2 f_\mu}{\partial x_\nu\partial x_\sigma}}{\beta_\nu\cfrac{\partial f_\mu}{\partial x_\nu}}=\cfrac{\beta_\nu\beta_\sigma\cfrac{\partial^2 f_0}{\partial x_\nu\partial x_\sigma}}{\beta_\nu\cfrac{\partial f_0}{\partial x_\nu}}
$$
(10)式共有4个等式，它们对独立变量$x_\nu,\beta_\nu$而言为恒等式，所以应当是$\beta_\nu$的有理分式，由(7)式知，联立方程组
$$
\beta_\nu\cfrac{\partial f_\mu}{\partial x_\nu}=0\quad (\mu=0,1,2,3)
$$
对于$\beta_\nu$而言仅有零解，但是由于$\beta_0>0$，所以$\beta_\nu$不能全为0，亦即(10)式的4个分母$\beta_\nu\cfrac{\partial f_\mu}{\partial x_\nu}$不能同时为0，假如某分母为零，由(10)式知此分式仍然有限，所以该分子必然同时为零，这仅当分母是分子的因式时才成立。

总之，(10)式中4个分式实际上是$\beta_\nu$的有理整式，令此有理考式等于$2\beta_\nu\psi_\nu$，其中 $\psi_\nu=\psi_\nu(x_0,x_1,x_2,x_3)$，则(10)式化为
$$
\beta_\nu\beta_\sigma\cfrac{\partial^2 f_\mu}{\partial x_\nu\partial x_\sigma}=2\beta_\nu\cfrac{\partial f_\mu}{\partial x_\nu}\beta_\sigma\psi_\sigma
$$
上式对$\beta_\nu\beta_\sigma$而言是恒等式，故对某一对$(\nu,\sigma)$有
$$
\begin{aligned}
\beta_\nu\beta_\sigma\cfrac{\partial^2 f_\mu}{\partial x_\nu\partial x_\sigma}
&=\beta_\nu\cfrac{\partial f_\mu}{\partial x_\nu}\beta_\sigma\psi_\sigma
+\beta_\sigma\cfrac{\partial f_\mu}{\partial x_\sigma}\beta_\nu\psi_\nu \\
&=\beta_\nu\beta_\sigma(\cfrac{\partial f_\mu}{\partial x_\nu}\psi_\sigma+\cfrac{\partial f_\mu}{\partial x_\sigma}\psi_\nu)
\end{aligned}
$$
式中重复指标不求和，最后得到
$$
\tag{11}\cfrac{\partial^2 f_\mu}{\partial x_\nu\partial x_\sigma}=\cfrac{\partial f_\mu}{\partial x_\nu}\psi_\sigma+\cfrac{\partial f_\mu}{\partial x_\sigma}\psi_\nu
$$
上式说明，若匀速直线运动在变换下保持不变，则函数$f_\mu$应当满足此偏微分方程，但不一定是线性变换。

# 考虑光速不变性对变换的限制

利用(9)式可把(5)式的第二式化为
$$
\eta_{\mu\nu}\beta'_{\mu}\beta'_\nu=\eta_{\mu\nu}\beta_{\sigma}\beta_\lambda
\cfrac{\partial f_\mu}{\partial x_\sigma}\cfrac{\partial f_\nu}{\partial x_\lambda}=0
$$


由(5)式的第一式，可知上式也应当等于
$$
\eta_{\sigma\lambda}\beta_\sigma\beta_\lambda=0
$$
故上两式中$\beta_\nu\beta_\sigma$的二次式系数应成正比，令其为某一函数 $\lambda(x_0,x_1,x_2,x_3)$ ，则得
$$
\eta_{\mu\nu}\cfrac{\partial f_\mu}{\partial x_\alpha}\cfrac{\partial f_\nu}{\partial x_\beta}=\lambda\eta_{\alpha\beta}\tag{12}
$$
(12)式对$x_\rho$微分并令$\cfrac{\partial\lambda}{\partial x_\rho}=2\lambda\varphi_\rho$，则有
$$
\eta_{\mu\nu}(\cfrac{\partial^2 f_\mu}{\partial x_\rho\partial x_\alpha}\cfrac{\partial f_\nu}{\partial x_\beta}+\cfrac{\partial^2 f_\mu}{\partial x_\rho\partial x_\beta}\cfrac{\partial f_\nu}{\partial x_\alpha}
)=2\lambda\eta_{\alpha\beta}\varphi_\rho
$$
以(11)式取代上式中的二阶导数，再利用(12)式得
$$
2\eta_{\alpha\beta}\psi_\rho+\eta_{\rho\alpha}\psi_\beta+\eta_{\rho\beta}\psi_\alpha=2\eta_{\alpha\beta}\varphi_\rho
$$
上式对任意$\alpha,\beta,\rho$均恒等，令$\rho\neq\alpha,\rho\neq\beta\quad (\eta_{\rho\alpha}=\eta_{\rho\beta}=0)$得
$$
\psi_\rho=\varphi_\rho\quad(\rho=0,1,2,3)
$$
再令$\rho=\alpha=\beta$，得 $\psi_\rho=\varphi_\rho$，联立上式可知
$$
\psi_\rho=\varphi_\rho=0\quad(\rho=0,1,2,3)
$$
以及
$$
\cfrac{\partial\lambda}{\partial x_\rho}=2\lambda\varphi_\rho=0,\quad\lambda=\text{const}
$$
于是 (11) 式化为
$$
\cfrac{\partial^2 f_\mu}{\partial x_\alpha\partial x_\beta}=0\tag{13}
$$
此即变换函数所应满足的条件，即**线性条件**。
所以，上而两式证明了惯性系之间满足狭义相对的原理(1)(2)要求的时空坐标变换一定是线性变换。

# 确定线性变换的形式

由(12)式知道，$f_\mu$中应含有因子$\sqrt{\lambda}$，令线性变换为
$$
x'_\mu =f_\mu=\sqrt{\lambda}(a_\mu+\eta_{\nu\nu}a_{\mu\nu}x_\nu)\tag{14}
$$
这里及以下的$\eta_{\nu\nu}$只是一个符号，即
$$
\eta_{00}=1,\quad\eta_{11}=\eta_{22}=\eta_{33}=-1
$$
引入该符号可以使得表述简洁，将(14)式代入(12)式得
$$
\begin{cases}
\eta_{\mu\nu}a_{\mu\alpha}a_{\nu\beta}=\eta_{\alpha\beta} \\
\eta_{\mu\nu}a_{\alpha\mu}a_{\beta\nu}=\eta_{\alpha\beta}
\end{cases}\tag{15}
$$
此即$a_{\mu\nu}$应满足的正交条件。

利用正交条件由(14)式解出
$$
x_\mu=\eta_{\nu\nu}a_{\nu\mu}(\cfrac{x'_\nu}{\sqrt \lambda}-a_\nu)\tag{16}
$$
从$S$系看$S'$系中的固定点 $\mathrm dx'_i=0$以速度$v_i$运动，由 (16) 式得
$$
\mathrm dx_i=\cfrac{1}{\sqrt\lambda}a_{0i}\mathrm dx'_0,\quad \mathrm dx_0=\cfrac{1}{\sqrt\lambda}a_{00}\mathrm dx'_0
$$
所以
$$
\cfrac{v_i}{c}=\cfrac{\mathrm dx_i}{\mathrm dx_0}=\cfrac{a_{0i}}{a_{00}}
$$
上式表明示惯性系$S$和$S'$之间的相对速度$\lambda$无关。如令$v_i=0$ ，即$S$和$S'$重合而无运动。但是利用正交条件(15)式，由(14)式可得
$$
\eta_{\mu\nu}\mathrm dx'_\mu\mathrm dx'_\nu=\lambda\eta_{\alpha\beta}\mathrm dx_\alpha\mathrm dx_\beta\tag{17}
$$
令 $\mathrm dx'_i=0$，则有
$$
\mathrm dx_0^{'2}=\lambda\mathrm dx_0^2(1-\cfrac{v^2}{c^2})
$$
当 $v=0$ 时，应该 $\mathrm dx'_0=\mathrm dx'_0$ ，故系数 $\lambda=1$。于是 (17) 式表示为 $\mathrm dx_\mu$ 得二次齐式
$$
\mathrm ds^2=\eta_{\mu\nu}\mathrm dx_\mu\mathrm dx_\nu=\eta_{\alpha\beta}\mathrm dx'_\alpha\mathrm dx'_\beta\tag{18}
$$
这就是时空间隔不变性，正好是闵可夫斯基空间时空度规的显示式。将$\lambda=1$代入(14)和(16)式，就成为
$$
\begin{cases}
x'_\mu =a_\mu+\eta_{\nu\nu}a_{\mu\nu}x_\nu \\
x_\mu=\eta_{\nu\nu}a_{\nu\mu}(x'_\nu-a_\nu)
\end{cases}\tag{19}
$$
(19) 式就是惯性系间的一般洛伦兹变换。

至此我们严格证明了：惯性系之间所容许的时空坐标变换为一般洛伦兹变换；采用实数时间生标时，闵可夫斯基空间的度规为伪欧几里得空间度规。

# 根据正交归一化条件确定变换系数

先令 $t=0$ 时 $t'=0$，则原点 $O$ 与 $O'$ 重合，则 $a_\mu=0$ （如果是坐标微分变换 $\mathrm dx_\mu \to \mathrm dx'_\mu$，这一条件可取消），于是 (19)式化为
$$
\begin{cases}
x'_\mu =\eta_{\nu\nu}a_{\mu\nu}x_\nu \\
x_\mu=\eta_{\nu\nu}a_{\nu\mu}x'_\nu
\end{cases}\tag{20}
$$
再设 $S'$ 系相对于$S$系以速度$v_i$做惯性运动，则从$S$系观测，$S'$系的固定点（$\mathrm dx'_i=0$）的速度为 $v_i$；又在$S'$系观测，$S$系的固定点（$\mathrm dx'_i=0$）以速度$v'_i$运动，故从上式得
$$
\begin{cases}
a_{00}v_i=a_{0i}c \\
a_{00}v'_i=a_{i0}c
\end{cases}\tag{21}
$$
另外，根据实践我们引入单向顺时性条件，此条件要求洛伦兹变换不改变时间进程的方句，即
$$
a_{00}=\cfrac{\partial t'}{\partial t}>0\tag{22}
$$
根据上面两式和正交规一化条件(15)式即可确定变换系数。
在(15)式中令 $\alpha=\beta=0$ 得
$$
\begin{cases}
a_{00}^2-(a_{10}^2+a_{20}^2+a_{30}^2)=1 \\
a_{00}^2-(a_{01}^2+a_{02}^2+a_{03}^2)=1
\end{cases}\tag{23}
$$
将(21)式代入(23)式可得
$$
v'_iv'_i=v_iv_i=v^2 \tag{24}
$$
尽管两坐标系相对速度的分量不同，但速度的大小是相等的，这一点和直观相符合，考虑到条件(22)式，由(21)和(23)两式可以解出变换系数为
$$
\begin{cases}
a_{00}=\cfrac{1}{\sqrt{1-\cfrac{v^2}{c^2}}}=\gamma \\
a_{0i}=\gamma\cfrac{v_i}{c} \\
a_{i0}=\gamma\cfrac{v'_i}{c}
\end{cases}\tag{25}
$$
当(15)式中的$\alpha,\beta$ 一个为0时，利用(25)式中的 $a_{0i},a_{i0}$ 推导出
$$
\tag{26}\begin{cases}
a_{00}v'_i=a_{ik}v_{k} \\
a_{00}v_i=a_{ki}v'_{k}
\end{cases}
$$
当(15)式中的$\alpha,\beta$ 均不为0时，又得
$$
\tag{27}a_{ki}a_{kj}=-\eta_{ij}+a_{0i}a_{0j}=\delta_{ij}+\gamma^2\cfrac{v_iv_j}{c^2}
$$
定义
$$
\tag{28}d_{ik}=-a_{ik}+\cfrac{a_{i0}a_{0k}}{a_{00}+1}=-a_{ik}+(\gamma-1)\cfrac{v'_iv_k}{v^2}
$$
根据上面三式得到
$$
\begin{cases}
d_{ik}v_k=-v'_i \\
d_{ik}v'_i=-v_k \\
d_{ki}d_{kj}=\delta_{ij}
\end{cases}\tag{29}
$$
故所定义的 $d_{ik}$ 乃是三维空间的正交变换矩阵，如果坐标系对应平行，则 $d_{ij}=\delta_{ij}$，(29)式的第一式和第二式成为
$$
\delta_{ik}v_k=v_i=-v'_i,\quad \delta_{ik}v'_i=v'_k=-v_k
$$
综合(25)(26)(28)式，我们终于求得一般固有洛伦兹变换的系数为
$$
\begin{cases}
a_{00}=\gamma \\
a_{0i}=\gamma\cfrac{v_i}{c} \\
a_{i0}=-\gamma\cfrac{d_{ij}v_i}{c} \\
a_{ik}=-d_{ik}-(\gamma-1)\cfrac{d_{ij}v_jv_k}{v^2}
\end{cases}\tag{30}
$$

# 固有洛伦兹变换

**讨论如下：**

(1) 如果 $S$ 和 $S'$ 系的坐标轴对应平行，即不存在空间转动或反射，这时的正交变换矩阵为单位矩阵 （ $d_{ij}=\delta_{ij}$），再设坐标系的相对速度沿 $x_1(x'_1)$ 轴方向，即 $v_1=v,v_2=v_3=0$，对变换系数简化为（记作$\bar a$）
$$
\begin{cases}
\bar a_{00}=-\bar a_{11}=\gamma \\
\bar a_{01}=-\bar a_{10}=\gamma\cfrac{v}{c} \\
\bar a_{22}=\bar a_{33}=-1
\end{cases}\tag{31}
$$
其余变换系数为0。带入 (20) 式即得到特殊洛伦兹变换式。

(2) 如果 $S$ 和 $S'$ 系的坐标轴对应平行（ $d_{ij}=\delta_{ij}$），但坐标系的相对速度沿任意方向，则变换系数简化为（记作$\bar a$）
$$
\begin{cases}
\bar a_{00}=\gamma \\
\bar a_{0i}=-\bar a_{i0}=\gamma\cfrac{v_i}{c} \\
a_{ik}=-\delta_{ik}-(\gamma-1)\cfrac{v_iv_k}{v^2}
\end{cases}\tag{32}
$$
带入 (20) 式即得无空间转动的固有洛伦兹变换式。
$$
\tag{33}\begin{cases}
\bar x_k=x_k+v_k[(\gamma-1)\cfrac{v_ix_i}{v^2}-\gamma t] \\
\bar t=\gamma(t-\cfrac{v_ix_i}{c^2})
\end{cases}
$$
(3) 如果 $S$ 和 $S'$ 系的坐标轴不是对应平行，可对上式的空间坐标再作一次三维空间转动或反射变换 $dij$
$$
\tag{34}\begin{aligned}
x'_i&=d_{ik}\bar x_k=d_{ik}x_k+d_{ik}v_k[(\gamma-1)\cfrac{v_ix_i}{v^2}-\gamma t] \\
&=d_{ik}x_k+v'_i[(\gamma-1)\cfrac{v_ix_i}{v^2}-\gamma t]
\end{aligned}
$$
其中利用了(29)式，事实上，因为
$$
x'_i=d_{ik}\bar x_k=d_{ik}(\eta_{\nu\nu}\bar a_{k\nu}x_\nu)=\eta_{\nu\nu}a_{i\nu}x_\nu
$$
其中 $\bar a_{k\nu}$ 由(32)式给出，$a_{i\nu}=d_{ik}\bar a_{k\nu}$即为(30)式。这个过程相当于分两步进行
$$
S(x_\mu)\to\bar S(\bar x_\mu)\to S'(x'_\mu)
$$
$S\to\bar S$代表空间坐标轴无转动且无反射的洛伦兹变换；$\bar S\to S'$代表纯空间轴的转动或反射变换，令 $D$ 代表正交变换矩阵 $d_{ik}$，则一般固有洛伦兹变换为
$$
\begin{cases}
\mathbf x=D\mathbf x-\mathbf v'[(\gamma-1)\cfrac{\mathbf{v\cdot x}}{v^2}-\gamma t] \\
t'=\gamma(t-\cfrac{\mathbf{v\cdot x}}{c^2})
\end{cases}\tag{34}
$$

# 附录

(1) 欧几里得空间的任意一点 $A(x_1,x_2,x_3)$ 相对于原点的失径为
$$
\mathbf x=x_1\mathbf e_1+x_2\mathbf e_2+x_3\mathbf e_3=x_i\mathbf e_i
$$
式中 $(\mathbf e_1,\mathbf e_2,\mathbf e_3)$ 是沿坐标轴的单位矢量，称作基矢。对上式两边微分（注意在笛卡尔坐标系中 $\mathrm d\mathbf e_i=0$ ），可知
$$
\mathrm d\mathbf x=\cfrac{\partial\mathbf x}{\partial x_i}\mathrm dx_i =\mathrm dx_i\mathbf e_i
$$
所以基矢实际上就是矢径对坐标的偏导
$$
\mathbf e_i=\cfrac{\partial\mathbf x}{\partial x_i}\quad (i=1,2,3)
$$
根据偏导的几何意义，可知笛卡尔基矢的内积满足
$$
\mathbf e_i\cdot \mathbf e_j=\delta_{ij}=\begin{cases}
1 &(i=j) \\
0 &(i\neq j)
\end{cases}
$$
$\delta_{ij}$ 是克罗内克符号，它的9个分量构成 $3\times 3$ 单位矩阵
$$
\delta_{ij}=\begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 
\end{pmatrix}
$$
设 $A(x_i)$ 和 $B(x_i+\mathrm dx_i)$ 是三维欧几里得空间中的作意两个相邻的点，两个矢径$\mathbf x_A,\mathbf x_B$的矢量差
$$
\mathrm d\mathbf x=\mathrm dx_1\mathbf e_1+\mathrm dx_2\mathbf e_2+\mathrm dx_3\mathbf e_3=\mathrm dx_i\mathbf e_i \\
|\mathrm d\mathbf x|^2=\mathrm d\mathbf x\cdot \mathrm d\mathbf x=(\mathbf e_i\cdot\mathbf e_j)\mathrm dx_i\mathrm dx_j
$$
当 $\mathrm dx_i\to 0$ 时，两点的弧长 $\mathrm dl$ 与矢量差的大小 $|\mathrm d\mathbf x|$ 相等，故有
$$
\mathrm dl^2=\delta_{ij}\mathrm dx_i\mathrm dx_j=\mathrm dx_i\mathrm dx_i
$$
这就是欧几里得线元，即无穷小弧长，而 $\delta_{ij}$ 又称作欧几里得度规。

(2) 将时时间坐标取为 $x_0=ct$，四维闵可夫斯基时空坐标统一记作
$$
(x_0,x_1,x_2,x_3)=(ct,\mathbf x)
$$
根据时空间隔不变性，任意两个邻近的时空点$P(x_\mu)$ 和 $Q(x_\mu+\mathrm dx_\mu)$ 的时空间隔为
$$
\mathrm ds^2=\mathrm dx_0^2-\mathrm dx_1^2-\mathrm dx_2^2-\mathrm dx_3^2=\eta_{\mu\nu}\mathrm dx_\mu\mathrm dx_\nu
$$
式中
$$
\eta_{\mu\nu}=\begin{pmatrix}
1 & 0 & 0 & 0\\
0 & -1 & 0 & 0\\
0 & 0 & -1 & 0\\
0 & 0 & 0 & -1
\end{pmatrix} \quad(\mu,\nu=0,1,2,3)
$$
时空间隔$ds$ 叫做闵可夫斯基线元，$\eta_{\mu\nu}$ 叫做闵可夫斯基度规。

(3) 狭义相对论中还采取另外一种复欧几里得坐标，即把时间坐标取成复数 $x_4=\mathrm ict$，并将空间和时间坐标统一记作
$$
(x_1,x_2,x_3,x_4)=(\mathbf x,\mathrm ict)
$$
由此构成的时空连续域 $\{x_\mu|\mu=1,2,3,4\}$称作（复）闵可夫斯基时空。在闵可夫斯基时空坐标下 $(x_\mu)$，时空间隔式成为
$$
\mathrm ds^2=-\delta_{\mu\nu}\mathrm dx_\mu\mathrm dx_\nu=-\mathrm dx_\mu\mathrm dx_\mu
$$
其中的二阶张量
$$
\delta_{\mu\nu}=\begin{pmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{pmatrix} \quad(\mu,\nu=1,2,3,4)
$$
就是四维欧几里得度规。