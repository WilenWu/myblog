---
title: 高等数学(Calculus II)
date: 2019-05-03 20:31:28
categories: [数学,高等数学]
tags: [数学,偏导数,全微分,重积分,曲线积分,曲面积分]
cover: 
top_img: 
katex: true
description: false
---

# 多元函数微分法

## 多元函数

- **平面点集** (planar point set)
(1) 由平面解析几何知道，有序实数对 $(x,y)$ 与平面点 P 视作等同。二元有序实数组$(x,y)$的全体，即 $\R^2=\R×\R=\{(x,y)|x,y\in\R\}$ 就表示坐标平面。
(2) 设 $P_0(x_0,y_0)$ 是 $xOy$ 平面的一个点 ，$δ$ 为一正数，定义 
邻域：
$$
U(P_0,δ)=\{(x,y)|\sqrt{(x-x_0)^2+(y-y_0)^2}<δ\}
$$
去心邻域：
$$
\mathring{U}(P_0,δ)=\{(x,y)|0<\sqrt{(x-x_0)^2+(y-y_0)^2}<δ\}
$$
不需要强调半径 $δ$ 时可记作 $U(P_0), \mathring{U}(P_0)$
(3) 点与点集的关系：任意一点 $P\in\R^2$ 与任意一点集 $E⊂\R^2$ 之间必有以下关系之一
内点(interior point)：$∃ U(P),U(P)⊂ E$
外点(exterior point)：$∃ U(P),U(P)∩ E=\empty$
边界点(boundary point)：若点 $P$ 的任一邻域 $U(P)$中既含 $E$ 的点也含不是 $E$ 的点,则称为 $E$ 的==边界==，记作$∂E$
聚点(point of accumulation)：对于 $∀δ>0$ ，点 $P$ 的去心邻域 $\mathring{U}(P,δ)$ 总有 $E$ 中的点。
![](https://gitee.com/WilenWu/images/raw/master/math/points.png)

   (4) 定义一些重要的平面点集，设平面点集 $E$
开集(open set)：$∀P\in E$，$P$ 都是 $E$ 的内点
闭集(closed set)：边界 $∂E⊂ E$
连通集(connected set)：点集 $E$ 内任何两点都可以用折线连接起来，且该折线上的点都属于 $E$
开区域(region)：连通的开集
闭区域(closed region)：开区域连同它的边界一起构成的点集
有界集(bounded set)：$∃r>0$ 点集 $E⊂ U(O,r),O$是坐标原点，则点集 $E$ 为有界集
无界集：不是有界集的点集

- **n维空间** (n-dimensional space)：n元有序实数组$(x_1,x_2,\cdots,x_n)$的全体构成的集合 $\R^n=\{(x_1,x_2,\cdots,x_n)|x_i\in\R,i=1,2,\cdots,n\}$，称为n维空间

| n维空间 $\R^n$ | $x=(x_1,x_2,\cdots,x_n) \\ y=(y_1,y_2,\cdots,y_n)$       |
| :------------- | :----------------------------------------------------------- |
| 线性运算       | $x+y=(x_1+y_1,x_2+y_2,\cdots,x_n+y_n)\\ λx=(λx_1,λx_2,\cdots,λx_n),λ\in\R$ |
| 距离(distance) | $ρ(x,y)=\sqrt{(x_1-y_1)^2+(x_2-y_2)^2+\cdots+(x_n-y_n)^2}$   |
| 模(modulus)    | $\|x\|=ρ(x,O)=\sqrt{x_1^2+x_2^2+\cdots+x_n^2}$               |
| 变元的极限     | $x\to a\iff x_1\to a_1,x_2\to a_2,\cdots,x_n\to a_n$         |
| 定义邻域及其他 | $a\in \R^n,δ>0\rightarrow U(a,δ)=\{x\mid x\in\R^n,ρ(x,a)<δ\}$ |

- **多元函数的概念**(Multi-element Function)
(1) 二元函数：
映射 $f：D\to \R \quad(D⊂\R^2) \implies$ 函数  $z=f(x,y) \quad (x,y)\in D$
(2) n元函数：
映射 $f：D\to \R \quad(D⊂\R^n) \implies$ 函数  $z=f(\mathbf x) \quad (\mathbf x\in D)$

- **多元函数的极限**：设 $z=f(x,y)\quad (x,y)\in D$ ，$P_0(x_0,y_0)$ 是 $D$ 的聚点，若存在常数 $A$ ，对于 $∀ϵ>0$ ，总存在$δ>0$ 使得 $P(x,y)\in D∪\mathring{U}(P_0,δ)$ 时，都有 $|f(P)-A|< ϵ$ ，则定义  
$$
\lim\limits_{(x,y)\to(x_0,y_0)}f(x,y)=A \\
$$
或记作 
$$
\lim\limits_{P\to P_0}f(P)=A
$$

- **多元函数的连续性**：设 $z=f(x,y)\quad (x,y)\in D$ ，$P_0(x_0,y_0)$ 是 $D$ 的聚点，若 
$$
\lim\limits_{(x,y)\to(x_0,y_0)}f(x,y)=f(x_0,y_0)
$$
则函数在点 $P_0$ 连续。

- **多元初等函数**：由常数及不同自变量的一元基本初等函数经过有限次四则运算和复合运算而得到的。一切多元初等函数在定义区域内都是连续的。

- **多元函数的性质**

   <kbd>有界性与最大值最小值定理</kbd>：在有界闭区域 $D$ 上的多元连续函数，必定在 $D$ 上有界，且能取得它的最大值最小值

   <kbd>介值定理</kbd>:在有界闭区域 $D$ 上的多元连续函数，必取得介于最大值最小值之间的任何值

   <kbd>一致连续性定理</kbd>:在有界闭区域D上的多元连续函数，必定在D上一致连续(uniformly continuous)

## 偏导数

(1) 函数 $z=f(x,y)$ 在点 $(x_0,y_0)$ 处对自变量 $x$ 的偏导数 (partial derivative)定义为
$$
f_x(x_0,y_0)=\lim\limits_{Δx\to0}\dfrac{f(x_0+Δx,y_0)-f(x_0,y_0)}{Δx}
$$
可记作 
$$
f_x(x_0,y_0), \dfrac{∂z}{∂x}|_{x=x_0 \atop y=y_0}, \dfrac{∂f}{∂x}|_{x=x_0 \atop y=y_0},z_x|_{x=x_0 \atop y=y_0}
$$
对自变量 $y$ 的偏导数类似。

(2) 同样的可以定义自变量 $x$ 在区间内任意处的偏导函数，记作
$$
f_x(x,y), \dfrac{∂z}{∂x}, \dfrac{∂f}{∂x},z_x
$$
偏导函数的概念还可推广到二元以上的函数。
(3) 偏导数的记号是一个整体记号，不能看作微商，和一元函数不一样。
(4) 偏导数的几何意义：$f_x(x_0,y_0)$ 就是平面 $y=y_0$上的曲线 $z=f(x,y_0)$ 在点$(x_0,y_0)$处的切线对 $x$ 轴的斜率

![](https://gitee.com/WilenWu/images/raw/master/math/partial-derivative.png)
(5) 二阶偏导数
$$
\frac{∂}{∂x}(\frac{∂z}{∂x})=\frac{∂^2 z}{∂x^2}=f_{xx}(x,y), \quad 
\frac{∂}{∂y}(\frac{∂z}{∂y})=\frac{∂^2 y}{∂y^2}=f_{yy}(x,y) \\ 
\frac{∂}{∂y}(\frac{∂z}{∂x})=\frac{∂^2 z}{∂x∂y}=f_{xy}(x,y),\quad 
\frac{∂}{∂x}(\dfrac{∂z}{∂y})=\frac{∂^2 z}{∂y∂x}=f_{yx}(x,y),\\
$$
其中 $f_{xy},f_{yx}$ 两个称为混合偏导数，同样可求得三阶、四阶...以及 $n$ 阶偏导数。

<kbd>定理</kbd> 高阶混合偏导数在连续条件下与求导次序无关。
以二阶混合偏导数为例， $z=f(x,y)$ 的混合偏导数 
$$
f_{xy}(x,y)=f_{yx}(x,y)
$$

证明思路：任取一点 $(x_0,y_0)$ ，记 $\phi(y)=f(x_0+Δx,y)-f(x_0,y)$
$\begin{aligned}
Δz&=f(x_0+Δx,y_0+Δy)-f(x_0,y_0+Δy)-f(x_0+Δx,y_0)+f(x_0,y_0) \\
&=[f(x_0+Δx,y_0+Δy)-f(x_0,y_0+Δy)]-[f(x_0+Δx,y_0)-f(x_0,y_0)] \\
&=\phi(y_0+Δy)-\phi(y_0) \\
&=\phi'(y_0+θ_1Δy)Δy \\
&=[f_y(x_0+Δx,y_0+θ_1Δy)-f_y(x_0,y_0+θ_1Δy)]Δy \\
&=f_{yx}(x_0+θ_2Δx,y_0+θ_1Δy)ΔxΔy
\end{aligned}$
其中 $0<θ_1<0,0<θ_2<1$ （应用两次拉格朗日中值定理）
同理，还可记 $\psi(x)=f(x,y_0+Δy)-f(x,y_0)$ ，从而求得
$Δz=f_{xy}(x_0+θ_3Δx,y_0+θ_4Δy)ΔxΔy$
令 $Δx\to0,Δy\to0$ 取极限可得
$f_{xy}(x_0,y_0)=f_{yx}(x_0,y_0)$
由于 $(x_0,y_0)$ 的任意性，所以结论得证。

## 全微分

- **偏增量** (partial increment)：根据一元函数的关系可得
$$
f(x+Δx,y)-f(x,y)= f_x(x,y)Δx+o(Δx) \\
f(x,y+Δy)-f(x,y)= f_y(x,y)Δy+o(Δy)
$$
等式左边为对 $x$ 或 $y$ 的偏增量，右端对 $x$ 或 $y$ 的偏微分。

- **全增量**(total increment)：
$$
Δz=f(x+Δx,y+Δy)-f(x,y)
$$

- **全微分的定义**：若函数$z=f(x,y)$在点$(x,y)$的全增量$Δz$可表示为
$$
Δz=AΔx+BΔy+o(ρ)
$$
其中 $A$ 和 $B$ 不依赖于 $Δx$ 和 $Δy$而仅与 $x,y$ 有关，$ρ=\sqrt{(Δx)^2+(Δy)^2}$，则称函数在点 $(x,y)$==可微分==(differentiable)，$AΔx+BΔy$叫做==全微分== (total differential)，记作 $\mathrm{d}z$，即
$$
\mathrm{d}z=AΔx+BΔy
$$

- **全微分与偏导数**
<kbd>必要条件</kbd>： 函数 $z=f(x,y)$ 在 $P(x,y)$ 可微分，那该函数在 $P(x,y)$ 偏导数 $\dfrac{∂z}{∂x}， \dfrac{∂z}{∂y}$ 必定存在，且全微分
$$
\mathrm{d}z=\dfrac{∂z}{∂x}Δx+\dfrac{∂z}{∂y}Δy
$$

   证明：取 $P(x,y)$ 邻域内任一点 $Q(x+Δx,y+Δy)$ ，全增量 $Δz=AΔx+BΔy+o(ρ)$
特别的，当 $Δy=0$ 时，$Δz=AΔx+o(|Δx|)$ ，于是
$\lim\limits_{Δx\to0}\dfrac{f(x+Δx,y)-f(x,y)}{Δx}=A$
从而偏导数 $\dfrac{∂z}{∂x}$ 存在且等于 $A$ 。
同理也可证 $\dfrac{∂z}{∂y}=B$ ，于是全微分得证。


   <kbd>充分条件</kbd>：函数 $z=f(x,y)$ 的偏导数 $\dfrac{∂z}{∂x}, \dfrac{∂z}{∂y}$ 在 $(x,y)$ 连续，那么函数在该点可微分。

   <kbd>叠加原理</kbd>：习惯上自变量的增量 $Δx,Δy$ 常记作 $\mathrm{d}x,\mathrm{d}y$ ，即全微分等于两个偏微分之和
$$
\mathrm{d}z=\dfrac{∂z}{∂x}\mathrm{d}x+\dfrac{∂z}{∂y}\mathrm{d}y
$$
叠加原理同样适用于二元以上函数

## 求导法则
- **多元复合函数的求导法则**
(1) $u=ϕ(t),v=ψ(t),z=f(u,v)$ 全导数
$$
\dfrac{\mathrm{d}z}{\mathrm{d}t}=\dfrac{∂z}{∂u}\dfrac{\mathrm{d}u}{\mathrm{d}t}+\dfrac{∂z}{∂v}\dfrac{\mathrm{d}v}{\mathrm{d}t}
$$
(2) $u=ϕ(x,y),v=ψ(x,y),z=f(u,v)$ 
$$
\dfrac{∂z}{∂x}=\dfrac{∂z}{∂u}\dfrac{∂u}{∂x}+\dfrac{∂z}{∂v}\dfrac{∂v}{∂x} ,\quad
\dfrac{∂z}{∂y}=\dfrac{∂z}{∂u}\dfrac{∂u}{∂y}+\dfrac{∂z}{∂v}\dfrac{∂v}{∂y}
$$
(3) ==全微分形式不变性== ：$u=ϕ(x,y),v=ψ(x,y),z=f(u,v)$
$$
\mathrm{d}z=\dfrac{∂z}{∂x}\mathrm{d}x+\dfrac{∂z}{∂y}\mathrm{d}y=\dfrac{∂z}{∂u}\mathrm{d}u+\dfrac{∂z}{∂v}\mathrm{d}v
$$

- **隐函数的求导公式**
(1) $F(x,y)=0$，隐函数 $y=f(x)$ 
$$
\dfrac{\mathrm{d}y}{\mathrm{d}x}=-\dfrac{F_x}{F_y}
$$
(2) $F(x,y,z)=0$，隐函数$z=f(x,y)$ 
$$
\dfrac{∂z}{∂x}=-\dfrac{F_x}{F_z},\quad \dfrac{∂z}{∂y}=-\dfrac{F_y}{F_z}
$$
(3) $\begin{cases} F(x,y,u,v)=0 \\ G(x,y,u,v)=0 \end{cases}$ 一般能确定两个二元函数 $u=u(x,y),v=v(x,y)$ 
$$
\dfrac{∂u}{∂x}=-\dfrac{1}{J}\dfrac{∂(F,G)}{∂(x,v)}, 
\dfrac{∂u}{∂y}=-\dfrac{1}{J}\dfrac{∂(F,G)}{∂(y,v)} \\ 
\dfrac{∂v}{∂x}=-\dfrac{1}{J}\dfrac{∂(F,G)}{∂(u,x)}, 
\dfrac{∂v}{∂y}=-\dfrac{1}{J}\dfrac{∂(F,G)}{∂(u,y)}
$$
其中雅可比行列式(Jacobian) $J= \dfrac{∂(F,G)}{∂(u,v)}=\begin{vmatrix} \dfrac{∂F}{∂u} & \dfrac{∂F}{∂v} \\  \\ \dfrac{∂G}{∂u} & \dfrac{∂G}{∂v}\end{vmatrix}$

## 多元函数微分学的几何应用
- **一元向量值函数**(vector objective function)：
(1) 空间曲线 $Γ$ 的参数方程$\begin{cases}x=x(t)\\ y=y(t)\\ z=z(t) \end{cases}$ 可以写成向量形式
$$
\mathbf r=\mathbf f(t)=x(t)\mathbf i+y(t)\mathbf j+z(t)\mathbf k,\quad\mathbf f(t)⊂\R^3
$$
![](https://gitee.com/WilenWu/images/raw/master/math/vector-objective-function.png)![](https://gitee.com/WilenWu/images/raw/master/math/vector-objective-function-limit.png) 
==极限== $\lim\limits_{t\to t_0}\mathbf f(t)=\mathbf r_0$
==连续== $\mathbf f(t)$ 在 $t_0$ 连续 $\iff\lim\limits_{t\to t_0}\mathbf f(t)=\mathbf f(t_0)$
==导数== 定义 $\mathbf f'(t)=\lim\limits_{Δt\to0}\dfrac{\mathbf f(t+Δt)-\mathbf f (t)}{Δt}$
$$
\mathbf f'(t)=x'(t)\mathbf i+y'(t)\mathbf j+z'(t)\mathbf k
$$

   ==导数运算法则==
$(\mathbf u\pm\mathbf v)'=\mathbf u'\pm\mathbf v'$ 
$(\phi \mathbf v)'=\phi'\mathbf v+\phi\mathbf v'$ 
$(\mathbf u\cdot\mathbf v)'=\mathbf u'\cdot\mathbf v+\mathbf u\cdot\mathbf v'$ 
$(\mathbf u×\mathbf v)'=\mathbf u'×\mathbf v+\mathbf u×\mathbf v'$ 
$\dfrac{\mathrm{d}\mathbf u}{\mathrm{d}t}=\dfrac{\mathrm{d}\mathbf u}{\mathrm{d}ϕ}\dfrac{\mathrm{d}ϕ}{\mathrm{d}t}$


   (2) 空间曲线$Γ$在点$M(x_0,y_0,z_0)$处的==切线方程==为
$$
   \dfrac{x-x_0}{x'(t_0)}=\dfrac{y-y_0}{y'(t_0)}=\dfrac{z-z_0}{z'(t_0)}
$$
(3) 通过点M且与切线垂直的==法平面方程==为
$$
x'(t_0)(x-x_0)+y'(t_0)(y-y_0)+z'(t_0)(z-z_0)=0
$$
(4) 若曲线$Γ$方程为$\begin{cases}F(x,y,z)=0\\ G(x,y,z)=0 \end{cases}$，则法平面方程为 
$$
\begin{vmatrix}F_y&F_z \\ G_y&G_z\end{vmatrix}(x-x_0)+
\begin{vmatrix}F_z&F_x \\ G_z&G_x\end{vmatrix}(y-y_0)+
\begin{vmatrix}F_x&F_y \\ G_x&G_y\end{vmatrix}(z-z_0)=0
$$

- **曲面** $Σ$ 方程 $F(x,y,z)=0$，曲面上一点 $M(x_0,y_0,z_0)$
  ![](https://gitee.com/WilenWu/images/raw/master/math/tangent-plane-fun.png)
   (1) 过点 $M$ 的切线形成的平面，==切平面==方程为
$$
F_x(x_0,y_0,z_0)(x-x_0)+F_y(x_0,y_0,z_0)(y-y_0)+F_z(x_0,y_0,z_0)(z-z_0)=0
$$
(2) 过点 $M$ 且垂直于切平面的==法线==方程
$$
\dfrac{x-x_0}{F_x(x_0,y_0,z_0)}=\dfrac{y-y_0}{F_y(x_0,y_0,z_0)}=\dfrac{z-z_0}{F_z(x_0,y_0,z_0)}
$$

## 方向导数和梯度

- **方向导数**(directional derivative)：函数沿坐标轴方向的变化率是偏导数，而函数任意沿射线 $l$ 的方向的变化率称为方向导数。
定义 $z=f(x,y)$ 在点 $P_0(x_0,y_0)$ 沿射线 $l$ 方向的方向导数为 ：
$$
\dfrac{∂f}{∂l}\Big|_{(x_0,y_0)}=\lim\limits_{\rho\to0}\dfrac{f(x_0+Δx,y_0+Δy)-f(x_0,y_0)}{\rho}
$$
其中 $\rho=\sqrt{(Δx)^2+(Δy)^2}$ 

   <kbd>定理</kbd>：若函数$z=f(x,y)$ 在点 $P_0(x_0,y_0)$处可微，那么函数在该点任意方向 $l$ 的方向导数存在，且为
$$
\dfrac{∂f}{∂l}\Big|_{(x_0,y_0)}=f_x(x_0,y_0)\cosα+f_y(x_0,y_0)\cosβ
$$
其中 $\cosα,\cosβ$ 为 $l$ 的方向余弦[^cos]，同样二元以上函数的方向导数类似。
![](https://gitee.com/WilenWu/images/raw/master/math/direction-cosine.png)
   证明：在 $P_0(x_0,y_0)$ 邻域内， $l$ 方向上任取一点 $P(x,y)$，由于函数在点 $P_0(x_0,y_0)$ 处可微，所以 
   增量 $f(P)-f(P_0)=Δz=f_x(x_0,y_0)Δx+f_x(x_0,y_0)Δy+o(ρ)$
   其中 $\rho=\sqrt{(Δx)^2+(Δy)^2}$ ，两边同除以 $ρ$ ，并取极限得到
   $\lim\limits_{\rho\to0}\dfrac{Δz}{\rho}=f_x(x_0,y_0)\cfrac{Δx}{\rho}+f_x(x_0,y_0)\cfrac{Δy}{\rho}$
   由于 $Δx=\rho\cosα,\quad Δy=\rho\cosβ$
   于是定理得证。

[^cos]: $l$ 的方向向量与坐标轴的夹角 $α,β$ 称为方向角，$cosα,cosβ$ 称为 $l$ 的方向余弦

- **梯度**(gradient)：定义函数$f(x,y)$在点$P_0(x_0,y_0)$处的梯度
$$
\mathrm{grad}f(x_0,y_0)=f_x(x_0,y_0)\mathbf i+f_y(x_0,y_0)\mathbf j
$$
设 $\mathbf e_l=(\cosα,\cosβ)$ 是与射线 $l$ 同向的单位向量，则方向导数可写作
$$
\dfrac{∂f}{∂l}\mid_{(x_0,y_0)}=\mathrm{grad}f(x_0,y_0)\cdot\mathbf e_l=|\mathrm{grad}f(x_0,y_0)|\cosθ
$$
其中 $θ$ 为 $\mathrm{grad}f(x_0,y_0)$ 与 $\mathbf e_l$ 的夹角，由此表明
(1) 当 $θ=0$ 或 $θ=π$ 时，函数$f(x,y)$变化最快，即最大方向导数与梯度方向一致
(2) 当 $θ=\dfrac{π}{2}$ 时，函数$f(x,y)$变化率为0


## 多元函数的极值
 <kbd>极值</kbd>：函数 $z=f(x,y)$定义域为 $D$，点 $P_0(x_0,y_0)$是 $D$ 的内点，对于 $∀ (x,y)\in\mathring{U}(P_0)$ 总有 $f(x,y)<f(x_0,y_0)$ 或 $f(x,y)>f(x_0,y_0)$，则称 $f(x_0,y_0)$ 是函数 $f(x,y)$ 的一个极大值或极小值，极大值和极小值统称为==极值==(extremum)，使函数取得极值的点称为==极值点==。

- **无条件极值** ：求函数 $z=f(x,y)$ 的极值

   ==(必要条件)==：设 $f(x,y)$ 在 $(x_0,y_0)$ 处有偏导数，$f(x_0,y_0)$ 为极值，则有
$$
f_x(x_0,y_0)=0,f_y(x_0,y_0)=0
$$
点 $(x_0,y_0)$ 称为驻点或稳定点。

   ==(充分条件)==：设 $f(x,y)$ 在 $(x_0,y_0)$ 处连续，且 $f_x(x_0,y_0)=0,f_y(x_0,y_0)=0$ ，令 
$$
f_{xx}(x_0,y_0)=A,f_{xy}(x_0,y_0)=B,f_{yy}(x_0,y_0)=C
$$
则有 （需用二元泰勒公式证明）
$$
AC-B^2\begin{cases}
<0,&有极值,A<0有极大值,A>0有极小值 \\  >0,&没有极值 \\=0,&另做讨论
\end{cases}
$$

- **条件极值** (conditional extremum)：对自变量有附加条件的极值称为条件极值。
(1) 有些可用消元法化为无条件极值问题求解。
例如，求体积为 $V$ 而表面积 $S$ 最小的长方体的表面积问题。
设长方体边长为 $x,y,z$ ，则表面积为 $S=2(xy+xz+yz)$ ，还必须满足约束条件 $V=xyz$ 。
由约束条件解出显函数 $z=\cfrac{V}{xy}$ ，再带入表面积 $S=2V(\cfrac{1}{x}+\cfrac{1}{y})+2xy$ ，求 $S$ 的无条件极值即可。
$S_x=-\cfrac{2V}{x^2}+2y=0,\quad S_y=-\cfrac{2V}{y^2}+2x=0$
从而可以解出 $x=y=z=\sqrt[3]{V}$
最小表面积 $S=6\sqrt[3]{V^2}$
(2) 然而在一般情况下，无法化为无条件极值，解决这类极值问题的办法叫做拉格朗日乘数法(Lagrange multiplier)
例如，求二元函数 $z=f(x,y)$ 在附加条件 $ϕ(x,y)=0$ 下的极值。
若由附加条件确定隐函数 $y=y(x)$ ，则可使目标函数成为一元函数 $z=f(x,y(x))$ ，再由一元函数极值的必要条件
$\cfrac{\mathrm dz}{\mathrm dx}=f_x+f_y\cfrac{\mathrm dy}{\mathrm dx}=f_x-f_y\cfrac{ϕ_x}{ϕ_y}=0$
求出稳定点 $P_0=(x,y)$ 
设 $\cfrac{f_y(x,y)}{ϕ_y(x,y)}=-λ$ ，则上述必要条件就变为
$\begin{cases}f_x(x,y)+λ ϕ_x(x,y)=0\\
f_y(x,y)+λ ϕ_y(x,y)=0\\
ϕ(x,y)=0\end{cases}$
引进辅助函数 $L(x,y)=f(x,y)+λϕ(x,y)$ 
我们容易看出方程组前两式就是 $L_x(x,y)=0,\quad L_y(x,y)=0$
函数 $L(x,y)$ 称为==拉格朗日函数==，参数 $λ$ 称为==拉格朗日乘子==。
由此，我们可以得到拉格朗日乘数法：
(i) 先做拉格朗日函数 
$$
L(x,y)=f(x,y)+λϕ(x,y)
$$
(ii) 求出拉格朗日函数的偏导数然后与附加条件联立成方程组
$$
\begin{cases}
L_x(x,y)=0\\
L_y(x,y)=0\\
ϕ(x,y)=0\end{cases}
$$
求出 $x,y,λ$，点$(x,y)$就是在附加条件下的可能极值
拉格朗日乘子法也适用于多个自变量的的多元函数与多条件，如果涉及到多个条件，只需引入多个拉格朗日乘子即可。

- **最小二乘法**(least square method)
求样本 $(x_i,y_i)$ 的经验公式$f(x)=ax+b$
(1) 为使偏差$y_i-f(x_i)$取得最小值，即求$M(a,b)=\sum [y_i-(ax_i+b)]^2$的最小值，来确定常数 $a,b$
(2) 根据本章的讨论得知，即求 $a,b$ 的驻点$\begin{cases}M_a(a,b)=0\\ M_b(a,b)=0\end{cases}$，解方程组可得 $a,b$

## 二元函数的泰勒公式
设函数 $z=f(x,y)$ 在点 $P_0(x_0,y_0)$ 的某一邻域 $U(P_0)$ 连续且有 $(n+1)$ 阶连续偏导数，$(x_0+h,y_0+k)\in U(P_0)$
**泰勒公式**(Taylor formula)
$$
\begin{aligned}
f(x_0+h,y_0+k)&=f(x_0,y_0)+(h\dfrac{∂}{∂x}+k\dfrac{∂}{∂y})f(x_0,y_0)+\dfrac{1}{2!}(h\dfrac{∂}{∂x}+k\dfrac{∂}{∂y})^2f(x_0,y_0)+\cdots+\dfrac{1}{n!}(h\dfrac{∂}{∂x}+k\dfrac{∂}{∂y})^nf(x_0,y_0)+R_n \\
&=\displaystyle\sum_{i=0}^{n}\dfrac{1}{i!}(h\dfrac{∂}{∂x}+k\dfrac{∂}{∂y})^i f(x_0,y_0)+R_n
\end{aligned}
$$
其中记号
$(h\dfrac{∂}{∂x}+k\dfrac{∂}{∂y})f(x_0,y_0)$ 表示 $hf_x(x_0,y_0)+kf_y(x_0,y_0)$
$(h\dfrac{∂}{∂x}+k\dfrac{∂}{∂y})^2f(x_0,y_0)$ 表示 $h^2f_{xx}(x_0,y_0)+2hkf_{xy}(x_0,y_0)+k^2f_{yy}(x_0,y_0)$

一般的，记号
$(h\dfrac{∂}{∂x}+k\dfrac{∂}{∂y})^mf(x_0,y_0)$ 表示 $\displaystyle\sum_{p=0}^{m}∁_m^ph^pk^{m-p}\dfrac{∂^mf}{∂x^p∂y^{m-p}}\mid_{(x_0,y_0)}$

拉格朗日余项
$R_n=\dfrac{1}{(n+1)!}(h\dfrac{∂}{∂x}+k\dfrac{∂}{∂y})^{n+1}f(x_0+θ h,y_0+θ k) \quad(0<θ<1)$
当 $n=0$时，即为二元函数的拉格朗日中值公式


# 重积分
## 二重积分概念和性质
- **引入意义**
曲顶 $z=f(x,y)$柱体的体积 $V=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i)Δσ_i$
密度为 $μ=μ(x,y)$平面薄片的质量 $m=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}μ(ξ_i,η_i)Δσ_i$
![](https://gitee.com/WilenWu/images/raw/master/math/double-integral-demo1.png) ![](https://gitee.com/WilenWu/images/raw/master/math/double-integral-demo2.png)


- **定义**：设$f(x,y)$ 是有界闭区域 $D$ 上的有界函数
将闭区域 $D$ 任意分成 $n$ 个小闭区域 
$$
Δσ_1,Δσ_2,\cdots,Δσ_n
$$
其中 $Δσ_i$第 $i$ 个小闭区域，也表示它的面积，在每个$Δσ_i$中任取一点$(ξ_i,η_i)$，作乘积 $f(ξ_i,η_i)Δσ_i(i=1,2,\cdots,n)$，并作出求和
$$
Ω=\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i)Δσ_i
$$
如果当各小闭区域直径中的最大值$λ\to0$时，和的极限总存在，且与闭区域 $D$ 的分法及点$(ξ_i,η_i)$的取法无关，则这个极限叫做函数 $f(x,y)$ 在闭区域$D$的二重积分(double integral)，记  
$$
\iint\limits_Df(x,y)\mathrm{d}σ=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i)Δσ_i
$$
其中 $D$ 称为积分区域(domain of integration)，$\mathrm{d}σ$叫做面积元素，$f(x,y)$是被积函数。
在直角坐标系中，有时把面积元素 $\mathrm{d}σ$ 记作 $\mathrm{d}x\mathrm{d}y$，把二重积分记作
$$
\displaystyle \iint\limits_Df(x,y)\mathrm{d}x\mathrm{d}y
$$

| 二重积分的性质                                               |
| :----------------------------------------------------------- |
| $\displaystyle\iint\limits_Dkf(x,y)\mathrm{d}σ=k\iint\limits_Df(x,y)\mathrm{d}σ$ |
| $\displaystyle\iint\limits_D [f(x,y)± g(x,y)]\mathrm{d}σ=\iint\limits_D f(x,y)\mathrm{d}σ ± \iint\limits_D g(x,y)\mathrm{d}σ$ |
| $\displaystyle\iint\limits_{D} f(x,y)\mathrm{d}σ=\iint\limits_{D_1} f(x,y)\mathrm{d}σ + \iint\limits_{D_2} f(x,y)\mathrm{d}σ\quad (D=D_1+D_2)$ |
| $\displaystyle f(x,y)⩽g(x,y)\implies \iint\limits_{D}f(x,y)\mathrm{d}σ⩽\iint\limits_{D}g(x,y)\mathrm{d}σ$ |
| $\displaystyle\iint\limits_{D}f(x,y)\mathrm{d}σ⩽\iint\limits_{D}f(x,y)\mathrm{d}σ$ |
| $\displaystyle m⩽f(x,y)⩽M\implies m S_D⩽\iint\limits_{D}f(x,y)\mathrm{d}σ⩽M S_D$ |

- **二重积分中值定理**：$f(x,y)$ 在闭区域 $D$上连续，$σ$ 是 $D$ 的面积 ，则在 $D$ 上至少存在一点 $(ξ,η)$ 使得
$$
\iint\limits_Df(x,y)\mathrm{d}σ=f(ξ,η)σ
$$

## 二重积分的计算
- **利用直角坐标计算二重积分**
下面用几何观点来讨论二重积分 $\displaystyle \iint\limits_Df(x,y)\mathrm{d}σ$
(1) 假设积分区域 $D$为 $X$ 型区域，即可以用不等式表示为
$$
y_1(x)⩽ y ⩽ y_2(x),\quad(a⩽ x⩽ b)
$$

   先计算截面面积 
$$
   \displaystyle S(x_0)=\int_{y_1(x_0)}^{y_2(x_0)}f(x_0,y)\mathrm{d}y
$$

   一般的 $x\in[a,b]$时，方程都适用。
再计算曲顶柱体体积 
$$
\displaystyle V=\int_a^bS(x)\mathrm{d}x=\int_a^b[\int_{y_1(x)}^{y_2(x)}f(x,y)\mathrm{d}y]\mathrm{d}x
$$

   这个体积就是二重积分的值，这样先对 $y$ 再对 $x$ 积分的方法叫==二次积分==，二次积分也常记作
$$
\iint\limits_Df(x,y)\mathrm{d}σ=\int_a^b \mathrm{d}x\int_{y_1(x)}^{y_2(x)}f(x,y)\mathrm{d}y
$$
![](https://gitee.com/WilenWu/images/raw/master/math/quadratic-integral-X.png) ![](https://gitee.com/WilenWu/images/raw/master/math/quadratic-integral-II.png)

   (2) 假设积分区域 $D$ 为 $Y$ 型区域，即可以用不等式表示为 
$$
   x_1(y)⩽ x ⩽ x_2(y),a⩽ y⩽ b
$$

   二次积分的值为 
$$
\iint\limits_Df(x,y)\mathrm{d}σ=\int_a^b \mathrm{d}y\int_{x_1(y)}^{x_2(y)}f(x,y)\mathrm{d}x
$$
![](https://gitee.com/WilenWu/images/raw/master/math/quadratic-integral-Y.png)
   (3) 如果积分区域既不是 $X$ 型区域，也不是 $Y$ 型区域，这时可以把 $D$ 分成几部分求和。
![](https://gitee.com/WilenWu/images/raw/master/math/quadratic-integral-XY.png)
**示例**：求两个底圆半径都是 $R$ 的直交圆柱面围成的立体的体积。
![](https://gitee.com/WilenWu/images/raw/master/math/double-integral-demo3.png)
解：设这两个圆柱面的方程为 
$x^2+y^2=R^2$ 及 $x^2+z^2=R^2$
利用直交立体关于坐标面的对称性，只求第一象限的体积即可。
第一象限可以看成一个曲顶柱体，它的底为 
$D=\{(x,y)|0⩽y⩽\sqrt{R^2-x^2},0⩽x⩽R\}$
如图，它的顶面方程为
$z=\sqrt{R^2-x^2}$
于是 $V_1=\displaystyle\iint\limits_D\sqrt{R^2-x^2}dσ \\
=\int_0^Rdx\int_{0}^{\sqrt{R^2-x^2}}\sqrt{R^2-x^2}dy \\
=\int_0^R[y\sqrt{R^2-x^2}\Big|_0^{\sqrt{R^2-x^2}}]dx \\
=\int_0^R(R^2-x^2)dx=\cfrac{2}{3}R^3$

- **利用极坐标计算二重积分**
(1) 按二重积分的定义来计算，小闭区域面积 
$$
Δσ_i=\dfrac{1}{2}(ρ_i+Δρ_i)^2\cdotΔθ_i=\bar{ρ_i}\cdotΔρ_i\cdotΔθ_i
$$

   其中$\bar{ρ_i}$表示相邻两圆弧半径的平均值，由极坐标和直角坐标间的关系[^1]得
$$
\iint\limits_Df(x,y)\mathrm{d}x\mathrm{d}y=\iint\limits_Df(ρ\cosθ,ρ\sinθ)ρ \mathrm{d}ρ \mathrm{d}θ
$$

   其中 $ρ \mathrm{d}ρ \mathrm{d}θ$ 就是极坐标中的面积元素

   (2) 极坐标的二重积分同样可以化为二次积分来求，设积分区域D可以用不等式表示为
$$
   ρ_1(θ)⩽ ρ ⩽ ρ_2(θ),α⩽θ⩽ β
$$
![](https://gitee.com/WilenWu/images/raw/master/math/double-integral-polar.png)![](https://gitee.com/WilenWu/images/raw/master/math/quadratic-integral-polar.png) 
   二次积分的值为
$$
\iint\limits_Df(ρ\cosθ,ρ\sinθ)ρ \mathrm{d}ρ \mathrm{d}θ
=\int_α^β \mathrm{d}θ\int_{ρ_1(θ)}^{ρ_2(θ)}f(ρ\cosθ,ρ\sinθ)ρ \mathrm{d}ρ
$$
(特例) 设$f(x,y)=1$，小面积区域 $\mathrm{d}σ=ρ \mathrm{d}ρ \mathrm{d}θ$，，可求得D的面积 
$$
σ=\iint\limits_Dρ \mathrm{d}ρ \mathrm{d}θ=\dfrac{1}{2}\int_α^β[ρ_2^2(θ)-ρ_1^2(θ)] \mathrm{d}θ
$$

[^1]: 平面向量极坐标$(ρ,θ)$和直角坐标$(x,y)$换算：$\begin{cases}x=ρ\cosθ \\ y=ρ\sinθ\end{cases},ρ\in[0,+∞),θ\in[0,2π)$


- **二重积分的换元法**：设$f(x,y)$在$xOy$平面上的闭区域$D$上连续，若变换
$$
T:x=x(u,v),y=y(u,v)
$$

   将$uOv$平面上的闭区域$D'$变为$xOy$平面上的$D$，且满足
(1) $x(u,v),y(u,v)在D'$上具有一阶连续偏导数
(2) 在$D'$上雅可比行列式[^2]$\det J(u,v)=\det\dfrac{∂(x,y)}{∂(u,v)}\neq0$
(3) 变换 $T:D'\to D$ 是一一对应的
则有
$$
\iint\limits_Df(x,y)\mathrm{d}x\mathrm{d}y=\iint\limits_{D'}f[x(u,v),y(u,v)]\ |J(u,v)|\ \mathrm{d}u\mathrm{d}v
$$
![](https://gitee.com/WilenWu/images/raw/master/math/double-integral-substitution.png)

[^2]: 雅可比矩阵(Jacobian matrix) $J(x_1,\cdots,x_n)=\dfrac{∂(y_1,\cdots,y_m)}{∂(x_1,\cdots,x_n)}=\begin{pmatrix}\dfrac{∂y_1}{∂x_1}\cdots\dfrac{∂y_1}{∂x_n} \\ \vdots \ddots \vdots \\ \dfrac{∂y_m}{∂x_1}\cdots\dfrac{∂y_m}{∂x_n} \end{pmatrix}$

## 三重积分概念和性质

(1) 定积分和二重积分作为和的极限的概念自然推广到三重积分(triple integral)
$$
\iiint\limits_{Ω}f(x,y,z)\mathrm{d}v=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i,ζ_i)Δv_i
$$

其中$f(x,y,z)$是被积函数，$Ω$称为积分区域(domain of integration)，$\mathrm{d}v$叫做体积元素
在直角坐标系中，有时把体积元素$\mathrm{d}v$记作$\mathrm{d}x\mathrm{d}y\mathrm{d}z$，把三重积分记作
$$
\displaystyle\iiint\limits_{Ω}f(x,y,z)\mathrm{d}x\mathrm{d}y\mathrm{d}z
$$
(2) 三重积分的性质与二重积分类似，这里就不在重复了

## 三重积分的计算
-  **利用直角坐标进行三重积分**
(1) 假设平行于$z$轴且穿过闭区域$Ω$的直线与闭区域的边界曲面$S$相交不多于两点，把闭区域$Ω$投影到$xOy$平面上，得一平面闭区域$D_{xy}$，如图，$S$ 的上下两部分方程为 
$$
\begin{aligned}S_1:z=z_1(x,y)\\ S_2:z=z_2(x,y)\end{aligned}
$$

   则积分区域可表示为 
$$
   Ω=\{(x,y,z)|z_1(x,y)⩽ z⩽ z_2(x,y),(x,y)\in D_{xy}\}
$$

   假如 
$$
   D_{xy}=\{(x,y)|y_1(x)⩽ y⩽ y_2(x),a⩽ x⩽ b\}
$$

   三重积分可化为==三次积分==
$$
\iiint\limits_{Ω}f(x,y,z)\mathrm{d}v=\int_a^b\mathrm{d}x\int_{y_1(x)}^{y_2(x)}\mathrm{d}y\int_{z_1(x,y)}^{z_2(x,y)}f(x,y,z)\mathrm{d}z
$$
(2) 也可把闭区域投影到$xOz$或$yOz$平面上，三重积分解法类似
(3) 如果平行于坐标轴且穿过闭区域$Ω$的直线与闭区域的边界曲面$S$相交多于两点，可以把闭区域分成若干部分分别计算。
(4) 有时候计算三重积分也可以先计算一个二重积分，在计算定积分，有下列公式。
设空间闭区域 
$$
Ω=\{(x,y,z)|(x,y)\in D_{z},c_1⩽z⩽c_2\}
$$
其中 $D_z$ 是竖坐标为 $z$ 的平面与闭区域的截面，则有
$$
\iiint\limits_{Ω}f(x,y,z)\mathrm{d}v=\int_{c_1}^{c_2}\mathrm{d}z\iint\limits_{D_z}f(x,y,z)\mathrm{d}x\mathrm{d}y
$$
![](https://gitee.com/WilenWu/images/raw/master/math/triple-integral-I.png)![](https://gitee.com/WilenWu/images/raw/master/math/triple-integral-II.png)  
**示例**：求解 $\displaystyle\iiint\limits_{Ω}z^2dxdydz$ 其中 $Ω$ 是椭球 $\cfrac{x^2}{a^2}+\cfrac{y^2}{b^2}+\cfrac{z^2}{c^2}=1$ 所围成的空间闭区域。
![](https://gitee.com/WilenWu/images/raw/master/math/triple-integral-demo.png)
解：空间闭区域可表示为
$Ω=\{(x,y,z)|\cfrac{x^2}{a^2}+\cfrac{y^2}{b^2}⩽1-\cfrac{z^2}{c^2},-c⩽z⩽c\}$
于是 
$\displaystyle\iiint\limits_{Ω}z^2dxdydz=\int_{-z}^{z}z^2dz\iint\limits_{D_z}dxdy \\
=\pi ab\int_{-z}^{z}(1-\cfrac{z^2}{c^2})z^2dz=\cfrac{4}{15}\pi abc^3$


-  **利用柱面坐标进行三重积分**[^3]：体积元素$\mathrm{d}v=ρ \mathrm{d}ρ \mathrm{d}θ \mathrm{d}z$
$$
\iiint\limits_{Ω}f(x,y,z)\mathrm{d}x\mathrm{d}y\mathrm{d}z=\iiint\limits_{Ω}f(ρ\cosθ,ρ\cosθ,z)ρ \mathrm{d}ρ \mathrm{d}θ \mathrm{d}z
$$
![](https://gitee.com/WilenWu/images/raw/master/math/triple-integral-cylindrical-I.png)  ![](https://gitee.com/WilenWu/images/raw/master/math/triple-integral-cylindrical-II.png)


-  **利用球面坐标进行三重积分**[^3]：体积元素$\mathrm{d}v=r^2\sinϕ \mathrm{d}r \mathrm{d}ϕ \mathrm{d}θ$
$$
\iiint\limits_{Ω}f(x,y,z)\mathrm{d}x\mathrm{d}y\mathrm{d}z=\iiint\limits_{Ω}f(r\sinϕ\cosθ,r\sinϕ\sinθ,r\cosϕ)r^2\sinϕ \mathrm{d}r \mathrm{d}ϕ \mathrm{d}θ
$$
![](https://gitee.com/WilenWu/images/raw/master/math/triple-integral-spherical-I.png)  ![](https://gitee.com/WilenWu/images/raw/master/math/triple-integral-spherical-II.png)


[^3]: 柱面坐标$(ρ,θ,z)$和直角坐标$(x,y,z)$的换算$\begin{cases}x=ρ\cosθ \\ y=ρ\sinθ \\ z=z\end{cases},ρ\in[0,+∞),θ\in[0,2π),z\in[0,+∞)$，球面坐标$(r,ϕ,θ)$和直角坐标$(x,y,z)$的换算$\begin{cases}x=r\sinϕ\cosθ \\ y=r\sinϕ\sinθ \\z=r\cosϕ\end{cases},ρ\in[0,+∞),ϕ\in[0,π),θ\in[0,2π)$


## 重积分的应用
- **曲面的面积**：
(1) 曲面方程 $z=f(x,y)$
面积元素 $\mathrm{d}A=\dfrac{\mathrm{d}σ}{\cosγ},\cosγ=\dfrac{1}{\sqrt{1+f_x^2(x,y)+f_y^2(x,y)}}$
由此面积公式 
$$
\displaystyle A=\iint\limits_D\sqrt{1+f_x^2(x,y)+f_y^2(x,y)}\mathrm{d}x\mathrm{d}y
$$
![](https://gitee.com/WilenWu/images/raw/master/math/area-of-surface-I.png)![](https://gitee.com/WilenWu/images/raw/master/math/area-of-surface-II.png)
(2) 曲面的参数方程$\begin{cases}x=x(u,v)\\y=y(u,v)\\z=z(u,v) \end{cases}\quad (u,v)\in D$，且[^2]
$\det\dfrac{∂(x,y)}{∂(u,v)}\cdot\det\dfrac{∂(x,z)}{∂(u,v)}\cdot\det\dfrac{∂(y,z)}{∂(u,v)}\neq0$
面积公式 
$$
\displaystyle A=\iint\limits_D\sqrt{EG-F^2}\mathrm{d}u\mathrm{d}v
$$
其中  $E=x_u^2+y_u^2+z_u^2 \\
F=x_ux_v+y_uy_v+z_uz_v \\
G=x_v^2+y_v^2+z_v^2$

- **质心**[^4]：占有空间区域 $Ω$，在 $\mathbf r=(x,y,z)$ 处密度为 $ρ(\mathbf r)=ρ(x,y,z)$的物体
质心坐标为 $\displaystyle (\bar x,\bar y,\bar z)=\dfrac{1}{M}\iiint\limits_{Ω}\mathbf rρ(\mathbf r)\mathrm{d}v$
质量为 $\displaystyle M=\iiint\limits_{Ω}ρ(\mathbf r)\mathrm{d}v$
[^4]: 力学上质心坐标的计算公式 $\mathbf r=\dfrac{\sum m_i\mathbf r_i}{M}, M=\sum m_i$

## 含参变量的积分

- **含参变量的积分**：设函数 $f(x,y)$是矩形$R=[a,b]×[c,d]$上的连续函数，在 $[a,b]$ 上任取 $x$ 的一个值，于是 $f(x,y)$ 是 $[c,d]$ 上关于 $y$ 的一元连续函数，从而积分

$$
\displaystyle ϕ(x)=\int_c^df(x,y)\mathrm{d}y\quad(a⩽ x⩽ b)
$$
可以确定一个定义在 $x\in[a,b]$ 关于 $x$  的函数，这里 $x$ 在积分过程中视作常量，称为==参变量==。 下面讨论关于 $ϕ(x)$ 的一些性质。

1.  函数 $ϕ(x)$ 在 $[a,b]$ 上连续

2. 对 $ϕ(x)$ 进行积分，可以得到
$$
\displaystyle \int_a^bϕ(x)\mathrm{d}x=\int_a^\mathrm{d}\mathrm{d}y\int_c^df(x,y)\mathrm{d}y=\iint\limits_Rf(x,y)\mathrm{d}x\mathrm{d}y
$$

   显然，二重积分可以化为二次积分并交换积分顺序，于是
$$
\displaystyle \int_a^b\mathrm{d}x\int_c^df(x,y)\mathrm{d}y=\int_c^\mathrm{d}\mathrm{d}y\int_a^bf(x,y)\mathrm{d}x
$$
3. 若 $f_x(x,y)$ 也在 $R$ 上连续，那么 $ϕ(x)$ 在 $[a,b]$ 上可微，并且
$$
\displaystyle ϕ'(x)=\cfrac{\mathrm d}{\mathrm dx}\int_c^df(x,y)\mathrm{d}y
=\int_c^df_x(x,y)\mathrm{d}y
$$
证明：由定义可计算 $\displaystyle ϕ(x+Δx)-ϕ(x)=\int_c^d[f(x+Δx)-f(x)]dy$
由拉格朗日中值定理和 $f_x(x,y)$ 的一致连续性得到
$\cfrac{f(x+Δx)-f(x)}{Δx}=f_x(x+θΔx,y)=f_x(x,y)+η(x,y,Δx)$
其中 $0<θ<1$，$|η|$ 可小于任意给定的正数 $ϵ$ ，只要 $|Δx|$ 小于某个正数 $δ$。
因此 $\displaystyle \Big|\int_c^dη(x,y,Δx)dy\Big|<\int_c^dϵdy=ϵ(d-c)\quad(|Δx|<δ)$
这就是说 $\displaystyle \lim\limits_{Δx\to0}\int_c^dη(x,y,Δx)dy=0$
所以 $\displaystyle \cfrac{ϕ(x+Δx)-ϕ(x)}{Δx}=\int_c^df_x(x,y)+\int_c^dη(x,y,Δx)$
令 $Δx\to0$ 取极限即证毕。

- **莱布尼茨公式**：对于实际中更一般的情形，设函数 $f(x,y)$是矩形$R=[a,b]×[c,d]$上的连续函数，函数 $α(x),β(x)$在 $x\in[a,b]$ 上连续，且 $c⩽α(x)⩽d,c⩽β(x)⩽d\quad(a⩽x⩽b)$ ，从而确定积分

$$
\displaystyle \Phi(x)=\int_{α(x)}^{β(x)}f(x,y)\mathrm{d}y\quad(a⩽ x⩽ b)
$$
1.  函数 $\Phi(x)$ 在区间 $[a,b]$ 上连续
2. 若 $f_x(x,y)$ 也在 $R$ 上连续，函数 $α(x),β(x)$在 $x\in[a,b]$ 上可微，那么 $\Phi(x)$ 在 $[a,b]$ 上可微，并且
$$
\displaystyle \Phi'(x)=\int_{α(x)}^{β(x)}f(x,y)\mathrm{d}y
=\int_{α(x)}^{β(x)}f_x(x,y)\mathrm{d}y+f[x,β(x)]β'(x)-f[x,α(x)]α'(x)
$$
上式称为==莱布尼茨公式==。

- **$Γ$函数**
$$
Γ(s)=\int_0^{+∞}e^{-x}x^{s-1}\mathrm{d}x\quad(s>0)
$$
Gamma 函数可以写成如下两个积分之和
$$
Γ(s)=\int_0^{1}e^{-x}x^{s-1}\mathrm{d}x+\int_1^{+∞}e^{-x}x^{s-1}\mathrm{d}x=\Phi(s)+\Psi(s)
$$
其中 $\Phi(s)$ 在$s⩾1$ 是正常积分，在 $0<s<1$ 区间是收敛的无界反常积分，$\Psi(x)$在 $s>0$ 区间是收敛的无界反常积分收敛
(1) $Γ(s)$ 在定义域 $s>0$ 内连续，且有任意阶导数
$$
Γ^{(n)}(s)=\int_0^{+∞}e^{-x}x^{s-1}(\ln x)^n\mathrm{d}x\quad(s>0)
$$
(2) 分部积分可得递推公式
$$
Γ(s+1)=sΓ(s)
$$
由此可推广阶乘：$s!\overset{def}{=} Γ(s+1),s>0$
(3) 其他公式
==余元公式==：$Γ(s)Γ(1-s)=\dfrac{π}{\sinπ s}$
==概率论常用积分==：
$Γ(s)$换元推导出 $\displaystyle\int_0^{+∞}e^{-u^2}\mathrm{d}u=\dfrac{\sqrt π}{2}$
![](https://gitee.com/WilenWu/images/raw/master/math/gamma-fun.png)

# 曲线积分与曲面积分

## 对弧长的曲线积分(第一类曲线积分)
- **引入意义**
线密度$μ=μ(x,y)$构件在曲线$L$的质量 $m=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}μ(ξ_i,η_i)Δs_i$
![](https://gitee.com/WilenWu/images/raw/master/math/curvilinear-integral.png)

- **概念**：设 $L$ 为$xOy$平面的一条光滑曲线弧，定义曲线积分(Curvilinear Integral )
$$
\int_Lf(x,y)\mathrm{d}s=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i)Δs_i
$$
其中$f(x,y)$叫被积函数，$L$叫做积分弧段
(1) 上述定义可以推广至 $f(x,y,z)$ 在空间弧段 $Γ$上的曲线积分
$$
\displaystyle\int_{Γ}f(x,y,z)\mathrm{d}s=\lim\limits_{λ\to0}\sum_{i=1}^{n}f(ξ_i,η_i,ζ_i)Δs_i
$$
(2) 如果 $L$ 是闭曲线，曲线积分记作 
$$
\displaystyle\oint_Lf(x,y)\mathrm{d}s
$$

| 对弧长曲线积分的部分性质                                     |
| :----------------------------------------------------------- |
| $\displaystyle\int_Lkf(x,y)\mathrm{d}s=k\int_Lf(x,y)\mathrm{d}s$          |
| $\displaystyle\int_L [f(x,y)± g(x,y)]\mathrm{d}s=\int_L f(x,y)\mathrm{d}s ± \int_L g(x,y)\mathrm{d}s$ |
|$\displaystyle\int_{L_1+L_2}f(x,y)\mathrm{d}s=\int_{L_1}f(x,y)\mathrm{d}s+\int_{L_2}f(x,y)\mathrm{d}s$ |

- **对弧长曲线积分的计算方法**
设被积函数$f(x,y)$，积分弧段 $L$ 的参数方程为$\begin{cases}x=ϕ(t)\\y=ψ(t)\end{cases}\quad t\in[α,β]$，且$ϕ^{'2}(t)+ψ^{'2}(t)\neq0$，则曲线积分
$$
\int_Lf(x,y)\mathrm{d}s=\int_{α}^{β}f[ϕ(t),ψ(t)]\sqrt{ϕ^{'2}(t)+ψ^{'2}(t)}\mathrm{d}t,\quad  (α<β)
$$
## 对坐标的曲线积分(第二类曲线积分)

- **引入意义**：求变力 $\mathbf F=P(x,y)\mathbf i+Q(x,y)\mathbf j$ 沿曲线 $L$所做的功
$ΔW_i=\mathbf F(ξ_i,η_i)\cdot \overrightarrow{M_{i-1}M_i}=P(ξ_i,η_i)Δx_i+Q(ξ_i,η_i)Δy_i$
$W=\displaystyle\sum_{i=1}^{n}ΔW_i=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}[P(ξ_i,η_i)Δx_i+Q(ξ_i,η_i)Δy_i]$
![](https://gitee.com/WilenWu/images/raw/master/math/curvilinear-integral-typeII.png)

- **概念**：设 $L$ 为$xOy$平面上从点A到点B的一条有向光滑曲线弧，定义
$$
\int_LP(x,y)\mathrm{d}x=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i)Δx_i \\ 
\int_LQ(x,y)\mathrm{d}y=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i)Δy_i 
$$
其中$P(x,y),Q(x,y)$叫被积函数，$L$叫做积分弧段
(1) 上述定义可以推广至$f(x,y,z)$在空间弧段$Γ$上的第二类曲线积分
(2) 应用上经常出现的是
$$
\int_LP(x,y)\mathrm{d}x+\int_LQ(x,y)\mathrm{d}y
$$
简写为
$$
\int_LP(x,y)\mathrm{d}x+Q(x,y)\mathrm{d}y
$$
也可以写成==向量形式==
$$
\int_L\mathbf F(x,y)\cdot \mathrm{d}\mathbf r
$$

   其中 $\mathbf F(x,y)=P(x,y)\mathbf i+Q(x,y)\mathbf j,\quad\mathbf r=\mathrm{d}x\mathbf i+\mathrm{d}y\mathbf j$

| 对坐标曲线积分的部分性质                                     |
| :----------------------------------------------------------- |
| $\displaystyle\int_Lk\mathbf F(x,y)\cdot \mathrm{d}\mathbf r=k\int_L\mathbf F(x,y)\cdot \mathrm{d}\mathbf r$ |
| $\displaystyle\int_L [\mathbf F_1(x,y)± \mathbf F_2(x,y)]\cdot \mathrm{d}\mathbf r=\int_L \mathbf F_1(x,y)\cdot \mathrm{d}\mathbf r ± \int_L \mathbf F_2(x,y)\cdot \mathrm{d}\mathbf r$ |
| $\displaystyle\int_{L_1+L_2}\mathbf F(x,y)\cdot \mathrm{d}\mathbf r=\int_{L_1}\mathbf F(x,y)\cdot \mathrm{d}\mathbf r+\int_{L_2}\mathbf F(x,y)\cdot \mathrm{d}\mathbf r$ |
| $\displaystyle\int_{L^-}\mathbf F(x,y)\cdot \mathrm{d}\mathbf r=-\int_L\mathbf F(x,y)\cdot \mathrm{d}\mathbf r$ ($L^-$与$L$反向) |

- **对坐标曲线积分的计算方法**
设被积函数 $P(x,y),Q(x,y)$，积分弧段 $L$ 的参数方程为$\begin{cases}x=ϕ(t)\\y=ψ(t)\end{cases}$，当 $t$ 单调的由$α$到$β$时，曲线上的点$M(x,y)$沿  $L$ 从起点 $A$ 到 $B$，且 $ϕ^{'2}(t)+ψ^{'2}(t)\neq0$，则曲线积分
$$
\begin{aligned}
& \int_LP(x,y)\mathrm{d}x+Q(x,y)\mathrm{d}y \\
= & \int_{α}^{β}\{P[ϕ(t),ψ(t)]ϕ'(t)+Q[ϕ(t),ψ(t)]ψ'(t)\}\mathrm{d}t
\end{aligned}
$$

- **两类曲线积分之间的联系**
空间有向曲线弧 $Γ$ 两类曲线积分的关系
$$
\int_{Γ}P\mathrm{d}x+Q\mathrm{d}y+R\mathrm{d}z=\int_{Γ}(P\cosα+Q\cosβ+R\cosγ) \mathrm{d}s
$$
其中$α(x,y,z),β(x,y,z),γ(x,y,z)$为有向曲线弧 $Γ$ 在点$(x,y,z)$处的切向量的方向角[^cos]，也可以用向量的形式表述
$$
\int_{Γ}\mathbf A\cdot \mathrm{d}\mathbf r=\int_{Γ}\mathbf A\cdot \mathbfτ \mathrm{d}s
$$
其中 $\mathbf A=(P,Q,R),\mathbfτ=(\cosα,\cosβ,\cosγ)$，$\mathrm{d}\mathbf r=\mathbfτ \mathrm{d}s=(\mathrm{d}x,\mathrm{d}y,\mathrm{d}z)$ 称为有向曲线元。

## 格林公式

- **平面闭区域D的一些概念**
  (1) 若D内任一闭曲线所围成的区域都属于D，则D为单连通区域，否则为复连通区域。
  (2) 如图复连通区域，外边界曲线的正向为逆时针方向，内边界曲线的正向为顺时针方向
  ![](https://gitee.com/WilenWu/images/raw/master/math/planar-region.png)

   <kbd>格林公式</kbd>(Green formula) 设闭区域 $D$ 由分段光滑曲线 $L$ 围成
$$
\iint\limits_D(\dfrac{∂Q}{∂x}-\dfrac{∂P}{∂y})\mathrm{d}x\mathrm{d}y=\oint_LP\mathrm{d}x+Q\mathrm{d}y
$$
其中 $L$ 是 $D$ 取正向的边界曲线，其中复连通区域 $L$ 应取全部边界曲线。

- **平面上曲线积分与积分路径无关的条件**
设区域 $G$为单连通区域，若函数$P(x,y),Q(x,y)$在 $G$ 内具有一阶连续偏导数，$L$ 为 $G$ 内任意闭曲线，则下面的四种说法等价
(1) 在区域 $G$ 内存在可微函数 $u(x,y)$，使全微分 $\mathrm{d}u=P\mathrm{d}x+Q\mathrm{d}y$
(2) 在区域内成立 $\dfrac{∂ Q}{∂ x}=\dfrac{∂ P}{∂ y}$
(3) 对于区域内的任何光滑曲线$L$，均有 $\displaystyle\oint_LP\mathrm{d}x+Q\mathrm{d}y=0$ (如图$L_1+L_2^-$) 
(4) 对于区域内的任何两点 $A, B$，积分$\displaystyle\int_{L_{AB}}P\mathrm{d}x+Q\mathrm{d}y$ 的值只与 $A, B$ 两点的位置有关，而与$L_{AB}$在区域 $G$内的==路径无关==
($\star$) 要求的所有条件都要满足，若区域 $G$ 内含有==奇点==（存在破坏函数$P,Q,\dfrac{∂ Q}{∂ x},\dfrac{∂ P}{∂ y}$连续性的点），则定理不成立
![](https://gitee.com/WilenWu/images/raw/master/math/path-independence.png)

## 对面积的曲面积分(第一类曲面积分)
- **引入意义**：求面密度为$μ=μ(x,y,z)$的曲面的质量$m=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}μ(ξ_i,η_i,ζ_i)ΔS_i$
![曲面积分](https://gitee.com/WilenWu/images/raw/master/math/surface-integral-typeI.png)
- **概念**：设曲面 $Σ$ 是光滑的，函数$f(x,y,z)在Σ$ 上有界，定义曲面积分(Surface Integral)
$$
\iint\limits_{Σ}f(x,y,z)\mathrm{d}s=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i,ζ_i)ΔS_i
$$
其中$f(x,y,z)$是被积函数，$Σ$称为积分曲面

- **对面积的曲面积分的计算法**：设积分曲面$Σ$由方程$z=z(x,y)$给出，$Σ在xOy$平面上的投影区域为$D_{xy}$
$$
\iint\limits_{Σ}f(x,y,z)\mathrm{d}s=\iint\limits_{D_{xy}}f(x,y,z(x,y))\sqrt{1+z_x^2(x,y)+z_y^2(x,y)} \mathrm{d}x\mathrm{d}y
$$
如果$Σ$ 由方程 $x=x(y,z)$ 或 $y=y(x,z)$ 给出，也可类似的把曲面积分化成对应的二重积分。
$(\star)$ 如果$Σ$是闭曲面，曲面积分记作 
$$
\displaystyle\oiint\limits_Σf(x,y,z)\mathrm{d}s
$$
## 对坐标的曲面积分(第二类曲面积分)
- **有向曲面的一些概念**：通常我们遇到的曲面都是双侧的，有上侧和下侧，闭合曲面有外侧和内侧之分，我们取==曲面法向量的方向确定曲面的方向==，若曲面的法向量 $\mathbf n$ 朝上，我们就认为取定曲面的上侧，法向量朝外，我们就认为取定曲面的外侧。

- **引入意义**
设稳定流动(流速与时间无关)不可压缩流体(假定密度为1)的速度场为$\mathbf v(x,y,z)=P(x,y,z)\mathbf i+Q(x,y,z)\mathbf j+R(x,y,z)\mathbf k$，求单位时间流过有向曲面 $Σ$ 指定侧的质量，即流量 $Φ$
![](https://gitee.com/WilenWu/images/raw/master/math/oriented-surface1.png) ![](https://gitee.com/WilenWu/images/raw/master/math/oriented-surface2.png)

(1) 取$Σ$ 的一小块$ΔS_i$，单位法向量$\mathbf n_i=\cosα_i\mathbf i+\cosβ_i\mathbf j+\cosγ_i\mathbf j$，流量(流体质量)$ΔΦ_i=\mathbf v_i\cdot\mathbf n_iΔS_i$
(2) 于是$Φ=\displaystyle\sum_{i=1}^{n}ΔΦ_i=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}[P(ξ_i,η_i,ζ_i)\cosα_i+Q(ξ_i,η_i,ζ_i)\cosβ_i+R(ξ_i,η_i,ζ_i)\cosγ_i]ΔS_i$
(3) 又 $ΔS_i$ 在 $yOz$ 面的投影 $(ΔS_i)_{yz}\approx\cosα_iΔS_i$，其余面的投影类似，所以
$Φ=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}[P(ξ_i,η_i,ζ_i)(ΔS_i)_{yz}+Q(ξ_i,η_i,ζ_i)(ΔS_i)_{xz}+R(ξ_i,η_i,ζ_i)(ΔS_i)_{xy}]$

- **概念**：设$Σ$为光滑有向曲面，函数$R(x,y,z)在Σ$上有界，定义函数对坐标 $x,y$ 的曲面积积分
$$
\iint\limits_{Σ}R(x,y,z)\mathrm{d}x\mathrm{d}y=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i,ζ_i)(ΔS_i)_{xy} 
$$
其中$R(x,y,z)$叫被积函数，$Σ$叫做积分曲面
(1) 类似还可定义其他坐标轴的曲面积分
$$
\iint\limits_{Σ}P(x,y,z)\mathrm{d}y\mathrm{d}z=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i,ζ_i)(ΔS_i)_{yz} \\ 
\iint\limits_{Σ}Q(x,y,z)\mathrm{d}x\mathrm{d}z=\lim\limits_{λ\to0}\displaystyle\sum_{i=1}^{n}f(ξ_i,η_i,ζ_i)(ΔS_i)_{xz}
$$
以上三个曲面积分也称第二类曲面积分。
(2) 上述流量 $Φ$ 可写为
$$
\displaystyle\iint\limits_{Σ}P(x,y,z)\mathrm{d}y\mathrm{d}z+\iint\limits_{Σ}Q(x,y,z)\mathrm{d}x\mathrm{d}z+\iint\limits_{Σ}R(x,y,z)\mathrm{d}x\mathrm{d}y
$$
简写为
$$
\iint\limits_{Σ}P(x,y,z)\mathrm{d}y\mathrm{d}z+Q(x,y,z)\mathrm{d}x\mathrm{d}z+R(x,y,z)\mathrm{d}x\mathrm{d}y
$$
也可以写成==向量形式==
$$
\iint\limits_{Σ}\mathbf v(x,y,z)\cdot \mathrm{d}\mathbf S
$$
其中 $\mathbf v(x,y,z)=P(x,y,z)\mathbf i+Q(x,y,z)\mathbf j+R(x,y,z)\mathbf k$ ，$\mathrm{d}\mathbf S=\mathrm{d}y\mathrm{d}z\mathbf i+\mathrm{d}x\mathrm{d}z\mathbf j+\mathrm{d}x\mathrm{d}y\mathbf k$ 称为有向曲面元。


| 对坐标曲面积分的部分性质                                     |
| :----------------------------------------------------------- |
| $\displaystyle\iint\limits_{Σ}k\mathbf v\cdot \mathrm{d}\mathbf S=k\iint\limits_{Σ}\mathbf v\cdot \mathrm{d}\mathbf S$ |
| $\displaystyle\iint\limits_{Σ} (\mathbf v_1± \mathbf v_2)\cdot \mathrm{d}\mathbf S=\iint\limits_{Σ} \mathbf v_1\cdot \mathrm{d}\mathbf S ± \iint\limits_{Σ} \mathbf v_2\cdot \mathrm{d}\mathbf S$ |
| $\displaystyle\iint\limits_{Σ_1+Σ_2}\mathbf v\cdot \mathrm{d}\mathbf S=\iint\limits_{Σ_1}\mathbf v\cdot \mathrm{d}\mathbf S+\iint\limits_{Σ_2}\mathbf v\cdot \mathrm{d}\mathbf S$  ($Σ_1$与$Σ_2$无公共点) |
| $\displaystyle\iint\limits_{Σ^-}\mathbf v\cdot \mathrm{d}\mathbf S=-\iint\limits_{Σ}\mathbf v\cdot \mathrm{d}\mathbf S$ 

- **对坐标曲面积分的计算方法**：设光滑曲面 $Σ:z=f(x,y),(x,y)\in D_{xy}, D_{xy}为Σ在xOy$平面上的投影区域，$R(x,y,z)为Σ$上的连续函数，按对坐标曲面积分的定义有
$$
\iint\limits_{Σ}R(x,y,z)\mathrm{d}x\mathrm{d}y=±\iint\limits_{D_{xy}}R[x,y,z(x,y)]\mathrm{d}x\mathrm{d}y
$$
等式右端的符号由法向量 $\mathbf n与z$轴的夹角$γ$决定，==锐正钝负==，同样可得
$$
\iint\limits_{Σ}P(x,y,z)\mathrm{d}y\mathrm{d}z=±\iint\limits_{D_{yz}}R[x(y,z),y,z]\mathrm{d}y\mathrm{d}z \\
\iint\limits_{Σ}Q(x,y,z)\mathrm{d}x\mathrm{d}y=±\iint\limits_{D_{xz}}R[x,y(y,z),z]\mathrm{d}y\mathrm{d}z
$$
![](https://gitee.com/WilenWu/images/raw/master/math/surface-integral-typeII.png)
- **两类曲面积分之间的联系**：空间有向曲面$Σ$两类曲面积分的关系
$$
\iint\limits_{Σ}P(x,y,z)\mathrm{d}y\mathrm{d}z=\iint\limits_{Σ}P(x,y,z)\cosα \mathrm{d}s \\
\iint\limits_{Σ}Q(x,y,z)\mathrm{d}x\mathrm{d}y=\iint\limits_{Σ}Q(x,y,z)\cosβ \mathrm{d}s \\
\iint\limits_{Σ}R(x,y,z)\mathrm{d}x\mathrm{d}y=\iint\limits_{Σ}R(x,y,z)\cosγ \mathrm{d}s
$$
其中$α(x,y,z),β(x,y,z),γ(x,y,z)$为有向曲面$Σ$在点$(x,y,z)$处的切向量的方向角[^cos]，合并上面的方程
$$
\iint\limits_{Σ}P\mathrm{d}y\mathrm{d}z+Q\mathrm{d}x\mathrm{d}z+R\mathrm{d}x\mathrm{d}y=\iint\limits_{Σ}(P\cosα+Q\cosβ+R\cosγ) \mathrm{d}s
$$
也可以用向量的形式表述
$$
\iint\limits_{Σ}\mathbf A\cdot \mathrm{d}\mathbf S=\iint\limits_{Σ}\mathbf A\cdot \mathbf n \mathrm{d}s
$$
其中$\mathbf A=(P,Q,R),\mathbf n=(\cosα,\cosβ,\cosγ)$为有向曲面在点$(x,y,z)$处的单位法向量，$\mathrm{d}\mathbf S=\mathbf n\mathrm{d}s=\mathrm{d}y\mathrm{d}z\mathbf i+\mathrm{d}x\mathrm{d}z\mathbf j+\mathrm{d}x\mathrm{d}y\mathbf k$ 称为有向曲面元。

## 高斯公式

<kbd>高斯公式</kbd>：(Gauss formula)  设空间闭区域 $Ω$ 由分片光滑的闭曲面 $Σ$ 所围成，函数$P(x,y,z), Q(x,y,z), R(x,y,z)$ 在 $Ω$上有连续的一阶偏导数，则有
$$
\begin{aligned} 
\iiint\limits_{Ω}(\dfrac{∂P}{∂x}+\dfrac{∂Q}{∂y}+\dfrac{∂R}{∂z})\mathrm{d}v&=\oiint\limits_{Σ}P\mathrm{d}y\mathrm{d}z+Q\mathrm{d}x\mathrm{d}z+R\mathrm{d}x\mathrm{d}y \\
&=\oiint\limits_{Σ}(P\cosα+Q\cosβ+R\cosγ) \mathrm{d}s
\end{aligned}
$$
取闭曲面 $Σ$ 的外侧为正向，$\cosα,\cosβ,\cosγ$为曲面$Σ$在点$(x,y,z)$处的方向余弦[^cos]

![](https://gitee.com/WilenWu/images/raw/master/math/Gauss-formula.png)

引进矢量微分算子[^nabla]
$$
∇=\dfrac{∂}{∂x}\mathbf i+\dfrac{∂}{∂y}\mathbf j+\dfrac{∂}{∂z}\mathbf k
$$
由此高斯公式可简写为向量形式
$$
\iiint\limits_{Ω}∇\cdot \mathbf A\mathrm{d}v=\oiint\limits_{Σ}\mathbf A\cdot \mathrm{d}\mathbf S=\oiint\limits_{Σ}\mathbf A\cdot \mathbf n \mathrm{d}s
$$
其中 $\mathbf A=(P,Q,R),\mathbf n=(\cosα,\cosβ,\cosγ),\mathrm{d}\mathbf S=\mathbf n\mathrm{d}s=\mathrm{d}y\mathrm{d}z\mathbf i+\mathrm{d}x\mathrm{d}z\mathbf j+\mathrm{d}x\mathrm{d}y\mathbf k$

[^nabla]: 矢量微分算子
$$
∇=\dfrac{∂}{∂x}\mathbf i+\dfrac{∂}{∂y}\mathbf j+\dfrac{∂}{∂z}\mathbf k
$$
称为哈密顿算子(Hamiltonian)，读作nabla。
哈密顿算子有如下基本性质：
(1) $∇(cu)=c∇u$ 
(2) $∇(u±v)=∇u+∇v$
(3) $∇(uv)=u∇v+v∇u,\quad ∇(u^2)=2u(∇u)$
(4) $∇\dfrac{u}{v}=\dfrac{v∇u-u∇v}{v^2}$

**沿任意闭曲面的曲面积分为零的条件**：若空间区域$Ω$内任一闭曲面所围成的区域全属于$Ω$,则称$Ω$为空间==二维单连通域==，如图
![](https://gitee.com/WilenWu/images/raw/master/math/2ds-simplyconnected-domain.png)
<kbd>定理</kbd>：若空间闭区域$Ω$为二维单连通域，函数在闭区域上有一阶连续偏导数，则 $Ω$ 沿任意闭曲面的曲面积分为零的充要条件是
$$
\dfrac{∂P}{∂x}+\dfrac{∂Q}{∂y}+\dfrac{∂R}{∂z}=0 \quad (x,y,z)\inΩ
$$


## 斯托克斯公式
<kbd>斯托克斯公式</kbd>：(Stokes formula) 设光滑曲面$Σ$的边界$Γ$是分段光滑曲线,$Σ$的侧与$Γ$的正向符合右手法则,函数$P(x,y,z), Q(x,y,z), R(x,y,z)在Σ$上具有连续一阶偏导数，则有
$$
\begin{aligned}
& \iint\limits_{Σ}(\dfrac{∂R}{∂y}-\dfrac{∂Q}{∂z})\mathrm{d}y\mathrm{d}z+
(\dfrac{∂P}{∂z}-\dfrac{∂R}{∂x})\mathrm{d}x\mathrm{d}z
+(\dfrac{∂Q}{∂x}-\dfrac{∂P}{∂y})\mathrm{d}x\mathrm{d}y \\
= & \oint_{Γ}P\mathrm{d}x+Q\mathrm{d}y+R\mathrm{d}z
\end{aligned}
$$
![](https://gitee.com/WilenWu/images/raw/master/math/right-hand-rule.png) ![](https://gitee.com/WilenWu/images/raw/master/math/Stokes-formula.png)


 为便于记忆，斯托克斯公式也可以写为
$$
\iint\limits_{Σ}
\begin{vmatrix} 
\mathrm{d}y\mathrm{d}z & \mathrm{d}z\mathrm{d}x &\mathrm{d}x\mathrm{d}y \\ 
\dfrac{∂}{∂x} & \dfrac{∂}{∂y} &\dfrac{∂}{∂z} \\
P & Q &R
\end{vmatrix}=\oint_{Γ}P\mathrm{d}x+Q\mathrm{d}y+R\mathrm{d}z
$$
或 (利用两类曲面积分的关系)
$$
\iint\limits_{Σ}
\begin{vmatrix} 
\cosα & \cosβ &\cosγ \\ 
\dfrac{∂}{∂x} & \dfrac{∂}{∂y} &\dfrac{∂}{∂z} \\
P & Q &R
\end{vmatrix}=\oint_{Γ}P\mathrm{d}x+Q\mathrm{d}y+R\mathrm{d}z
$$

**空间曲线积分和路径无关的条件**
设G是空间一维单连通域,函数 $P(x,y,z), Q(x,y,z), R(x,y,z)$在G内具有连续一阶偏导数,则下列四个说法等价:
(1) 对G内任意分段光滑闭曲线$Γ$，有$\displaystyle\oint_{Γ}P\mathrm{d}x+Q\mathrm{d}y+R\mathrm{d}z=0$
(2) 对G内任意分段光滑闭曲线$Γ$，有$\displaystyle\oint_{Γ}P\mathrm{d}x+Q\mathrm{d}y+R\mathrm{d}z$与路径无关
(3) 在G内存在某一函数 $u(x,y,z),\mathrm{d}u=P\mathrm{d}x+Q\mathrm{d}y+R\mathrm{d}z$，这时可求得$\displaystyle u=\int_{(x_0,y_0,z_0)}^{(x,y,z)}P\mathrm{d}x+Q\mathrm{d}y+R\mathrm{d}z$ 如下图
(4) 在G内处处有 $\dfrac{∂P}{∂y}=\dfrac{∂Q}{∂x},\dfrac{∂Q}{∂z}=\dfrac{∂R}{∂y},\dfrac{∂R}{∂x}=\dfrac{∂P}{∂z}$
![](https://gitee.com/WilenWu/images/raw/master/math/space-path-independence.png)

<kbd>曲线积分的基本定理</kbd>：设 $\mathbf A$ 是空间区域 $Ω$ 内的一个有势场，即存在数量函数满足  $∇f=\mathbf A$ ，则空间任意曲线积分 $\displaystyle \int_Γ\mathbf A\cdot \mathrm{d}\mathbf r$ 在 $Ω$ 内与路径无关，且
$$
\int_Γ\mathbf F\cdot \mathrm{d}\mathbf r=f(B)-f(A)
$$
其中 $Γ$ 是 $Ω$ 内起点为 $A$，终点为 $B$ 的任意一条分段光滑曲线。
## 场论初步

- **场的概念**：若对空间区域 $Ω$ 中每一点 $M$ 都有一个数量或向量与之对应，则称在 $Ω$ 上给定了一个==数量场==(scalar field)或==向量场==(vector field)。例如，温度和密度都是数量场，重力和速度都是向量场。
在引进了直角坐标系后，点 $M$ 的位置可由坐标确定。因此给定了某个数量场就等于给定了一个数量函数 $u(x,y,z)$ 。同理，每个向量场都与某个向量函数 $\mathbf A(x,y,z)=P(x,y,z)\mathbf i+Q(x,y,z)\mathbf j+R(x,y,z)\mathbf k$ 相对应，这里 $P,Q,R$ 是定义域上的数量函数。

    > 场的性质是它本身的属性，和坐标系的引进无关。引入或选择某种坐标系是为了便以通过数学方法来进行计算和研究它的性质。

- **等值面**：在三维数量场中，满足方程
$$
u(x,y,z)=c
$$
的所有点通常是一个曲面。在曲面上函数 $u$ 都取同一个常数 $c$ ，被称为==等值面==。例如温度场中等温面、三维等高线等。
![](https://gitee.com/WilenWu/images/raw/master/math/isosurface.png)

- **向量场线**：设 $L$ 为向量场中一条曲线，若 $L$ 上每点 $M$ 处的切线方向都与向量函数 $\mathbf A$ 在该点的方向一致，即
$$
\cfrac{\mathrm dx}{P}=\cfrac{\mathrm dy}{Q}=\cfrac{\mathrm dz}{R}
$$
则称曲线 $L$ 为向量场 $\mathbf A$ 的向量场线。例如电力线、磁力线等都是向量场线。

- **梯度场**：梯度是数量函数 $u(x,y,z)$ 所定义的向量函数
$$
\text{grad }u=\dfrac{∂u}{∂y}\mathbf i+\dfrac{∂u}{∂y}\mathbf j+\dfrac{∂u}{∂z}\mathbf k=∇u
$$
梯度 $\text{grad }u$ 是由数量场 $u$ 派生出的向量场，称为梯度场。由前文知道，$\text{grad }u$ 的方向就是使方向导数 $\dfrac{∂u}{∂l}$ 达到最大值的方向，它的大小就是 $u$ 在这个方向上的方向导数。

   又因为数量场 $u(x,y,z)$ 的等值面 $u(x,y,z)=c$ 的法线方向即是梯度 $\text{grad }u$ 方向，所以梯度方向与等值面正交。等值面上任意一点的单位法向量
$$
\mathbf n=\dfrac{∇u(x,y,z)}{|∇u(x,y,z)|}
$$
于是沿法线的方向导数 $\dfrac{∂u}{∂n}$ 满足
$$
∇u(x,y,z)=\dfrac{∂u}{∂n}\mathbf n
$$


   梯度除具有算符 $∇$ 表示的基本性质外[^nabla]，还有如下性质：
(1) 若 $\mathbf r=(x,y,z),u=u(x,y,z)$ ，则 $\mathrm du=\mathrm d\mathbf r\cdot∇u$
(2) 若 $f=f(u),u=u(x,y,z)$ ，则 $∇f=f'(u)∇u$
(3) 若 $f=f(u_1,u_2,\cdots,u_m),u_i=u_i(x,y,z)$ ，则 $\displaystyle∇f=\sum_{i=1}^{m}\dfrac{∂f}{∂u_i}∇u_i$

   **示例**：设质量为 $m$ 的质点位于原点，$r=\sqrt{x^2+y^2+z^2}$  表示原点与点 $M(x,y,z)$ 间的距离。求数量场 $\cfrac{m}{r}$ 产生的梯度场。
解： $\cfrac{∂}{∂x}(\cfrac{m}{r})=-\cfrac{m}{r^2}\cfrac{∂r}{∂x}=-\cfrac{mx}{r^3}$
同理可求得 $\cfrac{∂}{∂y}(\cfrac{m}{r})=-\cfrac{my}{r^3},\cfrac{∂}{∂z}(\cfrac{m}{r})=-\cfrac{mz}{r^3}$
若以 $\mathbf e_r=\cfrac{x}{r}\mathbf i+\cfrac{y}{r}\mathbf j+\cfrac{z}{r}\mathbf k$  表示 $\vec{OM}$ 的单位向量，则有
$$
\mathrm{grad}\dfrac{m}{r}=-\dfrac{m}{r^2}\mathbf e_r
$$
它表示两质点间的引力，方向朝着原点，大小与质量的乘积成正比，与两点间距离的平方成反比。数量场 $\cfrac{m}{r}$ 的梯度场 $\mathrm{grad}\dfrac{m}{r}=-\dfrac{m}{r^2}\mathbf e_r$ 称为引力场，而函数 $\dfrac{m}{r}$ 称为引力势。

- **散度场**：设 $\mathbf A=P(x,y,z)\mathbf i+Q(x,y,z)\mathbf j+R(x,y,z)\mathbf k$是  $Ω$ 上的向量场，定义数量函数
$$
\text{div }\mathbf{A}=\dfrac{∂P}{∂x}+\dfrac{∂Q}{∂y}+\dfrac{∂R}{∂z}
$$
为向量场 $\mathbf A$ 的==散度==(divergence)，这是由向量场派生出的数量场，称为散度场，也可用微分算子记作
$$
\text{div }\mathbf{A}=∇\cdot\mathbf A
$$
设 $Σ$ 是闭区域 $Ω$ 的边界曲面的外侧， $\mathbf n=(\cosα,\cosβ,\cosγ)$ 为有向曲面 $Σ$ 在各点的单位法向量，$\mathrm{d}\mathbf S=\mathbf n\mathrm{d}s=\mathrm{d}y\mathrm{d}z\mathbf i+\mathrm{d}x\mathrm{d}z\mathbf j+\mathrm{d}x\mathrm{d}y\mathbf k$ 为有向曲面元，于是高斯公式也可以写为
$$
\iiint\limits_Ω\text{div }\mathbf{A}=\oiint\limits_{Σ}\mathbf A\cdot \mathrm{d}\mathbf S
$$

   **流量和散度**：设在闭区域 $Ω$ 内有稳定流动(流速与时间无关)不可压缩流体(假定密度为1)，速度场为$\mathbf A$ 。
(1) 高斯公式右端可解释为单位时间内流向$Σ$指定侧的总==流量==(discharge)或==通量==(flux)
$$
\displaystyleΦ=\oiint\limits_{Σ}\mathbf A\cdot\mathrm{d}\mathbf S
$$
当$Φ>0$ 时, 说明流入 $Σ$ 的流体体积少于流出的；
当$Φ<0$ 时, 说明流入 $Σ$ 的流体体积多于流出的；
当$Φ=0$ 时, 说明流入与流出 $Σ$ 的流体体积相等
(2) 由于我们假定流体是不可压缩且稳定的，高斯公式左端可解释为分布在 $Ω$ 内的源头在单位时间内产生或吸收的流体的总质量。设 $Ω$ 的体积为 $V$，对高斯公式中的三重积分应用中值定理，存在点 $M'(ξ,η,ζ)\inΩ$ 使得
$$
\iiint\limits_Ω\text{div }\mathbf{A}\mathrm{d}v
=V(\text{div }\mathbf A)|_{M'}
=\oiint\limits_{Σ}\mathbf A\cdot \mathrm{d}\mathbf S
$$
在 $Ω$ 内任取一点 $M(x,y,z)$，令 $Ω$ 收缩到 $M$ （记作 $Ω\to M$），则同时有 $M'\to M$ ，对上式取极限得到
$$
\text{div }\mathbf A(M)=\lim\limits_{Ω\to M}\dfrac{Φ}{V}
$$
这个等式可以看作是散度的另一种定义形式。
上式表明散度 $\text{div }\mathbf A(M)$ 是流量对体积 $V$ 的变化率，并称为 $\mathbf A$ 在 $M$ 处的流量密度。
(1) 若 $\text{div }\mathbf{A}>0$，说明在每一单位时间内有一定数量的流体流出该点，则称该点为==源==； 
(2) 若 $\text{div }\mathbf{A}<0$，表明流体在该点被吸收，则称该点为==汇==； 
(3) 若在向量场 $\mathbf A$ 中每一点 $\text{div }\mathbf{A}=0$，则称 $\mathbf A$ 为==无源场==(field without sources)。
散度绝对值的大小反映了源的强度。
![](https://gitee.com/WilenWu/images/raw/master/math/divergence.png)

  **散度的性质**：用微分算符表示 $\mathbf A$ 的散度是
$$
\text{div }\mathbf{A}=∇\cdot\mathbf A
$$
   容易推得散度的以下一些基本性质：设 $\mathbf A,\mathbf B$ 是向量函数，$\phi$ 是数量函数。
(1) $∇\cdot(\mathbf{A+B})=∇\cdot\mathbf A+∇\cdot\mathbf B$
(2) $∇\cdot(\phi\mathbf{A})=\phi∇\cdot\mathbf A+\mathbf A\cdot∇\phi$
(3) $∇\cdot∇\phi=Δ\phi$
其中 $Δ=∇\cdot∇=∇^2$ 称为拉普拉斯算符

   **示例**：求引力场 $\mathbf F=-\dfrac{m}{r^2}\mathbf e_r$ 产生的散度场。其中 $\mathbf e_r=(\cfrac{x}{r},\cfrac{y}{r},\cfrac{z}{r})$  表示单位向量。
   解：因为 $\mathbf F=-\dfrac{m}{r^2}\mathbf e_r=-\dfrac{m}{r^3}(x,y,z)$ 所以
$∇\cdot\mathbf F=-m\Big[\dfrac{∂}{∂x}(\dfrac{x}{r^3})+\dfrac{∂}{∂y}(\dfrac{y}{r^3})+\dfrac{∂}{∂z}(\dfrac{z}{r^3})\Big]$
求偏导 $\dfrac{∂}{∂x}(\dfrac{x}{r^3})=\dfrac{r^2-3x^2}{r^5},\ \dfrac{∂}{∂y}(\dfrac{y}{r^3})=\dfrac{r^2-3y^2}{r^5},\ \dfrac{∂}{∂z}(\dfrac{z}{r^3})=\dfrac{r^2-3z^2}{r^5}$
于是  $∇\cdot\mathbf F=0$ 
因此引力场 $\mathbf A$ 除原点外，每点的散度都为零。

- **旋度场**：设 $\mathbf A=P(x,y,z)\mathbf i+Q(x,y,z)\mathbf j+R(x,y,z)\mathbf k$ 是  $Ω$ 上的向量场，定义向量函数
$$
\text{rot }\mathbf{A}=(\dfrac{∂R}{∂y}-\dfrac{∂Q}{∂z})\mathbf i+(\dfrac{∂P}{∂z}-\dfrac{∂R}{∂x})\mathbf j+(\dfrac{∂Q}{∂x}-\dfrac{∂P}{∂y})\mathbf k
$$
为向量场 $\mathbf A$ 的==旋度==(rotation;curl)，这是由向量场派生出的向量场，称为旋度场，也可由微分算子表示为
$$
\text{rot }\mathbf{A}=∇×\mathbf A
$$
还可用行列式简记为
$$
\text{rot }\mathbf{A}=\begin{vmatrix} 
\mathbf i & \mathbf j &\mathbf k \\ 
\dfrac{∂}{∂x} & \dfrac{∂}{∂y} &\dfrac{∂}{∂z} \\
P & Q &R
\end{vmatrix}
$$
设闭曲线 $Γ$ 是有向曲面 $Σ$ 的边界曲线， $\mathbf τ=(\cosα,\cosβ,\cosγ)$ 为有向曲线 $Γ$ 在各点的单位切向量，$\mathrm{d}\mathbf r=\mathbfτ \mathrm{d}s=(\mathrm{d}x,\mathrm{d}y,\mathrm{d}z)$ 称为有向曲线元，于是斯托克斯公式也可以写为
$$
\iint\limits_{Σ}\text{rot }\mathbf{A}\cdot\mathrm{d}\mathbf S=\oint_{Γ}\mathbf A\cdot\mathrm{d}\mathbf r
$$

   **环流量和旋度**：向量场 $\mathbf A$ 沿闭曲线 $Γ$ 的积分
$$
I=\oint_{Γ}\mathbf A\cdot  \mathrm{d}\mathbf r=\oint_{Γ}P\mathrm{d}x+Q\mathrm{d}y+R\mathrm{d}z
$$
称为环流量(circulation)，它表示流速为 $\mathbf A$ 的不可压缩流体，在单位时间内沿曲线 $Γ$ 流过的总量。
设 $Σ$ 的体积为 $S$，对斯托克斯公式中的曲面积分应用中值定理，存在点 $M'(ξ,η,ζ)\inΣ$ 使得
$$
\iint\limits_{Σ}\text{rot }\mathbf{A}\cdot\mathrm{d}\mathbf S
=S(\text{rot }\mathbf{A}\cdot\mathbf{n})|_{M'}
=\oint_{Γ}\mathbf A\cdot\mathrm{d}\mathbf r
$$
在 $Σ$ 内任取一点 $M(x,y,z)$，令 $Σ$ 收缩到 $M$ （记作 $Σ\to M$），则同时有 $M'\to M$ ，对上式取极限得到
$$
(\text{rot }\mathbf{A}\cdot\mathbf{n})|_{M}=\lim\limits_{Σ\to M}\dfrac{I}{S}
$$
这个等式也可以看作是旋度的另一种定义形式。
上式说明流速 $\mathbf A$ 在 $M$ 处的环流量密度，等于旋度 $\text{rot }\mathbf{A}(M)$ 在法线 $\mathbf n(M)$ 方向上的投影。
这同时指出了旋度的两个基本属性：
(1) 旋度 $\text{rot }\mathbf{A}(M)$ 的方向是 $\mathbf A$  在 $M$ 处的环流量密度最大的方向；
(2) $|\text{rot }\mathbf{A}(M)|$即为上述最大环流密度的数值。
(3) 若在向量场 $\mathbf A$ 中每一点 $\text{rot }\mathbf{A}=0$，则称 $\mathbf A$ 为==无旋场==(irrotational field)。
![](https://gitee.com/WilenWu/images/raw/master/math/rotation.png)

   **旋度的力学意义**：为了更好地认识旋度的物理意义及这一名称的来源，我们讨论刚体绕定轴旋转的问题。
   设某刚体绕定轴转动，角速度为 $\mathbf ω=(ω_x,ω_y,ω_z)$，如图，刚体上任一点 $M(x,y,z)$ 的线速度为
$$
\mathbf v=\mathbf ω×\mathbf r=\begin{vmatrix}
\mathbf i &\mathbf j &\mathbf k \\
ω_x&ω_y&ω_z \\
x&y&z
\end{vmatrix}\\
=(ω_yz-ω_zy,ω_zx-ω_xz,ω_xy-ω_yx)
$$
而
$$
\text{rot }\mathbf{v}=∇×\mathbf v=(2ω_x,2ω_y,2ω_z)=2\mathbf ω
$$
这结果表明线速度 $\mathbf v$ 的旋度除相差一个常数因子外，就是旋转的角速度 $\mathbf ω$ 。这也说明了旋度这个名称的来源。
![](https://gitee.com/WilenWu/images/raw/master/math/rotation-physics.png)

   **散度的性质**：用微分算符表示 $\mathbf A$ 的旋度是
$$
\text{rot }\mathbf{A}=∇×\mathbf A
$$
旋度有如下一些基本性质：设 $\mathbf A,\mathbf B$ 是向量函数，$\phi$ 是数量函数。
(1) $∇×(\mathbf{A+B})=∇×\mathbf A+∇×\mathbf B$
(2) $∇\cdot(\mathbf{A×B})=\mathbf B\cdot∇×\mathbf A-\mathbf A\cdot∇×\mathbf B$    
(3) $∇×(\phi\mathbf{A})=\phi\cdot(∇×\mathbf A)+∇\phi×\mathbf A$
(4) $∇\cdot(∇×\mathbf{A})=0,∇×∇\phi=0$
(5) $∇×(∇×\mathbf{A})=∇(∇\cdot\mathbf{A})-Δ\mathbf{A}$
其中 $Δ=∇\cdot∇=∇^2$ 称为拉普拉斯算符

- **管量场**：若向量场 $\mathbf A$  的散度恒为零，即 $\text{div }\mathbf{A}=0$ 我们曾称 $\mathbf A$ 为无源场。由高斯公式我们知道，此时沿任何封闭曲面的曲面积分都等于零，我们又把 $\mathbf A$ 称为==管量场==。这是因为，若在向量场 $\mathbf A$ 中作一向量管（如图），即由向量线围成的管状的曲面。用断面 $S_1,S_2$ 去截它，以 $S_3$ 表示所截出的管的表面，这就得到了由 $S_1,S_2,S_3$ 所围成的封闭曲面 $S$ 。
![](https://gitee.com/WilenWu/images/raw/master/math/tube-field.png)
于是由高斯公式得到
$$
\iint\limits_{S_1外侧}\mathbf A\cdot \mathrm{d}\mathbf S+
\iint\limits_{S_2外侧}\mathbf A\cdot \mathrm{d}\mathbf S+
\iint\limits_{S_3外侧}\mathbf A\cdot \mathrm{d}\mathbf S=0
$$
   而向量线与曲面 $S_3$ 的法线正交，所以
$$
\iint\limits_{S_3外侧}\mathbf A\cdot \mathrm{d}\mathbf S=0
$$
   于是可得到
$$
\iint\limits_{S_1内侧}\mathbf A\cdot \mathrm{d}\mathbf S=
\iint\limits_{S_2外侧}\mathbf A\cdot \mathrm{d}\mathbf S
$$
   这等式说明了流体通过向量管的任意断面的流量是相同的，所以把无源场称为管量场。
   例如，由 $\cfrac{m}{r}$ 的梯度 $∇(\cfrac{m}{r})$ 所成的引力场 $\mathbf F$ 是一个管量场。

- **有势场**：若向量场 $\mathbf A$  的旋度恒为零，即 $\text{rot }\mathbf{A}=0$ 我们曾称 $\mathbf A$ 为无旋场。由斯托克斯公式我们知道，这时在空间单连通区域内沿任何封闭曲线的曲线积分都等于零，我们又把 $\mathbf A$ 称为==有势场==(potential field)或==保守场==。。由斯托克斯公式知道，此时空间曲线积分与路线无关，且存在某函数 $u(x,y,z)$ 使得
$$
\mathrm du=P\mathrm dx+Q\mathrm dy+R\mathrm dz
$$
即
$$
\text{grad }u=(P,Q,R)
$$
通常称 $u$ 为==势函数==。因此若某向量场 $\mathbf A$ 的旋度为零，则必存在某个势函数  $u$ 使得 $\text{grad }u=\mathbf A$ 。这也是一个向量场是某个数量场的梯度场的充要条件。
例如，引力势 $u=\cfrac{m}{r}$ 就是势函数，所以 $∇u=\mathbf F=-\cfrac{m}{r^2}(\cfrac{x}{r},\cfrac{y}{r},\cfrac{z}{r})$。又因为 $∇×∇u=0$ 恒成立，所以 $∇×\mathbf F=0$ 它也是引力场 $\mathbf F$ 是有势场的充要条件。


- **调和场**：若一个向量场既是管量场，又是有势场，则称这个向量场为==调和场==(harmonic field)。引力场 $\mathbf F$ 就是调和场。
若向量场 $\mathbf A$ 是一个调和场，则必有
$$
∇\cdot\mathbf A=0 ,\quad ∇u=\mathbf A
$$
显然
$$
∇\cdot∇u=Δu=0
$$
即必有势函数 $u$ 满足
$$
\dfrac{∂^2u}{∂x^2}+ \dfrac{∂^2u}{∂y^2}+\dfrac{∂^2u}{∂z^2}=0
$$
这时称函数 $u$ 为调和函数。

