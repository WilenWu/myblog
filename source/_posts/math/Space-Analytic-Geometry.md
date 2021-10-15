---
title: 高等数学(空间解析几何)
date: 2019-05-03 11:22:02
categories: [数学]
tags: [数学,向量代数,解析几何]
cover: 
top_img: 
katex: true
---

# 向量代数

## 空间直角坐标系

(1) 过空间中一定点，作三条互相垂直的数轴构成的坐标系称为空间直角坐标系 (space rectangular coordinate system)
(2) 有序三元数组$(x,y,z)$称为点 $M$ 的坐标(coordinate)，也记为$M(x,y,z)$
(3) 空间中的两点$M_1(x_1,y_1,z_1),M_1(x_1,y_1,z_1)$ 间距离(distance)公式
$$
|M_1M_2|=\sqrt{(x_1-x_2)^2+(y_1-y_2)^2+(z_1-z_2)^2}
$$
![](https://gitee.com/WilenWu/images/raw/master/math/coordinate.png)

## 向量和线性运算 
- **向量的记法**：加粗 (如 **a、b、u、v**)或者加小箭头（如 $\vec a,\vec b, \overrightarrow{AB}$）

- **向量的大小**：叫做向量的模(modulus)，记作 |**a**| ,$|\vec b|, |\overrightarrow{AB}|$，模等于1的向量叫做单位向量(unit vector)记作 $\vec e$ 或 **e**，模等于0的叫做零向量(zero vector)，记作 $\vec 0$ 或 **0**

- **向量夹角**(vector angel)：$(\widehat{\mathbf a,\mathbf b})=\varphi,\varphi\in[0,\pi]$
![](https://gitee.com/WilenWu/images/raw/master/math/vector-angel.png) 
- **向量的线性运算**
加法：$\mathbf{c=a+b}$ 遵循平行四边形法则（或三角形法则）
数乘：$λ\mathbf a$ 方向与 $\mathbf a$ 相同，模 $|λ\mathbf a|=|λ||\mathbf a|$
![](https://gitee.com/WilenWu/images/raw/master/math/parallelogram-law.png) ![](https://gitee.com/WilenWu/images/raw/master/math/triangle-law.png)  

向量运算|表达式
:---|:---
交换律|$a+b=b+a$
结合律|$a+(b+c)=(a+b)+c$
数乘|$λ(μa)=μ(λa)=(λμ)a \\ (λ+μ)a=λa+μa \\ λ(a+b)=λa+λb$


- **向量坐标表示**
(1) 向量 $\mathbf a$ 平行于 $\mathbf b\iff$存在唯一的数 $\lambda$，使得 $\mathbf b=\lambda \mathbf a$
(2) 点 $M$，向量 $\mathbf r$ 与三个有序实数对有一一对应关系
$M\longleftrightarrow \mathbf r=\overrightarrow{OM}=x\mathbf i+y\mathbf j+z\mathbf k\longleftrightarrow (x,y,z)$
![](https://gitee.com/WilenWu/images/raw/master/math/vector-coordinate.png)
- **方向角和方向余弦**：向量 $\mathbf r$ 与三个坐标轴的夹角 $\alpha,\beta,\gamma$ 称为==方向角==，$\cosα,\cosβ,\cosγ$ 称为 $\mathbf r$ 的==方向余弦==。
![](https://gitee.com/WilenWu/images/raw/master/math/direction-cosine-math.png)

坐标运算|$\mathbf a=(x_1,y_1,z_1),\mathbf b=(x_2,y_2,z_2),\mathbf c=(x_3,y_3,z_3)\\ \mathbf r=(x,y,z)$
:---|:---
向量的模(modulus)|$\mid \mathbf r\mid=\sqrt{x^2+y^2+z^2}$
方向角余弦(direction cosine)|$(\cosα,\cosβ,\cosγ)=\dfrac{1}{\mathbf r}(x,y,z)=\mathbf e_r \\ \cos^2α+\cos^2β+\cos^2γ=1$
向量在坐标轴上的投影(projection)|$(x_1,y_1,z_1)=(\mathrm{Prj}_x\mathbf a,\mathrm{Prj}_y\mathbf a,\mathrm{Prj}_z\mathbf a)或\\ (x_1,y_1,z_1)=(\ (\mathbf a)_x,(\mathbf a)_y,(\mathbf a)_z\ )$
加法(plus)|$\mathbf a+\mathbf b=(x_1+x_2,y_1+y_2,z_1+z_2)$
数乘(scalar-multiplication)|$λ \mathbf a=(λ x_1,λ y_1,λ z_1)$
数量积(dot product)|$\mathbf a\cdot\mathbf b=\mid \mathbf a\mid\mid \mathbf b\mid\cos(\widehat{\mathbf a,\mathbf b})=\mid \mathbf a\mid\mathrm{Prj}_a\mathbf b=x_1x_2+y_1y_2+z_1z_2$
向量积(cross product)|$\mathbf a\times\mathbf b=\begin{vmatrix} \mathbf i&\mathbf j&\mathbf k  \\ x_1 &y_1&z_1\\  x_2 &y_2&z_2\end{vmatrix}$
混合积(mixed product)|$[\mathbf a\mathbf b\mathbf c]=(\mathbf a\times\mathbf b)\cdot\mathbf c=\begin{vmatrix}x_1 &y_1&z_1  \\ x_2 &y_2&z_2\\  x_3&y_3&z_3\end{vmatrix}$<br>几何意义：平行六面体的体积

**结论**
(1) $\mathbf a\cdot\mathbf b=0 \iff \mathbf a\bot\mathbf b$
(2) $\mathbf a \times\mathbf b=0 \iff \mathbf a\parallel\mathbf b$
(2) $[\mathbf a\mathbf b\mathbf c]=0 \iff \mathbf a,\mathbf b,\mathbf c$ 共面

![](https://gitee.com/WilenWu/images/raw/master/math/mixed-product.png)

# 空间解析几何
## 平面及其方程
- **平面点法式方程**：已知平面上一点 $M_0(x_0,y_0,z_0)$ 和平面法向量(normal vector) $\mathbf=(A,B,C)$ 利用向量的垂直关系 
$$
\mathbf n\cdot\overrightarrow{M_0M}=0
$$
于是得到方程
$$
A(x-x_0)+B(y-y_0)+C(z-z_0)=0
$$
![](https://gitee.com/WilenWu/images/raw/master/math/plane-equation.png)
- **平面一般方程** 
$$
Ax+By+Cz+D=0
$$
- **平面的截距方程** (intercept form)：平面依次与 $x,y,z$ 三轴的截距为 $a,b,c$ 
$$
\dfrac{x}{a}+\dfrac{y}{b}+\dfrac{z}{c}=0
$$
![](https://gitee.com/WilenWu/images/raw/master/math/intercept-form-equation.png)

- **两平面的夹角**(included angle)：两平面法线的夹角。
$$
\cos\theta=|\cos(\widehat{\mathbf n_1,\mathbf n_2})|
$$
![](https://gitee.com/WilenWu/images/raw/master/math/plane-included-angle.png)

## 空间直线及其方程
- **空间直线的一般方程**：两平面的交线
$$
\begin{cases} A_1x+B_1y+C_1z+D_1=0\\ A_2x+B_2y+C_2z+D_2=0 \end{cases}
$$
- **直线的对称式方程**(点向式方程)
如果一个非零向量平行于已知直线，则称为==方向向量==(direction vector) 
已知过直线上一点 $M_0(x_0,y_0,z_0)$ 和 它的方向向量 $\mathbf s=(m,n,p)$ ，由平行关系可知
$$
\mathbf s\times\overrightarrow{M_0M}=0
$$
于是可求得
$$
\dfrac{x-x_0}{m}= \dfrac{y-y_0}{n}= \dfrac{z-z_0}{p}
$$
方向向量 $\mathbf s$ 的坐标 $m,n,p$ 叫做直线的方向数，方向向量的方向余弦叫做直线的方向余弦。
很容易导出直线的参数方程，设 
$$
\dfrac{x-x_0}{m}= \dfrac{y-y_0}{n}= \dfrac{z-z_0}{p}=t
$$
于是**参数方程**为
$$
\begin{cases}x=x_0+mt \\y=y_0+nt\\z=z_0+pt\end{cases}
$$
- **两直线的夹角**：两直线方向向量的夹角 
$$
\cos\varphi=|\cos(\widehat{\mathbf s_1,\mathbf s_2})|
$$
-  **直线与平面的夹角**：直线与在平面上投影直线的夹角 
$$
\sin\varphi=|\cos(\widehat{\mathbf s,\mathbf n})|
$$
## 曲面及其方程
曲面(surface)的一般方程 $F(x,y,z)=0$
曲面参数方程 $\begin{cases}x=x(s,t)\\ y=y(s,t)\\ z=z(s,t) \end{cases}$

- **旋转曲面**(surface of revolution)：有一条平面曲线绕其平面上一条直线旋转一周围成的曲面叫旋转曲面。旋转曲线和定直线依次叫做曲面的母线(generating line)和轴(axis)。
设 $yOz$ 坐标面上有一曲线 $C$，方程为
$$
f(y,z)=0
$$
绕  $z$ 轴旋转的曲面方程为
$$
f(\pm\sqrt{x^2+y^2},z)=0
$$
$x$轴、$y$ 轴同理
![](https://gitee.com/WilenWu/images/raw/master/math/surface-of-revolution.png)
   (1) 圆锥曲面(conic surface)：$z^2=a^2(x^2+y^2)\quad a=\cot\alpha$
![](https://gitee.com/WilenWu/images/raw/master/math/conic-surface.png)

   (2) 双曲面(hyperboloid)：$xOz$坐标面上的双曲线$\dfrac{x^2}{a^2}-\dfrac{z^2}{c^2}=1$
旋转单叶双曲面：（绕 $z$ 轴旋转）$\dfrac{x^2+y^2}{a^2}-\dfrac{z^2}{c^2}=1$
旋转双叶双曲面：（绕 $x$ 轴旋转）$\dfrac{x^2}{a^2}-\dfrac{y^2+z^2}{c^2}=1$
![](https://gitee.com/WilenWu/images/raw/master/math/hyperboloid.png)

- **柱面**(cylinder)：直线 $L$（母线）沿曲线 $C$（准线）平行移动形成的轨迹。
一般的只含 $x,y$ 而缺 $z$ 的方程 $F(x,y)=0$ 表示母线平行于 $z$ 轴的柱面，其准线是 $xOy$ 平面上的曲线 $F(x,y)=0$，$x,y$ 轴类似。
![](https://gitee.com/WilenWu/images/raw/master/math/cylinder.png)

-  **二次曲面**(quadric surface)：与曲线类似，我们把三元二次方程形成的曲面叫做二次曲面，平面称为一次曲面。
(1) 椭圆锥面(elliptic cone) $\dfrac{x^2}{a^2}+\dfrac{y^2}{b^2}=z^2$
![](https://gitee.com/WilenWu/images/raw/master/math/elliptic-cone.png)
(2) 椭球面(ellipsoid) $\dfrac{x^2}{a^2}+\dfrac{y^2}{b^2}+\dfrac{z^2}{c^2}=1$
![](https://gitee.com/WilenWu/images/raw/master/math/ellipsoid.png)
(3) 单叶双曲面(hyperboloid of one sheet) $\dfrac{x^2}{a^2}+\dfrac{y^2}{b^2}-\dfrac{z^2}{c^2}=1$
(4) 双叶双曲面(hyperboloid of two sheets) $\dfrac{x^2}{a^2}-\dfrac{y^2}{b^2}-\dfrac{z^2}{c^2}=1$
(5) 椭圆抛物面(elliptic paraboloid) $\dfrac{x^2}{a^2}+\dfrac{y^2}{b^2}=z$
![](https://gitee.com/WilenWu/images/raw/master/math/elliptic-paraboloid.png)
(6) 双曲抛物面(hyperbolic paraboloid)（马鞍面） $\dfrac{x^2}{a^2}-\dfrac{y^2}{b^2}=z$
![](https://gitee.com/WilenWu/images/raw/master/math/hyperbolic-paraboloid.png)
(7) 椭圆柱面(elliptic cylinder) $\dfrac{x^2}{a^2}+\dfrac{y^2}{b^2}=1$
(8) 双曲柱面(hyperbolic cylinder) $\dfrac{x^2}{a^2}-\dfrac{y^2}{b^2}=1$
(9) 抛物柱面(parabolic cylinder)  $x^2=ay$

## 空间曲线及其方程
(1) 空间曲线(space curve)的一般方程 $\begin{cases}F(x,y,z)=0\\ G(x,y,z)=0 \end{cases}$
(2) 曲线参数方程 $\begin{cases}x=x(t)\\ y=y(t)\\ z=z(t) \end{cases}$
==螺旋线==(helix) $\begin{cases}x=a\cos\omega t\\ y=a\sin\omega t\\ z=vt \end{cases}$ 或 $\begin{cases}x=a\cos\theta\\ y=a\sin\theta\\ z=b\theta\end{cases}$
![](https://gitee.com/WilenWu/images/raw/master/math/helix.png)
(3) 空间曲线在坐标面上的投影
空间曲线 $C$ 一般方程消去 $z$ 得到 $H(x,y)=0$，由上节知道这是母线平行于 $z$ 轴的柱面，曲线 $C$ 的所有点满足方程，都在柱面上，此柱面叫做曲线 $C$ 在坐标平面 $xOy$上的投影柱面，投影曲线方程为
$$
\begin{cases}H(x,y)=0\\ z=0 \end{cases}
$$
其余坐标面类似。