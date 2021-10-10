---
title: ToE 概述
date: ‎2021‎-08‎-0‎8‎ ‏‎10:43:20
katex: true
---

[万有理论][toe]（英语：Theory of Everything或ToE）指的是假定存在的一种具有总括性、一致性的物理理论框架，能够解释宇宙的所有物理奥秘。

[toe]: https://baike.baidu.com/item/%E4%B8%87%E6%9C%89%E7%90%86%E8%AE%BA/630145?fr=aladdin

包括光学、电学等

或者直接更新四大力学

```mermaid
graph TD
A1(电学) --> B(电磁学)
A2(磁学) --> B
B --> C(弱电相互作用)
B2(弱相互作用) --> C
C --> D1(粒子物理学标准模型)
D1 --> E1(电核力<br />大统一理论)
D2(宇宙学标准模型) --> E2(引力)
E1 --> F(量子引力)
E2 --> F
F --> G(万有理论)
```

三角形内角和

$$
\alpha+\beta+\gamma
\begin{cases}
=\pi &\text{欧几里得几何} \\
<\pi &\text{双曲几何(罗巴切夫斯基几何)}\\  >\pi &\text{椭圆几何(黎曼几何)}
\end{cases}
$$
量子物理发展

```mermaid
graph TD
C3(广义相对论) -.-> D2(量子引力<br />弦论, 圈量子引力等)
C4(宇宙学) --> D2

A1(量子力学) --> B1(QED<br />量子电动力学)
A2(狭义相对论) --> B1
A3(电磁场论) --> B1

B1 --> C1(QWED<br />弱电统一理论)
B2(弱相互作用) --规范场论--> C1
B3(强相互作用) --规范场论--> C2(QCD<br />量子色动力学)

C1 --> D1(QFD<br />量子场论)
C2 --> D1

D2 -.-> E(大统一理论)
D1 -.-> E

E -.-> F(万有理论)
```

相对论性量子力学发展

```mermaid
graph LR
A(Klein-Gordan 方程) --> B(Dirac 方程<br />单粒子) 
B --> C(QED<br />多粒子) 
```

