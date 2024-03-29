---
title: 量子场论
categories:
  - Physics
  - 高等物理
katex: true
abbrlink: d56d6970
tags:
cover: /img/Quantum-Field-Theory.png
top_img: /img/math-top-img.png
date:
description: 量子力学、狭义相对论和经典场论相结合的物理理论
---

量子场论(Quantum Field Theory, QFT)是量子力学，狭义相对论和经典场论相结合的物理理论，已被广泛的应用于粒子物理学和凝聚态物理学中。量子场论为描述多粒子系统，尤其是包含粒子产生和湮灭过程的系统，提供了有效的描述框架。量子场论的最初建立历程是和量子力学以及狭义相对论密不可分的，它是基本粒子物理标准模型的理论框架。后来，非相对论性的量子场论也被应用于凝聚态物理学，比如描述超导性的BCS理论。2013年的诺贝尔物理学奖被授予量子场论中希格斯机制的发现者。希格斯粒子也是构造粒子物理标准模型的最后一环。

当前主流尝试理论有弦理论/超弦理论/M理论、超引力、圈量子引力、扭量理论等

[量子场论的数学基础和应用研究](https://zhuanlan.zhihu.com/p/24671230)

<iframe src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/physics/One_galaxy_Lede.mp4?autoplay=0" title='宇宙' scrolling="no" border="0" frameborder="no"
  framespacing="0" allowfullscreen="true" width="100%;" height="500"> </iframe>



量子场论：(Quantum Field Theory, QFT) 目前已知的四种相互作用中，除去引力，另三种相互作用都找到了合适满足特定规范对称性的量子场论（或者说Yang-Mills场）来描述。强相互作用有量子色动力学（QCD，Quantum Chromodynamics)；电磁相互作用有量子电动力学（QED,Quantum Electrodynamics)；弱相互作用有四费米子点作用理论。后来弱相互作用和电磁相互作用实现了形式上的统一，由Yang-Mills（杨-米尔斯）场来描述，通过希格斯机制（Higgs Mechanism）产生质量，建立了弱电统一的量子规范理论，即GWS（Glashow, Weinberg, Salam）模型。

[*如何自学量子场论？*](https://www.zhihu.com/question/24209758/answer/248307405)

# 量子场论导论

An Introduction to Quantum Field Theory



量子物理发展

{% mermaid %}
graph TD
C5(量子力学) -.-> D2
C3(广义相对论) -.-> D2(量子引力)
C4(宇宙学) --> D2

A1(量子力学) --> B1(QED<br />量子电动力学)
A2(狭义相对论) --> B1
A3(电磁场论) --> B1

B1 --> C1(QWED<br />弱电统一理论)
B2(弱相互作用) --规范场论--> C1
B3(强相互作用) --规范场论--> C2(QCD<br />量子色动力学)

C1 --> D1(QFD<br />量子场论)
C2 --> D1

D2 -.-> F(万有理论)
D1 -.-> E(大统一理论)

E -.-> F
{% endmermaid %}



# 量子电动力学

QED,Quantum Electrodynamics

# 规范场论

强，弱相互作用的理论框架

规范场论（Gauge Theory），为量子力学的学科，是基于对称变换可以局部也可以全局地施行这一思想的一类物理理论。非交换对称群（又称非阿贝尔群）的规范场论最常见的例子为杨-米尔斯理论。物理系统往往用在某种变换下不变的拉格朗日量表述，当变换在每一时空点同时施行，它们有全局对称性。规范场论推广了这一思想，它要求拉格朗日量必须也有局部对称性—应该可以在时空的特定区域施行这些对称变换而不影响到另外一个区域。这个要求是广义相对论的等效原理的一个推广。

# 量子色动力学

QCD，Quantum Chromodynamics

# 弱电统一理论

QWED