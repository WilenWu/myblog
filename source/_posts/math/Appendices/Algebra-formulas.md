---
title: 经典数学公式
categories:
  - 数学
  - 附录
tags:
  - 数学
  - 三角函数
  - 数列
  - 因式分解
  - 对数
cover: /img/Algebra-formulas.webp
top_img: /img/math-top-img.png
katex: true
description: false
abbrlink: 375d0704
date: 2019-04-19 13:07:29
---

# 因式分解

- 平方和 $a^2\pm 2ab+b^2=(a\pm b)^2$
- 平方差 $a^2-b^2=(a+b)(a-b)$
- 立方和 $a^3+b^3=(a+b)(a^2-ab+b^2)$
- 立方差 $a^3-b^3=(a-b)(a^2+ab+b^2)$
- $a^n-b^n=(a-b)\displaystyle\sum_{i=1}^n a^{n-i}b^{i-1}$

# 不等式

- 均值不等式 $\cfrac{2}{\cfrac{1}{a}+\cfrac{1}{b} } \leqslant \sqrt{ab} \leqslant \cfrac{a+b}{2} \leqslant \sqrt{\cfrac{a^2+b^2}{2} }$
- 绝对值不等式 $||a|-|b|| \leqslant |a±b| \leqslant |a|+|b|$
- 权方和不等式 $(ac+bd)^2 \leqslant (a^2+b^2)(c^2+d^2)$

# 一元二次方程

$ax^2+bx+c=0(a\neq0)$

- 一元二次方程的解 $x_{1,2}=\dfrac{-b\pm\sqrt{\Delta}}{2a}$，其中判别式 $\Delta=b^2-4ac$
-  根与系数的关系 $\begin{cases} x_1+x_2=-\dfrac{b}{a}\\ x_1x_2=\dfrac{c}{a}\end{cases}$

# 指数与对数

- 指数运算 
  $a^n\cdot a^m=a^{n+m}$
  $\cfrac{a^n}{a^m}=a^{n-m}$
  $(a^n)^m=a^{mn}$
  $(ab)^n=a^n\cdot b^n$
  $a^{-n}=\dfrac{1}{a^n}$
- 对数运算
  $\log_a1=0$
  $\log_ax=\dfrac{\ln x}{\ln a}$ (换底公式)
  $\log_axy=\log_ax+\log_ay$
  $\log_a\dfrac{x}{y}=\log_ax-\log_ay$
  $\log_ax^y=y\log_ax$
- $x=e^{\ln x}$

# 三角恒等式

- 平方关系 
  $\sin^2\alpha+\cos^2\alpha=1$
  $1+\tan^2\alpha=\sec^2\alpha$
  $1+\cot^2\alpha=\csc^2\alpha$
- 两角和差
  $\cos (\alpha \pm \beta )=\cos \alpha  \cos \beta \mp \sin \alpha   \sin \beta$
  $\sin (\alpha \pm \beta )=\sin \alpha   \cos \beta \pm \cos \alpha   \sin \beta$
  $\tan (\alpha \pm \beta )=\dfrac{\tan \alpha \pm \tan \beta }{1\mp \tan \alpha \tan \beta}$
- 和差化积
  $\sin \alpha +\sin \beta =2\sin \dfrac{\alpha +\beta}{2}\cos \dfrac{\alpha -\beta}{2}$
  $\sin \alpha -\sin \beta =2\cos \dfrac{\alpha +\beta}{2}\sin \dfrac{\alpha -\beta}{2}$
  $\cos \alpha +\cos \beta =2\cos \dfrac{\alpha +\beta}{2}\cos \dfrac{\alpha -\beta}{2}$
  $\cos \alpha -\cos \beta =-2\sin \dfrac{\alpha +\beta}{2}\sin \dfrac{\alpha -\beta}{2}$
  $\tan\alpha+\tan\beta=\dfrac{\sin(\alpha+\beta)}{\cos\alpha\cos\beta}$
- 积化和差
  $\sin \alpha \cos \beta =\dfrac{1}{2}[\sin (\alpha +\beta )+\sin (\alpha -\beta )]$
  $\cos \alpha \sin \beta =\dfrac{1}{2}[\sin (\alpha +\beta )-\sin (\alpha -\beta )]$
  $\cos \alpha \cos \beta =\dfrac{1}{2}[\cos (\alpha +\beta )+\cos (\alpha -\beta )]$
  $\sin \alpha \sin \beta =-\dfrac{1}{2}[\cos (\alpha +\beta )-\cos (\alpha -\beta )]$
- 倍角公式
  $\sin 2\alpha=2\sin \alpha   \cos \alpha =\dfrac{2}{\tan \alpha +\cot \alpha}$
  $\cos 2\alpha=\cos^2 \alpha-\sin^2 \alpha$
  $\tan 2\alpha =\dfrac{2\tan \alpha}{1-\tan^2 \alpha}$
  $\cot 2\alpha=\dfrac{\cot^2\alpha -1}{2\cot \alpha}$
  $\sin 3\alpha=3\sin \alpha -4\sin^3 \alpha$
  $\cos 3\alpha=4\cos^3 \alpha -3\cos \alpha$
  $\tan 3\alpha=\dfrac{3\tan \alpha -\tan^3 \alpha}{1-3\tan^2 \alpha}$
  $\cot 3\alpha=\dfrac{\cot^3 \alpha -3\cot \alpha}{3\cot \alpha -1}$
- 半角公式（正负由$\dfrac{\alpha}{2}$所在的象限决定）
  $\sin \dfrac{\alpha}{2}=\pm \sqrt{\dfrac{1-\cos \alpha }{2}}$
  $\cos\dfrac{\alpha}{2}=\pm \sqrt{\dfrac{1+\cos \alpha }{2}}$
  $\tan \dfrac{\alpha}{2}=\pm \sqrt {\dfrac{1-\cos \alpha }{1+\cos \alpha }}=\dfrac{\sin \alpha}  {1+\cos \alpha}=\dfrac{1-\cos \alpha}{\sin \alpha}$
  $\cot\dfrac{\alpha}{2}=\pm \sqrt {\dfrac{1+\cos \alpha}{1-\cos \alpha}}=\dfrac{1+\cos \alpha}  {\sin \alpha} =\dfrac{\sin \alpha}{1-\cot \alpha}$
- 辅助角公式
  $a\sin \alpha +b\cos \alpha =\sqrt{a^2+b^2}\sin (\alpha +\arctan\frac{b}{a})$
  $a\sin \alpha +b\cos \alpha =\sqrt{a^2+b^2}\cos (\alpha -\arctan\frac{a}{b})$
- 万能公式
  $\sin\alpha=\dfrac{2\tan\frac{\alpha}{2}}{1+\tan ^2\frac{\alpha}{2}}$
  $\cos\alpha=\dfrac{1-\tan ^2\frac{\alpha}{2}}{1+\tan ^2\frac{\alpha}{2}}$
  $\tan\alpha=\dfrac{2\tan\frac{\alpha}{2}}{1-\tan ^2\frac{\alpha}{2}}$
- 降幂公式
  $\sin^2 \alpha=\frac{1}{2}(1-\cos 2\alpha)$
  $\cos^2 \alpha=\frac{1}{2}(1+\cos 2\alpha)$
  $\tan^2 \alpha=\dfrac{1-\cos 2\alpha}{1+\cos 2\alpha}$
- 正弦定理 $\dfrac{a}{\sin A}=\dfrac{b}{\sin B}=\dfrac{c}{\sin C}=2R$ ，其中 $R$ 为外接圆半径
- 余弦定理 $c^2=a^2+b^2-2ab\cos C$

# 数列

- 等差数列
  通项公式：$a_n=a_1+(n-1)d$
  求和公式：$S_n=\dfrac{n(a_1+a_n)}{2}$
- 等比数列
  通项公式：$a_n=a_1q^{n-1}(a_n\neq 0,q\neq 0)$
  求和公式：$S_n=\begin{cases} \dfrac{a_1(1-q^n)}{1-q} & \text{if } q\neq 1 \\ na_1 & \text{if } q=1 \end{cases}$

# 排列和组合

- 阶乘：$n!=1\times2\times\cdots (n-2)(n-1)n$
- 排列：$A_n^m=\dfrac{n!}{(n-m)!}$
- 组合：$\complement_n^m=\dfrac{A_n^m}{m!}=\dfrac{n!}{m!(n-m)!}$

