---
title: KaTeX 基本数学符号
categories:
  - 'Markup Language'
tags:
  - 标记语言
  - 数学
  - markdown
  - katex
katex: true
cover: img/katex.png
sticky: 1
abbrlink: b9652820
date: 2019-04-22 18:37:20
---

[KaTeX](https://katex.org/docs/supported.html) 是一个快速，易于使用的JavaScript库，用于在Web上进行TeX数学渲染。
KaTeX兼容所有主流浏览器，包括Chrome，Safari，Firefox，Opera，Edge和IE 9-11。
KaTeX支持很多（但不是全部）LaTeX语法和许多LaTeX软件包。

<!-- more -->

# 字母符号

$\Alpha$ \Alpha|$\alpha$ \alpha|$\Tau$ \Tau|$\tau$ \tau
:--|:--|:--|:--
$\Beta$ \Beta|$\beta$ \beta|$\Upsilon$ \Upsilon|$\upsilon$ \upsilon
$\Gamma$ \Gamma|$\gamma$ \gamma|$\Phi$ \Phi|$\phi$ \phi
$\Delta$ \Delta|$\delta$ \delta|$\Chi$ \Chi|$\chi$ \chi
$\Epsilon$ \Epsilon|$\epsilon$ \epsilon|$\Psi$ \Psi|$\psi$ \psi
$\Zeta$ \Zeta|$\zeta$ \zeta|$\Omega$ \Omega|$\omega$ \omega
$\Eta$ \Eta|$\eta$ \eta|$\varPi$ \varPi|$\varpi$ \varpi
$\Theta$ \Theta|$\theta$ \theta|$\varSigma$ \varSigma|$\varsigma$ \varsigma
$\Iota$ \Iota|$\iota$ \iota|$\varTheta$ \varTheta|$\vartheta$ \vartheta
$\Kappa$ \Kappa|$\kappa$ \kappa|$\varPhi$ \varPhi|$\varphi$ \varphi
$\Lambda$ \Lambda|$\lambda$ \lambda|$\varGamma$ \varGamma|$\varepsilon$ \varepsilon
$\Mu$ \Mu|$\mu$ \mu|$\varDelta$ \varDelta|$\varkappa$ \varkappa
$\Nu$ \Nu|$\nu$ \nu|$\varLambda$ \varLambda|$\thetasym$ \thetasym
$\Xi$ \Xi|$\xi$ \xi|$\varXi$ \varXi|$\varrho$ \varrho
$\Omicron$ \Omicron|$\omicron$ \omicron|$\varUpsilon$ \varUpsilon|$\digamma$ \digamma
$\Pi$ \Pi|$\pi$ \pi|$\varPsi$ \varPsi|$\imath$ \imath
$\Rho$ \Rho|$\rho$ \rho|$\varOmega$ \varOmega|$\jmath$ \jmath
$\Sigma$ \Sigma|$\sigma$ \sigma|$\mho$ \mho|$\ell$ \ell


$\wp$ \wp;\weierp|$\aleph$ \aleph|$\Game$ \Game|$\Bbbk$ \Bbbk
:--|:--|:--|:--
$\alef$ \alef|$\Finv$ \Finv|$\text{\AA}$ \text{\AA}|$\text{\aa}$ \text{\aa}
$\beth$ \beth|$\gimel$ \gimel|$\daleth$ \daleth|$\eth$ \eth
|$\hbar$ \hbar|$\hslash$ \hslash|$\text{\AE}$ \text{\AE}|$\text{\oe}$ \text{\oe}

常用标记|定义|Latex
:---|:---|:---
$\breve{a}$||\breve{a}
$\check{a};\widecheck{ac}$||\check{a};\widecheck{ac}
$\tilde{a};\widetilde{ac};\utilde{AB}$|波浪|\tilde{a};\widetilde{ac};\utilde{AB}
$\acute{a}$||\acute{a}
$\grave{a}$||\grave{a}
$a_n$|下标|a_n
$\hat a$|帽子|\hat a
$\bar a$|短线|\bar a

# 运算符

二元运算|定义|Latex
:---|:---|:---
$=$|等于|=
$\approx$|约等于|\approx
$\propto$|正比于|\propto
$+$|加|+
$-$|减|-
$\pm; \mp$||\pm; \mp
$\times$|乘|\times
$\cdot$|点乘|\cdot; \centerdot
$*$|卷积|*;\ast
$\div; /$|除|\div; /
$<$|小于|<;\lt
$>$|大于|>;\gt
$\ll;\lll$|远小于|\ll\lll
$\gg;\ggg$|远大于|\gg;\ggg
$\geqslant;\ge$|大于等于|\geqslant\ge
$\leqslant;\le$|小于等于|\leqslant;\le
$\not=;\not\in$|前方加`\not`否定|\not=;\not\in
$\&$|与|\\&
$\mid$|或|\mid
$\dfrac{b}{a}$|分数|\frac{b}{a}; \dfrac{b}{a}
$\cfrac{a}{1 + \cfrac{1}{b}}$|复合分式|\cfrac{a}{1+\cfrac{1}{b}}
$f \circ g$|复合函数|f \circ g


一元运算|定义|Latex
:---|:---|:---
$\vert a \vert$|绝对值|\vert;  \|
$\|a\|$|范数|\Vert; \\|
$\lceil a\rceil$|ceiling|\lceil a\rceil
$\lfloor a\rfloor$|floor|\lfloor a\rfloor
$\lfloor a\rceil$|最接近的整数|\lfloor a\rceil
$a^n$|指数|a^n
$\sqrt{x}; \sqrt[n]{x}$|开方|\sqrt{x}; \sqrt[n]{x}
$\bar{a};\overline{a+bi}$|共轭|\bar{a}; \overline{a+bi}
$\lmoustache\rmoustache$|胡须|\lmoustache\rmoustache
$\ulcorner \urcorner$||\ulcorner\urcorner
$\llcorner\lrcorner$||\llcorner\lrcorner
$\uparrow;\downarrow;\updownarrow$||\uparrow;\downarrow;\updownarrow
$\Uparrow;\Downarrow;\Updownarrow$||\uparrow;\downarrow;\updownarrow


多元运算|定义|Latex|示例
:---|:---|:---|:---
$\sum$|求和|\sum
$\prod$|求积|\prod
$\bigcap$|交集|\bigcap
$\bigcup$|并集|\bigcup
$\amalg$|合并|\amalg

**括号**

`\left(\LARGE{AB}\right)`
$$
\left(\LARGE{AB}\right)
$$

`( \big( \Big( \bigg( \Bigg(`
$$
( \big( \Big( \bigg( \Bigg(
$$


# 逻辑符号

符号|定义|Latex
:---|:---|:---
$\because$|因为|\because
$\therefore$|所以|\therefore
$\neg;\sim$|逻辑非|\neg; \lnot; \sim
$\land$|逻辑与|\land
$\lor$|逻辑或|\lor
$\oplus; \veebar$|异或|\oplus; \veebar
$\iff; \leftrightarrow$|等价，当且仅当(if and only if)|\iff; \leftrightarrow
$\Rarr; \rarr$|条件运算,if ... then|\Rarr; \rarr ;\to
$\implies; \impliedby$||\implies; \impliedby
$\Larr; \larr$|左箭头|\Larr; \larr; \gets
$:=$|定义|:=
$\triangleq$|定义|\triangleq
$\forall$|任意|\forall
$\exists$|存在|\exists
$\exists!$|唯一存在|\exists!
$\vDash$|满足符|\vDash
$\vdash$|推断出|\vdash
$\square$|拟态词必然|\square
$\Diamond$|拟态词可能|\Diamond

堆叠|定义|Latex
:---|:---|:---
$\stackrel{!}{=}$|堆叠|\stackrel{!}{=}
$\overset{!}{=}$|上方|\overset{!}{=}
$\underset{!}{=}$|下方|\underset{!}{=}
$a \atop b$||a \atop b
$a\raisebox{0.25em}{b}c$||a\raisebox{0.25em}{b}c
# 注释

符号|定义|Latex
:---|:---|:---
$\text{\sect}$|分节|\text{\sect}
$\star$|星号|\star
$\cancel{5}$|左划线|\cancel{5}
$\bcancel{5}$|右划线|\bcancel{5}
$\xcancel{abc}$|交叉划线|\xcancel{5}
$\sout{5}$|划线|sout{5}
$\boxed{\pi=\frac c d}$|方框|\boxed{\pi=\frac c d}
$\overbrace{a+b+c}^{\text{note}}$|上备注|\overbrace{a+b+c}^{\text{note}}
$\underbrace{a+b+c}_{\text{note}}$|下备注|\underbrace{a+b+c}_{\text{note}}

`\tag{hi} x+y^{2x}`
$$
x+y^{2x} \tag{hi}
$$

`\tag*{hi} x+y^{2x}`
$$
x+y^{2x}\tag*{hi}
$$

`\left.\begin{bmatrix}a & b \\ c & d\end{bmatrix}\right\}rows`
$$
\left.\begin{bmatrix}a & b \\ c & d\end{bmatrix}\right\}\text{2 rows}
$$

# 集合和映射

符号|定义|Latex
:---|:---|:---
$\{x\mid x<5\}$|集合|`\{x\mid x<5\}`
$\set{x\mid x<5}$|集合|\set{x\mid x<5}
$\mathring{U}$|邻域|\mathring{U}
$\uplus$|多重集|\uplus
$\subset$|真子集|\subset
$\subseteq$|子集|\subseteq
$\supset$|真父集|\supset
$\supseteq$|父集|\supseteq
$\in$|属于|\in
$\ni$|属于|\ni 
$\cap$|交集|\cap
$\cup$|并集|\cup
$\setminus$|差集|\setminus
$\mathrm{card}(A)$|元素个数|\mathrm{card}(A)
$\emptyset; \varnothing$|空集|\emptyset; \varnothing
$\N$|自然数|\N
$\Z$|整数|\Z
$\R;\Re$|实数|\R;\Reals;\Re
$\Im$|虚数|\Im; \Image
$\Complex$|复数|\Complex
$n!$|阶乘|n!
$\binom{n}{k}$, ${n\brack k}$|组合|\binom{n}{k}; \dbinom{n}{k}; <br>{n \choose k}; n\brack k
${n\brace k}$||{n\brace k}
$A^k_n$|排列|A^n_m
$\complement^k_n$|组合|\complement^n_m
$\mapsto$|映射|\mapsto
$\xmapsto{over}$|条件映射|\xmapsto{over}
$\xrightarrow[under]{over}$|条件映射|\xrightarrow[under]{over}


# 几何

符号|定义|Latex
:---|:---|:---
$\backsim$|相似三角形|\backsim
$\backsimeq$||\backsimeq
$\overset{\backsim}{=}$|全等三角形|\overset{\backsim}{=}
$\parallel$|平行|\parallel
$\nparallel$|不平行|\nparallel
$\bot$|垂直|\bot; \perp
$\overline{AB}$|直线|\overline{AB}
$\underline{AB}$||\underline{AB}
$\overlinesegment{AB}$|线段|\\overlinesegment{AB}
$\overgroup{AB};\undergroup{AB}$||\overgroup{AB};\undergroup{AB}
$\underlinesegment{AB}$||\underlinesegment{AB}
$\overset{\frown}{AB}$|弧|\overset{\frown}{AB}
$\odot$|圆|\odot
$\bigcirc$||\bigcirc
$\boxdot$||\boxdot
$\square$|矩形|\square
$\mathrm{Rt}\triangle$|直角三角形|\mathrm{Rt}\triangle
$\Diamond$|菱形|\Diamond
$\angle$|角|\angle
$\measuredangle$||\measuredangle
$90\degree$|角度|90\degree
$\hat{\imath}$|坐标基|\hat{\imath}
$\hat{\jmath}$|坐标基|\hat{\jmath}
$\widehat{ac}$|向量夹角|\widehat{ac}
$\vec{a},\overrightarrow{AB};\overrightharpoon{ac}$|向量|\vec{a},\overrightarrow{AB};<br/>\overrightharpoon{ac}
$\underrightarrow{AB}$||\underrightarrow{AB}
$\overleftarrow{AB};\overleftharpoon{ac};\underleftarrow{AB}$||\overleftarrow{AB};<br/>\overleftharpoon{ac};<br/>\underleftarrow{AB}
$\Overrightarrow{AB}$||\Overrightarrow{AB}
# 函数

$\arcsin$  `\arcsin`|$\cotg$  `\cotg`|$\ln$  `\ln`|$\det$  `\det`
:---|:---|:---|:---
$\arccos$  `\arccos`|$\coth$  `\coth`|$\log$  `\log`|$\gcd$  `\gcd`
$\arctan$  `\arctan`|$\csc$  `\csc`|$\sec$  `\sec`|$\inf$  `\inf`
$\arctg$  `\arctg`|$\ctg$  `\ctg`|$\sin$  `\sin`|$\lim$  `\lim`
$\arcctg$  `\arcctg`|$\cth$  `\cth`|$\sinh$  `\sinh`|$\liminf$  `\liminf`
$\arg$  `\arg`|$\deg$  `\deg`|$\sh$  `\sh`|$\limsup$  `\limsup`
$\ch$  `\ch`|$\dim$  `\dim`|$\tan$  `\tan`|$\max$  `\max`
$\cos$  `\cos`|$\exp$  `\exp`|$\tanh$  `\tanh`|$\min$  `\min`
$\cosec$  `\cosec`|$\hom$  `\hom`|$\tg$  `\tg`|$\Pr$  `\Pr`
$\cosh$  `\cosh`|$\ker$  `\ker`|$\th$  `\th`|$\sup$  `\sup`
$\cot$  `\cot`|$\lg$  `\lg`|$\operatorname{f}$  `\operatorname{f}`|$\arg\max$  `\arg\max`
$\arg\min$  `\arg\min`|||


# 微积分

符号|定义|Latex
:---|:---|:---
$\to$|趋向于|\to
$\gets$||\gets
$\infty$|无穷大|\infty
$\lim\limits_{x\to 0}$|极限|\lim\limits_{x\to 0} 
$\dot{x}$|导数|\dot{x}
$\ddot{x}$|二阶导|\ddot{x}
$x'$|导数|x'; x^\prime
$x''$|二阶导|x''
$x^{(n)}$|n阶导|x^{(n)}
$\partial x$|偏导数|\partial x
$\mathrm{d}x$|微分|\mathrm{d}x
$\int$|积分|\int
$\iint$|积分|\iint
$\iiint$|积分|\iiint
$\oint$|积分|\oint
$\oiint$|积分|\oiint
$\oiiint$|积分|\oiiint
$\nabla$|微分算子|\nabla
$\Delta$|拉普拉斯算子|\Delta
$\Box$|非欧拉普拉斯算子|\Box
$\mathrm{grad}$|梯度|\mathrm{grad}

```
\iiint\limits_{Ω}(\dfrac{∂P}{∂x}+\dfrac{∂Q}{∂y}+\dfrac{∂R}{∂z})\mathrm{d}V=
\oiint\limits_{Σ}P\mathrm{d}y\mathrm{d}z+Q\mathrm{d}x\mathrm{d}z+R\mathrm{d}x\mathrm{d}y
```
$$
\iiint\limits_{Ω}(\dfrac{∂P}{∂x}+\dfrac{∂Q}{∂y}+\dfrac{∂R}{∂z})\mathrm{d}V=\oiint\limits_{Σ}P\mathrm{d}y\mathrm{d}z+Q\mathrm{d}x\mathrm{d}z+R\mathrm{d}x\mathrm{d}y
$$

# 线性代数

表示|定义|Latex
:---|:---|:---
$\mathbf{a}$|向量（粗体）|\mathbf{a}
$A$ |矩阵（大写）|A
$\overleftrightarrow{T}$|张量|\overleftrightarrow{T}
$\underleftrightarrow{AB}$||\underleftrightarrow{AB}
$\mathcal{T}$|张量（花体）|\mathcal{T}
$f(x)=\begin{cases} a &\text{if } b \\   c &\text{if } d \end{cases}$|定义方程|`f(x)=\begin{cases}`   <br>`a &\text{if } b \\` <br>   `c &\text{if } d` <br>`\end{cases}`
$\begin{alignedat}{2}   10&x+ &3&y = 2 \\  3&x+&13&y = 4 \end{alignedat}$|方程组|`\begin{alignedat}{2`}   <br>`10&x+ &3&y = 2 \\`   <br>`3&x+&13&y = 4` <br>`\end{alignedat}`
$\begin{aligned}  f(x) &=(m+n)^2 \\ & =m^2+2m+n^2 \end{aligned}$|多行等式|`\begin{aligned} ` <br>`f(x) &=(m+n)^2 \\` <br>`& =m^2+2m+n^2` <br>`\end{aligned}`
$\begin{matrix}   a & b \\   c & d\end{matrix}$|数组|`\begin{matrix}`   <br>`a & b \\`   <br>`c & d `<br>`\end{matrix}`
$\begin{array}{cc}   a & b \\   c & d\end{array}$|数组|`\begin{array}{cc} `   <br>`a & b \\`   <br>`c & d `<br>`\end{array}`
$\begin{pmatrix}   a & b \\   c & d\end{pmatrix}$|矩阵|`\begin{pmatrix}`   <br>`a & b \\`   <br>`c & d` <br>`\end{pmatrix}`
$\begin{bmatrix}   a & b \\   c & d\end{bmatrix}$|矩阵|`\begin{bmatrix`}   <br>`a & b \\`   <br>`c & d` <br>`\end{bmatrix}`
$\begin{vmatrix}   a & b \\   c & d\end{vmatrix}$|行列式|`\begin{vmatrix}`   <br>`a & b \\`   <br>`c & d`  <br>`\end{vmatrix}`
$\begin{Vmatrix}   a & b \\   c & d\end{Vmatrix}$|范式|`\begin{Vmatrix}`   <br>`a & b \\ ` <br>`c & d`  <br>`\end{Vmatrix}`
$\begin{Bmatrix}a & b \\ c & d\end{Bmatrix}$|花括号|`\begin{Bmatrix}`<br>`a & b \\` <br>`c & d `<br>`\end{Bmatrix}`
$\begin{bmatrix} \begin{array}{c:c:c}   a & b & c \\ \hline   d & e & f \\   \hdashline   g & h & i\end{array}\end{bmatrix}$|分块矩阵|`\begin{bmatrix}` <br>`\begin{array}{c:c:c}`   <br>`a & b & c \\` <br>`\hline`   <br>`d & e & f \\`   <br>`\hdashline`   <br>`g & h & i`<br>`\end{array}`<br>`\end{bmatrix}`
$\xrightarrow[under]{over}$|变换|\xrightarrow[under]{over}
$\to$|变换|\to
$A^\top$|矩阵转置|A^\top
$A\cong B$|矩阵等价|A\cong B
$A\sim B$|矩阵相似|A\sim B
$A\simeq B$|矩阵合同|A\simeq B
$\bar{A}$|增广矩阵|\bar{A}
$A^*$|伴随矩阵|A^*
$\det A;\vert A \vert$|矩阵的行列式|\det A
$\mathrm{diag}(1,2,3)$|对角阵|\mathrm{diag}(1,2,3)
$A\otimes B$|克罗内克积|\otimes
$\dim V$|空间维度|\dim V
$b\perp V$|正交|b \perp B
$\wedge$|外积|\wedge
$\text{tr}(A)$|迹|\text{tr}(A)
$\text{spec}(A)$|谱|\text{spec}(A)
$\cdots$|横点|\cdots
$\vdots$|竖点|\vdots
$\ddots$|对角点|\ddots
$\lang\psi\mid\phi\rang$|左矢;右矢|\lang\psi\mid\phi\rang
$\bra{\phi}$|左矢|\bra{\phi}; \Bra{\phi}
$\ket{\psi}$|右矢|\ket{\psi}; \Ket{\psi}
$\braket{\phi\mid\psi}$|狄拉克符号|\braket{\phi\mid\psi}
$\hom(U,V)$|线性映射的集合|\hom(U,V)
$\ker(A)$|零空间（核）|\ker(A)
$\text{span}(u,v)$|张成空间|\text{span}(u,v)

```md
\begin{pmatrix}
a_{11}&a_{12}&\cdots&a_{1n} \\
a_{21}&a_{22}&\cdots&a_{2n} \\
\vdots&\vdots&\ddots&\vdots \\
a_{m1}&a_{m2}&\cdots&a_{mn} \\
\end{pmatrix},

\begin{bmatrix} 
\begin{array}{cc:c|c} 
1&0 & 0 & 0 \\ 
0&1 & 0 &0 \\ 
\hdashline 
0&0 & 1 & 5
\end{array}
\end{bmatrix}
```

$$
\begin{pmatrix}
a_{11}&a_{12}&\cdots&a_{1n} \\
a_{21}&a_{22}&\cdots&a_{2n} \\
\vdots&\vdots&\ddots&\vdots \\
a_{m1}&a_{m2}&\cdots&a_{mn} \\
\end{pmatrix}, \quad
\begin{bmatrix} 
\begin{array}{cc:c|c} 
1&0 & 0 & 0 \\ 
0&1 & 0 &0 \\ 
\hdashline 
0&0 & 1 & 5
\end{array}
\end{bmatrix}
$$

# 现代数学

符号|定义|Latex
:---|:---|:---
$g \circ f$|复合|g \circ f
$b\pmod m$||b\pmod m
$a \bmod b$||a \bmod b
$x \pod a$||x \pod a
$\equiv$|同余关系|\equiv
$\gtrdot$||\gtrdot
$\lessdot$||\lessdot
$\intercal$|区间|\intercal
$\rhd$|双方关系对立|\rhd
$\lhd$|正规子群|\lhd
$\unrhd$||\unrhd
$\unlhd$||\unlhd
$\leftthreetimes$||\leftthreetimes
$\rightthreetimes$||\rightthreetimes
$\rtimes$||\rtimes
$\ltimes$||\ltimes
$\prec$|卡普可约|\prec
$\succ$||\succ
$\mid$|因式分解|\mid
$\nmid$||\nmid
$\gcd$|最大公约数|\gcd
$\deg$|多项式次数|\deg

