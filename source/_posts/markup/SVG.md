---
title: SVG 入门教程
date: 2021-12-12 20:13:00
updated: 
tags: [标记语言, XML]
categories: [标记语言]
katex: true
cover: /img/SVG-with-crown.svg
top_img: img/svg-cover.jpeg
description: 
---

# SVG 引言

## 简介

[**可缩放矢量图形**](https://www.w3.org/Graphics/SVG/) <img src="https://gitee.com/WilenWu/images/raw/master/common/SVG-with-crown.svg" width="20%" align="right"/>（Scalable Vector Graphics，SVG）是一种用于描述二维的矢量图形，基于 XML 的标记语言。SVG由W3C制定，是一个开放标准，能够优雅而简洁地渲染不同大小的图形，并和CSS，DOM[^dom]，JavaScript和SMIL等其他网络标准无缝衔接。本质上，SVG 相对于图像，就好比 HTML 相对于文本。

SVG 图像及其相关行为被定义于 XML 文本文件之中，这意味着可以对它们进行搜索、索引、编写脚本以及压缩。此外，这也意味着可以使用任何文本编辑器和绘图软件来创建和编辑它们。

和传统的点阵图模式不同，SVG格式提供的是矢量图，这意味着它的图像能够被无限放大而不失真或降低质量，并且可以方便地修改内容。

<img src="https://gitee.com/WilenWu/images/raw/master/common/SVG-Bitmap.svg" width="50%"/>

SVG1.1的第二个版本在2011年成为推荐标准，W3C工作组还在2003年推出了SVG Tiny (SVGT) 和SVG Basic (SVGB)。这两个配置文件主要瞄准移动设备。首先SVG Tiny主要是为性能低的小设备生成图元，而SVG Basic实现了完整版SVG里的很多功能，只是舍弃了难以实现的大型渲染（比如动画）。

[^dom]: DOM (Document Object Model) 译为**文档对象模型**，是 HTML 和 XML 文档的编程接口。DOM 定义了访问和操作网页文档的标准方法。

## 文件类型

SVG文件有两种形式。普通SVG文件是包含SVG标记的简单文本文件。推荐使用 `.svg`（全部小写）作为此类文件的扩展名。

由于在某些应用（比如地图应用等）中使用时，SVG文件可能会很大，SVG标准同样允许gzip压缩的SVG文件。推荐使用 `.svgz`（全部小写）作为此类文件扩展名 。

## 编辑工具

所有的主流浏览器都将支持SVG：Internet Explorer 9、Mozilla Firefox、Safari、Google Chrome和Opera。基于Webkit的移动设备浏览器（主要是指iOS和Android），都支持SVG。在较老或者较小的设备上，一般支持SVG Tiny。

- SVG 图像可以通过使用矢量图形编辑器来生成，开源软件有 [Inkscape](https://www.inkscape.org/)、Scribus、Karbon14 、Sodipodi 以及 [Apache Batik](https://xmlgraphics.apache.org/batik/) 工具集等
- 商用编辑工具有 Adobe Illustrator、Adobe Flash Professional 或 CorelDRAW
- 也有开放源码，功能简单但容易操作，免安装的在线SVG设计工具，例如 [SVG-Edit](https://github.com/SVG-Edit/svgedit) ，[SVG 菜鸟在线编辑器（国内）](https://c.runoob.com/more/svgeditor/) 和 [Figma: the collaborative interface design tool.](https://www.figma.com/)
- 在移动设备上的软件有安卓的 PainterSVG

## SVG in HTML

SVG 文件可以独立使用，或者通过以下几种方法嵌入到HTML文件中：

- 如果HTML是XHTML并且声明类型为`application/xhtml+xml`，可以直接使用`<svg>` 标签嵌入到XML源码中。

- 如果HTML是HTML5并且浏览器支持HTML5，同样可以直接嵌入SVG。然而为了符合HTML5标准，可能需要做一些语法调整。

- 可以通过 `<object>`元素引用SVG文件。缺点是不允许使用脚本。

  ```html
  <object data="image.svg" type="image/svg+xml" />
  ```

- `<embed>` 标签被所有主流的浏览器支持，并允许使用脚本：

  ```html
  <embed src="image.svg" type="image/svg+xml" />
  ```

- 类似的也可以使用 `<iframe>`元素引用SVG文件：

  ```html
  <iframe src="image.svg"></iframe>
  ```

- 同样可以使用`<img>` 标签以图片的形式直接引入，但是在低于4.0版本的Firefox 中不起作用：

  ```html
  <img src="image.svg" />
  ```

- 最后SVG可以通过JavaScript动态创建并注入到HTML DOM中。 这样具有一个优点，可以对浏览器使用替代技术，在不能解析SVG的情况下，可以替换创建的内容。

# SVG 基础

正式开始之前，你需要基本掌握XML和其它标记语言比如说HTML，如果你不是很熟悉XML，这里有几个重点一定要记住：

- SVG的元素和属性必须按标准格式书写，因为XML是区分大小写的（这一点和HTML不同）
- SVG里的属性值必须用引号引起来，就算是数值也必须这样做。

SVG是一个庞大的规范，本教程主要涵盖基础内容。掌握了这些内容之后，你就有能力使用[元素参考](https://developer.mozilla.org/en-US/SVG/Element)和[接口参考](https://developer.mozilla.org/zh-CN/docs/Web/API/Document_Object_Model#svg_接口)，学习其他你需要知道的内容。

## 根元素

HTML提供了定义标题、段落、表格等等内容的元素。与此类似，SVG也提供了一些元素，用于定义圆形、矩形、简单或复杂的曲线。一个简单的SVG文档由`<svg>`根元素和基本的形状元素构成。另外还有一个`g`元素，它用来把若干个基本形状编成一个组。

基于XML的SVG，语法和格式也是结构化的。所谓结构化，也就是文件中的对象通过特定的元素标签定义，任何元素都可以作为对象进行管理，文件是框架式的。掌握基本的文件框架，就可以阅读、编辑和创作自己的文件。

SVG使用一组元素标签，创建和组织文件以及文件中的对象。每一个SVG文件都包含最外层的`<svg>`和`</svg>`标签。该标签用于声明SVG文件的开始和结束。

下面是一个简单的 SVG 框架代码

```html
<?xml version="1.1" encoding="UTF-8" ?>
<svg width="100%" height="100%" 
     viewBox="0 0 100 100" 
     xmlns="http://www.w3.org/2000/svg" >
<!-- more tags here -->
</svg>
```

第一行是 XML 声明。它定义 XML 的版本和所使用的编码。

`<svg>` 标签创建一个SVG文档片段，基本属性如下：
- `version` 属性用于指明 SVG 文档遵循规范。 它只允许在根元素`<svg>` 上使用。 它纯粹是一个说明，对渲染或处理没有任何影响。
- `baseProfile` 描述了作者认为可以正确渲染内容所需要的最小的 SVG 语言。这个属性不会有任何处理限制，可以把它看作是元数据。 比如，这个属性的值可以被编辑工具用来发出警告信息。属性值如下：
  - none 默认值，代表了最小的 SVG 语言配置
  - full 代表一个正常的概述，适用于 PC
  - basic 代表一个轻量级的概述，适用于 PDA
  - tiny 代表更轻量的概述，适用于手机
- `x,y` 属性定义用户坐标系统中SVG片段左上角的坐标，默认为 (0,0) 。
- `width,height` 属性定义用户坐标系统中SVG片段的宽度和高度，默认 100%, 100%。除了相对单位，也可以采用绝对单位（像素：px）。
- `viewBox` 指定 SVG 文档绘图区域。属性的值有由空格或逗号分隔的4个值 `min-x`, `min-y`, `width` and `height`，分别是左上角的横坐标和纵坐标、视口的宽度和高度。
- 作为XML的一种方言，SVG 需要正确的声明命名空间 （在xmlns属性中绑定），以区别其他标签系统。 请阅读[命名空间速成](https://developer.mozilla.org/en/docs/Web/SVG/Namespaces_Crash_Course) 页面获取更多信息。

{% note info %}
如果不指定width属性和height属性，只指定viewBox属性，则相当于只给定 SVG 图像的长宽比。这时，SVG 图像的默认大小将等于所在的 HTML 元素的大小。
{% endnote %}

## 坐标定位

对于所有元素，SVG使用的坐标系统或者说网格系统，和[Canvas](https://developer.mozilla.org/zh-CN/docs/Web/API/Canvas_API)用的差不多（所有计算机绘图都差不多）。这种坐标系统是：以页面的左上角为(0,0)坐标点，坐标以像素为单位，x轴正方向是向右，y轴正方向是向下。注意，这和你小时候所教的绘图方式是相反的。但是在HTML文档中，元素都是用这种方式定位的。

![](https://gitee.com/WilenWu/images/raw/master/common/Canvas_default_grid.png)

**示例：**

```html
<rect x="0" y="0" width="100" height="100" />
```
定义一个矩形，即从左上角开始，向右延展100px，向下延展100px，形成一个100x100大的矩形。

**什么是 "像素"?**

基本上，在 SVG 文档中的1个像素对应输出设备（比如显示屏）上的1个像素。但是这种情况是可以改变的，否则 SVG 的名字里也不至于会有“Scalable”（可缩放）这个词。如同CSS可以定义字体的绝对大小和相对大小，SVG也可以定义绝对大小（比如使用“pt”或“cm”标识维度）同时SVG也能使用相对大小，只需给出数字，不标明单位，输出时就会采用用户的单位。

在没有进一步规范说明的情况下，1个用户单位等同于1个屏幕单位。要明确改变这种设定，SVG里有多种方法。我们从`svg`根元素开始：

```html
<svg width="100" height="100">
```
上面的元素定义了一个100x100 px的SVG画布，这里1用户单位等同于1屏幕单位。

```html
<svg width="200" height="200" viewBox="0 0 100 100">
```
这里定义的画布尺寸是200x200 px。但是，viewBox属性定义了画布上可以显示的区域：从(0,0)点开始，100宽x100高的区域。这个100x100的区域，会放到200x200的画布上显示。于是就形成了放大两倍的效果。

用户单位和屏幕单位的映射关系被称为**用户坐标系统**。除了缩放之外，坐标系统还可以旋转、倾斜、翻转。默认的用户坐标系统1用户像素等于设备上的1像素（但是设备上可能会自己定义1像素到底是多大）。在定义了具体尺寸单位的SVG中，比如单位是“cm”或“in”，最终图形会以实际大小的1比1比例呈现。

# 图形元素

## 基本形状

SVG 有一些预定义的形状元素，可被开发者使用和操作。要想插入一个形状，你可以在文档中创建一个元素。不同的元素对应着不同的形状，并且使用不同的属性来定义图形的大小和位置。下图展示了基本形状：

<svg width="200" height="250" version="1.1" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="30" height="30" stroke="black" fill="transparent" stroke-width="5"/>
  <rect x="60" y="10" rx="10" ry="10" width="30" height="30" stroke="black" fill="transparent" stroke-width="5"/>
  <circle cx="25" cy="75" r="20" stroke="red" fill="transparent" stroke-width="5"/>
  <ellipse cx="75" cy="75" rx="20" ry="5" stroke="red" fill="transparent" stroke-width="5"/>
  <line x1="10" x2="50" y1="110" y2="150" stroke="orange" fill="transparent" stroke-width="5"/>
  <polyline points="60 110 65 120 70 115 75 130 80 125 85 140 90 135 95 150 100 145"
      stroke="orange" fill="transparent" stroke-width="5"/>
  <polygon points="50 160 55 180 70 180 60 190 65 205 50 195 35 205 40 190 30 180 45 180"
      stroke="green" fill="transparent" stroke-width="5"/>
  <path d="M20,230 Q40,205 50,230 T90,230" fill="none" stroke="blue" stroke-width="5"/>
</svg>

```html
<?xml version="1.0" standalone="no"?>
<svg width="200" height="250" version="1.1" xmlns="http://www.w3.org/2000/svg">

  <rect x="10" y="10" width="30" height="30" stroke="black" fill="transparent" stroke-width="5"/>
  <rect x="60" y="10" rx="10" ry="10" width="30" height="30" stroke="black" fill="transparent" stroke-width="5"/>

  <circle cx="25" cy="75" r="20" stroke="red" fill="transparent" stroke-width="5"/>
  <ellipse cx="75" cy="75" rx="20" ry="5" stroke="red" fill="transparent" stroke-width="5"/>

  <line x1="10" x2="50" y1="110" y2="150" stroke="orange" fill="transparent" stroke-width="5"/>
  <polyline points="60 110 65 120 70 115 75 130 80 125 85 140 90 135 95 150 100 145"
      stroke="orange" fill="transparent" stroke-width="5"/>

  <polygon points="50 160 55 180 70 180 60 190 65 205 50 195 35 205 40 190 30 180 45 180"
      stroke="green" fill="transparent" stroke-width="5"/>

  <path d="M20,230 Q40,205 50,230 T90,230" fill="none" stroke="blue" stroke-width="5"/>
</svg>
```

- **圆形**：`<circle>` 元素会在屏幕上元素创建圆形，基本属性如下
  - cx, cy 圆心坐标。默认圆心坐标为 (0, 0)
  - r 圆的半径
- **椭圆**：`<ellipse>` 元素会在屏幕上创建椭圆
  - cx, cy 椭圆中心坐标
  - rx, ry 水平半径和垂直半径
- **矩形**：`<rect>`元素会在屏幕上绘制一个矩形
  - x, y 矩形左上角坐标
  - width, height 矩形的宽和高
  - 还可以用圆角半径 rx 和 ry 来创建圆角矩形
- **线条**：`<line>` 元素基于两个端点坐标 (x1, x2) 和 (x2, y2) 绘制一条直线
- **折线**：`<polyline>` 元素绘制一组连接在一起的直线。因为它可以有很多的点，折线的的所有点坐标都放在一个points属性中
  - points 点集数列。每个数字用空白、逗号、终止命令符或者换行符分隔开。每个点必须包含2个数字，一个是x坐标，一个是y坐标。
- **多边形**：`<polygon>` 创建多边形。和折线很像，它们都是由连接一组点集的直线构成。不同的是，`polygon`的路径在最后一个点处自动回到第一个点
  - points 点集数列。每个数字用空白符、逗号、终止命令或者换行符分隔开。每个点必须包含2个数字，一个是x坐标，一个是y坐标。路径绘制完后闭合图形。
- **路径**：`<path>`可能是SVG中最常见的形状。你可以用path元素绘制矩形（直角矩形或者圆角矩形）、圆形、椭圆、折线形、多边形，以及一些其他的形状，例如贝塞尔曲线、2次曲线等曲线。因为path很强大也很复杂，所以会在下一节进行详细介绍。这里只介绍一个定义路径形状的属性。
  - d 属性定义一个点集数列以及其它关于如何绘制路径的信息。

> **注意：**`stroke`、`stroke-width` 和 `fill` 等属性在后面的章节中解释。

## 路径

`<path>`元素是SVG基本形状中最强大的一个。  你可以用它创建线条, 曲线, 弧形等等。上一节提到过，path元素的形状是通过属性 `d` 定义的。

`d` 表示绘制顺序，它的值是一个是一个**命令+参数**的序列。每一个命令都用一个关键字母来表示绘制动作，后面跟着坐标。每一个命令都有两种表示方式，一种是用**大写字母**，表示采用绝对定位。另一种是用**小写字母**，表示采用相对定位（即相对于它前面的点需要移动多少距离）。

因为属性 `d` 采用的是用户坐标系统，所以**不需标明单位**。在后面的教程中，我们会学到如何变换路径，以满足更多需求。

SVG 定义了6种类型的 path 命令，一共20个命令：
- MoveTo: `M`, `m`
- LineTo: `L`, `l`, `H`, `h`, `V`, `v`
- Cubic Bézier Curve: `C`, `c`, `S`, `s`
- Quadratic Bézier Curve: `Q`, `q`, `T`, `t`
- Elliptical Arc Curve: `A`, `a`
- ClosePath: `Z`, `z`

### 移动命令

字母`M`表示的是Move to命令，当解析器读到这个命令时，它就知道你是打算移动到某个点。跟在命令字母后面的，是你需要移动到的那个点的x和y轴坐标。这一段字符结束后，解析器就会去读下一段命令。

```svg
M x y
m dx dy
```

假设，你的画笔当前位于一个点，在使用M命令移动画笔后，只会移动画笔，但不会在两点之间画线。因为M命令仅仅是移动画笔，但不画线。所以M命令经常出现在路径的开始处，用来指明从何处开始画。

### 直线命令

| 命令                 | 描述 |
| :-------------------- | :------ |
| `L x y`<br>(`l dx dy`) | 在当前位置和新位置 (x,y) 之间画一条线段。 |
| `H x`<br>(`h dx`) | 绘制水平线到 x |
| `V y`<br>(`v dy`) | 绘制垂直线到 y |
| `Z`<br>(`z`) | 从当前点画一条直线到路径的起点。另，Z命令不用区分大小写。 |

### 三次贝塞尔曲线

绘制平滑曲线的命令有三个，其中两个用来绘制贝塞尔曲线，另外一个用来绘制弧形或者说是圆的一部分。
我们从稍微复杂一点的三次贝塞尔曲线入手，三次贝塞尔曲线需要定义曲线终点和两个控制点，所需要设置三组坐标参数：

```svg
C x1 y1, x2 y2, x y 
c dx1 dy1, dx2 dy2, dx dy
```

这里的最后一个坐标 (x,y) 表示的是曲线的终点，另外两个坐标是控制点。(x1,y1)是起点的控制点，(x2,y2)是终点的控制点。如果你熟悉代数或者微积分的话，会更容易理解控制点，控制点描述的是曲线起点和终点的斜率，曲线上各个点的斜率，是从起点斜率到终点斜率的渐变过程。（文字描述不好，维基百科上有图示，更直观。）

![](https://gitee.com/WilenWu/images/raw/master/common/Cubic_Bezier_Curves.png)

你可以将若干个贝塞尔曲线连起来，从而创建出一条很长的平滑曲线。通常情况下，一个点某一侧的控制点是它另一侧的控制点的对称（以保持斜率不变）。这样，你可以使用一个简写的贝塞尔曲线命令S，如下所示：

```svg
S x2 y2, x y 
s dx2 dy2, dx dy
```

S命令可以用来创建与前面一样的贝塞尔曲线，但是，如果S命令跟在一个C或S命令后面，则它的第一个控制点会被假设成前一个命令曲线的第二个控制点的中心对称点。如果S命令单独使用，前面没有C或S命令，那当前点将作为第一个控制点。下面是S命令的语法示例，图中左侧红色标记的点对应的控制点即为蓝色标记点。(标记控制点的代码会比较庞大，所以在这里舍弃了)

![](https://gitee.com/WilenWu/images/raw/master/common/ShortCut_Cubic_Bezier.png)

```html
<?xml version="1.0" standalone="no"?>
<svg width="190px" height="160px" version="1.1" 
     xmlns="http://www.w3.org/2000/svg">
  <path d="M10 80 C 40 10, 65 10, 95 80 S 150 150, 180 80" 
  stroke="black" fill="transparent"/>
</svg>
```

下面的例子创建了一个螺旋线：

<svg width="200" height="200" version="1.1"  viewBox="150 250 60 150" >
  <path d="M153 334
    C153 334 151 334 151 334
    C151 339 153 344 156 344
    C164 344 171 339 171 334
    C171 322 164 314 156 314
    C142 314 131 322 131 334
    C131 350 142 364 156 364
    C175 364 191 350 191 334
    C191 311 175 294 156 294
    C131 294 111 311 111 334
    C111 361 131 384 156 384
    C186 384 211 361 211 334
    C211 300 186 274 156 274"
  style="fill:pink;stroke:red;stroke-width:2"/>
</svg>

```html
<svg width="200" height="200" version="1.1" 
     viewBox="150 250 60 150" >
  <path d="M153 334
    C153 334 151 334 151 334
    C151 339 153 344 156 344
    C164 344 171 339 171 334
    C171 322 164 314 156 314
    C142 314 131 322 131 334
    C131 350 142 364 156 364
    C175 364 191 350 191 334
    C191 311 175 294 156 294
    C131 294 111 311 111 334
    C111 361 131 384 156 384
    C186 384 211 361 211 334
    C211 300 186 274 156 274"
  style="fill:pink;stroke:red;stroke-width:2"/>
</svg>
```

### 二次贝塞尔曲线

另一种可用的贝塞尔曲线是二次贝塞尔曲线Q，它比三次贝塞尔曲线简单，只需要一个控制点，用来确定起点和终点的曲线斜率。因此它需要两组参数，控制点和终点坐标。

```svg
Q x1 y1, x y 
q dx1 dy1, dx dy
```

![](https://gitee.com/WilenWu/images/raw/master/common/Quadratic_Bezier.png)

```html
<?xml version="1.0" standalone="no"?>
<svg width="190px" height="160px" version="1.1" 
     xmlns="http://www.w3.org/2000/svg">
  <path d="M10 80 Q 95 10 180 80" 
        stroke="black" fill="transparent"/>
</svg>
```

就像三次贝塞尔曲线有一个S命令，二次贝塞尔曲线有一个差不多的T命令（一个点某一侧的控制点是它另一侧的控制点的对称）。通常情况下，可以通过更简短的参数，延长二次贝塞尔曲线。

```html
T x y
t dx dy
```

和之前一样，快捷命令T会通过前一个控制点，推断出一个新的控制点。这意味着，在你的第一个控制点后面，可以只定义终点，就创建出一个相当复杂的曲线。需要注意的是，T命令前面必须是一个Q命令，或者是另一个T命令，才能达到这种效果。如果T单独使用，那么控制点就会被认为和终点是同一个点，所以画出来的将是一条直线。
![](https://gitee.com/WilenWu/images/raw/master/common/Shortcut_Quadratic_Bezier.png)

```html
<?xml version="1.0" standalone="no"?>
<svg width="190px" height="160px" version="1.1" 
     xmlns="http://www.w3.org/2000/svg">
  <path d="M10 80 Q 52.5 10, 95 80 T 180 80" 
        stroke="black" fill="transparent"/>
</svg>
```

虽然三次贝塞尔曲线拥有更大的自由度，但是两种曲线能达到的效果总是差不多的。具体使用哪种曲线，通常取决于需求，以及对曲线对称性的依赖程度。

### 弧形

弧形命令A是另一个创建SVG曲线的命令。基本上，弧形可以视为圆形或椭圆形的一部分。假设，已知椭圆形的长轴半径和短轴半径，并且已知两个点（在椭圆上），根据半径和两点，可以画出两个椭圆，在每个椭圆上根据两点都可以画出两种弧形。所以，仅仅根据半径和两点，可以画出四种弧形。为了保证创建的弧形唯一，A命令需要用到比较多的参数：

```svg
 A rx ry x-axis-rotation large-arc-flag sweep-flag x y
 a rx ry x-axis-rotation large-arc-flag sweep-flag dx dy
```

- 前两个参数 rx,ry 分别是x轴半径和y轴半径
- 第三个参数 x-axis-rotation 表示弧形相对于x轴的旋转角度
- 参数是large-arc-flag（角度大小） 和sweep-flag（弧线方向），large-arc-flag决定弧线是大于还是小于180度，0表示小角度弧，1表示大角度弧。sweep-flag表示弧线的方向，0表示从起点到终点沿逆时针画弧，1表示从起点到终点沿顺时针画弧。下面的例子展示了这四种情况。
- 最后两个参数是指定弧形的终点
- 弧形命令解释起来较复杂，请参考文档 [路径 - SVG | MDN (mozilla.org)](https://developer.mozilla.org/zh-CN/docs/Web/SVG/Tutorial/Paths#arcs) 

![](https://gitee.com/WilenWu/images/raw/master/common/SVGArcs_Flags.png)

```html
<?xml version="1.0" standalone="no"?>
<svg width="325px" height="325px" version="1.1" 
     xmlns="http://www.w3.org/2000/svg">
  <path d="M80 80
           A 45 45, 0, 0, 0, 125 125
           L 125 80 Z" fill="green"/>
  <path d="M230 80
           A 45 45, 0, 1, 0, 275 125
           L 275 80 Z" fill="red"/>
  <path d="M80 230
           A 45 45, 0, 0, 1, 125 275
           L 125 230 Z" fill="purple"/>
  <path d="M230 230
           A 45 45, 0, 1, 1, 275 275
           L 275 230 Z" fill="blue"/>
</svg>
```

# 属性

可以使用几种方法来配置属性，包括指定对象的属性，内嵌CSS样式，或者使用外部CSS样式文件。大多数的web网站的SVG使用对象的属性，对于这些方法都有优缺点。

## id 属性

`id` 属性给予元素一个唯一名称。所有元素均可使用该属性。该ID在节点树中必须是唯一的，不能为空字符串，并且不能包含任何空格字符。

## 边框属性

SVG 提供了一系列边框属性，可应用于任何种类的线条，包括文字和元素的轮廓。

- `stroke` 属性定义线条颜色。你可以使用在HTML中的CSS颜色命名方案定义它们的颜色，比如说颜色名（像red这种）、rgb值（像rgb(255,0,0)这种）、十六进制值、rgba值，等等。
- `stroke-width` 属性定义了描边的宽度。注意，描边是以路径为中心线绘制的，路径的每一侧都有均匀分布的描边。
- `stroke-linecap` 属性控制边框终点的形状。有三种可能值：
  - `butt`用直边结束线段，它是常规做法，线段边界90度垂直于描边的方向、贯穿它的终点。
  - `square`的效果差不多，但是会稍微超出实际路径的范围，超出的大小由`stroke-width`控制。
  - `round`表示边框的终点是圆角，圆角的半径也是由`stroke-width`控制的。

  ![](https://gitee.com/WilenWu/images/raw/master/common/SVG_Stroke_Linecap_Example.png)

  ```html
  <?xml version="1.0" standalone="no"?>
  <svg width="160" height="140" xmlns="http://www.w3.org/2000/svg" version="1.1">
    <line x1="40" x2="120" y1="20" y2="20" stroke="black" stroke-width="20" stroke-linecap="butt"/>
    <line x1="40" x2="120" y1="60" y2="60" stroke="black" stroke-width="20" stroke-linecap="square"/>
    <line x1="40" x2="120" y1="100" y2="100" stroke="black" stroke-width="20" stroke-linecap="round"/>
  </svg>
  ```

- `stroke-linejoin` 属性用来控制两条描边线段之间，用什么方式连接。有三个可用的值：
  - `miter`是默认值，表示用方形画笔在连接处形成尖角
  - `round`表示用圆角连接，实现平滑效果
  - 最后还有一个值`bevel`，连接处会形成一个斜接

  ![](https://gitee.com/WilenWu/images/raw/master/common/SVG_Stroke_Linejoin_Example.png)

  ```html
  <?xml version="1.0" standalone="no"?>
  <svg width="160" height="280" xmlns="http://www.w3.org/2000/svg" version="1.1">
    <polyline points="40 60 80 20 120 60" stroke="black" stroke-width="20"
        stroke-linecap="butt" fill="none" stroke-linejoin="miter"/>
  
    <polyline points="40 140 80 100 120 140" stroke="black" stroke-width="20"
        stroke-linecap="round" fill="none" stroke-linejoin="round"/>
  
    <polyline points="40 220 80 180 120 220" stroke="black" stroke-width="20"
        stroke-linecap="square" fill="none" stroke-linejoin="bevel"/>
  </svg>
  ```

- `stroke-dasharray` 属性用于创建虚线

  ![](https://gitee.com/WilenWu/images/raw/master/common/SVG_Stroke_Dasharray_Example.png)

  ```html
  <?xml version="1.0" standalone="no"?>
  <svg width="200" height="150" version="1.1"
       xmlns="http://www.w3.org/2000/svg" >
    <path d="M 10 75 Q 50 10 100 75 T 190 75" stroke="black"
      stroke-linecap="round" stroke-dasharray="5,10,5" fill="none"/>
    <path d="M 10 75 L 190 75" stroke="red"
      stroke-linecap="round" stroke-width="1" 
      stroke-dasharray="5,5" fill="none"/>
  </svg>
  ```

  `stroke-dasharray`属性的参数，是一组用逗号分割的数字组成的数列。注意，和`path`不一样，这里的数字**必须**用逗号分割（空格会被忽略）。每一组数字，第一个用来表示填色区域的长度，第二个用来表示非填色区域的长度。所以在上面的例子里，第二个路径会先做5个像素单位的填色，紧接着是5个空白单位，然后又是5个单位的填色。如果你想要更复杂的虚线模式，你可以定义更多的数字。第一个例子指定了3个数字，这种情况下，数字会循环两次，形成一个偶数的虚线模式（奇数个循环两次变偶数个）。所以该路径首先渲染5个填色单位，10个空白单位，5个填色单位，然后回头以这3个数字做一次循环，但是这次是创建5个空白单位，10个填色单位，5个空白单位。通过这两次循环得到偶数模式，并将这个偶数模式不断重复。

- `stroke-opacity` 属性定义线条的不透明度，范围为 0 - 1
- 另外还有一些关于边框的属性，包括`stroke-miterlimit`，定义什么情况下绘制或不绘制边框连接的`miter`效果；还有`stroke-dashoffset`，定义虚线开始的位置。

## 填充属性

SVG 也提供了一系列属性用于填充，包括形状类元素和文字内容类元素。

- `fill` 属性用于填充颜色
- `fill-opacity` 属性定义填充的不透明度，范围为 0 - 1
- `fill-rule` 属性用于定义如何给图形重叠的区域上色。有三个可用的值 nonzero(default), evenodd, inherit用于判断画布上的某区域是否属于该图形**内部**，将被填充。

使用 `<polygon>` 元素创建一个星型

<svg height="210" width="500">
  <polygon points="100,10 40,198 190,78 10,78 160,198"
    fill="lime" fill-rule="nonzero"
    stroke="purple" stroke-width="5"
  />
</svg>

```html
<svg height="210" width="500">
  <polygon points="100,10 40,198 190,78 10,78 160,198"
    fill="lime" fill-rule="nonzero"
    stroke="purple" stroke-width="5"
  />
</svg>
```

**fill-rule 属性**：对一个简单的无交叉的路径，哪块区域是内部是很直观清楚的。但是，对一个复杂的路径，比如自相交或者一个子路径包围另一个子路径，内部的理解就不那么明确了。
- **nonzero**：字面意思是“非零”。按该规则，要判断一个点是否在图形内，从该点作任意方向的一条射线，然后检测射线与图形路径的交点情况。从0开始计数，路径从左向右穿过射线则计数加1，从右向左穿过射线则计数减1。得出计数结果后，如果结果是0，则认为点在图形外部，否则认为在内部。下图演示了nonzero规则

  ![](https://gitee.com/WilenWu/images/raw/master/common/SVG-nonzero.svg)

- **evenodd**：字面意思是“奇偶”。按该规则，要判断一个点是否在图形内，从该点作任意方向的一条射线，然后检测射线与图形路径的交点的数量。如果结果是奇数则认为点在内部，是偶数则认为点在外部。下图演示了evenodd 规则

  ![](https://gitee.com/WilenWu/images/raw/master/common/SVG-evenodd.svg)

**提示:** 上述解释未指出当路径片段与射线重合或者相切的时候怎么办，因为任意方向的射线都可以，那么只需要简单的选择另一条没有这种特殊情况的射线即可。

## 样式属性

除了定义对象的属性外，你也可以通过CSS来样式化填充和描边。语法和在html里使用CSS一样，只不过你要把`background-color`、`border`改成`fill`和`stroke`。

注意，不是所有的属性都能用CSS来设置。SVG 所有显示属性都可以作为 CSS 属性来使用。另外，`width`、`height`，以及路径的命令等等，都不能用css设置。

- `style` 属性指定了其元素的样式信息。它的功能与 HTML 中的style属性相同。

  ```html
  <svg version="1.1" viewbox="0 0 1000 500" 
       xmlns="http://www.w3.org/2000/svg">
    <rect height="300" width="600" x="200" y="100"
       style="fill: red; stroke: blue; stroke-width: 3"/>
  </svg>
  ```

- 或者利用`<style>`元素设置一段样式段落，通过 class 属性指定对应的 CSS 类

  ```html
  <?xml version="1.0" standalone="no"?>
  <svg width="200" height="200" version="1.1"
       xmlns="http://www.w3.org/2000/svg" >
    <style type="text/css">
      .fancy {
        fill: none;
        stroke: black;
        stroke-width: 3pt;
      }
    </style>
    <rect x="10" height="180" y="10" width="180" class="fancy"/>
  </svg>
  ```

- 你也可以定义一个外部的样式表，但是要符合[normal XML-stylesheet syntax](https://www.w3.org/TR/xml-stylesheet/)的CSS规则:

  ```html
  <?xml version="1.0" standalone="no"?>
  <?xml-stylesheet type="text/css" href="style.css"?>
  
  <svg width="200" height="150" version="1.1"
       xmlns="http://www.w3.org/2000/svg" >
    <rect height="10" width="10" id="MyRect"/>
  </svg>
  ```

  style.css 看起来就像这样：

  ```css
  #MyRect {
    fill: red;
    stroke: black;
  }
  ```

## 透明度

`opacity`属性指定了一个对象或一组对象的透明度，也就是说，元素后面的背景的透过率，范围为 0 - 1。

填充和描边还有两个属性是`fill-opacity`和`stroke-opacity`，分别用来控制填充和描边的不透明度。需要注意的是描边将绘制在填充的上面。因此，如果你在一个元素上设置了描边透明度，但它同时设有填充，则描边的一半应用填充色，另一半将应用背景色。

# 容器元素

## 超链接

使用 SVG 的锚元素 `<a>` 定义一个超链接
- `xlink:show` 
- `xlink:actuate`
- `xlink:href `属性规定链接指向的页面的 URL
- `target` 属性规定在何处打开链接文档，取值有 `'_blank', '_parent',   '_self', '_top', 'framename'`

<svg width="140" height="30" xmlns="http://www.w3.org/2000/svg">
  <a xlink:href="https://www.w3.org/Graphics/SVG/"
     target="_blank">
    <rect height="30" width="120" y="0" x="0" rx="15"/>
    <text fill="white" text-anchor="middle"
          y="21" x="60">W3C SVG</text>
  </a>
</svg>


```html
<svg width="140" height="30" xmlns="http://www.w3.org/2000/svg">

  <a xlink:href="https://www.w3.org/Graphics/SVG/"
     target="_blank">
    <rect height="30" width="120" y="0" x="0" rx="15"/>
    <text fill="white" text-anchor="middle"
          y="21" x="60">W3C SVG</text>
  </a>

</svg>
```

## 定义和引用

`<defs>`元素是 definitions 的缩写，用于定义以后需要重复使用的图元。它内部的图形元素不会直接显现，仅供引用。

`<use>`元素在SVG文档内取得目标节点，并在别的地方复制它们
- `xlink:href` 属性指定所要复制的节点 id 
- `x,y` 定义左上角端点坐标
- `width,height` 定义它的宽和高 

<svg width="300" height="100">
  <defs>
    <g id="c2">
      <text x="25" y="20">circle</text>
      <circle cx="50" cy="50" r="20"/>
    </g>
  </defs>
  <use href="#c2" x="0" y="0" />
  <use href="#c2" x="100" y="0" fill="blue" />
  <use href="#c2" x="200" y="0" fill="white" stroke="blue" />
</svg>


```html
<svg width="300" height="100">
  <defs>
    <g id="c2">
      <text x="25" y="20">circle</text>
      <circle cx="50" cy="50" r="20"/>
    </g>
  </defs>

  <use href="#c2" x="0" y="0" />
  <use href="#c2" x="100" y="0" fill="blue" />
  <use href="#c2" x="200" y="0" fill="white" stroke="blue" />
</svg>
```

## 分组

`<g>` 标签用于将多个元素组成一个组（group）。添加到g元素上的变换会应用到其所有的子元素上。添加到g元素的属性会被其所有的子元素继承。此外，g元素也可以用来定义复杂的对象，之后通过`<use>`元素来引用它们。

<svg width="50%"  viewBox="0 0 95 50" xmlns="http://www.w3.org/2000/svg">
  <g stroke="green" fill="white" stroke-width="5">
    <circle cx="25" cy="25" r="15" />
    <circle cx="40" cy="25" r="15" />
    <circle cx="55" cy="25" r="15" />
    <circle cx="70" cy="25" r="15" />
  </g>
</svg>

```html
<svg width="200" height="100" viewBox="0 0 95 50"
     xmlns="http://www.w3.org/2000/svg">
  <g stroke="green" fill="white" stroke-width="5">
    <circle cx="25" cy="25" r="15" />
    <circle cx="40" cy="25" r="15" />
    <circle cx="55" cy="25" r="15" />
    <circle cx="70" cy="25" r="15" />
  </g>
</svg>
```

## 图案

`<pattern>`（图案）元素用于自定义一个形状，该形状可以被引用来平铺一个区域。必须给该元素指定一个id属性，否则文档内的其他元素就不能引用它。为了让图案能被重复使用，`<pattern>` 元素需要放在SVG文档的`<defs>`内部。

- x,y 定义图案对象视口左上角偏移量，默认为 0,0
- viewBox 定义图案对象视口区域。由空格或逗号分隔的4个值。(min x, min y, width, height)
- width, height 定义图案对象宽和高，单位（％）。
- patternUnits="userSpaceOnUse" | "objectBoundingBox"。
  "userSpaceOnUse" 表示`<pattern>`的宽度和长度是实际的像素值。
- patternContentUnits='userSpaceOnUse' | 'objectBoundingBox'
- patternTransform 定义转换


<svg width="200" height="200">
  <defs>
    <pattern id="dots" x="0" y="0" width="10%" height="10%" patternUnits="userSpaceOnUse">
      <circle fill="#bee9e8" cx="5" cy="5" r="5" />
    </pattern>
  </defs>
  <rect x="0" y="0" width="100%" height="100%" fill="url(#dots)" />
</svg>

```html
<svg width="200" height="200">
  <defs>
    <pattern id="dots" x="0" y="0" width="10" height="10" patternUnits="userSpaceOnUse">
      <circle fill="#bee9e8" cx="5" cy="5" r="5" />
    </pattern>
  </defs>
  <rect x="0" y="0" width="100%" height="100%" fill="url(#dots)" />
</svg>
```

## 标记

`<marker>` 元素定义了在特定的`<path>`元素、`<line>`元素、`<polyline>`元素或者`<polygon>`元素上绘制的箭头或者多边标记图形。

这些元素可以使用marker元素的`marker-start`，`marker-mid`和`marker-end`。

<svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <marker id="Triangle" viewBox="0 0 10 10" refX="1" refY="5"
        markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" />
    </marker>
  </defs>
  <polyline points="10,90 50,80 90,20" fill="none" stroke="black"
      stroke-width="2" marker-end="url(#Triangle)" />
</svg>

```html
<svg width="120" height="120" viewBox="0 0 120 120"
    xmlns="http://www.w3.org/2000/svg" version="1.1">

  <defs>
    <marker id="Triangle" viewBox="0 0 10 10" refX="1" refY="5"
        markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" />
    </marker>
  </defs>

  <polyline points="10,90 50,80 90,20" fill="none" stroke="black"
      stroke-width="2" marker-end="url(#Triangle)" />
</svg>
```

## 图像元素

很像在HTML中的`img`元素，SVG有一个`<image>`元素，用于同样的目的。你可以利用它嵌入任意光栅（以及矢量）图像。它的规格要求应用至少支持PNG、JPG和SVG格式文件。

- `xlink:href` 属性定义图像的来源。
- `x,y` 定义左上角端点坐标
- `width,height` 定义它的宽和高 

# 文本元素

在SVG中有两种截然不同的文本模式。一种是写在图像中的文本，另一种是SVG字体。

## text

`<text>` 元素内部可以放任何的文字
- `x, y` 属性表示文本区块基线（baseline）起点的坐标，默认为 (0, 0)
- `text-anchor`可以有这些值：start、middle、end或inherit，决定从这一点开始的文本流的方向
- 和形状元素类似，属性`fill`可以给文本填充颜色，属性`stroke`可以给文本描边
- 设置字体属性：`font-family`、`font-style`、`font-weight`、`font-variant`、`font-stretch`、`font-size`、`font-size-adjust`、`kerning`、`letter-spacing`、`word-spacing`和`text-decoration`
- `dx, dy` 属性控制文本相对起点移动。这里，你可以提供一个值数列，可以应用到连续的字体，因此每次累积一个偏移。
- `rotate` 属性把所有的字符旋转一个角度。如果是一个数列，则使每个字符旋转分别旋转到那个值，剩下的字符根据最后一个值旋转。
- `textLength` 这是一个很模糊的属性，给出字符串的计算长度。它意味着如果它自己的度量文字和长度不满足这个提供的值，则允许渲染引擎精细调整字型的位置
- lengthAdjust='spacing' or 'spacingAndGlyphs' 如果指定长度将尝试调整文本

## tspan

该元素在文本内部嵌套分组，每个 `<tspan>` 元素可以包含不同的格式和位置。它必须是一个`text`元素或别的`tspan`元素的子元素。

<svg height="100" xmlns="http://www.w3.org/2000/svg" version="1.1">
  <text x="10" y="20" style="fill:red;">Several lines:
    <tspan x="10" y="45">First line</tspan>
    <tspan x="10" y="70">Second line</tspan>
  </text>
</svg>

```html
<svg height="100" version="1.1"
     xmlns="http://www.w3.org/2000/svg">
  <text x="10" y="20" style="fill:red;">Several lines:
    <tspan x="10" y="45">First line</tspan>
    <tspan x="10" y="70">Second line</tspan>
  </text>
</svg>
```

## tref

`tref`元素允许引用已经定义的文本，高效地把它复制到当前位置。你可以使用`xlink:href`属性，把它指向一个元素，取得其文本内容。你可以独立于源样式化它、修改它的外观。

```html
<text id="example">This is an example text.</text>

<text>
    <tref xlink:href="#example" />
</text>
```

## textPath

该元素利用它的`xlink:href`属性取得一个任意路径，把字符对齐到路径

<svg height="100" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <path id="my_path" d="M0,20 a1,1 0 0,0 100,0" fill="transparent" />
  <text>
    <textPath xlink:href="#my_path" stroke="red">
      This text follows a curve.
    </textPath>
  </text>
</svg>

```html
<svg height="100" 
     xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink">
  <path id="my_path" d="M0,20 a1,1 0 0,0 100,0" 
        fill="transparent" />
  <text>
    <textPath xlink:href="#my_path" stroke="red">
      This text follows a curve.
    </textPath>
  </text>
</svg>
```

# 描述性元素

SVG绘画中的每个容器元素或图形元素可以提供一个`<desc>` 和 `<title>` 描述性内容，这些描述只是纯文本的，以帮助用户通过多种方式进行索引、搜索和检索，并不作为图形的一部分来显示。

然而，一些用户可能会把`title`显示为一个提示冒泡。它显示了`title`元素但是不会显示路径元素或者别的图形元素。`title`元素通常提升了SVG文档的可访问性。

通常`title`元素必须是它的父元素的第一个子元素。注意，只有当`title`是它的父元素的第一个子元素的时候，那些编译器才会把`title`显示为一个提示冒泡。

<svg height="100" xmlns="http://www.w3.org/2000/svg">
  <g>
    <title>SVG Title Demo example</title>
    <rect x="10" y="10" width="200" height="50"
    style="fill:orange; stroke:blue; stroke-width:1px"/>
  </g>
</svg>

```html
<svg height="100" xmlns="http://www.w3.org/2000/svg">
  <g>
    <title>SVG Title Demo example</title>
    <rect x="10" y="10" width="200" height="50"
    style="fill:orange; stroke:blue; stroke-width:1px"/>
  </g>
</svg>
```

# 动画元素

SVG Animation 是一种基于XML的开放标准矢量图形格式，可以通过各种方式实现：ECMAScript、CSS Animations、SMIL。W3C 建议将 SMIL 作为 SVG 中动画的标准。

[SVG 在线动画编辑 | SVGA](https://svga.io/index.html)

## 为元素的属性添加动画效果

`<animate>`标签用于为元素的属性添加动画效果

- attributeName 发生动画效果的属性名
- from 属性的初始值
- to 属性的最终值
- dur 单次动画的持续时间
- repeatCount 动画的循环次数，可以定义为无限循环 "indefinite"

<svg width="120" height="120">
  <rect x="10" y="10" width="100" height="100" fill="#feac5e">
    <animate attributeName="rx" from="0" to="50" dur="2s" repeatCount="indefinite" />
    <animate attributeName="ry" from="0" to="50" dur="2s" repeatCount="indefinite" />
  </rect>
</svg>


```html
<svg width="120" height="120">
  <rect x="10" y="10" width="100" height="100" fill="#feac5e">
    <animate attributeName="rx" from="0" to="50" dur="2s" repeatCount="indefinite" />
    <animate attributeName="ry" from="0" to="50" dur="2s" repeatCount="indefinite" />
  </rect>
</svg>
```

## 为转换属性添加动画效果

`<animateTransform>` 对元素的transform属性添加动画效果。此标签对 CSS 的transform属性不起作用。
- attributeName 发生动画效果的属性名
- by 相对偏移值
- from 单次动画的初始值
- to 单次动画的结束值
- dur 单次动画的持续时间
- type 随时间变化的转换类型 'translate', 'scale', 'rotate', 'skewX', 'skewY'
- repeatCount 动画的循环次数，可以定义为无限循环 "indefinite"

下面的示例中，我们对旋转中心和角度进行动画处理

<svg width="300" height="100">
  <title>SVG SMIL Animate with transform</title>
  <rect x="0" y="0" width="300" height="100" fill="white" stroke="black" stroke-width="1" />
  <rect x="0" y="50" width="15" height="34" fill="blue" stroke="black" stroke-width="1">
    <animateTransform
       attributeName="transform"
       begin="0s"
       dur="20s"
       type="rotate"
       from="0 60 60"
       to="360 100 60"
       repeatCount="indefinite"
      />
  </rect>
</svg>


```html
<svg width="300" height="100">
  <title>SVG SMIL Animate with transform</title>
  <rect x="0" y="0" width="300" height="100" stroke="black" stroke-width="1" />
  <rect x="0" y="50" width="15" height="34" fill="blue" stroke="black" stroke-width="1">
    <animateTransform
       attributeName="transform"
       begin="0s"
       dur="20s"
       type="rotate"
       from="0 60 60"
       to="360 100 60"
       repeatCount="indefinite"
      />
  </rect>
</svg>
```

## 沿路径运行的动画

`<animateMotion>` 使元素沿着路径移动 
- `calcMode` 动画的插补模式。可以是'discrete', 'linear', 'paced', 'spline'
- `path` 运动路径
- `keyPoints` 
- `rotate` 应用旋转变换 

<svg xmlns="http://www.w3.org/2000/svg" width="300" height="100">
  <title>SVG SMIL Animate with Path</title>
  <rect x="0" y="0" width="300" height="100" fill="white" stroke="black" stroke-width="1" />
  <circle cx="0" cy="50" r="15" fill="blue" stroke="black" stroke-width="1">
    <animateMotion
       path="M 0 0 H 300 Z"
       dur="3s" repeatCount="indefinite" />
  </circle>
</svg>

```html
<svg xmlns="http://www.w3.org/2000/svg" width="300" height="100">
  <title>SVG SMIL Animate with Path</title>
  <rect x="0" y="0" width="300" height="100" fill="white" stroke="black" stroke-width="1" />
  <circle cx="0" cy="50" r="15" fill="blue" stroke="black" stroke-width="1">
    <animateMotion
       path="M 0 0 H 300 Z"
       dur="3s" repeatCount="indefinite" />
  </circle>
</svg>
```

# 滤镜元素

## 示例

滤镜（Filter）是 SVG 中用于创建复杂效果的一种机制。

下面是一个为 SVG 内容添加模糊效果的基本示例

<svg width="250" viewBox="0 0 200 85" xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <!-- Filter declaration -->
    <filter id="MyFilter" filterUnits="userSpaceOnUse"
            x="0" y="0"
            width="200" height="120">
      <!-- offsetBlur -->
      <feGaussianBlur in="SourceAlpha" stdDeviation="4" result="blur"/>
      <feOffset in="blur" dx="4" dy="4" result="offsetBlur"/>
      <!-- litPaint -->
      <feSpecularLighting in="blur" surfaceScale="5" specularConstant=".75"
                          specularExponent="20" lighting-color="#bbbbbb"
                          result="specOut">
        <fePointLight x="-5000" y="-10000" z="20000"/>
      </feSpecularLighting>
      <feComposite in="specOut" in2="SourceAlpha" operator="in" result="specOut"/>
      <feComposite in="SourceGraphic" in2="specOut" operator="arithmetic"
                   k1="0" k2="1" k3="1" k4="0" result="litPaint"/>
      <!-- merge offsetBlur + litPaint -->
      <feMerge>
        <feMergeNode in="offsetBlur"/>
        <feMergeNode in="litPaint"/>
      </feMerge>
    </filter>
  </defs>
  <!-- Graphic elements -->
  <g filter="url(#MyFilter)">
      <path fill="none" stroke="#D90000" stroke-width="10"
            d="M50,66 c-50,0 -50,-60 0,-60 h100 c50,0 50,60 0,60z" />
      <path fill="#D90000"
            d="M60,56 c-30,0 -30,-40 0,-40 h80 c30,0 30,40 0,40z" />
      <g fill="#FFFFFF" stroke="black" font-size="45" font-family="Verdana" >
        <text x="52" y="52">SVG</text>
      </g>
  </g>
</svg>

```html
<svg width="250" viewBox="0 0 200 85"
     xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <!-- Filter declaration -->
    <filter id="MyFilter" filterUnits="userSpaceOnUse"
            x="0" y="0"
            width="200" height="120">

      <!-- offsetBlur -->
      <feGaussianBlur in="SourceAlpha" stdDeviation="4" result="blur"/>
      <feOffset in="blur" dx="4" dy="4" result="offsetBlur"/>

      <!-- litPaint -->
      <feSpecularLighting in="blur" surfaceScale="5" specularConstant=".75"
                          specularExponent="20" lighting-color="#bbbbbb"
                          result="specOut">
        <fePointLight x="-5000" y="-10000" z="20000"/>
      </feSpecularLighting>
      <feComposite in="specOut" in2="SourceAlpha" operator="in" result="specOut"/>
      <feComposite in="SourceGraphic" in2="specOut" operator="arithmetic"
                   k1="0" k2="1" k3="1" k4="0" result="litPaint"/>

      <!-- merge offsetBlur + litPaint -->
      <feMerge>
        <feMergeNode in="offsetBlur"/>
        <feMergeNode in="litPaint"/>
      </feMerge>
    </filter>
  </defs>

  <!-- Graphic elements -->
  <g filter="url(#MyFilter)">
      <path fill="none" stroke="#D90000" stroke-width="10"
            d="M50,66 c-50,0 -50,-60 0,-60 h100 c50,0 50,60 0,60z" />
      <path fill="#D90000"
            d="M60,56 c-30,0 -30,-40 0,-40 h80 c30,0 30,40 0,40z" />
      <g fill="#FFFFFF" stroke="black" font-size="45" font-family="Verdana" >
        <text x="52" y="52">SVG</text>
      </g>
  </g>
</svg>
```

## 滤镜原始属性

滤镜通过 `<filter>` 元素进行定义，并且置于 `<defs>` 区块中。在  `<filter>` 标签中提供一系列图元（primitives），以及在前一个基本变换操作上建立的另一个操作（比如添加模糊后又添加明亮效果）。

`<filter>` 元素作用是作为原子滤镜操作的容器。它不能直接呈现。可以利用目标SVG元素上的 `filter` 属性引用一个滤镜。

**注意**：必须给`<filter>` 元素指定一个id属性，否则文档内的其他元素就不能引用它。

滤镜的原始属性有：
- `result` 属性定义了滤镜的输出名。如果提供了它，则经过滤镜处理的结果可以再次滤镜处理，在后继滤镜（即另一个`<filter>`元素）上通过一个`in`属性引用之前的结果。如果没有提供`result`值，而下一个滤镜也没有提供`in`属性值，则输出只可作为下一个滤镜的隐式输入。
- `in` 定义了滤镜的源。如果没有提供值并且这是`<filter>`中第一个源，那么将相当于使用`SourceGraphic`作为输入值。如果没有提供值并且这不是第一个源，那么将使用前面的`result`属性值作为输入。
  `in` 值可以是下面六种关键词中的一种
  - `SourceGraphic` 该关键词表示图形元素自身
  - `SourceAlpha` 该关键词表示图形元素自身的透明度
  - `BackgroundImage` 表示`<filter>`的背景
  - `BackgroundAlpha` 表示`<filter>`的背景的透明度
  - `FillPaint` 
  - `StrokePaint` 

## 高斯模糊

`<feGaussianBlur>` ：该滤镜对输入图像进行高斯模糊（Gaussian Blur）
- `stdDeviation` 属性可定义模糊的程度（模糊操作的标准差）

<svg width="230" height="120" xmlns="http://www.w3.org/2000/svg">
  <filter id="blurMe">
    <feGaussianBlur in="SourceGraphic" stdDeviation="5" />
  </filter>
  <circle cx="60"  cy="60" r="50" fill="green" />
  <circle cx="170" cy="60" r="50" fill="green"
          filter="url(#blurMe)" />
</svg>

```html
<svg width="230" height="120"
 xmlns="http://www.w3.org/2000/svg"
 xmlns:xlink="http://www.w3.org/1999/xlink">

  <filter id="blurMe">
    <feGaussianBlur in="SourceGraphic" stdDeviation="5" />
  </filter>

  <circle cx="60"  cy="60" r="50" fill="green" />

  <circle cx="170" cy="60" r="50" fill="green"
          filter="url(#blurMe)" />
</svg>
```

## SVG 阴影

`<feOffset>` 滤镜让图形整体相对当前位置偏移
- `dx, dy` 属性定义偏移量

`<feMerge>` 滤镜通过子元素 `<feMergeNode>` 合并多个滤镜

<svg width="120" height="120" xmlns="http://www.w3.org/2000/svg">
  <filter id="dropShadow">
    <feGaussianBlur in="SourceAlpha" stdDeviation="3" />
    <feOffset dx="2" dy="4" />
    <feMerge>
        <feMergeNode />
        <feMergeNode in="SourceGraphic" />
    </feMerge>
  </filter>
  <circle cx="60"  cy="60" r="50" fill="green"
          filter="url(#dropShadow)" />
</svg>

```html
<svg width="120" height="120"
 xmlns="http://www.w3.org/2000/svg">

  <filter id="dropShadow">
    <feGaussianBlur in="SourceAlpha" stdDeviation="3" />
    <feOffset dx="2" dy="4" />
    <feMerge>
        <feMergeNode />
        <feMergeNode in="SourceGraphic" />
    </feMerge>
  </filter>

  <circle cx="60"  cy="60" r="50" fill="green"
          filter="url(#dropShadow)" />
</svg>
```

## 颜色变换矩阵

`<feColorMatrix>`元素用于彩色滤光片转换。示例如下

```html
<filter id="f1" x="0%" y="0%" width="100%" height="100%">  
  <feColorMatrix   
     result="original" id="c1" type="matrix"   
     values="1 0 0 0 0    
             0 1 0 0 0   
             0 0 1 0 0   
             0 0 0 1 0" />  
</filter>
```
上述`<feColorMatrix>`滤镜的类型值为matrix，除此之外，还有saturate（饱和度）和hueRotate（色相旋转），取值比较简单，这里不做说明。

`<feColorMatrix>` 标签的 matrix 属性是一个 4x5 的矩阵。前面 4 列是颜色通道的比例系数，最后一列是常量偏移。

$$
\begin{pmatrix}
rr & rg & rb & ra & c1 \\
gr & gg & gb & ga & c2 \\
br & bg & bb & ba & c3 \\
ar & ag & ab & aa & c4 
\end{pmatrix} 
\begin{pmatrix}r \\ g \\ b \\ a \\ 1 \end{pmatrix} = 
\begin{pmatrix}
r*rr + g*rg + b*rb + a*ra + c1 \\
r*gr + g*gg + b*gb + a*ga + c2 \\
r*br + g*bg + b*bb + a*ba + c3 \\
r*ar + g*ag + b*ab + a*aa + c4 
\end{pmatrix}
$$

上面公式中的 rr 表示 red to red 系数。以此类推，c1-c4 表示常量偏移。

第一个 4x5 矩阵为变换矩阵，第二个单列矩阵为待变换对象的像素值。右侧单列矩阵为矩阵 1 和 2 的点积结果。

这个变换矩阵看起来比较复杂，在实践上常使用一个简化的对角矩阵，即除了 rr/gg/bb/aa 取值非零外，其余行列取值为 0，这就退化成了简单的各颜色通道的独立调整。

显然当变换矩阵为单位对角矩阵时，变换结果和原值相等。

我们可以尝试调整比例系数，比如把 rr 的值设置为0，即去除图像中的red颜色通道含量：

<svg  height=200 viewBox="0 0 150 70" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <filter id="colorMatrix">
        <feColorMatrix in="SourceGraphic" type="matrix" values="0 0 0 0 0
           0 1 0 0 0
           0 0 1 0 0
           0 0 0 1 0" />
    </filter>
    <g filter="">
        <circle cx="30" cy="30" r="20" fill="red" fill-opacity="0.5" />
    </g>
    <g filter="url(#colorMatrix)">
        <circle cx="80" cy="30" r="20" fill="red" fill-opacity="0.5" />
    </g>
</svg>

```html
<svg width="100%" height="100%" viewBox="0 0 150 120" 
     preserveAspectRatio="xMidYMid meet" 
     xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink">

    <filter id="colorMatrix">
        <feColorMatrix in="SourceGraphic" type="matrix" 
        values="0 0 0 0 0
                0 1 0 0 0
                0 0 1 0 0
                0 0 0 1 0" />
    </filter>
    <g filter="">
        <circle cx="30" cy="30" r="20" fill="red" fill-opacity="0.5" />
    </g>
    <g filter="url(#colorMatrix)">
        <circle cx="80" cy="30" r="20" fill="red" fill-opacity="0.5" />
    </g>
</svg>
```

# 渐变元素

渐变是从一种颜色到另一种颜色的平滑过渡。另外，可以把多个颜色的过渡应用到同一个元素上。

在 SVG 中，有两种主要的渐变类型：线性渐变和径向渐变

**注意**：必须给渐变内容指定一个id属性，否则文档内的其他元素就不能引用它。为了让渐变能被重复使用，渐变内容需要定义在`<defs>`标签内部，而不是定义在形状上面。

使用渐变时，我们需要在一个对象的属性`fill`或属性`stroke`中链接此渐变 id。

## 线性渐变

线性渐变沿着直线改变颜色，要插入一个线性渐变，你需要在SVG文件的 `defs` 元素内部，创建一个`<linearGradient>` 节点。

<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:rgb(255,255,0);stop-opacity:1" />
      <stop offset="100%" style="stop-color:rgb(255,0,0);stop-opacity:1" />
    </linearGradient>
  </defs>
  <ellipse cx="200" cy="70" rx="85" ry="55" fill="url(#grad1)" />
  <text fill="#ffffff" font-size="45" font-family="Verdana" x="150" y="86">
  SVG</text>
</svg>

```html
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:rgb(255,255,0);stop-opacity:1" />
      <stop offset="100%" style="stop-color:rgb(255,0,0);stop-opacity:1" />
    </linearGradient>
  </defs>
  <ellipse cx="200" cy="70" rx="85" ry="55" fill="url(#grad1)" />
  <text fill="#ffffff" font-size="45" font-family="Verdana" x="150" y="86">
  SVG</text>
</svg>
```

- 线性渐变内部有几个 `<stop>` 结点，这些结点通过指定位置的offset（偏移）属性和stop-color（颜色中值）属性来说明在渐变的特定位置上应该是什么颜色
  - offse属性用来指定位置（0-1 或 0％-100％）
  - stop-color 定义本stop元素的颜色中值
  - stop-opacity 定义本stop元素的不透明度 (0-1)
- 渐变的方向可以通过两个点来控制 (x1, y1) 和 (x2, y2) ，默认为 (0％, 0％) 和 (100％, 100％)
- gradientTransform 适用于渐变的转换，将在后续章节介绍变换

## 径向渐变

径向渐变与线性渐变相似，只是它是从一个点开始发散绘制渐变。创建径向渐变需要在文档的 `<defs>` 中添加一个`<radialGradient>`元素

<svg width="240" height="120" version="1.1" xmlns="http://www.w3.org/2000/svg">
  <defs>
      <radialGradient id="RadialGradient1">
        <stop offset="0%" stop-color="red"/>
        <stop offset="100%" stop-color="blue"/>
      </radialGradient>
      <radialGradient id="RadialGradient2" cx="0.25" cy="0.25" r="0.25">
        <stop offset="0%" stop-color="red"/>
        <stop offset="100%" stop-color="blue"/>
      </radialGradient>
  </defs>
  <rect x="10" y="10" rx="15" ry="15" width="100" height="100" fill="url(#RadialGradient1)"/>
  <rect x="120" y="10" rx="15" ry="15" width="100" height="100" fill="url(#RadialGradient2)"/>
</svg>

```html
<svg width="240" height="120" version="1.1" xmlns="http://www.w3.org/2000/svg">
  <defs>
      <radialGradient id="RadialGradient1">
        <stop offset="0%" stop-color="red"/>
        <stop offset="100%" stop-color="blue"/>
      </radialGradient>
      <radialGradient id="RadialGradient2" cx="0.25" cy="0.25" r="0.25">
        <stop offset="0%" stop-color="red"/>
        <stop offset="100%" stop-color="blue"/>
      </radialGradient>
  </defs>
  
  <rect x="10" y="10" rx="15" ry="15" width="100" height="100" fill="url(#RadialGradient1)"/>
  <rect x="120" y="10" rx="15" ry="15" width="100" height="100" fill="url(#RadialGradient2)"/>
  
</svg>
```

- 经向渐变内部有几个 `<stop>` 结点，这些结点通过指定位置的offset（偏移）属性和stop-color（颜色中值）属性来说明在渐变的特定位置上应该是什么颜色。由中心到环形边缘。
  - offset 属性用来定义渐变偏移量（0-1 或 0％-100％）。
  - stop-color 定义本stop元素的颜色
  - stop-opacity 定义本stop元素的不透明度 (0-1)
- cx, cy 描述了渐变中心位置（默认 -50%, 50%）
- r 定义渐变的半径（默认 50%）
- fx, fy 定义渐变的焦点（默认 0%, 0%）
- gradientTransform 适用于渐变的转换，将在后续章节介绍变换

## spreadMethod

线性渐变和径向渐变都需要一些额外的属性用于描述渐变过程，这里我希望额外提及一个`spreadMethod`属性，该属性控制了当渐变到达终点的行为。这个属性可以有三个值：pad、reflect或repeat。

- Pad为默认值，即当渐变到达终点时，最终的偏移颜色被用于填充对象剩下的空间。
- reflect会让渐变一直持续下去，不过它的效果是与渐变本身是相反的，以100%偏移位置的颜色开始，逐渐偏移到0%位置的颜色，然后再回到100%偏移位置的颜色。
- repeat也会让渐变继续，但是它不会像reflect那样反向渐变，而是跳回到最初的颜色然后继续渐变。

![](https://gitee.com/WilenWu/images/raw/master/common/SVG_Gradient_spreadMethod.svg)

## gradientUnits

两种渐变都有一个叫做 `gradientUnits`（渐变单元）的属性，它描述了用来描述渐变的大小和方向的单元系统。该属性有两个值：`userSpaceOnUse` 、`objectBoundingBox`。

- 默认值为objectBoundingBox，它大体上定义了对象的渐变大小范围，所以你只要指定从0到1的坐标值，渐变就会自动的缩放到对象相同大小。
- userSpaceOnUse使用绝对单元，所以你必须知道对象的位置，并将渐变放在同样地位置上。

# 坐标变换

SVG 还有一个功能，即可以定义自己的坐标系，其含义类似于解析几何中的坐标变换和映射规则。变换提供了一种整体的方式，用它可对一个或一组图像对象进行变换，改变其比例、位置、形状等，以达到使用自定义坐标系的目的。

SVG的坐标变换方式主要分为五种，包括平移变换，旋转变换，伸缩变换， 倾斜变换，还有矩阵变换。所有接下来的变换都会用元素的`transform`属性。变换可以连缀，只要把它们连接起来就行，用空格隔开。

<svg width="50%" viewBox="-40 0 150 100" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <g fill="grey"
     transform="rotate(-10 50 100)
                translate(-36 45.5)
                skewX(40)
                scale(1 0.5)">
    <path id="heart" d="M 10,30 A 20,20 0,0,1 50,30 A 20,20 0,0,1 90,30 Q 90,60 50,90 Q 10,60 10,30 z" />
  </g>
  <use xlink:href="#heart" fill="none" stroke="red"/>
</svg>

```html
<svg viewBox="-40 0 150 100" 
     xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink">
  <g fill="grey"
     transform="rotate(-10 50 100)
                translate(-36 45.5)
                skewX(40)
                scale(1 0.5)">
    <path id="heart" d="M 10,30 
                        A 20,20 0,0,1 50,30 
                        A 20,20 0,0,1 90,30 
                        Q 90,60 50,90 
                        Q 10,60 10,30 z" />
  </g>

  <use xlink:href="#heart" fill="none" stroke="red"/>
</svg>
```

## 平移变换

平移变换就是将元素移动一段距离，新坐标系的坐标轴方向不改变。语法为

```
transform="translate(x，y)"
```

这个表达式表示新得到的坐标系的原点平移到原来坐标系的点 `(x，y) `。

## 旋转变换

旋转变换是种相当常见的任务，语法为

```
transform="rotate(angle, x, y)"
```

其中有两个关键值，一个是旋转角度 angle （单位是度，其正值表示顺时针旋转，负值表示逆时针旋转）；另一个关键值是旋转中心的坐标 (x, y)，默认为 (0, 0) 。 

<svg width="50%" viewBox="0 0 200 50">
  <text x="0" y="15" fill="red" transform="rotate(30 20,40)">
    I love SVG
  </text>
</svg>

```html
<svg width="50%" viewBox="0 0 200 50">
  <text x="0" y="15" fill="red" transform="rotate(30 20,40)">
    I love SVG
  </text>
</svg>
```

## 伸缩变换

伸缩变换顾名思义就是将图像进行伸展和缩小。语法为

```
transform="scale(x，y)"
```

其中的关键值为坐标轴方向的比例因子，当比例因子大于1时表示图像被拉伸变大了，当比例因子小于1时表示图像被缩小了。 

## 倾斜变换

倾斜变换的关键值是倾斜的角度。语法为

```
transform="skewX(x-angle)"
transform="skewY(y-angle)"
```

其中 skewX是代表x轴上的变换，x-angle 是沿x轴歪斜的角度，skewY是代表y轴上的变换，y-angle 是沿y轴歪斜的角度。

## 矩阵变换

所有上面的变换都可以表达为一个2x3的变换矩阵。对于一些复杂的变换，可以直接用`matrix()`变换设置结果矩阵。语法为

```
transform="matrix(a, b, c, d, e, f)"
```

它通过以下矩阵将坐标从以前的坐标系映射到新的坐标系：

$$
\begin{pmatrix}
x_{new} \\
y_{new} \\
1
\end{pmatrix}=
\begin{pmatrix}
a & c & e \\
b & d & f \\
0 & 0 & 1  
\end{pmatrix}
\begin{pmatrix}
x_{prev} \\
y_{prev} \\
1
\end{pmatrix}=
\begin{pmatrix}
ax_{prev}+cy_{prev}+e \\
bx_{prev}+dy_{prev}+f \\
1
\end{pmatrix}
$$

# 附录

## 缩略图插件

Windows 系统默认是无法查看SVG图形文件的缩略图，很多时候我们想像预览 JPG、PNG 等图片一样批量预览 SVG 文件。Github 上已经有大神开发免费开源的 Windows 资源管理器的扩展模块 [tibold/svg-explorer-extension](https://github.com/tibold/svg-explorer-extension) 以呈现 SVG 缩略图，下载一个适合你电脑的版本，安装完成后，就可以直接查看SVG图形文件的缩略图了。如果安装后缩略图不显示，可以在 Github 查找故障方法。

![](https://gitee.com/WilenWu/images/raw/master/common/svg-see.png)

## ApacheCN 图标

<img src="https://wilenwu.gitee.io/img/apachecn.svg" width="50%"/>


> **参考链接：**
> [SVG 教程 | 菜鸟教程 (runoob.com)](https://www.runoob.com/svg/svg-tutorial.html)
> [SVG教程 - SVG | MDN (mozilla.org)](https://developer.mozilla.org/zh-CN/docs/Web/SVG/Tutorial)
