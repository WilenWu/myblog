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

# SVG 概述

## 简介

[**可缩放矢量图形**](https://www.w3.org/Graphics/SVG/) <img src="https://gitee.com/WilenWu/images/raw/master/common/SVG-with-crown.svg" width="20%" align="right"/>（Scalable Vector Graphics，SVG）是一种用于描述二维的矢量图形，基于 XML 的标记语言。SVG由W3C制定，是一个开放标准，能够优雅而简洁地渲染不同大小的图形，并和CSS，DOM[^dom]，JavaScript和SMIL等其他网络标准无缝衔接。本质上，SVG 相对于图像，就好比 HTML 相对于文本。

SVG 图像及其相关行为被定义于 XML 文本文件之中，这意味着可以对它们进行搜索、索引、编写脚本以及压缩。此外，这也意味着可以使用任何文本编辑器和绘图软件来创建和编辑它们。

和传统的点阵图模式不同，SVG格式提供的是矢量图，这意味着它的图像能够被无限放大而不失真或降低质量，并且可以方便地修改内容。

<img src="https://gitee.com/WilenWu/images/raw/master/common/SVG-Bitmap.svg" width="50%"/>

由于行业需求，SVG 1.1 引入了两个移动配置文件：SVG Tiny (SVGT) 和SVG Basic (SVGB)。这些是完整 SVG 标准的子集，主要用于功能有限的用户代理。SVG Tiny 被定义为高度受限的移动设备（如手机），它不支持样式或脚本。SVG Basic是为智能手机等更高级的移动设备定义的。

[^dom]: DOM (Document Object Model) 译为**文档对象模型**，是 HTML 和 XML 文档的编程接口。DOM 定义了访问和操作网页文档的标准方法。

## 编辑器

- SVG 图像可以通过使用矢量图形编辑器（如Inkscape，Adobe Illustrator，Adobe Flash Professional或CorelDRAW）来生成，并件渲染为常见的光栅图像格式（如PNG）。
- 而开放源代码的软件有Scribus、Karbon14、Inkscape以及Sodipodi等。
- 也有开放源码，功能简单但容易操作，免安装的在线SVG设计工具，例如 [SVG-Edit](https://github.com/SVG-Edit/svgedit) 。
- 在移动设备上的软件有安卓的 PainterSVG。
- 另介绍两款在线编辑器：[SVG 菜鸟在线编辑器](https://c.runoob.com/more/svgeditor/) 和 [Figma: the collaborative interface design tool.](https://www.figma.com/)。

## SVG in HTML

将 SVG 代码写入 `.svg` 文件可以独立使用，也可以嵌入到 XML 网页或 HTML 网页中。

- `<svg>` 标签可直接写入网页
  ```html
  <svg
    xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xml:space="preserve" width="500" height="440"
  >
  <!-- svg code -->
  </svg>
  ```

- `<img>` 标签以图片的形式直接引入
  ```html
  <img src="demo.svg" />
  ```

- `<iframe>` 标签引入，可工作在大部分的浏览器中
  ```html
  <iframe src="demo.svg" ></iframe>
  ```

- `<object>` 标签是 HTML4 的标准标签，被所有较新的浏览器支持。它的缺点是不允许使用脚本

  ```html
  <object data="demo.svg" type="image/svg+xml" ></object>
  ```

- `<embed>` 标签被所有主流的浏览器支持，并允许使用脚本

  ```html
  <embed src="demo.svg" type="image/svg+xml" />
  ```

# 文件结构

基于XML的SVG，语法和格式也是结构化的。所谓结构化，也就是文件中的对象通过特定的元素标签定义，任何元素都可以作为对象进行管理，文件是框架式的。掌握基本的文件框架，就可以阅读、编辑和创作自己的文件。

SVG使用一组元素标签，创建和组织文件以及文件中的对象。每一个SVG文件都包含最外层的`<svg>`和`</svg>`标签。该标签用于声明SVG文件的开始和结束。

下面是一个简单的 SVG 文件例子：

```html
<?xml version="1.1" encoding="UTF-8" ?>
<svg width="100%" height="100%" viewBox="0 0 100 100" 
   xmlns="http://www.w3.org/2000/svg" 
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xml:space="preserve"
   >
<!-- svg code -->
</svg>
```

- 第一行是 XML 声明。它定义 XML 的版本和所使用的编码。
- `<svg>` 标签创建一个SVG文档片段，这是根元素。
  - x,y 属性定义SVG片段左上角的坐标，默认为 (0,0) 。
  - width,height 属性定义SVG片段的宽度和高度，默认 100%, 100%。除了相对单位，也可以采用绝对单位（像素：px）。
  - 视口属性 viewBox 展示 SVG 文档绘图区域。属性的值有由空格或逗号分隔的4个值 (min x, min y, width, height)，分别是左上角的横坐标和纵坐标、视口的宽度和高度。
  - 如果不指定width属性和height属性，只指定viewBox属性，则相当于只给定 SVG 图像的长宽比。这时，SVG 图像的默认大小将等于所在的 HTML 元素的大小。
  - version 属性可定义所使用的 SVG 版本。
  - `<svg>` 元素都需要安装SVG和它的命名空间 xmlns,xmlns:xlink,xml:space 
  - zoomAndPan='magnify' | 'disable'。magnify 是默认值，允许用户平移和缩放文档。
- `<desc>` 和 `<title>` 元素 SVG 还可以提供描述性内容，以帮助用户通过多种方式进行索引、搜索和检索，并不作为图形的一部分来显示。

# 绘图元素

## 基本图形

SVG 有一些预定义的形状元素，可被开发者使用和操作（坐标都是相对于`<svg>`画布的左上角原点）

- `<circle>` 创建圆形，基于圆心坐标 (cx, cy) 和半径 r 创建圆。默认圆心坐标为 (0, 0)
- `<ellipse>` 创建椭圆，基于中心坐标 (cx, cy) 以及水平半径 rx 和垂直半径 ry。
- `<rect>`创建矩形以及矩形的变种，基于左上角端点坐标 (x, y) 以及它的宽 width 和高 height。它还可以用圆角半径 rx 和 ry 来创建圆角矩形。
- `<polygon>` 创建多边形，points 属性指定了每个端点的坐标，横坐标与纵坐标之间用逗号分隔，点与点之间用空格分隔。
- `<line>` 基于两个端点坐标 (x1, x2) 和 (x2, y2) 创建一条线段。
- `<polyline>` 创建折线，points 属性指定了每个端点的坐标，横坐标与纵坐标之间用逗号分隔，点与点之间用空格分隔。

<svg width="150" height="150" viewBox="-10 -10 270 270" >
<rect fill="#fff" stroke="#000" x="-10" y="-10" width="270" height="270" rx= "30" />
<g opacity="0.8">
  <rect x="25" y="25" width="200" height="200" fill="lime" stroke-width="4" stroke="pink" />
  <circle cx="125" cy="125" r="75" fill="orange" />
  <polyline points="50,150 50,200 200,200 200,100" stroke="red" stroke-width="4" fill="none" />
  <line x1="50" y1="50" x2="200" y2="200" stroke="blue" stroke-width="4" />
</g>
</svg> 

<svg width="150" height="150" version="1.1">
  <ellipse cx="240" cy="100" rx="220" ry="30" style="fill:purple" />
  <ellipse cx="220" cy="70" rx="190" ry="20" style="fill:lime" />
</svg>

```html
<svg width="150" height="150" viewBox="-10 -10 270 270" >
<rect fill="#fff" stroke="#000" x="-10" y="-10" width="270" height="270" rx= "30" />
<g opacity="0.8">
  <rect x="25" y="25" width="200" height="200" fill="lime" stroke-width="4" stroke="pink" />
  <circle cx="125" cy="125" r="75" fill="orange" />
  <polyline points="50,150 50,200 200,200 200,100" stroke="red" stroke-width="4" fill="none" />
  <line x1="50" y1="50" x2="200" y2="200" stroke="blue" stroke-width="4" />
</g>
</svg> 

<svg width="150" height="150" version="1.1">
  <ellipse cx="240" cy="100" rx="220" ry="30" style="fill:purple" />
  <ellipse cx="220" cy="70" rx="190" ry="20" style="fill:lime" />
</svg>
```

## 图形属性

**opacity 属性**：定义整个元素的透明值，范围为 0 - 1
**stroke 属性**：SVG 提供了一系列 stroke 属性，可应用于任何种类的线条，包括文字和元素的轮廓。
- stroke 属性定义线条颜色
- stroke-width 属性定义线条宽度
- stroke-linecap 属性定义不同类型的线头形状
- stroke-dasharray 属性用于创建虚线
- stroke-opacity 属性定义线条颜色透明度，范围为 0 - 1

<svg xmlns="http://www.w3.org/2000/svg" version="1.1"> 
  <g fill="none" stroke-width="4"> 
    <path stroke="red" stroke-dasharray="5,5" d="M5 20 l215 0" />    
    <path stroke="black" stroke-dasharray="10,10" d="M5 40 l215 0" />    
    <path stroke="blue" stroke-dasharray="20,10,5,5,5,10" d="M5 60 l215 0" />  
  </g>
  <g fill="none" stroke="black" stroke-width="6">
    <path stroke-linecap="butt" d="M5 100 l215 0" />
    <path stroke-linecap="round" d="M5 120 l215 0" />
    <path stroke-linecap="square" d="M5 140 l215 0" />
  </g>
</svg>

```html
<svg xmlns="http://www.w3.org/2000/svg" version="1.1"> 
  <g fill="none" stroke-width="4"> 
    <path stroke="red" stroke-dasharray="5,5" d="M5 20 l215 0" />    
    <path stroke="black" stroke-dasharray="10,10" d="M5 40 l215 0" />    
    <path stroke="blue" stroke-dasharray="20,10,5,5,5,10" d="M5 60 l215 0" />  
  </g>
  <g fill="none" stroke="black" stroke-width="6">
    <path stroke-linecap="butt" d="M5 100 l215 0" />
    <path stroke-linecap="round" d="M5 120 l215 0" />
    <path stroke-linecap="square" d="M5 140 l215 0" />
  </g>
</svg>
```

**fill 属性**：SVG 也提供了一系列 fill 属性用于填充，包括形状类元素和文字内容类元素。
- fill 属性用于颜色
- fill-opacity 属性定义填充颜色透明度，范围为 0 - 1
- fill-rule 属性用于指定使用哪一种算法去判断画布上的某区域是否属于该图形==内部==（内部区域将被填充），算法有 nonzero(default), evenodd, inherit。[见附录](#填充规则)

使用 `<polygon>` 元素创建一个星型:

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

改变 fill-rule 属性为 "evenodd":

<svg height="210" width="500">
  <polygon points="100,10 40,198 190,78 10,78 160,198"
    fill="lime" fill-rule="evenodd"
    stroke="purple" stroke-width="5"
  />
</svg>

**CSS 属性**

- SVG 内容可以使用 style 属性用来嵌入 CSS 属性，语法为 `style="key1:value1;key2:value2;..."`，一般以分号分隔各属性定义。常用的属性包括 fill,stroke 等
- SVG 内容 class 属性用来指定对应的 CSS 类

```css
.fancy {
  fill: none;
  stroke: black;
  stroke-width: 3pt;
}
```

## 路径元素

`<path>` 标签用于绘制路径，所有的基本形状都可以用 `<path>` 元素来创建。
- `d` 属性表示绘制顺序，它的值是一个长字符串。字符串中每个字母表示一个绘制动作，后面跟着坐标。
- pathLength 属性指定路径长度。如果存在，则计算各点相对于此值的路径。

| 动作                   | 说明   |
| :-------------------- | :----- |
| M | moveto（移动到）                      |
| L | lineto（画直线到）                     |
| H | horizontal lineto（画水平直线到）        |
| V | vertical lineto（画垂直直线到）        |
| C | curveto（曲线到）                 |
| S | smooth curveto（光滑曲线到）          |
| Q | quadratic Belzier curve（二次贝塞尔曲线到） |
| T | smooth quadratic Belzier curveto（光滑二次贝塞尔曲线到） |
| A | elliptical Arc（椭圆弧）              |
| Z | closepath（闭合路径）                  |

{% note info %} 
以上所有命令均允许小写字母。大写表示绝对定位，小写表示相对定位。
{% endnote %}

下面的例子创建了一个二次方贝塞尔曲线，A 和 C 分别是起点和终点，B 是控制点：

<svg xmlns="http://www.w3.org/2000/svg" version="1.1" height="300" viewBox="0 0 450 400">
<path id="lineAB" d="M 100 350 l 150 -300" stroke="red" stroke-width="3" fill="none" />
  <path id="lineBC" d="M 250 50 l 150 300" stroke="red" stroke-width="3" fill="none" />
  <path d="M 175 200 l 150 0" stroke="green" stroke-width="3" fill="none" />
  <path d="M 100 350 q 150 -300 300 0" stroke="blue" stroke-width="5" fill="none" />
  <!-- Mark relevant points -->
  <g stroke="black" stroke-width="3" fill="black">
    <circle id="pointA" cx="100" cy="350" r="3" />
    <circle id="pointB" cx="250" cy="50" r="3" />
    <circle id="pointC" cx="400" cy="350" r="3" />
  </g>
  <!-- Label the points -->
  <g font-size="30" font="sans-serif" fill="black" stroke="none" text-anchor="middle">
    <text x="100" y="350" dx="-30">A</text>
    <text x="250" y="50" dy="-10">B</text>
    <text x="400" y="350" dx="30">C</text>
  </g>
</svg>

```html
<svg xmlns="http://www.w3.org/2000/svg" version="1.1" height="300" viewBox="0 0 450 400">
  <path id="lineAB" d="M 100 350 l 150 -300" stroke="red" stroke-width="3" fill="none" />
  <path id="lineBC" d="M 250 50 l 150 300" stroke="red" stroke-width="3" fill="none" />
  <path d="M 175 200 l 150 0" stroke="green" stroke-width="3" fill="none" />
  <!-- quadratic Belzier curve -->
  <path d="M 100 350 q 150 -300 300 0" stroke="blue" stroke-width="5" fill="none" />
  <!-- Mark relevant points -->
  <g stroke="black" stroke-width="3" fill="black">
    <circle id="pointA" cx="100" cy="350" r="3" />
    <circle id="pointB" cx="250" cy="50" r="3" />
    <circle id="pointC" cx="400" cy="350" r="3" />
  </g>
  <!-- Label the points -->
  <g font-size="30" font="sans-serif" fill="black" stroke="none" text-anchor="middle">
    <text x="100" y="350" dx="-30">A</text>
    <text x="250" y="50" dy="-10">B</text>
    <text x="400" y="350" dx="30">C</text>
  </g>
</svg>
```

下面展示了一个螺旋线：

<svg width="200" height="200" version="1.1" viewBox="150 250 60 150" >
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
<svg width="200" height="200" version="1.1" viewBox="150 250 60 150" >
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

## 文本元素

`<text>` 标签绘制文本
- (x, y) 属性表示文本区块基线（baseline）起点的坐标，默认为 (0, 0)
- dx, dy 属性控制文本相对起点移动
- rotate 属性给出文本旋转列表
- textLength 属性如果给出，将尝试调整文本长度
- lengthAdjust=spacing' | 'spacingAndGlyphs' 如果指定长度将尝试调整文本
- font 属性定义字体	 
- font-face 系列属性描述字体的特征
- font-face-format
- font-face-name
- font-face-src
- font-face-uri

**路径上的文本**

<svg height="100" version="1.1">
  <defs>
    <path id="path1" d="M0,20 a1,1 0 0,0 100,0" />
  </defs>
  <text x="0" y="100" fill="red">
    <textPath xlink:href="#path1">
      I love SVG I love SVG
    </textPath>
  </text>
</svg>

```html
<svg height="100" version="1.1">
  <defs>
    <path id="path1" d="M0,20 a1,1 0 0,0 100,0" />
  </defs>
  <text x="0" y="100" fill="red">
    <textPath xlink:href="#path1">
      I love SVG I love SVG
    </textPath>
  </text>
</svg>
```

`<tspan>` 元素等同于 `<text>`，但可以在文本内部嵌套分组。每个 `<tspan>` 元素可以包含不同的格式和位置。

<svg height="100" xmlns="http://www.w3.org/2000/svg" version="1.1">
  <text x="10" y="20" style="fill:red;">Several lines:
    <tspan x="10" y="45">First line</tspan>
    <tspan x="10" y="70">Second line</tspan>
  </text>
</svg>

```html
<svg height="100" xmlns="http://www.w3.org/2000/svg" version="1.1">
  <text x="10" y="20" style="fill:red;">Several lines:
    <tspan x="10" y="45">First line</tspan>
    <tspan x="10" y="70">Second line</tspan>
  </text>
</svg>
```

## SVG 元素

- `<a>` 元素创建一个 SVG 元素链接 
  - xlink:show 
  - xlink:actuate
  - xlink:href 属性规定链接指向的页面的 URL
  - target 属性规定在何处打开链接文档，取值有 '_blank', '_parent',   '_self', '_top', 'framename'
  
  <svg height="30" version="1.1">
    <a xlink:href="https://www.w3.org/Graphics/SVG/" target="_blank">
      <text x="0" y="25" fill="red">W3C SVG Working Group</text>
    </a>
  </svg>
  
  ```html
  <svg height="30" version="1.1">
    <a xlink:href="https://www.w3.org/Graphics/SVG/" target="_blank">
      <text x="0" y="25" fill="red">W3C SVG Working Group</text>
    </a>
  </svg>
  ```

- `<use>` 标签用于引用一个元素。
  - xlink:href 属性指定所要复制的节点。当链接 id 时，必须使用 # 前缀。
  - (x, y) 定义左上角的坐标。另外，还可以指定width和height。
- `<image>` 标签用于插入图片文件。xlink:href 属性定义图像的来源。基于左上角端点坐标 (x, y) 以及它的宽 width 和高 height。

- `<g>` 标签用于将多个元素组成一个组（group），方便复用。在`<g>`元素内可以定义任何可视化的基本几何元素，例如
  - id 属性定义唯一组名
  - fill 属性定义该组填充颜色"
  - opacity 属性定义改组不透明度

  <svg width="300" height="100">
    <g id="myCircle">
      <text x="25" y="20">circle</text>
      <circle cx="50" cy="50" r="20"/>
    </g>
    <use href="#myCircle" x="100" y="0" fill="blue" />
    <use href="#myCircle" x="200" y="0" fill="white" stroke="blue" />
  </svg>

  ```html
  <svg width="300" height="100">
    <g id="myCircle">
      <text x="25" y="20">circle</text>
      <circle cx="50" cy="50" r="20"/>
    </g>
  
    <use href="#myCircle" x="100" y="0" fill="blue" />
    <use href="#myCircle" x="200" y="0" fill="white" stroke="blue" />
  </svg>
  ```

- `<defs>`标签是 definitions 的缩写，用于自定义被引用的元素（诸如滤镜等特殊元素），它内部的代码不会显示，仅供引用。

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

- `<pattern>`标签用于自定义一个形状，该形状可以被引用来平铺一个区域。
  - id 属性为模式定义一个唯一的名称（同一模式可被文档中的多个元素使用）。当链接 id 时，必须使用 # 前缀。
  - x,y 定义模式对象视口左上角偏移量，默认为 0,0
  - viewBox 定义模式对象视口区域。由空格或逗号分隔的4个值。(min x, min y, width, height)
  - width, height 定义模式对象宽和高，单位（％）。
  - patternUnits="userSpaceOnUse" | "objectBoundingBox"。
  "userSpaceOnUse" 表示`<pattern>`的宽度和长度是实际的像素值。
  - patternContentUnits='userSpaceOnUse' | 'objectBoundingBox'
  - patternTransform 属性允许整个表达式进行转换
  - xlink:href 模式链接

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

- `<marker>` 标签可以放在直线，折线，多边形和路径的顶点。这些元素可以使用marker属性的"marker-start"，"marker-mid"和"marker-end"，继承默认情况下或可设置为"none"或自定义的标记的URI。您必须先定义标记，然后才可以通过其URI引用。
  - markerUnits='strokeWidth' |'userSpaceOnUse'。
  - refx, refy 标记顶点连接的坐标，默认为 0,0
  - orient="auto" 始终显示标记的角度。
  - markerWidth, markerHeight 标记的宽度和高度，默认 3*3
  - viewBox标记视口区域区域。由空格或逗号分隔的4个值 (min x, min y, width, height)

- `<mask>` 
- `<script>` 脚本容器

## SVG 动画

SVG Animation 是一种基于XML的开放标准矢量图形格式，可以通过各种方式实现：ECMAScript、CSS Animations、SMIL。W3C 明确建议将 SMIL 作为 SVG 中动画的标准。

[SVG 在线动画编辑 | SVGA](https://svga.io/index.html)

`<animate>`标签用于产生动画效果
- attributeName 发生动画效果的属性名
- from 单次动画的初始值
- to 单次动画的结束值
- dur 单次动画的持续时间
- repeatCount 动画的循环模式，可以定义为无限循环 "indefinite"

<svg width="500" height="110">
  <rect x="0" y="0" width="100" height="100" fill="#feac5e">
    <animate attributeName="x" from="0" to="500" dur="5s" repeatCount="indefinite" />
    <animate attributeType="CSS" attributeName="opacity" from="1" to="0" dur="5s" repeatCount="indefinite" />
  </rect>
</svg>

```html
<svg width="500" height="110">
  <rect x="0" y="0" width="100" height="100" fill="#feac5e">
    <animate attributeName="x" from="0" to="500" dur="5s" repeatCount="indefinite" />
  </rect>
</svg>
```


`<animateTransform>` 元素进行动态的属性转换。此标签对 CSS 的transform属性不起作用。
- attributeName 发生动画效果的属性名
- by 相对偏移值
- from 单次动画的初始值
- to 单次动画的结束值
- dur 单次动画的持续时间
- type 随时间变化的转换类型 'translate', 'scale', 'rotate', 'skewX', 'skewY'
- repeatCount 动画的循环模式，可以定义为无限循环 "indefinite"


```html
<svg width="500px" height="500px">
  <rect x="250" y="250" width="50" height="50" fill="#4bc0c8">
    <animateTransform attributeName="transform" type="rotate" begin="0s" dur="10s" from="0 200 200" to="360 400 400" repeatCount="indefinite" />
  </rect>
</svg>
```

`<animateMotion>` 使元素沿着路径移动 
- calcMode 动画的插补模式。可以是'discrete', 'linear', 'paced', 'spline'
- path 运动路径
- keyPoints 
- rotate 应用旋转变换 
- xlink:href 引用到另一个动画

`<animateColor>` 随时间进行的颜色变换
- by 相对偏移值
- from 单次动画的初始值
- to 单次动画的结束值

# SVG 滤镜

基于 SVG 构建一个矢量图像时，我们使用的基本标签元素包括根元素`<svg>`、形状元素，还有特殊效果处理中的滤镜效果和渐变效果。 

## 滤镜简介

在 SVG 中，可用的滤镜有：

- feBlend 使用不同的混合模式把两个对象合成在一起
- [feColorMatrix](#颜色变换矩阵) 用于彩色滤光片转换
- feComponentTransfer 执行数据的 component-wise 重映射
- feComposite
- feConvolveMatrix
- feDiffuseLighting
- feDisplacementMap
- feFlood
- feGaussianBlur 高斯模糊
- feImage
- feMerge 建立在彼此顶部图像层
- feMorphology 对源图形执行 fattening 或者 thinning
- feOffset  相对其当前位置移动图像
- feSpecularLighting
- feTile
- feTurbulence
- feDistantLight 定义一个光源
- fePointLight 用于照明过滤
- feSpotLight 用于照明过滤

**注意**：必须用 `<filter>` 元素来自定义 SVG 滤镜，`<filter>` 元素嵌套在 `<defs>` 标签内。然后图形元素的 `filter` 属性使用 id 定向滤镜。

## 高斯模糊

**高斯模糊**（Gaussian Blur）：滤镜效果是通过 `<feGaussianBlur>` 标签进行定义的。fe 后缀可用于所有的滤镜。

- `<filter>` 标签的 id 属性为滤镜定义一个唯一的名称（同一滤镜可被文档中的多个元素使用）
- filter:url 属性用来把元素链接到滤镜。当链接滤镜 id 时，必须使用 # 前缀
- stdDeviation 属性可定义模糊的程度
- in 定义了滤镜的原始输入：SourceGraphic | SourceAlpha | BackgroundImage | BackgroundAlpha | FillPaint | StrokePaint | `<filter-primitive-reference>`

<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <filter id="f1" x="0" y="0">
      <feGaussianBlur in="SourceGraphic" stdDeviation="15" />
    </filter>
  </defs>
  <rect width="90" height="90" stroke="green" stroke-width="3"
  fill="yellow" filter="url(#f1)" />
</svg>

```html
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <filter id="f1" x="0" y="0">
      <feGaussianBlur in="SourceGraphic" stdDeviation="15" />
    </filter>
  </defs>
  <rect width="90" height="90" stroke="green" stroke-width="3"
  fill="yellow" filter="url(#f1)" />
</svg>
```

## SVG 阴影

`<feOffset>` 元素让图像相对其当前位置移动，适合用于创建阴影效果。
`<feBlend>` 把两个对象合成在一起	
- mode 图像混合模式：normal|multiply|screen|darken|lighten
- in 定义了滤镜的原始输入：SourceGraphic | SourceAlpha | BackgroundImage | BackgroundAlpha | FillPaint | StrokePaint | `<filter-primitive-reference>`

第一个例子偏移一个矩形（带`<feOffset>`），然后混合偏移图像（含`<feBlend>`）：

<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <filter id="f2" x="0" y="0" width="200%" height="200%">
      <feOffset result="offOut" in="SourceGraphic" dx="20" dy="20" />
      <feBlend in="SourceGraphic" in2="offOut" mode="normal" />
    </filter>
  </defs>
  <rect width="90" height="90" stroke="green" stroke-width="3"
  fill="yellow" filter="url(#f2)" />
</svg>

```html
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <filter id="f2" x="0" y="0" width="200%" height="200%">
      <feOffset result="offOut" in="SourceGraphic" dx="20" dy="20" />
      <feBlend in="SourceGraphic" in2="offOut" mode="normal" />
    </filter>
  </defs>
  <rect width="90" height="90" stroke="green" stroke-width="3"
  fill="yellow" filter="url(#f2)" />
</svg>
```

模糊偏移对象，并为阴影涂上一层颜色：

<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <filter id="f3" x="0" y="0" width="200%" height="200%">
      <feOffset result="offOut" in="SourceGraphic" dx="20" dy="20" />
      <feColorMatrix result="matrixOut" in="offOut" type="matrix"
      values="0.2 0 0 0 0 0 0.2 0 0 0 0 0 0.2 0 0 0 0 0 1 0" />
      <feGaussianBlur result="blurOut" in="matrixOut" stdDeviation="10" />
      <feBlend in="SourceGraphic" in2="blurOut" mode="normal" />
    </filter>
  </defs>
  <rect width="90" height="90" stroke="green" stroke-width="3"
  fill="yellow" filter="url(#f3)" />
</svg>

```html
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <filter id="f3" x="0" y="0" width="200%" height="200%">
      <feOffset result="offOut" in="SourceGraphic" dx="20" dy="20" />
      <feColorMatrix result="matrixOut" in="offOut" type="matrix"
      values="0.2 0 0 0 0 0 0.2 0 0 0 0 0 0.2 0 0 0 0 0 1 0" />
      <feGaussianBlur result="blurOut" in="matrixOut" stdDeviation="10" />
      <feBlend in="SourceGraphic" in2="blurOut" mode="normal" />
    </filter>
  </defs>
  <rect width="90" height="90" stroke="green" stroke-width="3"
  fill="yellow" filter="url(#f3)" />
</svg>
```

> 代码解析：
> - `<feGaussianBlur>` 元素的stdDeviation属性定义了模糊量
> - `<feOffset>` 元素的属性改为"SourceAlpha" 即在Alpha通道使用残影，而不是整个RGBA像素。
> - `<feColorMatrix>` 过滤器是用来转换偏移的图像使之更接近黑色。 


# SVG 渐变

渐变是从一种颜色到另一种颜色的平滑过渡。另外，可以把多个颜色的过渡应用到同一个元素上。

在 SVG 中，有两种主要的渐变类型：

- 线性渐变
- 放射性渐变

**注意**：渐变元素定义必须嵌套在 `<defs>` 标签内。图像的 fill:url属性链接此渐变 id

## 线性渐变

`<linearGradient>` 可用来定义 SVG 的线性渐变。线性渐变可以定义为水平，垂直或角渐变。`<linearGradient>` 标签必须嵌套在 `<defs>` 的内部。

- 渐变的颜色范围可由两种或多种颜色组成。每种颜色通过一个 `<stop>` 标签来规定。`<stop>` 标签嵌套在`<linearGradient>` 之内。
  - offset 属性用来定义渐变偏移量（0-1 或 0％-100％）。
  - stop-color 定义本stop元素的颜色
  - stop-opacity 定义本stop元素的不透明度 (0-1)
- id 属性可为渐变定义一个唯一的名称。当链接 id 时，必须使用 # 前缀。
- (x1, y1) 和 (x2, y2) 定义线性渐变的起点和终点，默认为 (0％, 0％) 和 (100％, 100％)
- gradientUnits='userSpaceOnUse' or 'objectBoundingBox'(default) 使用视口或对象，以确定相对位置矢量点。
- gradientTransform 适用于渐变的转换
- spreadMethod='pad' or 'reflect' or 'repeat' 
- xlink:href 引用到另一个渐变

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


## 放射性渐变

`<radialGradient>` 用来定义放射性渐变。`<radialGradient>` 创建渐变椭圆，沿中心向外渐变。

- 渐变的颜色范围可由两种或多种颜色组成。每种颜色通过一个 `<stop>` 标签来规定。`<stop>` 标签嵌套在`<linearGradient>` 之内。
  - offset 属性用来定义渐变偏移量（0-1 或 0％-100％）。
  - stop-color 定义本stop元素的颜色
  - stop-opacity 定义本stop元素的不透明度 (0-1)
- id 属性可为渐变定义一个唯一的名称。当链接 id 时，必须使用 # 前缀。
- cx, cy 定义渐变的中心（默认 -50%, 50%）
- r 定义渐变的半径（默认 50%）
- fx, fy 定义渐变的焦点（默认 0%, 0%）
- gradientUnits='userSpaceOnUse' or 'objectBoundingBox'(default) 使用视口或对象，以确定相对位置矢量点。
- gradientTransform 适用于渐变的转换
- spreadMethod='pad' or 'reflect' or 'repeat' 
- xlink:href 引用到另一个渐变


定义一个放射性渐变从白色到蓝色椭圆：

<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <radialGradient id="grad2" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
      <stop offset="0%" style="stop-color:rgb(255,255,255);stop-opacity:0" />
      <stop offset="100%" style="stop-color:rgb(0,0,255);stop-opacity:1" />
    </radialGradient>
  </defs>
  <ellipse cx="200" cy="70" rx="85" ry="55" fill="url(#grad2)" />
</svg>

```html
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
  <defs>
    <radialGradient id="grad2" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
      <stop offset="0%" style="stop-color:rgb(255,255,255);stop-opacity:0" />
      <stop offset="100%" style="stop-color:rgb(0,0,255);stop-opacity:1" />
    </radialGradient>
  </defs>
  <ellipse cx="200" cy="70" rx="85" ry="55" fill="url(#grad2)" />
</svg>
```


# 坐标变换

SVG 还有一个功能，即可以定义自己的坐标系。其方法是在一段SVG文本中定义一种叫做变换(Transformation)的格式，其含义类似于解析几何中的坐标变换和映射规则。变换提供了一种整体的方式，用它可对一个或一组图像对象进行变换，改变其比例、位置、形状等，以达到使用自定义坐标系的目的。

SVG的坐标变换方式主要分为五种，包括平移变换，旋转变换，伸缩变换， 倾斜变换，还有矩阵变换。

## 平移变换

平移变换的特点是变换后，新坐标系的坐标轴方向不改变。
它的表达式为：`transform="translate(x，y)"`
这个表达式表示新得到的坐标系的原点平移到原来坐标系的点 (x，y) 。

## 旋转变换

旋转变换的关键是旋转角度。
它的表达式为：`transform="rotate(angle, x, y)"` 。其中有两个关键值，一个是旋转角度 angle （单位是度，其正值表示顺时针旋转，负值表示逆时针旋转）；另一个关键值是旋转中心的坐标 (x, y)，默认为 (0, 0) 。 

<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 200 50">
  <text x="0" y="15" fill="red" transform="rotate(30 20,40)">
    I love SVG
  </text>
</svg>

```html
<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 200 50">
  <text x="0" y="15" fill="red" transform="rotate(30 20,40)">
    I love SVG
  </text>
</svg>
```

## 伸缩变换

伸缩变换顾名思义就是将图像进行伸展和缩小。
它的表达式为 `transform="scale(x，y)"` 。其中的关键值为坐标轴方向的比例因子，当比例因子大于1时表示图像被拉伸变大了，当比例因子小于1时表示图像被缩小了。 

## 倾斜变换

倾斜变换的关键是倾斜的角度。
它的表达式为 `transform="skewX(x-angle)"` 和 `transform="skewY(y-angle)"`。其中 skewX是代表x轴上的变换，x-angle 是沿x轴歪斜的角度，skewY是代表y轴上的变换，y-angle 是沿y轴歪斜的角度。

## 矩阵变换

矩阵变换的最大特点就是他有6个参数，所以，这种变换很灵活，对于特别复杂的变换，对于矩阵变换也只需进行一次运算就行了。
它的表达式为：`transform="matrix(abcdef)"`


# 附录

## 填充规则

图像的 **fill-rule** 属性用于指定使用哪一种算法去判断画布上的某区域是否属于该图形==内部==（内部区域将被填充），算法有 nonzero(default), evenodd, inherit。

对一个简单的无交叉的路径，哪块区域是==内部==是很直观清楚的。但是，对一个复杂的路径，比如自相交或者一个子路径包围另一个子路径，==内部==的理解就不那么明确了。

- **nonzero**：字面意思是“非零”。按该规则，要判断一个点是否在图形内，从该点作任意方向的一条射线，然后检测射线与图形路径的交点情况。从0开始计数，路径从左向右穿过射线则计数加1，从右向左穿过射线则计数减1。得出计数结果后，如果结果是0，则认为点在图形外部，否则认为在内部。下图演示了nonzero规则

  ![](https://gitee.com/WilenWu/images/raw/master/common/SVG-nonzero.svg)

- **evenodd**：字面意思是“奇偶”。按该规则，要判断一个点是否在图形内，从该点作任意方向的一条射线，然后检测射线与图形路径的交点的数量。如果结果是奇数则认为点在内部，是偶数则认为点在外部。下图演示了evenodd 规则

  ![](https://gitee.com/WilenWu/images/raw/master/common/SVG-evenodd.svg)

  **提示:** 上述解释未指出当路径片段与射线重合或者相切的时候怎么办，因为任意方向的射线都可以，那么只需要简单的选择另一条没有这种特殊情况的射线即可。

## 颜色变换矩阵

**变换矩阵的定义和说明**

`<feColorMatrix>` 标签的 matrix 属性是一个 4*5 的矩阵。前面 4 列是颜色通道的比例系数，最后一列是常量偏移。
$$
\begin{bmatrix}
rr & rg & rb & ra & c1 \\
gr & gg & gb & ga & c2 \\
br & bg & bb & ba & c3 \\
ar & ag & ab & aa & c4 
\end{bmatrix} \cdot
\begin{bmatrix}r \\ g \\ b \\ a \\ 1 \end{bmatrix} = 
\begin{bmatrix}
r*rr + g*rg + b*rb + a*ra + c1 \\
r*gr + g*gg + b*gb + a*ga + c2 \\
r*br + g*bg + b*bb + a*ba + c3 \\
r*ar + g*ag + b*ab + a*aa + c4 
\end{bmatrix}
$$

上面公式中的 rr 表示 red to red 系数，以此类推。c1-c4 表示常量偏移。

第一个 4*5 矩阵为变换矩阵，第二个单列矩阵为待变换对象的像素值。右侧单列矩阵为矩阵 1 和 2 的点积结果。

这个变换矩阵看起来比较复杂，在实践上常使用一个简化的对角矩阵，即除了 rr/gg/bb/aa 取值非零外，其余行列取值为 0，这就退化成了简单的各颜色通道的独立调整。

`<feColorMatrix>` 的语法:

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

上述feColorMatrix过滤器的类型值为matrix，除此之外，还有saturate（饱和度）和hueRotate（色相旋转），取值比较简单，这里不做说明。

显然当变换矩阵为单位对角矩阵时，变换结果和原值相等。

我们可以尝试调整比例系数，比如把rr的值设置为0，即去除图像中的red颜色通道含量：

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
<svg width="100%" height="100%" viewBox="0 0 150 120" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">

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
```


## 缩略图插件

Windows 系统默认是无法查看SVG图形文件的缩略图，很多时候我们想像预览 JPG、PNG 等图片一样批量预览 SVG 文件。Github 上已经有大神开发免费开源的 Windows 资源管理器的扩展模块 [tibold/svg-explorer-extension](https://github.com/tibold/svg-explorer-extension) 以呈现 SVG 缩略图，下载一个适合你电脑的版本，安装完成后，就可以直接查看SVG图形文件的缩略图了。如果安装后缩略图不显示，可以在 Github 查找故障方法。

![](https://gitee.com/WilenWu/images/raw/master/common/svg-see.png)



> **参考链接：**
> [HTML5如何使用SVG - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/54088196)
> [SVG 教程 | 菜鸟教程 (runoob.com)](https://www.runoob.com/svg/svg-tutorial.html)
> [最新全网最详细前端从零入门实战教程_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1ZE411c7yM?p=101&spm_id_from=pageDriver)
