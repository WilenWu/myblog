---
title: 标记语言概览
tags:
  - 标记语言
categories:
  - 标记语言
cover: /img/markup-cover.jpg
top_img: /img/markupdocs-ipad.jpg
abbrlink: 2957012b
date: 2021-11-23 21:13:52
description:
emoji: heart
---

**标记语言**（markup language，ML）是一种将文本（Text）以及文本相关的其他信息结合起来，展现出关于文档结构和数据处理细节的计算机文字编码。标记语言广泛应用于网页和网络应用程序。标记最早用于出版业，是作者、编辑以及出版商之间用于描述出版作品的排版格式所使用的。

<!-- more -->

# 标记语言的分类

标记语言通常可以分为三类：表示性的、过程性的以及描述性的。

- **表示性的标记语言**（Presentational Markup）是在编码过程中，标记文档的结构信息。
- **过程性标记语言**（Procedural Markup）一般都专门于文字的表达，但通常对于文本编辑者可见，并且能够被软件依其出现顺序依次解读。
- **描述性标记**（Descriptive Markup）也称**通用标记**，所描述的是文件的内容或结构，而不是文件的显示外观或样式，制定SGML的基本思想就是把文档的内容与样式分开，XML、SGML都是典型的通用标记语言。

**轻量级标记语言**（Lightweight Markup Language，LML）是一类用简单句法描述简单格式的文本语言。它语法简单，可方便地使用简单的文本编辑器输入，原生格式接近自然语言。所谓“轻量级”是相对于其他更丰富格式的标记语言而言的，比如：富文本格式语言RTF、超文本标记语言HTML、学术界排版语言TeX等。

# 常见标记语言

- **TeX** <img src="https://gitee.com/WilenWu/images/raw/master/common/TeX.svg" width="28%"  align="right"/>是一个由美国电脑教授高德纳（Donald E. Knuth）编写的功能强大的排版软件。它在学术界十分流行，特别是数学、物理学和计算机科学界。TeX被普遍认为是一个很好的排版工具，特别是在处理复杂的数学公式时。利用诸如是LaTeX等终端软件，TeX就能够排版出精美的文本。
- **LaTeX** <img src="https://gitee.com/WilenWu/images/raw/master/common/latex-logo-bird.svg" width="30%" align="right"/>是一种基于TEX的排版系统，遵循呈现与内容分离的设计理念，以便作者可以专注于他们正在编写的内容，而不必同时注视其外观。它非常适用于生成高印刷质量的科技和数学、物理文档。
- **KaTeX** 是一个在Web浏览器上显示数学符号的跨浏览器的JavaScript库。它特别强调快速和易于使用。它的布局基于TeX。与MathJax相比，它只处理LaTeX的数学符号的一个更小的子集。
- **SGML**（**S**tandard **G**eneralized **M**arkup **L**anguage，标准通用标记语言）是一种专门的标记语言，被用作编写《牛津英语词典》的电子版本。由于它的复杂，因而难以普及。
- **HTML** <img src="https://gitee.com/WilenWu/images/raw/master/common/HTML5-with-wordmark.svg" width="15%" height="15%" align="right"/>（**H**yper**T**ext **M**arkup **L**anguage，超文本标记语言）[^html]是一种用于创建网页的标准标记语言。HTML是一种基础技术，常与CSS、JavaScript一起被众多网站用于设计网页、网页应用程序以及移动应用程序的用户界面。
- **XML** （e**X**tensible **M**arkup **L**anguage，可扩展标记语言）是从标准通用标记语言（SGML）中简化修改出来的。XML 是对 HTML 的补充，而非替代。XML 被设计用来传输和存储数据，而 HTML 用于格式化并显示数据。
- **Markdown** <img src="https://gitee.com/WilenWu/images/raw/master/common/file-md.png" width="13%" height="13%" align="right"/>是一种轻量级标记语言。它允许人们使用易读易写的纯文本格式编写文档，然后转换成有效的XHTML（或者HTML）文档。这种语言吸收了很多在电子邮件中已有的纯文本标记的特性。
  由于Markdown的轻量化、易读易写特性，并且对于图片，图表、数学式都有支持，目前许多网站都广泛使用Markdown来撰写帮助文档或是用于论坛上发表消息。如GitHub、Reddit、Diaspora、Stack Exchange、OpenStreetMap 、SourceForge、简书等，甚至还能被用来撰写电子书。
- **RTF**<img src="https://gitee.com/WilenWu/images/raw/master/common/file-rtf.svg" width="15%" height="15%" align="right"/>（**R**ich **T**ext **F**ormat，富文本格式）是由微软公司开发的跨平台文档格式。大多数的文字处理软件都能读取和保存RTF文档。
- **YAML** 是 **Y**AML **A**in't a **M**arkup **L**anguage（YAML不是一种标记语言）的递归缩写。在开发的这种语言时，YAML 的意思其实是："Yet Another Markup Language"（仍是一种标记语言），但为了强调这种语言以数据做为中心，而不是以标记语言为重点，而用反向缩略语重命名
  YAML 是一个可读性高，用来表达资料序列化的格式。特别适合用来表达或编辑数据结构、各种配置文件、倾印调试内容、文件大纲。
- **JSON** <img src="https://gitee.com/WilenWu/images/raw/master/common/file-json.svg" width="15%" height="15%" align="right"/>（**J**ava**S**cript **O**bject **N**otation）是一种用于交换数据的轻量级标记语言。其内容由属性和值所组成，因此也有易于阅读和处理的优势。JSON是独立于编程语言的资料格式，其不仅是JavaScript的子集，也采用了C语言家族的习惯用法，目前也有许多编程语言都能够将其解析和字符串化，其广泛使用的程度也使其成为通用的资料格式。

# 基于XML的应用

很多新的互联网语言是基于 XML 创建的

- **XHTML** （e**X**tensible **H**yper**T**ext **M**arkup **L**anguage，可扩展超文本标记语言）表现方式与HTML类似，不过语法上更加严格，是以 XML 应用的方式定义的 HTML。
- **RSS** <img src="https://gitee.com/WilenWu/images/raw/master/common/feed-icon.svg" width="12%" align="right"/>（Really Simple Syndication，简易信息聚合）是一种消息来源格式规范，用以聚合多个网站更新的内容并自动通知网站订阅者。使用 RSS 后，网站订阅者便无需再手动查看网站是否有新的内容，同时 RSS 可将多个网站更新的内容进行整合，以摘要（feed）的形式呈现，有助于订阅者快速获取重要信息，并选择性地点阅查看。
- **WSDL**（Web Services Description Language，Web服务描述语言）是描述Web服务的公共接口。这是一个基于XML的关于如何与Web服务通讯和使用的服务描述。
- **同步多媒体集成语言**（Synchronized Multimedia Integration Language，SMIL）是W3C为采用XML描述多媒体而提出的建议标准。它定义了时间标签、布局标签、动画、视觉渐变（visual transitions）和媒体嵌入等。

- [**可缩放矢量图形**](https://www.w3.org/Graphics/SVG/) <img src="https://gitee.com/WilenWu/images/raw/master/common/file-svg.svg" width="15%" height="15%" align="right"/>（Scalable Vector Graphics，SVG）基于XML，用于描述二维矢量图形的图形格式。SVG文本格式的描述性语言来描述图像内容，因此是一种和图像分辨率无关的矢量图形格式。
  SVG Animation 是一种基于XML的开放标准矢量图形格式，可以通过各种方式实现：ECMAScript、CSS Animations、SMIL。
- **数学标记语言**（Mathematical Markup Language，MathML），是一种基于XML的标准，用来描述数学符号和公式。它的目标是把数学公式集成到W3C和其他文档中。从2015年开始，MathML成为了HTML5的一部分和ISO标准。
  由于数学符号和公式的结构复杂且符号与符号之间存在多种逻辑关系，MathML的格式十分繁琐。因此，大多数人都不会去手写MathML，而是利用其它的工具来编写，其中包括TEX到MathML的转换器。
- **地理标记语言**（Geography Markup Language，GML）<img src="https://gitee.com/WilenWu/images/raw/master/common/GML-demo.svg" width="15%" height="15%" align="right"/>是由OGC开放地理信息系统协会定义的XML格式，用来表达地理信息要素。
- [**化学标记语言**](http://www.xml-cml.org/)（Chemical Markup Language，ChemML或CML）是一种基于XML语言，用于描述化学分子、化学反应、光谱等化学数据的标记语言。可以使用Jumbo浏览器查看CML文件。
- **语音可扩展标记语言**（Voice Extensible Markup Language，VoiceXML或者VXML）是于交互式语音回应应用程序创建音频对话的标准，用于开发音频及声音回应应用程序，例如银行系统及自动客户服务。来自网页服务器的超文本标记语言（HTML）被网页浏览器接收后，网页浏览器能对其进行解析并视觉呈现出来。
- **语音合成标记语言**（Speech Synthesis Markup Language，SSML）是以XML为基础的标记语言，主要是用来支持语音合成应用程序。SSML经常内嵌于VoiceXML语言内以操控交互语音系统，但它也经常被单独使用，如制作有声书的时候。
- **系统生物学标记语言**（Systems Biology Markup Language，SBML）是基于XML的标记语言，用于描述生化反应等网络的计算模型。SBML可以描述代谢网络、细胞信号通路、调节网络、以及在系统生物学研究范畴中的其它系统。
- **墨水标记语言**（Ink Markup Language，InkML）是用于表达数字墨水数据的XML数据格式，这类数据的输入是通过作为多通路系统组成部分的电子笔或输入笔。
- **DocBook** 是一种基于XML的技术文件语义标记语言。它本来是设计用来编写有关计算机硬件和软件的技术文件，但它可以用于任何其它类型的文件。作为一个语义语言， DocBook让使用者能建立自定义的样式文件，使其能将内容转为不同格式，例如HTML、XHTML、EPUB、PDF、手册页、Web help、Microsoft 的HTML Help档案等。

[^html]: HTML 是 web 开发人员必须学习的 3 门语言中的基础：HTML 定义了网页的内容，CSS 描述了网页的布局， JavaScript 控制了网页的行为。
