---
title: 主流的博客框架及部分主题
categories: 
  - 博客搭建
tags:
  - Hexo
  - Jekyll
  - Hugo
  - MkDocs
  - ReadTheDocs
abbrlink: 9e8cfeb8
date: 2022-03-06 12:11:11
cover:
top_img: /img/hexo-top-img.png
description: false
swiper_index: 2
---

[Hexo还是Hugo？Typecho还是Wordpress？读完这篇或许你就有答案了！](https://blog.laoda.de/archives/blog-choosing)

[Static Site Generators - Top Open Source SSGs | Jamstack](https://jamstack.org/generators/)

# Hexo

[Hexo](https://hexo.io/zh-cn/) 是基于Node.js的静态站点生成框架。

## 主题

[Hexo theme](https://hexo.io/themes/)

- [NexT](https://theme-next.js.org/)：Elegant and Powerful Theme for Hexo（功能强大）
- [Butterfly ](https://butterfly.js.org/)：A Simple and Card UI Design theme for Hexo（中文支持性很好）
- [doc](https://zalando-incubator.github.io/hexo-theme-doc/)：这是Hexo的**文档主题**。它与其他Hexo主题的不同之处在于允许您呈现文档 - 特别是REST API文档。
- [matery](http://blinkfox.com/2018/09/28/qian-duan/hexo-bo-ke-zhu-ti-zhi-hexo-theme-matery-de-jie-shao/)：这是一个采用`Material Design`和响应式设计的 Hexo 博客主题

![hexo-next](https://gitee.com/WilenWu/images/raw/master/common/hexo-next.png)
![hexo-butterfly](https://gitee.com/WilenWu/images/raw/master/common/hexo-butterfly.png)
![hexo-doc](https://gitee.com/WilenWu/images/raw/master/common/hexo-doc.png)
![hexo-blinkfox](https://gitee.com/WilenWu/images/raw/master/common/matery-20181202-1.png)

# Jekyll

[Jekyll](https://www.jekyll.com.cn/) 由Ruby编写，是一个简洁的博客、静态网站生成工具。用你喜欢的标记语言书写内容并交给 Jekyll 处理，它将利用模板为你创建一个静态网站。

[Jekyll Themes](http://jekyllthemes.org/)


# Hugo

[Hugo](https://gohugo.io/) 是一个基于 Go 语言开发的静态网站生成器，号称世界上最快的构建网站工具

[Hugo 与 Hexo 的异同 - 云+社区 - 腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1578634)

## 主题

[Hugo Themes](https://themes.gohugo.io/)

- [Eureka](https://www.wangchucheng.com/zh/docs/hugo-eureka/) 是一个功能丰富且可高度定制化的Hugo主题，由国人开发。
- [Docsy](https://www.docsy.dev/) 是 Hugo 技术文档集的主题，提供简单的导航、网站结构等。
- [Learn](https://learn.netlify.app/en/) 完全是为文档设计的主题

![Hugo-Eureka](https://gitee.com/WilenWu/images/raw/master/common/Hugo-Eureka.png)
![hugo-docsy](https://gitee.com/WilenWu/images/raw/master/common/hugo-docsy.png)
![hugo-learn](https://gitee.com/WilenWu/images/raw/master/common/hugo-learn.png)

# MkDocs

## 文档

[笔记文档一把梭——MkDocs 快速上手指南 ｜ 少数派会员 π+Prime (sspai.com)](https://sspai.com/prime/story/mkdocs-primer)

[MkDocs中文文档 (zimoapps.com)](https://mkdocs.zimoapps.com/)：MkDocs（**M**ar**k**down **Doc**ument**s**）是一个**快速**、**简单**、**华丽**的静态网站生成器，适用于构建简单的文档网站。MkDocs 基于 Python 编写，也贯彻了 Python 里「简洁胜于复杂」的理念，与其他常见的静态网站生成器相比，无需繁琐的环境配置，所有配置都用只有简单的一个YAML配置文件管理。

```
.
├── docs
│   ├── index.md
└── mkdocs.yml
```

![mkdocs](https://gitee.com/WilenWu/images/raw/master/common/mkdocs.png)

## 主题

MkDocs内置了两个主题， mkdocs 和readthedocs， 也可以从[MkDocs wiki(github.com)](https://github.com/mkdocs/mkdocs/wiki/MkDocs-Themes)中选择第三方主题

- [ReadTheDocs Dropdown for MkDocs](https://github.com/cjsheets/mkdocs-rtd-dropdown)：ReadTheDocs 的修改版本
- [Material for MkDocs ](https://squidfunk.github.io/mkdocs-material/)：Google 推行的 Material Design 风格

![readthedocs](https://gitee.com/WilenWu/images/raw/master/common/mkdocs-readthedocs.png)
![material](https://gitee.com/WilenWu/images/raw/master/common/mkdocs-material.jpg)

## Markdown 扩展

MkDocs 使用 [Python-Markdown](https://github.com/Python-Markdown/markdown) 库（Markdown 规范的 Python 实现）来渲染 Markdown 内容，因此 MkDocs 中对于 Markdown 内容渲染的扩展也是来自于此。我们可以在 Python-Markdown 的 [官方文档](https://python-markdown.github.io/extensions/) 中浏览到目前所支持的 Markdown 扩展有哪些。

比如我在当中开启对于 Markdown 内容标题的固定标识符、脚注以及表格。

除此之外，Python-Markdown 还支持一些来自于第三方的 Markdown 扩展，你能在 Python-Markdown 的 Github 仓库中的 [Wiki](https://github.com/Python-Markdown/markdown/wiki/Third-Party-Extensions) 页面找到符合你需要的扩展，比如支持数学公式、emoji 表情、增强 Markdown 语法等。

但这些第三方扩展都还是需要通过 `pip`工具（安装 Python 解释器时会自带）来进行安装，安装之后在 `mkdocs.yml`中的`markdown_extension`部分继续追加。

# Sphinx

[Sphinx](https://www.sphinx.org.cn/) 是一个基于 Python 的文档生成工具。最早只是用来生成 Python 的项目文档，使用 reStructuredText[.rst] 作为标记语言。但随着 Sphinx 项目的逐渐完善，目前已发展成为一个大众可用的框架，很多非 Python 的项目也采用 Sphinx 作为文档写作工具，甚至完全可以用 Sphinx 来写书。

## 主题

Sphinx 为我们提供了好多可选的主题，在 [Sphinx Themes Gallery](https://sphinx-themes.org/#theme-sphinx-rtd-theme) 都可以找到。大家最熟悉的应该是 [Read the Docs](https://sphinx-themes.org/#theme-sphinx-rtd-theme) 主题和[Book](https://sphinx-themes.org/#theme-sphinx-book-theme)主题

![Sphinx-readthedocs](https://gitee.com/WilenWu/images/raw/master/common/Sphinx-readthedocs.png)
![Sphinx-book](https://gitee.com/WilenWu/images/raw/master/common/Sphinx-book.png)

## Markdown 支持

Sphinx 本身不支持Markdown文件生成文档，需要我们使用第三方库recommonmark进行转换。另外，如果需要支持 markdown 的表格语法，还需要安装 sphinx-markdown-tables 插件。

# docsify

[docsify](https://docsify.js.org/#/zh-cn/) 可以快速帮你生成文档网站，基于 Node.js。不同于 GitBook、Hexo 的地方是它不会生成静态的 `.html` 文件，所有转换工作都是在运行时。

![docsify](https://gitee.com/WilenWu/images/raw/master/common/docsify.png)