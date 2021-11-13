---
ID: a6be1f1f2719bfe477cd1420bd44b1c6
title: Hexo 标签插件的使用
date: 2019-09-11 14:10:48
categories: [博客搭建]
tags: [Hexo]
cover: /img/hexo.jpg
top_img: /img/hexo-logo.svg
noticeOutdate: true
description: 
---
「tag 插件」(Tag Plugin) 是 Hexo 提供的一种快速生成特定内容的方式，并不是标准的Markdown格式。 Hexo 内置来许多标签来帮助写作者可以更快的书写， [完整的标签列表](https://hexo.io/zh-cn/docs/tag-plugins) 可以参考 Hexo 官网。 另外，Hexo 也开放来接口给主题，使主题有可能提供给写作者更简便的写作方法。 

<!-- more -->

# Hexo 标签插件

## 引用块

在文章中插入引言，可包含作者、来源和标题，均可选。

标签方式：使用 `blockquote` 或者 简写 `quote`。

```sh
{% blockquote author, source link source_link_title %}
content
{% endblockquote %}
```

{% blockquote author, source link source_link_title %}
content
{% endblockquote %}


## 代码块

在文章中插入代码，包含指定语言、附加说明和网址，均可选。
标签方式：使用 `codeblock` 或者 简写 `code`。

```sh
{% codeblock [title] [lang:language] [url] [link text] %}
code snippet
{% endcodeblock %}
```

{% codeblock title lang:language url link text %}
code snippet
{% endcodeblock %}

反引号代码块

\`\`\`[language] [title] [url] [link text] 
code snippet 
\`\`\`

## iframe

在文章中插入 iframe。

```
{% iframe url [width] [height] %}
```

## Image

在文章中插入指定大小的图片。

```
{% img [class names] /path/to/image [width] [height] "title text 'alt text'" %}
```

## Link

在文章中插入链接，并自动给外部链接添加 `target="_blank"` 属性。

```
{% link text url [external] [title] %}
```

## Include Code

插入 `source/downloads/code` 文件夹内的代码文件。`source/downloads/code` 不是固定的，取决于你在配置文件中 `code_dir` 的配置。

```sh
{% include_code [title] [lang:language] path/to/file %}
```

## 文章摘要和截断

在文章中使用 `<!-- more -->`，那么 `<!-- more -->` 之前的文字将会被视为摘要。首页中将只出现这部分文字，同时这部分文字也会出现在正文之中。

注意，摘要可能会被 Front Matter 中的 `excerpt` 覆盖。

## 相对路径引用的标签插件

通过常规的 markdown 语法和相对路径来引用图片和其它资源可能会导致它们在存档页或者主页上显示不正确。在Hexo 2时代，社区创建了很多插件来解决这个问题。但是，随着Hexo 3 的发布，许多新的[标签插件](https://hexo.io/docs/tag-plugins#Include-Assets)被加入到了核心代码中。这使得你可以更简单地在文章中引用你的资源。

```
{% asset_path slug %}
{% asset_img slug [title] %}
{% asset_link slug [title] %}
```

比如说：当你打开文章资源文件夹功能后，你把一个 `example.jpg` 图片放在了你的资源文件夹中，如果通过使用相对路径的常规 markdown 语法 `![](example.jpg)` ，它将 *不会* 出现在首页上。（但是它会在文章中按你期待的方式工作）

正确的引用图片方式是使用下列的标签插件而不是 markdown ：

```
{% asset_img example.jpg This is an example image %}
```

通过这种方式，图片将会同时出现在文章和主页以及归档页中。

