---
title: R手册(Communicate)--R Markdown
tags:
  - R
  - markdown
categories:
  - R
  - Communicate
cover: /img/rmarkdown-cover.png
top_img: /img/r-top-img.svg
description: 用于创建可重复性报告和动态文档
abbrlink: 36ba9291
date: 2018-04-30 15:45:28
---

R Markdown是通过R语言制作动态文档的文件格式
 Cheat Sheet：[R Markdown](https://rstudio.com/wp-content/uploads/2015/03/rmarkdown-reference.pdf)

<!-- more -->

# Overview

**Installation**

- 如果你在RStudio中工作，那么你可以简单地安装[当前版本](http://www.rstudio.com/ide/download/preview)的RStudio（包括rmarkdown包和pandoc）。
- 如果您想在RStudio之外使用rmarkdown软件包
`install.packages("rmarkdown")`


**Workflow**

1. 编辑 R Markdown文件（Rmd 或 纯文本）：
 Rstudio菜单 File ▶ New File ▶ R Markdown
2. 渲染：use **knitr button** or **render()** to knit
3. 发布到网站（可选）
4. 重复使用

# Rmd Structure

R Markdown主要由三部分组成： 
1. YAML标头：文档开始部分，由`---`包围的`key:value`对

2. Text：主要由Markdown语法书写的文本

3. Knitr 处理 R代码块

   ~~~r
   ```{r}
   code
   ```
   ~~~

# YAML Header

## Parameters

1. Add parameters: 在YAML Header中定义
2. Call parameters: 在代码块中以`params$<name>`格式引用
3. Set parameters: Knit或者render() 时重设参数

```markdown
# YAML Header
---
params:
 num: 100
 date: !r Sys.Date()
---
# R Markdown文本
Today’s date is `r params$d`
# 渲染
render("doc.Rmd", params = list(n = 1,d = as.Date("2015-01-01"))
```

## Set render options with YAML

```markdown
---
output: html_document
---
Body
```

output value| creates
:------|:------
html_document |html
pdf_document |pdf (requires Tex )
word_document |Microsof Word (.docx)
odt_document |OpenDocument Text
rtf_document |Rich Text Format
md_document |Markdown
github_document |Github compatible markdown
ioslides_presentation ioslides |HTML slides
slidy_presentation |slidy HTML slides
beamer_presentation |Beamer pdf slides (requires Tex)

**ouput** 也可以设置子选项，详情请查看官网：
```markdown
---
output: 
  html_document:
    code_folding: hide
    toc: TRUE
    theme: united
    highlight: zenburn
---
Body
```

## 初始文档信息 

title, author, and date information,etc

```markdown
---
title: "Sample Document"
author: "Hadley Wickham"
---
```

# Text

更多语法请参考 [GitHub Flavored Markdown 规范](/posts/GitHub-Flavored-Markdown-Spec/)

# Embed code with knitr syntax

**Inline code**
以 `` `r <code>` ``格式插入，将以文本的格式显示

**Code chunks**
以 \`\`\`{r options}  code  \`\`\`围绕的一行或多行代码

chunks options	|说明
:------|:------
cache |为knits缓存结果 (default = FALSE)
cache.path |缓存结果的保存路径 (default = "cache/")
child|file(s) to knit and then include (default =NULL)
collapse| 把所有的输出汇聚到单个块中(default = FALSE)
comment|结果的每一行加前缀(default = '#')
dependson |chunk dependencies for caching (default = NULL) 
echo	|是否在输出中包含源代码
engine| 代码语言 (default ='R')
error	|是否在输出中包含错误信息（TRUE或FALSE）
eval | 在块中运行代码(default = TRUE)
fig.align | 'lef', 'right', or 'center' (default = 'default')
fig.cap |图片标签 (default= NULL)
fig.height, fig.width|图片宽度和高度（英寸）
highlight |高亮显示源码 (default = TRUE)
include | 运行后是否在文档中显示块 (default = TRUE)
message	|是否在输出中包含参考的信息（TRUE或FALSE）
results	|是否输出原生结果 (default = 'markup'){'markup','asis','hide','hold'}
warning	|是否在输出中包含警告(default = TRUE)

**Global options**
用`knitr::opts_chunk$set()` 等函数设置

```r
knitr::opts_chunk$set(echo = TRUE)
```

# 表格美化

```r
data <- faithful[1:4, ]

knitr::kable(data, caption = "Table with kable”)
print(xtable::xtable(data, caption = "Table with xtable”),
   type = "html", html.table.attributes = "border=0")) 
stargazer::stargazer(data, type = "html", title = "Table
   with stargazer")
```

# render(渲染)

`render()` 或`knitr button`
运行时先用Knitr将Rmd文件中的代码块编译成md文件
然后再用pandoc将md文件转化为HTML, PDF或 Word等最终文档(the default is HTML)

```r
render("input.Rmd")
render("input.md")
```

**render函数主要参数：**
input : file to render
output_format: 输出格式
output_options: 要渲染的选项list (in YAML)
output_file, output_dir : 输出文件名和路径
params： list of params to use
envir : environment to evaluate code chunks in
encoding - of input file

**Output Formats:** str or Output Format Function
all: 输出全部格式
*Output Format Functions：*
html_document
pdf_document
word_document
md_document
beamer_presentation
ioslides_presentation
slidy_presentation

```r
render("input.Rmd", html_document(toc = TRUE))
render("input.Rmd", pdf_document(latex_engine = "lualatex"))
render("input.Rmd", beamer_presentation(incremental = TRUE))
```
