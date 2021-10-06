---
ID: 46266b3c282a41b71bae70d35c6021f4
title: R手册(Import)--rvest
tags: [R,rvest,爬虫]
mathjax: false
copyright: true
date: 2018-05-01 17:49:56
categories: [R,Import]
sticky: false
---
rvest: Simple web scraping for R

<!-- more -->


# 解析html

函数|说明
:---|:---
read_html(x, ..., encoding = "") |x为a url或 a local path
html_nodes(x, css, xpath)|通过使用 XPath and css ，**selectors**`( read vignette("selectorgadget")` to learn about it)选择文档的一部分

# 提取组件

函数|说明
:---|:---
html_text(x)|提取标签内的文本
html_attr (x,name)|提取单个属性的内容，如href超链接
html_attrs(x)| 所有属性
html_tag(x) |标签名
html_table()|Parse html tables into data frames

>**for XML:**  read_xml , xml_node, xml_attr, xml_attrs, xml_text and xml_tag

# 提取，修改和提交形式的函数

```r
 html_form()
 set_values() 
 submit_form()
```
# 浏览网站

```r
html_session()
jump_to()
follow_link()
back(), forward()
submit_form()
# and so on
```


**Examples**

```r
lego_movie <-read_html("http://www.imdb.com/title/tt1490017/")
rating <- lego_movie %>%
  html_nodes("strong span") %>%
  html_text() %>%
  as.numeric()
```



