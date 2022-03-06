---
title: R手册(Tidy+Transform)--stringr
tags:
  - R
  - tidyverse
  - 数据清洗
  - 正则表达式
categories:
  - R
  - 数据清洗
cover: /img/stringr-cover.png
description: 'Simple, Consistent Wrappers for Common String Operations'
abbrlink: c4e3d4d8
date: 2018-05-01 22:35:55
top_img:
---


stringr: Simple, Consistent Wrappers for Common String Operations

<!-- more -->

# stringr

函数|说明
:---|---
str_c(..., sep = "", collapse =NULL)|将单个字符向量合并成单个字符串时用collapse<br>*(base::paste, base::paste0)*
str_length(string) |字符串中的字符数.*(base::nchar)*
str_trim(string, side = c("both", "left", "right"))|去空白
str_pad(string, width, side,pad = " ")|补充字符串的长度,side = c("left", "right", "both")
str_dup(string, times)|# 重复和连接strings
str_to_upper(string); <br>str_to_lower; <br>str_to_title (首字母大写)|
**起始位置取子集**|
str_sub(string, start , end)| *base::substr*
**匹配字符向量**|
str_detect(string, pattern)|   返回逻辑值*(base::grepl)*
str_subset(string, pattern)|  返回匹配到的字符串(base::grep)
**提取匹配到的字符**|
str_extract(string, pattern)|提取首个匹配模式的字符
str_extract_all(string, pattern, simplify = FALSE)|`simplify`: If FALSE, the default, returns a list of character vectors
str_count(string, pattern)|返回匹配到的数量
**替换匹配项**|
str_replace(string, pattern, replacement)|
str_replace_all(string, pattern, replacement)|`str_replace_all(“fruits”, "[aeiou]", toupper)`
str_replace_na(string, replacement = "NA")|
**分割字符串**|
str_split(string, pattern, n = Inf, simplify = FALSE)|n为分割数
str_split_fixed(string, pattern, n)|
**查找匹配项位置**|
str_locate(string, pattern)| #返回start ,end
str_locate_all(string, pattern)|
**字符串排序**|
str_order(x, decreasing =FALSE, na_last = TRUE)|  返回对应的顺序数字<br>`na_last`(NA位置): TRUE (末尾), FALSE(开始), NA(删除) 
str_sort(x, decreasing =FALSE, na_last)|返回排序后的向量
**pattern转化函数**|
regex(pattern, ignore\_case = FALSE, multiline = FALSE, comments = FALSE, dotall = FALSE, ...) <br>`>>> str_detect(c('Apple','a car'), regex('a',ignore_case = T))`<br>`[1] TRUE TRUE` |ignore\_case可以忽略大小写<br>multiline: If TRUE, `$`和`^`匹配开始和结尾位置<br>comments: If TRUE,以空格和#开头的将被忽略<br>dotall: If TRUE, . will also match line terminators.



----------

# stringi

函数|说明
:---|---
e1 %s+% e2| 连接两个字符
stri_enc_get()|获取默认编码
stri_encode(str,from ,to,to_raw = FALSE)|转换编码
stri_conv(str,from,to ,to_raw = FALSE)| 转换编码
stri_read_lines(fname, encoding = "auto")|*base::readLines*
stri_write_lines(str, fname, encoding = "UTF-8")|

----------

# R语言编码

> Encoding(someX)<-'UTF-8'/'GB2312'  #将正确的编码标记告诉 R 
> iconv(x,from,to)   #转换编码



