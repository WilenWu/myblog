---
ID: 3a5d91e9124d3889c9afef5c3e331c0d
title: R手册(Tidy+Transform)--tidyr
tags: [R,tidyverse,数据清洗]
date: 2018-05-28 16:33:39
categories: [R,Tidy+Transform]
cover: /img/tidyr.png
top_img: 
---

**tidyr**: Easily tidy data with spread and gather functions.

<!-- more -->

# Reshape Data

- **gather将多列聚集成键值对(key-value pairs)** *(reshap2::melt)*
`gather(data,key, value, ..., na.rm = FALSE, convert = FALSE,factor_key = FALSE)`
 > 参数：
 > **key, value:** 要聚合成的新key,value列的名字
 >  **... :**  要gather的列名. （选择x到z的所有列 `x:z`, 除了y  `-y`）

- **spread 将key扩展成多列，value为要显示的值**  *(reshap2::dcast)*
`spread(data, key, value, fill = NA , drop = TRUE,sep = NULL)`
 > 说明：
 > 1. 若行和key组成的索引不唯一，报错
 > 2. sep: If NULL, 列名为变量key的值. If non-NULL, 列名为 `<key_name><sep><key_value>`

# Split or Unit Cells

```r
separate(data, col, into, sep, remove = TRUE)  #单列分割成多列
separate_rows(data, ..., sep = "[^[:alnum:].]+", convert = FALSE)  #当列分裂成多行
unite(data,col,sep =”_”,remove = TRUE)  #多列联合成单列

# 示例
table3 %>% 
  separate(rate, into = c("cases", "population"), convert = TRUE)
```

# Handle Missing Values

```r
replace_na(data, replace = list() )
drop_na(data)
fill(data, ..., .direction = c("down", "up")) 
```



