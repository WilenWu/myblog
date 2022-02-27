---
title: Python手册(Standard Library)--re
tags:
  - Python
  - 正则表达式
categories:
  - Python
  - Python标准库
cover: /img/regexp-demo.jpg
top_img: /img/python-top-img.svg
abbrlink: cb9fa048
date: 2018-05-09 23:15:58
description: 正则表达式调用库
---

re 模块使 Python 语言拥有全部的正则表达式功能。

compile 函数根据一个模式字符串和可选的标志参数生成一个正则表达式对象。该对象拥有一系列方法用于正则表达式匹配和替换。

re 模块也提供了与这些方法功能完全一致的函数，这些函数使用一个模式字符串做为它们的第一个参数。

<!-- more -->

# re模块

## 匹配

返回re对象 `MatchObject`

```python 
re.match(pattern,string,flags)        # 起始位置匹配
re.search(pattern,string,flags)       # 扫描整个字符串并返回第一个成功的匹配  
```

我们可以通过 `re.compile(pattern,flags=0)` 编译pattern，提高效率。
`re.match`和`re.search`匹配成功则返回 `re.MatchObject`对象，匹配不到则返回 `None`

| `re.MatchObject方法`|说明|
| :------------- | :------------- |
| obj.group(num=0) | 返回num对应索引位置的元组`(num,str)`|
| obj.groups()  | 返回所有元组|
| obj.start()| 返回第一个字符在字符串中的起始位置  |
| obj.end()  | 返回第一个字符在字符串中的结束位置  |
| obj.span() | 返回匹配到的字符串位置元祖(start,end)  |

## 查找

```python
re.findall(pattern,string,flags=0)    # 返回匹配到所有子串list 
re.finditer(pattern, string,  flags=0)# 类似findall，返回迭代器  
```

## 检索、替换和分割

```python 
re.sub(pattern,replace,string,count=0)    # replace为字符或函数，count为替换的次数
re.split(pattern,string,maxsplit=0,flags) # 返回字符串列表，maxsplit为最大分割次数
```

 ## flags标志

| flags |说明|
| :------------- | :------------- |
| re.I | 使匹配对大小写不敏感 |
| re.L | 做本地语言识别（locale-aware）匹配|
| re.M | 多行匹配，影响 ^ 和 $ |
| re.S | 使.(dot)匹配所有字符，包括换行  |
| re.U | Unicode字符集解析  |
| re.X |该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解|

