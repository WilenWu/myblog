---
title: Python手册(Standard Library)--string
date: 2021-11-13 22:23:11
categories: [Python,Python标准库]
tags: [Python]
cover: /img/sudoku.jpeg
top_img: '#66CCFF'
description: Python3 字符串方法和格式化
---


```python
>>> # demo
>>> s = 'Hello World
```

# 字符串方法

| 计数   |      |
| :---------------- | :----------------- |
| s.count(substr,beg=0,end=len(string)) | 返回substr出现的次数|
| **去空格**     ||
| s.lstrip(chars)  | 删除str左边的字符（或空格）     |
| s.rstrip(chars)  | 删除str右边的字符（或空格）     |
| s.strip(chars)   | 删除str两边的字符（或空格）     |
| **字符串补齐**   ||
| s.center(width,fillchar)| 返回str居中，宽度为width的字符串(fillchar为填充字符)      |
| s.ljust(width,fillchar) | str左对齐|
| s.rjust(width,fillchar) | str右对齐|
| s.zfill (width)  | str右对齐，前面填充0 |
| **大小写转换**   ||
| s.capitalize()   | str的第一个字符大写  |
| s.title() | 每个单词首字母大写    |
| s.lower() | 小写    |
| s.upper() | 大写    |
| s.swapcase()     | 大小写互换 |
| **字符串条件判断** ||
| s.isalnum()      | 所有字符都是字母或数字  |
| s.isalpha()      | 所有字符都是字母     |
| s.isdigit()      | 所有字符都是数字     |
| s.isnumeric()    | 只包含数字字符      |
| s.isspace()      | 只包含空白 |
| s.istitle()      | 字符串是标题化      |
| s.islower()      | 都是小写  |
| s.isupper()      | 都是大写  |
| s.startswith(substr)    | 以substr开头    |
| s.endswith(substr)      | 以substr结尾    |
| **字符串搜索定位与替换**     |    |
| s.find(substr)    | 返回substr的索引位置，如果找不到，返回-1     |
| s.rfind(str)     ||
| s.index(substr)  | 返回substr的索引位置，如果找不到，返回异常   |
| s.rindex(str)    ||
| s.replace(old,new,max)  | 字符串替换，不超过 max 次（默认为1）。     |
| **字符串分割变换** ||
| s.join(seq)      | 以str分隔符，合并seq中所有的元素 |
| s.split(sep="",num)     | 分割str，num=str.count(sep)默认 |
| s.splitlines(keepends)  | 按照行('\r','\r\n',\n')分隔，参数 keepends为False则不包含换行符 |
| **字符串编码与解码**||
| s.encode(encoding='UTF-8')     | 以 encoding 指定的编码格式编码字符串 |


# 格式化字符串

格式字符串由 `{}`包围的`replacement_field` 和任何不包含在大括号中的普通文本组成。由 `s.format()` 方法传递参数。

`replacement_field`简单组成： `{field_name:format_spec}`

- **field_name** : 是一个数字，表示位置参数（element_index），如果`field_name`依次为`0,1,2，...`，则它们可以全部省略，并且数字`0,1,2，...`将按照该顺序自动插入。
或者关键字（attribute_name）,`str.format()`可通过关键字传递参数。

- **format_spec**：`[width][.precision][type]`
  `width`：（数字）表示宽度
  `.precision`：（dot+数字）小数位数
  `type`：表示类型
   `s`：表示字符格式
   `d`：十进制整数
   `f`：固定精度
   `e`：科学记数法
   `n`：数字
   `%`：百分比显示

for example
```python
print('Life is short, {} need {}'.format('You','Python'))  # 忽略数字
print('Life is short, {0} need {1}'.format('You','Python'))  # 带数字编号
print('Life is short, {1} need {0}'.format('Python','You'))  # 打乱顺序

#上面代码统一输出为: 'Life is short, You need Python'
print('Life is short, {name} need {language}' \
 .format(name='You',language='Python Language'))  # 关键字

import math
print('r={r:.1e}'.format(r=10**5))
print('π={pi:.2f}'.format(pi=math.pi))
print('e/PI={percent:.2%}'.format(percent=math.e/math.pi))

# 解包作为参数
>>> bob = {'name': 'Bob', 'age': 30}
>>> "{name}'s age is {age}".format(**bob)
"Bob's age is 30"
```

# 字符串常量

| 标准库   | `import string`     |
| ------------ | ---------- |
| string.digits      | 数字0~9 |
| string.letters     | 所有字母（大小写）    |
| string.lowercase   | 所有小写字母|
| string.printable   | 可打印字符的字符串    |
| string.punctuation | 所有标点  |
| string.uppercase   | 所有大写字母|

