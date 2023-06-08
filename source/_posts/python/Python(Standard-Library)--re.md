---
title: Python手册(Standard Library)--re
tags:
  - Python
  - 正则表达式
categories:
  - Python
  - 'Standard Library'
cover: /img/regexp-demo.jpg
top_img: /img/python-top-img.svg
abbrlink: cb9fa048
date: 2018-05-09 23:15:58
description: re 模块使 Python 语言拥有全部的正则表达式功能。
---

re 模块使 Python 语言拥有全部的正则表达式功能。

<!-- more -->

# 匹配和查找

## re.match 和re.search

`re.match(pattern,string,flags=0)`  从起始位置匹配，且只匹配字符串的开始
`re.search(pattern,string,flags=0)` 扫描整个字符串并返回第一个成功的匹配  

- `re.match`和`re.search`匹配成功则返回 **`re.MatchObject`**对象，匹配不到则返回 `None`。可以通过调用 [`re.MatchObject`](#正则表达式对象) 的方法获取匹配到字符串
- `pattern` 匹配的正则表达式
- `string` 要匹配的字符串
- `flags` 表达式的匹配模式，如：是否区分大小写，多行匹配等等

```python
In [22]: line = "Cats are smarter than dogs"
In [23]: print(re.search( r'dogs', line).group())
'dogs'
In [24]: print(re.match( r'dogs', line))
None
```


**`(?P...)`分组匹配**：将匹配结果直接转为字典模式，方便使用

例如，身份证 `1102231990xxxxxxxx`

```python
In [3]: s = '1102231990xxxxxxxx'
In [4]: res = re.search('(?P<province>\d{3})(?P<city>\d{3})(?P<born_year>\d{4})',s)
In [5]: print(res.groupdict())
{'province': '110', 'city': '223', 'born_year': '1990'}
```

## re.findall 和 re.finditer

`re.findall(pattern,string,flags=0)`  在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果有多个匹配模式，则返回元组列表，如果没有找到匹配的，则返回空列表。
`re.finditer(pattern, string,  flags=0)` 类似 `findall`，返回迭代器  

> **注意：** match 和 search 是匹配一次 findall 匹配所有。

# 检索和替换

`re.sub(pattern,repl,string,count=0,flags=0)` 用于替换字符串中的匹配项

- `pattern` : 正则中的模式字符串。
- `repl` : 替换的字符串，也可为一个函数。
- `string` : 要被查找替换的原始字符串。
- `count` : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。

```python
In [27]: phone = "2004-959-559 # 这是一个国外电话号码"
In [28]: # 删除字符串中的 Python注释 
In [29]: re.sub(r'#.*$', "", phone)
Out[29]: '2004-959-559 '
```

**repl 参数是一个函数**：以下实例中将字符串中的匹配的数字乘以 2

```python
In [31]: s = 'A23G4HFD567'
In [33]: double = lambda x:str(int(x.group())*2)
In [34]: re.sub('\d',double,s)
Out[34]: 'A46G8HFD101214'
```

**Python局部替换**：有两种方式，如下例

```python
In [40]: s = 'NumberInt<5>,NumberInt<2>,NumberInt<0>'
In [41]: reg = r'(NumberInt)<(\d+)>'
```

- 使用 repl 参数的函数形式

  ```python
  In [42]: rre.sub(reg, lambda x:f'({x.group(2)})', s)
  Out[42]: '(5),(2),(0)'
  ```

- 使用 `\n` 获取分组的内容

  ```python
  In [43]: re.sub(reg, r'(\2)', s)
  Out[43]: '(5),(2),(0)'
  ```

# 分割字符串

`re.split(pattern,string,maxsplit=0,flags)` 按照能够匹配的子串将字符串分割后返回列表。
参数 `maxsplit` 限制分隔次数，默认为 0，不限制次数

```python
In [38]: re.split('\s', 'hello world')
Out[38]: ['hello', 'world']
```

# 正则表达式修饰符

正则表达式可以包含一些可选修饰符来控制匹配的模式。多个标志可以通过按位 OR(|) 来指定。如 `re.I | re.M` 被设置成 I 和 M 标志

| flags |说明|
| :------------- | :------------- |
| re.I | 使匹配对大小写不敏感 |
| re.L | 做本地语言识别（locale-aware）匹配|
| re.M | 多行匹配，影响 ^ 和 $ |
| re.S | 使.(dot)匹配所有字符，包括换行  |
| re.U | Unicode字符集解析  |
| re.X |该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解|

在正则表达式字符串中使用标志修饰符：

| 表达式            | 说明                        | 示例                      |
| ----------------- | --------------------------- | ------------------------- |
| `(?imx:pattern)`  | 在括号中使用 imx 可选标志   | `(?i:new)`可以匹配到`New` |
| `(?-imx:pattern)` | 在括号中不使用 imx 可选标志 |                           |

# 编译正则表达式

compile 函数用于编译正则表达式，生成一个正则表达式（ Pattern ）对象，供 `match()` 和 `search() `这两个函数使用。编译正则表达式，可以提高效率。语法如下

```python
re.compile(pattern,flags=0)
```

# 正则表达式对象

- **`re.RegexObject`**：编译过的正则表达式，`re.compile()` 函数返回
- **`re.MatchObject`**：正则表达式匹配到的对象。
  - `group([group1, ...])` 返回对应分组的**字符串**
  - `groups()` 返回一个包含所有小组字符串的元组
  - `groupdict()`返回分组的字典模式
  - `start(group=0)` 返回对应分组子字符串的起始位置
  - `end(group=0)` 返回对应分组子字符串的结束位置
  - `span(group=0)` 返回一个元组包含匹配 (start, end) 的位置
