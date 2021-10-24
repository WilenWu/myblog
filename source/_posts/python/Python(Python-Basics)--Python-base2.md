---
ID: 8abb5d61acdaebba851a62d7fb10954d
title: Python手册(Python Basics)--Python基础（二）
date: 2018-05-20 15:36:50
categories: [Python,Python基础]
tags: [Python]
cover:  /img/pycad.png
top_img: /img/python-logo-wide.svg
---

Python 标准库非常庞大，所提供的组件涉及范围十分广泛，正如以下内容目录所显示的。这个库包含了多个内置模块 (以 C 编写)，Python 程序员必须依靠它们来实现系统级功能。

<!-- more -->

# 字符串

`s='Hello World'`

## 字符串方法

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


## 格式化字符串

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

## 字符串常量

| 标准库   | `import string`     |
| ------------ | ---------- |
| string.digits      | 数字0~9 |
| string.letters     | 所有字母（大小写）    |
| string.lowercase   | 所有小写字母|
| string.printable   | 可打印字符的字符串    |
| string.punctuation | 所有标点  |
| string.uppercase   | 所有大写字母|


# file 对象

> 参考链接：http://www.runoob.com/python/file-methods.html

```python
f=open(file, mode='r', encoding=None)  # 返回file对象
f.close()                              # 关闭
```
mode参数:

- `r` 只读
- `r+` 读写
- `w` 写入
- `w+` 读写
- `a` 追加
- `a+` 读写追加

> 默认为文本模式，如果要以二进制模式打开，加上 `b` 。
> 注意：使用 open() 方法一定要保证关闭文件对象，即调用 close() 方法。

若配合 `with` 使用，读取完可自动关闭
```python
with open("/tmp/python.csv") as f:
    data = f.read()
```

属性||
:---|:---
f.closed|返回true如果文件已被关闭，否则返回false。
f.mode|返回被打开文件的访问模式。
f.name|返回文件的名称。
f.softspace|如果用print输出后，必须跟一个空格符，则返回false。否则返回true。

方法||
:---|:---
f.close()|关闭文件
f.next()|返回文件下一行，迭代时使用
f.read([size])|从文件中读取的字节数
f.readline([size])|读取整行，包括 "\n" 字符
f.readlines([sizeint])|读取所有行并返回列表
f.seek(offset[, whence])|设置文件当前位置
f.tell()|返回文件当前位置
f.write(str)|将字符串写入文件，返回的是写入的字符长度。	
f.writelines(sequence)|向文件写入一个序列字符串列表，如果需要换行则要自己加入每行的换行符。

# os 模块

提供了很多与操作系统交互的函数

```python
>>> import os
>>> os.getcwd()      # Return the current working directory
'C:\\Python35'
>>> os.chdir('/server/accesslogs')   # Change current working directory
>>> os.system('mkdir today')   # Run the command mkdir in the system shell

>>> path=os.path.abspath(__file__) #返回脚本运行绝对路径
>>> dirname,filename=os.path.split(path) #分离出路径和脚本名
```

# 命令行参数

通用工具脚本经常调用命令行参数。这些命令行参数以链表形式存储于 sys 模块的 argv 变量。例如在命令行中执行 `python demo.py one two three` 后可以得到以下输出结果:
```python
>>> import sys
>>> print(sys.argv)
['demo.py', 'one', 'two', 'three']
```

# 错误输出重定向和程序终止

sys 还有 stdin， stdout 和 stderr 属性，即使在 stdout 被重定向时，后者也可以用于显示警告和错误信息:
```python
>>> sys.stderr.write('Warning, log file not found starting a new one\n')
Warning, log file not found starting a new one
```
大多数脚本的直接终止都使用 `sys.exit()`

# 数据压缩

以下模块直接支持通用的数据打包和压缩格式：zlib， gzip， bz2， lzma， zipfile 以及 tarfile。
```python
>>> import zlib
>>> s = b'witch which has which witches wrist watch'
>>> len(s)
41
>>> t = zlib.compress(s)
>>> len(t)
37
>>> zlib.decompress(t)
b'witch which has which witches wrist watch'
>>> zlib.crc32(s)
226805979
```

# 性能度量

有些用户对了解解决同一问题的不同方法之间的性能差异很感兴趣。Python 提供了一个度量工具，为这些问题提供了直接答案。

例如，使用元组封装和拆封来交换元素看起来要比使用传统的方法要诱人的多。timeit 证明了后者更快一些:
```python
>>> from timeit import Timer
>>> Timer('t=a; a=b; b=t', 'a=1; b=2').timeit()
0.57535828626024577
>>> Timer('a,b = b,a', 'a=1; b=2').timeit()
0.54962537085770791
```
相对于 timeit 的细粒度，profile 和 pstats 模块提供了针对更大代码块的时间度量工具。

# 十进制浮点数算法

decimal 模块提供了一个 Decimal 数据类型用于浮点数计
Decimal 的结果总是保有结尾的 0，自动从两位精度延伸到4位。Decimal 重现了手工的数学运算，这就确保了二进制浮点数无法精确保有的数据精度。

高精度使 Decimal 可以执行二进制浮点数无法进行的模运算和等值测试:
```python
>>> Decimal('1.00') % Decimal('.10')
Decimal('0.00')
>>> 1.00 % 0.10
0.09999999999999995

>>> sum([Decimal('0.1')]*10) == Decimal('1.0')
True
>>> sum([0.1]*10) == 1.0
False
```
decimal 提供了必须的高精度算法:
```python
>>> getcontext().prec = 36
>>> Decimal(1) / Decimal(7)
Decimal('0.142857142857142857142857142857142857')
```
