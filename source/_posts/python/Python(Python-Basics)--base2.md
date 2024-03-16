---
title: Python(Python Basics)--Python基础（二）
categories:
  - Python
  - General
tags:
  - Python
cover: /img/python-base2-cover.png
top_img: /img/python-top-img.svg
abbrlink: ec28e3a3
date: 2018-05-20 15:36:50
---

Python 标准库非常庞大，所提供的组件涉及范围十分广泛，正如以下内容目录所显示的。这个库包含了多个内置模块 (以 C 编写)，Python 程序员必须依靠它们来实现系统级功能。

<!-- more -->

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

# 错误输出重定向和程序终止

sys 还有 stdin， stdout 和 stderr 属性，即使在 stdout 被重定向时，后者也可以用于显示警告和错误信息:
```python
>>> sys.stderr.write('Warning, log file not found starting a new one\n')
Warning, log file not found starting a new one
```
大多数脚本的直接终止都使用 `sys.exit()` 。实现方式是抛出一个 SystemExit 异常。

参数（可选）可以是表示退出状态的整数（默认为 0），也可以是其他类型的对象。

- 如果它是整数，则 shell 等将 0 视为**成功终止**，非零值视为**异常终止**。
- Unix 程序通常用 2 表示命令行语法错误，用 1 表示所有其他类型的错误。
- 其他类型的对象：传入 None 等同于传入 0，传入其他对象则将其打印至 stderr，且退出代码为 1。

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

# JOSN 对象

JSON(JavaScript Object Notation) 是一种轻量级的数据交换格式，易于人阅读和编写。

```python
import json
```

| 函数       | 描述                                     |
| :--------- | :--------------------------------------- |
| json.dumps | 将 Python 对象编码成 JSON 字符串         |
| json.loads | 将已编码的 JSON 字符串解码为 Python 对象 |

JSON 类型和 python 类型对照表

| JSON   | Python           |
| :----- | :--------------- |
| object | dict             |
| array  | list             |
| string | str              |
| number | int, long, float |
| true   | True             |
| false  | False            |
| null   | None             |
