---
title: Python手册(Standard Library)--Python 代码调试工具
date: 2021-11-13 22:23:11
categories: [Python,Python标准库]
tags: [Python]
cover: /img/python-debug.jpg
top_img:  # /img/python-logo-wide.svg
description: Python 代码调试工具快速入门
---

# assert

Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况。

```python
assert expression [, arguments]  # 参数可选
```

等价于：

```python
if not expression:
    raise AssertionError(arguments)   # 参数可选
```

以下实例判断当前系统是否为 Linux，如果不满足条件则直接触发异常，不必执行接下来的代码：

```python
>>> import sys
>>> assert ('linux' in sys.platform), "该代码只能在 Linux 下执行"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AssertionError: 该代码只能在 Linux 下执行

>>> # ... do something ...
```

# pdb

[`pdb`](https://docs.python.org/zh-cn/3/library/pdb.html#module-pdb) (**P**ython **D**e**B**ugger) 模块定义了一个交互式源代码调试器，用于 Python 程序。它支持在源码行间设置（有条件的）断点和单步执行，检视堆栈帧，列出源码列表，以及在任何堆栈帧的上下文中运行任意 Python 代码。它还支持事后调试，可以在程序控制下调用。

以猜骰子游戏为例，源码如下：

```python
# dicegame.py
import random

def GameRunner(guess_num):
  rand_num = random.randint(1, 6) 
  win = rand_num == int(guess_num)
  if win:
    print("Congratulations !")
  else:
    print("Sorry that's wrong")
    print(f"The answer is: {rand_num}")
  return win

print("Welcome\nAdd the values of the dice\n"
      "It's really that easy" )
n = wins = 0
while True:
  n += 1
  print(f"Round {n}")
  guess_num = input("What is your guess?: ")
  win = GameRunner(guess_num)
  wins += win 
  print(f"Wins: {wins} of {n}")
  play = input("Would you like to play again?[Y/n]: ")
  if play == 'n':
    break 
```

## 用法

- **侵入式方法**：需要在代码中插入 `pdb.set_trace()` 设置断点，当执行到断点时会停下来，进入交互式调试模式（内置函数 `breakpoint()`，当以默认参数调用它时，可以用来代替 `import pdb; pdb.set_trace()` ）
   例如：
   
   ```python
   # dicegame.py
   import random
   
   import pdb; pdb.set_trace() # add pdb here
   def GameRunner(guess_num):
     rand_num = random.randint(1, 6) 
     win = rand_num == int(guess_num)
     ... ...
     return win
   # do something ...
   ```
   
   让我们运行脚本 `python dicegame.py`
   
   ```python
   $ python -m pdb dicegame.py
   > c:\users\admin\documents\project\dicegame.py(2)<module>()
   -> import random
   (Pdb)
   ```
   
   当你在命令行看到提示符 `(Pdb)` 时，说明已经正确打开了pdb 。然后就可以输入 pdb 命令调试了。
   其中`>` 是文件标识； `dicegame.py(2)` 括号中的数字代表当前行数；`->` 是下一步要执行的代码标识。

- **非侵入式方法**：不用额外修改源代码，可以将 pdb.py 作为脚本在命令行调用，来调试其他脚本。调试器将暂停在待执行脚本第一行前。例如

```shell
$ python -m pdb dicegame.py
> c:\users\admin\documents\project\dicegame.py(2)<module>()
-> import random
(Pdb) 
```

- **事后调试模式**：调用代码抛出错误后，可以使用 `pdb.pm()` 捕获异常并进入其事后调试模式

```python
import pdb;pdb.pm()
```

例如：

```python
>>> import dicegame
Welcome
Add the values of the dice
It's really that easy
Round 1
What is your guess?: a
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\Admin\Documents\Project\dicegame.py", line 21, in <module>
    win = GameRunner(guess_num)
  File "C:\Users\Admin\Documents\Project\dicegame.py", line 6, in GameRunner
    win = rand_num == int(guess_num)
ValueError: invalid literal for int() with base 10: 'a'

>>> pdb.pm()
> c:\users\admin\documents\project\dicegame.py(6)GameRunner()
-> win = rand_num == int(guess_num)
(Pdb)
```

## pdb 命令

```shell
$ python -m pdb dicegame.py
> c:\users\admin\documents\project\dicegame.py(2)<module>()
-> import random
(Pdb)
```

进入 pdb 交互式界面后，就可以输入pdb命令了，下面是常用命令。大多数命令可以缩写为一个或两个字母。命令的参数必须用空格（空格符或制表符）分隔。

- `h(help) [command]` - 不带参数显示可用的命令列表，带参数则显示有关该命令的帮助文档 

- `l(list)  [first[, last]]`  - 列出当前文件的源代码。当前行用 `->` 标记，异常行用`>>` 标记
  - 如果不带参数，则列出当前行周围的 11 行
  - 如果只带参数 `first`，则列出那一行周围的 11 行
  - 如果带有两个参数，则列出所给范围中的代码 
  
  ```python
  (Pdb) l
    9       else:
   10         print("Sorry that's wrong")
   11         print(f"The answer is: {rand_num}")
   12       return win
   13
   14  -> print("Welcome\nAdd the values of the dice\n"
   15           "It's really that easy" )
   16     n = wins = 0
   17     while True:
   18       n += 1
   19       print(f"Round {n}")
  ```
  
- `ll(longlist)` - 列出所有源代码

- `w(where) ` -  打印目前所在的行号位置以及上下文信息

- `j(jump) lineno`  - 跳转到指定行（注意，被跳过的代码不执行）

- `n(next)` - 继续执行，直接运行函数 

- `s(step)` - 继续执行，可进入函数内部

- `c(continue)` - 继续运行，直到遇到下一个断点 

- `unt(until) [lineno]` - 继续运行，直到超过指定行（或遇到断点）

- `r(return)` - 继续运行，直接执行到函数返回处 

- `a(args)` - 进入函数内部后，打印函数的所有参数

- `u(up) [count]` - 将当前帧向上移动count层，默认为1 

- `d(down) [count]` - 将当前帧向下移动count层，默认为1 

- `b(break) [filename:]lineno [,condition]` - 在指定文件（默认当前文件）指定行设置断点。condition 判断为True时断点才设置断点

- `b(break) function [,condition]` - 指定函数第一行设置断点

- `tbreak ` - 临时断点，执行一次自动删除。用法与 break 相同。

- `p(print) expression` - 打印表达式的值 

  ```python
  (Pdb) n
  What is your guess?: 5
  > c:\users\admin\documents\project\dicegame.py(21)<module>()
  -> win = GameRunner(guess_num)
  (Pdb) p guess_num
  '5'
  ```

- `pp(prettyprint) expression` - 格式化打印表达式的值 

- `whatis expression` - 打印 expression 的类型

- `cl(clear) [filename:lineno | bpnumber ...]` - 清除断点
  - 如果参数是 `filename:lineno`，则清除此行上的所断点
  - 如果参数是空格分隔的断点编号列表，则清除这些断点
  - 如果不带参数，则清除所有断点（但会先提确认）
  
- `disable [bpnumber ...]` - 禁用断点（并不会删除）

- `enable [bpnumber ...]` - 启用断点

- `run [args ...]`

- `restart [args ...]` - 重新启动调试器，断点等信息都会保留。restart 是 run 的一个别名

- `q(quit)` - 退出调试模式，被执行的程序将被中止

- `debug code` - 进入一个对代码参数执行步进的递归调试器（该参数是在当前环境中执行的任意表达式或语句）

- `! statement` - 执行 (单行) Python 语句，代码中变量名与调试命令不冲突时，`!` 可省略

- `Enter` - 重复最后输入的命令

# ipdb

Python 提供了一个默认的 debug 工具 pdb，而 ipdb 是一款集成了 Ipython 的 Python 代码命令行调试工具，可以看做 pdb 的升级版，提供了补全、语法高亮等功能。

ipdb不是python内置的，需要手动安装

```shell
pip install ipdb
```

ipdb 的用法和 pdb 类似

```python
import ipdb
ipdb.set_trace()
ipdb.set_trace(context=5)  # will show five lines of code
                           # instead of the default three lines
                           # or you can set it via IPDB_CONTEXT_SIZE env variable
                           # or setup.cfg file
ipdb.pm()
```

命令行调用

```shell
$ bin/ipdb mymodule.py
$ bin/ipdb3 mymodule.py
$ python -m ipdb mymodule.py
```


