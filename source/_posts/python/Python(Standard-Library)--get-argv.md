---
title: Python(Python Basics)--命令行参数
categories:
  - Python
  - 'Standard Library'
tags:
  - Python
cover: /img/python-argparse-cover.png
description: 轻松编写更友好的命令行参数接口，包括 sys.argv, getopt, argparse
abbrlink: 97d915c4
date: 2021-11-10 21:52:50
top_img: /img/python-top-img.svg
---

# 命令行参数

Python 命令行可以使用 `-h` 参数查看帮助信息

```python
$ python -h 
usage: python [option] ... [-c cmd | -m mod | file | -] [arg] ... 
Options and arguments (and corresponding environment variables): 
-c cmd : program passed in as string (terminates option list) 
-d     : debug output from parser (also PYTHONDEBUG=x) 
-E     : ignore environment variables (such as PYTHONPATH) 
-h     : print this help message and exit
... ... 
```

命令行参数包含位置参数与选项参数，位置参数不带横杠，而选项参数带横杠。如果是短选项参数加前缀 `-`，长选项参数加前缀 `--`。
位置参数：按照位置直接赋值即可，如 `python script.py a b c`。 
选项参数：可以在终端赋值，也可以不赋值。 赋值时需要按照格式加参数名，如 `python script.py --foo a -f b c`。

# sys 模块

Python 将命令行参数以列表形式存储于 `sys.argv` 。例如demo.py 文件代码如下：

```python
import sys
print(sys.argv)
```

在命令行中执行以上代码，输出结果为：

```shell
$ python demo.py one two three
['demo.py', 'one', 'two', 'three']
```

其中 `argv[0] `为脚本的名称（是否是完整的路径名取决于操作系统）。

# getopt 模块

此模块专门协助脚本解析 `sys.argv` 中的命令行参数。支持短选项模式 `-` 和长选项模式 `--`，使得程序的参数更加灵活。该模块提供了两个方法及一个异常处理来解析命令行参数。

```python
getopt.getopt(args, shortopts, longopts=[])
```

该方法用于解析命令行选项与形参列表。

- **args**: 要解析的命令行参数列表，不包含脚本的名称 argv[0] 。
- **shortopts** : 由短选项 (`-`)字母名称组成的字符串，若字母有后缀 `:` 表示该选项，必须附加参数。
- **longopts** : 由长选项 (`--`)名称组成的列表，若名称后加 `=` 表示该选项必须有附加的参数，不带等号表示该选项不附加参数。命令行中的长选项只要提供了恰好能匹配可接受选项之一的选项名称前缀即可被识别。 举例来说，如果 longopts 为 ['foo', 'frob']，则选项 --fo 将匹配为 --foo，但 --f 将不能得到唯一匹配，因此将引发 GetoptError。

返回值由两个元素组成：第一个是 `(option, value)` 对的列表；第二个是在去除该选项列表后余下参数列表。列表中选项的排列顺序与它们被解析的顺序相同，因此允许多次出现。 长选项与短选项可以混用。

```python
getopt.gnu_getopt(args, shortopts, longopts=[])
```

此方法与 `getopt()`类似，区别在于它默认使用 GNU 风格的扫描模式。 这意味着选项和非选项参数可能会混在一起。`getopt()`函数将在遇到非选项参数时立即停止处理选项。

```python
Exception getopt.GetoptError
```

当参数列表中出现不可识别的选项或者当一个需要参数的选项未带参数时将引发此异常。 此异常的参数是一个指明错误原因的字符串。属性 msg 和 opt 为相关选项的错误信息。

一个仅使用 Unix 风格选项的例子:

```python
>>> import getopt
>>> args = '-a -b -cfoo -d bar a1 a2'.split()
>>> args
['-a', '-b', '-cfoo', '-d', 'bar', 'a1', 'a2']
>>> optlist, args = getopt.getopt(args, 'abc:d:')
>>> optlist
[('-a', ''), ('-b', ''), ('-c', 'foo'), ('-d', 'bar')]
>>> args
['a1', 'a2']
```

在脚本中，典型的命令行参数用法类似这样:

```python
import getopt, sys

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["help", "input=", "output="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        print(sys.argv[0] + ' -i <input> -o <output>')
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            print('test.py -i <input> -o <output>')
            sys.exit()
        elif o in ("-i", "--input"):
            input = a 
        elif o in ("-o", "--output"):
            output = a
        else:
            assert False, "unhandled option"
    # ...

if __name__ == "__main__":
    main()
```

# argparse模块

## 概要

[argparse](https://docs.python.org/zh-cn/3/library/argparse.html) 模块可以轻松编写更友好的命令行参数接口，并从`sys.argv`解析出那些参数。 该模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。 

argparse 模块解析命令行参数通常有以下几步：

1. 创建解析器： `parser = argparse.ArgumentParser()`，该 ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
2. 添加参数：通过调用 `parser.add_argument()` 方法添加参数信息。
3. 解析参数：通过调用 `args = parser.parse_args()` 方法创建储存参数的Namespace对象。通常该方法会被不带参数调用，自动从 `sys.argv` 中确定命令行参数。最终通过Namespace对象的属性使用参数。

```python
# filename: script.py
import argparse
parser = argparse.ArgumentParser(description = 'hello world !')
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
args = parser.parse_args()
# ... do something ...
```

运行程序时，使用 `-h` 或 `--help` 的方式获得帮助信息。包括自己在 `add_argument()` 的help中添加的内容。

```shell
$ python script.py --help
usage: script.py [-h] [-i INPUT] [-o OUTPUT]

hello world !

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
  -o OUTPUT, --output OUTPUT
```

## 创建解析器

```python
parser = argparse.ArgumentParser()
```

所有的参数都应当作为关键字参数传入，常用参数列表：

- prog - 程序的名称，默认值`sys.argv[0]`。可以在帮助消息里可以通过 `%(prog)s` 格式来引用。
- description - 在参数帮助文档之前显示的文本，简要描述这个程序的用途。
- add_help - 为解析器添加一个 `-h/--help` 选项，默认值 True

## 解析参数

```python
parser.parse_args(args, namespace)
```

将参数字符串转换为 Namespace 对象，并将参数设为该对象的属性。 

- args - 要解析的字符串列表。 默认值是从 `sys.argv` 获取。
  - 对于长选项（名称长度超过一个字符的选项），选项和值也可以作为单个命令行参数传入，使用 `=` 分隔它们即可。如 `--foo=FOO`
  - 对于短选项（长度只有一个字符的选项），选项和它的值可以拼接在一起。如`-xX`
- namespace - 用于获取属性的对象。 默认值是一个新的空 Namespace 对象

## 添加参数

```python
parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
```

- name or flags - 位置参数或选项参数名，例如 `foo` 或 `-f, --foo`。选项参数会以 `-` 前缀识别，剩下的参数则会被假定为位置参数。

  ```python
  >>> parser.add_argument('bar')          # 位置参数
  >>> parser.add_argument('-f', '--foo')  # 可选参数
  ```

- const - 不从命令行中读取但被 action 和 nargs 需求的常数。

- action - 将命令行参数与动作相关联。

  - `'store'` - 默认，存储参数的值。
  - `'store_const'` - 存储 `const` 指定的常数。当命令行参数启用，但未被赋值时使用`const`的值。
  - `'store_true'` and `'store_false'` - 这些是 `'store_const'` 存储 `True` 和 `False` 特例。
  - `'append'` - 存储一个列表，并将每个参数值追加到列表中。此时允许多次使用选项参数。
  - `'append_const'` - 这存储一个列表，并将 `const` 命名参数指定的值追加到列表中。
  - 等等

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo', action='store_const', const=42)
  >>> parser.add_argument('--bar', action='store_true')
  >>> parser.add_argument('--baz', action='store_true')
  >>> parser.parse_args('--foo --bar'.split())
  Namespace(foo=42, bar=True, baz=False)
  
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo', action='append')
  >>> parser.parse_args('--foo 1 --foo 2'.split())
  Namespace(foo=['1', '2'])
  ```

- nargs - 应当关联的终端参数值的个数：

  - `N` : （一个整数）命令行中的 N 个参数值会被聚集到一个列表中
  - `?` : 从命令行中消耗 0 或 1 个参数值
  - `*` : 命令行中 0 或所有参数值会被聚集到一个列表中
  - `+` : 命令行中所有参数值会被聚集到一个列表中，并且至少提供1个值

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo', nargs=2)
  >>> parser.add_argument('bar', nargs='*')
  >>> parser.parse_args('a b c --foo 1 2'.split())
  Namespace(bar=['a', 'b', 'c'], foo=['1', '2'])
  ```

- default - 选项参数未在命令行出现时，采用的默认值。

- type - 输入参数被转换成的类型，默认 str。普通内置类型和函数可被用作类型转换器。

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--length', default='10', type=int)
  >>> parser.add_argument('--width', default=10.5, type=int)
  >>> parser.parse_args()
  Namespace(length=10, width=10.5)
  ```

- choices - 容器中是可用的参数值。

  ```python
  >>> parser = argparse.ArgumentParser(prog='game.py')
  >>> parser.add_argument('move', choices=['rock', 'paper', 'scissors'])
  >>> parser.parse_args(['rock'])
  Namespace(move='rock')
  >>> parser.parse_args(['fire'])
  usage: game.py [-h] {rock,paper,scissors}
  game.py: error: argument move: invalid choice: 'fire' (choose from 'rock',
  'paper', 'scissors')
  ```

- required - 选项参数是否必须 True/False。

- help - 参数的简单描述，一般是通过在命令行中使用 `-h` 或 `--help` 的方式获得帮助信息。

- metavar - 在使用帮助消息时的参数显示名称。`parse_args()` 返回对象的属性名称仍然会由 `dest` 值确定。
  当使用 `nargs` 值大于1时，提供一个元组给 `metavar` 即为每个参数值指定不同的显示信息。

  ```python
  >>> parser = argparse.ArgumentParser(prog='PROG')
  >>> parser.add_argument('-x', nargs=2)
  >>> parser.add_argument('--foo', nargs=2, metavar=('bar', 'baz'))
  >>> parser.print_help()
  usage: PROG [-h] [-x X X] [--foo bar baz]
  
  options:
   -h, --help     show this help message and exit
   -x X X
   --foo bar baz
  ```

- dest - 自定义 `parse_args()` 返回对象上的参数名。

  - 对于位置参数，`dest `通常会作为 `add_argument() `的第一个参数提供
  - 对于选项参数，长/短选项字符串会去掉开头的 `--` 或 `-`来生成 `dest` 的值
  - 参数内部的 `-` 字符都将被转换为 `_` 字符以确保字符串是有效的属性名称

  ```python
  >>> parser = argparse.ArgumentParser()
  >>> parser.add_argument('--foo', dest='bar')
  >>> parser.parse_args('--foo XXX'.split())
  Namespace(bar='XXX')
  ```

## 典型案例

以下代码是一个 Python 程序，它获取一个整数列表并计算总和或者最大值：

```python
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))
```

假设上面的 Python 代码保存在名为 `prog.py` 的文件中，它可以在命令行运行并提供有用的帮助信息：

```shell
$ python prog.py -h
usage: prog.py [-h] [--sum] N [N ...]

Process some integers.

positional arguments:
 N           an integer for the accumulator

options:
 -h, --help  show this help message and exit
 --sum       sum the integers (default: find the max)
```

当使用适当的参数运行时，它会输出命令行传入整数的总和或者最大值：

```shell
$ python prog.py 1 2 3 4
4

$ python prog.py 1 2 3 4 --sum
10
```

## 互斥参数组

```python
parser.add_mutually_exclusive_group(required=False)
```

创建一个互斥组。 `argparse`将会确保互斥组中只有一个参数在命令行中可用

```python
# filename: prog.py
import argparse

parser = argparse.ArgumentParser(description="calculate X to the power of Y")
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("x", type=int, help="the base")
parser.add_argument("y", type=int, help="the exponent")
args = parser.parse_args()
answer = args.x**args.y

if args.quiet:
    print(answer)
elif args.verbose:
    print("{} to the power {} equals {}".format(args.x, args.y, answer))
else:
    print("{}^{} == {}".format(args.x, args.y, answer))
```

注意 `[-v | -q]`，它的意思是说我们可以使用 `-v` 或 `-q`，但不能同时使用两者：

```shell
$ python3 prog.py --help
usage: prog.py [-h] [-v | -q] x y

calculate X to the power of Y

positional arguments:
  x              the base
  y              the exponent

options:
  -h, --help     show this help message and exit
  -v, --verbose
  -q, --quiet
  
$ python prog.py 2 3
2^3 == 8

$ python prog.py 2 3 --verbose
2 to the power 3 equals 8

$ python prog.py --quiet 2 3
8

$ python prog.py -q -v 2 3
usage: prog.py [-h] [-v | -q] x y
prog.py: error: argument -v/--verbose: not allowed with argument -q/--quiet
```

