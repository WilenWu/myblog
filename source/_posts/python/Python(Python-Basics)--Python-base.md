---
ID: f60ac84d9061b41dcd04b167b62173f5
title: Python手册(Python Basics)--Python基础（一）
tags: [python]
date: 2018-05-09 00:10:30
categories: [python,Python基础]
cover: /img/python-basic.jpeg
---

Python 是一种易于学习又功能强大的编程语言。它提供了高效的高层次的数据结构，还有简单有效的面向对象编程。Python 优雅的语法和动态类型，以及解释型语言的本质，使它成为在很多领域多数平台上写脚本和快速开发应用的理想语言。

<!-- more -->

**Tips**

- Python中多行语句书写需在行末添加反斜杠`\`
- 在`[]{}()`中的多行语句不需要使用反斜杠
- `?` 显示对象的信息 
  `??` 显示对象的源码 
  `help()` 显示对象的帮助信息


# 对象

> python 中万物皆对象
> `reset`关键字或`reset()`函数可清空所有对象

**访问对象的属性和方法**

`obj.some_method(args)`
`obj.attribute_name`

# 标量

| 类型 | 说明|示例|
| --- | :---|:---|
| None  | 空值  |常常作为函数的默认参数|
| str| 字符串，存有Unicode（UTF-8编码）字符串 ||
| bytes | 原生ASCII字节  ||
| float | 浮点数  |0.0, 10.3e-3|
| int| 整数|10<br/> -0x260,0x69(0x开头的为16进制数字)<br/>0o69(0o开头的为八进制)<br/>0b1101(0b开头的为二进制)|
| bool  | 布尔型   |True/False|
|complex|复数|3+2i|

str, bool, int和float也是函数，可以用来转换类型

| 字符串  | 说明  |
| :--- | :--- |
| 单引号  | word = '字符串'  |
| 双引号  | sentence = "这是一个句子"  |
| 三引号  | 字符串换行  |
| r'\n strings'  | `\`可以用来转义，前面加r(raw)则不发生转义 |
| u'中文字符'  | 中文常加u(unicode)前缀编译  |
| **文档起始中文编码说明** | `# -*- coding: utf-8-*- `  |
| `string[start:end:step]` | 切片（左闭右开区间）  |

![hello](https://img-blog.csdnimg.cn/20191021135651986.png)

# 运算符

| 成员运算符 | 身份运算符 | 逻辑运算符 | 比较运算符 |
| :--- | :--- | :--- | :--- |
| in         | is         | and        | `>,<`      |
| not in     | is not     | or         | `>=,<=`    |
|            |            | not        | `!= ，==`  |


```python
>>> 2<3 and 'a'<'b'
True
>>> 2<3<1   # 支持连用
False
```

| 按位运算符 | 描述（二进制编码运算）  | 实例<br>`a,b=60,13`   |
| :--- | :--- | :--- |
| &          | 按位与：参与运算的两个值,如果两个相应位都为1,则该位的结果为1,否则为0 | (a & b) 输出结果 12 ，二进制解释： 0000 1100                 |
| \|         | 按位或运算符：只要对应的二个二进位有一个为1时，结果位就为1。 | (a \| b) 输出结果 61 ，二进制解释： 0011 1101                |
| ^          | 按位异或运算符：当两对应的二进位相异时，结果为1              | (a ^ b) 输出结果 49 ，二进制解释： 0011 0001                 |
| ~          | 按位取反运算符：对数据的每个二进制位取反,即把1变为0,把0变为1。~x 类似于 -x-1 | (~a ) 输出结果 -61 ，二进制解释： 1100 0011， 在一个有符号二进制数的补码形式。 |
| <<         | 左移动运算符：运算数的各二进位全部左移若干位，由"<<"右边的数指定移动的位数，高位丢弃，低位补0。 | a << 2 输出结果 240 ，二进制解释： 1111 0000                 |
| \>\>       | 右移动运算符：把">>"左边的运算数的各二进位全部右移若干位，">>"右边的数指定移动的位数 | a >> 2 输出结果 15 ，二进制解释： 0000 1111                  |

| 数学运算符 | 说明  |
| :--- | :--- |
| `**`  | 乘方  |
| `%`  | 求余数  |
| `//`  | 整除  |
| `+`  | 合并str,tupple,list，数学加号 |
| `*`  | 重复str,tupple,list，数学乘号 |
| `- `  | 减号  |
| `/`  | 除号  |

| 赋值运算符 | 描述  | 实例  |
| :--- | :--- | :--- |
| `=`  | 简单的赋值运算符 | `c = a + b 将 a + b 的运算结果赋值为 c`<br>`a=b=c=1` 连续赋值 |
| `+=`  | 加法赋值运算符  | `c += a 等效于 c = c + a`  |
| `-=`  | 减法赋值运算符  | `c -= a 等效于 c = c - a`  |
| `*=`  | 乘法赋值运算符  | `c *= a 等效于 c = c * a`  |
| `/=`  | 除法赋值运算符  | `c /= a 等效于 c = c / a`  |
| `%=`  | 取模赋值运算符  | `c %= a 等效于 c = c % a`  |
| `**=`  | 幂赋值运算符  | `c **= a 等效于 c = c ** a`  |
| `//=`  | 取整除赋值运算符 | `c //= a 等效于 c = c // a`  |


# 数据类型

## tuple (元祖)

元组是一个固定长度，不可改变的Python序列对象

- **创建tuple**
```python
tup = (4, 5,6), (7, 8)
tup = tuple(obj)
tup=()
```

- **拆分元祖**
```python
tup = 4,5,(6,7)
a,b,(c,d) = tup  # 可迭代对象都可以这样拆分
a,b,c = 'abc'
```

- **星号的使用**：`*`代表剩余部分

```python
>>> values = 94, 85, 73, 46
>>> first, *middle, last = values
>>> middle
[85, 73]

# 当一些元素不用时，作为惯用写法用'_'命名
>>> person = ('Bob', 20, 50, (11, 20, 2000))
>>> name, *_, (*_, year) = person

# 也可赋值使用
>>> *a, = 1, 2  # *a = 1, 2 这样会语法报错
[1, 2]
```
注意上面用星号解压出的变量永远都是列表类型。

- **元祖数学运算**
```python
>>> tup1=(1,2,3)
>>> tup2=('a','b')
>>> tup1 + tup2
(1,2,3,'a','b')
>>> tup2 * 3
('a','b','a','b','a','b')
```

- **索引和切片**
`tup[index]`
`tup[start:stop:step]`

- **方法**  
`tup.count(value)`  统计元组中某元素的个数
`tup.index(value)` 元素第一次出现的索引位置

## list (列表)

- **创建list**
`a = [2, 3,7, None]` 
`a=list(obj)`
`a=[]`

```python
>>> list('abcd')  
['a','b','c','d'] 
```

- **列表数学运算**
同元组

- **索引和切片**
`a[index]`
`a[start:stop:step]`
`a[::-1] ` 倒叙切片

| 方法  | 说明  |
| --- | --- |
| `a.count(value)`  | 统计列表中某元素的个数  |
| `a.index(value)`  | 元素第一次出现的索引位置  |
| a.append(obj)  | 向列表尾部追加一个新元素，只占一个索引位  |
| a.extend(iterator)  | 向列表尾部追加多个元素，将迭代器中的每个元素都追加进来 |
| a.insert(index,obj)  | 指定位置插入元素  |
| a.pop(index)  | 删除并返回指定位置元素  |
| a.clear()  | 清空列表  |
| a.remove(obj)  | 寻找第一个值并除去  |
| a.copy()  | 列表浅复制  |
| a.reverse()  | 反转列表  |
| a.sort(reverse=False) | 将list中的元素升序排列  |
```python
>>> a=[1,2]
>>> a.append([3,4]) 
[1,2,[3,4]]
>>> a.append(['a','b']) 
[1,2,[3,4],'a','b']
```

## dict (字典)

字典的值可以是任意Python对象，而键通常是不可变的标量类型（整数、浮点型、字符串）或元组（元组中的对象必须是不可变的），这被称为“可哈希性”。

**创建dict**（哈希映射或关联数组）

```python
d={'key1':value1,'key2':value2, ...}  # 由key-value对创建
d=dict(obj)`
d=dict.fromkeys(seq, value=None) # 由key value序列创建
d=dict(zip(key_list,value_list))` # zip函数创建（key-value)元组迭代器
d=dict(d1,**d2) # 合并字典d1和d2
d={} # 空字典
```

```python
d['key'] # 访问dict元素
d['new_key']='new_value' # 添加新key-value对或更新value
'key' in d  # 是否包含键值'key' 

del d['key'] #  删除元素
del d # 删除字典
```

| 方法  | 说明  |
| --- | :--- |
| d.copy()  | 浅复制  |
| d.pop(key,default=None)  | 删除并返回对应value<br>`ret = dict.pop('a')`  |
| d.popitem()  | 随机返回并删除字典中的一对键和值(一般删除末尾对)  |
| d.clear()  | 删除所有元素  |
| d.update(dict2)  | 将dict2中元素融入dict  |
| d.keys()  | 返回keys迭代器  |
| d.values()  | 返回values迭代器  |
| d.items()  | 返回dict中所有的(key,val)元祖数组  |
| d.get(key,default=None)  | 如果key存在，返回value，否则返回default  |
| d.setdefault(key, default=None) | 如果key存在，返回value，则插入 key 及设置的默认值 default，并返回 default |

## set (集合)
集合是无序的不可重复的元素的集合。
```python
>>> {1,2,3} == {3,2,1}
True
```

**创建set**

```python 
a = {1, 2, 3, 4, 5} 
a = set(obj) 
a = set()  # 空集合，{} 只能用来创建空字典
```

| 方法  | 说明  | 替代语法 |
| --- | --- | --- |
| a.add(value)  | 将value添加到a  |  |
| a.remove(value)  | 寻找第一个值并除去，不存在会报错  |  |
| a.discard(value) | 删除集合中指定的元素，不会报错  |  |
| d.copy()  | 浅复制  |  |
| a.pop(index)  | 删除并返回指定位置元素  |  |
| a.clear()  | 清空a  |  |
| a.update(b)  | 将b中元素融入a  | `a\=b`  |
| a.union(b)  | a和b的不重复元素  | `a\b`  |
| a.intersection(b | 交集  | `a&b`  |
| a.difference(b)  | 差集（存在于a，不存在于b）  | `a-b`  |
| a.issubset(b)  | if a∈b return True  |  |
| a.issuperset(b)  | if b∈a return True  |  |
| a.isdisjoint(b)  | if $a∩b=\emptyset$(交集)  return True |  |


## 列表、集合和字典推导式 
```python
list_comp=[expr for val in collection if condition]  
set_comp = {expr for value in collection if condition}  
dict_comp = {key-expr :value-expr for value in collection if condition} 

# list推导式等价for循环
 list_comp = []
 for val in collection:  
   if condition: 
     list_comp.append(expr) 
```


## iterator(迭代器) and generator(生成器)

- 迭代器
` iterator = iter(obj)` 
`range(start=0,end,step=1)`  返回一个迭代器，它产生一个均匀分布的整数序列 `[start,end) by step`
序列对象（tuple, dict, ...）都可以作为迭代器使用
`next(iterator)`  （迭代函数）下一个
- 生成器
在 Python 中，使用了 yield 的函数被称为生成器（generator），生成器是一个返回迭代器的函数，只能用于迭代操作
- [Python itertools模块详解](https://www.cnblogs.com/fengshuihuan/p/7105545.html)


```python
# 排列，组合，笛卡儿积：
from itertools import combinations,permutations,product,combinations_with_replacement

# 自建迭代器
def squares(n=10):  
  print('Generating squares from1 to {0}'.format(n ** 2))  
  for i in range(1, n + 1): 
    yield i ** 2  #将return替换为yield即可返回迭代器
gen = squares() 
for x in gen: 
  print(x, end=' ') 
```

# if 语句

```python
if expression:
  statements 
elif expression: 
  statements
else:
  statements
```

if 语句变体：`value=[false-expr,true-expr][condition]`（利用list索引判断）
三元表达式：`value=true-expr if condition else false-expr`

# 循环语句


## for循环
```python
for value in collection:
  statements
```


## while循环
```python
while expression:
  statements
else:
  statements
```

## 关键字
`pass`: 占位
`break`: 跳出循环
`coutine`:  跳出本次循环

## python中Switch/Case实现


- `if...elif...else` 实现
- 字典映射
```python
def function_1(...):
    statements
... ...

functions = {'a': function_1,
             'b': function_2,
             'c': self.method_1, ...}

func = functions[value]
func()
```

- example
```python
switch={
        'a': 1，
        'b': 2,
        'c': 3,
       }
result = switch.get('key','default_value') # 找不到'key'，返回默认值
```

# 函数

## 内建函数

| **输入输出**  |  |  |
| :--- | :--- | --- |
| input(prompt)  | 输入 |  |
| print(values,sep=' ',end='\n') | 输出 |  |

| **数学函数**  |  |  |
| :--- | :--- | --- |
| round(x,n)  | 四舍五入  |  |
| abs(x)  |  |  |
| len(obj)  | 序列元素数  |  |
| max(obj)  |  |  |
| min(obj)  |  |  |
| pow()  |  |  |
| divmod(a,b)  | 返回元组(a // b, a % b)  |  |
| hex(x)  | 10进制转16进制，以字符表示（0x开头的格式，如0x10） |  |
| int(x, base=10)  | 用于将一个字符串或数字转换为整型，默认十进制  |  |
| oct(x)  | 将一个整数转换成8进制字符串  |  |
| bin(x)  | 返回一个整数 int 或者长整数 long int 的二进制表示  |  |
| complex(real, imag) | 返回复数  |  |
| hash(object)  | 获取哈希值  |  |

| **面向对象**  |  |  |
| :--- | :--- | --- |
| type(obj)  | 输出对象类型  |  |
| dir(object)  | 输出对象属性和方法  |  |
| isinstance(obj,class_or_tuple)  | 判断对象类型('int', 'str',' list'...)  |  |
| setattr(object, name, value)  | 设置对象属性  |  |
| getattr(object, name[, default]) | 返回一个对象属性值  |  |
| hasattr(object, name)  | 判断对象是否包含对应的属性  |  |
| delattr(object, name)  | 删除属性  |  |
| issubclass(class, classinfo)  | 判断是否子类  |  |
| vars([object])  | 函数返回对象object的属性和属性值的字典对象 |  |

| **字符表达式**  | 说明  |
| --- | --- |
| eval(expression) | 执行一个字符串表达式，并返回表达式的值<br>`eval('pow(2,2)')` |
| exec(object)  | 执行复杂的表达式，返回None  |

```python
x = 10
expr = """
z = 30
sum = x + y + z
print(sum)
"""
exec(expr)
```

| **内建函数**  | 说明  |
| --- | --- |
| chr()  | 把编码转换为对应的字符  |
| id([object])  | 获取对象内存地址  |
| iter(object[, sentinel]) | 生成迭代器  |
| all(iterable)  | 给定的可迭代对象全部为True，类似于and  |
| any(iterable)  | 给定的可迭代对象任一为True,类似于or  |
| reversed(seq)  | 反转序列，返回迭代器  |
| callable(object)  | 判断对象是否是可调用，对于函数, 方法, lambda 函式, 类, 以及实现了 `__call__ `方法的类实例, 它都返回 True |

| **高阶函数**  | 说明  |
| --- | :--- |
| map(function,obj)  | 对序列中所有元素函数映射  |
| reduce(function, iterable[, initializer]) | 对参数序列中元素进行累积<br>python3从内置模块移除`from functools import reduce` |
| filter(function, iterable)  | 返回一个迭代器  |

```python
strings = ['a','as','bat','car','dove','python']
set(map(len, strings))
reduce(lambda x, y: x+y, [1,2,3,4,5])
filter(lambda x: x%2==0, range(1, 101))
```

## 自定义函数

```python
def fun_name(var1,var2=default,*args,**kwargs): 
  '说明文档'
  statements  
  return result 
```

- 函数代码块以 def 关键词开头，后接函数标识符名称和圆括号 ()。
- 任何传入参数和自变量必须放在圆括号中间，圆括号之间可以用于定义参数。
- 函数的第一行语句可以选择性地使用文档字符串—用于存放函数说明。
- 函数内容以冒号起始，并且缩进。
- return [表达式] 结束函数，选择性地返回一个值给调用方。不带表达式的return相当于返回 None。

- **参数顺序**：必选参数，默认参数，可变参数， 关键字参数
  - `*args`是可变参数，以tuple的形式返回
  - `**kwargs`是关键字参数，以dict的形式返回
  - 可变参数(`*`)之后的参数必须指定参数名，否则就会被归到可变参数之中。
  - 关键字参数都只能作为最后一个参数。

```python
>>> def test(a,*args, b, **kwargs): 
 	   print(a)
 	   print(args)
       print(b)
 	   print(kwargs)
>>> test('a',1, 2, b='b',key1=2,key2='c')
'a'
(1,2)
'b'
{'key1':2,'key2':'c'}

>>> def myfun(*, a, b): # 可变参数也可不用命名
...   pass

>>> def fib(n):    # write Fibonacci series up to n
...     """Print a Fibonacci series up to n."""
...     a, b = 0, 1
...     while a < n:
...         print(a, end=' ')
...         a, b = b, a+b
...     print()
...
>>> # Now call the function we just defined:
>>> fib(2000)
0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597
```

- **解压序列作为参数传入函数中**

```python
>>> def myfun(a, b):
...   print(a + b)

# 解压列表元组
>>> n = [1, 2]
>>> myfun(*n)
3
>>> m = (1, 2)
>>> myfun(*m)
3

# 解压字典
>>> mydict = {'a':1, 'b': 2}
>>> myfun(**mydict)
3
>>> myfun(*mydict)
ab
```

### 函数注解

- 我们知道 Python 是一种动态语言，变量以及函数的参数是不区分类型。
Python解释器会在运行的时候动态判断变量和参数的类型，这样的好处是编写代码速度很快，很灵活，但是坏处也很明显，不好维护，可能代码写过一段时间重新看就很难理解了，因为那些变量、参数、函数返回值的类型，全都给忘记了。
- 所以Python3里有了这个新特性，可以给参数、函数返回值和变量的类型加上注解，不过这个仅仅是注释而已，对代码的运行来说没有任何影响，变量的真正类型还是会有Python解释器来确定，你所做的只是在提高代码的可读性，让 IDE 了解类型，从而提供更准确的代码提示、补全和语法检查，仅此而已。
- 注解是以字典形式存储在函数的 `__annotations__ `属性中，对函数的其它部分没有任何影响。

```python
>>> def add(x:int, y:int) -> int:
...     return x+y
>>> def f(ham: 42, eggs: int = 'spam') -> "Nothing to see here":
...     print("Annotations:", f.__annotations__)
...     print("Arguments:", ham, eggs)
...
>>> f('wonderful')
Annotations: {'eggs': <class 'int'>, 'return': 'Nothing to see here', 'ham': 42}
Arguments: wonderful spam
```

### 匿名（lambda）函数

`lambda vars:expr` 不用写return，返回值就是该表达式的结果 
` lambda x,y:x**y`


### 变量的作用域

Python的作用域一共有4种，分别是：

L （Local） 局部作用域
E （Enclosing） 闭包函数外的函数中
G （Global） 全局作用域
B （Built-in） 内建作用域
以 L –> E –> G –>B 的规则查找，即：在局部找不到，便会去局部外的局部找（例如闭包），再找不到就会去全局找，再者去内建中找。

```python
x = int(2.9)  # 内建作用域
 
g_count = 0  # 全局作用域
def outer():
    o_count = 1  # 闭包函数外的函数中
    def inner():
        i_count = 2  # 局部作用域
```

Python 中只有模块（module），类（class）以及函数（def、lambda）才会引入新的作用域，其它的代码块（如 if/elif/else/、try/except、for/while等）是不会引入新的作用域的，也就是说这些语句内定义的变量，外部也可以访问

### global 和 nonlocal关键字
当内部作用域想修改外部作用域的变量时，就要用到global和nonlocal关键字了

```python
num=1
def outer():
    num2 = 10
    def inner():
    	global num  # global关键字声明
        nonlocal num2   # nonlocal关键字声明
        print(num+num2)
```

## 序列函数

- **`enumerate`函数**，可以返回`(i, value)`元组序列，常用于loop：

```
for i, value in enumerate(collection): 
  #do something with value
```

- **sorted函数**
` sorted(list)` 返回排序好的list副本 
 `sorted(str)`  拆分str，返回排序好的list副本 

- **zip函数**
`zip(seq1,seq2)` 可以将多个列表、元组或其它序列成对组合成一个元组对
zip可以处理任意多的序列，元素的个数取决于最短的序列
`zip(*tup)`逆转用法

```python
In [89]: seq1 = ['foo', 'bar', 'baz'] 
In [90]: seq2 = ['one', 'two', 'three']
In [92]: list(zip(seq1, seq2)) 
Out[92]: [('foo', 'one'), ('bar', 'two'), ('baz', 'three')]

In [95]: for i, (a, b) in enumerate(zip(seq1, seq2)):
   ....:     print('{0}: {1}, {2}'.format(i, a, b))
   ....:
0: foo, one
1: bar, two
2: baz, three
```

- **排列组合**

`from itertools import product,permutations,combinations`

> 参考链接：https://blog.csdn.net/specter11235/article/details/71189486


# 闭包

> 参考链接：https://mp.weixin.qq.com/s/I9WIkm4ounSQSK5zpOv5eg

```python
def print_msg():
    # print_msg 是外围函数
    msg = "I'm python"
    def printer():
        # printer 是嵌套函数
        print(msg)
    return printer

another = print_msg()

# 输出 I'm python
another()
```

一般情况下，函数中的局部变量仅在函数的执行期间可用，一旦 print_msg() 执行过后，我们会认为 msg变量将不再可用。然而，在这里我们发现 print_msg 执行完之后，在调用 another 的时候 msg 变量的值正常输出了，这就是闭包的作用，闭包使得局部变量在函数外被访问成为可能。

看完这个例子，我们再来定义闭包，维基百科上的解释是:

> 在计算机科学中，闭包（Closure）是词法闭包（Lexical Closure）的简称，是引用了自由变量的函数。这个被引用的自由变量将和这个函数一同存在，即使已经离开了创造它的环境也不例外。所以，有另一种说法认为闭包是由函数和与其相关的引用环境组合而成的实体。

这里的 another 就是一个闭包，闭包本质上是一个函数，它由两部分组成，printer 函数和变量 msg。闭包使得这些变量的值始终保存在内存中。

**示例：**
闭包常用于解偏微分方程，如下例：求解陨石运动轨迹

运动半径：$r=\sqrt{x^2+y^2+z^2}$
单位向量：$\hat{r}=\cfrac{\vec{r}}{r}$    
加速度：$\cfrac{d\vec{v}}{dt}=-\cfrac{GM}{r^2}\cdot\hat{r}$   
速度：$\vec{v}=\cfrac{d\vec{r}}{dt}$
使用 `scipy` 包求解坐标 `(x,y,z)`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def gravity(GM=1):
    '''             
    dr/dt=v
    r_norm=sqrt(x^2+y^2+z^2)
    dv/dt=-GM/dot(r,r)*r/r_norm
    ''' 
    def force(t,X_and_V):
        'GM loading...'
        x,y,z,vx,vy,vz=X_and_V
        r=np.array([x,y,z])
        v=np.array([vx,vy,vz])
        dr=v
        r_norm=np.linalg.norm(r)
        dv=-GM/np.dot(r,r)*r/r_norm
        return np.concatenate([dr,dv])
    return force

r0=(20,30,0)
v0=(0.8,0.5,0)
t0,t1=0,100

F=gravity(GM=1000)
motion=solve_ivp(fun=F,t_span=(t0,t1),y0=r0+v0, t_eval=np.linspace(t0,t1,1001))

plt.plot(motion.y[0,:],motion.y[1,:])
plt.show()
```
![](https://img-blog.csdnimg.cn/20191021135843664.png)


# 装饰器
装饰器实际上是在不改变原程序的情况下，给某程序增添功能，避免大量雷同代码。
> 菜鸟教程（包括带参数的装饰器和装饰器类等）
> http://www.runoob.com/w3cnote/python-func-decorators.html
> 参考文档：http://lib.csdn.net/article/python/62942

```python
from functools import wraps
def decorator_name(f):
    @wraps(f)  #加载wraps是为了原函数的名字和注释文档不被重写(docstring)
    def decorated(*args, **kwargs):
    	print('I am do something')
        if not True:
            return "Function will not run"
        return f(*args, **kwargs)
    return decorated
 
@decorator_name
def func():
    return("Function is running")
```

# 错误和异常处理

```python
try: 
  statements 
except <Error(可选)>:  #try失败时执行，可以指定特定的Error或者Error元祖，例如(TypeError,ValueError)
  statements 
else:       #try成功时执行
  statements 
finally:    #始终执行
  statements 
```

# 模块

```python
 import some_module #导入整个模块 
 from some_module import func1,func2  #从某个模块中导入多个函数 
 from some_module import *  #将某个模块中的全部函数导入
 dir(some_module) #列出模块里的函数 
```
参考链接：
[python自定义包]( https://blog.csdn.net/lxy4239/article/details/79107008)
[python模块官方文档](http://www.pythondoc.com/pythontutorial3/modules.html)
[Python3 教程 | 菜鸟教程](http://www.runoob.com/python3/python3-module.html)