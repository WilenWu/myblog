---
title: Python手册(Standard Library)--string
categories:
  - Python
  - 'Standard Library'
tags:
  - Python
cover: /img/sudoku.jpeg
top_img: /img/python-top-img.svg
description: Python3 字符串方法和格式化输出
abbrlink: 64b738be
date: 2021-11-13 22:23:11
---

# 字符串方法

| 计数   |      |
| :---------------- | :----------------- |
| str.count(substr,start,end) | 返回substr出现的次数|
| **去空格**     ||
| str.lstrip(chars) | 删除str左边的chars字符（默认空格） |
| str.rstrip(chars) | 删除str右边的chars字符（默认空格） |
| str.strip(chars) | 删除str两边的chars字符（默认空格） |
| **字符串补齐**   ||
| str.center(width,fillchar) | 返回str居中，宽度为width的字符串(fillchar为填充字符)      |
| str.ljust(width,fillchar) | 左对齐 |
| str.rjust(width,fillchar) | 右对齐 |
| str.zfill (width) | 右对齐，前面填充0 |
| **大小写转换**   ||
| str.capitalize() | 第一个字符大写，其余为小写 |
| str.title() | 每个单词首字母大写    |
| str.lower() | 小写    |
| str.upper() | 大写    |
| str.swapcase()   | 大小写互换 |
| **字符串条件判断** |至少有一个字符|
| str.isalnum()    | 所有字符都是字母或数字 |
| str.isalpha()    | 所有字符都是字母 |
| str.isdigit()    | 所有字符都是数字 |
| str.isnumeric()  | 只包含数字字符      |
| str.isspace()    | 只包含空白字符 |
| str.istitle()    | 字符串是标题化，即单词首字母大写 |
| str.islower()    | 都是小写  |
| str.isupper()    | 都是大写  |
| str.startswith(substr)  | 以substr开头    |
| str.endswith(substr)    | 以substr结尾    |
| **字符串搜索定位与替换**     |    |
| str.find(substr,start,end) | 返回substr的索引位置，如果找不到，返回-1     |
| str.rfind(str)   |返回substr的最右索引位置，如果找不到，返回-1|
| str.index(substr) | 返回substr的索引位置，如果找不到，返回 ValueError |
| str.rindex(str)  |返回substr的最右索引位置，如果找不到，返回 ValueError|
| str.replace(old,new,count) | 字符串替换，不超过 count 次（默认为1）。 |
| **字符串分割联合** ||
| str.join(iterable) | 以str分隔符，合并iterable中所有的元素 |
| str.split(sep=None,maxsplit=- 1) | 分割字符串 |
| str.splitlines(keepends) | 按照行('\r','\r\n',\n')分隔，参数 keepends为False则不包含换行符 |
| str.partition(sep) | 在 sep 首次出现的位置拆分字符串，返回一个 3 元素元组 |
| **字符串编码与解码**||
| str.encode(encoding='UTF-8')   | 以 encoding 指定的编码格式编码字符串 |

# 格式化字符串

##  %-formatting

这是旧式字符串格式化方法，相关信息可以阅读[官方文档](https://docs.python.org/zh-cn/3/library/stdtypes.html#printf-style-bytes-formatting)。值得注意的是，官方文档其实并不推荐使用这种方式。

```python
'string' % values
```

其中 `string` 为一个字节串对象，字符串对象内有操作符 `%` 占位，可以用于格式化操作，具体用法如下：

```python
>>> import math
>>> print('The value of pi is approximately %5.3f.' % math.pi)
The value of pi is approximately 3.142.
```

如果要在字符串中嵌入多个变量，则必须使用元组，例如：

```python
>>> name = "Eric"
>>> age = 74
>>> "Hello, %s. You are %s." % (name, age)
'Hello Eric. You are 74.'
```

也可以使用关键字传递

```python
>>> b'%(language)s has %(number)03d quote types.' %
      {b'language': b"Python", b"number": 2}
b'Python has 002 quote types.'
```

## str.format

这种方式是在 Python 2.6 引入的，可以在 [官方文档](https://docs.python.org/zh-cn/3/library/stdtypes.html#printf-style-bytes-formatting) 找到相关介绍。

该方法该方法用 `{` 和 `}` 标记替换变量的位置，由格式化字符串的 `str.format()` 方法传递参数。该语法在大多数情况下与旧式的 `%` 格式化类似，只是增加了 `{}` 和 `:` 来取代 `%`。 

替换文本的简单组成： 

```python
[field_name][!conversion][:format_spec]
```

**field_name** : 被替换的字符

- 可以是一个数字，表示位置参数。如果`field_name` 依次为`0,1,2，...`，则它们可以全部省略。
- 或者是命名关键字，`str.format()`可通过关键字传递参数。

```python
# 按位置访问参数
>>> 'I am {}, I love {}'.format('Bill','Python')      # 忽略数字
>>> 'I am {0}, I love {1}'.format('Bill','Python')     # 带数字编号
>>> 'I am {1}, I love {0}'.format('Python','Bill')     # 打乱顺序
'I am Bill, I love Python'

# 按名称访问参数
>>>'I am {name}, I love {lang}'.format(name='Bill',lang='Python')  

# 在引用字典时，可以用 ** 操作符进行字典拆包
>>> bob = {'name': 'Bob', 'age': 30}
>>> "{name}'s age is {age}".format(**bob)
"Bob's age is 30"

# 访问参数的属性
>>> c = 3-5j
>>> ('The complex number {0} is formed from the real part {0.real} '
...  'and the imaginary part {0.imag}.').format(c)
'The complex number (3-5j) is formed from the real part 3.0 and the imaginary part -5.0.'
>>> class Point:
...     def __init__(self, x, y):
...         self.x, self.y = x, y
...     def __str__(self):
...         return 'Point({self.x}, {self.y})'.format(self=self)
...
>>> str(Point(4, 2))
'Point(4, 2)'

# 访问参数的项
>>> coord = (3, 5)
>>> 'X: {0[0]};  Y: {0[1]}'.format(coord)
'X: 3;  Y: 5'
```

**conversion** :  (`"r" | "s" | "a"`) 在格式化之前进行类型强制转换。 通常，格式化值的工作由值本身的 `__format__()` 方法来完成。 但是，在某些情况下最好强制将类型格式化为一个字符串，覆盖其本身的格式化定义。 

目前支持的转换旗标有三种: `'!s'` 会对值调用 [`str()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#str)，`'!r'` 调用 [`repr()`](https://docs.python.org/zh-cn/3/library/functions.html#repr) 而 `'!a'` 则调用 [`ascii()`](https://docs.python.org/zh-cn/3/library/functions.html#ascii)。

**format_spec**：`[fill align][sign][#][0][width][grouping_option][.precision][type]`
称为格式规格迷你语言，在格式字符串所包含的替换字段内部使用，用于定义单个值应如何呈现，包含字段宽度、对齐、填充、小数精度等细节信息。

## f-strings 

Python 3.6 引入了新的字符串格式化方式 f-strings，与其它格式化方式相比，不仅简洁明了，可读性更好，更不容易出错，而且运行效率也更高。

f-strings 也称作==格式化的字符串字面量== (formatted string literals)，它是一个带有 `f`或`F` 前缀的字符串。通过 `{}` 间的表达式，把 Python 表达式的值添加到字符串内。

```python
{f_expression[=][!conversion][:format_spec]}
```

其中表达式 `f_expression` 是替换并填入字符串的内容，可以是变量、表达式或函数等，这些表达式的具体值是在运行时确定的，背后依赖的也是嵌入对象的 `__format()__` 接口。查看 [官方文档](https://docs.python.org/zh-cn/3/reference/lexical_analysis.html#formatted-string-literals) 可以获得更多信息。格式说明符是可选的，写在表达式后面，可以更好地控制格式化值的方式。

**基本使用**：直接填入变量名

```python
>>> year = 2016
>>> event = 'Referendum'
>>> f'Results of the {year} {event}'
'Results of the 2016 Referendum'
```

**支持任意表达式**：可以在字符串中嵌入任意有效的 Python 表达式，从而写出更优雅的代码

```python
>>> f"{2 * 37}"
'74'
```

也可以在里面调用函数：

```python
>>> def to_lowercase(input):
...     return input.lower() 
>>> name = "Eric Idle"
>>> f"{to_lowercase(name)} is funny."
'eric idle is funny.'
```

或者直接调用对象的方法：

```python
>>> f"{name.lower()} is funny."
'eric idle is funny.'
```

甚至可以在对象的字符串方法中直接使用 f-strings，例如有以下类：

```python
class Comedian:
    def __init__(self, first_name, last_name, age):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age

     def __str__(self):
        return f"{self.first_name} {self.last_name} is {self.age}."

     def __repr__(self):
        return f"{self.first_name} {self.last_name} is {self.age}. Surprise!"
```

你可以有如下代码：

```python
>>> new_comedian = Comedian("Eric", "Idle", "74")
>>> f"{new_comedian}"
'Eric Idle is 74.'
```

`__str__()`方法与`__repr__()`方法用于处理对象的字符串显示方式，我们有必要至少定义其中一个。如果必须二选一的话，建议使用`__repr__()`，在`__str__()`方法没有定义的情况下，解释器会自动调用`__repr__()`方法。

`__str__()`方法返回的是对象的非正式字符串表示，主要考虑可读性，而`__repr__()`方法返回的是对象的正式字符串表示，主要考虑精确性。调用这两个函数时，比较推荐的方式是直接使用内置函数`str()`和`repr()`。

f-strings 会默认调用对象的`__str__()`方法，如果要强制使用`__repr__()`方法，则可以在变量之后加上转换标志`!r`：

```python
>>> f"{new_comedian}"'Eric Idle is 74.'
>>> f"{new_comedian!r}"
'Eric Idle is 74. Surprise!'
```

**使用lambda匿名函数**：可以做复杂的数值计算

f-string大括号内也可填入lambda表达式，但lambda表达式的 `:` 会被f-string误认为是表达式与格式描述符之间的分隔符，为避免歧义，需要将lambda表达式置于括号 `()` 内：

```python
>>> f'result is {(lambda x: x ** 2 + 1) (2)}'
'result is 5'
```

**多行字符串中使用 f-Strings**

要注意的是，在每一行字符串之前，都要加上`f`前缀。

```python
>>> name = "Eric"
>>> profession = "comedian"
>>> affiliation = "Monty Python"
>>> message = (
...     f"Hi {name}. "
...     f"You are a {profession}. "
...     f"You were in {affiliation}."
... )
>>> message
'Hi Eric. You are a comedian. You were in Monty Python.'
```

**关于大括号**：如果想在表达式中使用大括号，我们必须输入连续两个大括号：

```python
>>> f"{{74}}"
'{74}'
```

**关于反斜线符号**：大括号外的引号还可以使用 `\` 转义，但大括号内不能使用 `\` 转义。有需要时，可以提前定义一个变量来绕过这种限制

```python
>>> name = "Eric Idle"
>>> f"{name}"
'Eric Idle'
```

**关于行内注释**：f-strings 中不应包括带 `#` 号的注释，否则会导致句法错误：

```python
>>> f"Eric is {2 * 37 #Oh my!}."
  File "<stdin>", line 1
    f"Eric is {2 * 37 #Oh my!}."
                      ^
SyntaxError: f-string expression part cannot include '#'
```

**conversion** :  (`"r" | "s" | "a"`) 在格式化之前进行类型强制转换。 通常，格式化值的工作由值本身的 `__format__()` 方法来完成。 但是，在某些情况下最好强制将类型格式化为一个字符串，覆盖其本身的格式化定义。 

目前支持的转换旗标有三种: `'!s'` 会对值调用 [`str()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#str)，`'!r'` 调用 [`repr()`](https://docs.python.org/zh-cn/3/library/functions.html#repr) 而 `'!a'` 则调用 [`ascii()`](https://docs.python.org/zh-cn/3/library/functions.html#ascii)。

**format_spec**：`[fill align][sign][#][0][width][grouping_option][.precision][type]`
称为格式规格迷你语言，在格式字符串所包含的替换字段内部使用，用于定义单个值应如何呈现，包含字段宽度、对齐、填充、小数精度等细节信息。

```python
>>> table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
>>> for name, phone in table.items():
...     print(f'{name:10} ==> {phone:10d}')
...
Sjoerd     ==>       4127
Jack       ==>       4098
Dcab       ==>       7678
```

**=**：（Python3.8  新功能）表达式里含等号 `'='` 时，输出内容包括表达式文本、`'='` 和求值结果。

```python
>>> line = "The mill's closed"
>>> f"{line = }"
'line = "The mill\'s closed"
```

## 格式规格迷你语言

`type`：确定数据显示的类型

| 显示类型  | 含义                                        | 适用变量类型     |
| :-------- | :------------------------------------------ | :--------------- |
| `'s'`     | 字符串格式（默认）                          | str              |
| `'b'`     | 二进制整数格式                              | int              |
| `'c'`     | 字符格式，按unicode编码将整数转换为对应字符 | int              |
| `'d'`     | 十进制整数格式                              | int              |
| `'o'`     | 八进制整数格式                              | int              |
| `'x'`     | 十六进制整数格式（小写字母）                | int              |
| `'X'`     | 十六进制整数格式（大写字母）                | int              |
| `'n'`     | 十进制数字，带数字分隔符                    | 数字             |
| `'e | E'` | 科学计数法                                  | float 和 Decimal |
| `'f | F'` | 固定精度。默认精度(`precision`)是6          | float 和 Decimal |
| `'g | G'` | 通用格式，小数用 `f | F`，大数用 `e | E`    | float 和 Decimal |
| `'%'`     | 百分比                                      | float 和 Decimal |

```python
>>> points = 19
>>> total = 22
>>> 'Correct answers: {:.2%}'.format(points/total)
'Correct answers: 86.36%'
```

`align`：(` "<" | ">" | "=" | "^"` ) 字符串对齐方式。如果指定了一个有效的 *align* 值，则可以在该值前面加一个 *fill* 字符，它可以为任意字符，如果省略则默认为空格符。各种对齐选项的含义如下：

| 选项  | 含意                                                         |
| :---- | :----------------------------------------------------------- |
| `'<'` | 强制字段在可用空间内左对齐（这是大多数对象的默认值）。       |
| `'>'` | 强制字段在可用空间内右对齐（这是数字的默认值）。             |
| `'='` | 强制在符号（如果有）之后数字之前放置填充。 这被用于以 '+000000120' 形式打印字段。 这个对齐选项仅对数字类型有效。 这是当 '0' 紧接在字段宽度之前时的默认选项。 |
| `'^'` | 强制字段在可用空间内居中。                                   |

请注意，除非定义了最小字段宽度 `width`，否则字段宽度将始终与填充它的数据大小相同，因此在这种情况下，对齐选项没有意义。

`width` ：(数字) 定义最小宽度，包括任何前缀、分隔符和其他格式化字符。 
当未显式给出对齐方式时，在 *width* 字段前加一个零 (`'0'`) 将为数字高位填充零字符。 这相当于设置 *fill* 字符为 `'0'` 且 *alignment* 类型为 `'='`。

```python
# 对齐文本以及指定宽度
>>> '{:<30}'.format('left aligned')
'left aligned                  '
>>> '{:>30}'.format('right aligned')
'                 right aligned'
>>> '{:^30}'.format('centered')
'           centered           '
>>> '{:*^30}'.format('centered')  # use '*' as a fill char
'***********centered***********'
```

`sign`：(`"+" | "-" | " "`) 数字符号，仅对数字类型有效。

```python
>>> '{:+f}; {:+f}'.format(3.14, -3.14)  # show it always
'+3.140000; -3.140000'
>>> '{: f}; {: f}'.format(3.14, -3.14)  # show a space for positive numbers
' 3.140000; -3.140000'
>>> '{:-f}; {:-f}'.format(3.14, -3.14)  # show only the minus -- same as '{:f}; {:f}'
'3.140000; -3.140000'
```

`'#'` 仅适用于数字类型。
对于整数类型，当使用二进制、八进制或十六进制输出时，此选项会为输出值分别添加相应的 `'0b'`, `'0o'`, `'0x'` 或 `'0X'` 前缀。 
对于浮点数和复数类型，替代形式会使得转换结果总是包含小数点符号，即使其不带小数部分。 
此外，对于 `'g'` 和 `'G'` 转换，末尾的零不会从结果中被移除。

`grouping_option` ：(`"_" | ","`) 数字分隔符
`','` 选项表示使用逗号作为千位分隔符。 对于感应区域设置的分隔符，请改用 `'n'` 整数表示类型。
`'_'` 选项表示对浮点表示类型和整数表示类型 `'d'` 使用下划线作为千位分隔符。 对于整数表示类型 `'b'`, `'o'`, `'x'` 和 `'X'`，将为每 4 个数位插入一个下划线。 对于其他表示类型指定此选项则将导致错误。

```python
>>> '{:,}'.format(1234567890)
'1,234,567,890'
```

`.precision` ：(dot+数字) 表示精度
对于以 `'f'` and `'F'` 格式化的浮点数值表示几位小数。
对于以 `'g'` 或 `'G'` 格式化的浮点数值表示几位有效数字。 
对于非数字类型，该字段表示最大字段宽度。
对于整数值则不允许使用 *precision*。

**日期格式化**：

```python
>>> import datetime
>>> d = datetime.datetime(2010, 7, 4, 12, 15, 58)
>>> '{:%Y-%m-%d %H:%M:%S}'.format(d)
'2010-07-04 12:15:58'
```

# Linux终端ANSI控制码


```python
print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)

Prints the values to a stream, or to sys.stdout by default.
Optional keyword arguments:
file:  a file-like object (stream); defaults to the current sys.stdout.
sep:   string inserted between values, default a space.
end:   string appended after the last value, default a newline.
flush: whether to forcibly flush the stream.
```

|ANSI控制码|说明|
|:---|:---|
|`\033[0m` | 关闭所有属性 |
|`\033[1m` | 设置高亮度 |
|`\03[4m` |下划线 |
|`\033[5m` |闪烁 |
|`\033[7m` |反显 |
|`\033[8m` |消隐 |
|`\033[30m`  ~ `\033[37m` |设置字体颜色 |
|`\033[40m` ~  `\033[47m` |设置背景色 |
|`\033[nA` |光标上移n行，清除光标后内容 |
|`\03[nB` |光标下移n行 |
|`\033[nC` |光标右移n行 |
|`\033[nD` |光标左移n行 |
|`\033[nF` |光标上移n行，保留光标后内容 |
|`\033[y;xH`|设置光标位置 |
|`\033[2J` |清屏 |
|`\033[K` |清除从光标到行尾的内容 |
|`\033[s` |保存光标位置 |
|`\033[u` |恢复光标位置 |
|`\033[?25l` |隐藏光标 |
|`\33[?25h` | 显示光标|

设置前字体颜色或者背景色的控制码中中间的数字代表不同的颜色

|40~47|背景色| 30~37 |文字颜色|
|---:|:---|---:|:---|
|40 | 黑      | 30 | 黑  |
|41 | 红      | 31 | 红  |
|42 | 绿      | 32 | 绿  |
|43 | 黄      | 33 | 黄  |
|44 | 蓝      | 34 | 蓝  |
|45 | 紫      | 35 | 紫  |
|46 | 深绿    | 36 | 深绿|
|47 | 白色    | 37 | 白色|

比如需要输出灰底红色带有下划线的"你好"
格式: `\033[44;31m\033[4m你好\033[0m`
说明:  一共有3个控制串: `\033[44;31m` (灰底红色字),  `\033[4m`(下划线), `\033[0m`(关闭所有设置)