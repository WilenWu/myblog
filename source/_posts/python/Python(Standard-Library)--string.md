---
title: Python手册(Standard Library)--string
date: 2021-11-13 22:23:11
categories: [Python,Python标准库]
tags: [Python]
cover: /img/sudoku.jpeg
top_img: '#66CCFF'
description: Python3 字符串方法和格式化
---

# 字符串方法

| 计数   |      |
| :---------------- | :----------------- |
| str.count(substr,beg=0,end=len(string)) | 返回substr出现的次数|
| **去空格**     ||
| str.lstrip(chars) | 删除str左边的字符（或空格）     |
| str.rstrip(chars) | 删除str右边的字符（或空格）     |
| str.strip(chars) | 删除str两边的字符（或空格）     |
| **字符串补齐**   ||
| str.center(width,fillchar) | 返回str居中，宽度为width的字符串(fillchar为填充字符)      |
| str.ljust(width,fillchar) | str左对齐|
| str.rjust(width,fillchar) | str右对齐|
| str.zfill (width) | str右对齐，前面填充0 |
| **大小写转换**   ||
| str.capitalize() | str的第一个字符大写  |
| str.title() | 每个单词首字母大写    |
| str.lower() | 小写    |
| str.upper() | 大写    |
| str.swapcase()   | 大小写互换 |
| **字符串条件判断** ||
| str.isalnum()    | 所有字符都是字母或数字  |
| str.isalpha()    | 所有字符都是字母     |
| str.isdigit()    | 所有字符都是数字     |
| str.isnumeric()  | 只包含数字字符      |
| str.isspace()    | 只包含空白 |
| str.istitle()    | 字符串是标题化      |
| str.islower()    | 都是小写  |
| str.isupper()    | 都是大写  |
| str.startswith(substr)  | 以substr开头    |
| str.endswith(substr)    | 以substr结尾    |
| **字符串搜索定位与替换**     |    |
| str.find(substr)  | 返回substr的索引位置，如果找不到，返回-1     |
| str.rfind(str)   ||
| str.index(substr) | 返回substr的索引位置，如果找不到，返回异常   |
| str.rindex(str)  ||
| str.replace(old,new,max) | 字符串替换，不超过 max 次（默认为1）。     |
| **字符串分割变换** ||
| str.join(seq)    | 以str分隔符，合并seq中所有的元素 |
| str.split(sep="",num)   | 分割str，num=str.count(sep)默认 |
| str.splitlines(keepends) | 按照行('\r','\r\n',\n')分隔，参数 keepends为False则不包含换行符 |
| **字符串编码与解码**||
| str.encode(encoding='UTF-8')   | 以 encoding 指定的编码格式编码字符串 |

# 格式化字符串

## % 格式化

Python 3.6 引入了新的字符串格式化方式 f-strings，与其它格式化方式相比，不仅简洁明了，可读性更好，更不容易出错，而且运行效率也更高。

在 Python 代码中，一个 f-string 就是一个带有`f`前缀的字符串，并通过大括号嵌入表达式，这些表达式最后将由他们的具体值取代。

同时值得注意的是，f-string就是在format格式化的基础之上做了一些变动，核心使用思想和format一样，因此大家可以学习完%s和format格式化，再来学习f-string格式化

## str.format 格式化

格式化字符串由 `{}`包围的替换文本 `replacement_field` 和普通文本组成。

由 `str.format()` 方法传递参数。

该语法在大多数情况下与旧式的 `%` 格式化类似，只是增加了 `{}` 和 `:` 来取代 `%`。 例如，，`'%03.2f'` 可以被改写为 `'{:03.2f}'`。

`replacement_field` 简单组成： `[field_name][!conversion][:format_spec]`

- **field_name** : 被替换的字符

  - 可以是一个数字，表示位置参数。如果`field_name` 依次为`0,1,2，...`，则它们可以全部省略。
  - 或者是命名关键字，,`str.format()`可通过关键字传递参数。

  ```python
  # 按位置访问参数
  >>> 'I am {}, I love {}'.format('Bill','Python')      # 忽略数字
  >>> 'I am {0}, I love {1}'.format('Bill','Python')     # 带数字编号
  >>> 'I am {1}, I love {0}'.format('Python','Bill')     # 打乱顺序
  'I am Bill, I love Python'
  
  # 按名称访问参数
  >>>'I am {name}, I love {lang}'.format(name='Bill',lang='Python')  
  
  # 解包作为参数
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

- **conversion** :  (`"r" | "s" | "a"`) 在格式化之前进行类型强制转换。 通常，格式化值的工作由值本身的 `__format__()` 方法来完成。 但是，在某些情况下最好强制将类型格式化为一个字符串，覆盖其本身的格式化定义。 通过在调用 `__format__()` 之前将值转换为字符串，可以绕过正常的格式化逻辑。

  目前支持的转换旗标有三种: `'!s'` 会对值调用 [`str()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#str)，`'!r'` 调用 [`repr()`](https://docs.python.org/zh-cn/3/library/functions.html#repr) 而 `'!a'` 则调用 [`ascii()`](https://docs.python.org/zh-cn/3/library/functions.html#ascii)。

- **format_spec**：`[fill align][sign][#][0][width][grouping_option][.precision][type]`
  称为格式规格迷你语言，在格式字符串所包含的替换字段内部使用，用于定义单个值应如何呈现，包含字段宽度、对齐、填充、小数精度等细节信息。

  - `type`：确定数据显示的类型
    可用的字符串类型：

    | 类型  | 含意                                         |
    | :---- | :------------------------------------------- |
    | `'s'` | 字符串格式。这是字符串的默认类型，可以省略。 |
    | None  | 和 `'s'` 一样。                              |

    可用的整数类型：

    | 类型  | 含意                                                         |
    | :---- | :----------------------------------------------------------- |
    | `'b'` | 二进制格式。 输出以 2 为基数的数字。                         |
    | `'c'` | 字符。在打印之前将整数转换为相应的unicode字符。              |
    | `'d'` | 十进制整数。 输出以 10 为基数的数字。                        |
    | `'o'` | 八进制格式。 输出以 8 为基数的数字。                         |
    | `'x'` | 十六进制格式。 输出以 16 为基数的数字，使用小写字母表示 9 以上的数码。 |
    | `'X'` | 十六进制格式。 输出以 16 为基数的数字，使用大写字母表示 9 以上的数码。 在指定 `'#'` 的情况下，前缀 `'0x'` 也将被转为大写形式 `'0X'`。 |
    | `'n'` | 数字。 这与 `'d'` 相似，不同之处在于它会使用当前区域设置来插入适当的数字分隔字符。 |
    | None  | 和 `'d'` 相同。                                              |

    float 和 Decimal 值的可用表示类型有：

    | 类型      | 含意                                               |
    | :-------- | :------------------------------------------------- |
    | `'e | E'` | 科学计数法。                                       |
    | `'f | F'` | 固定精度。默认对 float 采用小数点之后 6 位精度。   |
    | `'g | G'` | 常规格式。舍入到 `p` 个有效数字。                  |
    | `'n'`     | 数字。 这与 `'g'` 相似，会插入适当的数字分隔字符。 |
    | `'%'`     | 百分比。                                           |
    | None      |                                                    |

    ```python
    >>> points = 19
    >>> total = 22
    >>> 'Correct answers: {:.2%}'.format(points/total)
    'Correct answers: 86.36%'
    ```

  - `align`：(` "<" | ">" | "=" | "^"` ) 字符串对齐方式。如果指定了一个有效的 *align* 值，则可以在该值前面加一个 *fill* 字符，它可以为任意字符，如果省略则默认为空格符。各种对齐选项的含义如下：

    | 选项  | 含意                                                         |
    | :---- | :----------------------------------------------------------- |
    | `'<'` | 强制字段在可用空间内左对齐（这是大多数对象的默认值）。       |
    | `'>'` | 强制字段在可用空间内右对齐（这是数字的默认值）。             |
    | `'='` | 强制在符号（如果有）之后数字之前放置填充。 这被用于以 '+000000120' 形式打印字段。 这个对齐选项仅对数字类型有效。 这是当 '0' 紧接在字段宽度之前时的默认选项。 |
    | `'^'` | 强制字段在可用空间内居中。                                   |

    请注意，除非定义了最小字段宽度 `width`，否则字段宽度将始终与填充它的数据大小相同，因此在这种情况下，对齐选项没有意义。

  - `width` ：(数字) 定义最小宽度，包括任何前缀、分隔符和其他格式化字符。 如果未指定，则字段宽度将由内容确定。
    当未显式给出对齐方式时，在 *width* 字段前加一个零 (`'0'`) 字段将为数字类型启用感知正负号的零填充。 这相当于设置 *fill* 字符为 `'0'` 且 *alignment* 类型为 `'='`。

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

  - `sign`：(`"+" | "-" | " "`) 数字符号，仅对数字类型有效。

    ```python
    >>> '{:+f}; {:+f}'.format(3.14, -3.14)  # show it always
    '+3.140000; -3.140000'
    >>> '{: f}; {: f}'.format(3.14, -3.14)  # show a space for positive numbers
    ' 3.140000; -3.140000'
    >>> '{:-f}; {:-f}'.format(3.14, -3.14)  # show only the minus -- same as '{:f}; {:f}'
    '3.140000; -3.140000'
    ```

  - `'#'` 仅适用于数字类型。 对于整数类型，当使用二进制、八进制或十六进制输出时，此选项会为输出值分别添加相应的 `'0b'`, `'0o'`, `'0x'` 或 `'0X'` 前缀。 对于浮点数和复数类型，替代形式会使得转换结果总是包含小数点符号，即使其不带小数部分。 通常只有在带有小数部分的情况下，此类转换的结果中才会出现小数点符号。 此外，对于 `'g'` 和 `'G'` 转换，末尾的零不会从结果中被移除。

  - `grouping_option` ：(`"_" | ","`) 数字分隔符
    `','` 选项表示使用逗号作为千位分隔符。 对于感应区域设置的分隔符，请改用 `'n'` 整数表示类型。
    `'_'` 选项表示对浮点表示类型和整数表示类型 `'d'` 使用下划线作为千位分隔符。 对于整数表示类型 `'b'`, `'o'`, `'x'` 和 `'X'`，将为每 4 个数位插入一个下划线。 对于其他表示类型指定此选项则将导致错误。

    ```python
    >>> '{:,}'.format(1234567890)
    '1,234,567,890'
    ```

  - `.precision` ：(dot+数字) 表示精度
    对于以 `'f'` and `'F'` 格式化的浮点数值要在小数点后显示多少个数位。
    对于以 `'g'` 或 `'G'` 格式化的浮点数值要在小数点前后共显示多少个数位。 
    对于非数字类型，该字段表示最大字段宽度。
    对于整数值则不允许使用 *precision*。

  - 使用特定类型的专属格式化：

    ```python
    >>> import datetime
    >>> d = datetime.datetime(2010, 7, 4, 12, 15, 58)
    >>> '{:%Y-%m-%d %H:%M:%S}'.format(d)
    '2010-07-04 12:15:58'
    ```

## f-string 格式化



# 字符串常量

| 标准库   | `import string`     |
| ------------ | ---------- |
| string.digits      | 数字0~9 |
| string.letters     | 所有字母（大小写）    |
| string.lowercase   | 所有小写字母|
| string.printable   | 可打印字符的字符串    |
| string.punctuation | 所有标点  |
| string.uppercase   | 所有大写字母|

