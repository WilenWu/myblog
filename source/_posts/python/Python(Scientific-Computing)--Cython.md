---
title: Python(Scientific Computing)--Cython
tags:
  - Python
  - 科学计算
categories:
  - Python
  - 'Scientific Computing'
cover: /img/cython-cover.jpg
top_img: /img/cython-logo.png
abbrlink: 619a34fc
date: 2025-03-26 16:05:01
description: 旨在融合 Python 的易用性和 C 语言的高性能
---

Cython 是一个编程语言和编译器，旨在融合 Python 的易用性和 C 语言的高性能。它的主要功能是允许在 Python 代码中使用静态类型声明。它通过将 Python 代码转换为优化的 C / C ++代码，从而显著提升 Python 程序的运行速度。 —— [官方网站](https://cython.org/)

**安装 Cython**

```bash
$ pip install cython
```

在终端中输入 `cython -V` 查看是否安装成功。

# 编译与运行

Cython 有两种不同的语法变体：Cython 和 Pure Python ，代表了用C数据类型注释代码的不同方式。

- Cython特定语法，旨在从C/C++的角度使类型声明简洁且易于阅读。这种语法主要用于使用高级C或C++功能。语法在Cython 源文件  `.pyx`  中使用。

- 纯Python语法允许在常规Python语法中进行静态Cython类型声明，遵循[PEP-484](https://www.python.org/dev/peps/pep-0484/)类型提示和[PEP 526](https://www.python.org/dev/peps/pep-0526/)变量注释。纯Python语代码作为正常的Python模块运行，但在编译时，Cython将它们解释为C数据类型，并使用它们生成优化的C/C\+\+代码。
  要在Python语法中使用C/C\+\+数据类型，需要在要编译的Python模块中导入 `cython` 模块。本文使用纯Python语法时，默认导入。
  
  ```python
  import cython
  ```

Cython 代码需经过两个阶段生成 Python 扩展模块：

1. **Cython 编译器**：将 Cython 源文件 `.pyx` 或 `.py` 转换为优化的 C/C++ 代码。
2. **C/C++ 编译器**：将生成的代码编译为共享库（`.so/.pyd`）。

## 使用 setuptools 包

**编写 Cython 代码**：先创建一个 Cython 源文件

```python
# fib.py
import cython

def fib(n):
    i: cython.int
    a: cython.double = 0.0
    b: cython.double = 1.0
    for i in range(n):
        a, b = a + b, a
    return a
```

**创建setup.py文件**：用Python标准库编译 Cython 代码

```python
# setup.py
from setuptools import Extension, setup
from Cython.Build import cythonize

ext = Extension(
    name='fib',  # if use path, seperate with '.'.
    sources=['fib.py'],  # like .c files in c/c++.
    language='c',  # c or c++.
)
setup(ext_modules=cythonize(ext, annotate=True, language_level=3))
```

构建扩展模块的过程分为三步：

- 首先 `Extension` 对象负责配置：name 表示编译之后的文件名，sources 则是代表 pyx 和 C/C++ 源文件列表；
- 然后 `cythonize` 负责将 Cython 代码转成 C 代码，参数 `language_level=3` 表示只需要兼容 python3 即可，而默认是 2 和 3 都兼容；`annotate`参数显示 Cython 的代码分析。
- 最后 `setup` 根据 C/C++ 代码生成扩展模块

**编译 Cython 代码**：在命令行中运行以下命令进行编译，即可生成相应的 Python 扩展模块

```bash
$ python setup.py build_ext --inplace
```

> 可选的 `--inplace` 标志指示 `setup` 将每个扩展模块放置在其各自的 Cython `.pyx` 源文件旁边。

编译后会多出一个 `.pyd` 文件或 `.so` 文件，这就是根据 `fib.pyx` 生成的扩展模块，至于其它的可以直接删掉了。

**调用编译后的扩展模块**：在 Python 中导入并使用该模块

```python
# main.py
import fib
result = fib.fib(10)
```

注意：`setuptools` 74.1.0 版本增加了对 `pyproject.toml` 中扩展模块的实验性支持（而不是使用 `setup.py`）：

```toml
[build-system]
requires = ["setuptools", "cython"]
build-backend = "setuptools.build_meta"

[project]
name = "main"
version = "0.1.0"

[tool.setuptools]
ext-modules = [
  {name = "fib", sources = ["fib.pyx"]}
]
```

在这种情况下，您可以使用任何构建前端，例如

```bash
$ python -m build
```

## 使用 Jupyter Notebook

要启用对 Cython 编译的支持，需要先用魔法命令加载 `Cython`扩展：

```python
In [1]: %load_ext Cython
```

然后，在单元格前加上  `%%cython` 来编译

```python
In [2]: %%cython
...: def fib(int n):
...:     i: cython.int
...:     a: cython.double = 0.0
...:     b: cython.double = 1.0
...:     for i in range(n):
...:         a, b = a + b, a
...:     return a

In [3]: fib(90)
Out[3]: 2.880067194370816e+18
```

## 即时编译

Cython 提供的 `pyximport` 包改造了 `import` 语句，使其能够识别 `.pyx` 扩展模块。在导入 Cython 扩展模块之前调用 `pyximport.install()`，自动编译。

```python
In [1]: import pyximport; pyximport.install()
Out[1]: (None, <pyximport.pyximport.PyxImporter at 0x101548a90>)

In [2]: import fib
```

这样可以省去编写 `setup.py` 脚本的需要。如果修改了 Cython 源文件，`pyximport` 会自动检测到修改，并在新的 Python 解释器会话中重新编译源文件。

## 性能分析

Cython 编译器有一个可选的 `--annotate` 选项（简写为 `-a`），用于生成一个 HTML 代码注释，根据每行代码调用 Python/C API 的次数进行性能评估。

```bash
$ cython --annotate integrate.pyx
```

在标准流程中通过设置扩展模块的 `annotate` 参数生成

```python
setup(ext_modules=cythonize(ext, annotate=True, language_level=3))
```

在 IPython 魔法命令中添加 `--annotate` 选项（简写为 `-a`）生成

```python
In [1]: %%cython -a
```

## 编译器指令

Cython 提供了编译器指令，用于控制 Cython 源代码的编译方式。

**常用的编译器指令如下**：

| 指令                       | 说明                                        |
| ------------------------ | ----------------------------------------- |
| language_level           | 全局设置用于模块编译的 Python 语言级别，默认值为 `None`       |
| infer_types              | 在函数体中推断未加类型注解的变量类型，默认值为 `None`            |
| annotation_typing        | 是否使用函数参数注解编译，默认值为 `True`                  |
| cdivision                | 获得C语义的除法和模运算，默认值为 `False`                 |
| boundscheck              | 假设代码中的索引操作不会引发任何 `IndexError`，默认值为 `True` |
| wraparound               | 是否支持负索引，默认值为 `True`                       |
| nonecheck                | 检查None值，默认值为 `False`                      |
| overflowcheck            | 整数溢出检查，默认值为 `True`                        |
| initializedcheck         | 检查内存视图是否已初始化，默认值为 `True`                  |
| freethreading_compatible | 表明该模块可以在没有活动 GIL 的情况下安全运行，默认值为 `False`    |

**全局指令：** 可以通过在文件顶部附近添加特殊的头注释来设置编译器指令，如下所示：

```python
# cython: language_level=3, boundscheck=False
```

或者分别写在不同的行上

```python
# cython: language_level=3
# cython: boundscheck=False
```
该注释必须出现在任何代码之前（但可以在其他注释或空白之后）。

您也可以通过在命令行中使用 `-X` 或 `--directive` 选项来传递指令：

```bash
$ cython -X language_level=3 boundscheck=True main.pyx
```

在命令行中使用 `-X` 选项设置的指令将覆盖头注释中设置的指令。

**局部指令**：某些指令支持通过装饰器进行局部控制

```python
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_indexing():
    # ...
```

还可以使用上下文管理器的形式，如关闭边界检查进一步优化循环。

```python
cimport cython

def fast_indexing(a):
    with cython.boundscheck(False), cython.wraparound(False):
        for i in range(len(a)):
            sum += a[i]
```

每次我们访问内存视图时，Cython 都会检查索引是否在范围内。如果索引超出范围，Cython 会引发一个 `IndexError`。此外，Cython 允许我们使用负索引对内存视图进行索引（即索引环绕），就像 Python 列表一样。

如果我们事先知道我们永远不会使用超出范围的索引或负索引，因此我们可以指示 Cython 关闭这些检查以获得更好的性能。为此，我们使用 `cython` 特殊模块与 `boundscheck` 和 `wraparound` 编译器指令


**无论是装饰器形式还是上下文管理器形式的指令，都不会受到注释或命令行指令的影响。**

**在 `setup.py` 中设置**：也可以在 `setup.py` 文件中通过将关键字参数传递给 `cythonize` 来设置编译器指令：

```python
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="My hello app",
    ext_modules=cythonize('hello.pyx', compiler_directives={'embedsignature': True}),
)
```

这将覆盖在 `compiler_directives` 字典中指定的默认指令。注意，明确在文件中或局部设置的指令（如上所述）将优先于传递给 `cythonize` 的值。

## 已编译开关

`compiled` 是一个特殊变量，当编译器运行时，它被设置为 `True`，在 CPython 解释器中则为 `False`。因此，以下代码：

```python
import cython

if cython.compiled:
    print("Yep, I'm compiled.")
else:
    print("Just a lowly interpreted script.")
```

根据代码是作为编译后的扩展模块（`.so`/`.pyd`）运行，还是作为普通的 `.py` 文件运行，其行为会有所不同。


# 静态类型声明

## 静态数据类型

静态变量可以通过以下方式声明

- 使用Cython特定的`cdef` 语句，
- 使用带有C数据类型的PEP-484/526类型注释或
- 使用函数`cython.declare()`

`cdef` 语句和`declare()`可以定义本地和模块级变量以及类中的属性，但类型注释只影响本地变量和属性，在模块级别被忽略。这是因为类型注释不是特定于Cython的，因此Cython将变量保留在模块字典中。

{% tabs datatypes %}

<!-- tab Pure Python -->

```python
global_var = declare(cython.int, 42)

def main():
    i: cython.int
    f: cython.float = 2.5
    g: cython.int[4] = [1, 2, 3, 4]
    h: cython.p_float = cython.address(f)
    c: cython.doublecomplex = 2 + 3j

    i = 5
```

还支持使用 `cython.typedef() `函数为类型命名，这与C中的`typedef`语句类似

```python
ULong = cython.typedef(cython.ulong)
IntPtr = cython.typedef(cython.p_int)
```

<!-- endtab -->

<!-- tab Cython -->

在Cython中，通过`cdef`关键字来声明静态变量，具有C类型的变量使用C语法

```python
cdef int global_var = 42

def main():
    cdef int i = 10, j, k
    cdef float f = 2.5
    cdef int[4] g = [1, 2, 3, 4]
    cdef float *h = &f
    cdef double complex c = 2 + 3j

    j = k = 5
```

也可以使用Python风格的缩进进行声明，这与C中的`typedef`语句类似

```python
cdef:
    int x, y
    double pi = 3.14159
```

还支持使用`ctypedef`语句为类型命名

```python
ctypedef unsigned long ULong
ctypedef int* IntPtr
```

<!-- endtab -->

<!-- tab local -->

使用 `@cython.locals` 装饰器指定函数体中的局部变量的类型（包括参数）。使用 `@cython.returns` 指定函数的返回类型

```python
@cython.cfunc
@cython.returns(cython.bint)
@cython.locals(a=cython.int, b=cython.int)
def c_compare(a, b):
    return a == b
```

<!-- endtab -->

{% endtabs %}


Cython支持所有标准的C类型以及它们的无符号版本：

| Cython type           | Pure Python type           |
|:--------------------- |:-------------------------- |
| `bint`                | `cython.bint`              |
| `char`                | `cython.char`              |
| `signed char`         | `cython.schar`             |
| `unsigned char`       | `cython.uchar`             |
| `short`               | `cython.short`             |
| `unsigned short`      | `cython.ushort`            |
| `int`                 | `cython.int`               |
| `unsigned int`        | `cython.uint`              |
| `long`                | `cython.long`              |
| `unsigned long`       | `cython.ulong`             |
| `long long`           | `cython.longlong`          |
| `unsigned long long`  | `cython.ulonglong`         |
| `float`               | `cython.float`             |
| `double`              | `cython.double`            |
| `long double`         | `cython.longdouble`        |
| `float complex`       | `cython.floatcomplex`      |
| `double complex`      | `cython.doublecomplex`     |
| `long double complex` | `cython.longdoublecomplex` |
| `size_t`              | `cython.size_t`            |
| `Py_ssize_t`          | `cython.Py_ssize_t`        |
| `Py_hash_t`           | `cython.Py_hash_t`         |
| `Py_UCS4`             | `cython.Py_UCS4`           |


## C 派生数据类型

Cython 支持同样指针、数组、结构体、枚举等复杂类型

### 指针

{% tabs pointer %}

<!-- tab Pure Python -->

纯python模式下，指针类型可以使用`cython.pointer[]` 构建

```python
pi: cython.double = 3.14
ptr: cython.pointer[cython.double] = cython.address(pi)
ptr: cython.p_double = cython.address(pi)

p2: cython.p_int = cython.NULL
```

简单的指针类型支持带有 p 前缀的快捷命名方案，如`cython.p_int`等同于`cython.pointer[cython.int]`。

<!-- endtab -->

<!-- tab Cython -->

Cython 构建指针的语法和C一致

```python
cdef double pi = 3.14
cdef double* ptr = &pi

cdef int* p2 = NULL
```

<!-- endtab -->

{% endtabs %}

注意：在Python 中，`*` 有特殊含义，指针无法像 C 中那样解引用。在 Cython 中通过数组索引  `ptr[0]`  的方式获取指针变量的值。

```python
print(ptr[0])
```

或者使用`cython.operator.dereference`函数式运算符来解引用指针

```python
from cython cimport operator
print(operator.dereference(ptr))
```

### 数组

C 数组可以通过添加 `[ARRAY_SIZE]` 来声明

{% tabs c-array %}

<!-- tab Pure Python -->

```python
def main():
    g: cython.float[42]
    f: cython.int[5][5][5]
    ptr_char_array: cython.pointer[cython.char[4]]  # pointer to the array of 4 chars
    array_ptr_char: cython.p_char[4]                # array of 4 char pointers
```

<!-- endtab -->

<!-- tab Cython -->

```python
def main():
    cdef float[42] g
    cdef int[5][5][5] f
    cdef char[4] *ptr_char_array     # pointer to the array of 4 chars
    cdef (char *)[4] array_ptr_char  # array of 4 char pointers
```

注意：Cython 语法目前支持两种声明数组的方式：

```python
cdef int arr1[4], arr2[4]  # C style array declaration
cdef int[4] arr1, arr2     # Java style array declaration
```

它们都生成相同的 C 代码，建议使用 Java 样式声明。

<!-- endtab -->

{% endtabs %}


### 结构体

结构体定义如下

{% tabs struct %}

<!-- tab Pure Python -->

```python
Point = cython.struct(
    x=cython.double,
    y=cython.double)

def main():
    point: Point = Point(5.0, 3.0)
    print(point.x, point.y)
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef struct Point:
    double x, y

def main():
    cdef Point point = Point(5.0, 3.0)
    print(point.x, point.y)
```

我们还可以通过以下两种方式初始化结构体：

```python
cdef Point point = Point(x=1.0, y=2.0)
cdef Point point = {"x": 1.0, "y": 2.0}
```

<!-- endtab -->

{% endtabs %}

与C不同，Cython 使用**点操作符**来访问结构体成员


### 枚举

目前，纯Python模式不支持`enum`。 Cython 支持使用 `cdef` 和 `cpdef` 定义，我们可以在单独的行上定义成员，或者在一行上用逗号分隔：

```python
cpdef enum Color:
    RED = 1
    YELLOW = 3
    GREEN = 5

cdef enum Color:
    RED, YELLOW, GREEN
```

> 注意：在Cython语法中，struct、union和enum 关键字仅在定义类型时使用，声明和引用时省略。

## Python 内置类型

Cython 同样支持Python的数据类型进行静态声明。前提是它们必须是用C实现的，并且Cython必须能够访问到它们的声明。内置的Python类型（如`list`、`tuple`和`dict`）已经满足这些要求。

{% tabs build-in %}

<!-- tab Pure Python -->

```python
a: str = "hello"
t: tuple = (1, 2, 3)
lst: list = []
lst.append(1)
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef str a = "hello"
cdef tuple t = (1, 2, 3)
cdef list lst = []
lst.append(1)
```

<!-- endtab -->

{% endtabs %}

Cython目前支持几种内置的可静态声明的Python类型，包括：

- `type`、`object`  
- `bool`  
- `complex`  
- `basestring`、`str`、`unicode`、`bytes`、`bytearray`  
- `list`、`tuple`、`dict`、`set`、`frozenset`  
- `array`  
- `slice`  
- `date`、`time`、`datetime`、`timedelta`、`tzinfo`  

另外，Cython 提供了一个 Python 元组的有效替代品 `ctuple`。 `ctuple`由任何有效的 C 类型组装而成

{% tabs ctuple %}

<!-- tab Pure Python -->

```python
def main():
    bar: tuple[cython.double, cython.int]
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef (double, int) bar
```

<!-- endtab -->

{% endtabs %}

## 静态和动态混合使用

Cython 同时允许静态 C 变量和 Python 动态类型变量的混合使用

{% tabs c-python %}

<!-- tab Pure Python -->

```python
def main():
    a: cython.int
    b: cython.int
    c: cython.int
    t = (a, b, c)
```

<!-- endtab -->

<!-- tab Cython -->

```python
def main():
    cdef int a, b, c
    t = (a, b, c)
```

<!-- endtab -->

{% endtabs %}

Cython 允许静态 C 变量赋值给 Python 动态变量，同时会自动转换类型

{% tabs datatypes %}

<!-- tab Pure Python -->

```python
num: cython.int = 6
a = num
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef int num = 6
a = num
```

<!-- endtab -->

{% endtabs %}

内置Python类型与C或C++类型的对应关系

| C types                                                     | From Python types      | To Python types |
| ----------------------------------------------------------- | ---------------------- | --------------- |
| `bint`                                                      | `bool`                 | `bool`          |
| `[unsigned] char`<br>`[unsigned] short`<br>`int`<br>`long`  | `int`, `long`          | `int`           |
| `unsigned int`<br>`unsigned long`<br>`[unsigned] long long` | `int`, `long`          | `long`          |
| `float`<br>`double`<br>`long double`                        | `int`, `long`, `float` | `float`         |
| `char *`<br>`std::string` (C++)                             | `str/bytes`            | `str/bytes`     |
| `array`                                                     | `iterable`             | `list`          |
| `struct`                                                    |                        | `dict`          |

**注意：** 在Python 3中，所有的`int`对象都具有无限精度。当将整数类型从Python转换为C时，Cython会生成检查溢出的代码。如果C类型无法表示Python整数，则会在运行时抛出`OverflowError`。

使用 C 字符指针时，需要使用临时变量赋值，否则会编译错误

{% tabs p_char %}

<!-- tab Pure Python -->

```python
def main():
    s: cython.p_char
    p = pystring1 + pystring2
    s = p
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef char *s
p = pystring1 + pystring2
s = p
```

<!-- endtab -->

{% endtabs %}



## 强制类型转换

Cython 支持类型转换，无论是内置类型还是自定义类型

{% tabs cast %}

<!-- tab Pure Python -->

在纯 python 模式下，使用 `cast` 函数，如果希望在转换前检查类型，我们可以设置参数 `typecheck=True`

```python
def main():
    p: cython.p_char
    q: cython.p_float
    p = cython.cast(cython.p_char, q)

    safe_list: list = cython.cast(list, array)
```

<!-- endtab -->

<!-- tab Cython -->

Cython 语法的转换运算符与 C 类似，其中 C 使用 `()` ，Cython 使用 `<>`。如果希望在转换前检查类型，我们可以使用检查类型转换运算符 `?`

```python
cdef char *p
cdef float *q
p = <char*>q

cdef list safe_list = <list?>array
```

<!-- endtab -->

{% endtabs %}

## 融合类型

融合类型允许您定义一个类型，它可以指代多种类型。这使得您可以编写一个单一的静态类型化的 Cython 算法，能够操作多种类型的值。因此，融合类型允许泛型编程，类似于 C++ 中的模板或 Java/C# 中的泛型。

> 注意：融合类型目前不支持作为扩展类型的属性。只有变量和函数/方法的参数可以声明为合成类型。

### 声明融合类型

{% tabs fused_type %}

<!-- tab Pure Python -->

纯Python语法通过 `cython.fused_type` 函数自定义一个融合类型

```python
int_or_float = cython.fused_type(cython.char, cython.double)

@cython.ccall
def plus(a: int_or_float, b: int_or_float) -> int_or_float:
    # a and b always have the same type here
    return a + b

def show_me():
    a: cython.int = 127
    b: cython.double = 127.0
    print('int', plus_one(a, 1))
    print('float', plus_one(b, 1.0))
```

<!-- endtab -->

<!-- tab Cython -->

Cython 支持通过 `ctypedef fused` 自定义一个融合类型，支持的类型可以写在块里面

```python
ctypedef fused int_or_float:
    int
    double

cpdef int_or_float plus(int_or_float a, int_or_float b):
    # a and b always have the same type here
    return a + b

def show():
    cdef:
        int a = 127
        float b = 127.0
    print('int', plus_one(a, 1))
    print('float', plus_one(b, 1.0))
```

<!-- endtab -->

{% endtabs %}

如果同一个融合类型在函数参数中多次出现，那么它们将具有相同的特化类型。在上述示例中，两个参数的类型要么是 `int`，要么是 `double`，因为它们使用了相同的融合类型名称。如果函数或方法使用了融合类型，则至少有一个参数必须声明为该融合类型，以便Cython能够在编译时或运行时确定实际的函数特化版本。

但是，我们不能混合同一融合类型的特化版本，这样做会产生编译时错误，因为Cython没有可以分派的特化版本，从而导致`TypeError`。：

```python
plus(cython.cast(float, 1), cython.cast(int, 2)) # not allowed
```

### 选择特化

**索引**：可以通过对函数进行类型索引来获取某些特化，例如：

{% tabs SelectingSpecializations %}

<!-- tab Pure Python -->

```python
fused_type1 = cython.fused_type(cython.double, cython.float)
fused_type2 = cython.fused_type(cython.double, cython.float)

@cython.cfunc
def cfunc(arg1: fused_type1, arg2: fused_type1):
    print("cfunc called:", cython.typeof(arg1), arg1, cython.typeof(arg2), arg2)

@cython.ccall
def cpfunc(a: fused_type1, b: fused_type2):
    print("cpfunc called:", cython.typeof(a), a, cython.typeof(b), b)

def func(a: fused_type1, b: fused_type2):
    print("func called:", cython.typeof(a), a, cython.typeof(b), b)

# called from Cython space
cfunc[cython.double](5.0, 1.0)
cpfunc[cython.float, cython.double](1.0, 2.0)
# Indexing def functions in Cython code requires string names
func["float", "double"](1.0, 2.0)
```

<!-- endtab -->

<!-- tab Cython -->

```python
cimport cython

ctypedef fused fused_type1:
    double
    float

ctypedef fused fused_type2:
    double
    float

cdef cfunc(fused_type1 arg1, fused_type1 arg2):
    print("cfunc called:", cython.typeof(arg1), arg1, cython.typeof(arg2), arg2)


cpdef cpfunc(fused_type1 a, fused_type2 b):
    print("cpfunc called:", cython.typeof(a), a, cython.typeof(b), b)

def func(fused_type1 a, fused_type2 b):
    print("func called:", cython.typeof(a), a, cython.typeof(b), b)

# called from Cython space
cfunc[double](5.0, 1.0)
cpfunc[float, double](1.0, 2.0)
# Indexing def function in Cython code requires string names
func["float", "double"](1.0, 2.0)
```

<!-- endtab -->

{% endtabs %}

索引函数可以直接从 Python 中调用：

```python
>>> import cython
>>> import indexing
cfunc called: double 5.0 double 1.0
cpfunc called: float 1.0 double 2.0
func called: float 1.0 double 2.0
>>> indexing.cpfunc[cython.float, cython.float](1, 2)
cpfunc called: float 1.0 float 2.0
>>> indexing.func[cython.float, cython.float](1, 2)
func called: float 1.0 float 2.0
```

如果合成类型被用作更复杂类型的组成部分（例如指向合成类型的指针，或合成类型的内存视图），则应对函数进行单个组成部分的索引，而不是完整的参数类型：

{% tabs SelectingSpecializations2 %}

<!-- tab Pure Python -->

```python
@cython.cfunc
def myfunc(x: cython.pointer[A]):
    # ...

# Specialize using int, not int *
myfunc[cython.int](myint)
```

对于从 Python 空间进行内存视图索引，可以按以下方式操作：

```python
import numpy as np

myarray: cython.int[:, ::1] = np.arange(20, dtype=np.intc).reshape((2, 10))

my_fused_type = cython.fused_type(cython.int[:, ::1], cython.float[:, ::1])

def func(array: my_fused_type):
    print("func called:", cython.typeof(array))

func["int[:, ::1]"](myarray)
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef myfunc(A *x):
    # ...

# Specialize using int, not int *
myfunc[int](myint)
```

对于从 Python 空间进行内存视图索引，可以按以下方式操作：

```python
import numpy as np
import cython

myarray: cython.int[:, ::1] = np.arange(20, dtype=np.intc).reshape((2, 10))

ctypedef fused my_fused_type:
    int[:, ::1]
    float[:, ::1]

def func(my_fused_type array):
    print("func called:", cython.typeof(array))

func["int[:, ::1]"](myarray)
```

<!-- endtab -->

{% endtabs %}

### 内置融合类型

为了方便使用，Cython 提供了一些内置的融合类型：

- `cython.integral`：将 C 的`short`、`int`和`long`标量类型组合在一起
- `cython.floating`：将`float`和`double` C类型组合在一起
- `cython.numeric`：最通用的类型，将所有`integral`和`floating`类型以及`float complex`和`double complex`组合在一起

## Cython 函数

Cython中的函数支持三种类型：Python 函数、C函数和混合函数

### Python 函数

 `def` 函数是经过编译的Python原生函数，他们把Python对象作为参数，并返回Python对象。支持外部文件通过 import 语句直接调用。

{% tabs def %}

<!-- tab Pure Python -->

```python
def func(x: cython.double) -> cython.double:
    return x ** 2 - x
```

<!-- endtab -->

<!-- tab Cython -->

```python
def func(double x):
    return x ** 2 - x
```

<!-- endtab -->

{% endtabs %}

### C 函数

C函数的参数和返回值都要求指定明确的类型。它可以处理我们见过的任何静态类型，包括指针、结构体、C数组以及静态Python类型。我们也可以将返回类型声明为`void`。如果省略返回类型，则默认为`object`。

{% tabs cdef %}

<!-- tab Pure Python -->

使用 `@cfunc` 装饰器

```python
@cython.cfunc
def divide(a: cython.int, b: cython.int) -> cython.int: 
    return a / b
```

<!-- endtab -->

<!-- tab Cython -->

使用 Cython语法中的 `cdef` 语句

```python
cdef int divide(int a, int b):
    return a / b
```

<!-- endtab -->

{% endtabs %}

C 函数可以被同一Cython源文件中的任何其他函数（无论是`def`还是`cdef`）调用，但不允许从外部代码调用 C 函数。由于这一限制，我们需要在 Cython 中通过 `def` 函数包装下才能被外部模块识别。

```python
def wrap_divide(a, b):
    return divide(a, b)
```

### 混合函数

混合函数是`def`和`cdef`的混合体，相当于定义了C版本的函数和一个Python包装器。当我们从Cython调用该函数时，我们调用仅C版本，当我们从Python调用该函数时，调用包装器。由于这一特性， 混合函数的参数和返回类型必须同时兼容 Python 和 C。

{% tabs cpdef %}

<!-- tab Pure Python -->

使用 `@ccall` 装饰器

```python
@cython.ccall
def divide(a: cython.int, b: cython.int) -> cython.int: 
    return a / b
```

<!-- endtab -->

<!-- tab Cython -->

使用 Cython语法中的 `cpdef` 语句

```python
cpdef int divide(int a, int b):
    return a / b
```

<!-- endtab -->

{% endtabs %}

### 函数指针

纯Python模式目前不支持指向函数的指针

以下示例显示了声明`ptr_add`函数指针并指向`add`函数：

```python
cdef int(*ptr_add)(int, int)

cdef int add(int a, int b):
    return a + b

ptr_add = add

print(ptr_add(1, 3))
```

`struct`中声明的函数会自动转换为函数指针：

```python
cdef struct Bar:
    int sum(int a, int b)

cdef int add(int a, int b):
    return a + b

cdef Bar bar = Bar(add)

print(bar.sum(1, 2))
```

### 异常捕获

C函数或混合函数通常通过返回代码或错误标志来传达错误状态，但都无法引发 Python 异常（比如 ZeroDivisionError），这导致程序不会报错停止。

{% tabs exceptval %}

<!-- tab Pure Python -->

使用 `@cython.exceptval` 装饰器将 C 异常自动转换为 Python 异常

```python
@cython.cfunc
@cython.exceptval(-1, check=True)
def divide(a: cython.int, b: cython.int) -> cython.int: 
    return a / b
```

这里把C的返回值 `-1` 作为可能的异常。值`-1`是任意的，我们可以使用返回类型范围内的任意其他值。在这个例子中，我们使用关键字 `check=True`，是因为`-1`可能是`divide` 的有效结果。或者，为了使Cython检查是否发生了异常而不考虑返回值，我们可以用 `check` 参数，这将带来一些额外开销。

```python
@cython.cfunc
@cython.exceptval(check=True)
def divide(a: cython.int, b: cython.int) -> cython.int: 
    return a / b
```

<!-- endtab -->

<!-- tab Cython -->

Cython提供了一个`except`子句将 C 异常自动转换为 Python 异常

```python
cpdef int divide(int a, int b) except? -1:
    return a / b
```

这里把C的返回值 `-1` 作为可能的异常。值`-1`是任意的，我们可以使用返回类型范围内的任意其他值。在这个例子中，我们在`except? -1`子句中使用问号，因为`-1`可能是`divide` 的有效结果。或者，为了使Cython检查是否发生了异常而不考虑返回值，我们可以使用`except *`子句，这将带来一些额外开销。

```python
cpdef int divide(int a, int b) except *:
    return a / b
```

<!-- endtab -->

{% endtabs %}

# 扩展类型

## 扩展类型

Cython 支持直接使用 Python/C API 定义一个C级别的类，称为扩展类。扩展类型是可以被外部文件访问。和Python 类的主要区别在于它们使用 C 结构体来存储属性和方法，而不是 Python dict。

{% tabs cclass %}

<!-- tab Pure Python -->

通过 `@cclass`装饰器创建

```python
@cython.cclass
class Shrubbery:
    width: cython.int
    height: cython.int

    def __init__(self, w, h):
        self.width = w
        self.height = h

    def describe(self):
        print("This shrubbery is", self.width, "by", self.height, "cubits.")
```

<!-- endtab -->

<!-- tab Cython -->

通过 `cdef class` 语句创建

```python
cdef class Shrubbery:
    cdef int width
    cdef int height

    def __init__(self, w, h):
        self.width = w
        self.height = h

    def describe(self):
        print("This shrubbery is", self.width, "by", self.height, "cubits.")
```

<!-- endtab -->

{% endtabs %}

## 类属性和访问控制

注意，我们在 `__init__` 中实例化的属性，都必须在类中先声明。它们是C语言级别的实例属性，这种属性声明风格类似于C++和Java等语言。当扩展类型被实例化时，属性直接存储在对象的 C 结构体中。 在编译时需要知道该结构体的大小和字段，因此要声明所有属性。

扩展类型的属性默认是私有的，并且只能通过类的方法访问。若要想让外部 Python代码访问，可以声明为 `readonly` 或 `public`。

{% tabs public-readonly %}

<!-- tab Pure Python -->

```python
@cython.cclass
class Shrubbery:
    # Default. Not available in Python-space:
    width: cython.int 

    # Available in Python-space:
    height = cython.declare(cython.int, visibility='public') 

    # Available in Python-space, but only for reading:
    depth = cython.declare(cython.float, visibility='readonly') 
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef class Shrubbery:
    # Default. Not available in Python-space:
    cdef int width

    # Available in Python-space:
    cdef public int height

    # Available in Python-space, but only for reading:
    cdef readonly float depth
```

<!-- endtab -->

{% endtabs %}

默认情况下，无法在运行时向扩展类型添加属性。这是因为，C语言结构体是固定的。

## 类方法

和函数一样，扩展类型同样支持 Python 方法、C 方法和混合方法，但 C 方法和混合方法只能用于 `cdef class`  或 `@cython.cclass` 定义的扩展类，不能用于普通 Python 类。

{% tabs class-func %}

<!-- tab Pure Python -->

```python
@cython.cclass
cdef class Shrubbery:

    width: cython.double
    height: cython.double

    def __init__(self, w, h):
        self.width = w
        self.height = h

    @cython.cfunc
    def get_area(self) -> cython.double:
        return self.width * self.height
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef class Shrubbery:

    cdef double width, height

    def __init__(self, w, h):
        self.width = w
        self.height = h

    cdef double get_area(self):
        return self.width * self.height
```

<!-- endtab -->

{% endtabs %}

Cython 目前不支持使用 `@classmethod` 装饰器声明为类方法，但支持使用 `@staticmethod` 装饰器声明为静态方法。这在构造接受非 Python 兼容类型的类时特别有用：

{% tabs staticmethod %}

<!-- tab Pure Python -->

```python
from cython.cimports.libc.stdlib import free

@cython.cclass
class OwnedPointer:
    ptr: cython.p_void

    def __dealloc__(self):
        if self.ptr is not cython.NULL:
            free(self.ptr)

    @staticmethod
    @cython.cfunc
    def create(ptr: cython.p_void):
        p = OwnedPointer()
        p.ptr = ptr
        return p
```

<!-- endtab -->

<!-- tab Cython -->

```python
from libc.stdlib cimport free

cdef class OwnedPointer:
    cdef void* ptr

    def __dealloc__(self):
        if self.ptr is not NULL:
            free(self.ptr)


    @staticmethod
    cdef create(void* ptr):
        p = OwnedPointer()
        p.ptr = ptr
        return p
```

<!-- endtab -->

{% endtabs %}

## 继承

扩展类只能继承单个基类，并且继承的基类必须是直接指向 C 实现的类型，可以是使用扩展类型，也可以是内置类型，因为内置类型也是直接指向 C 一级的结构。

{% tabs subclassing %}

<!-- tab Pure Python -->

```python
@cython.cclass
class Parrot:

    @cython.cfunc
    def describe(self) -> cython.void:
        print("This parrot is resting.")

@cython.cclass
class Norwegian(Parrot):

    @cython.cfunc
    def describe(self) -> cython.void:
        Parrot.describe(self)
        print("Lovely plumage!")
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef class Parrot:

    cdef void describe(self):
        print("This parrot is resting.")


cdef class Norwegian(Parrot):

    cdef void describe(self):
        Parrot.describe(self)
        print("Lovely plumage!")
```

<!-- endtab -->

{% endtabs %}

扩展类不可以继承 Python 类，但 Python 类是可以继承扩展类的。此外，当使用 Python 类继承扩展类时，纯 C 函数可以被纯 C 函数 / 混合函数覆盖，但不能被 Python 函数覆盖。

## 初始化方法

Cython 支持初始化两个方法：普通的 Python `__init__()` 方法和新增的 `__cinit__()` 方法。`__cinit__` 和 `__init__` 只能通过 `def` 来定义。

`__init__()` 方法的工作方式与 Python 中完全相同。它是在对象分配和基本初始化之后被调用的，包括完整的继承链。如果通过直接调用对象的 `__new__()` 方法来创建对象（而不是调用类本身），那么任何 `__init__()` 方法都不会被调用。

而 `__cinit__()` 方法保证在对象分配时被调用，可以在其中执行基本的 C 结构初始化。我们实例化一个扩展类的时候，参数会先传递给`__cinit__`，然后`__cinit__`再将接收到的参数原封不动的传递给`__init__`。

{% tabs init %}

<!-- tab Pure Python -->

```python
@cython.cclass
class Penguin:
    food: object

    def __cinit__(self, food):
        self.food = food

    def __init__(self, food):
        print("eating!")

normal_penguin = Penguin('fish')
fast_penguin = Penguin.__new__(Penguin, 'wheat')  # note: not calling __init__() !
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef class Penguin:
    cdef object food

    def __cinit__(self, food):
        self.food = food

    def __init__(self, food):
        print("eating!")

normal_penguin = Penguin('fish')
fast_penguin = Penguin.__new__(Penguin, 'wheat')  # note: not calling __init__() !
```

<!-- endtab -->

{% endtabs %}

请注意，通过 `__new__()` 不会调用类型的 `__init__()` 方法（这在 Python 中也是已知的）。因此，在上面的例子中，第一次实例化会打印 eating!，但第二次不会。这只是 `__cinit__()` 方法比普通的 `__init__()` 方法更安全的原因之一。

Cython保证`__cinit__`只被调用一次，并且在`__init__`、`__new__`或 `staticmethod`之前被调用。Cython将任何初始化参数传递给`__cinit__`。

**注意**：所有构造函数参数都将作为 Python 对象传递。这意味着不能将不可转换的 C 类型（如指针或 C++ 对象）作为参数传递给构造函数，无论是从 Python 还是从 Cython 代码中。如果需要这样做，通常在该函数中直接调用 `__new__()` 方法，以明确绕过对 `__init__()` 构造函数的调用。

如果您的扩展类型有一个基类型，基类型层次结构中任何现有的 `__cinit__()` 方法都会在您的 `__cinit__()` 方法之前自动被调用。

## 内存分配和释放

Cython 添加了构造函数`__cinit__` 和析构函数 `__dealloc__`，用于执行 C 级别的内存分配和释放。

相比 C 的动态内存管理函数，Python 在 malloc、realloc、free 基础上做了一些简单的封装，这些函数对较小的内存块进行了优化，通过避免昂贵的操作系统调用来加快分配速度。

{% tabs py_malloc %}

<!-- tab Pure Python -->

```python
from cython.cimports.cpython.mem import PyMem_Malloc, PyMem_Realloc, PyMem_Free

@cython.cclass
class SomeMemory:
    data: cython.p_double

    def __cinit__(self, number: cython.size_t):
        # allocate some memory (uninitialised, may contain arbitrary data)
        self.data = cython.cast(cython.p_double, PyMem_Malloc(
            number * cython.sizeof(cython.double)))
        if not self.data:
            raise MemoryError()

    def resize(self, new_number: cython.size_t):
        # Allocates new_number * sizeof(double) bytes,
        # preserving the current content and making a best-effort to
        # reuse the original data location.
        mem = cython.cast(cython.p_double, PyMem_Realloc(
            self.data, new_number * cython.sizeof(cython.double)))
        if not mem:
            raise MemoryError()
        # Only overwrite the pointer if the memory was really reallocated.
        # On error (mem is NULL), the originally memory has not been freed.
        self.data = mem

    def __dealloc__(self):
        PyMem_Free(self.data)  # no-op if self.data is NULL
```

<!-- endtab -->

<!-- tab Cython -->

```python
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class SomeMemory:
    cdef double* data

    def __cinit__(self, size_t number):
        # allocate some memory (uninitialised, may contain arbitrary data)
        self.data = <double*> PyMem_Malloc(
            number * sizeof(double))
        if not self.data:
            raise MemoryError()

    def resize(self, size_t new_number):
        # Allocates new_number * sizeof(double) bytes,
        # preserving the current content and making a best-effort to
        # reuse the original data location.
        mem = <double*> PyMem_Realloc(
            self.data, new_number * sizeof(double))
        if not mem:
            raise MemoryError()
        # Only overwrite the pointer if the memory was really reallocated.
        # On error (mem is NULL), the originally memory has not been freed.
        self.data = mem

    def __dealloc__(self):
        PyMem_Free(self.data)  # no-op if self.data is NULL
```

<!-- endtab -->

{% endtabs %}

## nonecheck

{% tabs nonecheck %}

<!-- tab Pure Python -->

当使用注释语法时，行为遵循 PEP-484 的 Python 类型语义。当变量仅用其普通类型注解时，None 值是不允许的：

```python
def widen_shrubbery(sh: Shrubbery, extra_width):  
    sh.width = sh.width + extra_width
```

当 `sh` 为 `None` 时会引发 `TypeError`。要允许 `None`，必须显式使用 `typing.Optional[]` 

```python
import typing

def widen_shrubbery(sh: typing.Optional[Shrubbery], extra_width):
    sh.width = sh.width + extra_width
```

对于默认参数为 `None` 时，也会自动允许。

<!-- endtab -->

<!-- tab Cython -->

```python
def widen_shrubbery(Shrubbery sh, extra_width): 
    sh.width = sh.width + extra_width
```

如果我们传入一个非 `Shrubbery`对象，我们会得到一个`TypeError`。但是 Python的 `None`对象本质上没有C接口，因此尝试在其上调用方法或访问属性会导致程序崩溃。为了使这些操作安全，可以在调用之前检查 `sh` 是否为 `None`。这是一个非常常见的操作，因此 Cython为此提供了特殊的语法：

```python
def widen_shrubbery(Shrubbery sh not None, extra_width):
    sh.width = sh.width + extra_width
```

现在，该函数将自动检查 sh 是否为 not None，同时检查它是否具有正确的类型。

<!-- endtab -->

{% endtabs %}

注意：`not None` 和 `typing.Optional` 只能在 Python 函数中使用，不能在 C 函数中使用。如果你需要检查 C 函数的参数是否为 `None`，需要自行定义。

Cython还提供了一个`nonecheck` 编译器指令（默认关闭），它使所有函数和方法调用都对`None`安全。

# 模块管理

## 实现和声明

Cython 也允许我们将项目拆分为多个 `pyx` 模块。但 `import` 语句无法让两个Cython模块访问彼此的`cdef/cpdef` 函数、`ctypedef`、结构体等C级别构造。

类似于 C 中的头文件，Cython 提供了`pxd`文件来组织 Cython 文件。`pxd` 文件用于存放供外部代码使用的C声明，而它们的具体实现是在同名的 `pyx/py` 文件中。外部 Cython 文件可以通过 `cimport`语句将 `pxd`文件导入使用。

一个 `.pxd` 声明文件可以包含：

- 任何类型的C型声明
- 外部C函数或变量声明
- 模块中定义的C函数声明
- 扩展类型的定义部分

它不能包含任何C或Python函数的实现，或任何Python类定义，或任何可执行语句的实现。

`pxd` 声明文件用于编译时访问，只允许在其中放置C级别声明，不允许放置Python的声明，比如`def`函数。Python对象是在运行时是可访问的，因此它们仅在实现文件中。

扩展类型的定义部分只能声明 C 属性和 C 方法，不能声明 Python 方法，并且必须声明该类型的所有 C 属性和 C 方法。

{% tabs modules %}

<!-- tab Pure Python -->

假设我们有一个名为 shrubbing.py 的实现文件： 

```python
# shrubbing.py

real_t = cython.typedef(cython.double)

@cclass
class Shrubbery:
    width: real_t
    length: real_t

    def __init__(self, w: real_t, l: real_t):
        self.width = w
        self.length = l

    @cfunc
    def get_area(self) -> real_t:
        return self.width * self.length

def standard_shrubbery():
    return Rectangle(3., 7.)
```

<!-- endtab -->

<!-- tab Cython -->

假设我们有一个名为 shrubbing.pyx 的实现文件： 

```python
# shrubbing.pyx

ctypedef double real_t

cdef class Shrubbery:
    cdef int width
    cdef int length

    def __init__(self, real_t w, real_t l):
        self.width = w
        self.length = l

    cdef real_t get_area(self):
        return self.width * self.length

def standard_shrubbery():
    return Rectangle(3., 7.)
```

<!-- endtab -->

{% endtabs %}

为此，我们首先需要创建一个同名的 shrubbing.pxd 声明文件。在其中，我们放置希望共享的C级别构造的声明。

```python
# shrubbing.pxd

ctypedef double real_t

cdef class Shrubbery:
    cdef public:
        real_t width
        real_t length

    cdef real_t get_area(self)
```

如果我们在 `pxd` 文件中声明了一个函数或者变量，那么在对应的实现文件中不可以再次声明，否则会发生编译错误。因此实现文件也需要更改

{% tabs implementation-file %}

<!-- tab Pure Python -->

```python
# shrubbing.py

@cclass
class Shrubbery:
    def __init__(self, w: real_t, l: real_t):
        self.width = w
        self.length = l

    @cfunc
    def get_area(self) -> real_t:
        return self.width * self.length

def standard_shrubbery():
    return Rectangle(3., 7.)
```

在编译 shrubbing.py 时，`cython`编译器将自动检测到 shrubbing.pxd 文件，并使用其声明。

<!-- endtab -->

<!-- tab Cython -->

```python
# shrubbing.pyx

cdef class Shrubbery:
    def __init__(self, real_t w, real_t h):
        self.width = w
        self.length = l

    cdef real_t get_area(self):
        return self.width * self.height

def standard_rectangle():
    return Rectangle(1, 1)
```

在编译 shrubbing.pyx 时，`cython`编译器将自动检测到 shrubbing.pxd 文件，并使用其声明。

<!-- endtab -->

{% endtabs %}

为了避免重复（以及潜在的未来不一致），默认参数值在声明中（`.pxd`文件）中不可见，而仅在实现中可见。

## `cimport` 语句

Cython提供了 `cimport`语句，语法与`import` 一致。我们可以另一个 `pyx` 文件中，使用`cimport` 语句导入 `pxd`文件中静态声明的对象。

{% tabs cimport %}

<!-- tab Pure Python -->

使用纯 Python 语法时，可以通过从 `cython.cimports` 包中导入 `pxd`文件

```python
# main.py 
from cython.cimports.shrubbing import Shrubbery
import shrubbing

def main():
    sh: Shrubbery
    sh = shrubbing.standard_shrubbery()
    print("Shrubbery size is", sh.width, 'x', sh.length)
```

<!-- endtab -->

<!-- tab Cython -->

Cython 提供了 `cimport` 关键字用来导入 `pxd`文件

```python
# main.pyx
cimport shrubbing
import shrubbing

def main():
    cdef shrubbing.Shrubbery sh
    sh = shrubbing.standard_shrubbery()
    print("Shrubbery size is", sh.width, 'x', sh.length)
```

<!-- endtab -->

{% endtabs %}

注意，`cimport`语句只能用于导入C数据类型、C函数和变量以及扩展类型，并且这种导入发生在编译时（扩展类型除外）。任何Python对象，只能使用`import`语句在运行时导入。

最后，编译这两个模块

{% tabs compile-modules %}

<!-- tab Pure Python -->

```python
# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["main.py", "shrubbing.py"]))
```

<!-- endtab -->

<!-- tab Cython -->

```python
# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["main.pyx", "shrubbing.pyx"]))
```

<!-- endtab -->

{% endtabs %}

如果头文件不在同一个目录中，那么编译的时候还需要通过 `include_dirs` 参数指定头文件的所在目录。

## 外部库声明

Cython 提供了一个 `extern` 语句，可以直接调用 C/C++ 源码。一旦一个 C 函数在 `extern` 块中声明，它就可以像在 Cython 中定义的普通C函数一样被使用和调用。

Cython 目前不支持在纯 Python 模式中声明为 `extern` 。

例如，有一个头文件 `mymodule.h`，里面是函数声明，源文件 `mymodule.c` 里面是函数实现

```c
// mymodule.h
#define M_PI 3.1415926
#define MAX(a, b) ((a) >= (b) ? (a) : (b))

long fib(long);
typedef double real;
real* arrays(long[], long[][10], real**);

typedef struct {
    double x, y;
} Point;
```

通过 `cdef extern from` 声明在 `pxd` 文件中，然后 Cython 可以直接调用

```python
# mymodule.pxd
cdef extern from "mymodule.h":

    double M_PI
    float MAX(float a, float b)

    long fib(long n)
    ctypedef double real
    real* func_arrays(long[] i, long[][10] j, real** k)

    ctypedef struct Point:
        double x, y
```

在前面的 `extern` 块中，我们为函数参数添加了变量名称。这是推荐的，但并非强制性的：这样做可以让我们使用关键字参数调用这些函数。

上例在 Cython 编译时，不会自动为声明的对象生成 Python 包装器，我们仍然需要在 Cython 中使用 `def`、或者 `cpdef` 将 `extern` 块中声明的 C 级结构包装一下才能给 Python 调用。

或者，我们在导入的时候直接声明 `cpdef` ，这将生成一个 Python 包装器

```python
# mymodule.pxd

# declare a C function as "cpdef" to export it to the module
cdef extern from "mymodule.h":
    cpdef long fib(long n)
```

```python
# main.py
from cython.cimports.mymodule import fib
```

编译的时候，我们必须确保将 `mymodule.c` 源文件包含在 `sources` 列表中：

```python
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension("example", sources=["main.py", "mymodule.c"])
setup(ext_modules = cythonize([ext]))
```

还可以在 `.pxd` 文件中重命名外部函数，如下所示 `sin` 被重命名为 `_sin`：

```python
cdef extern from "math.h":
    cpdef double _sin "sin" (double x)
```

在某些情况下，您可能不需要结构的任何成员，在这种情况下，您可以将pass放入结构声明的正文中，例如：

```python
cdef extern from "foo.h":
    struct spam:
        pass
```

请注意，您只能在`from`块的`cdef extern`内执行此工作；其他地方的结构声明必须是非空的。

## C/C++ 标准库

方便的是，Cython附带了常用的 C/C++ 标准库、Python/C API 和 Numpy包的  `pxd` 声明文件，位于主Cython源目录下的 Includes 目录中。

- C 标准库 libc：包含 stdlib、stdio、math、string 和 stdint 等头文件
- C++ 标准模板库（STL）libcpp：包含  string、vector、list、map、pair 和 set 等容器

使用 `cimport` 导入预定义模块

```python
from libc.math cimport sin as csin
result = csin(0.5)
```

通过设置编译器指令 `language=c++`，Cython 可以编译为 C++ 代码，从而支持 C++ 标准库

```python
# distutils: language=c++
from libcpp.vector cimport vector

cdef vector[int]* vec = new vector[int](3)
```

## 初始化模块

Cython 同样支持初始化模块 `__init__.pxd`，类似于Python 包中的 `__init__.py`。

例如，树目录

```
CyIntegration/
├── __init__.py
├── __init__.pxd
├── integrate.py
└── integrate.pxd
```

在 `__init__.pxd` 中，用于声明任何 `cimport` 语法导入的C结构。


## 增强 `.pxd` 文件

增强 `.pxd` 文件可以在不改变原始 `.py` 文件的情况下实现静态声明。如果编译器找到与正在编译的 `.py` 文件同名的 `.pxd` 文件，编译器将查找 `cdef` 类和 `cdef`/`cpdef` 函数及方法。然后，将 `.py` 文件中对应的类/函数/方法转换为声明的类型。

例如，如果有一个文件 `A.py`：

```python
def myfunction(x, y=2):
    a = x - y
    return a + x * y

def _helper(a):
    return a + 1

class A:
    def __init__(self, b=0):
        self.a = 3
        self.b = b

    def foo(self, x):
        print(x + _helper(1.0))
```

并添加 `A.pxd`：

```python
cpdef int myfunction(int x, int y=*)
cdef double _helper(double a)

cdef class A:
    cdef public int a, b
    cpdef foo(self, double x)
```

那么 Cython 将编译 `A.py`，就好像它是这样写的：

```python
cpdef int myfunction(int x, int y=2):
    a = x - y
    return a + x * y

cdef double _helper(double a):
    return a + 1

cdef class A:
    cdef public int a, b
    def __init__(self, b=0):
        self.a = 3
        self.b = b

    cpdef foo(self, double x):
        print(x + _helper(1.0))
```

# 包装 C++ 库

## 声明 C++ 类

假设我们有一个简单C++头文件 `Rectangle.h`

```cpp
#ifndef RECTANGLE_H
#define RECTANGLE_H

namespace shapes {
    class Rectangle {
        public:
            int x0, y0, x1, y1;
            Rectangle();
            Rectangle(int x0, int y0, int x1, int y1);
            ~Rectangle();
            int getArea();
            void getSize(int* width, int* height);
            void move(int dx, int dy);
    };
}

#endif
```

以及在名为 `Rectangle.cpp` 的文件中的实现：

```cpp
#include <iostream>
#include "Rectangle.h"

namespace shapes {

    // Default constructor
    Rectangle::Rectangle () {}

    // Overloaded constructor
    Rectangle::Rectangle (int x0, int y0, int x1, int y1) {
        this->x0 = x0;
        this->y0 = y0;
        this->x1 = x1;
        this->y1 = y1;
    }

    // Destructor
    Rectangle::~Rectangle () {}

    // Return the area of the rectangle
    int Rectangle::getArea () {
        return (this->x1 - this->x0) * (this->y1 - this->y0);
    }

    // Get the size of the rectangle.
    // Put the size in the pointer args
    void Rectangle::getSize (int *width, int *height) {
        (*width) = x1 - x0;
        (*height) = y1 - y0;
    }

    // Move the rectangle by dx dy
    void Rectangle::move (int dx, int dy) {
        this->x0 += dx;
        this->y0 += dy;
        this->x1 += dx;
        this->y1 += dy;
    }
}
```

为了在 Cython 中声明此类接口，我们需要像之前一样使用 `extern` 块。这个 `extern` 块需要三个额外的元素来处理 C++ 的特性：

- 使用 `cppclass` 关键字声明 C++ 类
- 使用 Cython 的 `namespace` 子句声明 C++ 命名空间。如果没有命名空间，可以省略 `namespace` 子句。如果有多个嵌套的命名空间，可以将它们声明为 `namespace "outer::inner"`。也可以声明类的静态成员，例如 `"namespace::MyClass"` 。

接下来，我们将这些声明放在一个名为 `Rectangle.pxd` 的文件中。可以将其视为 Cython 可读的头文件：

```python
cdef extern from "Rectangle.cpp":
    pass

# Declare the class with cdef
cdef extern from "Rectangle.h" namespace "shapes":
    cdef cppclass Rectangle:
        Rectangle() except +
        Rectangle(int, int, int, int) except +
        int x0, y0, x1, y1
        int getArea()
        void getSize(int* width, int* height)
        void move(int, int)
```

Cython 只能包装 `public` 方法和成员，任何 `private` 或 `protected` 方法或成员都无法访问，因此也无法包装。

注意：构造函数被声明为 `"except +"`。如果 C++ 代码或初始内存分配由于失败而引发异常，这将允许 Cython 安全地引发适当的 Python 异常（见下文）。如果没有此声明，源自构造函数的 C++ 异常将不会被 Cython 处理。

现在我们可以在 `.pyx` 文件中使用 `cdef` 或 C++ 的 `new` 语句声明一个类的变量：

```python
# distutils: language = c++

from Rectangle cimport Rectangle

def main():
    rec_ptr = new Rectangle(1, 2, 3, 4)  # Instantiate a Rectangle object on the heap
    try:
        rec_area = rec_ptr.getArea()
    finally:
        del rec_ptr  # delete heap allocated object

    cdef Rectangle rec_stack  # Instantiate a Rectangle object on the stack
```

显然，使用默认构造函数的版本更加方便，消除了对 `try`/`finally` 块的需求。

## 包装 C++ 类

若要从 Python 中访问 C++ 类，我们仍然需要编写可从 Python 访问的扩展类型来包装

```python
# rect.pyx
# distutils: language = c++

from Rectangle cimport Rectangle

cdef class PyRectangle:
    cdef Rectangle* c_rect  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self):
        self.c_rect = new Rectangle()

    def __init__(self, int x0, int y0, int x1, int y1):
        self.c_rect.x0 = x0
        self.c_rect.y0 = y0
        self.c_rect.x1 = x1
        self.c_rect.y1 = y1

    def __dealloc__(self):
        del self.c_rect

    def get_area(self):
        return self.c_rect.getArea()

    def get_size(self):
        cdef int width, height
        self.c_rect.getSize(&width, &height)
        return width, height

    def move(self, dx, dy):
        self.c_rect.move(dx, dy)
```

我们已经看到了如何将一个简单的 C++ 类包装在一个扩展类型中。Cython 将 `new` 操作符传递到生成的 C++ 代码中。`new` 操作符只能与 C++ 类一起使用。每次调用 `new` 都必须与一个 `delete` 调用匹配。

## 使用 C++ 编译

在编译 C++ 项目时，我们需要指定编译器指令 `language=c++`，并将所有 C++ 源文件包含在 `sources` 列表参数中

```python
from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("rect",
                sources=["rect.pyx", "Rectangle.cpp"],
                language="c++")

setup(ext_modules=cythonize(ext))
```

我们可以简化 `setup` 脚本。在 rect.pyx 的顶部，我们添加以下指令注释：

```python
# distutils: language = c++
# distutils: sources = Rectangle.cpp
```

有了这些指令，`cythonize` 命令可以自动提取必要的信息，以正确构建扩展

```python
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("rect.pyx"))
```

## 静态成员方法

如果 `Rectangle` 类有一个静态成员：

```cpp
namespace shapes {
    class Rectangle {
    ...
    public:
        static void do_something();

    };
}
```

可以使用 Python 的 `@staticmethod` 装饰器来声明它，即：

```python
cdef extern from "Rectangle.h" namespace "shapes":
    cdef cppclass Rectangle:
        ...
        @staticmethod
        void do_something()
```

## 重载方法和运算符

重载方法非常简单，只需声明具有不同参数的方法，并使用它们中的任何一个即可：

```python
cdef extern from "Foo.h":
    cdef cppclass Foo:
        Foo(int)
        Foo(bool)
        Foo(int, bool)
        Foo(int, int)
```

Cython 使用 C++ 的命名方式来重载运算符：

```python
cdef extern from "foo.h":
    cdef cppclass Foo:
        Foo()
        Foo operator+(Foo)
        Foo operator-(Foo)
        int operator*(Foo)
        int operator/(int)
        int operator*(int, Foo) # allows 1*Foo()
    # nonmember operators can also be specified outside the class
    double operator/(double, Foo)
```

## 模版

Cython 使用方括号语法来实现模板函数。模版参数列表跟在函数名之后，用方括号括起来：

```python
# distutils: language = c++

cdef extern from "<algorithm>" namespace "std":
    T max[T](T a, T b) except +

print(max[long](3, 4))
print(max(1.5, 2.5))  # simple template argument deduction
```

可以定义多个模板参数，例如 `[T, U, V]` 或 `[int, bool, char]`。可选的模板参数可以通过写 `[T, U, V=*]` 来表示。

类模板的定义方式与模板函数类似，一个简单的包装 C++ `vector` 的例子如下：

```python
# distutils: language = c++

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        vector() except +
        vector(vector&) except +
        vector(size_t) except +
        vector(size_t, T&) except +
        T& operator[](size_t)
        void clear()
        void push_back(T&)
```

我们使用 `T` 作为模板类型，并声明了 `vector` 的四个构造函数以及一些更常见的方法。

假设我们想在一个包装函数中声明和使用一个 `int` 类型的 `vector`。对于模板化类，我们需要在模板类名后用方括号指定一个具体的模板类型：

```python
def wrapper_func(elts):
    cdef vector[int] v
    for elt in elts:
        v.push_back(elt)
    # ...
```

这适用于栈分配的 `vector`，但创建一个堆分配的 `vector` 需要使用 `new` 操作符：

```python
def wrapper_func(elts):
    cdef vector[int] *v = new vector[int]()
    # ...
    del v
```

当使用 `new` 进行堆分配时，我们需要确保在使用完 `vector` 指针后调用 `del`，以防止内存泄漏。

## 标准库

Cython 已经在 `/Cython/Includes/libcpp` 中的 `.pxd` 文件中声明了 C++ 标准模版库（STL）的大部分容器。这些容器包括：`deque`、`list`、`map`、`pair`、`queue`、`set`、`stack` 和 `vector`。

```python
# distutils: language = c++

from libcpp.vector cimport vector
cdef vector[int] *vec_int = new vector[int](10)
```

Cython 支持自动将 STL 容器转换为对应的 Python 内置类型。

```python
# cython: language_level=3
# distutils: language=c++

from libcpp.map cimport map

cdef vector[int] vect = range(1, 10, 2)
cdef vector[string] cpp_strings = b'It is a good shrubbery'.split()
```

下表列出了当前支持的从 Python 到 C++ 容器的所有内置转换

| Python type =>   | C++ type           | => Python type |
| ---------------- | ------------------ | -------------- |
| bytes            | std::string        | bytes          |
| iterable         | std::vector        | list           |
| iterable         | std::list          | list           |
| iterable         | std::set           | set            |
| iterable         | std::unordered_set | set            |
| mapping          | std::map           | dict           |
| mapping          | std::unordered_map | dict           |
| iterable (len 2) | std::pair          | tuple (len 2)  |
| complex          | std::complex       | complex        |

所有转换都会创建一个新的容器，并将数据复制到其中。容器中的项目会自动转换为对应类型，这包括递归转换容器内的容器，例如一个 C++ `vector` 的 `map` 的字符串。

这一强大特性允许我们直接从 `def` 或 `cpdef` 函数或方法返回一个支持的 C++ 容器，前提是该容器及其模板类型是受支持的。Cython 会自动将容器的内容转换为正确的 Python 容器。

```python
from libcpp.string cimport string
from libcpp.map cimport map

def periodic_table():
    cdef map[string, int] table
    table = {"H": 1, "He": 2, "Li": 3}
    # ...use table in C++...
    return table
```

Cython 支持通过 `for .. in` 语法（包括在列表推导式中）迭代标准库容器（或任何具有返回支持递增、解引用和比较的对象的 `begin()` 和 `end()` 方法的类）。例如，可以编写如下代码：

```python
# distutils: language = c++

from libcpp.vector cimport vector

def main():
    cdef vector[int] v = [4, 6, 5, 10, 3]

    cdef int value
    for value in v:
        print(value)

    return [x*x for x in v if x % 2 == 0]
```

尽管 Cython 没有 `auto` 关键字，但未显式使用 `cdef` 类型化的 Cython 局部变量会根据其所有赋值的右侧类型进行推导（参见 `infer_types` 编译器指令）。这在处理返回复杂、嵌套、模板化类型的函数时特别方便。

## 标准异常

Cython 不能抛出 C++ 异常，也不能使用 `try-except` 语句捕获它们。但Cython 具有检测它们何时发生并将它们自动转换为相应的 Python 异常的功能。要启用此功能，我们只需在可能引发 C++ 异常的函数或方法声明中添加一个 `except +` 子句。

```python
cdef extern from "some_file.h":
    cdef int foo() except +
```

有了 `except +` 子句，Cython 会自动为我们进行检查，并将异常传播到 Python 代码中。

当前支持的异常及其 Python 对应如下表：

| C++  ( `std::` )    | Python            |
|:------------------- |:----------------- |
| `bad_alloc`         | `MemoryError`     |
| `bad_cast`          | `TypeError`       |
| `bad_typeid`        | `TypeError`       |
| `domain_error`      | `ValueError`      |
| `invalid_argument`  | `ValueError`      |
| `ios_base::failure` | `IOError`         |
| `out_of_range`      | `IndexError`      |
| `overflow_error`    | `OverflowError`   |
| `range_error`       | `ArithmeticError` |
| `underflow_error`   | `ArithmeticError` |
| (all others)        | `RuntimeError`    |

如果有 `what()` 消息，将被保留。

为了指示 Cython 抛出特定类型的 Python 异常，我们可以在 `except +` 子句中添加 Python 异常类型：

```python
cdef int bar() except +MemoryError
```

这将捕获任何 C++ 错误，并用 Python `MemoryError` 替换它。（任何 Python 异常在这里都是有效的。）

还有一个特殊形式：

```cython
cdef int bar() except +*
```


# 类型化内存视图

## 内存视图

Python 的缓冲协议（Buffer Protocol）是一种用于访问对象底层内存数据的机制，它允许 Python 对象将用于存储数据的一块连续内存区域（即缓冲区 Buffer）暴露出来，从而支持高效的内存操作和数据共享。

缓冲协议最重要的特性是其能够以不同的方式表示相同的底层数据。它允许支持缓冲区协议的对象共享相同的数据而无需复制，例如 `numpy.ndarray`、Python 标准库中的 `array.array` 、`cython.array` 等。

内存视图（Memoryview）是 Python 提供的内置类型，用于访问支持缓冲协议的对象。

Cython 提供了C级别的类型化内存视图对象 `memoryview`，它允许你以更高效的方式操作内存。内存视图使用 Python 切片语法，类似于 NumPy：

```python
import numpy as np

arr1D = np.ones((20,), dtype=np.intc)
arr2D = np.ones((20, 15), dtype=np.intc)
arr3D = np.ones((20, 15, 10), dtype=np.intc)
```

{% tabs memoryview %}

<!-- tab Pure Python -->

```python
# Memoryview on a NumPy array
mv1D: cython.int[:] = arr1D        # 1D memoryview
mv2D: cython.int[:, :] = arr2D     # 2D memoryview
mv3D: cython.int[:, :, :] = arr3D  # 3D memoryview
```

<!-- endtab -->

<!-- tab Cython -->

```python
import numpy as np

# Memoryview on a NumPy array
cdef int[:] mv1D = arr1D        # 1D memoryview
cdef int[:, :] mv2D = arr2D     # 2D memoryview
cdef int[:, :, :] mv3D = arr3D  # 2D memoryview
```

<!-- endtab -->

{% endtabs %}

在这里，NumPy 数组和 memoryview 共享内存数据。其中 `int` 指定了内存视图的底层数据类型。

## 函数参数

内存试图也可以方便地作为函数参数使用：

{% tabs memoryview-func %}

<!-- tab Pure Python -->

```python
# A function using a memoryview does not usually need the GIL
@cython.nogil
@cython.ccall
def sum2d(arr: cython.int[:, :]) -> cython.int:
    i: cython.size_t
    j: cython.size_t
    I: cython.size_t
    J: cython.size_t
    total: cython.int = 0
    I = arr.shape[0]
    J = arr.shape[1]
    for i in range(I):
        for j in range(J):
            total += arr[i, j, k]
    return total
```

<!-- endtab -->

<!-- tab Cython -->

```python
# A function using a memoryview does not usually need the GIL
cpdef int sum2d(int[:, :, :] arr) nogil:
    cdef size_t i, j, I, J
    cdef int total = 0
    I = arr.shape[0]
    J = arr.shape[1]
    for i in range(I):
        for j in range(J):
            total += arr[i, j, k]
    return total
```

<!-- endtab -->

{% endtabs %}

当我们从 Python 调用 `sum2d` 时，我们会传递一个 Python 对象，它在函数调用过程中被隐式地赋值给 memoryview 对象。当一个对象被赋值给类型化内存视图时，内存视图会尝试访问该对象的底层数据缓冲区。如果传递的对象无法提供缓冲区（即它不支持该协议），则会引发 `ValueError`。如果它支持该协议，那么它会为内存视图提供一个 C 级别的缓冲区以供使用。

memoryview 对象既支持简单的标量类型，也支持用户定义的结构化类型。

```python
import numpy as np

CUSTOM_DTYPE = np.dtype([
    ('x', np.uint8),
    ('y', np.float32),
])

a = np.zeros(100, dtype=CUSTOM_DTYPE)

cdef packed struct custom_dtype_struct:
    # The struct needs to be packed since by default numpy dtypes aren't aligned
    unsigned char x
    float y

def sum(custom_dtype_struct [:] a):

    cdef:
        unsigned char sum_x = 0
        float sum_y = 0.

    for i in range(a.shape[0]):
        sum_x += a[i].x
        sum_y += a[i].y

    return sum_x, sum_y
```

注意：纯 Python 模式目前不支持打包结构体。

## 索引和切片

我们可以通过类似 NumPy 的方式对类型化内存视图进行索引，以访问和修改单个元素。在 Cython 中，对内存视图的索引访问会自动转换为内存地址。

```python
mv3D[1, 2, 1]
mv3D[1, :, -1]

# These are all equivalent
mv3D[10]
mv3D[10, :, :]
mv3D[10, ...]

# NumPy-style syntax for assigning a single value to all elements.
mv3D[:, :, :] = 3
```

省略号（...）表示获得每个未指定维度的连续切片。

也可以用一个具有相同元素类型且形状正确的另一个内存视图修改。如果左右两侧的形状不匹配，将会引发 `ValueError`。

## 复制数据

内存视图可以直接复制：

{% tabs memoryview-copy %}

<!-- tab Pure Python -->

```python
import numpy as np

def main():

    to_view: cython.int[:, :, :] = np.empty((20, 15, 30), dtype=np.intc)
    from_view: cython.int[:, :, :] = np.ones((20, 15, 30), dtype=np.intc)

    # copy the elements in from_view to to_view
    to_view[...] = from_view
    to_view[:] = from_view
    to_view[:, :, :] = from_view
```

<!-- endtab -->

<!-- tab Cython -->

```python
import numpy as np

cdef int[:, :, :] to_view, from_view
to_view = np.empty((20, 15, 30), dtype=np.intc)
from_view = np.ones((20, 15, 30), dtype=np.intc)

# copy the elements in from_view to to_view
to_view[...] = from_view
to_view[:] = from_view
to_view[:, :, :] = from_view
```

<!-- endtab -->

{% endtabs %}

## C 连续内存布局

最简单的数据布局可能是 C 连续数组。这是 NumPy 和 Cython 数组的默认布局。C 连续意味着数组数据在内存中是连续的，并且数组的第一个维度的相邻元素在内存中相距最远，而最后一个维度的相邻元素在内存中相距最近。例如，在 NumPy 中：

```python
arr = np.array([['0', '1', '2'], ['3', '4', '5']], dtype='S1')
```

这里，`arr[0, 0]` 和 `arr[0, 1]` 在内存中相距一个字节，而 `arr[0, 0]` 和 `arr[1, 0]` 在内存中相距 3 个字节。这引出了 步长 的概念。数组的每个轴都有一个步长，即从该轴的一个元素移动到下一个元素所需的字节数。在上面的例子中，轴 0 和轴 1 的步长分别为：

```python
arr.strides  # (3, 1)
```

声明一个 C 连续的类型化内存视图只需要对最后一个维度使用切片语法 `::1` 来指定。例如，声明一个二维 C 连续的类型化内存视图:

{% tabs memoryview-c-contig %}

<!-- tab Pure Python -->

```python
c_contig_mv: float[:, ::1] = np.ones((3, 4), dtype=np.float32)
```

<!-- endtab -->

<!-- tab Cython -->

```python
cdef float[:, ::1] c_contig_mv = np.ones((3, 4), dtype=np.float32)
```

<!-- endtab -->

{% endtabs %}

## NumPy

Cython 内置了可以访问 C-level 接口 NumPy 包，通过 `cimport` 语句导入

{% tabs memoryview-cnp %}

<!-- tab Pure Python -->

```python
import cython.cimport.numpy as np

arr: double[:, :] = np.zeros((10, 10))
```

<!-- endtab -->

<!-- tab Cython -->

```python
cimport numpy as np

cdef double[:, :] arr = np.zeros((10, 10))
```

<!-- endtab -->

{% endtabs %}

由于我们使用了 NumPy/C API，需要在编译时包含一些 NumPy 头文件。NumPy 提供了一个 `get_include` 函数，返回其头文件目录的完整路径。

```python
from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("cnp", ["cnp.pyx"],
                include_dirs=['.', get_include()])

setup(ext_modules=cythonize(ext))
```

Cython 还可以通过使用装饰器 `@cython.ufunc` 将 C 函数来生成 NumPy ufunc 函数，输入和输出参数类型应该是标量变量。

{% tabs numpy-ufunc %}

<!-- tab Pure Python -->

```python
import cython

@cython.ufunc
@cython.cfunc
def add_one(x: cython.double) -> cython.double:
    # of course, this simple operation can already by done efficiently in Numpy!
    return x+1
```

<!-- endtab -->

<!-- tab Cython -->

```python
cimport cython

@cython.ufunc
cdef double add_one(double x):
    # of course, this simple operation can already by done efficiently in Numpy!
    return x+1
```

<!-- endtab -->

{% endtabs %}

还可以使用 ctuple 类型

{% tabs numpy-ufunc-tuple %}

<!-- tab Pure Python -->

```python
import cython

@cython.ufunc
@cython.cfunc
def add_one_add_two(x: cython.int) -> tuple[cython.int, cython.int]:
    return x+1, x+2
```

<!-- endtab -->

<!-- tab Cython -->

```python
cimport cython

@cython.ufunc
cdef (int, int) add_one_add_two(int x):
    return x+1, x+2
```

<!-- endtab -->

{% endtabs %}

# 多线程并行

Cython允许我们绕过CPython的全局解释器锁，只要我们清晰地将与Python交互的代码与独立于Python的代码分开。做到这一点后，我们可以通过Cython内置的`prange`轻松实现基于线程的并行性。

## nogil

在我们深入探讨`prange`之前，我们必须首先理解CPython全局解释器锁（GIL），用于确保与 Python 解释器相关的数据不会被破坏。在 Cython 中，当不访问 Python 数据时，有时可以释放这个锁。

Cython提供了两种机制来管理 GIL：标记 `nogil` 函数属性和 `with nogil`上下文管理器。

### 标记 `nogil` 函数属性

要在没有GIL的上下文中调用一个函数，该函数必须具有`nogil`属性。这种函数必须是来自外部库的，或者是用C函数或混合函数。`def`函数不能在没有GIL的情况下被调用，因为这些函数总是与Python对象交互。

{% tabs nogil-func %}

<!-- tab Pure Python -->

使用 `@cython.nogil` 装饰器将整个函数（无论是 Cython 函数还是外部函数）标记为 `nogil`：

```python
@cython.nogil
@cython.cfunc
@cython.noexcept
def kernel() -> None:
    ...
```

<!-- endtab -->

<!-- tab Cython -->

通过在函数签名后添加 `nogil` 将整个函数（无论是 Cython 函数还是外部函数）标记为 `nogil`：

```python
cdef void kernel() noexcept nogil:
    ....
```

<!-- endtab -->

{% endtabs %}

在`kernel`函数体中，我们不能创建或以其他方式与Python对象交互，包括静态类型的Python对象，如`list`或`dict`。

请注意，这并不会在调用函数时释放 GIL。它只是表明该函数适合在释放 GIL 的情况下使用。在持有 GIL 的情况下调用这些函数也是可以的。

在本例中，我们将函数标记为 `noexcept`，以表明它不会引发 Python 异常。请注意，具有 `except *` 异常规范的函数（通常是返回 `void` 的函数）调用成本较高，因为 Cython 需要在每次调用后暂时重新获取 GIL 以检查异常状态。在 `nogil` 块中，大多数其他异常规范的处理成本较低，因为只有在实际抛出异常时才会获取 GIL。

### 上下文管理器

在 Cython 中可以通过 `with nogil` 上下文管理来实际释放 GIL。

```python
cpdef int func(int a, int b) nogil except? -1:
    return a / b

cdef int res
with nogil:
    res = func(22, 33)

print(res) 
```

在这个代码片段中，我们使用 `with nogil`上下文管理器在调用 `func`之前释放GIL，并在退出上下文管理器块后重新获取它。

如果在 `with nogil` 里面如果出现了函数调用，那么该函数必须是使用 `nogil` 声明的函数。而使用 `nogil` 声明的函数，其内部必须是纯 C 操作、不涉及 Python。

通常，一个外部库根本不会与Python对象交互。在这种情况下，我们可以在`cdef extern from`行中放置 `nogil` 声明，从而将 `extern` 块中的每个函数都声明为 `nogil`：

```python
cdef extern from "math.h" nogil:
    double sin(double x)
    double cos(double x)
    double tan(double x)
```

我们还可以在 `with nogil`上下文中使用 `with gil` 子上下文暂时重新获取GIL。这允许一个`nogil`函数重新获取GIL以执行涉及Python对象的操作。

{% tabs with-gil %}

<!-- tab Pure Python -->

```python
with cython.nogil:
    ...              # some code that runs without the GIL
    with cython.gil:
        ...          # some code that runs with the GIL
    ...              # some more code without the GIL
```

也可以通过使用 `@cython.with_gil` 装饰器，确保在调用函数时立即获取 GIL。

```python
@cython.with_gil
@cython.cfunc
def some_func() -> cython.int
    ...

with cython.nogil:
    ...          # some code that runs without the GIL
    some_func()  # some_func() will internally acquire the GIL
    ...          # some code that runs without the GIL
some_func()      # GIL is already held hence the function does not need to acquire the GIL
```

<!-- endtab -->

<!-- tab Cython -->

```python
with nogil:
    ...      # some code that runs without the GIL
    with gil:
        ...  # some code that runs with the GIL
    ...      # some more code without the GIL
```

也可以通过将函数标记为 `with gil` ，确保在调用函数时立即获取 GIL。

```python
cdef int some_func() with gil:
    ...

with nogil:
    ...          # some code that runs without the GIL
    some_func()  # some_func() will internally acquire the GIL
    ...          # some code that runs without the GIL
some_func()      # GIL is already held hence the function does not need to acquire the GIL
```

<!-- endtab -->

{% endtabs %}

### 条件性地获取 GIL

融合类型函数可能需要处理 Cython 原生类型（例如 `cython.int` 或 `cython.double`）和 Python 类型（例如 `object` 或 `bytes`）。条件性获取/释放 GIL 提供了一种方法，可以在运行相同的代码时，根据需要释放 GIL（针对 Cython 原生类型）或持有 GIL（针对 Python 类型）：

{% tabs fused_type_gil %}

<!-- tab Pure Python -->

```python
import cython

double_or_object = cython.fused_type(cython.double, object)

def increment(x: double_or_object):
    with cython.nogil(double_or_object is not object):
        # Same code handles both cython.double (GIL is released)
        # and python object (GIL is not released).
        x = x + 1
    return x

increment(5.0)  # GIL is released during increment
increment(5)    # GIL is acquired during increment
```

<!-- endtab -->

<!-- tab Cython -->

```python
cimport cython

ctypedef fused double_or_object:
    double
    object

def increment(double_or_object x):
    with nogil(double_or_object is not object):
        # Same code handles both cython.double (GIL is released)
        # and python object (GIL is not released).
        x = x + 1
    return x

increment(5.0)  # GIL is released during increment
increment(5)    # GIL is acquired during increment
```

<!-- endtab -->

{% endtabs %}

### 异常和 GIL

在 `nogil` 块中可以执行少量的Python 操作，而无需显式使用 `with gil`。主要例子是抛出异常。在这里，Cython 知道异常总是需要 GIL，因此会隐式地重新获取它。同样，如果一个 `nogil` 函数抛出异常，Cython 能够正确地传播它，而无需你编写显式的代码来处理它。在大多数情况下，这是高效的，因为 Cython 可以使用函数的异常规范来检查错误，然后只有在需要时才获取 GIL，但 `except *` 函数的效率较低，因为 Cython 必须始终重新获取 GIL。

## prange

Cython通过OpenMP API实现`prange`，用于原生并行化。`prange`是一个仅在Cython中存在的特殊函数。它可以轻松地帮我们将普通的 for 循环转成使用多个线程的循环，接入所有可用的 CPU 核心。 

```python
cython.parallel.prange(start=0, stop=None, step=1, 
    nogil=False, use_threads_if=True, schedule=None, 
    chunksize=None, num_threads=None)
```

- `start, stop, step` 参数和 `range` 的用法一样
- `nogil` 用来打开 GIL。该函数只能在释放 GIL 的情况下使用。
- `use_threads_if` 是否启用并行
- `schedule`：传递给 OpenMP，用于线程分配
  - static：整个循环在编译时会以一种固定的方式分配给多个线程，如果 chunksize 没有指定，那么会分成 num_threads 个连续块，一个线程一个块。如果指定了 chunksize，那么每一块会以轮询调度算法（Round Robin）交给线程进行处理，适用于任务均匀分布的情况。
  - dynamic：线程在运行时动态地向调度器申请下一个块，chunksize 默认为 1，当任务负载不均时，动态调度是最佳的选择。
  - guided：块是动态分布的，但与 dynamic 不同，chunksize 的比例不是固定的，而是和 剩余迭代次数 / 线程数 成比例关系。
  - runtime：调度策略和块大小将从运行时调度变量中获取，该变量可以通过 `openmp.omp_set_schedule()` 函数调用或 `OMP_SCHEDULE` 环境变量设置。这允许在不重新编译的情况下探索不同的`schedule`和`chunksize`，但可能会由于没有编译时优化而导致整体性能较差。

`prange` 只能与 for 循环搭配使用，不能独立存在。变量的线程局部性和归约操作会自动推断。

**规约并行**

{% tabs prange-reduce %}

<!-- tab Pure Python -->

```python
from cython.parallel import prange

def func(x: cython.double[:], alpha: cython.double):
    i: cython.Py_ssize_t

    for i in prange(x.shape[0], nogil=True):
        x[i] = alpha * x[i]
```
<!-- endtab -->

<!-- tab Cython -->

```python
from cython.parallel import prange

cdef int i
cdef int n = 30
cdef int sum = 0

for i in prange(n, nogil=True):
    sum += i

print(sum)
```
<!-- endtab -->

{% endtabs %}

**内存视图并行**

{% tabs prange-mv %}

<!-- tab Pure Python -->

```python
from cython.parallel import prange

def func(x: cython.double[:], alpha: cython.double):
    i: cython.Py_ssize_t

    for i in prange(x.shape[0], nogil=True):
        x[i] = alpha * x[i]
```
<!-- endtab -->

<!-- tab Cython -->

```python
from cython.parallel import prange

def func(double[:] x, double alpha):
    cdef Py_ssize_t i

    for i in prange(x.shape[0], nogil=True):
        x[i] = alpha * x[i]
```
<!-- endtab -->

{% endtabs %}

**条件并行**

{% tabs prange-if %}

<!-- tab Pure Python -->

```python
from cython.parallel import prange

def psum(n: cython.int):

    i: cython.int
    sum: cython.int = 0

    for i in prange(n, nogil=True, use_threads_if=n>1000):
        sum += i

    return sum

psum(30)        # Executed sequentially
psum(10000)     # Executed in parallel
```
<!-- endtab -->

<!-- tab Cython -->

```python
from cython.parallel import prange

def psum(int n):

    cdef int i
    cdef int sum = 0

    for i in prange(n, nogil=True, use_threads_if=n>1000):
        sum += i

    return sum

psum(30)        # Executed sequentially
psum(10000)     # Executed in parallel
```
<!-- endtab -->

{% endtabs %}

一旦使用了 `prange`，那么必须确保在编译的时候启用 OpenMP。对于 gcc，可以在 `setup.py` 中如下操作：

```python
from setuptools import Extension, setup
from Cython.Build import cythonize
import sys

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'


ext_modules = [
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    ),
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    )
]

setup(
    name='parallel-tutorial',
    ext_modules=cythonize(ext_modules),
)
```

而在 Cython 源文件中我们可以通过注释的方式指定

```python
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
```
