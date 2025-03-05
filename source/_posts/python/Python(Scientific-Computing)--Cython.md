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
date: 2025-03-05 00:32:31
description: 旨在融合 Python 的易用性和 C 语言的高性能
---

Cython 是一个编程语言和编译器，旨在融合 Python 的易用性和 C 语言的高性能。它的主要功能是允许在 Python 代码中使用静态类型声明。它通过将 Python 代码转换为优化的 C / C ++代码，从而显著提升 Python 程序的运行速度。 —— [官方网站](https://Cython.readthedocs.io/en/latest/)

**安装 Cython**

```bash
pip install cython
```

在终端中输入 `Cython -V` 查看是否安装成功。

# 开发流程

**编写 Cython 代码**：先创建一个扩展名为 `.pyx` 的 Cython 源文件 `fibonacci.pyx` 

```python
# fibonacci.pyx
def fib(n):
    cdef int i
    cdef double a = 0.0, b = 1.0
    for i in range(n):
        a, b = a + b, a
    return a
```

**创建setup.py文件**：用于编译 Cython 代码。

```python
# setup.py
from setuptools import Extension, setup
from Cython.Build import cythonize

ext = Extension(
    name='fibonacci',  # if use path, seperate with '.'.
    sources=['fibonacci.pyx'],  # like .c files in c/c++.
    language='c',  # c or c++.
)
setup(ext_modules=cythonize(ext, annotate=True, language_level=3))
```

构建扩展模块的过程分为三步：

- 首先 `Extension` 对象负责配置：name 表示编译之后的文件名，sources 则是代表 pyx 和 c/c++ 源文件列表；

- 然后 `cythonize` 负责将 Cython 代码转成 C 代码，参数 `language_level=3` 表示只需要兼容 python3 即可，而默认是 2 和 3 都兼容；`annotate`参数显示 Cython 的代码分析。

- 最后 `setup` 根据 c/c++ 代码生成扩展模块

**编译 Cython 代码**：在命令行中运行以下命令进行编译，即可生成相应的 Python 扩展模块

```bash
python setup.py build_ext --inplace
```

编译后会多出一个 `.pyd` 文件或 `.so` 文件，这就是根据 `fibonacci.pyx` 生成的扩展模块，至于其它的可以直接删掉了。

**调用编译后的模块**：在 Python 中导入并使用该模块。

```python
# example.py
import fibonacci
result = fibonacci.fib(10)
```

我们看到扩展模块可以被正常使用了。

**在 jupyter notbook使用**：要启用对 Cython 编译的支持，需要先加载 `Cython`扩展：

```python
In [1]: %load_ext cython
```

然后，使用`%%cython`标记为单元格添加前缀以进行编译：

```python
In [2]: %%cython -a
...: def fib(int n):
...:     cdef int i
...:     cdef double a = 0.0, b = 1.0
...:     for i in range(n):
...:         a, b = a + b, a
...:     return a
```

上例中通过传递`--annotate`选项（简写为 `-a`）来显示 Cython 的代码分析。

**Cython 最佳实践**

1. **类型声明**：对循环变量和数值计算变量强制静态类型
2. **避免Python对象**：在核心算法中减少Python对象操作
3. **内存管理**：使用`memoryview`替代Python列表操作
4. **混合编程**：将性能关键部分用Cython实现，其他保持Python

# 静态类型声明

## 变量声明

在Cython中，通过`cdef`关键字来声明静态变量，同时允许静态类型变量和python动态类型变量的混合使用

```python
cdef int x, y
cdef double pi = 3.14159
t = (x, y)
```

也可以使用Python风格的缩进进行声明

```python
cdef:
    int x, y
    float k = 0.0
```

支持指针、结构体、枚举等C语言的数据类型

```python
cdef double a = 3.14
cdef double* p = &a
print(p[0])

cdef struct Point:
    int x, y

cdef Point p1 = Point(1, 2)
cdef Point p2 = {"x": 1, "y": 2}

cdef enum Color:
    RED = 1
    YELLOW = 2
    GREEN = 3
```

注意：在 Cython 中通过 `p[0]` 的方式获取指针变量的值。因为，在Python 中，`*` 有特殊含义，无法像 C 中那样使用。

Cython 同样支持内置的python的数据类型进行静态声明

```python
cdef str a = "hello"
cdef tuple t = (1, 2, 3)
cdef list lst = []
lst.append(1)
```

Cython 提供了一个 Python 元组的有效替代品 `ctuple`。 `ctuple`由任何有效的 C 类型组装而成。例如：

```python
cdef (double, int) bar
```

Cython 允许 cdef 定义的c变量赋值给python动态变量，同时会自动转换类型

```python
cdef int num = 6
a = num
```

类型的对应关系如下：

- `bool` - `bint`

- `int` - `short, int, long, char`

- `float` - `float, double`

- `bytes` - `char*`

- `str` - `std::string`

- `dict` - `struct`

## 函数

Cython中的函数支持三种修饰类型：

- `def`函数 - 编译的Python原生函数。支持外部文件通过 import 语句直接调用。

- `cdef`函数 - C-level 级函数，其参数和返回值都要求指定明确的类型。需要在 Cython 中通过 `def` 函数包装下才能被python模块识别。`

- `cpdef`混合函数 - 相当于定义了 Python 包装器的C函数。但 `cpdef` 定义的函数需要同时兼容 Python 和 C。支持文件直接调用。

> 在 Cython 模块中，`def`函数、`cdef`函数和`cpdef`函数可以自由地相互调用。

```python
# example.pyx
def greet(str name = "Alice"):
    print("Hello, " + name)

cdef int _add(int a, int b):
    return a + b

def add(a, b):
    return _add(a, b)

cpdef int divide(int a, int b):
    return a / b
```

不管`cdef` 和 `cpdef` 定义的函数都无法捕获异常（比如 ZeroDivisionError），这导致程序不会报错停止。Cython 提供了一个 except 字句，允许捕获和抛出异常。

```python
# example.pyx
cpdef int divide(int a, int b) except? -1:
    return a / b
```

## 泛型

Cython 定义了一个融合类型，允许我们用一个类型来引用多个类型

- `integral`：代指C中的 short、int、long
- `floating`：代指C中的 float、double
- `numeric`：最通用的类型，包含上面的integral和floating以及复数

```python
from Cython cimport integral

cpdef integral integral_max(integral a, integral b):
    return a if a >= b else b 
```

Cython 支持通过 `ctypedef fused` 定义一个融合类型，支持的类型可以写在块里面

```python
ctypedef fused sequence:
    tuple
    list
    dict
```

# 扩展类型

## 静态属性

Cython 支持通过 `cdef class` 使用 Python/C API 定义一个C级别的类，称为扩展类。扩展类是可以被外部文件访问的。

```python
# example.pyx
cdef class Rectangle:
    cdef int width, height

    def __init__(self, w, h):
        self.width = w
        self.height = h

    def describe(self):
        print("This shrubbery is", self.width,
              "by", self.height, "cubits.")
```

注意：我们在 `__init__` 中实例化的属性，都必须在类中先声明。且声明的类属性默认是私有的，不能被外部访问和修改。若要想让python代码访问，可以声明为 `readonly` 或 `public`。

```python
cdef class Shrubbery:
    cdef public int width, height
    cdef readonly float depth
```

默认情况下，无法在运行时向扩展类型添加属性。

## 构造函数和析构函数

Cython 扩充了构造函数`__cinit__` 和析构函数 `__dealloc__`，用于执行 C 级别的内存分配和释放。

```python
# example.pyx
from libc.stdlib cimport malloc, free

cdef class Rectangle:
    cdef public:
        long width, height, n
    cdef double *array  

    def __cinit__(self, *args, **kwargs):
        self.n = n
        self.array = <double *>malloc(n * sizeof(double))

    def __dealloc__(self):
        free(self.array)

    def __init__(self, w, h):
        self.width = w
        self.height = h
```

`__cinit__` 和 `__init__` 只能通过 def 来定义，在不涉及 malloc 动态分配内存的时候， 他们是等价的。我们实例化一个扩展类的时候，参数会先传递给`__cinit__`，然后`__cinit__`再将接收到的参数原封不动的传递给`__init__`。

相比 C 的动态内存管理函数，Python 在 malloc、realloc、free 基础上做了一些简单的封装，这些函数对较小的内存块进行了优化，通过避免昂贵的操作系统调用来加快分配速度。

```python
# example.pyx
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class Rectangle:
    cdef long n
    cdef double *array  

    def __cinit__(self, n):
        self.n = n
        self.array = <double *> PyMem_Malloc(sizeof(double) * number)

    def resize(self, size_t new_number):
        mem = <double *> PyMem_Realloc(self.data, sizeof(double) * new_number)
        self.data = mem

    def __dealloc__(self):
        PyMem_Free(self.array)
```

## 类方法

和函数一样，可以用`cdef` 和 `cpdef` 定义类方法，但只能用于 `cdef class` 定义的静态类。

```python
# example.pyx
cdef class Rectangle:

    cdef long width, height

    def __init__(self, w, h):
        self.width = w
        self.height = h

    cdef long get_area(self):
        return self.width * self.height
```

## 继承

扩展类只能继承单个基类，并且继承的基类必须是直接指向 C 实现的类型，可以是使用 cdef class 定义的扩展类型，也可以是内置类型，因为内置类型也是直接指向 C 一级的结构。

```python
# example.pyx
cdef class Girl:
    cdef public:
        str name
        long age

    def __init__(self, name, age):
        self.name = name
        self.age = age

    cpdef str get_info(self):
        return f"name: {self.name}, age: {self.age}"


cdef class CGirl(Girl):
    cdef public str where

    def __init__(self, name, age, where):
        self.where = where
        super().__init__(name, age)


class PyGirl(Girl):
    def __init__(self, name, age, where):
        self.where = where
        super().__init__(name, age)
```

扩展类不可以继承 Python 类，但 Python 类是可以继承扩展类的。但是 cdef 定义的方法，即使在 Cython 环境中，Python 类也是不能直接继承的。但是我们可以通过`<>` 类型转化，从而间接使用。

```python
class PyGirl(Girl):
    def __init__(self, name, age, where):
        self.where = where
        super().__init__(name, age)

    def get_info2(self):
        return (<Girl?> self).get_info()
```

## 魔法方法

函数名以双下划线开头和结尾的方法称为魔法方法，通过魔法方法可以对运算符进行重载等。

```python
cdef class Iter:
    cdef public:
        list values
        long __index

    def __init__(self, values):
        self.values = values
        self.__index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            ret = self.values[self.__index]
            self.__index += 1
            return ret
        except IndexError:
            raise StopIteration
```

注意：魔法方法只能用def定义，不可以使用cdef或者cpdef。

# 模块管理

## 动态链接

对于两个 pyx 模块，他们无法通过 import 互相访问。但我们可以像 C 一样使用 include 将别的 pyx 文件导入当前的 pyx 文件，就相当于在当前文件中定义的一样。

```python
# udlib.pyx
cdef PI = 3.14
```

```python
# example.pyx
include "./udlib.pyx"

double r = 5.0
print(PI * r ** 2)
```

除此之外，Cython 还提供了`pxd`文件来组织 Cython 文件以及 C 文件。`pxd` 文件类似于 C 中的头文件，用于存放一些声明，而它们的具体实现是在同名的 `pyx` 文件中。

```python
# example.pyx
cdef class Girl:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    cpdef str get_info(self):
        return f"name: {self.name}, age: {self.age}"


# udlib.pxd
cdef class Girl:
    cdef public :
        str name  
        long age  
        str gender  

    cpdef list get_score(self)
```

然后我们可以另一个 pyx 文件中，使用`cimport`关键字将`pxd`文件导入。`cimport` 和 `import` 语法一致，但是 `cimport` 可以用来导入静态声明的对象。

```python
# example.pyx
from udlib cimport Girl

girl = Girl("Alice", 18)
```

如果我们在 pxd 文件中声明了一个函数或者变量，那么在 pyx 文件中不可以再次声明，否则会发生编译错误。

## 标准库

Cython 可以直接访问和使用 Python 的标准库，如`os`、`sys`、`re` 等

```python
import os
```

同时还支持位于 libc 目录下的 c 标准库，如`stdio`、`math`、`string`等

```python
from libc.math cimport sin, cos

result = sin(0.5)
```

通过设置 `language_level=3` 和 `language=c++`，Cython 可以编译为 C++ 代码，从而支持 C++ 标准库，如 `string`、`vector`、`list`、`map`、`pair`、`set`等

```python
# distutils: language=c++
from libcpp.vector cimport vector

cdef vector[int] *vec = new vector[int](3)
```

注意：如果想要编译成功，那么必须要在开头加上注释 `# distutils: language=c++`。

## 外部声明

Cython 提供了一个 extern 语句，可以直接调用 C/C++ 源码。例如，有一个头文件 `fibonacci.h`，里面是函数声明；一个源文件 `fibonacci.c`，里面是函数实现

```c
// fibonacci.c
double cfib(int n) {
    int i;
    double a=0.0, b=1.0, tmp;
    for (i=0; i<n; ++i) {
        tmp = a; a = a + b; b = tmp;
    }
     return a;
}
```

```c
// fibonacci.h
double cfib(int n);
```

通过 `cdef extern from` 导入头文件，并声明函数`cfib`，然后 Cython 可以直接调用。如果头文件不在同一个目录中, 那么编译的时候还需要通过 `include_dirs` 参数指定头文件的所在目录。

```python
# example.pyx
cdef extern from "fibonacci.h":
    double cfib(int n)

def fib(n):
    """Returns the nth Fibonacci number."""
    return cfib(n)
```

我们仍然需要在 Cython 中使用 def、或者 cpdef 将 extern 块中声明的 C 级结构包装一下才能给 Python 调用。

# 内存视图

Cython 提供了内存视图对象 `memoryview`，它允许你以更高效的方式操作内存。以下是如何声明整数的内存视图：

```python
cdef int [:] foo         # 1D memoryview
cdef int [:, :] foo      # 2D memoryview
cdef int [:, :, :] foo   # 3D memoryview
```

NumPy给Cython留了调用的c-level的接口，使用cimport可以在Cython中导入c模块从而加快程序运行的速度。

```python
cimport numpy as np

cdef double[:, :] array = np.zeros((10, 10))
```



{% tabs memoryview %}

<!-- tab Pure Python -->

```python
import cython
@cython.boundscheck(False)
@cython.wraparound(False)
def compute(array_1: cython.int[:, ::1]):
    # get the maximum dimensions of the array
    x_max: cython.size_t = array_1.shape[0]
    y_max: cython.size_t = array_1.shape[1]
    
    #create a memoryview
    view2d: int[:,:] = array_1

    # access the memoryview by way of our constrained indexes
    for x in range(x_max):
        for y in range(y_max):
            view2d[x,y] = something()
```

<!-- endtab -->

<!-- tab Cython -->

```python
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def compute(int[:, ::1] array_1):
    # get the maximum dimensions of the array
    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]
    
    #create a memoryview
    cdef int[:, :] view2d = array_1

    # access the memoryview by way of our constrained indexes
    for x in range(x_max):
        for y in range(y_max):
            view2d[x,y] = something()
```

<!-- endtab -->

{% endtabs %}

# Cython装饰器

Cython提供了装饰器，可以在Python函数上关闭边界检查和环绕检查。

```python
from Cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
def summer(double[:] numbers):
    cdef double s = 0
    cdef int i, N
    N = numbers.shape[0]
    for i in range(N):
        s += numbers[i]
    return s
```

如果想关闭全局的边界检测，那么可以使用注释的形式。

```python
# Cython: boundscheck=False
# Cython: wraparound=False
```

# 多线程并行

Cython支持OpenMP，这使得在多核处理器上执行并行计算变得简单。

## nogil

在 Cython 中可以通过 `with nogil` 上下文管理来利用多核。进入 `nogil` 上下文，释放 GIL，独立执行，完事了再获取 GIL 退出上下文。

```python
cpdef int func(int a, int b) nogil except ? -1:
    return a / b

cdef int res
with nogil:
    res = func(22, 33)

print(res) 
```

如果在 `with nogil`  里面如果出现了函数调用，那么该函数必须是使用 `nogil` 声明的函数。而使用 `nogil` 声明的函数，其内部必须是纯 C 操作、不涉及 Python。因此当我们需要执行一个耗时的纯 C 函数，那么便可以将其申明为 nogil 函数，然后通过 with nogil 的方式实现并行执行。

## prange

在 Cython 中有一个 `prange` 函数，它可以轻松地帮我们将普通的 for 循环转成使用多个线程的循环，接入所有可用的 CPU 核心。**且 prange 只能与 for 循环搭配使用，不能独立存在。**

```python
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
from Cython cimport boundscheck, wraparound
from Cython.parallel cimport prange

@boundscheck(False)
@wraparound(False)
def calc_julia(int[:, :: 1] counts, int val):
    cdef:
        int total = 0
        int i, j, M, N
    N, M = counts.shape[: 2]
    for i in prange(M, nogil=True):
        for j in range(N):
            if counts[i, j] == val:
                total += 1

    return total / float(counts.size)
```

**如果是交给 Numpy 来做的话，那么等价于如下：**

```python
np.sum(counts == val) / float(counts.size)
```

一旦使用了 prange，那么必须确保在编译的时候启用 OpenMP，像 gcc 这样的编译器的链接标志是 -fopenmp，而在 Cython 中我们可以通过注释的方式指定。

```python
prange(start=0, stop=None, step=1, nogil=False, schedule=None, 
    chunksize=None, num_threads=None)
```

- `start, stop, step` 参数和 range 的用法一样

- `nogil` 用来打开 GIL

- `schedule`：用于线程分配
  
  - static：整个循环在编译时会以一种固定的方式分配给多个线程，如果 chunksize 没有指定，那么会分成 num_threads 个连续块，一个线程一个块。如果指定了 chunksize，那么每一块会以轮询调度算法（Round Robin）交给线程进行处理，适用于任务均匀分布的情况。
  
  - dynamic：线程在运行时动态地向调度器申请下一个块，chunksize 默认为 1，当任务负载不均时，动态调度是最佳的选择。
  
  - guided：块是动态分布的，但与 dynamic 不同，chunksize 的比例不是固定的，而是和 剩余迭代次数 / 线程数 成比例关系。
