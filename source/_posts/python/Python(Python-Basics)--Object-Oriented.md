---
title: Python(Python Basics)--Python面向对象
tags:
  - Python
  - 面向对象
categories:
  - Python
  - General
cover: /img/Object-Oriented-in-Python.jpg
abbrlink: 79a9fda5
date: 2018-06-23 20:12:35
top_img: /img/python-top-img.svg
---

Python从设计之初就已经是一门面向对象(Object Oriented,OO)的语言，正因为如此，在Python中创建一个类和对象是很容易的。本章节我们将详细介绍Python的面向对象编程。

<!-- more -->

# 面向对象

类把数据与功能绑定在一起。创建新类就是创建新的对象 **类型**，从而创建该类型的新 **实例** 。类实例支持维持自身状态的属性，还支持（由类定义的）修改自身状态的方法。

- **类(Class)**: 用来描述具有相同的属性和方法的对象的集合。对象是类的实例。

- **属性和方法**：类中定义的变量和函数。加双下划线`__`私有化属性和方法，`__private_attrs`,`__private_method`

- **继承**：派生类（derived class）自动共享基类（base class）数据结构和方法的机制，这是类之间的一种关系。python支持多继承性。

- **多态**：指相同的操作或函数、过程可作用于多种类型的对象上并获得不同的结果。不同的对象，收到同一消息可以产生不同的结果，这种现象称为多态性。

- `dir(object)`：输出类的属性列表

- **访问对象的属性和方法**

  `obj.some_method(args)`
  `obj.attribute_name`

[面向对象参考链接](http://www.runoob.com/python3/python3-class.html)


# 定义类

> 定义属性，方法，方法装饰器等

```python
class ClassName:
    <statement-1>
    ...
    <statement-N>
```

示例：

```python
class ClassName:

    '''类文档''' # __doc__属性可返回文档字符串

	cls_attr=0  # 类本身的属性，所有实例均可调用
	
    def __init__(self,...): # 初始化（需要传递初始化参数）
    	self.attr=1   # 初始化实例属性
    	self._x='read only'
    	
    def method(self,...): # 方法，第一个参数self代表实例
    	pass
    	
 	@staticmethod  #静态方法（纯粹定义在类内的函数），无self参数，无法访问类属性及方法
	def static_fun(...): #不需要实例化，可直接调用，实例化后也可调用
		pass
		
	@classmethod   # 定义类的方法，可直接调用，不需要实例化
	def cls_fun(cls,...):  #第一个参数cls代表类本身（可调用及修改类本身）
		pass
		
	@property # 很容易的操作只读属性
	def get_x(self):
		"""Get the current voltage."""
		return self._x
	#property 的 getter,setter 和 deleter 方法同样可以用作装饰器
	# @x.setter,@x.deleter	
```

调用类
```python
ClassName.cls_attr # 调用类属性
ClassName.static_fun(...) #调用静态方法
ClassName.cls_fun(...) #调用类方法

x = ClassName(...)  #实例化类
x.attr #访问属性
x.method(...) #调用方法
super(base,x).method(...) #调用父类的方法
```

# 内建函数
```python
getattr(obj, name[, default]) #访问对象的属性。
hasattr(obj,name) #检查是否存在一个属性。
setattr(obj,name,value) #设置一个属性。如果属性不存在，会创建一个新属性。
delattr(obj, name)  #删除属性。

type(object) #判断类别，不考虑继承关系
isinstance(object, classinfo) #判断类别，考虑继承关系，classinfo可以传递tuple
issubclass(class, classinfo)

dir(np) # 列出模块的方法和属性
```

# 继承
```python
class DerivedClassName(Base1, Base2, ...):
    <statement-1>
    ...
    <statement-N>
```

# 面向对象实例
```python
#!/usr/bin/python3

# ---------类定义---------
class People:
    count = 0  # 所有子类通用属性

    def __init__(self, name, age, weight):     # 定义构造方法（初始化）
        #用类名来定义类属性
        People.count+=1  #每次初始执行，可以用来统计子类数量
        
        #self代表实例而不是类
        self.name = name
        self._age = age  # 私有属性，可以访问
        self.__weight = weight  # 私有属性，不能直接访问

    @classmethod  # 定义类的方法，可直接调用，不需要实例化
    def get_sex(cls):  # cls代表类本身
        return cls.sex

    @property  
    def get_name(self):
        return self.name

    def speak(self):
        print("My name is {:s}, my age is {:d} .".format(self.name, self._age))
        
#------实例化
Tim = People("Tim", 25, 80)  
print(Tim._age)      #25 
print(Tim.get_name)  #'Tim'
print(Tim.speak())   #'My name is Tim, my age is 25 .'

print('count:'+str(People.count))   # count: 1

#-----添加，修改，删除属性
Tim.sex = 'male'  # 添加一个sex属性
Tim.age = 8  # 修改age属性
del Tim.sex  # 删除sex属性

# ---------继承---------
class Student(People):

    #子类不重写 __init__，实例化子类时，会自动调用父类定义的 __init__
    def __init__(self, name, age, weight, grade):
        # 调用父类的构造函数（多继承的话也需要调用其他父类的构造函数）
        People.__init__(self, name=name, age=age, weight=weirht)  
        self.grade = grade

    def speak(self):  # 覆写父类的方法
        print("My name is {:s}, my age is {:d},I'm a student.".format(self.name, self._age))
 
#-----------实例化
print(dir(People)
Lucy = Student("Lucy", 18, 45, 3)
print(Lucy.name)      #'Lucy'
print(Lucy.get_name)  #'Lucy'
print(Lucy.speak())   #'My name is Lucy, my age is 18,I'm a student.'
print(super(People,Lucy).speak())  #My name is Lucy, my age is 18 .
```
# 类的内置属性

内置属性|说明
:--------|:--------
`__dict__` :|类的属性（包含一个字典，由类的数据属性组成）
`__doc__` |类的文档（字符串）
`__name__`|类名
`__module__`:|类定义所在的模块（类的全名是`__main__.className`，如果类位于一个导入模块mymod中，那么`className.__module__` 等于 mymod）
`__bases__` :|类的所有父类构成元素（包含了一个由所有父类组成的元组）

# 类的魔术方法

魔术方法|说明|调用方法
--|:--|:--
`__init__(self,[,args...] )`|构造函数，在生成对象时调用| obj = className(args)
`__del__(self)`|析构函数，释放对象时使用|del obj
`__dir__(self)`|控制`dir()`输出|dir(obj)
`__setitem__(self)`|按照索引赋值|
`__getitem__(self)`| 按照索引获取值|
`__len__(self)`| 获得长度|len(obj)
`__call__(self)`| 函数调用|
`__name__`| 对象名字 |
`__file__`| 指向该对象的导入文件名 |
**转换成字符串**||
`__repr__(self)`| 打印，转换| repr(obj)
`__str__(self)`|转换成字符，print对象时会调用此方法|str(obj)
`__unicode__(self)`||

**算术运算符**||
--|:--|
`__add__`| +|
`__sub__`| -|
`__mul__`| \*|
`__div__`| /|
`__mod__`| %|
`__pow__`| \**|
**比较运算符**||
`__cmp__`|比较运算|
`__eq__`|==|
`__lt__`|>|
`__gt__`|<|
**逻辑运算符**||
`__and__`|and|
`__or__`|or|


```python
#!/usr/bin/python3

class num:

    def __init__(self,num):
        self.num=num

    def __add__(self,other):
        if isinstance(other,num):
            return self.num+other.num
        else:
            raise Exception("The type of object must be num")


    def __dir__(self):
        return self.__dict__.keys()

print(num(10)+num(2))   #12
print(num(10)==num(2))  #False
print(dir(num(2)))      #['num']
```

# 迭代器

 迭代器的使用非常普遍并使得 Python 成为一个统一的整体。 在幕后，for 语句会在容器对象上调用 `iter()`。 该函数返回一个定义了 `__next__() `方法的迭代器对象，此方法将逐一访问容器中的元素。 当元素用尽时，`__next__()` 将引发 StopIteration 异常来通知终止 for 循环。 你可以使用 `next()` 内置函数来调用 `__next__() `方法；这个例子显示了它的运作方式:

```python
>>> s = 'abc'
>>> it = iter(s)
>>> it
<str_iterator object at 0x10c90e650>
>>> next(it)
'a'
>>> next(it)
'b'
>>> next(it)
'c'
>>> next(it)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    next(it)
StopIteration
```

看过迭代器协议的幕后机制，给你的类添加迭代器行为就很容易了。 定义一个 `__iter__()` 方法来返回一个带有 `__next__()`方法的对象。 如果类已定义了 `__next__()`，则 `__iter__()` 可以简单地返回 `self`:

```python
class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]
```

```python
>>> rev = Reverse('spam')
>>> iter(rev)
<__main__.Reverse object at 0x00A1DB50>
>>> for char in rev:
...     print(char)
...
m
a
p
s
```

# 生成器

生成器是一个用于创建迭代器的简单而强大的工具。 它们的写法类似于标准的函数，但当它们要返回数据时会使用 `yield` 语句。 每次在生成器上调用 `next()` 时，它会从上次离开的位置恢复执行（它会记住上次执行语句时的所有数据值）。 一个显示如何非常容易地创建生成器的示例如下:

```python
def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]
```

```python
>>> for char in reverse('golf'):
...     print(char)
...
f
l
o
g
```

可以用生成器来完成的操作同样可以用前一节所描述的基于类的迭代器来完成。 但生成器的写法更为紧凑，因为它会自动创建 `__iter__()` 和 `__next__()`方法。

另一个关键特性在于局部变量和执行状态会在每次调用之间自动保存。 这使得该函数相比使用 `self.index` 和 `self.data` 这种实例变量的方式更易编写且更为清晰。

除了会自动创建方法和保存程序状态，当生成器终结时，它们还会自动引发 `StopIteration`。 这些特性结合在一起，使得创建迭代器能与编写常规函数一样容易。

# 生成器表达式

某些简单的生成器可以写成简洁的表达式代码，所用语法类似列表推导式，但外层为圆括号而非方括号。 这种表达式被设计用于生成器将立即被外层函数所使用的情况。 生成器表达式相比完整的生成器更紧凑但较不灵活，相比等效的列表推导式则更为节省内存。

示例:

```python
>>> sum(i*i for i in range(10))                 # sum of squares
285

>>> xvec = [10, 20, 30]
>>> yvec = [7, 5, 3]
>>> sum(x*y for x,y in zip(xvec, yvec))         # dot product
260

>>> unique_words = set(word for line in page  for word in line.split())

>>> valedictorian = max((student.gpa, student.name) for student in graduates)

>>> data = 'golf'
>>> list(data[i] for i in range(len(data)-1, -1, -1))
['f', 'l', 'o', 'g']
```

