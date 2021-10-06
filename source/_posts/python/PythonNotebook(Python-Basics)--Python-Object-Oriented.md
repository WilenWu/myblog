---
ID: d8c7d52bfa564e00185bfb5015e722e4
title: Python手册(Python Basics)--Python面向对象
tags: [python,数据类型,面向对象,类]
mathjax: true
copyright: true
date: 2018-06-23 20:12:35
categories: [python,Python Basics]
sticky: false
---

Python从设计之初就已经是一门面向对象的语言，正因为如此，在Python中创建一个类和对象是很容易的。本章节我们将详细介绍Python的面向对象编程。

<!-- more -->



# 面向对象(Object Oriented,OO)

- **类(Class)**: 用来描述具有相同的属性和方法的对象的集合。对象是类的实例。
- **属性和方法**：类中定义的变量和函数。加双下划线`__`私有化属性和方法，`__private_attrs`,`__private_method`
- **继承**：派生类（derived class）自动共享基类（base class）数据结构和方法的机制，这是类之间的一种关系。python支持多继承性。
- **多态**：指相同的操作或函数、过程可作用于多种类型的对象上并获得不同的结果。不同的对象，收到同一消息可以产生不同的结果，这种现象称为多态性。
- `dir(object)`：输出类的属性列表

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
--------|:--------
`__dict__` :|类的属性（包含一个字典，由类的数据属性组成）
`__doc__` |类的文档（字符串）
`__name__`|类名
`__module__`:|类定义所在的模块（类的全名是`__main__.className`，如果类位于一个导入模块mymod中，那么`className.__module__` 等于 mymod）
`__bases__` :|类的所有父类构成元素（包含了一个由所有父类组成的元组）

# 类的魔术方法(magic methods)

魔术方法|说明|调用方法
--|:--|:--
`__init__( self,[,args...] )`|构造函数，在生成对象时调用| obj = className(args)
`__del__(self)`|析构函数，释放对象时使用|del obj
`__dir__(self)`|控制`dir()`输出
`__setitem__(self)`|按照索引赋值
`__getitem__(self)`| 按照索引获取值
`__len__(self)`| 获得长度
`__call__(self)`| 函数调用
**转换成字符串**|
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

