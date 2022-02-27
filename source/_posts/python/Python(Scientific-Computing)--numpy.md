---
title: Python手册(Scientific Computing)--NumPy
categories:
  - Python
  - 科学计算
tags:
  - Python
  - 科学计算
cover: /img/numpy.svg
top_img: /img/numpy-logo.png
description: 使用 Python 进行科学计算的基础包
abbrlink: 5ba1c941
date: 2018-04-30 12:02:50
---

NumPy（Numerical Python的简称）是Python数值计算最重要的基础包。大多数提供科学计算的包都是用NumPy的数组作为构建基础。

<!-- more -->

NumPy的部分功能如下：

- ndarray，一个具有矢量算术运算和复杂广播能力的快速且节省空间的多维数组。
- 用于对数组执行元素级计算以及直接对数组执行数学运算的函数。
- 用于读写硬盘上基于数组的数据集的工具。	
- 线性代数运算、傅里叶变换，以及随机数生成。
- 用于集成由C、C++、Fortran等语言编写的代码的API。

```python
import numpy as np
```


# NumPy的ndarray

- ndarray 可以矢量化（vectorization）运算。
- 不同大小的数组之间的运算叫做广播（broadcasting）。

## 创建ndarray

`arr=np.array(obj,dtype=None)`

用法|说明
:---|:---
np.asarray(a, dtype=None|将a转化为数组
np.arange(n)|1:(n-1)
np.arange(start,stop,step)|等差数列
np.linspace(start,end,num=50)|返回等间隔的数字
np.empty(shape, dtype=float)<br>np.empty_like|没有任何具体值的数组，shape一个表示各维度大小的tuple
np.ones(shape, dtype=None)<br>np.zeros(shape, dtype=None)<br>np.ones_like; np.zeros_like|生成全是1或0数组，shape类型为tuple
np.eye(n)|n*n单位矩阵，对角线为1，其余为0
np.full(shape, fill_value, dtype=None)|生成全是fill_value数组
np.full_like(a,fill_value, dtype=None)|根据a的形状生成全是ill_value的数组

## ndarrary索引和切片

用法|说明
:---|:---
`arr[axis0,axis1]`<br>`arr[axis0][axis1]`|
`arr[arr < 0]`|bool索引
`arr[indexes_list]`|花式索引（总是将数据复制到新数组中） `arr[[4, 3, 0, 6]]`
`arr[start:stop:step,start:stop:step]`|slice

> Python关键字and和or在布尔型数组中无效。要使用`&`与`|`

## ndarrary属性

用法|说明
:---|:---
arr.ndim|维度数量
arr.shape|维度信息tupple
arr.size|元素总个数
arr.dtype|numpy的数据类型（详情请查阅）
arr.itermsize|每个元素的大小，以字节为单位
arr.T|转置属性

## ndarrary方法

变换数组方法|说明
:---|:---
arr.transpose()|转置方法
arr.reshape(shape)/arr.resize(shape)|修改数组维度
arr.swapaxes(ax1,ax2)|多维数组维度调换
arr.flatten()|降成一维数组
arr.astype(new_type)|修改数组类型
**统计方法**|
arr.sum(axis=None); arr.mean(axis=None)|
arr.std(axis=None); arr.var(axis=None)|
arr.min(axis=None); arr.max(axis=None)|
arr.argmin(axis=None); arr.argmax(axis=None)|返回最大值和最小值的索引
arr.cunsum(); arr.cumprod()|累积和/累积乘积
arr.sort(axis=None)|数组本身排序
**线性代数**|
arr.dot(arr2)|矩阵点积


# numpy函数

合并和分裂||
:---|:---
where(cond, xarr, yarr)|`xarr if condition else yarr`
concatenate((a1,a2,...),axis=0)|将数组a1,a2,...合并
hstack(tup)|行合并
vstack(tup)|列合并
split(ary, indices_or_sections, axis=0)|indices_or_sections : int or 1-D array
hsplit(arr, indices_or_sections)|行分裂`hsplit(x, np.array([3, 6]))`
vsplit(arr, indices_or_sections)|列分裂
**统计**|
max(a,axis=None); np.min(a,axis=None)|
fmax(a,b)|数组元素间取较大值
meshgrid(x,y)|接受两个一维数组，并产生两个二维矩阵，对应于两个数组中所有的(x,y)对
**线性代数**|
dot(arr1, arr2)|矩阵点积
gradient(f)|计算斜率(a[0]+a[1])/2
**排序**|
sort(arr,axis=-1)|返回的是数组的已排序副本
**去重**|
unique(arr,axis=None)|返回数组的唯一值，并排序
**重复和堆叠**|
repeat(a, repeats, axis=None)|repeats : int or array of ints
tile(A, reps)|堆叠

**数学运算**||
:---|:---
|sum,prod|求和，求积
|cumsum,cumprod|累积求和，累积求积
| abs  | 绝对值
| sqrt| 计算机各元素的平方根 |
| square   | 计算各元素的平方   |
| exp | 计算各元素的e的x次方|
| log,log10,log2,log1p | 分别自然对数（底数为e）、底数为10的log、底数为2的log、log（1+x） |
| sign  | 计算各元素的正负号：1（正数）、0（零）、-1（负数）    |
| ceil  | 计算各元素的ceiling值，即大于等于该值的最小正数    |
| floor | 计算各元素的floor值，即小于等于该值的最大正数 |
| rint  | 将各元素值四舍五入到最接近的整数，保留dtype  |
| modf  | 将数组的小数和整数部分以两个独立数组的形式返回   |
| isnan | 返回bool数组 |
| isfinite,isinf  |  返回bool数组|
| cos,cosh,sin,sinh,tan,tanh| 普通型和双曲型三角函数|
| arccos，arccosh，arcsin，arcsinh，arctan，arctanh | 反三角函数 |
| logical_not| 计算各元素notx的真值。相当于-arr |
| add  | 将数组中对应的元素想加|
| subtract  | 相减  |
| multiply  | 数组元素对应相乘   |
| divide（arr1,arr2），floor_divide（arr1,arr2） | 除法或向下圆整除法（丢弃余数） |
| power| 乘方  |
| mod  | 取余数  |
| greater（arr1,arr2）,greater_equal（arr1,arr2）,<br>less（arr1,arr2）,less_equal（arr1,arr2）,<br>equal（arr1,arr2）,not_equal（arr1,arr2）|执行元素级的比较运算，最终产生布尔型数组，相当于运算符`>,>=,<,<=,==,!=` |
| logical_and,logical_or,logical_xor  | 逻辑运算&、\|、^ |

ufunc方法**|`np.ufunc`
:---|:---
reduce(a, axis=0, dtype=None, out=None, keepdims=False)|对值进行连续聚合<br>`np.add.reduce(arr)`
accumulate(array, axis=0, dtype=None, out=None, keepdims=None)|保留所有局部聚合结果
reduceat(a, indices, axis=0, dtype=None, out=None)|分组约简
outer(A, B, **kwargs)|Apply the ufunc `op` to all pairs (a, b) with a in `A` and b in `B`.<br>`ufunc.outer(A,B).ndim=A.ndim+B.ndim`
np.frompyfunc(func, nin, nout)|nin : input(int), nout : output(int), return: ufunc

# NumPy的random随机库（生成n维随机数组）

用法|说明
:---|:---
np.random.rand(d0,d1,…,dn)|[0,1]区间均匀分布
np.random.randn(d0,d1,…,dn)|标准正态分布
np.random.randint(low,high,size)|[low,high] 随机整数数组
np.random.seed(s)|随机种子数
**随机排列**|
np.random.shuffle(a)|按a第一轴随机排列，改变a
np.random.permutation(a)|按a第一轴随机排列，生成副本
np.random.choice(a,size,replace,p)|以概率p数组a抽样，样本大小为size
**随机分布**|
np.random.uniform(low,high,size)|均匀分布数组
np.random.normal(loc,scale,size)|正态分布数组
np.random.poisson(lam,size)|泊松分布数组

```python
# 示例：随机漫步
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
plt.plot(walk[:100])
```

# Numba

Numba是一个开源项目，它可以利用CPUs、GPUs或其它硬件为类似NumPy的数据创建快速函数。它使用了[LLVM项目](http://llvm.org/)，将Python代码转换为机器代码



