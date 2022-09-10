---
title: Progress Bars with Python
tags:
  - Python
categories:
  - Python
  - General
cover: /img/progressbar-python.png
top_img: /img/progressbars-ui-ux-design.png
abbrlink: 6fe4228f
description: 在Python中实现进度条
date: 2022-08-18 22:16:33
---

在Python中，我们有许多可用的库，可以帮助我们为程序创建一个简单的进度条。在本文中，我们将逐一讨论每种方法。

# Using print

创建自定义函数打印进度条，我们计算所需的总迭代次数，在每次迭代中，我们递增符号并打印它。

## 创建打印函数

```python
import time

def progress(percent=0, width=50, desc="Processing"):
    left = width * percent // 100
    right = width - left
    
    tags = "#" * left
    spaces = "-" * right
    percents = f"{percent:.0f}%"
    
    print(f"\r{desc}: [{tags}{spaces}]{percents}", end="", flush=True)

# Example run
for i in range(101):
    progress(i)
    time.sleep(0.05)
```

当`print`函数指定`end=""`时，表示不换行，再使用`"\r"`将光标移至当前行首，相当于覆盖了之前打印出来的东西，看起来就像只有百分比在变化。

## 自定义进度条类

创建进度条类，[代码见 Github 库](https://github.com/WilenWu/Packages/)

**单进度条**：使用`ProgressBar(iterable)`包装任何可迭代对象

```python
import time 
import progressbar as pbar

for char in pbar.ProgressBar('abcd'):
    time.sleep(0.05)
```

或者手动调用 `update` 方法更新

```python
import time 
import progressbar as pbar

x = range(100)
progress = pbar.ProgressBar(x, circle=True)
for i,value in enumerate(x):
    progress.update(i)
    time.sleep(0.05)
```

**多任务进度条**：创建多进度条实例，多线程调用任务

```python
import time 
import progressbar as pbar

# 处理迭代任务中的函数
def test_func(num,sec):
    time.sleep(sec)
    return num * 2

bars = pbar.MultiProgressBar()
bars.add_task('task1', pbar.ProgressBar(range(3),desc='task1'), test_func, 1)
bars.add_task('task2', pbar.ProgressBar('ab',desc='task2'), test_func, 1)
bars.add_task('task3', pbar.ProgressBar('ABCD',desc='task3'), test_func, 1)
bars.start(['task2','task1', 'task3'])
```

# Using tqdm

tqdm 库用于在Python和命令行创建快速有效的进度条。详细的用法可以参考[官方文档](https://pypi.org/project/tqdm/)

## 包装迭代对象

使用`tqdm(iterable)`包装任何可迭代对象。默认情况下，该进度条显示剩余时间、每秒迭代次数和百分比。

```python
from tqdm import tqdm
from time import sleep

for char in tqdm(["a", "b", "c", "d"], desc="Processing"):
    sleep(0.25)
```

`trange(N)` 是 `tqdm(range(N))` 的快捷方式 

```python
from tqdm import trange

for i in trange(100, desc="Processing"):
    sleep(0.01)
```

## 手动更新

使用`with`语句手动控制`tqdm()`更新

```python
with tqdm(total=100) as pbar:
    for i in range(10):
        sleep(0.1)
        pbar.update(10)
```

`with` 也是可选的，您可以将 `tqdm()`分配给一个变量，但在这种情况下，不要忘记在末尾使用`del`或`close`

```python
pbar = tqdm(total=100)
for i in range(10):
    sleep(0.1)
    pbar.update(10)
pbar.close()
```

## 嵌套进度条

```python
from tqdm.auto import trange
from time import sleep

for i in trange(4, desc='1st loop'):
    for j in trange(5, desc='2nd loop'):
        for k in trange(50, desc='3rd loop', leave=False):
            sleep(0.01)
```

## Pandas 集成

```python
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
# (can use `tqdm.gui.tqdm`, `tqdm.notebook.tqdm`, optional kwargs, etc.)
tqdm.pandas(desc="my bar!")

# Now you can use `progress_apply` instead of `apply`
# and `progress_map` instead of `map`
df.progress_apply(lambda x: x**2)
# can also groupby:
# df.groupby(0).progress_apply(lambda x: x**2)
```

# Using progressbar2

一个 Python 进度条库，为长时间运行的操作提供可视化（基于文本）进度。详细的用法可以参考[官方文档](https://pypi.python.org/pypi/progressbar2)

## 包装迭代对象

```python
import time
import progressbar

for i in progressbar.progressbar(range(100)):
    time.sleep(0.02)
```

## 手动更新

```python
import time
import progressbar

with progressbar.ProgressBar(max_value=10) as bar:
    for i in range(10):
        time.sleep(0.1)
        bar.update(i)
```

# Using rich

Rich主要是用于在终端中打印丰富多彩的文本。详细的用法可以参考[官方文档](https://pypi.org/project/rich/)

## 包装迭代对象

将任何序列包装在`track`函数中并迭代结果。内置列包括完成百分比、文件大小、文件速度和剩余时间。

```python
import time
from rich.progress import track

for i in track(range(20), description="Processing..."):
    time.sleep(1)  # Simulate work being done
```

## 多任务进度条

如果您需要在终端中显示多个任务进度条，则可以直接使用`Progress`类。构建 `Progress` 对象后，使用 ( `add_task()`添加任务，并使用 `update()` 更新进度。

```python
import time

from rich.progress import Progress

with Progress() as progress:

    task1 = progress.add_task("[red]Downloading...", total=1000)
    task2 = progress.add_task("[green]Processing...", total=1000)
    task3 = progress.add_task("[cyan]Cooking...", total=1000)

    while not progress.finished:
        progress.update(task1, advance=0.5)
        progress.update(task2, advance=0.3)
        progress.update(task3, advance=0.9)
        time.sleep(0.02)
```

