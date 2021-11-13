---
title: Python手册(Standard Library)--Python 代码调试工具
date: 2021-11-13 22:23:11
categories: [Python,Python标准库]
tags: [Python]
cover: /img/python-debug.jpg
top_img:  # /img/python-logo-wide.svg
description: Python 代码调试工具快速入门
---

1. python 提供了一个默认的 debugger：pdb，而 ipdb 则是 pdb 的增强版，提供了补全、语法高亮等功能，类似于 ipython 与 python 默认的交互终端的关系，通过 pip install ipdb 即可安装 ipdb。
2. ipdb 的使用方法一般有两种：集成到源代码或通过命令交互。
3. 集成到源代码可以直接在代码指定位置插入断点。如下所示：

ipdb.set_trace()

插入的断点代码会污染原来的代码空间



1. 交互式的命令式调试方法更加方便。启动命令式调试环境的方法（非侵入式）

```
python -m ipdb code.py
```
