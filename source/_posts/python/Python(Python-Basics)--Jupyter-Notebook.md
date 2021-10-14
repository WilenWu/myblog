---
ID: 55bb3a806d14cc2233d359ca511408f6
title: Python手册(IDE)--Jupyter Notebook
tags: [python,IDE]
date: 2018-04-30 11:39:49
categories: [python,Python基础]
cover: /img/jupyter-logo.png
---

# Anaconda
Anaconda 是一个用于科学计算的 Python 发行版，支持 Linux, Mac, Windows, 包含了众多流行的科学计算、数据分析的 Python 包，还自带Spyder和Jupyter Notebook等IDE，不需要配置系统路径，安装后可直接运行。

<!-- more -->

> 清华大学开源软件镜像站 [下载链接](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/ )，下载速度快。
[win10+python3下Anaconda的安装及环境变量配置](https://blog.csdn.net/u013211009/article/details/78437098?locationNum=7&fps=1)

Anaconda作为管理平台，包含以下应用程序：

- Anaconda Navigator ：用于管理工具包和环境的图形用户界面，后续涉及的众多管理命令也可以在 Navigator 中手工实现。
- Jupyter notebook ：基于web的交互式计算环境，可以编辑易于人们阅读的文档，用于展示数据分析的过程。
- Anaconda Prompt：交互式命令终端，可以用来管理工具包和环境。
- spyder ：一个使用Python语言、跨平台的、科学运算集成开发环境。

![pro](https://gitee.com/WilenWu/images/raw/master/common/Anaconda-Navigator.png)

**包管理**：在Shell窗口运行

| **conda**      | conda将conda、python等都视为package |
| ------------------- | ---------------- |
| conda list                      | 查看已经安装的包                      |
| conda install package_name        | 导入包                           |
| conda update package_name         | 更新包                           |
| conda search package_name         | 查找package信息                   |
| conda update python               | 更新python                      |
| conda update anaconda             | 更新anaconda                    |
| **pip**                           |                               |
| pip installpackage_name           | 导入包                           |
| pip install --upgradepackage_name | 更新包                           |

**shell命令**：在Jupyter, windows cmd 或Linux Shell窗口运行

| shell命令（部分）      | 说明 |
| ------------- | ---------------- |
| cd E:\Jupyter | 修改工作目录           |
| ls            | 展示目录下的文件         |
| pwd           | 展示工作目录           |


# Jupyter Notebook

Jupyter Notebook（此前被称为 IPython notebook）是一个交互式笔记本，支持运行 40 多种编程语言。

Jupyter Notebook 的本质是一个 Web 应用程序，便于创建和共享文学化程序文档，支持实时代码，数学方程，可视化和 markdown。 用途包括：数据清理和转换，数值模拟，统计建模，机器学习等等.

> **Tips:**
> [最详尽使用指南：超快上手Jupyter Notebook](https://blog.csdn.net/datacastle/article/details/78890469)
> [Jupyter Notebook修改默认工作目录](https://blog.csdn.net/u014552678/article/details/62046638)
> [3步实现Jupyter Notebook直接调用R](https://blog.csdn.net/blackrosetian/article/details/77939295)
> [用jupyter notebook同时写python 和 R](https://blog.csdn.net/vincentluo91/article/details/76832264)


## 快捷键

| 快捷键 |   说明   |
| ------------------- | --------- |
| Shift+Enter         | 执行        |
| Ctrl+C              | 中断运行      |
| a/b             | 上/下插入cell |
| esc+dd              | 删除cell    |
| Tab                 | 自动补全      |
| Ctrl+↑/↓            | 搜索命令      |
| Ctrl+L              | 清空屏幕      |
| Ctrl+H              | 快捷键帮助      |
|Shift+M	|合并选中的cells|

## 魔术命令
1. Magic 关键字是可以在单元格中运行的特殊命令，能让你控制 notebook 本身或执行系统调用（例如更改目录）。
2. Magic 命令的前面带有一个或两个百分号（% 或 %%），分别对应行 Magic 命令和单元格 Magic 命令。行 Magic 命令仅应用于编写 Magic 命令时所在的行，而单元格 Magic 命令应用于整个单元格。

| magic | 说明  |
 |------ |:-------|
| %quickref        | 显示IPython的快速参考                           |
| %magic           | 显示所有魔术命令的详细文档                            |
| %debug           | 从最新的异常跟踪的底部进入交互式调试器                      |
| %hist            | 打印命令的输入（可选输出）历史                          |
| %pdb             | 在异常发生后自动进入调试器                            |
| %paste           | 执行剪贴板中的Python代码                          |
| %cpaste          | 打开一个特殊提示符以便手工粘贴待执行的Python代码              |
| %reset           | 删除interactive命名空间中的全部变量/名称               |
| %page            | 通过分页器打印输出OBJECT                          |
| %run             | 在IPython中执行一个Python脚本文件(Python解释器:$ python) |
| %prunstatement   | 通过cProfile执行statement，并打印分析器的输出结果        |
| %timestatement   | 报告statement的执行时间                         |
| %timeitstatement | 多次执行statement以计算系综平均执行时间。对那些执行时间非常小的代码很有用 |
| %matplotlib inline     | Jupyter Notebook中集成Matplotlib                             |
|%matplotlib|直接调用matplotlib窗口弹出显示|



