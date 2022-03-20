---
title: Python手册(IDE)--常用的Python IDE
tags:
  - Python
  - IDE
categories:
  - Python
  - General
cover: /img/jupyterlab-logo.png
top_img: # /img/python-top-img.svg
abbrlink: 62196d29
date: 2018-05-25 14:40:25
emoji: heart
---

集成开发环境（IDE，Integrated Development Environment ）是用于提供程序开发环境的应用程序，一般包括代码编辑器、编译器、调试器和图形用户界面等工具。

<!-- more -->

# 常用 IDE

常用的 Python IDE 主要有以下几款：

- **文本工具类**：Sublime Text、 Atom、 **VSCode**
- **集成工具类**：**PyCharm**、**Anaconda**
- **命令行交互式**：**IPython**

[**IPython**](https://ipython.org/install.html)<img src="https://ipython.org/_static/IPy_header.png" width="35%" height="35%" align="right"/>

- IPython 是一个 Python 交互式 shell
- 支持代码高亮，自动补全，自动缩进，支持 bash shell 命令
- 大家经常遇到的魔法命令，就是IPython的众多功能之一
- 常会看到 IPython 中的`In[1]:`/`Out[1]:`形式的提示,它们并不仅仅是好看的装饰形式，还是包含输入、输出的变量。

[**Sublime Text**](http://www.sublimetext.com/)<img src="https://gitee.com/WilenWu/images/raw/master/common/sublime.jpg" width="10%" height="10%" align="right"/>

- 专为程序员开发的第三方专用编程工具
- 专业编程体验（支持自动补全、提示、语法高亮等插件）
- 多种编程风格（主题丰富）
- 工具非注册免费使用
- **Sublime Text 配置 python 环境**
   1. 打开工具 > 编译系统 > 新建编译系统..
   2. 点击 **新建编译系统** 后，会生成一个空配置文件，在这个配置文件内覆盖配置信息
   ```bash
       "encoding": "gbk",
       "cmd": ["C:/Users/Administrator/Programs/Python/python.exe","-u","$file"],
       "file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)",
       "selector": "source.python",
   ```
   3. 保存配置文件到默认路径，例如重命名为 python3
   4. 打开工具 > 编译系统 ，选择新建号的 python3 
   5. 打开工具 > 编译或者快捷键 ctrl+B 即可运行脚本。**需特别注意，脚本必须保存到本地，否则会报错！**

[**PyCharm**](https://www.jetbrains.com/pycharm/)<img src="https://gitee.com/WilenWu/images/raw/master/common/pycharm.jpg" width="10%" height="10%" align="right"/>


- JetBrains 公司开发，社区版免费
- 简单，集成度高
- 适合较复杂工程
- **Linux系统自定义pycharm命令**
   1. shell窗口打开配置文件`sudo ~/.bashrc`
   2. 添加语句`alias pycharm = "bash /download/pycharm-community-2018.1.4/bin/pycharm.sh"`(**pycharm.sh所在的路径**)
   3. 重新加载`source ~/.bashrc`
   4. shell命令行输入`pycharn`即可打开

[**Anaconda**](https://www.anaconda.com/download/)<img src="https://gitee.com/WilenWu/images/raw/master/common/anaconda.png" width="20%" height="20%" align="right"/>


- 开源的Python发行版本。
- 其包含了conda、Python等180多个科学包及其依赖项。
- 内含Anaconda Prompt，命令交互窗口，不需要设置路径。
- 集成Jupyter Notebook 和 Spyder等主流工具
- 适合数据计算领域开发

# Anaconda

## 简介

Anaconda 是一个用于科学计算的 Python 发行版，支持 Linux, Mac, Windows, 包含了众多流行的科学计算、数据分析的 Python 包，还自带Spyder和Jupyter Notebook等IDE，不需要配置系统路径，安装后可直接运行。

> 清华大学开源软件镜像站 [下载链接](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/ )，下载速度快。
> [win10+python3下Anaconda的安装及环境变量配置](https://blog.csdn.net/u013211009/article/details/78437098?locationNum=7&fps=1)

Anaconda作为管理平台，包含以下应用程序：

- Anaconda Navigator ：用于管理工具包和环境的图形用户界面，后续涉及的众多管理命令也可以在 Navigator 中手工实现。
- Jupyter notebook ：基于web的交互式计算环境，可以编辑易于人们阅读的文档，用于展示数据分析的过程。
- Anaconda Prompt：交互式命令终端，可以用来管理工具包和环境。
- spyder ：一个使用Python语言、跨平台的、科学运算集成开发环境。

![](https://gitee.com/WilenWu/images/raw/master/common/Anaconda-Navigator.png)

## 包管理

| conda (shell command)             | conda将conda、python等都视为package |
| --------------------------------- | ----------------------------------- |
| conda list                        | 查看已经安装的包                    |
| conda install package_name        | 导入包                              |
| conda update package_name         | 更新包                              |
| conda search package_name         | 查找package信息                     |
| conda update python               | 更新python                          |
| conda update anaconda             | 更新anaconda                        |
| **pip**                           |                                     |
| pip install package_name           | 导入包                              |
| pip install --upgrade package_name | 更新包                              |


# Jupyter Notebook

Jupyter Notebook（此前被称为 IPython notebook）是一个交互式笔记本，支持运行 40 多种编程语言。

Jupyter Notebook 的本质是一个 Web 应用程序，便于创建和共享文学化程序文档，支持实时代码，数学方程，可视化和 markdown。 用途包括：数据清理和转换，数值模拟，统计建模，机器学习等等.

> **Tips:**
> [最详尽使用指南：超快上手Jupyter Notebook](https://blog.csdn.net/datacastle/article/details/78890469)
> [Jupyter Notebook修改默认工作目录](https://blog.csdn.net/u014552678/article/details/62046638)
> [3步实现Jupyter Notebook直接调用R](https://blog.csdn.net/blackrosetian/article/details/77939295)
> [用jupyter notebook同时写python 和 R](https://blog.csdn.net/vincentluo91/article/details/76832264)

## 安装和使用

可以使用pip、conda安装Jupyter Lab

```sh
pip install notebook
conda install -c conda-forge notebook
```

安装后可以在命令行使用 `jupyter notebook` 运行

## 快捷键

| 快捷键      | 说明            |
| ----------- | --------------- |
| Shift+Enter | 执行            |
| Ctrl+C      | 中断运行        |
| a/b         | 上/下插入cell   |
| esc+dd      | 删除cell        |
| Tab         | 自动补全        |
| Ctrl+↑/↓    | 搜索命令        |
| Ctrl+L      | 清空屏幕        |
| Ctrl+H      | 快捷键帮助      |
| Shift+M     | 合并选中的cells |

## 魔术命令

1. Magic 关键字是可以在单元格中运行的特殊命令，能让你控制 notebook 本身或执行系统调用（例如更改目录）。
2. Magic 命令的前面带有一个或两个百分号（% 或 %%），分别对应行 Magic 命令和单元格 Magic 命令。行 Magic 命令仅应用于编写 Magic 命令时所在的行，而单元格 Magic 命令应用于整个单元格。

| magic              | 说明                                                         |
| ------------------ | :----------------------------------------------------------- |
| %quickref          | 显示IPython的快速参考                                        |
| %magic             | 显示所有魔术命令的详细文档                                   |
| %debug             | 从最新的异常跟踪的底部进入交互式调试器                       |
| %hist              | 打印命令的输入（可选输出）历史                               |
| %pdb               | 在异常发生后自动进入调试器                                   |
| %paste             | 执行剪贴板中的Python代码                                     |
| %cpaste            | 打开一个特殊提示符以便手工粘贴待执行的Python代码             |
| %reset             | 删除interactive命名空间中的全部变量/名称                     |
| %page              | 通过分页器打印输出OBJECT                                     |
| %run               | 在IPython中执行一个Python脚本文件(Python解释器:$ python)     |
| %prunstatement     | 通过cProfile执行statement，并打印分析器的输出结果            |
| %timestatement     | 报告statement的执行时间                                      |
| %timeitstatement   | 多次执行statement以计算系综平均执行时间。对那些执行时间非常小的代码很有用 |
| %matplotlib inline | Jupyter Notebook中集成Matplotlib                             |
| %matplotlib        | 直接调用matplotlib窗口弹出显示                               |

# JupyterLab

JupyterLab是Jupyter主推的最新数据科学生产工具，某种意义上，它的出现是为了取代Jupyter Notebook。不过不用担心Jupyter Notebook会消失，JupyterLab包含了Jupyter Notebook所有功能。

JupyterLab作为一种基于web的集成开发环境，你可以使用它编写notebook、操作终端、编辑markdown文本、打开交互模式、查看csv文件及图片等功能。

你可以使用pip、conda安装Jupyter Lab
```sh
pip install jupyterlab
conda install -c conda-forge jupyterlab
```

在安装Jupyter Lab后，接下来要做的是运行它。
你可以在命令行使用`jupyter-lab`或`jupyter lab`命令，然后默认浏览器会自动打开Jupyter Lab。

![](https://gitee.com/WilenWu/images/raw/master/common/jupyterlab-example.png)
