---
ID: 667579e38e6f1d974c6c8f09efa072f4
title: Python手册(IDE)--常用的Python IDE
tags: [python,IDE]
date: 2018-05-25 14:40:25
categories: [python,Python基础]
---

集成开发环境（IDE，Integrated Development Environment ）是用于提供程序开发环境的应用程序，一般包括代码编辑器、编译器、调试器和图形用户界面等工具。

<!-- more -->

> 本文内容摘自MOOC.北京理工大学Python课程

常用的 Python IDE 主要有以下几款（本文主要介绍其中四款）：
文本工具类IDE| 集成工具类IDE
:-------|:--------
**Python IDLE**|**PyCharm**
Notepad++|Wing
**Sublime Text**|PyDev & Eclipse
Vim & Emacs|Visual Studio
Atom|**Anaconda**
Komodo Edit|Canopy



-------

[**Python IDLE**](https://www.python.org/downloads/)<img src="/images/python.png" width="20%" height="20%" align="right"/>

适用于

- Python入门
- 功能简单直接
- 少量代码


[**Sublime Text**](http://www.sublimetext.com/)<img src="/images/sublime.jpg" width="10%" height="10%" align="right"/>

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

[**PyCharm**](https://www.jetbrains.com/pycharm/)<img src="/images/pc.jpg" width="10%" height="10%" align="right"/>


- JetBrains 公司开发，社区版免费
- 简单，集成度高
- 适合较复杂工程

> **Linux系统自定义pycharm命令**
> 1. shell窗口打开配置文件`sudo ~/.bashrc`
> 2. 添加语句`alias pycharm = "bash /download/pycharm-community-2018.1.4/bin/pycharm.sh"`(**pycharm.sh所在的路径**)
> 3. 重新加载`source ~/.bashrc`
> 4. shell命令行输入`pycharn`即可打开


[**Anaconda**](https://www.anaconda.com/download/)<img src="/images/anaconda.png" width="15%" height="15%" align="right"/>


- 开源的Python发行版本。
- 其包含了conda、Python等180多个科学包及其依赖项。
- 内含Anaconda Prompt，命令交互窗口，不需要设置路径。
- 集成Jupyter Notebook 和 Spyder等主流工具
- 适合数据计算领域开发



