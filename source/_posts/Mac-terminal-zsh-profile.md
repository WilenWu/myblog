---
title: Mac终端配置文件
categories:
  - General
tags:
  - terminal
cover: /img/zshell-cover.png
top_img: /img/mac-bg.jpg
abbrlink: 2da646b7
date: 2024-01-12 21:59:32
description:
---

zsh，或 Z Shell，是一个 Unix-Like 系统（如 macOS 或 Linux）下的 shell 命令行解释器。它支持强大的自动补全能力，拥有丰富的插件，具有高可定制性，而且与 bash 充分兼容。

可以在terminal交互窗口使用vi命令编辑配置文件，Mac 也可使用文本编辑命令 open。

```sh
vi ~/.zshrc
open -e ~/.zshrc
```

让我们一起看看 zsh 配置文件吧。

- `~/.zshrc` 主要用在交互shell，在每次启动 shell 都会运行
- `~/.zlogin` 登录 shell 时运行
- `~/.zprofile` 是`.zlogin`的替代品，如果使用了`.zlogin`就不必再关心此文件
- `~/.zlogout` 退出 shell 的时候读取，用于做一些清理工作
- `~/.zshenv` 用于设置环境变量，在任何场景下都能被读取

**读取顺序**：

`.zshenv -> [.zprofile if login] -> [.zshrc if interactive] -> [.zlogin if login] -> [.zlogout sometimes]`

注意，以上所有的文件都有一个系统级别的对应文件，位于 /etc/zsh*，如 .zshrc 对应于 /etc/zshrc。通常，不同的 Linux 发行版会有自己的专属配置。

**命令行提示**：

配置文件为 /etc/zshrc ，默认为

```sh
# Default prompt
PS1="%n@%m %1~ %# "
```

其中 %n 为用户名，%m 为 hostname，%1~ 表示当前路径。hostname 可通过 scutil 命令查看和修改

```sh
scutil --get HostName # 查看hostname
scutil --get LocalHostName
scutil --get ComputerName

scutil --set HostName myMac # 修改hostname
```

