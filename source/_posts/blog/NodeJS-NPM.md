---
title: NPM简单使用
categories:
  - Blog
tags:
  - npm
cover: /img/npm-cover.png
top_img: /img/nodejs-white.svg
noticeOutdate: true
abbrlink: e0446c8e
date: 2021-09-21 09:32:20
description:
---

# 概要

NPM(Node Package Manager) 是随同NodeJS一起安装的 javascript 包管理工具，能解决NodeJS代码部署上的很多问题，常见的使用场景有以下几种：

- 允许用户从NPM服务器下载别人编写的第三方包到本地使用。
- 允许用户从NPM服务器下载并安装别人编写的命令行程序到本地使用。
- 允许用户将自己编写的包或命令行程序上传到NPM服务器供别人使用。

<!-- more -->

# 查看版本号

```shell
node -v     # 查看NodeJS版本号
npm -v      # 查看npm版本号
npm version # 查看npm及依赖项版本号
```

如果你安装的是旧版本的 npm，可以通过 npm 命令来升级

```shell
npm install npm -g  
```

# 本地模式和全局模式

npm 有两种操作模式：

- 本地模式：npm 将包安装到当前项目目录中 `./node_modules`。
- 全局模式：npm 将包安装到 node 的安装目录

本地模式是默认模式。使用`-g`或`--global`在任何命令上改为在全局模式下运行。

```shell
npm config get prefix  # 获取全局安装的默认目录
```

# npm 配置

package.json是项目的配置管理文件，定义了这个项目所需要的各个依赖模块以及项目的配置信息

```shell
npm config get registry  # 查看镜像源
npm config set registry https://registry.npm.taobao.org  # 使用淘宝镜像
npm config list  # 查看配置列表
npm config ls -l # 查看所有的默认配置
npm config delete key # 删除配置
```

# 安装模块

npm install 默认安装最新版本，如果想要安装指定版本，可以在库名称后加 @版本号

```shell
npm list        # 已安装模块信息             
npm list <pkg>  # 查看某个模块的版本号
# 别名 ls

npm install  # 安装项目所有的依赖包（在包目录中，没有参数）
npm install <pkg>               # 本地安装
npm install <pkg>@<version>     # 指定版本安装
npm install <pkg>@latest        # 最新版本
# 别名i, add
```

**安装选项**

- `-g` 或 `--global` 全局模式。例如

  ```shell
  npm install <pkg> -g     # 全局安装
  ```

- `-S` 或 `--save`  将已安装的包作为依赖项保存到 package.json 文件中

# 卸载模块

我们可以使用以下命令来卸载 Node.js 模块。

```shell
npm uninstall <pkg>
# 别名 remove, rm, r, un, unlink
```

# 更新模块

我们可以使用以下命令更新模块：

```shell
npm update <pkg>
# 别名 up, upgrade
```

此命令会将列出的所有包更新为最新版本，同时受包及其依赖项的约束。它还将安装缺少的软件包。

# 搜索模块

使用以下来搜索模块：

```shell
npm search <pkg>
```

# 附录

## 更新包

目前我们前端项目还挺多的，许多依赖都没办法统一管理，推荐一个npm包管理工具：npm-check：

```shell
npm install -g npm-check
```

`npm-check` 命令会检查项目中没有使用过的包、有更新的包、推荐安装的包

- `-u, --update` 出现一个**交互式更新界面**，我们可以根据需要更新并同步package.json

- `-y, --update-all` 更新所有的依赖

- `-g, --global` 全局模式，例如

  ```shell
  npm-check -gu
  ```
