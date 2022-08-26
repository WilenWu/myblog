---
title: Hexo5博客搭建及配置
categories:
  - Blog
tags:
  - Hexo
cover: /img/hexo-cover.png
top_img: /img/hexo-top-img.png
noticeOutdate: true
abbrlink: a31c2363
date: 2019-09-10 13:48:53
description:
---

[Hexo](https://hexo.io/zh-cn/) 是高效的静态站点生成框架，基于 [Node.js](https://nodejs.org/)。 通过 Hexo 你可以轻松地使用 Markdown （或其他渲染引擎）解析文章，在几秒内，即可利用靓丽的主题生成静态网页。除了 Markdown 本身的语法之外，还可以使用 Hexo 提供的 [tag 插件](https://hexo.io/zh-cn/docs/tag-plugins.html) 来快速的插入特定形式的内容。Hexo 还拥有丰富的主题和插件。

<!-- more -->

# 前提环境

- [Node.js](http://nodejs.org/) （建议使用 Node.js 12.0 及以上版本）
- [Git](http://git-scm.com/)：[Git命令官方文档](https://git-scm.com/book/zh/v2)
- 所有必备的应用程序安装完成后，即可使用 npm 安装 Hexo。

  ```shell
  npm install -g hexo-cli
  ```

安装以后，可以使用以下两种方式执行 Hexo：

1. `npx hexo <command>`
2. 将 Hexo 所在的目录下的 `node_modules` 添加到环境变量之中即可直接使用 `hexo <command>`：

  ```shell
  echo 'PATH="$PATH:./node_modules/.bin"' >> ~/.profile
  ```

# 相关命令

安装 Hexo 完成后，请执行下列命令，Hexo 将会在指定文件夹中新建所需要的文件。选择博客目录文件夹，右键打开 `Git Bash Here`

- 建站 
```bash
hexo init <folder>        # 初始化项目文件夹
cd <folder>               # 进入项目根目录
npm install               # 安装依赖包
```

- 常用操作
```bash
hexo clean           # 清除缓存
hexo generate        # 编译文件 生成静态页面至 \public 目录
hexo serve           # 本地预览  http://localhost:4000 
hexo deploye         # 发布 https://username.github.io

hexo new "PostName"        # 新建文章至 \source\_posts
hexo new page  "pageName"  # 新建页面

hexo list <type>   # 列出网站资料
hexo version       # 显示 Hexo 版本
```

>  在服务器启动期间，Hexo 会监视文件变动并自动更新，您无须重启服务器。

- 命令简写
```bash
hexo clean && hexo g && hexo s    # 清除缓存+生成+预览
hexo clean && hexo g -d           # 清除缓存+生成+发布
```

# 站点文档

新建完成后，指定文件夹的目录如下：

```
.
├── _config.yml
├── package.json
├── scaffolds
├── scripts
├── source
|   ├── _drafts
|   └── _posts
└── themes
```

- _config.yml：网站的 [配置](https://hexo.io/zh-cn/docs/configuration) 信息，您可以在此配置大部分的参数。

- package.json：应用程序的信息。[EJS](https://ejs.co/), [Stylus](http://learnboost.github.io/stylus/) 和 [Markdown](http://daringfireball.net/projects/markdown/) renderer 已默认安装，您可以自由移除。
- scaffolds：[模版](https://hexo.io/zh-cn/docs/writing) 文件夹。当您新建文章时，Hexo 会根据 scaffold 来建立文件。
  Hexo的模板是指在新建的文章文件中默认填充的内容。例如，如果您修改scaffold/post.md中的Front-matter内容，那么每次新建一篇文章时都会包含这个修改。
- scripts：脚本目录，此目录下的JavaScript文件会被自动执行
- source：资源文件夹是存放用户资源的地方。除 `_posts` 文件夹之外，开头命名为 `_` (下划线)的文件 / 文件夹和隐藏的文件将会被忽略。Markdown 和 HTML 文件会被解析并放到 `public` 文件夹，而其他文件会被拷贝过去。
- themes：[主题](https://hexo.io/zh-cn/docs/themes) 文件夹。Hexo 会根据主题来生成静态页面。

# 站点配置

Hexo 框架主要配置两方面的内容：站点配置文件和主题配置文件。

- **站点配置文件** `_config.yml`[中文官方文档](https://hexo.io/zh-cn/docs/configuration)
- **主题配置文件**：通常情况下，Hexo 主题是一个独立的项目，并拥有一个独立的 `_config.yml` 配置文件。除了自行维护独立的主题配置文件，你也可以在其它地方对主题进行配置。
- **独立的 `_config.[theme].yml` 文件**：（该特性自 Hexo 5.0.0 起提供）
  你可将独立的主题配置文件应放置于站点根目录下命名为 `_config.[theme].yml` 。支持 `yml` 或 `json` 格式。需要配置站点 `_config.yml` 文件中的 `theme` 以供 Hexo 寻找 `_config.[theme].yml` 文件。

# 一键部署

Hexo 提供了快速方便的[一键部署功能](https://hexo.io/zh-cn/docs/one-command-deployment)，让您只需一条命令就能将网站部署到服务器上。

1. [创建 github pages 库](#创建-github-pages-库)：库名为 `<username>.github.io` 
2. 请在库设置（Repository Settings）中将默认分支设置为`_config.yml`中的分支名称。
3. [安装并配置 Git 部署器插件](hexo-deployer-git/)
4. 执行 `hexo clean && hexo deploy` 生成站点文件并推送至远程库。 
  除非您使用[**令牌**](hexo-deployer-git/)或配置 [**SSH keys**](#配置-SSH-keys) 进行身份验证，否则将使用目标存储库的用户名和密码提示您。
5.  稍等片刻，您使用浏览器访访问 `https://<username>.github.io` 就可以成功访问我们的博客。

当执行 hexo deploy 时，Hexo 会将 public 目录中的文件和目录推送至_config.yml 中指定的远端仓库和分支中，并且**完全覆盖**该分支下的已有内容。

此外，如果您的 Github Pages 需要使用 CNAME 文件**自定义域名**，请将 CNAME 文件置于 source 目录下，只有这样 hexo deploy 才能将 CNAME 文件一并推送至部署分支。

# 插件

Hexo 有强大的插件系统，使您能轻松扩展功能而不用修改核心模块的源码。在 Hexo 中有两种形式的插件：

- **脚本（Scripts）**：如果您的代码很简单，建议您编写脚本，您只需要把 JavaScript 文件放到 hexo 根目录下的 scripts 文件夹(如不存在，    可自行创建)，在启动时就会自动载入脚本。
    scripts 其实就是一个迷你插件，它可以实现类似于插件的功能，同时可以无侵入式的增强我们的Hexo。
    在scripts中我们可以尽情使用Hexo的Api。可参考博文[玩转Hexo的Scripts](https://blog.hvnobug.com/post/hexo-script.html)

- **插件（Packages）**：下载已发布在 NPM 的hexo插件使用。

# 附录

## 创建 github pages 库

1. 新建库 `<username>.github.io` ，库名必须以`github.io`结尾。 
2. 点击 settings，进入仓库设置页面。
3. 找到 GitHub Pages ，设置`Source`为默认分支。

## 配置 SSH keys

1. 生成新的 SSH keys
   ```sh
   ssh-keygen -t rsa -C "email@example.com"
   ```
2. 添加 SSH Key 到 GitHub
   - 输入指令 `cat ~/.ssh/id_rsa.pub` 查看本机公钥
   - 将公钥添加进 Github 账户 `setting` -> `SSH  and GPG keys` -> `SSH keys `->`New SSH key`
3. 测试设置是否成功
   ```sh
   ssh -T git@github.com
   ```
4. 配置个人信息
   ```sh
   git config --global user.name "username"
   git config --global user.email "email@example. com"
   git config -l     # 查看是否成功
   ```
   
   这里建议用户名和邮箱与你的 GitHub 用户名和邮箱保持 一致。每次 Git 提交时都会附带这两条信息，用于记录是 谁提交的更新，并且会随更新内容一起被记录到历史记录 中。简单说，是用来标记的你的身份的～

## Hexo命令设置别名

打开安装的Git文件夹，在文件夹中搜索.bashrc文件，也可以循着路径找，默认路径是C:\Program Files\Git\etc\bash.bashrc，找到后在文档末尾添加快捷命令：

```shell
alias hc='hexo clean && python ./demo.py'
alias gp='git add . && git commit -m "update" && git push -f'
```

以管理员身份保存后，重新打开git命令行即可使用快捷命令。