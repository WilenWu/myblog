---
title: Hexo的Git部署插件
categories:
  - Blog
tags:
  - Hexo
cover: /img/hexo-deployer-cover.png
top_img: /img/hexo-top-img.png
noticeOutdate: true
description: false
abbrlink: 7682dd2b
date: 2021-09-20 18:37:20
---


<!-- more -->

## 安装

安装插件 [hexo-deployer-git](https://github.com/hexojs/hexo-deployer-git)

```shell
$ npm install hexo-deployer-git --save
```

## 配置

修改站点配置文件`_config.yml`

```yaml
# You can use this:
deploy:
  type: git
  repo: <repository url>   # 存储库的url
  branch: [branch]         # （可选）将静态站点部署到的 Git 分支
  token: ''                # （可选）令牌值，用于对 repo 进行身份验证
  message: [message]
  name: [git user]
  email: [git email]
  extend_dirs: [extend directory]
  ignore_hidden: false # default is true
  ignore_pattern: regexp  # 任何与正则表达式匹配的文件在部署时都会被忽略

# or this:
deploy:
  type: git
  message: [message]
  repo: <repository url>[,branch]
  extend_dirs:
    - [extend directory]
    - [another extend directory]
  ignore_hidden:
    public: false
    [extend directory]: true
    [another extend directory]: false
  ignore_pattern:
    [folder]: regexp  # 或者你可以指定某个目录下的ignore_pattern

# Multiple repositories
deploy:
  repo:
    # 任何一种语法都支持
    [repo_name]: <repository url>[,branch]
    [repo_name]:
      url: <repository url>
      branch: [branch]
```

- **branch**：将静态站点部署到的 Git 分支
  
  - 在 GitHub 上默认`gh-pages`
  - 在 Coding.net 上默认`coding-pages`
  - 否则默认为`master`
  
- **token** ：令牌值，用于对 repo 进行身份验证。`$`前缀，可从Hexo环境变量中读取令牌（推荐）。

- **repo_name**：部署到多个存储库时的唯一名称。例如

  ```yaml
  deploy:
    repo:
      # 两种语法都支持
      github: https://github.com/user/project.git,branch 
      gitee:
        url: https://gitee.com/user/project.git 
        branch : branch_name
  ```
  
- **message**：提交消息。默认为`Site updated: {{ now('YYYY-MM-DD HH:mm:ss') }}`.

- **name** 和 **email**：用于提交更改的用户信息，覆盖全局配置。此信息与 git login 无关

- **extend_dirs**：要发布的其他目录。例如`demo`，`examples`

- **ignore_hidden**：(Boolean|Object)是否忽略要发布的隐藏文件

- **ignore_pattern**：(Object|RegExp)部署时选择忽略模式

  ```yaml
  # _config.yaml
  deploy:
    - type: git
      repo: git@github.com:<username>/<username>.github.io.git
      branch: master
    - type: git
      repo: git@gitee.com:<username>/<username>.git
      branch: src
      extend_dirs: /
      ignore_hidden: false
      ignore_pattern:
          public: .
  ```
  ```yaml
  deploy:
    type: git
    repo: 
         github: git@github.com:<username>/<username>.github.io.git
         coding: git@git.coding.net:<username>/<username>.coding.me.git
         gitee: git@gitee.com:<username>/<username>.git
    branch: master
  ```

## 使用令牌部署

虽然此插件可以从配置中解析身份验证令牌，但仅当您确定不会提交配置时才使用此方法，包括提交到私有存储库。更安全的方法是将它作为环境变量添加到 CI 中，然后简单地将环境变量的名称添加到此插件的配置中（例如`$GITHUB_TOKEN`）。

附加指南：

- [创建 GitHub 个人访问令牌](https://docs.github.com/cn/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- 将身份验证令牌添加到 Travis CI。[[链接\]](https://easyhexo.com/1-Hexo-install-and-config/1-5-continuous-integration.html#%E4%BB%80%E4%B9%88%E6%98%AF%E6%8C%81%E7%BB%AD%E9%9B%86%E6%88%90)

## 工作方式

`hexo-deployer-git`通过在 config 中生成站点`.deploy_git`并强制推送到 repo(es) 来工作。如果`.deploy_git`不存在，将初始化一个 repo ( `git init`)。否则将使用当前的 repo（及其提交历史）。

用户可以将部署的 repo 克隆到`.deploy_git`以保留提交历史记录。

```shell
git clone <gh-pages repo> .deploy_git
```

## 重启

删除`.deploy_git`文件夹。

```shell
$ rm -rf .deploy_git
```

## 大小写敏感

git 对大小写不敏感，因此仅修改文件名大小写在 Hexo 中并不会被重新部署，但是网页路径却对大小写敏感，这样常常会引起 404，对此只需删除`.deploy_git`下对应的文件夹即可。
