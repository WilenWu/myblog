---
ID: 
title: 建立Hexo博客时踩过的坑
tags: [Hexo]
copyright: true
date: 
categories: [博客搭建]
sticky: 
noticeOutdate: true
---

# Hexo 安装

Hexo 建议原封不动正常安装 `npm install -g hexo-cli`，千万不要装逼安装 `npm install -g hexo`，可能导致Hexo不能正常初始化。

# 关于 npm 升级

安装完成 `Node.js` 后建议不要升级 npm ，否则 Hexo 可能不能正常初始化 `hexo init <folder>`，且不能正常安装依赖 `npm install`


# skip_render

{% note info %} 站点配置`/_config.yml/skip_render`  {% endnote %}

`skip_render`：跳过指定文件的渲染。匹配到的文件将会被不做改动的复制到 `public` 目录中。

- `skip_render: "mypage/**"` 将会直接将 `source/mypage/`下的所有文件和目录不做改动地输出到 'public' 目录
  注意：这里只能填相对于source文件夹的**相对路径**。千万不要手贱加上个`/`

- `skip_render: "_posts/test-post.md"` 这将会忽略对 'test-post.md' 的渲染

- `skip_render: "mypage/*"`将会忽略`source/mypage/`文件夹下所有文件的渲染

- `skip_render: "mypage/*.html"` 将会忽略`source/mypage/`文件夹下`.html`文件

- 如果要忽略多个路径的文件或目录，可以这样配置：

  ```shell
  skip_render: 
    - "_posts/test-post.md"   
    - "mypage/*
  ```

# 数学公式

butterfly 主题中使用 Katex 数学公式的时候，建议使用 `@upupming/hexo-renderer-markdown-it-plus` ，文章 front-matter 中配置 `katex: true`

# 本地搜索插件 

本地搜索插件`hexo-generator-search`[官方说明](https://github.com/wzpan/hexo-generator-search)对根目录下站点配置 `_config.yml` 会造成编译错误，建议修改成

```yaml
search:
  path: search.xml
  field: post
  content: true
- template: ./search.xml
+ format: html
```

# 拼音插件

中文链接自动转拼音插件 `hexo-permalink-pinyin`会把网页url自动转小写。启用插件后，如果你的 post 名或分类中含有大写字母，本地预览正常，`hexo d`部署端则会出现 404 异常，主要在于服务端大小写区分。

