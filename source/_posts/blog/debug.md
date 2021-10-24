---
title: 建立博客时踩过的坑
date: 2021-09-25 13:47:05
categories: [博客搭建]
tags: [Hexo]
cover: /img/hexo-page.png
noticeOutdate: true
description: false
---

# Hexo

## Hexo 安装

Hexo 建议原封不动正常安装 `npm install -g hexo-cli`，千万不要装逼安装 `npm install -g hexo`，可能导致Hexo不能正常初始化。

## 关于 npm 升级

安装完成 `Node.js` 后建议不要升级 npm ，否则 Hexo 可能不能正常初始化 `hexo init <folder>`，且不能正常安装依赖 `npm install`


## skip_render

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

## 数学公式

butterfly 主题中使用 Katex 数学公式的时候，建议使用 `@upupming/hexo-renderer-markdown-it-plus` ，文章 front-matter 中配置 `katex: true`

## 本地搜索插件 

本地搜索插件`hexo-generator-search`[官方说明](https://github.com/wzpan/hexo-generator-search)对根目录下站点配置 `_config.yml` 会造成编译错误，建议修改成

```yaml
search:
  path: search.xml
  field: post
  content: true
- template: ./search.xml
+ format: html
```

## 拼音插件

中文链接自动转拼音插件 `hexo-permalink-pinyin`会把网页url自动转小写。启用插件后，如果你的 post 名或分类中含有大写字母，本地预览正常，`hexo d`部署端则会出现 404 异常，主要在于服务端大小写区分。

## 豆瓣插件

- 安装豆瓣影音插件后 `hexo-butterfly-douban` ，可以使用 `hexo douban` 自动爬取并生成豆瓣页面。
  由于`hexo douban` 和原始的部署命令 `hexo deploy` 均为 `d`开头，因此 `hexo d` 将不再适用，建议使用完整命令 `hexo deploy` 部署。
- 如果 `hexo douban` 爬取不到任何数据，有可能是豆瓣官方开启了反爬虫机制。豆瓣每天对爬取次数有限制，超过限制则不再允许爬取，不过第二天便会恢复正常。

##  `{+#`编译报错

{% note warning %} Hexo 对 `{`+`#`连起来的文本不能正常编译 ，可在文档中加空格处理 `{ #`  {% endnote %}

## KaTex 内的中文报错

开启 `hexo-renderer-markdown-it-plus` 作为渲染器并且用 `katex` 进行公式解析时，如果 `$...$` 中有中文的话会报错

```
LaTeX-incompatible input and strict mode is set to 'warn': Unicode text character "中" used in math mode [unicodeTextInMathMode]
LaTeX-incompatible input and strict mode is set to 'warn': Unicode text character "文" used in math mode [unicodeTextInMathMode]
No character metrics for '中' in style 'Main-Regular'
No character metrics for '文' in style 'Main-Regular'
```

这种情况的话在 `$...$` 中用 `\text{}` 对中文包裹应该可以解决这个问题，比如 `$\text{中文}$` 。另外这样的警告应该不会对渲染造成任何的影响，一定程度上可以忽视。

# Gitbook

## TypeError: cb.apply is not a function

Gitbook 安装完毕后检查版本 `gitbook -V`的时候提示

```shell
C:\Users\Admin\AppData\Roaming\npm\node_modules\gitbook-cli\node_modules\npm\node_modules\graceful-fs\polyfills.js:287
      if (cb) cb.apply(this, arguments)
                 ^

TypeError: cb.apply is not a function
    at C:\Users\Admin\AppData\Roaming\npm\node_modules\gitbook-cli\node_modules\npm\node_modules\graceful-fs\polyfills.js:287:18
    at FSReqCallback.oncomplete (fs.js:169:5)
```

按错误提示的路径找到 `polyfills.js`，在287行处定义了一个函数 `statFix()` ，看注释似乎是为了修复 npm老版本的问题。于是我们搜索调用 `statFix()`函数的地方，将其注释掉即可。

```js
  // fs.stat = statFix(fs.stat)
  // fs.fstat = statFix(fs.fstat)
  // fs.lstat = statFix(fs.lstat)
```

