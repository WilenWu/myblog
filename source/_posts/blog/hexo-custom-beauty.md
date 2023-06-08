---
title: Hexo 博客魔改优化方法
date: 2022-03-06 14:26:06
categories:
  - Blog
tags:
  - Hexo
cover: /img/blogging-relevant-anymore.webp
top_img: 
noticeOutdate: true
abbrlink: 18fbd3bc
description:
swiper_index: 2
---

魔改方法千千万，作为一个小白，我是一个也不熟练！下面列出几种方法，哪条简单走那条。
<!-- more -->

# Hexo 插件

Hexo 有强大的插件系统，使您能轻松扩展功能而不用修改核心模块的源码。在 Hexo 中有两种形式的插件：

- **脚本（Scripts）**：如果您的代码很简单，建议您编写 JS 脚本或Node.js应用，您只需要把 JavaScript 文件放到 hexo 根目录下的 scripts 文件夹（如不存在，可自行创建），在启动时就会自动载入脚本。

    scripts 其实就是一个迷你插件，它可以实现类似于插件的功能，同时可以无侵入式的增强我们的Hexo。
    在scripts中我们可以尽情使用 [Hexo的 API](https://hexo.io/zh-cn/api/)。可参考博文 [玩转Hexo的Scripts](https://blog.hvnobug.com/post/hexo-script.html)

- **插件（Packages）**：下载已发布在 NPM 的hexo插件使用。插件会被安装在根目录下的 `node_modules` 文件夹中，Hexo 会在启动时载入。

- 您可以使用 Hexo 提供的官方工具插件来开发插件，请参考[官网链接](https://hexo.io/zh-cn/docs/plugins.html)。需要在 `node_modules` 文件夹中建立文件夹，文件夹名称开头必须为 `hexo-`，如此一来 Hexo 才会在启动时载入；否则 Hexo 将会忽略它。

# Hexo 调用脚本

当然像我这样不懂 JavaScript 的用户，只能用Hexo提供的接口自动加载自己的脚本了...

我们在根目录下的scripts文件夹种使用简单的 JS 脚本执行shell命令加载其他语言的脚本（譬如python）。下面简单介绍几种调用方法：

- **child_process**：NodeJS 可以利用子进程来调用系统命令或者文件，生成子进程的命令如下：

  ```javascript
  var child_process = require('child_process');
  ```

  子进程和系统交互的方法如下：

  ```javascript
  child_process.exec(cmd, [options], callback)  // 提供直接执行系统命令的方法
  child_process.execFile(file, [args], [options], [callback])  // 提供调用脚本文件的方法
  child_process.spawn(cmd, args=[], [options])  
  ```

  子进程提供了与系统交互的重要接口，其主要 API 有： 标准输入、标准输出及标准错误输出

  ```javascript
  child.stdin  // 标准输入
  child.stdout // 获取标准输出
  child.stderr // 获取标准错误输出
  ```

  完整的示例如下

  ```js
  var child_process = require('child_process');
  child_process.exec('python ./demo.py', function(error,stdout,stderr){
      if(error){
          console.info(stderr);
      } else {
          console.log(stdout);
      }
  });
  ```

- **shelljs** 是一个nodeJS插件，此模块重新包装了 child_process，调用系统命令更加方便。

# Butterfly 主题魔改

如想添加额外的 `js/css/meta` 等等东西，可以通过主题配置文件`Inject`里添加，支持添加到head(``</body>``标签之前)和bottom(`</html>`标签之前)。

{% note info %} 请注意：以标准的html格式添加内容 {% endnote %}

```yaml
inject:
  head:
  	- <link rel="stylesheet" href="/self.css">
  bottom:
  	- <script src="xxxx"></script>
```

每添加一个新的`js/cs`文件，都可以单独引入。也可以将所有的代码都复制到一个`js/css` 文件中。

本博客统一放到根目录的`source\js\custom.js`和`source\css\custom.css`文件中，然后修改 Butterfly 配置文件的 inject 引入

```yaml
inject:
  head:
    - <link rel="stylesheet" href="/css/custom.css">   # 加载css文件
  bottom:
    - <script src="/js/custom.js"></script>  # 加载js文件
```

# Butterfly 主题封面大小

**cover**: 1080\*720 = 3:2

**top\_img**: 1920\*600 =16:5