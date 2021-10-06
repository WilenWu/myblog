---
ID: 5980ef6822399cf8ecb3c84fd29d6252
title: NexT主题配置
tags: [Hexo,NexT,Github pages]
copyright: true
date: 2019-09-11 14:14:04
categories: [博客搭建]
---

[NexT](https://theme-next.js.org/) 是 [Hexo](http://hexo.io) 框架中最为流行的主题之一。精于心，简于形。
NexT 支持多种常见第三方服务，使用  **第三方服务**  来扩展站点的功能 。
除了 Markdown 支持的语法之外，NexT 借助 Hexo 提供的  **tag 插件** ， 为您提供在书写文档时快速插入带特殊样式的内容。

<!-- more -->

# 安装主题

1. 安装 NexT 主题
  - 如果你在使用 Hexo 5.0 或更新版本，最简单的安装方式是通过 npm：
    ```shell
    $ cd hexo-site
    $ npm install hexo-theme-next
    ```
  - 你也可以直接克隆整个仓库：
    ```shell
    $ cd hexo-site
    $ git clone https://github.com/next-theme/hexo-theme-next themes/next
    ```

2. 启用主题

   与所有 Hexo 主题启用的模式一样。 当 克隆/下载 完成后，打开站点配置文件， 找到 theme 字段，并将其值更改为 next
   
   ```yaml
   theme: next
   ```

# 主题配置

目前 NexT 鼓励用户使用 [官网博客](https://theme-next.js.org/docs/getting-started/configuration.html) 进行配置。并且可以轻松地通过 [Custom Files](https://theme-next.js.org/docs/advanced-settings/custom-files.html) (`source/_data`) 自定义主题的布局和样式。网上的若干优化教程估计已非必要。

不推荐直接修改 NexT 主题的文件。因为这可能导致错误（例如 git merge 冲突），并且在升级主题时修改的文件可能丢失。Hexo 5.0 版本起，可将主题配置文件 `_config.yml` 应放置于站点根目录下命名为 `_config.next.yml` 进行独立配置，便于升级。

NexT 8 自定义样式支持已与主题核心组件分离，如博客背景、文本结束标记等。这样用户可放心升级主题而不会破坏自定义配置。在路径 `source/_data` 下添加自定义文件，并在主题配置文件 `custom_file_path` 取消注释。

另，tag 插件也是NexT的一大亮点，会单独开一篇文章具体介绍，请参考 [NexT tag 插件](/hexo/Hexo-tag-plugins/)

# 更新

NexT 每个月都会发布新版本。请在更新前阅读[更新说明](https://github.com/next-theme/hexo-theme-next/releases)。你可以通过如下命令更新 NexT。

通过 npm 安装最新版本：

```shell
$ cd hexo-site
$ npm install hexo-theme-next@latest
```

或者通过 git 更新到最新的 master 分支：

```shell
$ cd themes/next
$ git pull
```

# 主题优化

插件丰富和拓展了 NexT 的功能。这些插件分为两种：核心插件和第三方插件。核心插件被 NexT 的基础功能所依赖。第三方插件提供了大量的可选功能。

## 置顶和置顶标签

【推荐】Hexo 自带置顶功能，`hexo-generator-index` 从 2.0.0 开始，已经支持文章置顶功能。你可以直接在文章的front-matter区域里添加`sticky: 1`属性来把这篇文章置顶。数值越大，置顶的优先级越大。

> 安装 `hexo-generator-index-pin-top` 插件就目前来说对Next主题并不友好，并没有置顶标签 <i class="fa fa-thumb-tack"></i>

## RSS支持

RSS(Really Simple Syndication)称为简易信息聚合，是一种描述和同步网站内容的格式。RSS搭建了信息迅速传播的一个技术平台，使得每个人都成为潜在的信息提供者。发布一个RSS文件后，这个RSS Feed中包含的信息就能直接被其他站点调用，而且由于这些数据都是标准的XML格式，所以也能在其他的终端和服务中使用，是一种描述和同步网站内容的格式。

NexT 中 RSS 有三个设置选项，满足特定的使用场景。 更改主题配置文件，设定 rss 字段的值。
{% tabs rss, 1 %}
<!-- tab false -->
禁用 RSS，不在页面上显示 RSS 连接。
{% code next\_config.yml %}
rss: false 
{% endcode %}
<!-- endtab -->

<!-- tab blank -->
留空：使用 Hexo 生成的 Feed 链接，需先安装 `hexo-generator-feed` 插件
{% code next\_config.yml %}
rss:  
{% endcode %}

- 安装 `hexo-generator-feed` 插件
   {% code %}
   npm install hexo-generator-feed --save 
   {% endcode %}
   
- 在站点配置文件 `_config.yml` 修改/添加
   {% code \_config.yml %}
   feed:
     type: atom
     path: atom.xml
     limit: 20     # Feed中的最大帖子数(使用0或false显示所有帖子)
   {% endcode %}
   <!-- endtab -->

<!-- tab URL -->
具体的URL：适用于已经烧制过 Feed 的情形。
{% code next\_config.yml %}
rss: /atom.xml
{% endcode %}

- 安装 `hexo-generator-feed` 插件
   {% code %}
   npm install hexo-generator-feed --save 
   {% endcode %}
   
- 在站点配置文件 `_config.yml` 修改/添加
   {% code \_config.yml %}
   feed:
     type: atom
     path: atom.xml
     limit: 20     # Feed中的最大帖子数(使用0或false显示所有帖子)
   {% endcode %}
   <!-- endtab -->
   {% endtabs %}



## 书签

Bookmark是一个插件，允许用户保存他们的阅读进度。用户只需单击页面左上角的书签图标即可保存滚动位置。当他们下次访问您的博客时，他们可以自动恢复每个页面的最后滚动位置。
在主题配置文件启用

```yaml
bookmark:
  enable: true
  color: "#222"   # 自定义书签颜色
  save: auto  # auto | manual 自动保存进度或点击保存进度`
```

## 字数统计和阅读时长

- 导入插件
  
   ```shell
   npm install hexo-word-counter
   ```
   
- Hexo 配置
  
   ```yaml
   symbols_count_time:
     symbols: true              # 是否启用
     time: true                 # 估计阅读时间
     total_symbols: true        # 页脚部分中所有帖子字数
     total_time: true           # 页脚部分中所有帖子的估计阅读时间
     awl: 4                     # 平均字长
     wpm: 275                   # 每分钟的平均字数
   ```
   
- Next 主题配置
  
   ```yaml
   symbols_count_time:
     separated_meta: true       # 以分隔线显示单词计数和估计读取时间
     item_text_total: false     # 在页脚部分显示单词计数和估计阅读时间的文本描述
   ```

## 文章热度

- 配置 leancloud [官方使用文档](https://theme-next.org/docs/third-party-services/comments#Disqus)

- 修复NexT的leancloud计数器的安全插件
1. 导入插件
   {% code %}
   npm install hexo-leancloud-counter-security --save
   {% endcode %}

2. 站点配置文件添加
   {% code \_config.yml %}
   leancloud_counter_security:
     enable_sync: true
     app_id: <<your app id>>
     app_key: <<your app key>>
     username: <<your username>> # 部署时会询问是否留空
     password: <<your password>> # 建议留空。部署时会询问是否留空
   {% endcode %}

3. 主题配置文件修改 
   {% code next\_config.yml %}
   leancloud_visitors:
     enable: true
     app_id: <<your app id>>
     app_key: <<your app key>>
   
   # Dependencies: https://github.com/theme-next/hexo-leancloud-counter-security
   
     security: true
    betterPerformance: false
   {% endcode %}
   
4. 控制台命令：在Leancloud数据库中注册用户以进行权限控制
   {% code %}
   hexo lc-counter register <<username>> <<password>>
   {% endcode %}

## 相关热门帖子

[NexT](https://github.com/tea3/hexo-related-popular-posts)根据[hexo-related-popular-posts](https://github.com/tea3/hexo-related-popular-posts)支持相关的帖子功能

- 导入插件

  ```shell
  npm install hexo-related-popular-posts --save
  ```

- Next 主题配置

  ```yaml
  related_posts:
    enable: true 
    title:                   # 默认标题“相关帖子”
    display_in_home: false   # 是否在主页展示
    params:
      maxCount: 5            # 最大数量
      #PPMixingRate: 0.0     # 热门帖子和相关帖子的混合比例
      #isDate: false         # 显示相关帖子的日期
      #isImage: false        # 显示相关帖子的图像
      #isExcerpt: false      # 显示相关帖子的摘录
  ```

- 在markdown中添加标签

  如果文章中包含标签，可以将相关文章显示为列表。例如，添加类似以下 markdown 文件的标记。

  ```markdown
  ---
  title: Hello World
  tags:
    - program
    - diary
    - web
  ---
  Welcome to [Hexo](https://hexo.io/)! This is a sample article. Let's add some tags as above.
  ...
  ```

  匹配标签的数量越多，显示为候选的相关文章就越多。

# 自定义页面

## 自定义页面

Hexo 中除了home和categories外，均需要添加页面。

- 添加新页面

  ```shell
  $ cd hexo-site
  $ hexo new page <custom-name>
  ```

- 在 `custom-name/index.md` 设置 front-matter

  ```markdown
  title: custom-name
  date: 2014-12-22 12:39:04
  ---
  ```

- 编辑 Next 配置，比如添加`about`页面

  ```yaml
  menu:
    ... ...
    about: /about/ || fa fa-user
  ```

## 标签页 

- 添加新页面

  ```shell
  $ cd hexo-site
  $ hexo new page tags
  ```

- 在 `custom-name/index.md` 编辑 front-matter，type 设置为 tags

  ```markdown
  title: Tags
  date: 2014-12-22 12:39:04
  type: tags
  ---
  ```

- 编辑 Next 配置

  ```yaml
  menu:
    ... ...
    tags: /tags/ || fa fa-tags
  ```

- **标签云**：默认情况下，NexT 已在标签页面中为标签云设置了字体颜色和大小。
  从 NexT v7.0.2 开始，您可以自定义它们，只需在Next 配置  `tagcloud`

## 类别页

分类页面可以用类似的方式新建，唯一的区别是：type 由 tags更换为 categories。

## 公益404

如果你想启用`commonweal 404`（腾讯在中国提供的服务），`source/404/index.md` 像这样编辑：

```markdown
---
title: '404'
date: 2014-12-22 12:39:04
comments: false
---
<script src="//qzonestyle.gtimg.cn/qzone/hybrid/app/404/search_children.js"
        charset="utf-8" homePageUrl="/" homePageName="Back to home">
</script>
```

您还可以添加任何您想要的内容。

 Next 主题配置

```yaml
menu:
  ... ...
  commonweal: /404/ || fa fa-heartbeat
```

# SEO支持

SEO(Search Engine Optimization)意为搜索引擎优化，利用搜索引擎的规则提高网站在有关搜索引擎内的自然排名。

> 参考链接：[站点（e.g. Hexo Blog）提交百度搜索引擎收录实现SEO_LL_Leung的博客-CSDN博客](https://blog.csdn.net/liangllhahaha/article/details/105343008)

## Sitemap

Sitemap 可方便网站管理员通知搜索引擎他们网站上有哪些可供抓取的网页。最简单的 Sitemap 形式，就是XML 文件，在其中列出网站中的网址以及关于每个网址的其他元数据（上次更新的时间、更改的频率以及相对于网站上其他网址的重要程度为何等），以便搜索引擎可以更加智能地抓取网站。

- 安装站点地图(sitemap)插件 [hexo-generator-sitemap](https://github.com/hexojs/hexo-generator-sitemap)

  ```sh
  npm install hexo-generator-sitemap --save
  npm install hexo-generator-baidu-sitemap --save
  ```

- Hexo配置文件修改/添加

  ```yaml
  # SEO 
  sitemap:
    path: sitemap.xml                  # 站点地图路径
    template: ./sitemap_template.xml   # 自定义模板路径
    rel: false                         # 添加rel-sitemap到站点的标题
    tags: true                         # 添加站点的标签
    categories: true                   # 添加站点的类别
  
  baidusitemap:
    path: baidusitemap.xml
  ```

  > 不想被抓取的页面可以在front-matter配置 sitemap: false

- 启用百度推送功能，博客会自动将网址推送到百度，这对搜索引擎优化非常有帮助。
  主题配置文件修改 `baidu_push: true`

## 百度站长工具

- 登录[百度站长工具](https://ziyuan.baidu.com/site/)，进入验证方式，选择**HTML标签验证**，获取验证字符串：

  ```html
  <meta name="baidu-site-verification" content="code-XXXXXXXXXX">
  ```

- 复制`content`的值 `code-XXXXXXXXXX`，编辑Next 配置文件

  ```yaml
  baidu_site_verification: code-XXXXXXXXXX
  ```

> 默认情况下，百度会缓存并重写您的网站，为移动用户提供网页快照。您可以通过设置站点类型来禁用此功能。路径：搜索展现 --> 站点属性 --> 基础信息 --> 站点类型 --> 修改  --> **自适应站**

## 百度分析

- 登录[百度分析](https://tongji.baidu.com/) 并定位到站点代码获取页面：管理 --> 代码管理 --> 代码获取
- 复制`hm.js?` 后的脚本ID，在Next中配置 `baidu_analytics: your_id`

# 第三方服务

静态网站在某些功能上受到限制，因此我们需要第三方服务来扩展我们的网站。
您可以随时使用NexT支持的第三方服务扩展所需的功能。第三方插件提供了大量可选功能。它们默认从 jsDelivr CDN 加载，因为它在任何地方都很快。

## 数学公式

NexT提供了两个用于显示数学公式的渲染引擎。

1. 启用数学公式

   ```yaml
   math:
     per_page: true 
   ```
   
   - per_page: true 默认只渲染 Front-matter 标记 `mathjax: true` 的文档
   - per_page: false 每一页都会导入 `mathjax / katex` 脚本（建议打开，是个坑）
   
2. 选择渲染引擎

   目前，NexT提供了两个渲染引擎：MathJax和KaTeX。
   
   {% tabs math, 2 %}
   <!-- tab mathjax -->
   
   - 需要卸载原始渲染器
   {% code %}
   npm un hexo-renderer-marked --save 
   npm i hexo-renderer-pandoc --save 
   {% endcode %}
   
   - 打开主题配置文件渲染引擎
     {% code next\_config.yml %}
     mathjax：    
       enable：true
     {% endcode %}

     <!-- endtab -->
   
   <!-- tab katex -->
   - 需要卸载原始渲染器 
   {% code %}
   npm un hexo-renderer-marked --save 
   npm i hexo-renderer-markdown-it-plus --save 
   {% endcode %}
   
   - 打开主题配置文件渲染引擎
   {% code next\_config.yml %}
   katex:
       enable：true
   {% endcode %}
   <!-- endtab -->
   {% endtabs %}

>  Note: 除了需要的渲染器外，不需要任何其他 Hexo 数学插件，无需手动导入任何 JS 或 CSS 文件。如果您安装了`hexo-math`或插件`hexo-katex`，它们可能会与 NexT 的内置渲染引擎冲突。

# 设置 CDN

第三方插件默认通过 [jsDelivr](https://www.jsdelivr.com/) CDN 服务加载。我们也提供了其它的 CDN 服务供选择，包括著名的 [UNPKG](https://unpkg.com/) 和 [CDNJS](https://cdnjs.com/)。

> CDN 是什么？
> CDN 就是一项非常有效的**缩短时延**的技术。这个技术其实说起来并不复杂，最初的核心理念，就是**将内容缓存在终端用户附近**。
> 内容源不是远么？那么，我们就在靠近用户的地方，建一个缓存服务器，把远端的内容，复制一份，放在这里，不就OK了？
> 因为这项技术是把内容进行了分发，所以，它的名字就叫做CDN——**Content Delivery Network，内容分发网络**。
> 具体来说，CDN就是采用更多的缓存服务器（CDN边缘节点），布放在用户访问相对集中的地区或网络中。当用户访问网站时，利用全局负载技术，将用户的访问指向距离最近的缓存服务器上，由缓存服务器响应用户请求。

例如，你想要使用 `unpkg` 代替 `jsdelivr` 作为默认的 CDN 提供商，你需要在 NexT 配置文件中进行如下设置：

```yaml
vendors:
  # ...
  # Some contents...
  # ...
  plugins: unpkg
```

# 自定义样式支持

NexT 建议大家使用 Hexo 官方推荐的 Data Files 系统（Hexo 3.x 及以上）来分离个人配置，这样就可以在尽可能少地修改 NexT 工程代码的情况下进行个性化配置，方便主题升级。

## 覆盖默认翻译

如果您想自定义默认翻译，则无需修改languages目录中的翻译文件。您可以使用数据文件覆盖所有翻译。

1. 在数据文件`source/_data`创造一个`languages.yml`。

2. 插入以下代码：（注意两个空格的缩进）

   ```yaml
   # language
   zh-CN:
     # items
     reward:
       donate: 点赞
       funny: 开玩笑
   ```


## 文本结束标记

主题配置文件取消注释
```sh next\_config.yml
custom_file_path:
  postBodyEnd: source/_data/post-body-end.swig
```

在路径 `/source/_data` 下创建/修改 `post-body-end.swig`文件，并添加以下内容
```sh
<div>
    {% if not is_index %}
        <div style="text-align:center;color: #ccc;font-size:14px;">-------------本文结束<i class="fa fa-paw"></i>感谢您的阅读-------------</div>
    {% endif %}
</div>
```

## 添加背景图

首先主题配置文件取消注释
```sh next\_config.yml
custom_file_path:
  style: source/_data/styles.styl
```
在路径 `/source/_data` 下创建/修改 `styles.styl`文件，并添加以下内容

```stylus
// 添加背景 url(https://source.unsplash.com/random/1600x900); 
body {
    background:url(/images/background6.jpg);
    background-repeat: no-repeat;
    background-attachment:fixed;
    background-position:50% 50%;
    background-size:cover;
}

// 修改主体透明度
.main-inner{
    background: #fff;
    opacity: 0.95;
}


// 主页文章添加阴影效果
.post {
   margin-top: 60px;
   margin-bottom: 60px;
   padding: 25px;
   -webkit-box-shadow: 0 0 5px rgba(202, 203, 203, .5);
   -moz-box-shadow: 0 0 5px rgba(202, 203, 204, .5);
}
```

## 修改主副标题字体颜色

继续在`/source/_data/styles.styl`文件中添加，帮你挑选颜色的网站： [color-hex](http://www.color-hex.com/)

```stylus
//主标题颜色
.brand{
    color: $white
}

//副标题颜色
.site-subtitle {
    margin-top: 10px;
    font-size: 13px;
    color: #ffffff;
}
```

## 修改按钮，选择区域，代码块，表格等样式

首先主题配置文件取消注释

```sh next\_config.yml
custom_file_path:
  style: source/_data/variables.styl
```
在路径 `/source/_data` 下创建/修改 `variables.styl`文件（相当于修改主题文件 `next/source/css/_variables/base.styl`），并添加以下内容

```stylus
// Buttons
// --------------------------------------------------
$btn-default-bg                 = white;
$btn-default-color              = #49b1f5;
$btn-default-font-size          = $font-size-small;
$btn-default-border-width       = 2px;
$btn-default-border-color       = #49b1f5;
$btn-default-hover-bg           = #49b1f5;
$btn-default-hover-color        = white;
$btn-default-hover-border-color = #49b1f5;

// Selection
$selection-bg                 = #49b1f5;
$selection-color              = white;

// Code & Code Blocks
// --------------------------------------------------
$code-font-family               = $font-family-monospace;
$code-border-radius             = 3px;
$code-foreground                = $black-light;
$code-background                = #edf1ff;

// Table
// --------------------------------------------------
$table-width                    = normal;  //next默认100%
$table-border-color             = $gray-lighter;
$table-font-size                = $font-size-small;
$table-content-alignment        = left;
$table-content-vertical         = middle;
$table-th-font-weight           = 700;
$table-cell-padding             = 8px;
$table-cell-border-right-color  = $gainsboro;
$table-cell-border-bottom-color = $gray-lighter;
$table-row-odd-bg-color         = #f9f9f9;
$table-row-hover-bg-color       = $whitesmoke;
```

------

参考链接：

[hexo的next主题个性化配置](https://blog.csdn.net/weixin_44815733/article/details/88817220)
[Hexo Next主题进阶详细教程](https://blog.csdn.net/qq_31279347/article/details/82427562)
[hexo个人博客next主题优化](https://www.linjiujiu.xyz/2018/12/11/hexo%E4%B8%AA%E4%BA%BA%E5%8D%9A%E5%AE%A2next%E4%B8%BB%E9%A2%98%E4%BC%98%E5%8C%96/)
[NexT主题统一网站颜色](https://www.jianshu.com/p/2a8d399f1266)
[Hexo Theme NexT 主题个性化配置最佳实践](https://blog.csdn.net/colton_null/article/details/97622079)
[Hexo+NexT 主题配置备忘](https://blog.ynxiu.com/2016/hexo-next-theme-optimize.html)



