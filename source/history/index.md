---
title: 本站更新日志
date: 2022-03-05 10:13:00
top_img: false
---

{% timeline 2022 %}
<!-- timeline Unknow -->
添加 Python MkDocs 主题文档
<!-- endtimeline -->

<!-- timeline 2022-03-05 -->
1. 升级主题至hexo-theme-butterfly@4.1.0
2. 安装大量插件，美化 Butterfly 主题
   - 安装插件[`hexo-butterfly-charts`](https://github.com/kuole-o/hexo-butterfly-charts)，添加 hexo    posts、类和标签的统计数据
   - ~~安装插件[`hexo-butterfly-footer-beautify`](https://github.com/Akilarlxh/hexo-butterfly-footer-beautify)添加页脚 Github 徽标~~
   - 安装插件[`hexo-butterfly-swiper`](https://github.com/Akilarlxh/hexo-butterfly-swiper)，给   Butterfly主题添加首页轮播图
   - 安装插件[`hexo-butterfly-tag-plugins-plus`](https://github.com/Akilarlxh/hexo-butterfly-tag-plugins-plus)，给Butterfly主题添加大量外挂标签
   - 安装插件[`hexo-filter-gitcalendar`](https://github.com/Akilarlxh/hexo-filter-gitcalendar)，给`hexo`添加首页git提交日历
3. 在归档页面引入十二生肖图标，参考大神 Akilar 教程[Archive Beautify](https://akilar.top/posts/22257072/)（需要修改源码）
<!-- endtimeline -->
{% endtimeline %}

{% timeline 2021 %}
<!-- timeline 2021-09-25 -->
1. 更换博客主题至hexo-theme-butterfly，并添加Gitee存储库
2. 引入阿里图标库 [iconfont](https://www.iconfont.cn/)
3. 按照主题官方文档，安装以下美化插件
   - 安装 `hexo-generator-index` 支持文章置顶功能
   - 安装 [`hexo-generator-search`](https://github.com/PaicHyperionDev/hexo-generator-search) 配置本   地搜索系统
   - 卸载掉 marked 插件，然后安装新的 [`@upupming/hexo-renderer-markdown-it-plus`](https://github.com/upupming/hexo-renderer-markdown-it-plus) 渲染插件，使用了 @neilsustc/markdown-it-katex 来渲染数学方程。
   - 安装 `hexo-wordcount` 为主题配上子数统计等特征
   - 安装支持 APlayer 播放器的 Hexo 标签插件 [`hexo-tag-aplayer`]((https://github.com/MoePlayer/hexo-tag-aplayer)
   - ~~安装电影界面插件 [`hexo-butterfly-douban`](https://github.com/jerryc127/butterfly-plugins/tree/main/hexo-butterfly-douban)~~
   - 安装 [`hexo-generator-feed`](https://github.com/hexojs/hexo-generator-feed) 生成RSS文件的插件
   - 安装 [`hexo-filter-nofollow`](https://github.com/hexojs/hexo-filter-nofollow) 有效地加强网站SEO   和防止权重流失
   - 安装 [`hexo-generator-sitemap`](https://github.com/hexojs/hexo-generator-sitemap) 和 [`hexo-generator-baidu-sitemap`](https://github.com/coneycode/hexo-generator-baidu-sitemap) 生成sitemap的插件
   - 引入 permalink 永久链接插件 [`hexo-abbrlink`](https://github.com/rozbo/hexo-abbrlink)
<!-- endtimeline -->
{% endtimeline %}

{% timeline 2019 %}
<!-- timeline 2019-09-07 -->
新建个人博客，框架Hexo，主题Next
<!-- endtimeline -->
{% endtimeline %}

{% timeline 2018 %}
<!-- timeline 2018-04-30 -->
完成第一篇博客
<!-- endtimeline -->
{% endtimeline %}

------

本博客使用 Butterfly 主题，主要通过以下两种途径魔改优化：
1. 下载已发布在 NPM 的hexo插件使用
2. 如想添加额外的 `js/css/meta` 等等东西，可以通过主题配置文件`Inject`里添加，支持添加到head(``</body>``标签之前)和bottom(`</html>`标签之前)。
   
    额外的 `js/css` 文件先放入博客根目录的`source\js`或`source\css`目录下，然后修改 Butterfly 配置文件的 inject 引入
    
    ```yaml
    inject:
      head:
        - <link rel="stylesheet" href="/css/custom.css">   # 加载css文件
      bottom:
        - <script src="/js/custom.js"></script>  # 加载js文件
    ```
    
    