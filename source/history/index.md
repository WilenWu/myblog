---
title: 本站更新日志
date: 2022-03-05 10:13:00
top_img: '#66CCFF' 
---

{% timeline 2022 %}
<!-- timeline Unknow -->
添加 Python MkDocs 主题文档
<!-- endtimeline -->

<!-- timeline 2022-03-05 -->
升级主题至hexo-theme-butterfly@4.1.0
安装大量插件，美化 Butterfly 主题
- 安装插件[`hexo-butterfly-charts`](https://github.com/kuole-o/hexo-butterfly-charts)，添加 hexo posts、类和标签的统计数据
- ~~安装插件[`hexo-butterfly-footer-beautify`](https://github.com/Akilarlxh/hexo-butterfly-footer-beautify)添加页脚 Github 徽标~~
- 安装插件[`hexo-butterfly-swiper`](https://github.com/Akilarlxh/hexo-butterfly-swiper)，给Butterfly主题添加首页轮播图
- 安装插件[`hexo-butterfly-tag-plugins-plus`](https://github.com/Akilarlxh/hexo-butterfly-tag-plugins-plus)，给Butterfly主题添加大量外挂标签
- 安装插件[`hexo-filter-gitcalendar`](https://github.com/Akilarlxh/hexo-filter-gitcalendar)，给`hexo`添加首页git提交日历

在归档页面引入十二生肖图标，参考大神 Akilar 教程[Archive Beautify](https://akilar.top/posts/22257072/)（需要修改源码）
<!-- endtimeline -->
{% endtimeline %}

{% timeline 2021 %}
<!-- timeline 2021-10-05 -->
更换博客主题至hexo-theme-butterfly，并添加Gitee存储库
<!-- endtimeline -->
{% endtimeline %}

{% timeline 2019 %}
<!-- timeline 2019-10-13 -->
新建个人博客，框架Hexo，主题Next
<!-- endtimeline -->
{% endtimeline %}

{% timeline 2018 %}
<!-- timeline 2018-04-30 -->
完成第一篇博客，也有若干篇同时完成
<!-- endtimeline -->
{% endtimeline %}


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