---
title: R(Visualise)--gganimate(ggplot2 extensions)
categories:
  - R
  - Visualise
  - 'ggplot2 extensions'
tags:
  - R
  - ggplot2
cover: /img/gganimate-cover.png
top_img: /img/ggplot2-top-img.png
description: 用ggplot2创建简单的动画
abbrlink: a8b6353d
date: 2018-05-28 16:45:17
emoji: heart
---

# [gganimate][gganimate]: Create easy animations with ggplot2

[gganimate]: https://github.com/dgrtwo/gganimate

![gganimate-personifyr](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/gganimate-personifyr.jpg)

## ggplot对象

```R
gganimate(p=last_plot(),aes(frame),filename = NULL, interval=1,title_frame = TRUE )
```

**参数：**

- aes(frame)：包括时间维度
- filename：输出文件
- interval：动画间隔
- title_frame：是否当前frame值附加到标题

**aes参数:**
cumulative = TRUE：设置是否路径累积

## geom图层

geom必须设置aes(group)，group为frame的变量，用来指定时间维度

**for example:**

```R
  p <- ggplot(gapminder, aes(gdpPercap, lifeExp, size = pop, frame = year)) +
    geom_point() +geom_smooth(aes(group = year)) +
    facet_wrap(~continent, scales = "free") +
    scale_x_log10()
  gganimate(p, "output.swf", interval=.2)
```

  ![gganimate-demo](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/ggplot2/gganimate-demo.gif)



