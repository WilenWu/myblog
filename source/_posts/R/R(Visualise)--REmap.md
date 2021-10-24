---
ID: 72a689c6504caa9db44e4f114cb3b23a
title: R手册(Visualise)--REmap
tags: [R,地图,可视化]
date: 2018-06-01 14:24:23
categories: [R,可视化]
cover: /img/remap-nanchang.png
top_img: /img/world-map.jpeg
description: 基于Echarts的简易动态交互地图
---

**REmap**:（简单动态地图）Create html maps by Echarts

[REmap](http://langdawei.com/REmap/2015/07/intro)是一个基于 [Echarts](http://echarts.baidu.com) 的R包。主要的目的是为广大数据玩家提供一个简便的，可交互的地图数据可视化工具。目前托管在[github](https://github.com/lchiffon/REmap)

<!-- more -->

# 获取经纬度

`get_geo_position(city_vec)` 获得城市向量的经纬度

```r
city_vec = c("北京","Shanghai","广州")
get_geo_position (city_vec)
        lon      lat     city
1  116.6212 40.06107     北京
2  121.4786 31.21562 Shanghai
3  113.3094 23.39237     广州
```

# 迁移图（无缩放中国地图）

`remap(mapdata,title = "", subtitle = "", theme = get_theme("Dark")) `

> 参数
> mapdata格式：data.frame(出发地, 目的地)
> theme：Dark/Bright/Sky

# 填色地图（+迁移图）

```r
mapNames(mapType) # 获得所有的maptype（china/world/省份名）

remapC(data,                     #data.frame(地名, 数值)
  maptype = 'china',             #china/world/省份名
  color = c('1e90ff','f0ffff'),  #填充色阶(“skyblue”,etc)
  theme = get_theme("Bright"),
  title = "",subtitle = "",
  markLineData = NULL,            #data.frame(出发地, 目的地)
  markPointData = NULL,           #地名向量
  markLineTheme = markLineControl(),
  markPointTheme = markPointControl() )
```
**示例**
```r
data = data.frame(country = mapNames("world"),
                   value = 5*sample(178)+200)
remapC(data,maptype = "world",color = 'skyblue')

remapC(chinaIphone,
        title = "remapC实例地图",
        theme = get_theme('none',backgroundColor = '#fff',
                          titleColor = "#1b1b1b",
                          pointShow = T),
        max = 2000,
        markLineData = demoC,
        markPointData = demoC[,2])
```
![](https://gitee.com/WilenWu/images/raw/master/r-echarts/remap_world.png)![](https://gitee.com/WilenWu/images/raw/master/r-echarts/remap_china.png)


# 调用baiduAPI绘图

`remapB` 是用于创建一个以百度地图为底图的recharts效果，有以下特点:

- 支持滚轮缩放,拖拽地图
- 详细的地图效果
- 可视化主要以标线与标点的形式做出

```r
remapB(center = c(lon,lat)   #地图中心向量
  zoom = 5,                  #地图尺寸(country :5, city:15)
  color = "Bright",title = "",subtitle = "",
  color,                    #color: Bright", "Blue", "light", "dark", "redalert", "googlelite", "grassgreen", "midnight", "pink", "darkgreen", "bluish", "grayscale", "hardedge"
  markLineData = NA,        #data.frame(出发地, 目的地)
  markPointData = NA,       #地名向量
  markLineTheme = markLineControl(),
  markPointTheme = markPointControl(),
  geoData = NA)              #similar as get_geo_position()
```

```r
remapB(title = "Remap:  百度迁徙模拟", markPointData = demoC[,2])
```

![](https://gitee.com/WilenWu/images/raw/master/r-echarts/baidumap.png)


# For shiny

```r
REmapOutput(outputId, inline = FALSE, container,...)`

renderREmap(expr, env = parent.frame(),height = "800px", width = "100%", quoted = FALSE, func = NULL)
```
