---
ID: 39ef142e1b7cbb12e245c59a99c60fe6
title: R手册(Visualise)--ggmap+baidumap
tags: [R,ggplot2,可视化,地图]
date: 2018-05-03 23:25:52
categories: [R,可视化]
---

# ggmap

ggmap可以轻松地从流行的在线地图服务（如Google Maps， OpenStreetMap，Stamen Maps）获得地图图块，并使用ggplot2绘制**静态地图**。
<!-- more -->

## 获取地图图块

```r
get_map(location,zoom,scale,maptype,source,...) #主函数，返回ggmap对象

get_cloudmademap()	#Get a CloudMade map(需获取密钥)
get_googlemap()	#Get a Google Map (谷歌地图国内暂不能用）
get_navermap()	#Get a Naver Map
get_openstreetmap()	#Get an OpenStreetMap
get_stamenmap()	#Get a Stamen Map
```
**主要参数**

- location：地址，经度/纬度对（按顺序）或 左/下/右/上经纬度边界框
- zoom = "auto"：地图大小
- scale = "auto"
- maptype：获取地形图，卫星图，路线图，混合图或其他地图
- source =c("google", "osm",  "stamen", "cloudmade")：地图来源
- filename = "ggmapTemp"
- color = c("color", "bw")：color ("color") or black-and-white ("bw")
- language = "en-EN"：谷歌地图语言
- api_key：api_key for cloudmade maps

> **maptype参数:**  获取地形图，卫星图，路线图，混合图或其他地图
> from googlemap： terrain/terrain-background/ satellite/roadmap/hybrid
> from stamen maps：terrain, "watercolor", and "toner" 
> from cloudmade maps)：integer (from function get_cloudmademap)


## 作图函数

```R
ggmap(ggmap, extent = "panel", base_layer, maprange = FALSE,
  legend = "right", padding = 0.02, darken = c(0, "black"), ...)

qmap(location = "houston", ...)   #将get_map和ggmap结合，用于快速制图
qmplot(lon,lat,...,data，zoom)    #相当与ggplot2函数qplot
```

**参数**

- ggmap：用ggmap对象作图(from function `get_map`)
- extent = "panel"：地图细节("normal", "device", or "panel")
- base_layer：ggplot图层
- maprange = FALSE：与base_layer配合使用，定义x和y限制
- legend = "right"：`left/right/bottom/top/bottomleft/bottomright/topleft/topright`
`none`(当extent="device"时使用)
- padding = 0.02：距离 from legend to corner(和legend一起使用)
- darken = c(0, "black")


**示例：一般作图**
```r
us <- c(left = -125, bottom = 25.75, right = -67, top = 49)
map <- get_stamenmap(us, zoom = 5, maptype = "toner-lite")
ggmap(map)
```
![](https://gitee.com/WilenWu/images/raw/master/ggplot2/us_map.png)

**示例：qmplot作图**

```r
library("dplyr")
library("forcats")

# define helper
`%notin%` <- function(lhs, rhs) !(lhs %in% rhs)

# reduce crime to violent crimes in downtown houston
violent_crimes <- crime %>% 
  filter(
    offense %notin% c("auto theft", "theft", "burglary"),
    -95.39681 <= lon & lon <= -95.34188,
     29.73631 <= lat & lat <=  29.78400
  ) %>% 
  mutate(
    offense = fct_drop(offense),
    offense = fct_relevel(offense, 
      c("robbery", "aggravated assault", "rape", "murder")
    )
  )

#qmplot函数作图
qmplot(lon, lat, data = violent_crimes, maptype = "toner-lite", geom = "density2d", color = I("red"))  #geom参数添加图层

robberies <- violent_crimes %>% filter(offense == "robbery")
qmplot(lon, lat, data = violent_crimes, geom = "blank", 
  zoom = 15, maptype = "toner-background", darken = .7, legend = "topleft"
) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", alpha = .3, color = NA) +  #添加图层
  scale_fill_gradient2("Robbery\nPropensity", low = "white", mid = "yellow", high = "red", midpoint = 650) #设置标尺
```
![](https://gitee.com/WilenWu/images/raw/master/ggplot2/qmplot.png)![](https://gitee.com/WilenWu/images/raw/master/ggplot2/qmplot2.png)

```r
#刻面
qmplot(lon, lat, data = violent_crimes, maptype = "toner-background", color = offense) + 
  facet_wrap(~ offense)

#获取欧洲地图
europe <- c(left = -12, bottom = 35, right = 30, top = 63)
get_stamenmap(europe, zoom = 5) %>% ggmap()
```
![](https://gitee.com/WilenWu/images/raw/master/ggplot2/ggmap1.png)![](https://gitee.com/WilenWu/images/raw/master/ggplot2/ggmap2.png)


## 经纬度信息

```r
 #返回经纬度信息（谷歌地图国内暂不能用）
geocode(location, output = c("latlon", "latlona", "more", "all"),   
  source = c("google", "dsk")) #参数source：地理位置编码

#经纬度反编码（谷歌地图国内暂不能用）
revgeocode(location, output = c("address", "more", "all") 
#编码信息，传递给geocode使用（谷歌地图国内暂不能用）
mutate_geocode(data，location) 
```

## 从Google Maps获取路线图

```r
route(from, to, mode = c("driving", "walking", "bicycling", "transit"),
  structure = c("legs", "route"), output = c("simple", "all")) 
  #（谷歌地图国内暂不能用）
```

## 使用Google Maps计算地图距离

```r
mapdist(from, to, mode = c("driving", "walking", "bicycling"), 
  output = c("simple", "all") #（谷歌地图国内暂不能用）
```

## Google认证

如果您拥有Google API密钥，则可以超出Google对查询的标准限制。

------

# baidumap

为百度地图API提供R接口，和ggmap一样，但从百度api而不是谷歌或openstreet获取地图。

## 密钥

需要从 [lbsyun.baidu.com](http://lbsyun.baidu.com/apiconsole/ke)申请密钥。然后在R注册你的钥匙。

```r
library(baidumap)
options(baidumap.key = 'XXX fill your key here XXX')
```

## 从坐标数据获取位置

`getLocation(location,output = "json")` 
参数ouput： 设置返回数据类型('json', 'xml')

```r
lon = matrix(c(117.93780, 24.55730, 117.93291, 24.57745, 117.23530, 24.64210, 117.05890, 24.74860), byrow=T, ncol=2)
## json 
location_json = getLocation(lon[1,], output='json')
## formatted
location = getLocation(lon[1, ], formatted = T) 
```

## 从地址获取坐标数据

`getCoordinate(address, city = NULL,output = "json",formatted = F) `#返回坐标或初始信息
参数ouput：设置返回数据类型('json', 'xml')
```r
getCoordinate('北京大学', output='xml') # xml
getCoordinate('北京大学', formatted = T) # character
getCoordinate(c('北京大学', '清华大学'), formatted = T) # matrix
```

## 获取百度地图图块

```r
getBaiduMap(location, width = 400, height = 400, zoom = 10, scale = 2,
  color = "color", messaging = TRUE) # 返回ggmap对象
```

**参数**

- location：中心坐标或位置字符，
- width, height, zoom：图块大小
- scale：像素数的乘法因子
- color ：color or "bw",(color or black-and-white)
- messaging：提示信息

```r
getBaiduMap('中国',zoom = 4) %>% ggmap
```
![China](https://gitee.com/WilenWu/images/raw/master/ggplot2/baidumap_china.png)

## 搜索
```r
getPlace(place = NULL, city = "北京", page_size = 20, pages = Inf,
  scope = 1, verbose = TRUE)  #用 baiduAPI搜索信息

getRoute(origin, destination, mode,origin_region, destination_region)  #获取路线图
```
**getRoute参数**

- origin, destination：初始位置和目的地
- mode：出行方式，'driving'(default), 'walking', or 'transit'
- region：城市
- origin_region, destination_region：不在一个城市时，设置原始城市和目的地城市

```r
>>> getPlace('麦当劳', '上海')%>%names
Get 33 records, 2 page. 
    Getting  0 th page 
    Getting  1 th page 
Done! 
[1] "name"      "address"   "lat"       "lon"       "telephone"

>>> bjMap = getBaiduMap('北京', color='bw')
>>> df = getRoute('首都国际机场', '北京南苑机场', region = '北京')
>>> ggmap(bjMap) + 
       geom_path(data = df, aes(lon, lat), alpha = 0.5, col = 'red')
```
![route](https://gitee.com/WilenWu/images/raw/master/ggplot2/baidumap_route.png)

## 百度地理编码

`geoconv(geocode, from = 3, to = 5)`  别的地理编码转化成百度地理编码

