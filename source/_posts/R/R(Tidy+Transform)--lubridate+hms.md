---
ID: bf69e033e54a444ffad5a188861990c3
title: R手册(Tidy+Transform)--lubridate+hms
tags: [R,tidyverse,数据清洗,时间序列]
date: 2018-05-01 23:06:29
categories: [R,Tidy+Transform]
cover: /img/lubridate.png
top_img: /img/lubridate-hms.png
description: for date and times
---

日期和时间变量处理包

<!-- more -->

# lubridate:  Dates and times

## Date-times

![](https://gitee.com/WilenWu/images/raw/master/lubridate/date-time.png)

- **date-time** 是时间线上的一个点，以秒为存储单位，起始于1970-01-01 00:00:00 UTC   # `base::as.POSIXct(x,farmat,origin)`
```r
dt <- as_datetime(1511870400)
## "2017-11-28 12:00:00 UTC"
```
- **date**是从1970-01-01开始的天数 # `base::as.Date(x,"%Y-%m-%d")`
```r
d <- as_date(17498)
## "2017-11-28"
```
- **hms**是从00:00:00开始的秒数
```r
t <- hms::as.hms(85)
## 00:01:25
```

1. 解析时间 (Convert strings or numbers to date-times)

- 识别出数据中 year (**y**), month (**m**), day (**d**), hour (**h**), minute (**m**) and second (**s**) 元素的顺序
- 用下面的函数，函数的名字代表顺序，函数接收**多种多样**的输入形式
![](https://gitee.com/WilenWu/images/raw/master/lubridate/ymd-hms.png)

**date_decimal**(decimal, tz = "UTC")  解析小数形式
*date_decimal(2017.5)*

**fast_strptime**() 快速解析字符格式
*fast_strptime('9/1/01', '%y/%m/%d')*

**parse_date_time**() 简单解析字符格式
*parse_date_time("9/1/01", "ymd")*

**获取当前日期/时间**
**today(tzone = "")**   Date object （base::Sys.Date() ）
**now(tzone = "")**   POSIXct object（base::date() ）

2. 获取/设置时间组件

```r
>>> d <- ymd("2017-11-28")
>>> day(d)
28
>>> day(d) <- 1
>>> d 
"2017-11-01"
```
![floor](https://gitee.com/WilenWu/images/raw/master/lubridate/lubridate-parts.png)

![](https://gitee.com/WilenWu/images/raw/master/lubridate/am-pm.png)

**quarter**(x, with_year = FALSE, fiscal_start = 1)  季度
**semester**(x, with_year = FALSE)  （尤指美国的大专院校的） 学期，半学年

**dst(x)**  夏令时(daylight savings time)? 
**leap_year(x)** 闰年?
**update**(object, ..., simple = FALSE)
*update(dt, mday = 2, hour = 1)*


## Round Date-times

```r
floor_date(x, unit = "second") # 向下滚动到最近的单位
round_date(x, unit = "second") # 滚动到最近的单位
ceiling_date(x, unit = "second",change_on_boundary = NULL) #向上滚动到最近的单位
rollback(dates, roll_to_first =FALSE, preserve_hms = TRUE) #滚动到上月最后一天或本月第一天
```
**图示**
 ![](https://gitee.com/WilenWu/images/raw/master/lubridate/floor_date.png)
`floor_date(dt, unit = "month")`
![round](https://gitee.com/WilenWu/images/raw/master/lubridate/ceiling_date.png)
`round_date(dt, unit = "month")`
![ceiling](https://gitee.com/WilenWu/images/raw/master/lubridate/round_date.png)
`ceiling_date(dt, unit = "month")`

## Stamp Date-times

自定义一个Date-times的字符显示模板函数
`stamp() , stamp_date() and stamp_time()`

```r
# 1. 创建模板函数
sf <- stamp("Created Sunday, Jan 17, 1999 3:34")
# 2. 使用模板
sf(ymd("2010-04-05"))
[1] "Created Monday, Apr 05, 2010 00:00" 
```

## Time Zones

R可以识别 ~600 个时区，每一个区块都有编码时区，夏令时，历史日历（time zone, Daylight Savings Time, and historical calendar）变量，R为每个赋值一个 time zone向量。

通常用**UTC** 时区来避开夏令时
**OlsonNames()**  返回完整的时区名列表
**with_tz()** 返回另一个时区的date-time
**force_tz** 强制抓换时区，date-time数字不变
![](https://gitee.com/WilenWu/images/raw/master/lubridate/lubridate-tz.png)

```r
meeting <- ymd_hms("2011-07-01 09:00:00", tz = "Pacific/Auckland")
with_tz(meeting, "America/Chicago")
#> [1] "2011-06-30 16:00:00 CDT"

mistake <- force_tz(meeting, "America/Chicago")
with_tz(mistake, "Pacific/Auckland")
#> [1] "2011-07-02 02:00:00 NZST"
```

## Time Spans

lubridate提供三种类型的时间块

- **period**：人类定义的时间单位， like years, months, and days.
- **duration**：以秒为单位的时间跨度（**difftime**是base R 中一种特殊的duration），当这个数字很大时，period也会以较大的单位显示一个估计值，底层总是为固定的秒数，常见的转换有Minutes = 60 seconds, hours = 3600 seconds, days = 86400 seconds, weeks = 604800，由于其变化性，未使用大于周的单位。
- **interval**：用指定的开始和结束日期创建的时间间隔。如果开始日期在结束日期之前发生，则间隔将为正值。否则，为负值。


1. Period and Duration

period|duration
---|---
period(num = NULL, units = "second", ...)<br>`period(5, unit = "years")`|duration(num = NULL, units = "second", ...)<br>duration(5, unit = "years") 
as.period(x, unit) |as.duration(x, …) 
is.period()|is.duration(),is.difime()
period_to_seconds(x) |make_difime(x)<br>`make_difftime(99999)`
seconds_to_period(x)|

period unit|说明|duration unit|说明
---|---|---|---
years(x = 1)| years|dyears(x = 1) |31536000 seconds
months(x)| months|
weeks(x = 1)| weeks|dweeks(x = 1) |604800 seconds
days(x = 1)| days|ddays(x = 1) |86400 seconds
hours(x = 1)| hours|dhours(x = 1) |3600 seconds
minutes(x = 1)| minutes|dminutes(x = 1) |60 seconds
seconds(x = 1)| seconds|dseconds(x = 1)| seconds
milliseconds(x = 1)| milliseconds|dmilliseconds(x = 1)| 10**3 seconds
microseconds(x = 1)| microseconds|dmicroseconds(x = 1)| 10**6 seconds
nanoseconds(x = 1)| nanoseconds|dnanoseconds(x = 1)| 10**9 seconds
picoseconds(x = 1)| picoseconds |dpicoseconds(x = 1)| 10**12 seconds 

**示例**：period和duration也可由组件的数学计算得到
```r
period(c(90, 5), c("second", "minute"))  #  "5M 90S"
period(second = 90, minute = 5)  #  "5M 90S"
months(3) + days(12) # "3m 12d 0H 0M 0S"

ddays(14)  # "1209600s (~2 weeks)"
```

2. Interval

**interval**：用指定的开始和结束日期创建的时间间隔。如果开始日期在结束日期之前发生，则间隔将为正值。否则，为负值。

创建|说明
---|---
interval(start, end, tzone = tz(start))|
start %--% end|符号创建
as.interval(x, start, …)|参数x：duration, difftime, period, or numeric object
base::difftime(time1, time2, tz,units ="auto")|
is.interval(x)|判断函数
**组件**|
int_start(int)<br>int_start(int) <- value|date-time
int_end(int)<br>int_end(int) <- value|结束date-time

> interval的长度由使用时的对象(period, duration)划定

```r
s<- ymd("2017-01-01"); e <- ymd("2017-11-28")
interval(s, e) ## 2017-01-01 UTC--2017-11-28 UTC
s %--% e ## 2017-01-01 UTC--2017-11-28 UTC

as.interval(days(1), start = now()) 
##  2018-06-03 15:34:25 CST--2018-06-04 15:34:25 CST
```

函数|说明
---|---
a %within% b|logical，a在b区间内,{a: interval or date-time object,b: interval}
int_length(int)| Length in seconds
int_flip(int)| 反转
int_standardize(int)|标准化
int_shift(int, by)|平移interval
int_overlaps(int1, int2)|logical,判断int1和int2是否重叠
int_aligns(int1, int2)|int1, int2是否分享相同边界
int_diff(times)|用n个date-time分割成n-1个intervals向量

```r
>>> dt <- today()
>>> c(dt, dt + 100, dt + 1000)%>%int_diff()
2018-06-03 UTC--2018-09-11 UTC     2018-09-11 UTC--2021-02-27 UTC

>>> dt1 <- interval(dt,dt+1)
>>> int_shift(dt1, days(-1))
2018-06-02 UTC--2018-06-03 UTC
```

## Math with Date-times

1. 夏令时(Daylight Saving Time, DST)

夏令时（Daylight Saving Time：DST），又称“日光节约时制”和“夏令时间”，是一种为节约能源而人为规定地方时间的制度，在这一制度实行期间所采用的统一时间称为“夏令时间”。一般在天亮早的夏季人为将时间调快一小时，可以使人早起早睡，减少照明量，以充分利用光照资源，从而节约照明用电。各个采纳夏令时的国家具体规定不同。目前全世界有近110个国家每年要实行夏令时。

![](https://gitee.com/WilenWu/images/raw/master/lubridate/lubridate-math.png)

2. Date-times数学运算

Date-times数学运算依赖于计算的时间点，位于夏令时开始、结束时间、润年、闰秒时，计算有所不同。
并不是所有的年都是365d，不是所有的分钟都是60s，普通的数据运算可能会产生虚拟日期，lubridate提供了操作符和函数。

数学运算|说明
---|---
`%m+%` and `%m-%` |滚动到前一月的最后一天
add_with_rollback(e1, e2, roll_to_first =TRUE) |滚动到新一月的第一天
```r
jan31 <- ymd(20180131)
jan31 + months(1)
## NA 

jan31 %m+% months(1)
## "2018-02-28"

add_with_rollback(jan31, months(1),roll_to_first = TRUE)
## "2018-03-01"
```

----------

# hms: time-of-day values

hms包提供了一个简单的类来存储持续时间或时间值，并以`hh:mm:ss`格式显示它们。该类旨在简化与数据库，电子表格和其他数据源的数据交换。

`hms(seconds, minutes, hours, days)`
`as.hms(x)` x可以是字符，数字等

**Usage**
```r
library(hms)
hms(56, 34, 12)
#> 12:34:56
as.hms(1)
#> 00:00:01
as.hms("12:34:56")
#> 12:34:56
as.hms(Sys.time())
#> 16:45:12.828186
as.POSIXct(hms(1))
#> [1] "1970-01-01 00:00:01 UTC"

data.frame(hours = 1:3, hms = hms(hours = 1:3))
#>   hours      hms
#> 1     1 01:00:00
#> 2     2 02:00:00
#> 3     3 03:00:00
```

