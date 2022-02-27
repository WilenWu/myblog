---
title: Python手册(Standard Library)--日期时间模块
tags:
  - Python
categories:
  - Python
  - Python标准库
cover: /img/python-datetime-cover.png
top_img: /img/python-top-img.svg
abbrlink: 592885c8
date: 2018-05-09 22:49:26
description: 日基础包：datetime, time, calendar
---

Python 程序能用很多方式处理日期和时间，转换日期格式是一个常见的功能。
Python 提供了一个 time 和 calendar 模块可以用于格式化日期和时间。
时间间隔是以秒为单位的浮点小数。
每个时间戳都以自从1970年1月1日午夜（历元）经过了多长时间来表示。


<!-- more -->

# datetime

| datetime模块定义了6个类 |  |
| ------------- | --------- |
| datetime.date | 表示日期的类 |
| datetime.datetime| 表示日期时间的类  |
| datetime.time | 表示时间的类 |
| datetime.timedelta  | 表示时间间隔 |
| datetime.tzinfo  | 时区的相关信息|
| datetime.timezone|将tzinfo抽象基类实现为UTC固定偏移量的类|

```python
from datetime import date,datetime,time,timedelta
```

## datetime.timedelta

```python
datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
```

|**属性**|	read-only|
|------|-------|
|days  |Between -999999999 and 999999999 inclusive
|seconds  |Between 0 and 86399 inclusive
|microseconds  |Between 0 and 999999 inclusive
**方法**|
total_seconds()|返回持续时间中包含的总秒数


## datetime.date

`date(year,month,day)`返回 'year-month-day' 
`date.today()`返回today（datetime.date类）
`date.fromtimestamp(timestamp)`由时间戳转化
`date.fromordinal(ordinal)`
> timestamp（时间戳）是指格林威治时间1970年01月01日00时00分00秒起至现在的总秒数。

|**属性**|	read-only|
|------|-------|
|date.year|Between MINYEAR and MAXYEAR inclusive.
|date.month|Between 1 and 12 inclusive.
|date.day|Between 1 and the number of days in the given month of the given year.

**运算**

```python
date2 = date1 + timedelta
date2 = date1 - timedelta
timedelta = date1 - date2
date1 < date2	
```

| **方法** | 说明 |
|:---|:---|
|replace(year=self.year, month=self.month, day=self.day)|替换给定日期，但不改变原日期|
|timetuple()|返回 time.struct_time对象(时间元祖)|
|toordinal()|回归原始日期|
|weekday()|Return the day of the week as an integer, where Monday is 0 and Sunday is 6
| isoweekday()|Return the day of the week as an integer, where Monday is 1 and Sunday is 7  |
| isocalendar()  | Return a 3-tuple, (ISO year, ISO week number, ISO weekday) |
| isoformat() | Return a string 'YYYY-MM-DD'  |
|ctime()|return a string (`date(2002, 12, 4).ctime() == 'Wed Dec 4 00:00:00 2002'`)
|strftime(format)| 返回指定格式字符


## datetime.datetime


```python
datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0) 
```

**创建**|说明
:---|:---
`datetime.today()`| 返回当天date
`datetime.now()`|返回当前系统时间  
`datetime.fromtimestamp(timestamp, tz=None)`|根据时间戮返回datetime对象 
`datetime.fromordinal(ordinal)`|
`datetime.combine(date, time, tzinfo=self.tzinfo)`|date对象和time对象组合成新的datetime对象
`datetime.strptime(date_string, format)`|**字符串格式创建**
`datetime.strftime()`| 转换为字符格式


**属性** |(read-only)
:---|:---
year|Between MINYEAR and MAXYEAR inclusive.
month|Between 1 and 12 inclusive.
day|Between 1 and the number of days in the given month of the given year.
hour|In range(24).
minute|In range(60).
second|In range(60).
microsecond|In range(1000000).（微秒）
tzinfo|时区
fold|


**运算**
```python
datetime2 = datetime1 + timedelta
datetime2 = datetime1 - timedelta
timedelta = datetime1 - datetime2
datetime1 < datetime2
```

| **方法**  |  说明 |
|:------|:------|
| date()|返回date对象|
| time()|返回time对象|
| replace()  | 替换 |
| ctime() | 返回格式如 Sun Apr 16 00:00:00 2017 |
| timetuple()| 返回time.struct_time对象  |
|utctimetuple()|
|toordinal()||
|timestamp()|返回时间戳(float)|
|weekday()|Monday is 0 and Sunday is 6
|isoweekday()|Monday is 1 and Sunday is 7  |
|isocalendar()  | Return a 3-tuple, (ISO year, ISO week number, ISO weekday) |
|isoformat(sep='T', timespec='auto') | Return a string 'YYYY-MM-DDTHH:MM:SS.mm'  |
|ctime()|return a string ('Wed Dec  4 20:30:40 2002')
| strftime(format) | 由日期格式转化为字符串格式|
> 关于时区的方法暂时不计入

## datetime.time

```python
time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
```

属性| (read-only)
:---|:---
hour|In range(24).
minute|In range(60).
second|In range(60).
microsecond|In range(1000000).
tzinfo|
fold|

| **方法**  | 说明 |
|:------|:------|
|replace()|  |
|isoformat(timespec='auto')||
|strftime(format)|转字符格式|
| tzname() | 返回时区名字 |
| utcoffset() | 返回时区的时间偏移量|
|dst()||

## python中时间日期格式化符号

| 格式 | 说明    | Example    |
| :---- |:---------- | :---------------- |
| %a   | 周日期缩写   | Sun, Mon, …, Sat (en_US); |
| %A   | 周日期全称   | Sunday, Monday, …, Saturday (en_US);|
| %w   | 周数字  | 0, 1, …, 6 |
| %d   | 月中天数| 01, 02, …, 31   |
| %b   | 月份缩写| Jan, Feb, …, Dec (en_US); |
| %B   | 月份全称| January, February, …, December (en_US);  |
| %m   | 月份数字| 01, 02, …, 12   |
| %y   | 年数字，两位 | 00, 01, …, 99   |
| %Y   | 年数字，四位 | 0001, 0002, …, 2013, 2014, …, 9998, 9999 |
| %H   | 24小时制| 00, 01, …, 23   |
| %I   | 12小时制| 01, 02, …, 12   |
| %p   | AM or PM| AM, PM (en_US); |
| %M   | 分钟    | 00, 01, …, 59   |
| %S   | 秒 | 00, 01, …, 59   |
| %f   | 微秒    | 000000, 000001, …, 999999 |
| %z   | UTC offset in the form +HHMM or -HHMM | (empty), +0000, -0400, +1030   |
| %Z   | 时区名  | (empty), UTC, EST, CST    |
| %j   | 年中的天数   | 001, 002, …, 366|
| %U   | 年中的周日期（周日为第一天）| 00, 01, …, 53   |
| %W   | 年中的周日期（周一为第一天）| 00, 01, …, 53   |
| %c   | date and time| Tue Aug 16 21:30:00 1988 (en_US);   |
| %x   | date    | 08/16/88 (None)<br>08/16/1988 (en_US)    |
| %X   | time    | 21:30:00 (en_US);    |
| %%   | %'字符  | %|
| %G   | ISO 8601 year| 0001, 0002, …, 2013, 2014, …, 9998, 9999 |
| %u   | ISO 8601 weekday  | 1, 2, …, 7 |
| %V   | ISO 8601 week| 01, 02, …, 53   |



----------


# time

| 函数 |  |
| ------------- | --------- |
| time.clock()  | 用CPU花费的时间（秒） |
| time.sleep(secs) |  |

## time.struct_time(时间戳类)

| **属性**| 说明 |
| :------------- | :--------- |
| tm_year | yyyy|
| tm_mon  | 1 到 12 |
| tm_mday | 1 到 31 |
| tm_hour | 0 到 23 |
| tm_min  | 0 到 59 |
| tm_sec  | 0 到 61 (60或61  是闰秒)|
| tm_wday | 0到6 (0是周一)|
| tm_yday | 1 到 366|
| tm_isdst| 1(夏令时)0(不是夏令时)-1(未知)  |

----------


# calendar

星期一是默认的每周第一天，星期天是默认的最后一天。更改设置需调用`calendar.setfirstweekday()`函数。

| calendar函数|  |
| :------------- |:--------- |
| calendar.calendar(year,w=2,l=1,c=6)| 年日历 |
|calendar.firstweekday()|返回当前每周起始日期的设置|
| calendar.month(year,month)| 月日历 |
| calendar.isleap(year)  | 是否闰年|
| calendar.leapdays(y1,y2)  | 返回在Y1，Y2两年之间的闰年总数  |
|calendar.monthrange(year, month)|Returns weekday of first day of the month and number of days in month|
|calendar.weekday(year,month,day)|Returns the day of the week (0 is Monday) |






