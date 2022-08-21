---
title: 大数据手册(Oracle)--Oracle SQL(Introduction)
date: '2021-05-03 16:18:01'
categories:
  - Big Data
  - Oracle
tags:
  - 大数据
  - oracle
  - SQL
cover: /img/introduction-to-oracle-sql.png
top_img: /img/SQL.png
abbrlink: eb28907e
description:
---

# 引言

结构化查询语言(Structured Query Language)简称SQL，是一种数据库查询和程序设计语言，用于存取数据以及查询、更新和管理关系数据库系统（RDBMS）。

主流关系型数据库，比如Oracle, MS SQL Server 以及 MySQL，其数据库语言都是基于 SQL-92 标准开发的。

 SQL 语言包括：

- 数据定义语言 (Data Definition Language, DDL)：定义数据库对象
  - *CREATE* - 创建对象
  - *ALTER* - 修改对象
  - *DROP* - 删除对象
- 数据查询语言 (Data Query Language, DQL)
  - *SELECT* - 从数据库表中获取数据
- 数据操作语言 (Data Manipulation Language, DML) ：修改数据
  - *UPDATE* - 更新数据
  - *DELETE* - 删除数据
  - *INSERT* - 插入数据
- 数据控制语言 (Data Control Language, DCL)：设置或修改对象权限
  - *GRANT* - 授予权限
  - *DENY* - 拒绝权限
  - *REVOKE* - 撤销权限
- 事务控制语言 (Transaction Control Language, TCL)：控制事务的执行
  - *COMMIT* - 提交事务
  - *ROLLBACK* - 回滚事务

# Oracle 登录

Oracle 数据库可以用以下客户端工具进行连接访问

- SQL*PLUS
- SQL Developer

Oracle 系统用户

- sys、system : 系统用户
- sysman
- scott：默认密码 tiger

```bash
# 系统用户登录
sqlplus [username/password][@server][as sysdba|sysoper]
```

> server ：自己设置的服务器名
> as sysdba：提供管理员权限

```sql
-- 启用用户 scott
SQL> alter user scott account unlock
-- 切换用户
SQL> connect username/password
-- 退出
SQL> exit
-- 显示当前用户名
SQL> show user
```

> ```bash
> # mysql 的启动和关闭
> service mysql start
> service mysql stop
> # mysql 登录
> mysql -u root -p
> ```

# 数据类型

| 文本类型            | 描述                            |
| :------------------ | :------------------------------ |
| CHAR(size)          | 固定长度字符，size 规定字符长度 |
| NCHAR(size)         | 固定长度字符，支持unicode       |
| VARCHAR2(max_size)  | 可变长度字符                    |
| NVARCHAR2(max_size) | 可变长度字符，支持unicode       |

| 数值类型    | 描述                                                         | 示例                             |
| ----------- | ------------------------------------------------------------ | -------------------------------- |
| NUMBER(p,s) | p 代表有效数字，s 为小数位数<br/>number 也有几个别名，例如 `INT = NUMBER(38), DECIMAL(p,s) = NUMBER(p,s)` | NUMBER(5, -2)<br/>四舍五入到百位 |
| FLOAT(size) | 存储二级制数字                                               |                                  |

| 日期类型  | 描述                       |
| --------- | -------------------------- |
| DATE      | 日期类型，精确到秒         |
| TIMESTAMP | 时间戳，可以达到纳秒级精度 |
| INTERVAL  | 时间间隔                   |

在Oracle数据库系统中使用数字存储日期，包括世纪、年、月、日、小时、分、秒。标准日期格式是`DD-MON-YY`，例如`01-JAN-17`
想要将标准日期格式更改为`YYYY-MM-DD`，那么可以使用`ALTER SESSION`语句来更改`NLS_DATE_FORMAT`参数的值，如下所示：

```sql
ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD';
```

| 大对象类型 | 描述                |
| ---------- | ------------------- |
| BLOB       | 二进制存储，最大4GB |
| CLOB       | 字符型存储，最大4GB |


# 运算符和表达式

| 运算符                         | 描述                                                         |
| :----------------------------- | :----------------------------------------------------------- |
| `+, -, *, /`                   | 算数运算符                                                   |
| `=,<>,>,<,>=,<=`               | 比较运算符                                                   |
| `AND, OR, NOT`                 | 逻辑运算符                                                   |
| `IS [NOT] NULL`                | 判断空值                                                     |
| `[NOT] BETWEEN ... AND`        | 范围查询                                                     |
| `[NOT] LIKE pattern`           | 模糊查询，配合通配符使用                                     |
| `RLIKE pattern`                | 使用正则表达式匹配                                           |
| `REGEXP pattern`               | 同 `RLIKE`.                                                  |
| `[NOT] IN (value1,value2,...)` | 在值列表中查找                                               |
| `[NOT] IN (subquery)`          | 在子查询中查找                                               |
| `ANY/SOME/ALL`                 | 将值与列表或子查询进行比较。<br/>它必须以另一个逻辑运算符(例如：`=`，`>`，`<`)作为前缀。 |
| `[NOT] EXISTS`                 | 子查询中是否返回数据                                         |

| 通配符 | 描述               |
| :----- | :----------------- |
| `%`    | 替代一个或多个字符 |
| `_`    | 仅替代一个字符     |


# 常用函数

> Oracle 中常用 dual目标表测试

## 日期函数

| 函数                      | 说明                                     |
| :----------------------------- | :---------------------------------------- |
| CURRENT_DATE                  | 当前日期，默认 "DD-MON-RR"               |
| CURRENT_TIMESTAMP             | 当前时间戳，默认 "DD-MON-RR HH:MI:SS.FF" |
| SYSDATE                       | 系统时间                                 |
| SYSTIMESTAMP                  | 系统时间戳                               |
| ADD_MONTHS(date,n)       | 日期+月数                                |
| NEXT_DATE(date,weekchar)      | 下周某一天的日期                       |
| LAST_DAY(date)                | 当月最后一天                             |
| MONTHS_BETWEEN(date1,date2)   | 两个日期间隔月数                        |
| date1 - date2                 | 两个日期间隔天数                         |
| EXTRACT(fmt FROM date) | 提取日期组件                             |
| TRUNC(date[, fmt])              | 截断到所在期间的第一天                           |
| ROUND(date[, fmt])              | 四舍五入到所在期间的第一天                       |

> fmt 参数：YEAR, MONTH, DAY, HOUR, MINUTE, SECOND

```sql
select NEXT_DATE(SYSDATE,'Tuesday') FROM dual; -- 返回下周二的日期
select EXTRACT(YEAR FROM SYSDATE) FROM dual;  -- 返回年份
select EXTRACT(HOUR FROM TIMESTAMP'2021-05-05 16:45:12') FROM dual;  -- 返回小时数
```
Oracle 获取本周、本月、本季、本年的第一天和最后一天
```sql
--本周
select trunc(sysdate, 'd') from dual; -- 本周第一天，即周日

--本月
select trunc(sysdate, 'mm') from dual;
select last_day(trunc(sysdate)) from dual;

--本季
select trunc(sysdate, 'Q') from dual;
select add_months(trunc(sysdate, 'Q'), 3) - 1 from dual;

--本年
select trunc(sysdate, 'yyyy') from dual;
select add_months(trunc(sysdate, 'yyyy'), 12) - 1 from dual;
```

## 聚合函数

| 聚合函数(Aggregate) | 说明（NULL不计入） |
| :------------------- | :------------------ |
| AVG()               | 平均值             |
| SUM()               | 总和               |
| COUNT(*)            | 返回表中的记录数   |
| COUNT(col)          | 指定列的值的数目   |
| MAX()               | 最大值             |
| MIN()               | 最小值             |
| WM_CONCAT()         | 分组收集           |
| CORR(X,Y)           | 相关系数           |

```sql
select job,wm_concat(ename) from employee group by job;
```

## 数学函数


| 函数                 | 描述                     |
| :------------------------ | :------------------------ |
| ROUND(x[,n])             | 四舍五入，默认取整         |
| TRUNC(x[,n])             | 截断，默认取整 |
| CEIL(x)                | 取上限                     |
| FLOOR(x)               | 取下限                     |
| ABS(x)                 | 绝对值                   |
| MOD(x,y)                 | 取余数                   |
| POWER(x,y)               | 乘方                     |
| SQRT(x)                  | 平方根                   |
| SIN(), ASIN(),TAN(), ... | 三角函数                 |

## 字符函数

| 函数                    | 说明                                   |
| :--------------------------- | :-------------------------------------- |
| UPPER(x)                 | 换为大写                               |
| LOWER(x)                 | 转换为小写                             |
| INITCAP(x)               | 首字母大写                             |
| SUBSTR(x, start[, length]) | 提取子字符串(默认到结尾)<br/> start 参数可以取负数 |
| LENGTH(char)                | 获取字符串长度                         |
| x \|\|  y | 字符串连接运算符                       |
| CONCAT(x,y)               | 字符串连接函数                         |
| TRIM( [trim_str FROM] x)           | 从 x 两边去除  trim_str(默认空格)                |
| LTRIM(x [,trim_str])              | 从 x 左边去除第一个 trim_str(默认空格)               |
| RTRIM(x [,trim_str])              | 从 x 右边去除第一个 trim_str(默认空格)             |
| REPLACE(x ,old[,new])       | 将 old 替换成 new（默认空格）                |
| INSTR(x, str \[,start\] \[,n\])               | 在x中搜索str位置                       |
| LPAD(x, length[, fill])      | 在x左端填充字符串 fill(默认空格) ，长度为length |
| RPAD(x, length[, fill])      | 在x右边填充字符串 fill(默认空格) ，长度为length                     |

```sql
SQL> select lpad('bill',8,'*') from dual;
****bill
```

## 格式化函数

| 函数           | 说明                         |
| :-------------------- | :---------------------------- |
| TO_CHAR(date, fmt)   | 日期转字符，默认 "DD-MON-RR" |
| TO_CHAR(num, fmt)    | 数字转化为指定格式字符串     |
| TO_DATE(char, fmt)   | 字符转日期                   |
| DATE 'YYYY-MM-DD'    | 字符转日期                   |
| TO_NUMBER(char, fmt) | 字符转数字                   |

> 隐性转换：Oracle 会自动将对应格式的字符类型的数据转化为数字或者日期格式

```sql
SQL> select TO_CHAR(3.1415, '9,999.999') FROM dual;
SQL> select TO_CHAR(-31415.926, 'L9.9EEEPR') FROM dual;
<￥3.1E+04>
```

常用的数字格式：

|参数|说明|
|:---|:---|
|9|显示数字|
|0|显示零|
|.|小数点|
|,|千位符|
|$|美元符号|
|L|本地货币符号|
|EEEE|科学计数法|
|PR|将负数用尖括号包括表示|

```sql
SQL> select TO_CHAR(SYSDATE, 'YYYY-MM-DD HH24:MI:SS') FROM dual;
SQL> select TO_CHAR(LOCALTIMESTAMP(2),'YYYY-MM-DD HH24:MI:SS.FF') FROM dual;
```

常用的日期格式：

| 参数 | 说明 |
| :--- | :--- |
| YYYY | 年   |
| MM   | 月   |
| DD   | 日   |

## 通用函数


| 函数                           | 描述                                 |
| :------------------------------------ | :------------------------------------ |
| DECODE(x,value1,result1,...,default) | 和 CASE...WHEN类似的函数             |
| NVL(x, value)                    | 返回的是它第一个非空值的参数         |
| NVL2(x, value1, value2)                    | 返回的是它第一个非空值的参数         |
| COALESCE(value1, value2, ...)        | 返回第一个不是空值的参数             |
| NULLIF(valuel, value2)               | 如果value1和value2相等，函数返回空值 |
| GREATEST(value1, value2, ...)        | 最大值                               |
| LEAST(value1, value2, ...)           | 最小值                               |
| PIVOD(aggfun FOR  col IN (value1, value2, ...) ) | 透视 |
| UNPIVOD(valueName FOR keyName IN(col1,col2,...)) | 逆透视 |

```sql
-- 透视（交叉表）
select * from (
 select times_purchased as "Puchase Frequency", state_code
 from customers t
 )
 pivot 
 (
 count(state_code)
 for state_code in ('NY' as "New York",'CT' "Connecticut",'NJ' "New Jersey",'FL' "Florida",'MO' as "Missouri")
 )
 order by 1;
 
 -- 逆透视
 select *
 from cust_matrix
 unpivot
 (
 state_counts
 for state_code in ("New York","Conn","New Jersey","Florida","Missouri")
 )
 order by "Puchase Frequency", state_code;
```

------

> **参考文献：**
> [易百教程 - Oracle](https://www.yiibai.com/oracle)
> [PL/SQL教程](https://www.yiibai.com/plsql2)
> [W3school 新学院 - Oracle](http://www.hechaku.com/Oracle/Sql_Oracle_info.htm3)