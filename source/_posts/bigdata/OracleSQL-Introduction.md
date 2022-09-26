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



------

> **参考文献：**
> [易百教程 - Oracle](https://www.yiibai.com/oracle)
> [PL/SQL教程](https://www.yiibai.com/plsql2)
> [W3school 新学院 - Oracle](http://www.hechaku.com/Oracle/Sql_Oracle_info.htm3)