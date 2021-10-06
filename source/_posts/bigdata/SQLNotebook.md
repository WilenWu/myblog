---
ID: e9159e236fc5a451035f6cf278723e0d
title: SQL手册
tags: [数据库,SQL]
mathjax: false
copyright: true
date: 2018-07-04 15:33:56
categories: [Big Data]
sticky: false
---
结构化查询语言(Structured Query Language)简称SQL，是一种数据库查询和程序设计语言，用于存取数据以及查询、更新和管理关系数据库系统。

<!-- more -->

数据库|Oracle|MySQL
---|---|---
启动和关闭|sqlplus|NET START mysql57<br>NET STOP mysql57
登录（user+password）|sys/p as sysdba<br>[conn] system/p|MYSQL –uroot –p<br>EXIT



# 数据库操作

```sql
SHOW DATABASES;            --查看数据库
CREATE DATABASE db_name;   --创建数据库
DROP DATABASE db_name;     --删除
USE db_name;               --连接数据库
SELETE DATABASE();
```

# 数据表操作

```sql
--------查看数据表
SHOW TABLES;
--------创建数据表
CREATE TABLE tbl_name(
col_name col_type,
col_name col_type,
);
--------删除数据表
DROP TABLE tbl_name,table_name;
--------数据表结构
DESC tbl_name;
---------修改数据表结构
ALTER TABLE tbl_name
RENAME TO new_name,                    --重命名表名
ADD col_name col_type,                 --添加字段
DROP col_name,                         --删除字段
CHANGE old_name new_name new_type;     --重命名字段名

-----------------mysql导入数据库
mysql> LOAD DATA LOCAL INFILE 'dump.txt' --utf-8 txt文件
    -> INTO TABLE mytbl                  --已创建的表
    -> FIELDS TERMINATED BY ':'          --分隔符
    -> LINES TERMINATED BY '\r\n';       --换行符
-----------------插入记录
INSERT INTO  tbl_name  VALUES (....);
-----------------更新记录
UPDATE tbl_name
SET col_name = new_value,
SET col_name = new_value
WHERE ……;
-----------------删除记录
DELETE FROM tbl_name WHERE ……;
```
# 查询语句

```sql
select -> from -> where -> group by -> having -> order by -> limit
```

# 常用函数

Oracle(PL/SQL)|MySQL|说明
---|---|---
concat()|concat()|字符连接
nvl()|ifnull()|空值替换
sysdate()|now()|系统时间
to_date()|date_formate|日期格式化
add_month()|date_sub()|日期+时间间隔
rownum|limit|查询部分记录



