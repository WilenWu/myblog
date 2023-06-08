---
title: 大数据手册(Hive)--HiveQL(DML)
categories:
  - Big Data
  - Hive
tags:
  - 大数据
  - hive
  - SQL
cover: /img/apache-hive-dml.png
top_img: /img/apache-hive-bg.png
description: 数据操作语言(Data Manipulation Language, DML）：其语句包括动词 INSERT、UPDATE、DELETE。它们分别用于添加、修改和删除。
abbrlink: d2c808ff
date: 2018-07-03 17:57:36
---

# 数据操作语句

```sql
-- 例表
CREATE EXTERNAL TABLE page_view_stg(
    viewTime INT, 
    userid BIGINT,
    page_url STRING, 
    referrer_url STRING,
    ip STRING COMMENT 'IP Address of the User',
    country STRING COMMENT 'country of origination'
) COMMENT 'This is the staging page view table'
ROW FORMAT DELIMITED FIELDS TERMINATED BY '44' LINES TERMINATED BY '12'
STORED AS TEXTFILE
LOCATION '/user/data/staging/page_view';
```

## 将文件加载到表中

Hive 在将数据加载到表中时不做任何转换。加载操作目前是纯复制/移动操作，将数据文件移动到对应于 Hive 表的位置。

```sql
LOAD DATA [LOCAL] INPATH 'filepath' 
[OVERWRITE] INTO TABLE tablename 
[PARTITION (partition_spec)] 
[INPUTFORMAT 'inputformat' SERDE 'serde']    -- (hive 3.0 or late)
```

- filepath 可以是相对路径或绝对路径
- 关键字 LOCAL 标识符指定本地路径，否则为hdfs路径
- 关键字 OVERWRITE 指定覆盖目标表（或分区）中的数据，否则将添加到表中。

```sql
LOAD DATA LOCAL INPATH /tmp/pv_2008-06-08_us.txt INTO TABLE page_view PARTITION(date='2008-06-08', country='US')

LOAD DATA INPATH '/user/data/pv_2008-06-08_us.txt' INTO TABLE page_view PARTITION(date='2008-06-08', country='US')
```

也可以先将文件移动到 Hive 表的位置，然后手动修复，将有关分区的元数据更新到 Hive Metastore

```sql
hadoop dfs -put /tmp/pv_2008-06-08.txt /user/data/staging/page_view
hive> MSCK TABLE table_name;  -- 修复分区信息 
```

## 将查询结果插入表中

```sql
INSERT OVERWRITE TABLE tablename 
	[PARTITION (partition_spec) [IF NOT EXISTS]] 
	select_statement1 FROM from_statement;
INSERT INTO TABLE tablename 
	[PARTITION (partition_spec)] 
	select_statement1 FROM from_statement;
```

可以在同一个查询中指定多个插入子句（也称为*多表插入*）。多表插入最大限度地减少了所需的数据扫描次数。通过对输入数据仅扫描一次（并应用不同的查询运算符），Hive 可以将数据插入到多个表中。

```sql
FROM from_statement
INSERT OVERWRITE TABLE tablename1 
	[PARTITION (partition_spec) [IF NOT EXISTS]] 
	select_statement1
	... ...
[INSERT INTO TABLE tablename2 
	[PARTITION (partition_spec)] 
 	select_statement2] 
 	... ...;
```

```sql
FROM page_view_stg pvs
INSERT OVERWRITE TABLE page_view PARTITION(dt='2008-06-08', country='US')
SELECT pvs.viewTime, pvs.userid, pvs.page_url, pvs.referrer_url, null, null, pvs.ip
WHERE pvs.country = 'US';
```

## 动态分区插入

动态分区插入（或多分区插入）旨在通过在扫描输入表时动态确定应创建和填充哪些分区。如果尚未创建该分区，它将自动创建该分区。
动态分区列必须在 SELECT 语句中的列中**最后指定**，并且必须与在 PARTITION 子句中出现的**顺序相同**。
动态分区字段一定要放在所有静态字段的后面。

```sql
INSERT OVERWRITE TABLE tablename 
	PARTITION (partcol1[=val1], partcol2[=val2] ...) 
	select_statement FROM from_statement;
INSERT INTO TABLE tablename 
	PARTITION (partcol1[=val1], partcol2[=val2] ...) 
	select_statement FROM from_statement;
```

**例子**：这里country 是一个动态分区列

```sql
-- 动静分区结合
FROM page_view_stg pvs
INSERT OVERWRITE TABLE page_view PARTITION(dt='2008-06-08', country)
SELECT pvs.viewTime, pvs.userid, pvs.page_url, pvs.referrer_url, null, null, pvs.ip, pvs.country;

-- 动态（自动）分区
hive.exec.dynamic.partition.mode = nonstrict;
FROM page_view_stg pvs
INSERT OVERWRITE TABLE page_view PARTITION(dt, country)
SELECT pvs.viewTime, pvs.userid, pvs.page_url, pvs.referrer_url, null, null, pvs.ip,  pvs.dt, pvs.country;
```

这些是动态分区插入的相关配置属性：

| 配置属性                                   | 默认     | 笔记                                                         |
| :----------------------------------------- | :------- | :----------------------------------------------------------- |
| `hive.exec.dynamic.partition`              | `true`   | 需要设置`true`为启用动态分区插入                             |
| `hive.error.on.empty.partition`            | `false`  | 动态分区插入产生空结果是否抛出异常                           |
| `hive.exec.dynamic.partition.mode`         | `strict` | 在`strict`模式下，用户必须指定至少一个静态分区。在`nonstrict`模式的所有分区被允许是动态 |
| `hive.exec.max.created.files`              | 100000   | MapReduce 作业中所有映射器/还原器创建的 HDFS 文件的最大数量  |
| `hive.exec.max.dynamic.partitions`         | 1000     | 总共允许创建的最大动态分区数                                 |
| `hive.exec.max.dynamic.partitions.pernode` | 100      | 每个mapper/reducer节点允许创建的最大动态分区数               |
| `hive.exec.default.partition.name`         |          | 如果分区列值为 NULL 或空字符串，则该行将被放入一个特殊分区，默认值为`HIVE_DEFAULT_PARTITION`。 |

## 将数据写入文件系统

```sql
-- Standard syntax:
INSERT OVERWRITE [LOCAL] DIRECTORY 'directory'
[ROW FORMAT row_format] 
[STORED AS file_format]
  SELECT select_statement FROM from_statement ;

-- Hive extension (multiple inserts):
FROM from_statement
INSERT OVERWRITE [LOCAL] DIRECTORY 'directory1' select_statement1
[INSERT OVERWRITE [LOCAL] DIRECTORY 'directory2' select_statement2] ...;
```

- 关键字 LOCAL 标识符指定本地路径，否则为hdfs路径
- 写入文件系统的数据被序列化为文本。如果任何列不是原始数据类型，则这些列将序列化为 JSON 格式。
- row_format 语法如下，用法见 CREATE TABLE

```sql
-- row_format:
DELIMITED 
[FIELDS TERMINATED BY 'char' [ESCAPED BY char]] 
[COLLECTION ITEMS TERMINATED BY char]
[MAP KEYS TERMINATED BY char] 
[LINES TERMINATED BY char]
[NULL DEFINED AS char]
```

```sql
-- 输出写入本地文件
INSERT OVERWRITE LOCAL DIRECTORY '/tmp/pv_gender_sum'
SELECT pv_gender_sum.* FROM pv_gender_sum;
```

也可以使用 shell 命令写入本地

```bash
hive -e "SELECT ... FROM ...;" > local_path
```

## 向表中插入值

```sql
INSERT INTO TABLE tablename 
[PARTITION (partcol1[=val1], partcol2[=val2] ...)] 
VALUES (value1, value2, ...)
	  [,(value1, value2，...) ... ...]
```

- VALUES 子句中列出的每一行都插入到表*tablename 中*。
- 必须为表中的每一列提供值。尚不支持允许用户仅将值插入某些列的标准 SQL 语法。为了模仿标准 SQL，可以为用户不希望为其分配值的列提供空值。
- 动态分区的支持方式与 INSERT...SELECT 相同。
- Hive 不支持INSERT INTO...VALUES 子句中使用复杂类型（数组、映射、结构、联合）。

```sql
CREATE TABLE pageviews (userid VARCHAR(64), link STRING, came_from STRING)
PARTITIONED BY (datestamp STRING) 
CLUSTERED BY (userid) INTO 256 BUCKETS STORED AS ORC;
 
INSERT INTO TABLE pageviews PARTITION (datestamp = '2014-09-23')
  VALUES ('jsmith', 'mail.com', 'sports.com'),
  ('jdoe', 'mail.com', null);
 
INSERT INTO TABLE pageviews PARTITION (datestamp)
  VALUES ('tjohnson', 'sports.com', 'finance.com', '2014-09-23'), 
  ('tlee', 'finance.com', null, '2014-09-21');
  
INSERT INTO TABLE pageviews
  VALUES ('tjohnson', 'sports.com', 'finance.com', '2014-09-23'),
  ('tlee', 'finance.com', null, '2014-09-21');
```

## 更新数据

```sql
UPDATE tablename SET column = value [, column = value ...] [WHERE expression]
```

- 分配的值必须是 Hive 在 select 子句中支持的表达式。因此支持算术运算符、UDF、强制转换、文字等。不支持子查询。
- 只有匹配 WHERE 子句的行才会被更新。
- 无法更新分区列，无法更新分桶列。
- 成功完成此操作后，将自动提交更改。

## 删除数据

```sql
DELETE FROM tablename [WHERE expression]
```

- 只有匹配 WHERE 子句的行才会被删除。
- 成功完成此操作后，将自动提交更改。

## 合并操作

MERGE 从Hive 2.2开始可用，允许根据与源表的连接结果对目标表执行操作。

```sql
MERGE INTO <target table> AS T USING <source expression/table> AS S
ON <boolean expression1>
WHEN MATCHED [AND <boolean expression2>] THEN UPDATE SET <set clause list>
WHEN MATCHED [AND <boolean expression3>] THEN DELETE
WHEN NOT MATCHED [AND <boolean expression4>] THEN INSERT VALUES<value list>
```

- 可能存在 1、2 或 3 个 WHEN 子句；每种类型最多 1 个：UPDATE/DELETE/INSERT。
- WHEN NOT MATCHED 必须是最后一个 WHEN 子句。
- 如果 UPDATE 和 DELETE 子句都存在，则语句中的第一个子句必须包含 `[AND <boolean expression>]`。

```sql
CREATE DATABASE merge_data;

CREATE TABLE merge_data.transactions(
 ID int,
 TranValue string,
 last_update_user string)
PARTITIONED BY (tran_date string)
CLUSTERED BY (ID) into 5 buckets 
STORED AS ORC TBLPROPERTIES ('transactional'='true');

CREATE TABLE merge_data.merge_source(
 ID int,
 TranValue string,
 tran_date string)
STORED AS ORC;
```

```sql
INSERT INTO merge_data.transactions PARTITION (tran_date) VALUES
(1, 'value_01', 'creation', '20170410'),
(2, 'value_02', 'creation', '20170410'),
(3, 'value_03', 'creation', '20170410'),
(4, 'value_04', 'creation', '20170410'),
(5, 'value_05', 'creation', '20170413'),
(6, 'value_06', 'creation', '20170413'),
(7, 'value_07', 'creation', '20170413'),
(8, 'value_08', 'creation', '20170413'),
(9, 'value_09', 'creation', '20170413'),
(10, 'value_10','creation', '20170413');

INSERT INTO merge_data.merge_source VALUES 
(1, 'value_01', '20170410'),
(4, NULL, '20170410'),
(7, 'value_77777', '20170413'),
(8, NULL, '20170413'),
(8, 'value_08', '20170415'),
(11, 'value_11', '20170415');
```

```sql
MERGE INTO merge_data.transactions AS T 
USING merge_data.merge_source AS S
ON T.ID = S.ID and T.tran_date = S.tran_date
WHEN MATCHED AND (T.TranValue != S.TranValue AND S.TranValue IS NOT NULL) 
	THEN UPDATE SET TranValue = S.TranValue, last_update_user = 'merge_update'
WHEN MATCHED AND S.TranValue IS NULL THEN DELETE
WHEN NOT MATCHED THEN INSERT VALUES (S.ID, S.TranValue, 'merge_insert', S.tran_date);
```

```sql
SELECT * FROM merge_data.transactions order by ID;

+----+-----------------------+------------------------------+-----------------------+
| id | transactions.tranvalue| transactions.last_update_user| transactions.tran_date|
+----+-----------------------+------------------------------+-----------------------+
| 1  | value_01              | creation                     | 20170410              |
| 2  | value_02              | creation                     | 20170410              |
| 3  | value_03              | creation                     | 20170410              |
| 5  | value_05              | creation                     | 20170413              |
| 6  | value_06              | creation                     | 20170413              |
| 7  | value_77777           | merge_update                 | 20170413              |
| 8  | value_08              | merge_insert                 | 20170415              |
| 9  | value_09              | creation                     | 20170413              |
| 10 | value_10              | creation                     | 20170413              |
| 11 | value_11              | merge_insert                 | 20170415              |
+----+-----------------------+------------------------------+-----------------------+
```

## 导入导出

`EXPORT`命令将表或分区的数据以及元数据导出到指定位置。`IMPORT`命令从指定位置导入。
`EXPORT`和`IMPORT`命令中使用的源和目标metastore DBMS的独立工作。例如，它们可以在 Derby 和 MySQL 数据库之间使用。

```sql
-- 导出语法
EXPORT TABLE tablename [PARTITION (partition_spec)]
  TO 'export_target_path' 
  [ FOR replication('eventid') ]  -- 复制
-- 导入语法
IMPORT [[EXTERNAL] TABLE new_or_original_tablename [PARTITION (partition_spec)]]
  FROM 'source_path'
  [LOCATION 'import_target_path']
```

如果目标不存在，`IMPORT`将创建目标表/分区。所有表属性/参数都将是用于`EXPORT`生成存档的表的属性/参数。如果目标存在，则检查它是否具有适当的架构、输入/输出格式等。如果目标表存在且未分区，则它必须为空。如果目标表存在且已分区，则表中不得存在要导入的分区。

**例子**

简单的导出和导入：

```sql
export table department to 'hdfs_exports_location/department';
import from 'hdfs_exports_location/department';
```

导入时重命名表：

```sql
export table department to 'hdfs_exports_location/department';
import table imported_dept from 'hdfs_exports_location/department';
```

导出分区和导入：

```sql
export table employee partition (emp_country="in", emp_state="ka") 
to 'hdfs_exports_location/employee';
import from 'hdfs_exports_location/employee';
```

导出表和导入分区：

```sql
export table employee to 'hdfs_exports_location/employee';
import table employee partition (emp_country="us", emp_state="tn") 
from 'hdfs_exports_location/employee';
```

指定导入位置：

```sql
export table department to 'hdfs_exports_location/department';
import table department from 'hdfs_exports_location/department' 
       location 'import_target_location/department';
```

导入为外部表：

```sql
export table department to 'hdfs_exports_location/department';
import external table department from 'hdfs_exports_location/department';
```
