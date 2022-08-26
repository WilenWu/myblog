---
title: 大数据手册(Hive)--HiveQL(DDL)
categories:
  - Big Data
  - Hive
tags:
  - 大数据
  - hive
  - SQL
cover: /img/apache-hive-ddl.png
top_img: /img/apache-hive-bg.png
description: '数据定义语言(Data Definition Language, DDL)：其语句包括动词CREATE、ALTER和DROP。'
abbrlink: dd193370
date: 2018-07-03 17:57:36
---

# 数据库

```sql
USE database_name;
USE DEFAULT;
SELECT current_database();  -- 查看使用的数据库
SHOW (DATABASE|SCHEMA) [LIKE "pattern_with_wildcards"]；  -- 查看数据库
DESCRIBE (DATABASE|SCHEMA) [EXTENDED] db_name;; -- 查看数据库位置等信息
```

## 创建数据库

```sql
CREATE [REMOTE] (DATABASE|SCHEMA) [IF NOT EXISTS] database_name
  [COMMENT database_comment]
  [LOCATION hdfs_path]
  [MANAGEDLOCATION hdfs_path]
  [WITH DBPROPERTIES (property_name=property_value, ...)];
```

- HIVE 中 SCHEMA 和 DATABASE 是等同的
- LOCATION 现在指的是外部表的默认目录
- MANAGEDLOCATION 已添加到 Hive 4.0.0 中，指的是内部表的默认目录 (metastore.warehouse.dir)。

## 删除数据库

```sql
DROP (DATABASE|SCHEMA) [IF EXISTS] database_name [RESTRICT|CASCADE];
```

- 默认 RESTRICT，如果数据库不为空，则删除将失败。
- 要删除数据库时，删除库中的表则使用 CASCADE

## 修改数据库

```sql
ALTER (DATABASE|SCHEMA) database_name SET DBPROPERTIES (property_name=property_value, ...);
ALTER (DATABASE|SCHEMA) database_name SET OWNER [USER|ROLE] user_or_role;
ALTER (DATABASE|SCHEMA) database_name SET LOCATION hdfs_path; -- Hive 2.2
ALTER (DATABASE|SCHEMA) database_name SET MANAGEDLOCATION hdfs_path; 
```

# 连接器

hive 4.0.0 中添加了对数据连接器的支持。初始版本包括基于 JDBC 的数据源（如 MYSQL、POSTGRES、DERBY）的连接器实现。后续将添加其他连接器。 

```sql
SHOW CONNECTORS;  -- 列出 Metastore 中定义的所有连接器
DESCRIBE CONNECTOR [EXTENDED] connector_name;  -- 显示连接器的名称、注释等
```

## 创建连接器

```sql
CREATE CONNECTOR [IF NOT EXISTS] connector_name
   [TYPE datasource_type]
   [URL datasource_url]
   [COMMENT connector_comment]
   [WITH DCPROPERTIES (property_name=property_value, ...)];
```

- TYPE - 此连接器连接到的远程数据源的类型。例如MYSQL。类型决定了 Driver 类和特定于此数据源的任何其他参数。
- URL - 远程数据源的 URL。如果是 JDBC 数据源，它将是 JDBC 连接 URL。对于 hive 类型，它将是 thrift URL。
- COMMENT - 此连接器的简短描述。
- DCPROPERTIES：连接器属性设置。

## 删除连接器

```sql
DROP CONNECTOR [IF EXISTS] connector_name;
```

## 修改连接器

```sql
ALTER CONNECTOR connector_name SET DCPROPERTIES (property_name=property_value, ...);
ALTER CONNECTOR connector_name SET URL new_url;
ALTER CONNECTOR connector_name SET OWNER [USER|ROLE] user_or_role;
```

# 数据表

```sql
SHOW TABLES [IN database_name] [LIKE "pattern_with_wildcards"]; -- 显示数据库中的表和视图
SHOW TABLE EXTENDED [IN|FROM database_name] LIKE 'pattern_with_wildcards' 
	[PARTITION(partition_spec)];  -- 列出基本表信息和文件系统信息
SHOW TBLPROPERTIES tblname[(property_name)];  -- 列出表属性信息
SHOW CREATE TABLE [db_name.]table_name;  -- 查看建表语句
SHOW COLUMNS (FROM|IN) table_name [(FROM|IN) db_name];  -- 显示表中的所有列
DESCRIBE [EXTENDED|FORMATTED] [db_name.]table_name;  -- 将以序列化形式|表格格式显示元数据
```

## 数据存储

- 基于HDFS
- 分区表：对于数据库中的超大型表，可以通过把它的数据分成若干个小表，从而简化数据库的管理活动
- 内部表：存储在 hive.metastore.warehouse.dir 路径属性下，默认情况下类似于`/user/hive/warehouse/databasename.db/tablename/`.。默认位置可以在表创建期间被`location`属性覆盖。如果删除内部表或分区，则删除与该表或分区关联的数据和元数据。如果未指定 PURGE 选项，则数据将在定义的时间内移动到垃圾文件夹。
- 外部表：外部表描述了外部文件的元数据。在删除这个表的时候只删除了表定义。当文件已经存在或位于远程位置时使用外部表。
- 桶表 bucket：经过hash运算后放在不同的桶中，比非分桶表允许更有效的采样。

## 创建表

```sql
CREATE [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS] [dbname.]table_name 
(
    col_name data_type COMMENT col_comment, 
    ... ...
)
COMMENT table_comment
PARTITIONED BY (
    col_name data_type COMMENT col_comment, 
    ... ...
)
[CLUSTERED BY (col_name, ...) 
	[SORTED BY (col_name [ASC|DESC], ...)] 
	INTO num BUCKETS
]
[ROW FORMAT DELIMITED
	[FIELDS TERMINATED BY 'char' [ESCAPED BY 'char']] 
	[COLLECTION ITEMS TERMINATED BY 'char']
    [MAP KEYS TERMINATED BY 'char'] 
	[LINES TERMINATED BY 'char']
    [NULL DEFINED AS 'char']
]
[STORED AS file_format]
[LOCATION hdfs_path]
[TBLPROPERTIES (name=value, ...)]
```

- 默认创建内部表，其中文件、元数据和统计信息由内部 Hive 进程管理。EXTERNAL 关键字创建外部表，同时提供 LOCATION
- TEMPORARY 创建临时表，只在当前交互时使用，会话结束时被删除
- PARTITIONED BY 子句建立表分区
- TBLPROPERTIES 子句允许预定义表属性，例如
  - "comment"="*table_comment*" ：设置表注释
  - "EXTERNAL"="TRUE" ：定义为外部表
  - "skip.header.line.count"="n"  ：忽略文件前n行
  - "skip.footer.line.count"="n"  ：忽略文件后n行
- STORED AS file_format：指定文件格式，默认 TEXTFILE 纯文本格式，还支持 SEQUENCEFILE|RCFILE|ORC|PARQUET|AVRO|JSONFILE|INPUTFORMAT input_format_classname OUTPUTFORMAT output_format_classname
- ROW FORMAT row_format：使用 DELIMITED 子句读取分隔文件
  - FIELDS TERMINATED BY：定义块分隔符
  - ESCAPED BY：如果要处理包含分隔符的数据，为分隔符启用转义
  - LINES TERMINATED BY：定义行分隔符
  - NULL DEFINED AS：自定义 NULL 格式，默认为 '\N'
- CLUSTERED BY：对表或分区进行分桶，并且可以通过 SORT BY 在该桶内对数据进行排序。

 **示例**：

```sql
CREATE TABLE page_view(
    viewTime INT, 
    userid BIGINT,
    page_url STRING, 
    referrer_url STRING,
    friends ARRAY<BIGINT>, 
    properties MAP<STRING, STRING>
    ip STRING COMMENT 'IP Address of the User'
) COMMENT 'This is the page view table'
PARTITIONED BY(dt STRING, country STRING)
CLUSTERED BY(userid) SORTED BY(viewTime) INTO 32 BUCKETS
ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '<hdfs_location>';
```

该表对 userid 进行HASH运算分到 32 个桶中。在每个存储桶中，数据按 viewTime 的递增顺序排序。该表将会以更高的效率抽样和查询。如果没有分桶，仍然可以对表进行随机抽样，但效率不高，因为查询必须扫描所有数据。

 **CTAS 语句**

还可以通过一个 create-table-as-select (CTAS) 语句中的查询结果来创建和填充表。

```sql
CREATE [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS] table_name
[AS select_statement];  
```

**复制表结构**

允许您精确复制现有表定义（不复制其数据）

```sql
-- 创建空表，复制已存在的表结构
CREATE [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS] table_name
LIKE existing_table_or_view_name;
```

## 创建约束

```sql
CREATE TABLE table_name 
(
    column_1 data_type [column_constraint ENABLE|DISABLE NOVALIDATE RELY|NORELY],
    ... 
    [table_constraint]
);
```

Hive 支持的约束类型：PRIMARY KEY|UNIQUE|NOT NULL|DEFAULT [default_value]|CHECK [check_expression]
default_value支持 LITERAL|CURRENT_USER()|CURRENT_DATE()|CURRENT_TIMESTAMP()|NULL

```sql
-- table_constraint:
[, PRIMARY KEY (col_name, ...) DISABLE NOVALIDATE RELY/NORELY ]
[, PRIMARY KEY (col_name, ...) DISABLE NOVALIDATE RELY/NORELY ]
[, CONSTRAINT constraint_name FOREIGN KEY (col_name, ...) REFERENCES table_name(col_name, ...) DISABLE NOVALIDATE 
[, CONSTRAINT constraint_name UNIQUE (col_name, ...) DISABLE NOVALIDATE RELY/NORELY ]
[, CONSTRAINT constraint_name CHECK [check_expression] ENABLE|DISABLE NOVALIDATE RELY/NORELY ]
```

**示例**

```sql
create table pk(
    id1 integer, 
    id2 integer,
    primary key(id1, id2) disable novalidate
);
 
create table fk(
    id1 integer, 
    id2 integer,
    constraint c1 foreign key(id1, id2) references pk(id2, id1) disable novalidate
);
  
create table constraints1(
    id1 integer UNIQUE disable novalidate, 
    id2 integer NOT NULL,
    usr string DEFAULT current_user(), 
    price double CHECK (price > 0 AND price <= 1000)
);
```

## 删除表

```sql
DROP TABLE [IF EXISTS] table_name [PURGE]; -- 删除表
TRUNCATE TABLE table_name; -- 截断表
```

- DROP TABLE 删除此表的元数据和数据。如果配置了 Trash（并且未指定 PURGE），则数据实际上会移动到 .Trash/Current 目录。元数据完全丢失。
- 删除 EXTERNAL 表时，表中的数据不会从文件系统中删除。如果设置表属性 external.table.purge=true，也会删除数据。
- 删除视图引用的表时，不会给出警告（视图失效，必须由用户删除或重新创建）。
- TRUNCATE 从表删除所有行。

## 修改表/分区

统一用 partition_spec 代替分区语句

```sql
(partition_column = partition_col_value, partition_column = partition_col_value, ...)
```

**修改表**

```sql
ALTER TABLE table_name RENAME TO new_table_name;  -- 重名名表名
ALTER TABLE table_name SET TBLPROPERTIES(  -- 修改表属性
    key = value,
    ... ...
);
ALTER TABLE table_name SET TBLPROPERTIES ('comment' = new_comment); -- 修改表注释

ALTER TABLE table_name 
	CLUSTERED BY (col_name, ...) 
	[SORTED BY (col_name, ...)]
    INTO num_buckets BUCKETS;  -- 更改表的物理存储属性
```

**修改表/分区**

```sql
ALTER TABLE table_name [PARTITION partition_spec] 
	SET FILEFORMAT file_format;  -- 修改表/分区文件格式
ALTER TABLE table_name [PARTITION partition_spec] 
	SET LOCATION "new location";  -- 修改表/分区位置
ALTER TABLE table_name TOUCH [PARTITION partition_spec]; -- 修改表/分区 TOUCH
ALTER TABLE table_name [PARTITION partition_spec]  
	COMPACT 'compaction_type'[AND WAIT]  -- 更改表/分区压缩
    [WITH OVERWRITE TBLPROPERTIES ("property"="value" , ...)];
```

```sql
-- 更改表/分区保护
ALTER TABLE table_name [PARTITION partition_spec] ENABLE|DISABLE NO_DROP [CASCADE];  
ALTER TABLE table_name [PARTITION partition_spec] ENABLE|DISABLE OFFLINE;
```

- 可以在表或分区级别设置对数据的保护。启用 NO_DROP 可防止表被删除。启用OFFLINE 可以防止查询表或分区中的数据，但仍然可以访问元数据。
- 如果表中的任何分区启用了 NO_DROP，则该表也不能被删除。相反，如果一个表启用了 NO_DROP，那么分区可能会被删除，但是启用 NO_DROP CASCADE 分区也不能被删除，除非drop partition 命令指定 IGNORE PROTECTION 。

```sql
-- 更改表/分区合并
ALTER TABLE table_name [PARTITION partition_spec] CONCATENATE;
```

如果表或分区包含很多小的 RCFiles 或 ORC 文件，那么上面的命令会将它们合并成更大的文件。在 RCFile 的情况下，合并发生在块级别，而对于 ORC 文件，合并发生在条带级别，从而避免解压缩和解码数据的开销。

```sql
-- 更改表/分区更新列
ALTER TABLE table_name [PARTITION partition_spec] UPDATE COLUMNS;
```

在 Hive 3.0.0版中，添加了此命令以让用户将 serde 存储的架构信息同步到 Metastore。

**修改约束**

```sql
ALTER TABLE table_name ADD CONSTRAINT constraint_name PRIMARY KEY (column, ...) DISABLE NOVALIDATE;
ALTER TABLE table_name ADD CONSTRAINT constraint_name FOREIGN KEY (column, ...) REFERENCES table_name(column, ...) DISABLE NOVALIDATE RELY;
ALTER TABLE table_name ADD CONSTRAINT constraint_name UNIQUE (column, ...) DISABLE NOVALIDATE;
ALTER TABLE table_name CHANGE COLUMN column_name column_name data_type CONSTRAINT constraint_name NOT NULL ENABLE;
ALTER TABLE table_name CHANGE COLUMN column_name column_name data_type CONSTRAINT constraint_name DEFAULT default_value ENABLE;
ALTER TABLE table_name CHANGE COLUMN column_name column_name data_type CONSTRAINT constraint_name CHECK check_expression ENABLE;
-- 删除约束
ALTER TABLE table_name DROP CONSTRAINT constraint_name;
```

## 修改列

```sql
ALTER TABLE table_name [PARTITION partition_spec] 
CHANGE [COLUMN] col_old_name col_new_name column_type 
	[COMMENT col_comment] 
    [FIRST|AFTER column_name] 
    [CASCADE|RESTRICT];
```

- 此命令将允许用户更改列的名称、数据类型、注释或位置，或它们的任意组合
- CASCADE 命令更改表元数据的列，并将相同的更改级联到所有分区元数据。RESTRICT 是默认设置，将列更改限制为仅对表元数据进行更改
- 无论表或分区的保护模式如何，启用CASCADE 都将覆盖表分区的列元数据。谨慎使用。
- FIRST|AFTER column_name 子句更改列位置

```sql
ALTER TABLE table_name [PARTITION partition_spec]
	ADD|REPLACE COLUMNS (col_name data_type [COMMENT col_comment], ...)
    [CASCADE|RESTRICT];
```

- ADD COLUMNS 允许您将新列添加到现有列的末尾但在分区列之前
- REPLACE COLUMNS 删除所有现有列并添加新的列集
- REPLACE COLUMNS 也可用于删除列，添加新的列集时去掉需要删除的列即可

## 统计信息

用户可以通过仅查询存储的统计信息而不是触发长时间运行的执行计划来获得。统计信息现在存储在 Hive Metastore 中，用于新创建的或现有的表。当前支持以下统计信息：行数、文件数、存储大小、列分析等

对于新创建的表或分区，默认情况下会自动计算统计信息。参数如下

```sql
set hive.stats.autogather=true;
```

对于现有的表或分区，用户可以发出 ANALYZE 命令来收集统计信息并将它们写入 Hive MetaStore。该命令的语法如下

```sql
ANALYZE TABLE [db_name.]tablename [PARTITION(partition_spec)]
  COMPUTE STATISTICS 
  [FOR COLUMNS]      
  [CACHE METADATA]       -- (Note: Hive 2.1.0 and later.)
  [NOSCAN];
```

当指定可选参数 NOSCAN 时，该命令不会扫描文件，只收集文件数和物理大小，而不是所有统计信息。

显示这些统计信息的语法如下

```sql
DESCRIBE FORMATTED [db_name.]table_name column_name [PARTITION (partition_spec)];  -- 查看收集的列统计信息
```

# 分区管理

统一用 partition_spec 代替分区语句

```sql
(partition_column = partition_col_value, partition_column = partition_col_value, ...)
```

```sql
SHOW PARTITIONS table_name;   -- 查看分区
SHOW PARTITIONS [db_name.]table_name [PARTITION(partition_spec)];
SHOW PARTITIONS [db_name.]table_name [PARTITION(partition_spec)] 
	[WHERE where_condition] 
	[ORDER BY col_list] 
	[LIMIT rows];   -- (Note: Hive 4.0.0 and later)
DESCRIBE [EXTENDED|FORMATTED] table_name[.column_name] PARTITION partition_spec;
```

## 修改分区

```sql
ALTER TABLE table_name ADD [IF NOT EXISTS]  -- 添加分区
	PARTITION partition_spec [LOCATION 'location']
	[, PARTITION partition_spec [LOCATION 'location'], ...];
ALTER TABLE table_name PARTITION partition_spec 
	RENAME TO PARTITION partition_spec; -- 重名民分区

-- Move partition from table_name_1 to table_name_2
ALTER TABLE table_name_2 EXCHANGE PARTITION (partition_spec) WITH TABLE table_name_1;
-- multiple partitions
ALTER TABLE table_name_2 EXCHANGE PARTITION (partition_spec, partition_spec2, ...) WITH TABLE table_name_1;
```

## 发现分区

表属性 "discover.partitions"="true"是， 自动发现并同步分区元数据到 Hive Metastore 中。
创建外部分区表时，会自动添加发现分区表属性，对于内部分区表，可以手动添加表属性。

## 保留分区

当指定了保留时间间隔表属性时，将自动删除时间间隔前的分区。例如，同时设置表属性 "discover.partitions"="true" 和 "partition.retention.period"="7d" ，将自动删除7天前的分区。

## 修复分区

Hive Metastore中存储每个表的分区信息。但是，如果将新分区直接添加到 HDFS（例如通过使用`hadoop fs -put`命令）或从 HDFS 中删除，则Metastore（以及 Hive）将不会意识到分区信息的更改，除非用户`ALTER TABLE table_name ADD/DROP PARTITION`对每个新添加或删除的分区运行命令。

但是，用户可以手动修复，这会将有关分区的元数据更新到 Hive Metastore

```sql
MSCK [REPAIR] TABLE table_name [ADD/DROP/SYNC PARTITIONS];
```

- 默认选项是 ADD PARTITIONS。会将 HDFS 上存在但不在 Metastore 中的任何分区添加到 Metastore
- DROP PARTITIONS 选项将从 Metastore 中删除分区信息，该信息已从 HDFS 中删除。
- SYNC PARTITIONS 选项等效于调用 ADD 和 DROP PARTITIONS。
- 不带 REPAIR 选项的 MSCK 命令可用于查找有关元数据不匹配Metastore的详细信息

## 删除分区

```sql
ALTER TABLE table_name DROP [IF EXISTS] 
	PARTITION partition_spec
	[, PARTITION partition_spec, ...]
    [IGNORE PROTECTION] 
    [PURGE]; 
TRUNCATE TABLE table_name PARTITION (col_name = col_value,...); -- 截断分区
```

- 对于受NO_DROP CASCADE保护的表，您可以使用谓词 IGNORE PROTECTION 删除指定的分区或分区集。在 2.0.0 及更高版本中不再可用。
- 如果指定了 PURGE，则分区数据不会进入 .Trash/Current 目录

## 归档分区

存档是一种将分区文件移动到 Hadoop 存档 (HAR) 的功能。

```sql
ALTER TABLE table_name ARCHIVE PARTITION partition_spec;
ALTER TABLE table_name UNARCHIVE PARTITION partition_spec;
```

## 部分分区

可以使用带有部分分区规范的单个 ALTER 语句一次更改许多现有分区：

```sql
SET hive.exec.dynamic.partition = true;
  
-- This will alter all existing partitions in the table with ds='2008-04-08' 
ALTER TABLE foo PARTITION (ds='2008-04-08', hr) CHANGE COLUMN dec_column_name dec_column_name DECIMAL(38,18);
 
-- This will alter all existing partitions in the table 
ALTER TABLE foo PARTITION (ds, hr) CHANGE COLUMN dec_column_name dec_column_name DECIMAL(38,18);
```

# 视图

视图就是由 SELECT 语句指定的一个纯逻辑对象，每次查询视图时都会导出该查询。与表不同，视图不会存储任何数据。

```sql
SHOW VIEWS [IN/FROM database_name] [LIKE 'pattern_with_wildcards'];  -- 列出视图
SHOW MATERIALIZED VIEWS [IN/FROM database_name] [LIKE 'pattern_with_wildcards’]; -- 列出实体化视图 
SHOW CREATE TABLE [db_name.]view_name;  -- 查看建表语句
```

## 创建视图

```sql
CREATE [OR REPLACE] VIEW [IF NOT EXISTS] [db_name.]view_name 
[(column_name [COMMENT column_comment], ...) ]
[COMMENT view_comment]
[TBLPROPERTIES (property_name = property_value, ...)]
AS SELECT ...;
```

- 如果未提供列名，则视图列的名称将自动从定义的 SELECT 表达式派生。（注释不会自动从基础列继承）
- 视图的结构在创建视图时被冻结；对基础表的后续更改（例如添加列）将不会反映在视图的架构中。如果以不兼容的方式删除或更改基础表，则后续尝试查询失败时视图将无效。
- 视图是只读的，不能用作 LOAD/INSERT/ALTER 的目标。要更改元数据，请参阅 ALTER VIEW。

```sql
SHOW CREATE TABLE view_name;  -- 显示创建视图的 CREATE VIEW 语句
SHOW VIEWS;  -- 显示数据库中的视图列表（从 Hive 2.2.0 开始）
```

## 删除视图

```sql
DROP VIEW [IF EXISTS] [db_name.]view_name;
```

- DROP VIEW 删除指定视图的元数据
- 当删除一个被其他视图引用的视图时，不会给出警告（依赖视图被悬空无效，必须由用户删除或重新创建）。

## 修改视图

```sql
-- 更改视图属性
ALTER VIEW [db_name.]view_name SET TBLPROPERTIES 
(property_name = property_value, ...);
-- 更改视图定义
ALTER VIEW [db_name.]view_name AS select_statement;
```

## 实体化视图

**创建实体化视图**

```sql
CREATE MATERIALIZED VIEW [IF NOT EXISTS] [db_name.]materialized_view_name
  [DISABLE REWRITE]
  [COMMENT materialized_view_comment]
  [PARTITIONED ON (col_name, ...)]
  [CLUSTERED ON (col_name, ...) | DISTRIBUTED ON (col_name, ...) SORTED ON (col_name, ...)]
  [
    [ROW FORMAT row_format]
    [STORED AS file_format]
      | STORED BY 'storage.handler.class.name' [WITH SERDEPROPERTIES (...)]
  ]
  [LOCATION hdfs_path]
  [TBLPROPERTIES (property_name=property_value, ...)]
AS SELECT ...;
```

**删除实体化视图**

```sql
DROP MATERIALIZED VIEW [db_name.]materialized_view_name;
```

**修改实体化视图**

创建实体化视图后，优化器将能够利用其定义语义自动重写传入的查询，从而加速查询执行。 
用户可以选择性地启用/禁用物化视图进行重写。使用以下语句：

```sql
ALTER MATERIALIZED VIEW [db_name.]materialized_view_name ENABLE|DISABLE REWRITE;
```

# 索引

Hive 索引的目标是提高对表的某些列的查询查找速度。如果没有索引，带有 `WHERE tab1.col1 = 10` 等条件的查询会加载整个表或分区并处理所有行。但是，如果 col1 存在索引，则只需加载和处理文件的一部分。

```sql
SHOW [FORMATTED] (INDEX|INDEXES) ON table_with_index [(FROM|IN) db_name];
```

SHOW INDEXES 显示某个列上的所有索引，以及有关它们的信息：索引名称、表名称、用作键的列名称、索引表名称、索引类型和注释。如果使用 FORMATTED 关键字，则为每一列打印列标题。

## 创建索引

```sql
CREATE INDEX index_name
  ON TABLE base_table_name (col_name, ...)
  AS index_type
  [WITH DEFERRED REBUILD]
  [IDXPROPERTIES (property_name=property_value, ...)]
  [IN TABLE index_table_name]
  [
     [ ROW FORMAT ...] STORED AS ...
     | STORED BY ...
  ]
  [LOCATION hdfs_path]
  [TBLPROPERTIES (...)]
  [COMMENT "index comment"];
```

## 删除索引

```sql
DROP INDEX [IF EXISTS] index_name ON table_name;
```

## 修改索引

```sql
ALTER INDEX index_name ON table_name [PARTITION partition_spec] REBUILD;
```

# 临时宏

## 创建临时宏

```
CREATE TEMPORARY MACRO macro_name([col_name col_type, ...]) expression;
```

CREATE TEMPORARY MACRO 使用给定的可选列列表作为表达式的输入来创建宏。宏在当前会话期间存在。

**例子：**

```sql
CREATE TEMPORARY MACRO fixed_number() 42;
CREATE TEMPORARY MACRO string_len_plus_two(x string) length(x) + 2;
CREATE TEMPORARY MACRO simple_add (x int, y int) x + y;
```

## 删除临时宏

```sql
DROP TEMPORARY MACRO [IF EXISTS] macro_name;
```

