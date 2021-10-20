---
ID: 93a63169509ac11fd77c443d5c8633e3
title: 大数据手册(Hive)--HiveQL
date: 2018-07-03 17:57:36
categories: [大数据]
tags: [大数据,hive,SQL]
cover: /img/apache-hive.png
top_img: /img/hive-bg.png
description: Hive 是一个基于Apache Hadoop的数据仓库基础设施
---

Hive 是一个基于[Apache Hadoop](http://hadoop.apache.org/)的数据仓库基础设施。Hadoop 为商用硬件上的数据存储和处理提供了大规模的横向扩展和容错能力。
<!-- more -->
Hive 旨在实现对大量数据的轻松数据汇总、即席查询和分析。它提供了 SQL，使用户能够轻松地进行临时查询、汇总和数据分析。

SQL语言包含6个部分：

1. 数据查询语言(DQL）：用以从表中获得数据，包括 SELECT，WHERE，ORDER BY，GROUP BY和HAVING等。

2. 数据操作语言(DML）：其语句包括动词 INSERT、UPDATE、DELETE。它们分别用于添加、修改和删除。

3. 事务控制语言(TCL)：它的语句能确保被DML语句影响的表的所有行及时得以更新。包括COMMIT（提交）命令、SAVEPOINT（保存点）命令、ROLLBACK（回滚）命令。

4. 数据控制语言(DCL)：它的语句通过GRANT或REVOKE实现权限控制，确定单个用户和用户组对数据库对象数据库对象)的访问。

5. 数据定义语言(DDL)：其语句包括动词CREATE、ALTER和DROP。

6. 指针控制语言(CCL)：它的语句，像DECLARE CURSOR，FETCH INTO和UPDATE WHERE CURRENT用于对一个或多个表单独行的操作。

<!-- more -->

# Hive 命令

命令是非SQL语句，例如设置属性或添加资源。它们可以在 HiveQL 脚本中使用，也可以直接在命令行界面 (Command Line Interface, CLI) 或 Beeline 中使用。

HiveServer2（在 Hive 0.11 中引入）有自己的 CLI，称为[Beeline](https://cwiki.apache.org/confluence/display/Hive/HiveServer2+Clients#HiveServer2Clients-Beeline–NewCommandLineShell)，它是一个基于 SQLLine 的 JDBC 客户端。由于新的开发集中在 HiveServer2 上，Hive CLI 将很快被弃用，以支持 Beeline ( HIVE-10511)。

## Hive 命令选项

`$HIVE_HOME/bin/hive` 是一个 shell 实用程序，可用于以交互或批处理模式运行 Hive 查询。
在没有选项的情况下运行时进入CLI交互界面，默认进入default数据库


**hive** 命令选项

- `hive --database dbname` 指定要使用的数据库。
- `hive -d <key=value>` 或 `hive --define <key=value>`应用于 Hive 命令的变量替换。
- `hive -e '<query-string>'` 执行查询字符串。
- `hive -f <filepath>` 从文件中执行一个或多个 SQL 查询。
- `hive -H` 或 `hive --help` 打印帮助信息。
- `hive  -h <hostname>` 连接到远程主机上的 Hive 服务器
	  `--hiveconf <property=value>` 设置hive的参数
    `--hivevar <key=value>` 设置 hive 内变量
- `hive -i <init.sql>` 进入交互界面时，先执行初始化SQL文件
- `hive -p <port>` 在端口号上连接到 Hive 服务器
- `hive -S` 或 `hive --silent` 静默模式，不显示转化MR-Job的信息，只显示最终结果
- `hive -v` 或 `hive --verbose` 详细模式，将执行的 SQL 回显到控制台

```bash
hive -e 'select a.col from tab1 a'
hive --hiveconf hive.cli.print.current.db=false
hive -S -e 'select a.col from tab1 a' > a.txt
hive -f /home/my/hive-script.sql
hive -f hdfs://<namenode>:<port>/hive-script.sql
hive -i /home/my/hive-init.sql
```

## Hive 交互式命令

`hive` 在没有`-e` 或者 `-f` 选项的情况下进入 hive交互式模式。

| 命令                | 描述                                                         |
| :------------------ | :----------------------------------------------------------- |
| exit, quit          | 离开交互式 shell。                                           |
| reset               | 将配置重置为默认值。                                         |
| `set <key>=<value>` | 设置特定配置变量（键）的值。[Hve官网](https://cwiki.apache.org/confluence/display/Hive/Configuration+Properties) |
| set                 | 打印由用户或 Hive 覆盖的配置变量列表。                       |
| set -v              | 打印所有 Hadoop 和 Hive 配置变量。                           |
| `! <command>`       | 从 Hive shell 执行 Linux shell 命令。                        |
| `dfs <dfs command>` | 从 Hive shell 执行 dfs 命令。                                |
| `<query string>`    | 执行 Hive 查询并将结果打印到标准输出。                       |
| `source <filepath>` | 在 CLI 中执行脚本文件。                                      |

```shell
hive> select * from dummy;
hive> source ./hive-script.sql;
hive> !echo 'hello hive';      # 执行 shell命令，前面加 ！即可
hive> dfs -ls;
hive> set mapred.reduce.tasks=32;
```

# 数据单位

按粒度顺序 - Hive 数据单位依次为：

- **数据库**：数据库命名空间的作用是避免表、视图、分区、列等的命名冲突。数据库还可用于为用户或用户组强制实施安全性。
- **表**：具有相同模式的同类数据单元。例如 page_views 表，可以包含以下列（模式）：
  - `timestamp`— INT 类型，对应于查看页面的 UNIX 时间戳。
  - `userid` —BIGINT 类型，标识查看页面的用户ID 。
  - `page_url`—STRING 类型，这是捕获页面位置的 。
  - `referer_url`—STRING 类型，用于捕获用户到达当前页面的页面位置。
  - `IP`—STRING 类型，用于捕获发出页面请求的 IP 地址。
- **分区**：每个表可以有一个或多个分区键，它决定了数据的存储方式。分区——除了作为存储单元——还允许用户有效地识别满足指定条件的行。分区列是虚拟列，
- **Buckets**（或**Clusters**）：每个分区中的数据可以根据表中某列的哈希函数值依次划分为Buckets。例如，page_views 表可以按 userid 进行分桶，userid 是 page_view 表的除分区列之外的列之一。这些可用于有效地采样数据。

请注意，不必对表进行分区或分桶，但这些抽象允许系统在查询处理期间修剪大量数据，从而加快查询执行速度。

# 数据类型和运算符

## 原始数据类型

| 数字类型      | 说明                                            | 示例 |
| ------------- | ----------------------------------------------- | ---- |
| TINYINT       | 1 字节有符号整数，$[-2^7, 2^7)$，后缀为 Y       | 10Y  |
| SMALLINT      | 2 字节有符号整数，$[-2^{15}, 2^{15})$，后缀为 S | 10S  |
| INT/INTEGER   | 4 字节有符号整数，$[-2^{31}, 2^{31})$           | 10   |
| BIGINT        | 8 字节有符号整数，$[-2^{63}, 2^{63})$，后缀为 L | 10L  |
| FLOAT         | 4 字节单精度浮点数                              |      |
| DOUBLE        | 8 字节双精度浮点数                              |      |
| DECIMAL(p, s) | 用户可定义的精度和有效长度                      |      |
| NUMERIC(p, s) | 与DECIMAL相同，从Hive 3.0.0开始引入             |      |

| 字符串类型        | 说明     |
| :---------------- | :------- |
| STRING(size)      | 字符串   |
| VARCHAR(max_size) | 可变长度 |
| CHAR(size)        | 固定长度 |

| 时间日期类型                   | 说明                                                         |
| :----------------------------- | :----------------------------------------------------------- |
| TIMESTAMP                      | 精确到纳秒的时间戳（9 位小数精度），格式 `"YYYY-MM-DD HH:MM:SS.fffffffff"` |
| TIMESTAMP WITH LOCAL TIME ZONE | 有时区的时间戳                                               |
| DATE                           | 格式 `"YYYY-­MM-­DD"`                                        |
| INTERVAL                       | 时间间隔                                                     |


| 布尔类型 | 说明       |
| -------- | ---------- |
| BOOLEAN  | TRUE/FALSE |

| 二进制类型 | 说明     |
| ---------- | -------- |
| BINARY     | 字节序列 |

## 复杂数据类型

| 复杂类型                                                | 说明     |
| ------------------------------------------------------- | -------- |
| ARRAY<data_type>                                        | 数组     |
| MAP<primitive_type, data_type>                          | 映射     |
| STRUCT<col_name : data_type [COMMENT col_comment], ...> | 结构     |
| UNIONTYPE<data_type, data_type, ...>                    | 联合类型 |

| 构建函数                                    | 描述                                         |
| :------------------------------------------ | :------------------------------------------- |
| map(key1, value1, key2, value2, ...)        | 使用给定的key/value对创建映射                |
| struct(val1, val2, val3, ...)               | 创建结构体，结构字段名称将是 col1, col2, ... |
| named_struct(name1, val1, name2, val2, ...) | 使用给定的字段名称和值创建一个结构体         |
| array(val1, val2, ...)                      | 用给定的元素创建一个数组                     |
| create_union(tag, val1, val2, ...)          | 使用 tag 参数指向的值创建联合类型            |

| 运算符 | 操作类型                 | 描述                                        |
| ------ | ------------------------ | ------------------------------------------- |
| A[n]   | A是一个数组，n是一个整数 | 它返回数组A的第n个元素，第一个元素的索引0。 |
| M[key] | M 是一个 Map             | 它返回对应于映射中key的值。                 |
| S.x    | S 是一个结构体           | 它返回S的x部分                              |

```sql
hive> desc student;
id	string
chinese	float
math	float
english float

hive> select id,array(chinese,math,english) as score from student;
id	score
001	[90,95,80]
002	[70,65,83]
... ...

hive> select id,map('c',chinese,'m',math,'e',english) as score from student;
id	score
001	{'c':90,'m':95,'e':80}
002	{'c':70,'m':65,'e':83}
... ...
```

```sql
-- 将学生按score随机平均分配班级
create table tmp
as
select id,score,
case when score='A' then array('class 1','class 2')
     when score='B' then array('class 3','class 4')
     when score='C' then array('class 5','class 6')
else null end as class
from students;

select id,score,
case when rn%2=0 then class[0] else class[1] end as class
from
(select *,row_number() over(partition by score order by rand()) as rn from tmp) a
```

## 运算符

关系运算符|说明
:---|:---
A = B<br />A==B|等于
A<=>B|等于，适用于NULL
A<>B<br />A != B|不等于
A < B|小于
A <= B|大于等于
A > B|大于
A >= B|大于等于
A [NOT] BETWEEN B AND C|[NOT] B<= A <=C
A IS [NOT] NULL|空值/非空
A IS [NOT]  TRUE\|FALSE|
A [NOT] LIKE B|如果字符串模式A匹配到B，使用通配符，例如`%`
A RLIKE B|匹配正则表达式B
A REGEXP B|同于RLIKE.

算数运算符|描述
:---|---
A + B|加
A - B|减
A * B|乘
A / B|除
A DIV B|整除
A % B|取余数
A & B|A和B的按位与结果
A \| B|A和B的按位或结果
A ^ B|A和B的按位异或结果
~A|A按位非的结果

逻辑运算符|描述
:---|---
A AND B|与
A OR B|或
NOT A|非
!A|同 NOT A
A IN (val1, val2, ...)|A等于其中任何一个
A NOT IN (val1, val2, ...)|A不等于其中任何一个
[NOT] EXISTS (subquery)|subquery返回至少一行，则返回TRUE

| 字符运算符 | 描述                             |
| ---------- | -------------------------------- |
| A \|\|B    | 字符串连接（从HIve 2.2开始支持） |

# 数据库操作

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

# 连接器操作

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

# 管理表结构

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


# 数据查询

## Select

```sql
SELECT [ALL | DISTINCT] select_expr, select_expr, ...
FROM table_reference
[WHERE where_condition]
[GROUP BY col_list] [HAVING having_condition]
[ORDER BY col_list]
[CLUSTER BY col_list] | [DISTRIBUTE BY col_list] [SORT BY col_list]
[LIMIT [offset,] rows]
```

- SELECT 语句可以是 union 查询的一部分，也可以是另一个查询的子查询。
- table_reference表示查询的输入。它可以是常规表、视图、连接查询或子查询。
- 在反引号 ( \`) 中指定的任何列名都按字面处理。
- 默认为 ALL 返回所有匹配的行。DISTINCT 指定从结果集中删除重复行。

**WHERE 语句**：是一个布尔表达式。Hive在 WHERE 子句中支持许多运算符、 UDF 和某些类型的子查询。

```sql
SELECT * FROM sales WHERE amount > 10 AND region = "US"
```

**LIMIT 子句**：可用于限制 SELECT 语句返回的行数。

LIMIT 需要一个或两个数字参数，它们都必须是非负整数常量。第一个参数指定要返回的第一行的偏移量，第二个参数指定要返回的最大行数。当给出单个参数时，它代表最大行数，偏移量默认为 0。

以下查询返回要创建的前 5 个客户

```sql
SELECT * FROM customers ORDER BY create_date LIMIT 5;
```

 以下查询返回要创建的第 3 个到第 7 个客户

```sql
SELECT * FROM customers ORDER BY create_date LIMIT 2, 5;
```

**REGEX 列规则**

配置属性 `hive.support.quoted.identifiers = none`，则反引号内被解释为 Java 正则表达式。以下查询选择除 ds 和 hr 之外的所有列。

````sql
hive.support.quoted.identifiers = none;
SELECT `(ds|hr)?+.+` FROM sales;
````

**分区查询**

通常，SELECT 查询会扫描整个表（除了用于采样）。如果表是使用PARTITIONED BY子句创建的，则查询可以进行分区修剪并仅扫描与查询指定的分区相关的表的一小部分。

## Group By

使用 group by 子句时，select 语句只能包含 group by 子句中包含的列。当然，您也可以尽可能多的聚合函数。例如，为了按性别计算不同用户的数量，可以编写以下查询：

```sql
SELECT pv_users.gender, count (DISTINCT pv_users.userid)
FROM pv_users
GROUP BY pv_users.gender;
```

可以同时进行多个聚合，但是，任何两个聚合都不能具有不同的 DISTINCT 列。例如，以下是可能的

```sql
SELECT pv_users.gender, count(*)
	count(DISTINCT pv_users.userid), 
	sum(DISTINCT pv_users.userid)
FROM pv_users
GROUP BY pv_users.gender;
```

但是，**不允许**在同一个查询中使用多个 DISTINCT 表达式。

```sql
SELECT pv_users.gender, 
	count(DISTINCT pv_users.userid), 
	count(DISTINCT pv_users.ip)
FROM pv_users
GROUP BY pv_users.gender;
```

**map 端聚合**

*hive.map.aggr*控制我们如何进行聚合，默认值为false。如果设置为true，Hive会直接在map任务中做一级聚合。
这通常提供更好的效率，但可能需要更多内存才能成功运行。

```sql
set hive.map.aggr=true;
SELECT COUNT(*) FROM table2;
```

## 排序

**ORDER BY** ：Hive QL 中的ORDER BY语法类似于SQL 语言中的ORDER BY语法。

```sql
SELECT expressions FROM src 
ORDER BY colName [ASC | DESC] [NULLS FIRST | NULLS LAST] , ...
```

- 在严格模式下（即hive.mapred.mode =strict），order by 子句后面必须跟一个 limit 子句。。原因是为了强制所有结果的总顺序，必须有一个reducer来对最终输出进行排序。如果输出中的行数太大，单个reducer可能需要很长时间才能完成。
- 默认排序顺序是升序 (ASC)。ASC 顺序的默认空排序顺序是 NULLS FIRST，而 DESC 顺序的默认空排序顺序是 NULLS LAST。

**SORT BY** ：类似于ORDER BY语法。区别在于ORDER BY保证输出中的总顺序，而SORT BY只保证reducer中行的排序。如果有多个reducer，SORT BY 可能会给出部分排序的最终结果。

```sql
SELECT expressions FROM src 
SORT BY colName [ASC | DESC] [NULLS FIRST | NULLS LAST] , ...
```

**Cluster By 和 Distribute By 的语法**

Cluster By和Distribute By主要与Transform/Map-Reduce Scripts 一起使用。但是，如果需要为后续查询的输出进行分区和排序，它有时在 SELECT 语句中很有用。

```sql
select * from emp 
	distribute by deptno -- 指定分区
	sort by sal;         -- 局部排序
select * from emp cluster by sal;   -- 同时指定分区和排序字段
```

Cluster By是Distribute By和Sort By的快捷方式。Hive 使用Distribute By中的列在 reducer 之间分配行。具有相同Distribute By列的所有行都将进入相同的 reducer。

用户可以指定Distribute By和Sort By，而不是指定Cluster By，因此分区列和排序列可以不同。通常的情况是分区列是排序列的前缀，但这不是必需的。

```sql
SELECT col1, col2 FROM t1 CLUSTER BY col1;
SELECT col1, col2 FROM t1 DISTRIBUTE BY col1;
SELECT col1, col2 FROM t1 DISTRIBUTE BY col1 SORT BY col1 ASC, col2 DESC;
```

## Joins



<img src="https://gitee.com/WilenWu/images/raw/master/common/hive_join.png" alt="join" width="80%" />

**连接语法**

```sql
select_statement
FROM from_statement
JOIN table_reference join_condition 
WHERE where_condition

-- join_condition: ON (expr = expr[ AND expr = expr ...])
```

用户可以使用关键字来限定连接类型

- 内连接：``[INNER] JOIN`，默认值，join_condition 可选（去除后等效于交叉连接）
- 外连接：`{LEFT|RIGHT|FULL} [OUTER] JOIN`，左保留、右保留或两侧保留
- 半连接：`LEFT SEMI JOIN` ，等价于 IN/EXISTS 子查询。只能返回左表记录，连接时遇到右表重复记录，左表会跳过。
- 交叉连接：`CROSS JOIN`，join_condition可选
- 联接发生在 WHERE 子句之前
- 无论是 LEFT JOIN 还是 RIGHT JOIN，连接都是从左到右关联的。

**Example**

简单连接

```sql
SELECT pv.*, u.gender, u.age FROM user u 
JOIN page_view pv ON (pv.userid = u.id)
WHERE pv.date = '2008-03-03';
SELECT a.* FROM a LEFT OUTER JOIN b ON (a.id <> b.id);
```

多表连接

```sql
SELECT pv.*, u.gender, u.age, f.friends FROM page_view pv 
	JOIN user u ON (pv.userid = u.id) 
	JOIN friend_list f ON (u.id = f.uid)
WHERE pv.date = '2008-03-03';
```

连接前预过滤

```sql
SELECT a.val, b.val FROM a 
LEFT OUTER JOIN b
ON (a.key=b.key AND b.ds='2009-07-07' AND a.ds='2009-07-07');
```

半连接

```sql
SELECT u.* FROM user u 
LEFT SEMI JOIN page_view pv ON (pv.userid = u.id)
WHERE pv.date = '2008-03-03';
-- 等价于
SELECT u.* FROM user u 
WHERE pv.date = '2008-03-03'
    AND u.id IN (SELECT pv.userid FROM page_view pv);
```

**map/reduce 作业**

如果每个表的连接子句中都使用了相同的列，则 Hive 会将多个表的连接转换为单个 map/reduce 作业，例如

```sql
SELECT a.val, b.val, c.val FROM a 
	JOIN b ON (a.key = b.key1) 
	JOIN c ON (c.key = b.key1)
```
转换为单个 map/reduce 作业，因为连接中只涉及 b 的 key1 列。在连接的每个 map/reduce 阶段，序列中的最后一个表通过 reducer 流式传输，而其他表则被缓存。因此，将最大的表出现在序列的最后，有助于减少reducer 中用于缓存连接键的特定值的行所需的内存。

另一方面

```sql
SELECT a.val, b.val, c.val FROM a 
	JOIN b ON (a.key = b.key1) 
	JOIN c ON (c.key = b.key2)
```
被转换为两个 map/reduce 作业，因为 b 中的 key1 列用于第一个连接条件，而 b 中的 key2 列用于第二个连接条件。第一个 map/reduce 作业将 a 与 b 连接，然后在第二个 map/reduce 作业中将结果与 c 连接。

**STREAMTABLE**：在连接的每个 map/reduce 阶段，可以通过提示指定要流式传输的表。例如

```sql
SELECT /*+ STREAMTABLE(a) */ a.val, b.val, c.val FROM a 
	JOIN b ON (a.key = b.key1) 
	JOIN c ON (c.key = b.key1)
```

所有三个表都连接在一个 map/reduce 作业中，表 b 和 c 的键的特定值的值缓冲在 reducer 的内存中。然后对于从 a 中检索的每一行，使用缓冲的行计算连接。如果省略 STREAMTABLE 提示，Hive 会流式传输连接中最右边的表。

**MAPJOIN**：如果要连接的表只有一个表且很小，则连接可以仅 map 作业执行

```sql
SELECT /*+ MAPJOIN(b) */ a.key, a.value
FROM a JOIN b ON a.key = b.key;
```


不需要 reducer，对于 A 的每个映射器，B 都被完全读取。限制是不能执行FULL/RIGHT OUTER JOIN b。

## Union

**语法**：UNION 用于将多个 SELECT 语句的结果组合成一个结果集

```sql
select_statement 
UNION [ALL | DISTINCT] select_statement 
UNION [ALL | DISTINCT] select_statement 
... ...;
```

- UNION的默认行为 是从结果中删除重复的行，DISTINCT 关键字可选。
- 可以在同一查询中混合使用 UNION ALL 和 UNION DISTINCT。混合 UNION 类型的处理方式是 DISTINCT 联合覆盖其左侧的任何 ALL 联合。
- 每个 select_statement 返回的列数、列名称、列类型必须相同。否则，将引发架构错误。

**FROM 子句中的 UNION**：如果必须对 UNION 的结果进行一些额外的处理，可以将整个语句表达式嵌入到 FROM 子句中，如下所示：

```sql
SELECT *
FROM (
  select_statement
  UNION ALL
  select_statement
) unionResult
```

```sql
SELECT u.id, actions.date
FROM (
    SELECT av.uid AS uid
    FROM action_video av
    WHERE av.date = '2008-06-03'
    UNION ALL
    SELECT ac.uid AS uid
    FROM action_comment ac
    WHERE ac.date = '2008-06-03'
 ) actions JOIN users u ON (u.id = actions.uid)
```

**子规范**：要将 ORDER BY、SORT BY、CLUSTER BY、DISTRIBUTE BY 或 LIMIT 应用于单个 SELECT，请将子句放在包含 SELECT 的括号内：

```sql
SELECT key FROM (SELECT key FROM src ORDER BY key LIMIT 10) subq1
UNION
SELECT key FROM (SELECT key FROM src1 ORDER BY key LIMIT 10) subq2
```

要将 ORDER BY、SORT BY、CLUSTER BY、DISTRIBUTE BY 或 LIMIT 子句应用于整个 UNION 结果，请将子句放在最后。

```sql
SELECT key FROM src
UNION
SELECT key FROM src1 
ORDER BY key LIMIT 10
```

# 内置函数

参考资料：[Hive 官方文档](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF#LanguageManualUDF-Built-inFunctions)

```sql
SHOW FUNCTIONS [LIKE "pattern"]; -- 列出用户定义和内置函数
DESCRIBE FUNCTION function_name;   -- 查看函数信息
DESCRIBE FUNCTION EXTENDED function_name; 
```

## 数学函数

| Return Type   | Name (Signature)                             | Description                                         |
| ------------- | -------------------------------------------- | --------------------------------------------------- |
| BIGINT        | round(DOUBLE a)                              | 四舍五入取整                                        |
| DOUBLE        | round(DOUBLE a, INT d)                       | 四舍五入，保留d位小数                               |
| BIGINT        | floor(DOUBLE a)                              | 向下取整                                            |
| BIGINT        | ceil(DOUBLE a), ceiling(DOUBLE a)            | 向上取整                                            |
| DOUBLE        | rand(), rand(INT seed)                       | 返回0到1间的随机数                                  |
| DOUBLE        | exp(DOUBLE a),                               | $e^a$                                               |
| DOUBLE        | ln(DOUBLE a)                                 | $\ln a$                                             |
| DOUBLE        | log10(DOUBLE a)                              | $\log_{10}a$                                        |
| DOUBLE        | log2(DOUBLE a)                               | $\log_{2}a$                                         |
| DOUBLE        | log(DOUBLE b, DOUBLE a)                      | $\log_{b}a$                                         |
| DOUBLE        | pow(DOUBLE a, DOUBLE p)                      | $a^p$                                               |
| DOUBLE        | sqrt(DOUBLE a)                               | $\sqrt{a}$                                          |
| STRING        | bin(BIGINT a)                                | 返回二进制数字对应的字段                            |
| STRING        | hex(BIGINT a), hex(STRING a), hex(BINARY a)  | 返回十六进制数字对应的字段                          |
| BINARY        | unhex(STRING a)                              | hex的逆方法                                         |
| DOUBLE        | abs(DOUBLE a)                                | 绝对值                                              |
| INT or DOUBLE | pmod(INT a, INT b), pmod(DOUBLE a, DOUBLE b) | $a \mod b$                                          |
| DOUBLE        | sin(DOUBLE a)                                | 正弦值                                              |
| DOUBLE        | asin(DOUBLE a)                               | 反正弦值                                            |
| DOUBLE        | cos(DOUBLE a)                                | 余弦值                                              |
| DOUBLE        | acos(DOUBLE a)                               | 反余弦值                                            |
| DOUBLE        | tan(DOUBLE a)                                | 正切值                                              |
| DOUBLE        | atan(DOUBLE a)                               | 反正切值                                            |
| DOUBLE        | degrees(DOUBLE a)                            | 将弧度值转换角度值                                  |
| DOUBLE        | radians(DOUBLE a)                            | 将角度值转换成弧度值                                |
| INT or DOUBLE | positive(INT a), positive(DOUBLE a)          | 返回a                                               |
| INT or DOUBLE | negative(INT a), negative(DOUBLE a)          | 返回-a                                              |
| DOUBLE or INT | sign(DOUBLE a)                               | 如果a是正数则返回1.0，是负数则返回-1.0，否则返回0.0 |
| DOUBLE        | e()                                          | 数学常数e                                           |
| DOUBLE        | pi()                                         | 数学常数$\pi$                                       |
| BIGINT        | factorial(INT a)                             | 求a的阶乘                                           |
| DOUBLE        | cbrt(DOUBLE a)                               | 求a的立方根                                         |
| TYPE          | greatest(T v1, T v2, ...)                    | 求最大值                                            |
| TYPE          | least(T v1, T v2, ...)                       | 求最小值                                            |

## 集合函数

| Return Type | Name(Signature)                                   | Description                                                  |
| ----------- | ------------------------------------------------- | ------------------------------------------------------------ |
| int         | `size(Map<K.V>)`                                  | 返回map的长度                                                |
| int         | `size(Array<T>)`                                  | 返回数组的长度                                               |
| array       | map_keys(Map<K.V>)                                | 返回map中的所有key                                           |
| array       | map_values(Map<K.V>)                              | 返回map中的所有value                                         |
| boolean     | `array_contains(Array<T>, value)`                 | 数组中是否包含value                                          |
| array       | `sort_array(Array<T>)`                            | 对数组进行排序并返回                                         |
| string      | `concat_ws(string SEP, array<string>)`            | Array中的元素拼接                                            |
| array       | sentences(string str, string lang, string locale) | 字符串str将被转换成单词数组                                  |
| array       | split(string str, string pat)                     | 按照正则表达式pat来分割字符串str                             |
| map         | str_to_map(text[, delimiter1, delimiter2])        | 将字符串str按照指定分隔符转换成Map，第一个参数是需要转换字符串，第二个参数是键值对之间的分隔符，默认为逗号;第三个参数是键值之间的分隔符，默认为"=" |
| ARRAY       | collect_set(col)                                  | 返回一组消除了重复元素的对象                                 |
| ARRAY       | collect_list(col)                                 | 返回具有重复项的对象列表                                     |

## 类型转换函数

| Return Type | Name(Signature)        | Description          |
| ----------- | ---------------------- | -------------------- |
| binary      | binary(string\|binary) | 转换成二进制         |
| type        | `cast(expr as <type>)` | 将expr转换成type类型 |

## 日期函数

| Return Type | Name(Signature)                                   | Description                                                  |
| ----------- | ------------------------------------------------- | ------------------------------------------------------------ |
| string      | from_unixtime(bigint unixtime[, string format])   | 将Unix时间戳 (1970-01-0100:00:00 UTC 为起始秒) 转化为时间字符 |
| bigint      | unix_timestamp()                                  | 获取本地时区下的时间戳                                       |
| bigint      | unix_timestamp(string date)                       | 将格式为 yyyy-MM-dd HH:mm:ss 的时间字符串转换成时间戳        |
| bigint      | unix_timestamp(string date, string fmt)           | 将指定时间字符串格式字符串转换成Unix时间戳                   |
| string      | to_date(string timestamp)                         | 返回时间字符串的日期部分                                     |
| int         | year(string date)                                 | 年份部分                                                     |
| int         | quarter(date/timestamp/string)                    | 季度部分                                                     |
| int         | month(string date)                                | 月份部分                                                     |
| int         | day(string date) dayofmonth(date)                 | 天                                                           |
| int         | hour(string date)                                 | 小时                                                         |
| int         | minute(string date)                               | 分钟                                                         |
| int         | second(string date)                               | 秒                                                           |
| int         | weekofyear(string date)                           | 一年中的第几个周内                                           |
| int         | extract(field FROM source)                        | 提取日期组件                                                 |
| int         | datediff(string enddate, string startdate)        | 相差的天数                                                   |
| string      | date_add(string startdate, int days)              | 从开始时间startdate加上days                                  |
| string      | date_sub(string startdate, int days)              | 从开始时间startdate减去days                                  |
| timestamp   | from_utc_timestamp(timestamp, string timezone)    | 如果给定的时间戳并非UTC，则将其转化成指定的时区下时间戳      |
| timestamp   | to_utc_timestamp(timestamp, string timezone)      | 如果给定的时间戳指定的时区下时间戳，则将其转化成UTC下的时间戳 |
| date        | current_date                                      | 返回当前时间日期                                             |
| timestamp   | current_timestamp                                 | 返回当前时间戳                                               |
| string      | add_months(string start_date, int num_months)     | 返回当前时间下再增加num_months个月的日期                     |
| string      | last_day(string date)                             | 返回这个月的最后一天的日期，忽略时分秒部分（HH:mm:ss）       |
| string      | next_day(string start_date, string day_of_week)   | 返回当前时间的下一个星期X所对应的日期                        |
| string      | trunc(string date, string format)                 | 返回时间的最开始年份或月份。注意所支持的格式为MONTH/MON/MM, YEAR/YYYY/YY |
| double      | months_between(date1, date2)                      | 返回date1与date2之间相差的月份                               |
| string      | date_format(date/timestamp/string ts, string fmt) | 按指定[Format][dt]返回日期字符                               |

[dt]: https://docs.oracle.com/javase/7/docs/api/java/text/SimpleDateFormat.htm

 ```sql
select extract(month from "2016-10-20") -- results in 10.
select extract(hour from "2016-10-20 05:06:07") -- results in 5.
select extract(dayofweek from "2016-10-20 05:06:07") -- results in 5.
select extract(month from interval '1-3' year to month) -- results in 3.
select extract(minute from interval '3 12:20:30' day to second) -- results in 20
 ```

## 条件函数

| Return Type | Name(Signature)                                            | Description                                      |
| ----------- | ---------------------------------------------------------- | ------------------------------------------------ |
| TYPE        | if(boolean testCondition, T valueTrue, T valueFalseOrNull) | 二分支语句                                       |
| TYPE        | nvl(T value, T default_value)                              | 返回第一个不是NULL的参数                         |
| TYPE        | COALESCE(T v1, T v2, ...)                                  | 返回第一个不是NULL的参数                         |
| TYPE        | CASE a WHEN b THEN c [WHEN d THEN e]* [ELSE f] END         | 多分支语句                                       |
| TYPE        | CASE WHEN a THEN b [WHEN c THEN d]* [ELSE e] END           | 多分支语句                                       |
| boolean     | isnull( a )                                                | 是否NULL                                         |
| boolean     | isnotnull ( a )                                            | 是否非NULL                                       |
| TYPE        | nullif( a, b )                                             | 如果 a=b，则返回 NULL；否则返回 a                |
| void        | assert_true(boolean condition)                             | 如果 condition 不为真，则抛出异常，否则返回 null |

## 字符函数

| Return Type | Name(Signature)                                              | Description                                                  |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| string      | A \|\| B                                                     | 字符串连接（从HIve 2.2开始支持）                             |
| string      | concat(string A, string B...)                                | 字符串连接                                                   |
| array       | `context_ngrams(array<array<string>>, array<string>, int K, int pf)` |                                                              |
| string      | concat_ws(string SEP, string A, string B...)                 | 指定分隔符拼接                                               |
| string      | `concat_ws(string SEP, array<string>)`                       | Array中的元素拼接                                            |
| string      | elt(N int,str1 string,str2 string,str3 string,...)           | 返回索引号处的字符串。例如 elt(2,'hello','world') 返回 'world'。如果 N 小于 1 或大于参数数量，则返回 NULL。 |
| int         | field(val T,val1 T,val2 T,val3 T,...)                        | 返回 val1,val2,val3,... 参数列表中 val 的索引，如果未找到则返回 0。例如field ('world','say','hello','world') 返回 3。 |
| int         | find_in_set(string str, string strList)                      | 返回以逗号分隔的字符串中str出现的位置，如果参数str为逗号或查找失败将返回0 |
| string      | format_number(number x, int d)                               | 数字转字符串                                                 |
| string      | get_json_object(string json_string, string path)             |                                                              |
| boolean     | in_file(string str, string filename)                         | 在文件中查找字符串                                           |
| int         | instr(string str, string substr)                             | 查找子字符串substr出现的位置，如果查找失败将返回0            |
| int         | length(string A)                                             | 字符串的长度                                                 |
| int         | locate(string substr, string str[, int pos])                 | 查找字符串str中的pos位置后字符串substr第一次出现的位置       |
| string      | lower(string A) <br />lcase(string A)                        | 小写                                                         |
| string      | lpad(string str, int len, string pad)                        | 在左端填充字符串 pad，长度为len                              |
| string      | ltrim(string A)                                              | 去掉左边空格                                                 |
| array       | `ngrams(array<array<string>>, int N, int K, int pf)`         |                                                              |
| string      | regexp_extract(string subject, string pattern, int index)    | 抽取字符串subject中符合正则表达式pattern的第index个部分的子字符串 |
| string      | regexp_replace(string INITIAL_STRING, string PATTERN, string REPLACEMENT) | 按照Java正则表达式PATTERN将字符串INTIAL_STRING中符合条件的部分成REPLACEMENT所指定的字符串 |
| string      | repeat(string str, int n)                                    | 重复输出n次字符串str                                         |
| string      | reverse(string A)                                            | 反转字符串                                                   |
| string      | rpad(string str, int len, string pad)                        | 在右端填充字符串 pad，长度为len                              |
| string      | rtrim(string A)                                              | 去掉右边空格                                                 |
| array       | sentences(string str, string lang, string locale)            | 字符串str将被转换成单词数组                                  |
| string      | space(int n)                                                 | 返回n个空格                                                  |
| array       | split(string str, string pat)                                | 按照正则表达式pat来分割字符串str                             |
| map         | str_to_map(text[, delimiter1, delimiter2])                   | 将字符串str按照指定分隔符转换成Map，第一个参数是需要转换字符串，第二个参数是键值对之间的分隔符，默认为逗号;第三个参数是键值之间的分隔符，默认为"=" |
| string      | substr(string A, int start) <br />substring(string A, int start) | 提取子字符串                                                 |
| string      | substr(string A, int start, int len) <br />substring(string A, int start, int len) | 提取长度为len的子字符串                                      |
| string      | substring_index(string A, string delim, int count)           | 截取第count分隔符之前的字符串，如count为正则从左边开始截取，如果为负则从右边开始截取 |
| string      | translate(string input, string from, string to)              | 字符串替换                                                   |
| string      | trim(string A)                                               | 去掉两边空格                                                 |
| string      | upper(string A) <br />ucase(string A)                        | 大写                                                         |
| string      | initcap(string A)                                            | 首字母大写                                                   |
| int         | levenshtein(string A, string B)                              | 计算两个字符串之间的差异大小                                 |

## 数据掩码函数

| Return Type | Name(Signature)                                              | Description                                                  |
| :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| string      | mask(string str[, string upper[, string lower[, string number]]]) | 返回 str 的掩码版本。默认情况下，大写字母转换为“X”，小写字母转换为“x”，数字转换为“n”。 |
| string      | mask_first_n(string str[, int n])                            | 返回前 n 个值被屏蔽的掩码版本。                              |
| string      | mask_last_n(string str[, int n])                             | 返回后 n 个值被屏蔽的掩码版本。                              |
| string      | mask_show_first_n(string str[, int n])                       | 返回前 n 个值未被屏蔽的掩码版本。                            |
| string      | mask_show_last_n(string str[, int n])                        | 返回后 n 个值未被屏蔽的掩码版本。                            |
| string      | mask_hash(string str)                                        | 返回hash掩码                                                 |

## Misc. Functions

| Return Type | Name(Signature)                                     | Description                           |
| :---------- | :-------------------------------------------------- | :------------------------------------ |
| int         | hash(a1[, a2...])                                   | 返回参数的哈希值                      |
| string      | current_user()                                      | 返回当前用户名                        |
| string      | logged_in_user()                                    | 从会话状态返回当前用户名              |
| string      | current_database()                                  | 返回当前数据库名称                    |
| string      | md5(string/binary)                                  | 返回MD5编码                           |
| string      | sha1(string/binary)<br />sha(string/binary)         | 计算SHA-1 摘要                        |
| bigint      | crc32(string/binary)                                |                                       |
| string      | sha2(string/binary, int)                            | 计算SHA-2 系列摘要                    |
| binary      | aes_encrypt(input string/binary, key string/binary) | 使用 AES 加密                         |
| binary      | aes_decrypt(input binary, key string/binary)        | 使用 AES 解密                         |
| string      | version()                                           | 返回 Hive 版本                        |
| bigint      | surrogate_key([write_id_bits, task_id_bits])        | 将数据输入表格时，自动为行生成数字 ID |

## 聚合函数

| Return Type               | Name(Signature)                                          | Description                                                  |
| ------------------------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| BIGINT                    | count(*)                                                 | 统计总行数，包括含有NULL值的行                               |
| BIGINT                    | count([DISTINCT ] expr, ...)                             | 统计提供非NULL的expr表达式值的行数                           |
| DOUBLE                    | sum(col)                                                 | 求和                                                         |
| DOUBLE                    | sum(DISTINCT col)                                        | 去重后求和                                                   |
| DOUBLE                    | avg(col)                                                 | 平均值                                                       |
| DOUBLE                    | avg(DISTINCT col)                                        | 去重后平均值                                                 |
| DOUBLE                    | min(col)                                                 | 最小值                                                       |
| DOUBLE                    | max(col)                                                 | 最大值                                                       |
| DOUBLE                    | variance(col)<br />var_pop(col)                          | 方差                                                         |
| DOUBLE                    | var_samp(col)                                            | 样本方差                                                     |
| DOUBLE                    | stddev_pop(col)                                          | 标准偏差                                                     |
| DOUBLE                    | stddev_samp(col)                                         | 样本标准偏差                                                 |
| DOUBLE                    | covar_pop(col1, col2)                                    | 协方差                                                       |
| DOUBLE                    | covar_samp(col1, col2)                                   | 样本协方差                                                   |
| DOUBLE                    | corr(col1, col2)                                         | 相关系数                                                     |
| DOUBLE                    | percentile(BIGINT col, p)                                | 返回col的p分位数（不适用于浮点类型）                         |
| ARRAY                     | percentile(BIGINT col, array(p1 [, p2]...))              | 与上面相同，接收并返回数组                                   |
| DOUBLE                    | percentile_approx(BIGINT col, p [, B])                   | 返回col的近似p分位数（包括浮点类型），B 参数控制近似精度。较高的值会产生更好的近似值，默认值为 10,000。当 col 中不同值的数量小于 B 时，这会给出精确的百分位值。 |
| ARRAY                     | percentile_approx(DOUBLE col, array(p1 [, p2]...) [, B]) | 与上面相同，接收并返回数组                                   |
| `array<struct {'x','y'}>` | histogram_numeric(col, b)                                | 使用 b 个非均匀间隔的 bin 计算组中数字列的直方图。输出是一个大小为 b 的双值 (x,y) 坐标数组，表示 bin 中心和高度 |
| ARRAY                     | collect_set(col)                                         | 行收集成数组，消除重复元素                                   |
| ARRAY                     | collect_list(col)                                        | 行收集成数组，具有重复项                                     |
| INTEGER                   | ntile(INTEGER x)                                         |                                                              |

```sql
hive> create table as student
	> select stack (12,
    > '001', 'Chinese', 87,
    > '001', 'Math'   , 87,
    > '001', 'English', 92,
    > '002', 'Chinese', 89,
    > '002', 'Math'   , 95,
    > '002', 'English', 93,
    > '003', 'Chinese', 93,
    > '003', 'Math'   , 82,
    > '003', 'English', 87,
    > '004', 'Chinese', 86,
    > '004', 'Math'   , 86,
    > '004', 'English', 100
    ) as (id, course, score);
hive> select id,collect_list(score) as score 
	> from student
	> group by id;
id	score
001	[90,95,80]
002	[70,65,83]
... ...
```

## 表生成函数

| Return Type | Name(Signature)                   | Description                                                  |
| ----------- | --------------------------------- | ------------------------------------------------------------ |
| N rows      | explode(ARRAY a)                  | 将数组a分解为单列，每行对应数组中的每个元素                  |
| N rows      | explode(MAP m)                    | 将映射m分解为两列，每行对应每个key-value对                   |
| N rows      | posexplode(ARRAY a)               | 与explode类似，不同的是还返回一列各元素在数组中的位置        |
| N rows      | stack(INT n, v_1, v_2, ..., v_k)  | 将k列转换为n行，每行有k/n个字段                              |
| tuple       | json_tuple(jsonStr, k1, k2, ...)  | 从一个JSON字符串中获取多个键并作为一个元组返回，与get_json_object不同的是此函数能一次获取多个键值 |
| tuple       | parse_url_tuple(url, p1, p2, ...) | 返回从URL中抽取指定N部分的内容                               |
| N rows      | `inline(ARRAY<STRUCT>)`           | 将结构数组分解为多行，数组中每个结构体一行                   |

```sql
-- stack (values) 创建示例表
create table student as 
select stack (4,
'001', 87, 92, 87,
'002', 89, 93, 95,
'003', 93, 87, 82,
'004', 86, 100, 86
) as (id, Chinese, English, Math);
```

```sql
-- 数组分解成单列，并聚合运算
hive> SELECT a.id, sum(tf.score) FROM 
    > (select id,array(chinese,math,english) as sc from student) a
    > LATERAL VIEW explode(sc) tf AS score GROUP BY a.id; 
id      _c1
001     266
002     277
003     262
004     272

-- 映射分解成两列，重命名时逗号分隔（union语句替代品）
hive> select id,course,score from
    > (select id,map('chinese',chinese,'math',math,'english',english) as dict 
       from student) a
    > LATERAL VIEW explode(dict) tf AS course,score; 
id      course  score
001     chinese 87
001     math    87
001     english 92
... ...
```

```sql
-- inline (array of structs)
select inline(array(
    struct('A',10,date '2015-01-01'),
    struct('B',20,date '2016-02-02')
));
select inline(array(
    struct('A',10,date '2015-01-01'),
    struct('B',20,date '2016-02-02'))) as (col1,col2,col3);
select tf.* 
from (select 0) t 
lateral view inline(array(
    struct('A',10,date '2015-01-01'),
    struct('B',20,date '2016-02-02'))) tf;
select tf.* 
from (select 0) t 
lateral view inline(array(
    struct('A',10,date '2015-01-01'),
    struct('B',20,date '2016-02-02'))) tf as col1,col2,col3;
```

| col1 | col2 | col3       |
| ---- | ---- | ---------- |
| A    | 10   | 2015-01-01 |
| B    | 20   | 2016-02-02 |

# 窗口函数和分析函数

## OVER 子句

OVER 子句支持函数定义查询结果集内的窗口，函数计算窗口中的值。OVER 子句支持标准聚合函数 (SUM, AVG, MAX, MIN, COUNT) 、窗口函数（Window Function）和分析函数（Analytics Function）。

**OVER 子句语法**

```sql
func(args) OVER([PARTITION BY expr, ...] [ORDER BY expr [ASC | DESC], ...] [window_clause])
```

- 当 ORDER BY 指定时缺少 WINDOW 子句，WINDOW 规范默认为 `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`
- 当 ORDER BY 和 WINDOW 子句都缺失时，WINDOW 规范默认为 `ROW BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`
- 窗口函数和分析函数使用 OVER 子句时不支持 WINDOW 子句。

**WINDOW 子句语法**

```sql
window_clause: 
(ROWS | RANGE) BETWEEN frame_bound AND frame_bound 
```

- WINDOW 子句对窗口进行进一步的分区，有两种范围限定方式：一种是使用ROWS子句，通过指定当前行之前或之后的固定数目的行来限制分区中的行数；另一种是RANGE子句，按照排序列的当前值，根据值的范围来确定。
- WINDOW 规范：

| frame_bound         | 说明                         |
| ------------------- | ---------------------------- |
| UNBOUNDED PRECEDING | 第一行                       |
| [num] PRECEDING     | 在当前行前面第num行，默认为1 |
| CURRENT ROW         | 当前行                       |
| [num] FOLLOWING     | 在当前行后面第num行，默认为1 |
| UNBOUNDED FOLLOWING | 最后一行                     |

**示例**

OVER 子句支持标准聚合函数：能够在同一行中返回原始行的同时返回聚合值。等价于原表 GROUP BY 聚合后再 JOIN 原表。以购物表 order_detail 为例

```sql
hive> SELECT * FROM order_detail;
user_id user_type sales
001   new    1
002   new    1
003   new    2
004   new    3
005   new    5
006   new    5
007   new    6
008   old    1
009   old    2
010   old    3
```

**OVER() 子句无参数**：`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`，先累计汇总，然后匹配源数据。

```sql
hive> SELECT *, SUM(sales) OVER() FROM order_detail;
user_id user_type       sales   sum_window_0
010   old     3       29.0
009   old     2       29.0
008   old     1       29.0
007   new     6       29.0
006   new     5       29.0
005   new     5       29.0
004   new     3       29.0
003   new     2       29.0
002   new     1       29.0
001   new     1       29.0
```

**只有PARTITION BY 参数**：`ROW BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`，按照指定列分区汇总

```sql
hive> SELECT *, SUM(sales) OVER(PARTITION BY user_type) FROM order_detail;
user_id user_type       sales   sum_window_0
007  new     6       23.0
006  new     5       23.0
005  new     5       23.0
004  new     3       23.0
003  new     2       23.0
002  new     1       23.0
001  new     1       23.0
010  old     3       6.0
009  old     2       6.0
008  old     1       6.0
```

**只有ORDER BY 参数**：`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`，按照指定列进行排序，然后累计汇总。若第 n 行的值唯一，返回累计到第 n 行的值即可。若第 n 行的值不唯一，累计到所有和第 n 行值相同的行，返回累计值。

```sql
hive> SELECT *, SUM(sales) OVER(ORDER BY sales) FROM order_detail;
user_id user_type       sales   sum_window_0
008    old     1       3.0
002    new     1       3.0
001    new     1       3.0
009    old     2       7.0
003    new     2       7.0
010    old     3       13.0
004    new     3       13.0
006    new     5       23.0
005    new     5       23.0
007    new     6       29.0
```

**PARTITION BY ... ORDER BY**：`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`，按照指定列分区，在分区内排序然后累计汇总。

```sql
hive> SELECT *, SUM(sales) OVER(PARTITION BY user_type ORDER BY sales) FROM order_detail;
user_id user_type       sales   sum_window_0
002  new     1       2.0
001  new     1       2.0
003  new     2       4.0
004  new     3       7.0
006  new     5       17.0
005  new     5       17.0
007  new     6       23.0
008  old     1       1.0
009  old     2       3.0
010  old     3       6.0
```

**WINDOW 子句**

`OVER`一个查询中可以有多个 子句。单个 `OVER`子句仅适用于前一个函数调用。

```sql
hive> SELECT *, 
	> sum(sales) OVER(ORDER BY sales asc),
	> -- 当前行及之前所有行
	> SUM(sales) OVER(ORDER BY sales ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
	> -- 当前行+往前3行
	> SUM(sales) OVER(ORDER BY sales ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
	> FROM order_detail;
user_id user_type       sales   sum_window_0    sum_window_1    sum_window_2
008    old     1       3.0     1.0     1.0
002    new     1       3.0     2.0     2.0
001    new     1       3.0     3.0     3.0
009    old     2       7.0     5.0     5.0
003    new     2       7.0     7.0     6.0
010    old     3       13.0    10.0    8.0
004    new     3       13.0    13.0    10.0
006    new     5       23.0    18.0    13.0
005    new     5       23.0    23.0    16.0
007    new     6       29.0    29.0    19.0
```

```sql
-- 滚动平均值
SELECT sale_date, 
	AVG(sales) OVER(ORDER BY sales ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
FROM order_detail;
```

## 窗口函数

| 偏移取值                     | 说明                                                         |
| :--------------------------- | :----------------------------------------------------------- |
| FIRST_VALUE(expr, skip_null) | 第一个值，skip_null设置是否跳过空值，默认false               |
| LAST_VALUE(expr, skip_null)  | 最后一个值，skip_null设置是否跳过空值，默认false             |
| LEAD(expr, n, default)       | 返回往下第n行值，默认为1。超出当前窗口时返回default，默认为NULL |
| LAG(expr, n, default)        | 返回往上第n行值，默认为1。超出当前窗口时返回default，默认为NULL |

## 分析函数

| 数据排名       | 说明                                                  |
| :------------- | :---------------------------------------------------- |
| ROW_NUMBER()   | 行序号                                                |
| RANK()         | 排名，排名相等则下一个序号间断                        |
| DENSE_RANK()   | 排名，排名相等则下一个序号不间断                      |
| PERCENT_RANK() | 百分比排名，(分组内当前行的RANK值-1)/(分组内总行数-1) |

```sql
hive> SELECT 
    user_id,user_type,sales,
    RANK() over (partition by user_type order by sales desc) as r,
    ROW_NUMBER() over (partition by user_type order by sales desc) as rn,
    DENSE_RANK() over (partition by user_type order by sales desc) as dr
FROM
    order_detail; 
user_id user_type       sales   r       rn      dr
007  new     6       1       1       1
006  new     5       2       2       2
005  new     5       2       3       2
004  new     3       4       4       3
003  new     2       5       5       4
002  new     1       6       6       5
001  new     1       6       7       5
010  old     3       1       1       1
009  old     2       2       2       2
008  old     1       3       3       3
```

| 累计分布    | 说明                              |
| :---------- | :-------------------------------- |
| CUME_DIST() | 小于等于当前值的行数/分组内总行数 |
| NTILE(n)    | 切成n个桶，返回当前切片号         |

```sql
hive> SELECT 
    user_id,user_type,sales,
	NTILE(2) OVER(ORDER BY sales),
	NTILE(3) OVER(ORDER BY sales)
FROM 
    order_detail;
user_id user_type       sales   NTILE_window_0  NTILE_window_1
008   old     1       1       1
002   new     1       1       1
001   new     1       1       1
009   old     2       1       1
003   new     2       1       2
010   old     3       2       2
004   new     3       2       2
006   new     5       2       3
005   new     5       2       3
007   new     6       2       3
```

## Distinct

Hive 2.1.0 及更高版本中，Distinct 支持聚合函数。例子如下：

```sql
SELECT COUNT(DISTINCT a) OVER (PARTITION BY c ORDER BY d ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM T
```

目前添加了对 OVER 子句中引用聚合函数的支持

```sql
SELECT rank() OVER (ORDER BY sum(b)) FROM T GROUP BY a;
```



# 自定义函数

Hive 中有三种 UDF

1. 用户定义函数(user-defined function,UDF)：作用于单个数据行，并且产生一个数据行作为输出。大多数函数都属于这一类（比如数学函数和字符串函数）。 
2. 用户定义聚合函数（user-defined aggregate function ,UDAF ）：接受多个输入数据行，并产生一个输出数据行。例如聚合函数。
3. 用户定义表生成函数（user-defined table-generating function,UDTF ）：作用于单个数据行，并且产生一个表（多个数据行）作为输出。

## 临时函数

**创建临时函数**

```sql
CREATE TEMPORARY FUNCTION function_name AS class_name;
```

此语句允许您创建一个由 class_name 实现的函数 （UDF）。只要会话持续，您就可以在 Hive 查询中使用此函数。

**删除临时函数**

```sql
DROP TEMPORARY FUNCTION [IF EXISTS] function_name;
```

## 永久函数

在 Hive 0.13 或更高版本中，函数可以注册到 Metastore，因此可以在查询中引用它们，而无需在每个会话中创建临时函数。

**创建函数**

```sql
CREATE FUNCTION [db_name.]function_name AS class_name
  [USING JAR|FILE|ARCHIVE 'file_uri' [, JAR|FILE|ARCHIVE 'file_uri'] ];
```

此语句允许您创建一个由 class_name 实现的函数。可以使用 USING 子句指定需要添加到环境中的 jar、文件或档案；当 Hive 会话第一次引用该函数时，这些资源将被添加到环境中，就像已发出 ADD JAR/FILE一样。如果 Hive 未处于本地模式，则资源位置必须是非本地 URI，例如 HDFS 位置。

该函数将被添加到指定的数据库中，或在创建该函数时添加到当前数据库中。可以通过完全限定函数名称 (db_name.function_name) 来引用该函数，如果该函数在当前数据库中，则可以不加限定地引用该函数。

**删除函数**

```sql
DROP FUNCTION [IF EXISTS] function_name;
```

**重新加载**

```sql
RELOAD (FUNCTIONS|FUNCTION);
```

在 HiveServer2 或 HiveCLI 已打开的会话中使用 RELOAD FUNCTIONS 将允许它获取对不同 HiveCLI 会话可能已完成的永久函数的任何更改。

# Lateral View

## Lateral View Syntax

```sql
SELECT select_statement FROM baseTable 
LATERAL VIEW [OUTER] udtf(expression) tableAlias [AS columnAlias, columnAlias, ...]
```

LATERAL VIEW 和表生成函数 (UDTF) 配合使用。LATERAL VIEW 首先将 UDTF 应用于基表的每一行，然后将输出结果 JOIN 基表输入行形成虚拟表，并重名名虚拟表。

**例如**：基表 pageAds

| pageid(STRING) | adid_list(Array\<int\>) |
| :------------- | :---------------------- |
| front_page     | [1, 2, 3]               |
| contact_page   | [3, 4, 5]               |

```sql
SELECT pageid, adid
FROM pageAds LATERAL VIEW explode(adid_list) adTable AS adid;
```

| pageid (string) | adid (int) |
| :-------------- | :--------- |
| "front_page"    | 1          |
| "front_page"    | 2          |
| "front_page"    | 3          |
| "contact_page"  | 3          |
| "contact_page"  | 4          |
| "contact_page"  | 5          |

然后为了计算特定广告出现的次数，可以使用 count/group by：

```sql
SELECT adid, count(1)
FROM pageAds LATERAL VIEW explode(adid_list) adTable AS adid
GROUP BY adid;
```

| adid | count(1) |
| ---- | -------- |
| 1    | 1        |
| 2    | 1        |
| 3    | 2        |
| 4    | 1        |
| 5    | 1        |

## Multiple Lateral Views

一个 FROM 子句可以有多个 LATERAL VIEW 子句。后续的 LATERAL VIEWS 可以引用 LATERAL VIEW 左侧出现的任何表中的列。例如：

```sql
SELECT * FROM exampleTable
LATERAL VIEW explode(col1) myTable1 AS myCol1
LATERAL VIEW explode(myCol1) myTable2 AS myCol2;
```

LATERAL VIEW 子句按它们出现的顺序应用。例如基表：

| Array\<int\> col1 | Array\<string\> col2 |
| ----------------- | -------------------- |
| [1, 2]            | [a", "b", "c"]       |
| [3, 4]            | [d", "e", "f"]       |

```sql
SELECT myCol1, col2 FROM baseTable
LATERAL VIEW explode(col1) myTable1 AS myCol1;
```

| int mycol1 | Array\<string\> col2 |
| ---------- | -------------------- |
| 1          | [a", "b", "c"]       |
| 2          | [a", "b", "c"]       |
| 3          | [d", "e", "f"]       |
| 4          | [d", "e", "f"]       |

添加额外 LATERAL VIEW 的查询：

```sql
SELECT myCol1, myCol2 FROM baseTable
LATERAL VIEW explode(col1) myTable1 AS myCol1
LATERAL VIEW explode(col2) myTable2 AS myCol2;
```

| int myCol1 | string myCol2 |
| ---------- | ------------- |
| 1          | "a"           |
| 1          | "b"           |
| 1          | "c"           |
| ...        | ...           |

## Outer Lateral Views

当使用的 UDTF 不生成任何行时，基表源行则永远不会出现在结果中。例如当`explode`要分解的列为空时很容易发生这种情况 。在这种情况下，用户也可以指定可选关键字 OUTER 来生成行，并且将使用`NULL`填充UDTF 生成的行值。例如：

```sql
SELECT * FROM src LATERAL VIEW OUTER explode(array()) C AS a limit 10;
```

# Group By 增强

## GROUPING SETS 子句

GROUP BY中的GROUPING SETS子句允许我们在同一数据集分组时指定多个分组选项，等效于多次分组后 UNION 。

```sql
SELECT a, b, SUM(c) FROM tab1 GROUP BY a, b GROUPING SETS ( (a,b) ) ;
-- 等效于
SELECT a, b, SUM(c) FROM tab1 GROUP BY a, b ;

SELECT a, b, SUM(c) FROM tab1 GROUP BY a, b GROUPING SETS ( (a,b), a);
-- 等效于
SELECT a, b, SUM(c) FROM tab1 GROUP BY a, b
UNION
SELECT a, null, SUM(c) FROM tab1 GROUP BY a ;

SELECT a,b, SUM(c) FROM tab1 GROUP BY a, b GROUPING SETS (a,b);
-- 等效于
SELECT a, null, SUM(c) FROM tab1 GROUP BY a
UNION
SELECT null, b, SUM(c) FROM tab1 GROUP BY b

SELECT a, b, SUM(c) FROM tab1 GROUP BY a, b GROUPING SETS ( (a, b), a, b, ( ) );
-- 等效于
SELECT a, b, SUM(c) FROM tab1 GROUP BY a, b
UNION
SELECT a, null, SUM(c) FROM tab1 GROUP BY a, null
UNION
SELECT null, b, SUM(c) FROM tab1 GROUP BY null, b
UNION
SELECT null, null, SUM(c) FROM tab1;
```

其中 GROUPING SETS子句中的空白集（）计算总体合计。缺失的分组名用 null 代替。

## GROUPING__ID 伪列

`GROUPING__ID` 表示结果属于哪一个分组集合，第一个集合 `GROUPING__ID` 为0，按数学中的组合顺序依次加1，如果部分分组集合未在统计范围， `GROUPING__ID` 序号仍会保留。

```sql
SELECT a, b, SUM(c), GROUPING__ID FROM tab1 GROUP BY a, b GROUPING SETS ( (a, b), a, b, ( ) ) ;
-- 等效于
SELECT a, b, SUM(c), 0 as GROUPING__ID FROM tab1 GROUP BY a, b
UNION
SELECT a, null, SUM(c), 1 as GROUPING__ID FROM tab1 GROUP BY a, null
UNION
SELECT null, b, SUM(c), 2 as GROUPING__ID FROM tab1 GROUP BY null, b
UNION
SELECT null, null, SUM(c), 3 as GROUPING__ID FROM tab1
```

## GROUPING 函数

如果某一分组列存在null值，该怎样区分null到底是由数据null还是由于 GROUPING SETS 分组生成的null呢。这里 GROUPING 函数来区分，如果是数据本身的值(null或其他值)，GROUPING (col) 将会在这一行返回0，如果这一行不是由分组列自身值聚合，将会返回1。

例如数据集

```sql
SELECT key, value,count(*) FROM tb GROUP BY key, value;
```

| key  | value | count(*) |
| ---- | ----- | -------- |
| 1    | 1     | 5        |
| 1    | NULL  | 2        |
| 2    | 2     | 3        |
| 3    | 3     | 6        |
| 3    | NULL  | 2        |
| 4    | 5     | 3        |

```sql
SELECT key, value, GROUPING__ID, grouping(key), grouping(value), count(*)
FROM tb
GROUP BY key, value WITH ROLLUP;
```

| key  | value | GROUPING__ID | grouping(key) | grouping(value) | count(*) |
| ---- | ----- | ------------ | ------------- | --------------- | -------- |
| 1    | 1     | 0            | 0             | 0               | 5        |
| 1    | NULL  | 0            | 0             | 0               | 2        |
| 2    | 2     | 0            | 0             | 0               | 3        |
| 3    | 3     | 0            | 0             | 0               | 6        |
| 3    | NULL  | 0            | 0             | 0               | 2        |
| 4    | 5     | 0            | 0             | 0               | 3        |
| 1    | NULL  | 1            | 0             | 1               | 7        |
| 2    | NULL  | 1            | 0             | 1               | 3        |
| 3    | NULL  | 1            | 0             | 1               | 8        |
| 4    | NULL  | 1            | 0             | 1               | 3        |
| NULL | NULL  | 3            | 1             | 1               | 21       |

GROUPING 函数使用示例

```sql
select if(grouping(month)=1,'all_months',month) as month,  
    if(grouping(area)=1,'all_area',area) as area, 
    sum(income)  
from people  
group by month,area wirh cube
order by month,area;
```

## CUBE 和 ROLLUP

WITH CUBE/ROLLUP 语法仅在 GROUP BY 语句中使用。

```sql
GROUP BY a, b, c WITH CUBE -- 等效于
GROUP BY a, b, c GROUPING SETS ( (a, b, c), (a, b), (b, c), (a, c), (a), (b), (c), ( ))
GROUP BY a, b, c, WITH ROLLUP -- 等效于
GROUP BY a, b, c GROUPING SETS ( (a, b, c), (a, b), (a), ( ))
```

CUBE 语句返回分组列 a, b, c 所有组合的分组聚合数据。
ROLLUP 语句进行分层聚合，从右向左依次将列值设为 NULL 层级聚合，直到全部分组列都为NULL统计整个表的聚合，对分组列存在包含关系的聚合特别实用（如果使用 CUBE 则会存在若干重复数据）。

```sql
select year,quarter,month,sum(income) 
from people 
group by year,quarter,month with rollup;
```

| year | quarter | month | sum(income) |
| ---- | ------- | ----- | ----------- |
| 2021 | 1       | 1     | 10          |
| 2021 | 1       | 2     | 11          |
| 2021 | 1       | 3     | 10          |
| 2021 | 2       | 4     | 9           |
| 2021 | 1       | NULL  | 31          |
| 2021 | 2       | NULL  | 9           |
| 2021 | NULL    | NULL  | 40          |
| NULL | NULL    | NULL  | 40          |

# CTE

复杂的SQL语句时，可能某个子查询在多个层级多个地方存在重复使用的情况，这个时候我们可以使用 WITH 语句将其独立出来，极大提高SQL可读性。
WITH  语句称为公用表表达式（Common Table Expression, CTE），是一个临时查询表。该查询必须紧接在SELECT或INSERT关键字之前。CTE仅在单个语句的执行范围内定义。一个或多个CTE可以在Hive SELECT，INSERT，CREATE TABLE AS SELECT 或 CREATE VIEW AS SELECT 语句中使用。

```sql
WITH CommonTableExpression 
(, CommonTableExpression)
SELECT select_statement FROM from_statement;
```

- 子查询中不支持WITH子句
- 视图，CTAS和INSERT语句均支持CTE
- 不支持递归查询
- 目前 oracle、sql server、hive等均支持 with as 用法，但 mysql并不支持！

```sql
-- select example
with q1 as (
    select key from q2 where key = '5'
),
q2 as ( 
    select key from src where key = '5'
)
select * from q1;

-- insert example
create table s1 like src;
with q1 as (
    select key, value from src where key = '5'
)
from q1
insert overwrite table s1
select *;
 
-- ctas example
create table s2 as
with q1 as (
    select key from src where key = '4'
)
select * from q1;
```

这里必须注意 WITH 语句和后面的语句是一个整体，中间不能有分号。同级 WITH 关键字只能使用一次，多个子句间用逗号分割；最后一个 with 子句与下面的查询之间不能有逗号，只通过右括号分割。

# 子查询

## FROM 子句

```sql
SELECT ... FROM (subquery) [AS] name 
```

FROM 子句中的子查询，必须为子查询命名，因为 FROM 子句中的每个表都必须有一个名称。

## WHERE 子句

IN 和 NOT IN 子查询：

```sql
SELECT * FROM A WHERE A.a IN (SELECT foo FROM B);
```

EXISTS 和 NOT EXISTS 子查询（相关子查询）：

```sql
SELECT A FROM T1 WHERE EXISTS (SELECT B FROM T2 WHERE T1.X = T2.Y);
```

限制：

- 这些子查询仅在表达式的右侧受支持。
- IN/NOT IN 子查询只能选择一列。
- EXISTS/NOT EXISTS 必须有一个或多个相关连接。
- 仅在WHERE 子句中的子查询支持对父查询的引用。

# 抽样

## 抽样分桶表

TABLESAMPLE 子句允许用户为数据样本而不是整个表编写查询。语法如下：

```sql
table_sample: TABLESAMPLE (BUCKET x OUT OF y [ON colname])
```

TABLESAMPLE 子句可以添加到 FROM 子句中的任何表后。表的行在 colname 上分桶到编号为 1 到 y 的 y 个桶中，返回属于桶 x 的行。colname表示进行采样的列，可以是表中的非分区列之一，也可以是 rand() 表示对整行进行随机采样。

在以下示例中，返回表 source 的 32 个桶中的第 3 个桶。s 是表别名。

```sql
SELECT * FROM source TABLESAMPLE(BUCKET 3 OUT OF 32 ON rand()) s;
```

通常，抽样是在 CREATE TABLE 语句的 CLUSTERED BY 子句中指定的列上完成的。如果表 'source' 是用 'CLUSTERED BY id INTO 32 BUCKETS' 创建的。在以下示例中，我们从 source 表的 32 个存储桶中选择第三个存储桶：

```sql
SELECT * FROM source TABLESAMPLE(BUCKET 3 OUT OF 32 ON id);
```

y 必须是表创建时指定的表中存储桶数的倍数或除数。

```
TABLESAMPLE(BUCKET 3` `OUT OF ``16` `ON id)
```

将挑选出第 3 个和第 19 个集群，因为每个桶将由 (32/16)=2 个集群组成。

另一方面， TABLESAMPLE 子句

```
TABLESAMPLE(BUCKET ``3` `OUT OF ``64` `ON id)
```

将挑选出第三个集群的一半，因为每个桶将由 (32/64)=1/2 的集群组成。

## 块抽样

```
block_sample: TABLESAMPLE (n PERCENT)
```

这将允许 Hive 选择至少 n% 的数据大小（注意它不一定意味着行数）作为输入。仅支持CombineHiveInputFormat，不处理一些特殊的压缩格式。如果采样失败，MapReduce 作业的输入将是整个表/分区。抽样的粒度为HDFS 块级别大小。例如，如果块大小为 256MB，即使输入大小的 n% 仅为 100MB，您也会获得 256MB 的数据。

在以下示例中，查询将使用 0.1% 或更大的输入大小。

```sql
SELECT * FROM source TABLESAMPLE(0.1 PERCENT) s;
```

有时你想用不同的块对相同的数据进行采样，你可以改变这个种子数：

```
set hive.sample.seednumber=<INTEGER>;
```

或者用户可以指定要读取的总长度，但它与 PERCENT 采样有相同的限制。

```sql
block_sample: TABLESAMPLE (ByteLengthLiteral)

ByteLengthLiteral : (Digit)+ ('b' | 'B' | 'k' | 'K' | 'm' | 'M' | 'g' | 'G')
```

在以下示例中，查询将使用 100M 或更大的输入大小。

```sql
SELECT * FROM source TABLESAMPLE(100M) s;
```

Hive 还支持按行数限制输入，但它的作用与以上两种不同。首先，它不需要CombineHiveInputFormat，这意味着它可以与非本地表一起使用。其次，用户给出的行数应用于每个拆分。因此总行数可能因输入拆分的数量而异。

```sql
block_sample: TABLESAMPLE (n ROWS)
```

例如，以下查询将从每个输入拆分中获取前 10 行。

```sql
SELECT * FROM source TABLESAMPLE(10 ROWS);
```

# TRANSFORM

## Transform/Map-Reduce Syntax

Hive 提供了在SQL中调用自定义 Map/Reduce 脚本的功能，适合实现Hive中没有的功能又不想写UDF的情况。

```sql
FROM (
  FROM src
  MAP col_expr1, col_expr2, ...
  [inRowFormat]
  USING 'my_map_script'
  [ AS colname1, colname2, ...]
  [outRowFormat] [RECORDREADER className]
  [CLUSTER BY col_list] | [DISTRIBUTE BY col_list] [SORT BY col_list]
   src_alias
)
REDUCE col_expr1, col_expr2, ...
  [inRowFormat]
  USING 'my_reduce_script'
  [ AS colname1, colname2, ...]
  [outRowFormat] [RECORDREADER className]
; 
```

- inRowFormat/outRowFormat：rowFormat
- 请注意，在提供给用户脚本之前，列将被转换为字符串并由 TAB 分隔，并且用户脚本的标准输出将被视为以 TAB 分隔的字符串列。类似地，所有 NULL 值都将转换为字符串 "\N"。这些默认值可以被 ROW FORMAT rowFormat 覆盖。

- rowFormat：

```sql
DELIMITED [FIELDS TERMINATED BY char]
          [COLLECTION ITEMS TERMINATED BY char]
          [MAP KEYS TERMINATED BY char]
          [ESCAPED BY char]
          [LINES SEPARATED BY char]
-- OR
SERDE serde_name [WITH SERDEPROPERTIES
                       property_name=property_value,
                       property_name=property_value, ...]
```

当然，MAP 和 REDUCE 都是更通用的TRANSFORM的语法糖

```sql
FROM (
  FROM src
  SELECT TRANSFORM (col_expr1, col_expr2, ...)
  [inRowFormat]
  USING 'my_map_script'
  [ AS colname1, colname2, ...]
  [outRowFormat] [RECORDREADER className]
  [CLUSTER BY col_list] | [DISTRIBUTE BY col_list] [SORT BY col_list]
   src_alias
)
SELECT TRANSFORM (col_expr1, col_expr2, ...)
  [inRowFormat]
  USING 'my_reduce_script'
  [ AS colname1, colname2, ...]
  [outRowFormat] [RECORDREADER className]
;
```

**示例#1**

```sql
FROM (
     FROM pv_users
     MAP pv_users.userid, pv_users.date
     USING 'map_script'
     AS dt, uid
     CLUSTER BY dt) map_output
INSERT OVERWRITE TABLE pv_users_reduced
     REDUCE map_output.dt, map_output.uid
     USING 'reduce_script'
     AS date, count;
```

示例map脚本 (weekday_mapper.py)

```python
import sys
import datetime
 
for line in sys.stdin:
  line = line.strip()
  userid, unixtime = line.split('\t')
  weekday = datetime.datetime.fromtimestamp(float(unixtime)).isoweekday()
  print ','.join([userid, str(weekday)])
```

当然，内部查询也可以这样写：

```sql
FROM (
  FROM pv_users
  SELECT TRANSFORM(pv_users.userid, pv_users.date)
  USING 'map_script'
  AS dt, uid
  CLUSTER BY dt) map_output
INSERT OVERWRITE TABLE pv_users_reduced
  SELECT TRANSFORM(map_output.dt, map_output.uid)
  USING 'reduce_script'
  AS date, count;
```

**示例#2**

```sql
FROM (
  FROM src
  SELECT TRANSFORM(src.key, src.value) 
  ROW FORMAT SERDE 'org.apache.hadoop.hive.contrib.serde2.TypedBytesSerDe'
  USING '/bin/cat'
  AS (tkey, tvalue) 
  ROW FORMAT SERDE 'org.apache.hadoop.hive.contrib.serde2.TypedBytesSerDe'
  RECORDREADER 'org.apache.hadoop.hive.contrib.util.typedbytes.TypedBytesRecordReader'
) tmap
INSERT OVERWRITE TABLE dest1 SELECT tkey, tvalue
```

## Schema-less map/reduce

如果 USING map_script 之后没有 AS 子句，Hive 假设脚本的输出包含两部分：key 位于第一个 tab 之前，value 位于第一个 tab 之后的其余部分。 请注意，这与指定 AS key, value 不同，因为在这种情况下，如果有多个 tab，value 仅包含第一个tab和第二个tab之间的部分。

注意，我们可以直接执行CLUSTER BY 键，而无需指定脚本的输出模式。

```sql
FROM (
    FROM pv_users
    MAP pv_users.userid, pv_users.date
    USING 'map_script'
    CLUSTER BY key) map_output
 
INSERT OVERWRITE TABLE pv_users_reduced
    REDUCE map_output.dt, map_output.uid
    USING 'reduce_script'
    AS date, count;
```

用户可以指定 Distribute By 和Sort By，而不是指定 cluster by ，所以分区列和排序列可以不同，排序列可选。

```sql
FROM (
    FROM pv_users
    MAP pv_users.userid, pv_users.date
    USING 'map_script'
    AS c1, c2, c3
    DISTRIBUTE BY c2
    SORT BY c2, c1) map_output
 
INSERT OVERWRITE TABLE pv_users_reduced
    REDUCE map_output.c1, map_output.c2, map_output.c3
    USING 'reduce_script'
    AS date, count;
```

## Typing the output of TRANSFORM

默认情况下，脚本的输出字段为字符串，例如

```sql
SELECT TRANSFORM(stuff) USING 'script' AS thing1, thing2
```

它们可以使用以下语法进行转换

```sql
SELECT TRANSFORM(stuff) USING 'script' AS (thing1 INT, thing2 INT)
```

# EXPLAIN

Hive 提供了一个`EXPLAIN`命令显示查询执行计划。该语句的语法如下：

```sql
EXPLAIN [EXTENDED|CBO|AST|DEPENDENCY|AUTHORIZATION|LOCKS|VECTORIZATION|ANALYZE] query
```

- 采用EXTENDED的EXPLAIN语句生成有关执行计划的额外信息。这通常是文件名等物理信息。

- 一个 Hive 查询被转换成一个stages序列（它更像是一个有向无环图，DAG）。这些stages可能是 map/reduce 阶段，或者它们甚至可能是执行元存储或文件系统操作（如移动和重命名）的阶段。解释输出分为三部分：

  - 查询的抽象语法树

  - 计划不同stages之间的依赖关系

  - 每个stage的描述

  stages本身的描述显示了一系列运算符以及与运算符关联的元数据。元数据可能包括诸如 FilterOperator 的过滤器表达式或 SelectOperator 的选择表达式或 FileSinkOperator 的输出文件名之类的内容。

- CBO 子句输出 Calcite 优化器生成的计划。从 Hive 版本 4.0.0开始。
  ```sql
  EXPLAIN [FORMATTED] CBO [COST|JOINCOST]
  ```
  - COST 选项打印使用 Calcite 默认成本模型计算的计划和成本。

  - JOINCOST 选项打印使用用于连接重新排序的成本模型计算的计划和成本。

- AST子句输出查询的抽象语法树

- DEPENDENCY子句显示了输入的各种属性，包含表和分区。如果通过视图访问表，则依赖项会显示父项。

- AUTHORIZATION子句显示所有的实体需要被授权执行（如果存在）的查询和授权失败。

- LOCKS子句了解系统将获得哪些锁来运行指定的查询。从 Hive 版本 3.2.0开始。

- VECTORIZATION子句向 EXPLAIN 输出添加详细信息，显示为什么 Map 和 Reduce 工作未矢量化。自 Hive 版本 2.3.0开始。

  ```sql
  EXPLAIN VECTORIZATION [ONLY] [SUMMARY|OPERATOR|EXPRESSION|DETAIL]
  ```

  - ONLY选项会抑制大多数非矢量化元素。
  - SUMMARY（默认）显示 PLAN 的矢量化信息（启用矢量化）以及 Map 和 Reduce 工作的摘要。
  - OPERATOR显示运算符的矢量化信息。例如Filter矢量化。包括SUMMARY的所有信息。
  - EXPRESSION显示表达式的矢量化信息。例如谓词表达式。包括SUMMARY和OPERATOR的所有信息。
  - DETAIL显示详细级别的矢量化信息。它包括SUMMARY、OPERATOR和EXPRESSION的所有信息。
  - 默认值是SUMMARY，但不带ONLY选项。

- ANALYZE子句使用实际行数注释计划，格式为：（估计行数）/（实际行数）
- User-level Explain Output


**示例**

```sql
EXPLAIN
FROM src INSERT OVERWRITE TABLE dest_g1 SELECT src.key, sum(substr(src.value,4)) GROUP BY src.key;
```

- 依赖图

  ```sql
  STAGE DEPENDENCIES:
    Stage-1 is a root stage
    Stage-2 depends on stages: Stage-1
    Stage-0 depends on stages: Stage-2
  ```

  这表明Stage-1是根阶段，Stage-1完成后执行Stage-2，Stage-2完成后执行Stage-0。

- 每个阶段的计划

# 配置属性

管理员可以使用 Hive 中的许多配置变量来更改其安装和用户会话的行为。这些变量可以通过以下任何一种方式配置，按优先顺序显示：

- 在CLI 或 Beeline 中 使用**set 命令**配置属性。例如，以下命令为所有后续语句设置临时目录（Hive 用于存储临时输出和计划） ：`/tmp/mydir`

  ```sql
  set hive.exec.scratchdir=/tmp/mydir;
  ```

- 在`hive`（在 CLI 中）或 `beeline`的命令选项 `--hiveconf`中配置。例如：

  ```sql
    bin/hive --hiveconf hive.exec.scratchdir=/tmp/mydir
  ```

- 在**`hive-site.xml`**，这用于设置永久属性。例如：

  ```sql
    <property>
      <name>hive.exec.scratchdir</name>
      <value>/tmp/mydir</value>
      <description>Scratch space for Hive jobs</description>
    </property>
  ```

| 属性                                     | 说明                                                         | 选项                                                       |
| ---------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| hive.support.quoted.identifiers          | none模式下，反引号内容被解释为正则表达式                     | column (默认) / none                                       |
| hive.execution.engine                    | 选择执行引擎                                                 | mr（Map Reduce，默认）/tez/spark                           |
| mapred.reduce.tasks                      | 每个作业的默认reduce 任务数                                  | 通常设置为接近可用主机的数量。默认为 -1，Hive 将自动计算。 |
| hive.cli.print.header                    | 是否在命令行界面打印列名                                     | false (默认) / true                                        |
| hive.cli.print.current.db                | 是否显示数据库名称                                           | false (默认) / true                                        |
| hive.resultset.use.unique.column.names   | 是否显示表名                                                 | false / true (默认)                                        |
| hive.exec.mode.local.auto                | 是否自动在本地模式下运行                                     | false (默认) / true                                        |
| hive.optimize.index.filter               | 是否启用索引的自动使用                                       | false (默认) / true                                        |
| hive.mapred.mode                         | 正在执行 Hive 操作的模式，在`strict`模式下，禁止全表扫描     | strict (默认) / nonstrict                                  |
| hive.script.auto.progress                | 是否应自动向 TaskTracker 发送进度信息，以避免任务因不活动而被终止 | false (默认) / true                                        |
| hive.exec.parallel                       | 是否并行执行作业                                             | false (默认) / true                                        |
| hive.exec.parallel.thread.number         | 最多可以并行执行多少个作业                                   | 8 (默认)                                                   |
| hive.merge.mapredfiles                   | 在 map-reduce 作业结束时合并小文件                           | false (默认) / true                                        |
| hive.merge.size.per.task                 | 作业结束时合并文件的大小                                     | 256000000 (默认)                                           |
| hive.exec.dynamic.partition              | 是否允许DML/DDL 中的动态分区                                 | false (默认) / true                                        |
| hive.exec.dynamic.partition.mode         | 在strict模式下，用户必须至少指定一个静态分区                 | strict (默认) / nonstrict                                  |
| hive.exec.max.dynamic.partitions         | 最大动态分区数                                               | 1000 (默认)                                                |
| hive.exec.max.dynamic.partitions.pernode | 允许在每个 mapper/reducer 节点中创建的最大动态分区数         | 100 (默认)                                                 |
| hive.error.on.empty.partition            | 动态分区插入产生空结果时是否抛出异常                         | false (默认) / true                                        |
| hive.output.file.extension               | 输出文件的扩展名                                             | 默认为空                                                   |
| hive.ddl.output.format                   | 用于 DDL 输出的数据格式 (e.g. `DESCRIBE table`)              | text (默认) / json                                         |
| hive.exec.script.wrapper                 | 设置调用脚本的运算符。如果将其设置为 python，则为 `python <script command>`。如果该值为 null 或未设置，则为调用`<script command>`。 | null (默认)                                                |
| hive.exec.local.scratchdir               | 当 Hive 在本地模式下运行时，此目录用于临时文件               | `/tmp/<user.name>`                                         |
| hive.variable.substitute                 | 替换之前使用set命令、系统变量或环境变量设置的 Hive 语句中的变量。 | false / true (默认)                                        |

# Shell 变量传递

- shell 中已有的变量，hive -e 命令可直接使用，例如

  ```bash
  #!/bin/bash
  tablename="student"
  limitcount="8"
  
  hive -e "select * from ${tablename} limit ${limitcount};"
  ```

- 在Hive选项 `--hiveconf` 中配置变量，在SQL脚本中使用

  ```shell
  hive --hiveconf tablename='student' --hiveconf limitcount=8 -f test.sql
  ```

  ```sql
  -- test.sql
  select * from ${hiveconf:tablename} limit ${hiveconf:limitcount};
  ```

  注意，`--hiveconf` 不能在 hive -e 中使用

- 配置Hive属性 `hivevar`

  ```sql
  hive > set hivevar: tablename="student";
  hive > set hivevar: limitcount="8";
  hive > set ${hivevar: tablename};  -- 查看变量
  tablename="student"
  hive > select * from ${tablename} limit ${limitcount}; --不使用前缀也可以引用使用
  ```

  也可以通过Hive属性`hiveconf` 在SQL脚本中以类似的方式设置，引用时必须加上`hiveconf`前缀

**示例**

```bash
# 数据备份
empl=(`hive -e "show partitions employee;"`)
emp2=(`hive -e "show partitions employee_backup;"`)
for((i=1;i<${#empl[@]};i++))
do
  update=${#empl[@]}
  tmp=(`echo ${emp2[@]} | grep ${update}`)

  if [ -z ${tmp} ]
  then
    hive "
    set hive.support.quoted.identifiers = none; 
    insert overwrite table tmp2 partition(${update})
    select `(y|m|d)?+.+` from  employee where ${update}
    ;" 
    printf "%-20s\n" ${update} > info 
  fi 
done 
```



