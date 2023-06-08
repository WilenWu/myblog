---
title: 大数据手册(Hive)--HiveQL(Introduction)
categories:
  - Big Data
  - Hive
tags:
  - 大数据
  - hive
  - SQL
cover: /img/hive-cover.jpg
top_img: /img/apache-hive-bg.png
description: Hive 命令、数据类型
abbrlink: e6bd78c5
date: 2018-07-03 17:57:36
---

# 前言

Apache Hive 是可实现大规模分析的分布式容错数据仓库系统。该数据仓库集中存储信息，您可以轻松对此类信息进行分析，从而做出明智的数据驱动决策。Hive 让用户可以利用 SQL 读取、写入和管理 PB 级数据。

Hive 建立在 Apache Hadoop 基础之上，后者是一种开源框架，可被用于高效存储与处理大型数据集。因此，Hive 与 Hadoop 紧密集成，其设计可快速对 PB 级数据进行操作。Hive 的与众不同之处在于它可以利用 Apache Tez 或 MapReduce 通过类似于 SQL 的界面查询大型数据集。

SQL语言包含6个部分：

1. 数据查询语言(Data Query Language, DQL）：用以从表中获得数据，包括 SELECT，WHERE，ORDER BY，GROUP BY和HAVING等。
2. 数据操作语言(Data Manipulation Language, DML）：其语句包括动词 INSERT、UPDATE、DELETE。它们分别用于添加、修改和删除。
3. 事务控制语言(Transaction Control Language, TCL)：它的语句能确保被DML语句影响的表的所有行及时得以更新。包括COMMIT（提交）命令、SAVEPOINT（保存点）命令、ROLLBACK（回滚）命令。
4. 数据控制语言(Data Control Language, DCL)：它的语句通过GRANT或REVOKE实现权限控制，确定单个用户和用户组对数据库对象数据库对象)的访问。
5. 数据定义语言(Data Definition Language, DDL)：其语句包括动词CREATE、ALTER和DROP。
6. 指针控制语言(CCL)：它的语句，像DECLARE CURSOR，FETCH INTO和UPDATE WHERE CURRENT用于对一个或多个表单独行的操作。


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

## Hive变量传递

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

## 内置运算符

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

