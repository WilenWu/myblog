---
title: 大数据手册(Hive)--HiveQL(Introduction)
categories:
  - Big Data
  - Hive
tags:
  - 大数据
  - hive
  - SQL
cover: /img/hive-sql.jpg
top_img: /img/apache-hive-bg.png
description: Hive 命令、数据类型和常用函数
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

