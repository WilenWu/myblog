---
title: 大数据手册(Hive)--HiveQL(DQL)
categories:
  - 'Big Data'
  - Hive
tags:
  - 大数据
  - hive
  - SQL
cover: /img/apache-hive-dql.png
top_img: /img/apache-hive-bg.png
description: 数据查询语言(Data Query Language, DQL）：用以从表中获得数据，包括 SELECT，WHERE，ORDER BY，GROUP BY和HAVING等。
abbrlink: 297bd708
date: 2018-07-03 17:57:36
---

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

## Order

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



<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/hive_join.png" alt="join" width="80%" />

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

## Distinct

Hive 2.1.0 及更高版本中，Distinct 支持聚合函数。例子如下：

```sql
SELECT COUNT(DISTINCT a) OVER (PARTITION BY c ORDER BY d ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM T
```

目前添加了对 OVER 子句中引用聚合函数的支持

```sql
SELECT rank() OVER (ORDER BY sum(b)) FROM T GROUP BY a;
```

# 增强聚合

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

`GROUPING__ID` 表示结果属于哪一个分组组合。`GROUP BY`语句中每个位置对应一个值，例如 `GROUP BY a, b, c` 中 `c` 对应 2^0^， `c` 对应 2^1^， `c` 对应 2^2^，依次类推。去除缺失分组后，`GROUPING__ID` 的值由剩余的值加和得到。例如，缺失的分组为 `b` ，则`GROUPING__ID` 等于 2^0^+2^2^=5。

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

# Transform

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

