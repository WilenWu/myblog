---
title: 大数据手册(Oracle)--Oracle SQL
date: ‎2021‎-05‎-03‎ ‏‎16:18:01
categories: [大数据]
tags: [大数据,oracle,SQL]
cover: /img/sql-vector-image.jpg
top_img: /img/SQL.png
---

结构化查询语言(Structured Query Language)简称SQL，是一种数据库查询和程序设计语言，用于存取数据以及查询、更新和管理关系数据库系统（RDBMS）。

主流关系型数据库，比如Oracle, MS SQL Server 以及 MySQL，其数据库语言都是基于 SQL-92 标准开发的。

<!-- more -->

 SQL 语言包括：

- 数据定义语言 (Data Definition Language, DDL)：定义数据库对象
  - *CREATE* - 创建对象
  - *ALTER* - 修改对象
  - *DROP* - 删除对象
- 数据查询语言 (Data Retrieval Language, DRL)
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

# Oracle 事务

事务在数据库中是工作的逻辑单元，单个事务是由一个或多个完成一组的相关行为的SQL语句组成，通过事务机制，可以确保这一组SQL语句所作的操作要么都成功执行，完成整个工作单元操作，要么一个也不执行。
一组SQL语句操作要成为事务，数据库管理系统必须保证这组操作的原子性（Atomicity）、一致性（consistency）、隔离性（Isolation）和持久性（Durability），这就是ACID特性。

## 提交事务

COMMIT 语句可以用来提交当前事务的所有更改。提交后，其他用户将能够看到您的更改。

```sql
COMMIT [ WORK ] [ COMMENT clause ] [ WRITE clause ] [ FORCE clause ];
```

**参数**：

- WORK：可选的。它被 Oracle 添加为符合 SQL 标准。使用或不使用 WORK 参数来执行 COMMIT 将产生相同的结果。
- COMMENT clause：可选的。 它用于指定与当前事务关联的注释。 该注释最多可以包含在单引号中的 255 个字节的文本中。 如果出现问题，它将与事务ID一起存储在名为 DBA_2PC_PENDING 的系统视图中。
- WRITE clause：可选的。 它用于指定将已提交事务的重做信息写入重做日志的优先级。 用这个子句，有两个参数可以指定：
  - WAIT 或 NOWAIT (如果省略，WAIT是默认值)
  - IMMEDIATE 或 BATCH(IMMEDIATE是省略时的默认值)
- FORCE clause：可选的。 它用于强制提交可能已损坏或有疑问的事务。 有了这个子句，可以用3种方式指定。

## 回滚事务

ROLLBACK 语句可以用来撤销当前事务或有问题的事务。

```sql
ROLLBACK [ WORK ] [ TO [SAVEPOINT] savepoint_name  | FORCE 'string' ];
```

**参数**

- `WORK`：可选的。 它被 Oracle 添加为符合 SQL 标准。 使用或不使用 WORK 参数来发出 ROLLBACK 会导致相同的结果。
- `TO SAVEPOINT` `savepoint_name`：可选的。 ROLLBACK语句撤消当前会话的所有更改，直到由 savepoint_name 指定的保存点。 如果省略该子句，则所有更改都将被撤消。
- `FORCE` `‘string’`：可选的。它用于强制回滚可能已损坏或有问题的事务。 使用此子句，可以将单引号中的事务ID指定为字符串。 可以在系统视图中找到名为 DBA_2PC_PENDING 的事务标识。
- 必须拥有 DBA 权限才能访问系统视图：DBA_2PC_PENDING 和 V$CORRUPT_XID_LIST。
- 您无法将有问题的事务回滚到保存点。

**Savepoint 示例**

```sql
ROLLBACK TO SAVEPOINT savepoint1;
```

## 设置事务

SET TRANSACTION 语句可以用来设置事务的各种状态，比如只读、读/写、隔离级别，为事务分配名称或将事务分配回滚段等等。

```sql
SET TRANSACTION [ READ ONLY | READ WRITE ]
                [ ISOLATION LEVEL SERIALIZE | READ COMMITED ]
                [ USE ROLLBACK SEGMENT 'segment_name' ]
                [ NAME 'transaction_name' ];
```

**参数**

- READ ONLY：可以将事务设置为只读事务。
- READ WRITE：可以将事务设置为读/写事务。
- ISOLATION LEVEL： 如果指定，它有两个选项：
- SERIALIZE：如果事务尝试更新由另一个事务更新并未提交的资源，则事务将失败。
- READ COMMITTED：如果事务需要另一个事务持有的行锁，则事务将等待，直到行锁被释放。
- USE ROLLBACK SEGMENT：可选的。 如果指定，它将事务分配给由 'segment_name' 标识的回退段，该段是用引号括起来的段名称。
- NAME：为 'transaction_name' 标识的事务分配一个名称，该事务用引号括起来。

## 锁表

LOCK TABLE 语句可以用来锁定表、表分区或表子分区。

```sql
LOCK TABLE tables IN lock_mode MODE [ WAIT [, integer] | NOWAIT ];
```

**参数**

- tables：用逗号分隔的表格列表。
- lock_mode：它是以下值之一：

| lock_mode           | 描述                                                         |
| :------------------ | :----------------------------------------------------------- |
| ROW SHARE           | 允许同时访问表，但阻止用户锁定整个表以进行独占访问。         |
| ROW EXCLUSIVE       | 允许对表进行并发访问，但阻止用户以独占访问方式锁定整个表并以共享方式锁定表。 |
| SHARE UPDATE        | 允许同时访问表，但阻止用户锁定整个表以进行独占访问。         |
| SHARE               | 允许并发查询，但用户无法更新锁定的表。                       |
| SHARE ROW EXCLUSIVE | 用户可以查看表中的记录，但是无法更新表或锁定`SHARE`表中的表。 |
| EXCLUSIVE           | 允许查询锁定的表格，但不能进行其他活动。                     |

- WAIT：它指定数据库将等待(达到指定整数的特定秒数)以获取 DML 锁定。
- NOWAIT：它指定数据库不应该等待释放锁。


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

# 数据库操作

```sql
SHOW DATABASES;            --查看数据库
CREATE DATABASE db_name;   --创建数据库
DROP DATABASE db_name;     --删除
USE db_name;               --连接数据库
SHOW TABLES [IN db_name];  --查看数据表
DESC tbl_name;          --查看表结构
```

# 管理表结构

## 创建表

```sql
-- 新建表结构
CREATE TABLE tbl_name(
col_name1 datatype COMMENT clause,
col_name2 datatype COMMENT clause,
... ...
);

-- 复制表结构
SELECT *
INTO newtable FROM table1
WHERE 1=0;   -- 限制成空表

DESC tbl_name; --查看表结构
```

## 修改表结构

ALTER TABLE 语句用于在已有的表中添加、修改或删除列。

```sql
--添加列
ALTER TABLE table_name
ADD column_name datatype;                 

--删除列
ALTER TABLE table_name 
DROP COLUMN column_name;

--修改列定义
ALTER TABLE table_name
MODIFY column_name datatype; 

--重命名列名
ALTER TABLE table_name
RENAME COLUMN column_name TO new_name;

--修改表名
RENAME table_name TO new_name;
```

从大表中删除列的过程可能耗费时间和资源。 因此，也会使用逻辑删除列。

```sql
ALTER TABLE table_name 
SET UNUSED COLUMN column_name;  -- 逻辑删除列

ALTER TABLE table_name
DROP UNUSED COLUMNS;  -- 物理删除隐藏的列
```

## 删除表

```sql
TRUNCATE TABLE table_name [CASCADE];  -- 删除表数据[并解除关联表约束]，又称截断表
DROP TABLE table_name [CASCADE CONSTRAINTS]; --删除表[并解除关联表约束]
```
>  TRUNCATE TABLE 仅仅需要除去表内的数据，但并不删除表本身 

# 操作表数据

## 插入数据

INSERT 语句向表中添加记录

```sql
INSERT INTO tbl_name VALUES (value1, value2, ...);
INSERT INTO tbl_name(col1, col2, ...) VALUES (value1, value2, ...);
```

## 复制表数据

```sql
-- 建表时复制表数据
CREATE TABLE new_table
AS Subquery;

-- 添加时复制数据
INSERT INTO tbl_name
SELECT Subquery;

INSERT INTO tbl_name(col1, col2, ...)
SELECT Subquery;
```

```sql
-- 示例
INSERT INTO Websites(name, country)
SELECT app_name, country FROM apps;
```

**SELECT INTO 语句**：从一个表中选取数据，然后把数据插入另一个表中。常用于创建表的备份复件或者用于对记录进行存档

```sql
SELECT col1, col2, ...
INTO　new_table_name [IN externaldatabase] 
FROM old_table_name

-- 示例
SELECT LastName,Firstname
INTO Persons_backup IN 'Backup.mdb'
FROM Persons
WHERE City='Beijing'
```

## 批量插入数据

1. **无条件的插入数据**

   ```sql
   INSERT ALL
   	INTO tbl_name1(...) VALUES (...)
   	INTO tbl_name2(...) VALUES (...)
   	... ...
   Subquery;
   ```

   ```sql
   -- 示例
   INSERT ALL
         INTO sales_1 (prod_id, cust_id, amount)
         VALUES (product_id, customer_id, amount)
   	  INTO sales_2 (product_id, customer_id, amount)
         VALUES (product_id, customer_id, amount)
   SELECT product_id, customer_id, amount
   FROM sales_detail;
   ```

2. **有条件的插入数据**

   ```sql
   INSERT ALL|FIRST
       WHEN condition1 THEN
           INTO tbl_name1(column_list) VALUES (value_list)
       WHEN condition2 THEN 
           INTO tbl_name2(column_list) VALUES (value_list)
       ... ...
       ELSE
           INTO tbl_name3(column_list) VALUES (value_list)
   Subquery;
   ```

   使用 ALL 关键字，对于子查询的每一行，会遍历所有 WHEN 子句，只要满足条件，就会插入数据。
   使用 ALL 只要有一个满足条件，后面的条件不再判断，不会造成重复插入。

> mysql 导入本地数据
>
> ```sql
> LOAD DATA LOCAL INFILE 'dump.txt' --utf-8 txt文件
> INTO TABLE mytbl                  --已创建的表
> FIELDS TERMINATED BY ','          --分隔符
> LINES TERMINATED BY '\r\n';       --换行符
> ```

## 修改表数据

Update 语句用于修改表中的数据。

```sql
UPDATE tbl_name
SET column1 = value1,
    column2 = value2,
    ... ...
[WHERE conditions];
```

## 删除表数据

DELETE 语句用于删除表中的行。

```sql
DELETE FROM tbl_name [WHERE conditions];
TRUNCATE TABLE table_name [CASCADE];  -- 删除表数据[并解除关联表约束]，又称截断表
```

> MySQL还支持多表连接删除，例如
>
> ```sql
> DELETE offices, employees  
> FROM offices INNER JOIN employees        
> ON employees.officeCode = employees.officeCode 
> WHERE offices.officeCode = 5;
> 
> DELETE customers 
> FROM customers LEFT JOIN orders 
> ON customers.customerNumber = orders.customerNumber 
> WHERE orderNumber IS NULL;
> ```


## MERGE 语句

MERGE - 使用单个语句逐步完成插入，更新和删除操作。

```sql
MERGE INTO target_table 
USING source_table 
ON search_condition
    WHEN MATCHED THEN
        UPDATE SET col1 = value1, col2 = value2,...
        WHERE <update_condition>
        [DELETE WHERE <delete_condition>]
    WHEN NOT MATCHED THEN
        INSERT (col1,col2,...)
        values(value1,value2,...)
        WHERE <insert_condition>;
```

示例

```sql
MERGE INTO member_staging x
USING (SELECT member_id, first_name, last_name, rank FROM members) y
ON (x.member_id  = y.member_id)
WHEN MATCHED THEN
    UPDATE SET x.first_name = y.first_name, 
                        x.last_name = y.last_name, 
                        x.rank = y.rank
    WHERE x.first_name <> y.first_name OR 
           x.last_name <> y.last_name OR 
           x.rank <> y.rank 
WHEN NOT MATCHED THEN
    INSERT(x.member_id, x.first_name, x.last_name, x.rank)  
    VALUES(y.member_id, y.first_name, y.last_name, y.rank);
```

# 约束（Constraints）

SQL 约束用于规定表中的数据规则。如果存在违反约束的数据行为，行为会被约束终止。
可以在创建表时规定约束（通过 CREATE TABLE 语句），或者在表创建之后也可以（通过 ALTER TABLE 语句）。我们将主要探讨以下几种约束：

- NOT NULL：强制列不接受 NULL 值
- UNIQUE：保证某列的每行必须有唯一的值
- PRIMARY KEY：主键，非空且唯一
- FOREIGN KEY：外键，指向另一个表中的 PRIMARY KEY(唯一约束的键)
- CHECK：保证列中的值符合指定的条件
- DEFAULT：规定没有给列赋值时的默认值

## 创建约束

创建约束有两种方法

1. 直接在字段后添加列约束 （column constraint）
2. 在末尾用 CONSTRAINT 关键字添加表约束（table constraint）

```sql
CREATE TABLE schema_name.table_name (
    column_1 data_type column_constraint,
    column_2 data_type column_constraint,
    ...
    table_constraint
 );
```

示例：

```sql
CREATE TABLE Persons
(
P_Id nchar(255) PRIMARY KEY,  --主键约束
Age number(3,0) CHECK (Age>0),
Sex varchar2(10),
LastName varchar2(255) NOT NULL,  -- 非空约束只能在字段后添加
FirstName varchar2(255),
Address varchar2(255),
City varchar2(255) DEFAULT 'HK',
CONSTRAINT ck_sex CHECK(Sex in ('male','female'))  -- 添加表约束
);

CREATE TABLE Orders
(
O_Id nchar(255) UNIQUE, 
OrderNo nchar(255) NOT NULL,
P_Id nchar(255) REFERENCES Persons(P_Id),  --外键约束
OrderDate date DEFAULT SYSDATE,    --默认值(调用函数SYSDATE)
CONSTRAINT ID PRIMARY KEY(O_Id)  --主键约束（表约束）
-- CONSTRAINT P_Id FOREIGN KEY REFERENCES Persons(P_Id)  -- 外键约束(表约束)
-- ON DELETE [CASCADE|SET NULL] -- 在主表被清除数据时删除外键或者重置为NULL
);
```

如需创建约束名字，或定义多个列的约束，可使用关键字 CONSTRAINT：

```sql
CREATE TABLE Persons
(
P_Id nchar(255) NOT NULL,
LastName varchar2(255) NOT NULL,
FirstName varchar2(255),
Address varchar2(255),
City varchar2(255),
CONSTRAINT uc_PersonID PRIMARY KEY (P_Id,LastName), --命名约束
CONSTRAINT chk_Person CHECK (P_Id>0 AND City='Sandnes')  --联合约束
);
```

## 修改约束

```sql
ALTER TABLE Persons
MODIFY P_Id nchar(255) NOT NULL; --添加非空约束
ALTER TABLE Persons
MODIFY P_Id nchar(255) NULL; --删除非空约束

ALTER TABLE Persons
ADD CONSTRAINT uc_PersonID 
PRIMARY KEY(P_Id);  --添加主键约束，并命名为 uc_PersonID
ALTER TABLE Persons
DROP PRIMARY KEY [CASECADE];  --删除主键[解除外键连接]

ALTER TABLE Orders
ADD CONSTRAINT P_Id FOREIGN KEY REFERENCES Persons(P_Id);  --添加外键约束

-- 以下为修改约束通用方法
ALTER TABLE Orders
ADD CONSTRAINT un_Id 
UNIQUE(O_id);  --添加约束

ALTER TABLE Persons
RENAME CONSTRAINT uc_PersonID TO new_ID; --修改约束名字

ALTER TABLE Persons
DISABLE|ENABLE CONSTRAINT new_ID; --禁用/启用约束

ALTER TABLE Persons
DROP CONSTRAINT new_ID; --删除约束
```

# 查询

## SELECT 语句

SELECT 语句用于从表中选取数据。

```sql
SELECT [DISTINCT] col1, col2, ... 
FROM table_name 
[WHERE conditions]
[ORDER BY col_list] [ASC|DESC] [NULLS FIRST|LAST];
```

> NULLS FIRST|LAST 表示将null值放置在前面或后面

```sql
SELECT country_id, city, state
FROM locations
ORDER BY state ASC NULLS FIRST;
```

## 运算符和表达式

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

## TOP 查询

Oracle数据库标准中没有`LIMIT`子句
Oracle 11g及以下版本可以用 `ROWNUM` 虚列控制

```sql
-- ROWNUM 序列关键字控制返回的记录数
SELECT col_list
FROM table_name
WHERE ROWNUM <= number
-- 示例
SELECT * FROM Person WHERE ROWNUM <= 5
```

Orace 12c以上版本中，``FETCH` 子句可以用来限制查询返回的行数

```sql
[OFFSET num ROWS]
 FETCH  NEXT [row_count | percent PERCENT] ROWS  [ONLY | WITH TIES]
```

`OFFSET` 子句指定在限制开始之前要跳过行数。如果跳过它，则偏移量为 0，行限制从第一行开始计算。
`WITH TIES`返回与最后一行相同的排序键。请注意，如果使用`WITH TIES`，则必须在查询中指定一个`ORDER BY`子句。如果不这样做，查询将不会返回额外的行。

```sql
-- 以下查询语句仅能在Oracle 12c以上版本执行
SELECT
 product_name,
 quantity
FROM
 inventories
INNER JOIN products
 USING(product_id)
ORDER BY
 quantity DESC 
FETCH NEXT 10 ROWS ONLY;
```

## GROUP BY 语句

```sql
SELECT col_list, aggregate_function(column_name)
FROM table_name
WHERE conditions
GROUP BY col_list
HAVING aggregate_condition  --筛选聚合后结果
ORDER BY col_list;
```

GROUP BY 语句用于结合聚合函数，根据一个或多个列对结果集进行分组。
在 SQL 中增加 HAVING 子句原因是，WHERE 关键字无法与聚合函数一起使用。

```sql
SELECT Customer,SUM(OrderPrice) FROM Orders
WHERE Customer='Bush' OR Customer='Adams'
GROUP BY Customer
HAVING SUM(OrderPrice)>1500;

SELECT job, AVG(salary) AS avg_sal FROM  employee
GROUP BY job
ORDER BY avg_sal;

-- 分组函数的嵌套（求平均工资的最大值）
SELECT MAX(AVG(salary)) FROM  employee 
GROUP BY job;
```

## GROUP BY 增强

GROUP BY 增强语法的含义见 [HiveQL](https://blog.csdn.net/qq_41518277/article/details/80902191/#GROUP-BY-增强语法) ，Oracle 中的语法如下示例

```sql
SELECT country_id, state, AVG(income) 
FROM locations
GROUP BY GROUPING SETS((country_id, state), country_id, ());

SELECT country_id, state, AVG(income) 
FROM locations
GROUP BY ROLLUP(country_id, state);

SELECT country_id, state, AVG(income) 
FROM locations
GROUP BY CUBE(country_id, state);
```

Oracle 还支持混合使用

```sql
SELECT a, b, SUM(x) FROM tab1 
GROUP BY GROUPING SETS (a),
         GROUPING SETS (b) ;
-- 等效于
SELECT a, b, SUM(x) FROM tab1 GROUP BY a, b;

SELECT a, b, c, SUM(x) FROM tab1 
GROUP BY a, 
         GROUPING SETS (b, c) ;
-- 等效于
SELECT a, b, c, SUM(x) FROM tab1 
GROUP BY GROUPING SETS (a), 
         GROUPING SETS (b, c) ;
-- 等效于
SELECT a, b, null, SUM(x) FROM tab1 GROUP BY a, b 
UNION
SELECT a, null, c, SUM(x) FROM tab1 GROUP BY a, c ;
```

Oracle 还支持 ROLLUP/CUBE 内组合列（HIVE没有类似的语法）

```sql
SELECT a, b, c, SUM(x) 
FROM tab1
GROUP BY ROLLUP(a, (b, c));
-- 等效于
SELECT a, b, c, SUM(x) FROM tab1 GROUP BY a, b, c 
UNION
SELECT a, null, null, SUM(x) FROM tab1 GROUP BY a 
UNION
SELECT null, null, null, SUM(x) FROM tab1;
```

GROUPING() 函数和GROUPING__ID的用法同HIVE

```sql
SELECT 
	DECODE(GROUPING(country_id),0,country_id,'All') country_id, 
	DECODE(GROUPING(state),0,state,GROUPING(country_id),0,country_id,'All') state, 
	AVG(income) 
FROM locations
GROUP BY ROLLUP(country_id, state);
```

# 常用函数

> Oracle 中常用 dual目标表测试

| 日期函数                      | 说明                                     |
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


| 数学函数                 | 描述                     |
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

| 字符函数                    | 说明                                   |
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

| 格式化函数           | 说明                         |
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


| 通用函数                           | 描述                                 |
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



# 子查询（Subquery）

## 子查询

问题不能一步解决问题时，使用子查询语句（SELECT 语句嵌套）。
可以使用子查询的位置：where, having, from, select

```sql
SELECT col_list FROM tb
WHERE expr operator (Subquery);
```

- 单行操作符：`=,<>,>,<,>=,<=`
- 多行操作符：`EXISTS,ANY,ALL,SOME,IN`
  `EXISTS`：检查子查询返回的行是否存在
  `ANY,ALL,SOME,IN`：将值与列表或子查询进行比较
  
  > oracle官方文档提醒：子查询中含有NULL的话使用 NOT IN 会出错值，建议先排除NULL后再使用

```sql
-- 单行子查询
select * from employee 
where salary > (select salary from employee where ename='Scott');

-- 多行子查询
select * from employee e
where salary > ALL(select salary from employee where deptno=20);

SELECT w.name, w.url 
FROM Websites w
WHERE EXISTS (SELECT count FROM access_log a WHERE w.id = a.site_id AND count > 200);
```

## 相关子查询 

了解相关的子查询，它是一个依赖于外部查询返回的值的子查询。

```sql
-- 大于部门平均工资的员工信息
select * from employee e
where salary > (select avg(salary) from employee where deptno=e.deptno);
```

# 表连接

1. **笛卡尔集合**

   ```sql
   SELECT col_list FROM tableA, tableB;
   SELECT col_list FROM tableA
   CROSS JOIN tableB;
   ```

2. **内连接**：根据连接条件中的运算符又分为 *等值连接* 与 *非等值连接*

   ```sql
   SELECT col_list FROM tableA, tableB 
   WHERE conditions;
   
   SELECT col_list FROM tableA 
   [INNER] JOIN tableB 
   ON conditions;
   ```

   **on** 和 **where** 条件的区别如下：

   - **on** 条件是在生成临时表时使用的条件，它不管 **on** 中的条件是否为真，都会返回左边表中的记录。

   - **where** 条件是在临时表生成好后，再对临时表进行过滤的条件。

   ```sql
   -- 等值连接
   SELECT empno, ename, dname 
   FROM employee emp, dept
   WHERE emp.deptno = dept.deptno AND dept.deptno = 30;
   
   SELECT empno, ename, dname 
   FROM employee emp
   JOIN dept
   ON emp.deptno = dept.deptno AND dept.deptno = 30;
   
   -- 显示雇员的编号,姓名,工资,工资级别,所在部门的名称;
   SELECT empno, ename,salary, grade, dname 
   FROM employee emp,dept,salgrade
   WHERE emp.depno = dept.depno 
   	AND emp.salary BETWEEN lowsal AND highsal;
   ```

3. **外连接**：对于外连接，在Oracle中也可以使用(+)来表示

   ```sql
   SELECT col_list FROM tableA 
   LEFT|RIGHT|FULL [OUTER] JOIN tableB 
   ON conditions;
   
   SELECT col_list FROM tableA a, tableB b
   WHERE a.col1 operator b.col2(+);  -- 左连接
   SELECT col_list FROM tableA a, tableB b
   WHERE a.col1(+) operator b.col2;  -- 右连接
   ```
   **INNER JOIN**: 取交集
   **LEFT JOIN**: 取左表全部，右表没有对应的值为 null
   **RIGHT JOIN**: 取右表全部，左表没有对应的值为 null
   **FULL JOIN**: 取并集，彼此没有对应的值为 null

   ```sql
   SELECT * FROM orders a,order_items b
   WHERE a.order_id=b.order_id(+);
   
   SELECT * FROM orders a
   LEFT JOIN order_items b
   ON a.order_id=b.order_id;
   ```

4. **自然连接**：在两张表中寻找那些数据类型和列名都相等的字段，然后自动地将他们连接起来。
   ```sql
   SELECT col_list FROM tableA
   NATURAL JOIN tableB;
   ```
   
5. 除`ON`子句外，还可以使用`USING`子句指定共有的相等字段。
   ```sql
   SELECT col_list FROM tableA 
   JOIN tableB 
   USING(col1, col2,...);
   
   -- 示例
   SELECT *
   FROM orders
   INNER JOIN order_items 
   USING(order_id);
   ```

# 集合操作

UNION 操作符用于合并两个或多个 SELECT 语句的结果集。
本节介绍使用集合运算符合并两个或多个独立查询的结果集的步骤。
● UNION - 将两个查询的结果合并为一个结果。
● INTERSECT - 实现两个独立查询的结果的交集。
● MINUS - 从一个结果集中减去另一个结果(也就是求差集)。

```sql
-- 并集
select_statement 
UNION [ALL] select_statement 
UNION [ALL] select_statement
... ... ;
-- 交集
select_statement 
INTERSECT select_statement 
INTERSECT select_statement
... ... ;
-- 差集
select_statement 
MINUS select_statement 
MINUS select_statement
... ... ;
```

> 加上`ALL` 关键字时不会自动去重。
> 请注意，UNION 内部的 SELECT 语句必须拥有相同数量的列。列也必须拥有相似的数据类型。
> 结果集中的列名总是等于 UNION 中第一个 SELECT 语句中的列名。

# 索引（INDEX）

索引是一个目录清单，每个索引条目记录着表中某行的索引列的值，以及此行的物理标识。在不读取整个表的情况下，索引使数据库应用程序可以更快地查找数据。用户无法看到索引，它们只能被用来加速搜索/查询。

```sql
CREATE [UNIQUE] INDEX index_name
ON table_name (column_list);

DROP INDEX index_name;  -- 删除索引
```

> column_list 规定需要索引的列。
> UNIQUE 意味着两个行不能拥有相同的索引值。
> 注释：更新一个包含索引的表需要比更新一个没有索引的表花费更多的时间，这是由于索引本身也需要更新。因此，理想的做法是仅仅在常常被搜索的列（以及表）上面创建索引。

```sql
CREATE INDEX PersonIndex
ON Person (LastName, FirstName)
```

> MySQL 删除索引
>
> ```sql
> ALTER TABLE table_name DROP INDEX index_name;
> ```

# 视图（VIEW）

视图就是由 SELECT 语句指定的一个逻辑对象，每次查询视图时都会导出该查询。与表不同，视图不会存储任何数据。
视图分为简单视图和复杂视图，简单视图可以支持 DML 操作。简单视图是指基于单个表建立的，不含任何函数、表达式和分组数据的视图。

## 创建视图

```sql
-- 创建或更新视图
CREATE [OR REPLACE] VIEW view_name AS
defining-query
[WITH READ ONLY]  -- 防止通过视图对底层表的 DML 操作
```

```sql
-- 示例
CREATE OR REPLACE VIEW backlogs AS
SELECT product_name,
    EXTRACT(YEAR FROM order_date) YEAR,
    SUM(quantity * unit_price) amount
FROM orders
INNER JOIN order_items
        USING(order_id)
INNER JOIN products
        USING(product_id)
WHERE status = 'Pending'
GROUP BY product_name,
    EXTRACT(YEAR FROM order_date);
    
-- 查询视图中可以更新的列
SELECT table_name, column_name,
	insertable, updatable, deletable
FROM dba_updatable_columns
WHERE table_name = 'backlogs';
```

## 删除视图

```sql
DROP VIEW schema_name.view_name 
[CASCADE CONSTRAINT];  -- 释放视图约束
```

# 自增序列

我们通常希望在每次插入新记录时，自动地创建主键字段的值。
在 Oracle 中，可以通过 sequence 创建自增序列（auto-increment）生成器。

```sql
CREATE SEQUENCE seq_name
[INCREMENT BY n] -- 序列步长，默认为 1
[START WITH n]  -- 序列的初始值(即产生的第一个值)，默认为1
[MINVALUE n]  -- 能产生的最小值
[MAXVALUE n | NOMAXVALUE] -- 能产生的最大值，默认 NOMAXVALUE
[CYCLE|NOCYCLE]  -- 值达到限制值后是否循环
[CACHE n|NOCACHE]; -- 序列的内存块的大小，默认为20
```

```sql
-- 示例
CREATE SEQUENCE seq_person
MINVALUE 1
START WITH 1
INCREMENT BY 1
NOMAXVALUE 
NOCYCLE
CACHE 10;
```

上面的代码创建名为 seq_person 的序列对象，它以 1 起始且以 1 递增。该对象缓存 10 个值以提高性能。

```sql
SELECT seq_person.CURRVAL FROM dual; -- 获得当前序列值
SELECT seq_person.NEXTVAL FROM dual; -- 获得下一个序列值

DROP SEQUENCE seq_person; -- 删除序列
```

第一次NEXTVAL返回的是初始值；随后的 NEXTVAL 会自动增加序列值，并返回增加后的值。
要在 "Persons" 表中插入新记录，我们必须使用 NEXTVAL 函数（该函数从 seq_person 序列中取回下一个值）：

```sql
INSERT INTO Persons (P_Id, FirstName, LastName)
VALUES (seq_person.nextval,'Lars','Monsen')
```

在ORACLE 12C以前的版本中,如果要实现列自增长,需要通过序列+触发器实现,到了12C ORACLE 引进了Identity Columns新特性,从而实现了列自增长功能。

```sql
GENERATED 
[ALWAYS | BY DEFAULT [ON NULL] ] 
AS IDENTITY [ (identity_options) ]
```
**参数**：

- 使用 BY DEFAULT  就是采用默认的序列生成器，此时用户可以插入自己的值。如果在添加 ON NULL 选项，则表示仅当插入NULL时，才会自增。
- 也可以使用  ALWAYS 指定独立的序列规则，此时自增列只能使用序列生成器提供的值，用户无法更改自增列。
  identity_options ：同上面序列生成器的参数
- 可以在创建table时指定identity columns的类型和规则，也可以创建之后使用alter table 来修改。

![](https://gitee.com/WilenWu/images/raw/master/sql/20210905143112.png)

```sql
-- BY DEFAULT
CREATE TABLE Persons
(
ID NUMBER(10) GENERATED BY DEFAULT AS IDENTITY,
Name varchar(255) NOT NULL,
);

-- ALWAYS
CREATE TABLE Persons
(
ID NUMBER(10) GENERATED ALWAYS AS IDENTITY(START WITH 0 INCREMENT BY 2),
Name varchar(255) NOT NULL,
);
```

MySQL 使用 AUTO_INCREMENT 关键字来执行 auto-increment 任务。
默认地，AUTO_INCREMENT 的开始值是 1，每条新记录递增 1。

```sql
CREATE TABLE Persons
(
ID int NOT NULL AUTO_INCREMENT,
Name varchar(255) NOT NULL,
PRIMARY KEY (ID)
);
```

要让 AUTO_INCREMENT 序列以其他的值起始：

```sql
ALTER TABLE Persons AUTO_INCREMENT=100
```

要在 "Persons" 表中插入新记录，我们不必为 "P_Id" 列规定值（会自动添加一个唯一的值）：

```sql
INSERT INTO Persons (FirstName,LastName)
VALUES ('Bill','Gates')
```

# CONNECT BY LEVEL

```sql
-- 获取连续的数
SQL> select level from dual connect by level <= 5;
LEVEL
1
2
3
4
5
-- 获取连续的日期
SQL> select trunc(sysdate, 'd') + level as thisWeek
SQL> from dual 
SQL> connect by level <= 7;
```



参考链接：
[易百教程 - Oracle](https://www.yiibai.com/oracle)
[PL/SQL教程](https://www.yiibai.com/plsql)
[W3school 新学院 - Oracle](http://www.hechaku.com/Oracle/Sql_Oracle_info.html)