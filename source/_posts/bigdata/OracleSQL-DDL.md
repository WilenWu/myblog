---
title: 大数据手册(Oracle)--Oracle SQL(DDL)
date: '2021-05-03 16:18:01'
categories:
  - Big Data
  - Oracle
tags:
  - 大数据
  - oracle
  - SQL
cover: /img/oracle-ddl.png
top_img: /img/SQL.png
abbrlink: 3928076d
description: '数据定义语言(Data Definition Language, DDL)：其语句包括动词CREATE、ALTER和DROP。'
---

# 数据库

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


# 约束

SQL 约束（Constraints）用于规定表中的数据规则。如果存在违反约束的数据行为，行为会被约束终止。
可以在创建表时规定约束（通过 CREATE TABLE 语句），或者在表创建之后也可以（通过 ALTER TABLE 语句）。我们将主要探讨以下几种约束：

- NOT NULL：强制列非空值
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

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/sql/20210905143112.png)

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
