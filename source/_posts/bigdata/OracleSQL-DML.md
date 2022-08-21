---
title: 大数据手册(Oracle)--Oracle SQL(DML)
date: '2021-05-03 16:18:01'
categories:
  - Big Data
  - Oracle
tags:
  - 大数据
  - oracle
  - SQL
cover: /img/oracle-dml.png
top_img: /img/SQL.png
abbrlink: 36f93ce2
description: >-
  数据操作语言(Data Manipulation Language, DML）：其语句包括动词
  INSERT、UPDATE、DELETE。它们分别用于添加、修改和删除。
---


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