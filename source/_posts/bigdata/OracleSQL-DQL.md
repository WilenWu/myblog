---
title: 大数据手册(Oracle)--Oracle SQL(DQL)
date: '2021-05-03 16:18:01'
categories:
  - 'Big Data'
  - Oracle
tags:
  - 大数据
  - oracle
  - SQL
cover: /img/oracle-dql.png
top_img: /img/SQL.png
abbrlink: 5122306f
description: 数据查询语言(Data Query Language, DQL）：用以从表中获得数据，包括 SELECT，WHERE，ORDER BY，GROUP BY和HAVING等。
---

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

## 增强聚合

GROUP BY 增强语法的含义见 [HiveQL](/posts/297bd708/#增强聚合) ，Oracle 中的语法如下示例

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

# 子查询

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
- UNION - 将两个查询的结果合并为一个结果。
- INTERSECT - 实现两个独立查询的结果的交集。
- MINUS - 从一个结果集中减去另一个结果(也就是求差集)。

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
