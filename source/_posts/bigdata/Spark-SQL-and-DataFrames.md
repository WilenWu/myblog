---
title: 大数据手册(Spark)--Spark SQL and DataFrames
categories:
  - Big Data
  - Spark
tags:
  - 大数据
  - Spark
  - python
cover: /img/apache-spark-sql.png
top_img: /img/apache-spark-top-img.svg
abbrlink: bb755aa3
date: 2020-01-03 16:20:25
description: DataFrame 抽象
---

# Spark SQL

Spark SQL用于对结构化数据进行处理，它提供了DataFrame的抽象，作为分布式平台数据查询引擎，可以在此组件上构建大数据仓库。

> [PySpark 3.5 Tutorial For Beginners with Examples](https://sparkbyexamples.com/pyspark-tutorial/#google_vignette)

## SparkSession

在过去，你可能会使用 SparkContext、SQLContext和HiveContext来分别配置Spark环境、SQL环境和Hive环境。SparkSession本质上是这些环境的组合，包括读取数据、处理元数据、配置会话和管理集群资源。

要创建基本的SparkSession，只需使用SparkSession.builder

```python
from pyspark.sql import SparkSession
spark = SparkSession \
   .builder \
	 .enableHiveSupport() \
   .appName('myApp') \
   .master('local') \
	 .config('spark.driver.memory','2g') \
	 .config('spark.driver.memoryOverhead','1g') \
	 .config('spark.executor.memory','2g') \
	 .config('spark.executor.memoryOverhead','1g') \
	 .config('spark.driver.cores','2') \
   .config('spark.executor.cores','2') \
   .getOrCreate()
sc = spark.sparkContext

spark.stop() # Stop the underlying SparkContext.
```
在 Spark 交互式环境下，默认已经创建了名为 spark 的 SparkSession 对象，不需要自行创建。

| SparkSession.builder||
|---|---|
|`SparkSession.builder.appName(name)` |设置Spark web UI上的应用名|
|`SparkSession.builder.master(master)`|设置Spark master URL|
|  `SparkSession.builder.config(key, value, …)`|配置选项|
| `SparkSession.builder.enableHiveSupport()` |启用Hive支持|
|`SparkSession.builder.getOrCreate`|创建SparkSession|
|`SparkSession.range()`|创建一个只含id列的DataFrame|

在使用Spark与Hive集成时，需要使用enableHiveSupport方法来启用Hive支持。启用Hive支持后，就可以在Spark中使用Hive的元数据、表和数据源。

## Catalog

pyspark.sql.Catalog 是面向用户的目录 API。

| pyspark.sql.Catalog                            |                                    |
| ---------------------------------------------- | ---------------------------------- |
| `Catalog.currentCatalog()`                       | 返回当前默认目录                   |
| `Catalog.currentDatabase()`                      | 返回当前默认数据库                 |
| `Catalog.databaseExists(dbName)`                 | 检查具有指定名称的数据库是否存在   |
| `Catalog.functionExists(functionName[, dbName])` | 检查具有指定名称的函数是否存在     |
| `Catalog.isCached(tableName)`                    | 检查指定表是否在内存中缓存         |
| `Catalog.tableExists(tableName[, dbName])`       | 检查具有指定名称的表或视图是否存在 |
| `Catalog.listDatabases([pattern])`               | 返回所有会话中可用的数据库列表     |
| `Catalog.listTables([dbName, pattern])`          | 返回指定数据库中的表/视图列表      |
| `Catalog.listFunctions([dbName, pattern])`       | 返回在指定数据库中注册的函数列表   |

```python
spark.range(1).createTempView("test_view")
spark.catalog.listTables()
[Table(name='test_view', catalog=None, namespace=[], description=None, ...
```


# DataFrame

DataFrame是一个分布式数据集，在概念上类似于传统数据库的表结构。

DataFrame的一个主要优点是：Spark引擎一开始就构建了一个逻辑执行计划，而且执行生成的代码是基于成本优化程序确定的物理计划。与Java或者Scala相比，Python中的RDD是非常慢的，而DataFrame的引入则使性能在各种语言中都保持稳定。

## 创建 DataFrame

PySpark DataFrame可以通过 SparkSession.createDataFrame创建。数据来源通常是结构化的python对象、RDD、Hive表或Spark数据源。并采用schema参数来指定DataFrame的模式，当schema被省略时，PySpark通过从数据中提取样本来推断相应的模式。

**从列表创建PySpark DataFrame**

```python
# spark is an existing SparkSession

# Create a DataFrame from a list of tuples.
spark.createDataFrame([
    (29, "Michael"),
    (30, "Andy"),
    (19, "Justin")
]).show()
+----+-------+
|  _1|     _2|
+----+-------+
|  29|Michael|
|  30|   Andy|
|  19| Justin|
+----+-------+

# Create a DataFrame from a list of dictionaries.
df = spark.createDataFrame([
    {"age": 29, "name": "Michael"},
    {"age": 30, "name": "Andy"},
    {"age": 19, "name": "Justin"}
])
```

通过采用schema参数来指定DataFrame的模式，schema 可以是字段名列表，DDL字符串，或者StructType

```python
# Create a DataFrame with column names specified.
df = spark.createDataFrame([
    (29, "Michael"),
    (30, "Andy"),
    (19, "Justin")
], schema=["age", "name"])

# Create a DataFrame with the schema in DDL formatted string.
df = spark.createDataFrame([
    (29, "Michael"),
    (30, "Andy"),
    (19, "Justin")
], schema="age: int, name: string")

# Create a DataFrame with the explicit schema specified.
from pyspark.sql.types import *
schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("name", StringType(), True)
])
df = spark.createDataFrame([
    (29, "Michael"),
    (30, "Andy"),
    (19, "Justin")
], schema=schema)
```

其中 StructField 包括 name（字段名）、dataType（数据类型）和nullable（此字段的值是否为空）。

**从 Rows 创建PySpark DataFrame**

```python
from pyspark.sql import Row
# Create a DataFrame from a list of Rows.
df = spark.createDataFrame([
    Row(age=29, name="Michael"),
    Row(age=30, name="Andy"),
    Row(age=19, name="Justin")
], schema="age int, name string")
```

**从pandas DataFrame创建PySpark DataFrame**

```python
import pandas as pd
# Create a DataFrame from a Pandas DataFrame.
pandas_df = pd.DataFrame({
    'age': [29, 30, 19],
    'name': ['Michael', 'Andy', 'Justin']
})
df = spark.createDataFrame(pandas_df)
```

**从RDD创建PySpark DataFrame**

```python
rdd = sc.parallelize([
    (29, "Michael"),
    (30, "Andy"),
    (19, "Justin")
])
df = rdd.toDF(schema="age int, name string")
```

**从 pandas-on-Spark DataFrame 创建 PySpark DataFrame**

```python
import pyspark.pandas as ps
psdf = ps.DataFrame({
    'age': [29, 30, 19],
    'name': ['Michael', 'Andy', 'Justin']
})
df = psdf.to_spark()
```

上面创建的DataFrame具有相同的结果和模式：

```python
# All DataFrames above result same.
df.show()
+---+-------+                                                                   
|age|   name|
+---+-------+
| 29|Michael|
| 30|   Andy|
| 19| Justin|
+---+-------+

df.printSchema()
# root
#  |-- age: integer (nullable = true)
#  |-- name: string (nullable = true)
```

PySpark DataFrame还提供了转化为其他结构化数据类型的方法

| pyspark.sql.DataFrame          |                        |
| ------------------------------ | ---------------------- |
| `DataFrame.toPandas()`           | 转化为Pandas DataFrame |
| `DataFrame.rdd`                  | 转化为RDD              |
| `DataFrame.to_pandas_on_spark()` | 转化为pandas-on-Spark  |
| `DataFrame.pandas_api()`         | 转化为pandas-on-Spark  |
| `DataFrame.toJSON()`             | 转化为Json字符串RDD    |
| `DataFrame.to(schema)` | 改变列顺序和数据类型 |

请注意，转换为Pandas DataFrame 时，会将所有数据收集到本地。当数据太大时，很容易导致 out-of-memory-error。

## DataFrame信息

| pyspark.sql.DataFrame   |                   |
| ----------------------- | ----------------- |
| `DataFrame.show(n, truncate, vertical)` | 预览前n行数据         |
| `DataFrame.collect()`     | 返回Row列表 |
| `DataFrame.take(num)`     | 返回前num个Row列表 |
| `DataFrame.head(n)`       | 返回前n个Row列表  |
| `DataFrame.tail(num)`     | 返回后num个Row列表 |
| `DataFrame.columns`       | 返回列名列表      |
| `DataFrame.dtypes`        | 返回数据类型列表  |
| `DataFrame.printSchema(level)` | 打印模式信息          |

```python
# spark, df are from the previous example
# Displays the content of the DataFrame to stdout
df.show()
+---+-------+                                                                   
|age|   name|
+---+-------+
| 29|Michael|
| 30|   Andy|
| 19| Justin|
+---+-------+
```

还可以通过启用`spark.sql.repl.eagerEval.enabled`，在Jupyter等notebook中美化 DataFrame 显示。显示的行数可以通过`spark.sql.repl.eagerEval.maxNumRows`配置。

```python
spark.conf.set('spark.sql.repl.eagerEval.enabled', True)
df.show()
```

当行太长而无法水平显示时，您也可以转置显示

```python
df.show(1, vertical=True)
# -RECORD 0-------
#  age  | 29    
#  name | Michael 
# only showing top 1 row
```

`DataFrame.collect()`将分布式数据收集到客户端，作为Python中的本地数据。

```python
df.collect()
# [Row(age=29, name='Michael'), Row(age=30, name='Andy'), Row(age=19, name='Justin')]
```

请注意，当数据集太大时，可能会引发 out-of-memory-error。为了避免抛出内存异常，请使用`DataFrame.take()`或`DataFrame.tail()`。

还可以查看DataFrame的列名和模式

```python
df.columns
# ['age', 'name']

# Print the schema in a tree format
df.printSchema()
# root
#  |-- age: integer (nullable = true)
#  |-- name: string (nullable = true)
```

## Input/Output

PySpark 支持许多外部数据源，如JDBC、text、binaryFile、Avro等。

### 外部数据源

Spark SQL提供pyspark.sql.DataFrameReader将文件读取到Spark DataFrame，并提供pyspark.sql.DataFrameWriter 写入文件。可用`option()`方法配置读写行为。

[Spark SQL Data Sources](https://spark.apache.org/docs/3.5.1/sql-data-sources.html)

本节先介绍使用Spark加载和保存数据的一般方法，然后介绍特定数据源。

```python
# spark, df are from the previous example

# "people" is a folder which contains multiple csv files and a _SUCCESS file.
df.write.save("people.csv", format="csv")

# Read a csv with delimiter and a header
df = spark.read.option("delimiter", ",").option("header", True).load("people.csv")

# You can also use options() to use multiple options
df = spark.read.options(delimiter=",", header=True).load("people.csv")

# Run SQL on files directly
df = spark.sql("SELECT * FROM parquet.`parquetFile.parquet`")
```

```python
# Parquet
parquetDF.write.parquet('parquetFile.parquet')
parquetDF = spark.read.parquet('parquetFile.parquet')
# ORC
orcDF.write.orc('orcFile.orc')
orcDF = spark.read.orc('orcFile.orc')
# Json
jsonDF.write.json('jsonFile.json')
jsonDF = spark.read.json('jsonFile.json')
# CSV 
csvDF.write.csv('csvFile.csv', header=True)
csvDF = spark.read.csv('csvFile.csv', sep=",", inferSchema="true", header=True)
# Text
textDF.write.txt('textFile.orc')
textDF = spark.read.txt('textFile.orc')
```

CSV简单明了，易于使用。Parquet和ORC是高效而紧凑的文件格式，可以更快地读取和写入。

### 通用选项/配置

仅适用于parquet、orc、avro、json、csv、text

| Options                            |                |
| ---------------------------------- | -------------- |
| spark.sql.files.ignoreCorruptFiles | 忽略损坏的文件 |
| spark.sql.files.ignoreMissingFiles | 忽略缺失的文件 |

```python
# enable ignore corrupt files via the data source option
# dir1/file3.json is corrupt from parquet's view
spark.read.option("ignoreCorruptFiles", "true")\
    .parquet("examples/src/main/resources/dir1/",
             "examples/src/main/resources/dir1/dir2/") \
    .show()
+-------------+
|         file|
+-------------+
|file1.parquet|
|file2.parquet|
+-------------+

# enable ignore corrupt files via the configuration
spark.sql("set spark.sql.files.ignoreCorruptFiles=true")
# dir1/file3.json is corrupt from parquet's view
spark.read.parquet("examples/src/main/resources/dir1/",
                   "examples/src/main/resources/dir1/dir2/")\
    .show()
+-------------+
|         file|
+-------------+
|file1.parquet|
|file2.parquet|
+-------------+
```

| Options                         |                                                    |
| ------------------------------- | -------------------------------------------------- |
| pathGlobFilter                  | 路径Glob过滤器。加载与给定glob模式匹配的路径的文件 |
| recursiveFileLookup             | 递归文件查找                                       |
| modifiedBefore<br>modifiedAfter | 加载给定修改时间范围内的文件                       |

```python
# pathGlobFilter is used to only include files with file names matching the pattern. 
spark.read.load("examples/src/main/resources/dir1",
    format="parquet", pathGlobFilter="*.parquet").show()
+-------------+
|         file|
+-------------+
|file1.parquet|
+-------------+

# recursiveFileLookup is used to recursively load files and it disables partition inferring. 
spark.read.format("parquet")\
    .option("recursiveFileLookup", "true")\
    .load("examples/src/main/resources/dir1").show()
+-------------+
|         file|
+-------------+
|file1.parquet|
|file2.parquet|
+-------------+

# Only load files modified before 07/1/2050 @ 08:30:00
spark.read.load("examples/src/main/resources/dir1",
    format="parquet", modifiedBefore="2050-07-01T08:30:00").show()
+-------------+
|         file|
+-------------+
|file1.parquet|
+-------------+
# Only load files modified after 06/01/2050 @ 08:30:00
spark.read.load("examples/src/main/resources/dir1",
    format="parquet", modifiedAfter="2050-06-01T08:30:00").show()
+-------------+
|         file|
+-------------+
+-------------+
```

### 保存模式

保存模式有以下几种

| DataFrame.write.mode                 |                                           |
| ------------------------------------ | ----------------------------------------- |
| "error" or "errorifexists" (default) | 如果数据已存在，则会报错                  |
| "append"                             | 如果数据/表已存在，则追加到现有数据中     |
| "overwrite"                          | 如果数据/表已经存在，则覆盖原数据         |
| "ignore"                             | 如果数据/表已经存在，则忽略，不改变原数据 |

```python
df.write.mode("overwrite").format("parquet").save("people.parquet")
```

### 分区

也可以对输出文件进行分区

```python
df.write.partitionBy("name").format("parquet").save("PartByName.parquet")
```

请注意，分区列的数据类型是自动推断的。目前，支持数字、日期、时间戳和字符串类型。如果不想自动推断分区列的数据类型，可以配置`spark.sql.sources.partitionColumnTypeInference.enabled`为`false`，则分区为字符串类型。

## Hive 表交互

### Hive配置

由于Hive有大量的依赖项，这些依赖项不包含在默认的Spark发行版中。一般我们需要将Hive的配置文件`hive-site.xml`、`core-site.xml`（用于安全配置）和`hdfs-site.xml`（用于HDFS配置）文件放在Spark的配置目录 `conf/`下。

请注意，自Spark 2.0.0以来，`hive-site.xml`中的`hive.metastore.warehouse.dir`属性已弃用。请使用`spark.sql.warehouse.dir` 来指定数仓的默认位置。

当然，对于没有部署Hive的用户仍然可以启用Hive支持，此时 Spark会自动在当前目录中创建`metastore_db`，并创建 spark数仓目录 `spark.sql.warehouse.dir`，数仓目录位于Spark应用程序启动目录下的 `spark-warehouse`。

### SQL 查询

通过调用SparkSession.sql()方法可以使用任意SQL语句，并返回DataFrame。

```python
SparkSession.sql(sqlQuery, args=None, **kwargs)
```

```python
spark.sql("SELECT * FROM range(10) where id > 7").show()
+---+
| id|
+---+
|  8|
|  9|
+---+

spark.sql(
    "SELECT * FROM range(10) WHERE id > {bound1} AND id < {bound2}", bound1=7, bound2=9
).show()
+---+
| id|
+---+
|  8|
+---+
```

### 保存HIve表

DataFrames可以使用 saveAsTable 方法和 insertInto 方法将表保存到Hive中。

- 调用 **saveAsTable** 保存，如果hive中不存在该表，则spark会自动创建此表。如果表已存在，则匹配插入数据和原表的 schema。
- 调用 **insertInto** 保存，无关schema，只需按数据的顺序插入，如果原表不存在则会报错。

```python
# Write a DataFrame into a Parquet file in a partitioned manner.
df.write.mode('overwrite').saveAsTable('people')
df.write.insertInto('people', overwrite=True)
```

同样，Hive表的保存模式有以下几种

| DataFrame.write.mode                 |                                           |
| ------------------------------------ | ----------------------------------------- |
| "error" or "errorifexists" (default) | 如果数据已存在，则会报错                  |
| "append"                             | 如果数据/表已存在，则追加到现有数据中     |
| "overwrite"                          | 如果数据/表已经存在，则覆盖原数据         |
| "ignore"                             | 如果数据/表已经存在，则忽略，不改变原数据 |

saveAsTable 支持分桶、排序和分区

```python
df.write.partitionBy("name").sortBy("age").saveAsTable("people")
# Write a DataFrame into a Parquet file in a bucketed manner.
df.write.bucketBy(2, "name").sortBy("age").saveAsTable("bucketed_table")
```

对于分区表，可以先开启**Hive动态分区**，则不需要指定分区字段。如果有一个分区，那么默认为数据中最后一列为分区字段，有两个分区则为最后两列为分区字段，以此类推。

```python
spark.conf.set("hive.exec.dynamic.partition", "true")
spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")

df.write.insertInto('people', overwrite=True)
```

若通过 `path`选项**自定义表路径**，当表被 DROP 时，自定义表路径不会被删除，表数据仍然存在。如果没有指定 `path`，Spark会将数据写入仓库目录下的默认表路径，当表被删除时，默认表路径也将被删除。

```python
df.write.option("path", "/some/path").saveAsTable("people")
```

请注意，在创建外部数据源表（具有`path`选项的表）时，默认不会收集分区信息。您可以调用`MSCK REPAIR TABLE`同步元存储中的分区信息。

## 常用方法

### Select

| pyspark.sql                                | 查询语句        |
| :----------------------------------------- | :-------------- |
| `DataFrame.select(*cols)`                 | `SELECT` in SQL |
| `DataFrame.selectExpr(*expr)`             | 解析SQL 语句    |
|`function.expr(*expr)`|解析SQL 语句(函数)|
| `Column.alias(*alias, **kwargs)`          | 重命名列        |
| `funtions.lit(col)`                        | 常数列（函数）  |
| `DataFrame.limit(num)`                     | `LIMIT` in SQL |


```python
df = spark.createDataFrame([
    (29, "Michael"),
    (30, "Andy"),
    (19, "Justin")
], schema=["age", "name"])

# Select everybody.
df.select('*').show()
+---+-------+                                                                   
|age|   name|
+---+-------+
| 29|Michael|
| 30|   Andy|
| 19| Justin|
+---+-------+

# Select the "name" and "age" columns
df.select("age", "name").show()
df.select(["age", "name"]).show()
+---+-------+                                                                   
|age|   name|
+---+-------+
| 29|Michael|
| 30|   Andy|
| 19| Justin|
+---+-------+

# Select everybody, but increment the age by 1
df.select(df['name'], df['age'] + 1).show()
+-------+---------+
|   name|(age + 1)|
+-------+---------+
|Michael|       30|
|   Andy|       31|
| Justin|       20|
+-------+---------+

from pyspark.sql import functions as fn
df.select("age", "name", fn.lit("2020").alias("update")).show()
+----+-------+------+
| age|   name|update|
+----+-------+------+
|  29|Michael|  2020|
|  30|   Andy|  2020|
|  19| Justin|  2020|
+----+-------+------+

df.selectExpr("age * 2", "abs(age)").show()
+---------+--------+
|(age * 2)|abs(age)|
+---------+--------+
|       58|      29|
|       60|      30|
|       38|      19|
+---------+--------+
```

### Column

|       pyspark.sql                          | 修改列               |
| :----------------------------------------- | :------------------- |
|`DataFrame.withColumnRenamed(existing, new)` | 重命名列             |
|`DataFrame.withColumnsRenamed(colsMap)`     | 重命名多列           |
|`DataFrame.toDF(*cols)` | 重命名所有列 |
|`DataFrame.withColumn(colName, col)`                   | 修改单列             |
|`DataFrame.withColumns(*colsMap)`                      | 修改多列             |
|`DataFrame.drop(*cols)`                             | 删除列               |
|`DataFrame.to(schema)`                              | 模式变换             |
|`Column.astype(dataType)`                    | 数据类型转换         |
|`Column.cast(dataType)`                      | 数据类型转换         |
|`funtions.cast(dataType)`                    | 数据类型转换（函数） |
|`Column.when(condition, value)` | `CASE WHEN...THEN` in SQL |
|`Column.otherwise(value)`  |    `ELSE` in SQL              |
|`funtions.when(condition, value)`| `CASE WHEN...THEN` in SQL |
|`funtions.col(col)` |返回Column实例|
|`funtions.lit(col)`|常数列|


在Python中，大多数情况都是通过属性或索引操作DataFrame，它会返回一个`Column`实例

```python
# spark, df are from the previous example

df.age
# Column<'age'>

df['age']
# Column<'age'>
```

```python
import pyspark.sql.functions as fn

df = spark.createDataFrame([
    (2, "Alice"), 
    (3, "Alice"), 
    (5, "Bob"), 
    (10, "Bob")
], ["age", "name"])
df.withColumn("baby", fn.when(df['age'] <= 3, 1).otherwise(0)).show()
+---+-----+----+
|age| name|baby|
+---+-----+----+
|  2|Alice|   1|
|  3|Alice|   1|
|  5|  Bob|   0|
| 10|  Bob|   0|
+---+-----+----+
```

### Distinct

| pyspark.sql                 | 去重       |
| :------------------------------- | :--------- |
| `DataFrame.distinct()`      | `DISTINCT` in SQL |
| `DataFrame.dropDuplicates(subset=None)` | 删除重复项 |

```python
df = spark.createDataFrame([
    (2, "Alice"), 
    (3, "Alice"), 
    (5, "Bob"), 
    (10, "Bob")
], ["age", "name"])

df.dropDuplicates(subset=["name"]).show()
+---+-----+
|age| name|
+---+-----+
|  5|  Bob|
|  2|Alice|
+---+-----+
```


### Filter

| pyspark.sql                            | 条件筛选              |
| :------------------------------------- | :-------------------- |
| `DataFrame.filter(condition)`            | 筛选，别名`where` |
| `Column.isin(*cols)`                     | `IN (...)` in SQL   |
| `Column.contains(other)`                 | `LIKE "%other%"` in SQL |
| `Column.like(pattern)`                   | `LIKE` in SQL       |
| `Column.rlike(pattern)`                  | 正则表达式匹配        |
| `Column.startswith(pattern)`             | 匹配开始              |
| `Column.endswith(pattern)`               | 匹配结尾              |
| `Column.between(lowerBound, upperBound)` | `BETWEEN ... AND ...` in SQL |


```python
df = spark.createDataFrame([
    (None, "Michael"),
    (30, "Andy"),
    (19, "Justin")
], schema=["age", "name"])

# Select people older than 21
df.filter(df['age'] > 21).show()
+---+----+
|age|name|
+---+----+
| 30|Andy|
+---+----+

# Filter by Column instances,
# Or SQL expression in a string.
df.filter(df.age == 30).show()
df.filter("age = 30").show()
+---+----+
|age|name|
+---+----+
| 30|Andy|
+---+----+

df.filter(df.name.rlike("^A")).show()
df.filter(df.name.startswith("A")).show()
+---+----+
|age|name|
+---+----+
| 30|Andy|
+---+----+

df.filter(df.name.isin('Andy','Justin')).show()
+---+-------+                                                                   
|age|   name|
+---+-------+
| 30|   Andy|
| 19| Justin|
+---+-------+
```

### Group

|          pyspark.sql              | 分组和聚合            |
| :------------------------------------- | :-------------------- |
| `DataFrame.groupBy(*cols)`           | 返回GroupedData |
| `GroupedData.count()`                  | 计数                  |
| `GroupedData.sum(*cols)`               | 求和                  |
| `GroupedData.avg(*cols)`               | 平均值                |
| `GroupedData.mean(*cols)`              | 平均值                |
| `GroupedData.max(*cols)`               | 最大值                |
| `GroupedData.min(*cols)`               | 最小值                |
| `GroupedData.agg(*exprs)`              | 应用表达式         |
| `GroupedData.pivot(pivot_col, values)` | 透视 |

PySpark DataFrame支持按特定条件对数据进行分组，将函数应用于每个组，然后将它们合并返回DataFrame。

```python
df = spark.createDataFrame([
    (2, "Alice", "pink"), 
    (3, "Alice", "pink"), 
    (5, "Bob", "pink"), 
    (10, "Bob", "orange")
], ["age", "name", "color"])

# Group-by name, and count each group.
df.groupBy('name').count().show()
df.groupBy(df.name).agg({"*": "count"}).show()
+-----+--------+
| name|count(1)|
+-----+--------+
|Alice|       2|
|  Bob|       2|
+-----+--------+
```

聚合函数还可以使用 pyspark.sql.functions 模块提供的其他函数，如countDistinct, kurtosis, skewness, stddev, sumDistinct, variance 等。或者使用 pyspark.pandas API对每个组应用Python原生函数

```python
import pyspark.sql.functions as fn
from pyspark.sql.functions import pandas_udf, PandasUDFType

# Group-by name, and calculate the minimum age.
df.groupBy('name').agg(fn.min(df.age)).show() 
+-----+--------+
| name|min(age)|
+-----+--------+
|  Bob|       5|
|Alice|       2|
+-----+--------+

# Same as above but uses pandas UDF.
@pandas_udf('int', PandasUDFType.GROUPED_AGG)  
def min_udf(v):
    return v.min()

df.groupBy(df.name).agg(min_udf(df.age)).sort("name").show()  
+-----+------------+                                                            
| name|min_udf(age)|
+-----+------------+
|Alice|           2|
|  Bob|           5|
+-----+------------+
```


|         pyspark.sql  | 增强分组         |
| -------------------------- | ---------------- |
| `DataFrame.cube(*cols)`     |`CUBE` in SQL |
| `DataFrame.rollup(*cols)`   |  `ROLLUP` in SQL |
|`functions.grouping(col)`|`GROUPING` in SQL|


```python
from pyspark.sql.functions import grouping, min
df.cube("name", "color").agg(grouping("name"), min("age")).show()
+-----+------+--------------+--------+
| name| color|grouping(name)|min(age)|
+-----+------+--------------+--------+
|  Bob|  NULL|             0|       5|
|  Bob|orange|             0|      10|
| NULL|orange|             1|      10|
|Alice|  NULL|             0|       2|
| NULL|  NULL|             1|       2|
|Alice|  pink|             0|       2|
|  Bob|  pink|             0|       5|
| NULL|  pink|             1|       2|
+-----+------+--------------+--------+
```

### Sort

| pyspark.sql          | 排序            |
| :-------------------------- | :--------------- |
| `DataFrame.sort(*col, **kwargs)` | `ORDER BY` in SQL，别名orderBy |
| `Column.asc()`              | `ASC` in SQL |
| `Column.asc_nulls_first()`  | `ASC` ，NULL放第一位 |
| `Column.asc_nulls_last()`   | `ASC` ，NULL放最后一位 |
| `Column.desc()`             | `DESC` in SQL |
|`Column.desc_nulls_first`|`DESC`，NULL放第一位|
|`Column.desc_nulls_last`|`DESC`，NULL放最后一位|
|`funtions.asc(col)`|`ASC` in SQL|
|`funtions.desc(col)`|`DESC` in SQL|


```python
import pyspark.sql.functions as fn

df = spark.createDataFrame([
    (2, "Alice"), 
    (3, "Alice"), 
    (5, "Bob"), 
    (10, "Bob")
], ["age", "name"])

# Sort the DataFrame in descending order.
df.sort(df.age.desc()).show()
df.sort("age", ascending=False).show()
df.sort(fn.desc("age")).show()
+---+-----+
|age| name|
+---+-----+
| 10|  Bob|
|  5|  Bob|
|  3|Alice|
|  2|Alice|
+---+-----+

# Specify multiple columns
df.orderBy(fn.desc("age"), "name").show()
+---+-----+
|age| name|
+---+-----+
| 10|  Bob|
|  5|  Bob|
|  3|Alice|
|  2|Alice|
+---+-----+

# Specify multiple columns for sorting order at ascending.
df.orderBy(["age", "name"], ascending=[False, False]).show()
+---+-----+
|age| name|
+---+-----+
| 10|  Bob|
|  5|  Bob|
|  3|Alice|
|  2|Alice|
+---+-----+
```

### Joins

| pyspark.sql            | 连接                |
| :--------------------- | :------------------ |
| `DataFrame.alias(alias)` | 重命名表            |
| `DataFrame.join(other, on, how)` | 表连接              |
| `DataFrame.crossJoin(other)` | 笛卡尔集            |
| `DataFrame.intersect(other)` | 交集，并移除重复行  |
| `DataFrame.intersectAll(other)` | 交集                |
| `DataFrame.subtract(other)` | 差集                |
| `DataFrame.exceptAll(other)` | `EXCEPT ALL` in SQL |

其中表连接 `DataFrame.join(other, on, how)`支持的参数有：

- on: str, list or Column.
- how: str, default inner. 
  Must be one of: inner, cross, outer, full, fullouter, full_outer, left, leftouter, left_outer, right, rightouter, right_outer, semi, leftsemi, left_semi, anti, leftanti and left_anti.

```python
df = spark.createDataFrame([(2, "Alice"), (5, "Bob")]).toDF("age", "name")
df2 = spark.createDataFrame([(80, "Tom"), (85, "Bob")]).toDF("height", "name")

# Inner join on columns (default)
df.join(df2, 'name').select(df.name, df2.height).show()
+----+------+
|name|height|
+----+------+
| Bob|    85|
+----+------+

# Outer join for both DataFrames on the ‘name’ column.
df.join(df2, df.name == df2.name, 'outer').select(
    df.name, df2.height).sort(fn.desc("name")).show()
+-----+------+
| name|height|
+-----+------+
|  Bob|    85|
|Alice|  NULL|
| NULL|    80|
+-----+------+
```

### Union

| pyspark.sql               | 合并               |
| :------------------------ | :----------------- |
| `DataFrame.union(other)`  | `UNION ALl` in SQL     |
| `DataFrame.unionAll(other)` | `UNION ALL` in SQL |


```python
df1 = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
df2 = spark.createDataFrame([(3, "Charlie"), (4, "Dave")], ["age", "name"])

df1.union(df2).show()
+---+-------+
|age|   name|
+---+-------+
|  2|  Alice|
|  5|    Bob|
|  3|Charlie|
|  4|   Dave|
+---+-------+
```

### Pivot

| pyspark.sql |    透视  |
| ---- | ---- |
| `GroupedData.pivot(pivot_col, values)` |   透视   |
| `DataFrame.unpivot(ids, values, variableColumnName, …)` |  逆透视，别名melt    |

```python
df = spark.createDataFrame([
    (2, "Alice", "pink"), 
    (3, "Alice", "pink"), 
    (5, "Bob", "pink"), 
    (10, "Bob", "orange")
], ["age", "name", "color"])

pivot_df = df.groupBy("name").pivot("color").avg("age")
pivot_df.show()
+-----+------+----+
| name|orange|pink|
+-----+------+----+
|  Bob|  10.0| 5.0|
|Alice|  NULL| 2.5|
+-----+------+----+

pivot_df.unpivot("name", ["pink", "orange"], "color", "avg_age").show()
+-----+------+-------+
| name| color|avg_age|
+-----+------+-------+
|  Bob|  pink|    5.0|
|  Bob|orange|   10.0|
|Alice|  pink|    2.5|
|Alice|orange|   NULL|
+-----+------+-------+
```

### Window

| pyspark.sql               |          窗口定义    |
| :------------------------ | :----------------- |
|`Column.over(window)`|`OVER` 子句|
|`functions.Window.partitionBy(*cols)`|窗口内分区|
|`functions.Window.orderBy(*cols)`|窗口内排序|
|`functionsWindow.rangeBetween(start, end)`|窗口区间|
|`functions.Window.rowsBetween(start, end)`|窗口区间|
|`functions.Window.unboundedFollowing`|最后一行|
|`functions.Window.unboundedPreceding`|第一行|
|`functions.Window.currentRow`|当前行|

```python
from pyspark.sql import Window
from pyspark.sql.functions import min, desc

df = spark.createDataFrame([
    (2, "Alice"), 
    (3, "Alice"), 
    (5, "Bob"), 
    (10, "Bob")
], ["age", "name"])

window = Window.partitionBy("name")

df.withColumn(
     "min", min('age').over(window)
).sort(desc("age")).show()
+---+-----+---+
|age| name|min|
+---+-----+---+
| 10|  Bob|  5|
|  5|  Bob|  5|
|  3|Alice|  2|
|  2|Alice|  2|
+---+-----+---+
```

| pyspark.sql.functions               |          窗口函数    |
| :------------------------ | :----------------- |
|`row_number()` |序号|
|`dense_rank()` | 排名，排名相等则下一个序号不间断|
|`rank()`|排名，排名相等则下一个序号间断|
|`percent_rank()`|百分比排名|
|`cume_dist()`| 累积分布|
|`lag(col, offset, default)` |返回往上第 offset 行值
|`lead(col, offset, default)` | 返回往下第 offset 行值
|`nth_value(col, offset, ignoreNulls)`|返回第 offset 行值|
|`ntile(n)`|切成 n 个桶，返回当前切片号|

```python
from pyspark.sql.functions import cume_dist, row_number

w = Window.orderBy("age")
df.withColumn("cume_dist", cume_dist().over(w)) \
  .withColumn("asc_order", row_number().over(w)).show()
+---+-----+---------+---------+
|age| name|cume_dist|asc_order|
+---+-----+---------+---------+
|  2|Alice|     0.25|        1|
|  3|Alice|      0.5|        2|
|  5|  Bob|     0.75|        3|
| 10|  Bob|      1.0|        4|
+---+-----+---------+---------+
```

## 函数

### Built-In Functions

Spark SQL 还拥有丰富的函数库，包括字符串操作、日期算术、常见数学运算、集合函数，聚合函数等。常用的函数有下面几个

[PySpark SQL functions API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)

| pyspark.sql                                           | 函数                          |
| :---------------------------------------------------- | :---------------------------- |
| `functions.coalesce(*cols)`                           | `COALESCE` in SQL             |
|`functions.nvl(col1, col2)`|`NVL` in SQL|
| `functions.greatest(*cols)`                           | 最大列的值                    |
| `functions.least(*cols)`                              | 最小列的值                    |
| `functions.monotonically_increasing_id()`             | 单调递增ID列                  |
| `functions.rand(seed)`                                | 生成随机列，服从 0-1 均匀分布 |
| `functions.randn(seed)`                               | 生成随机列，服从标准正态分布  |
| `Column.substr(startPos, length)`                     | 截取字符串                    |
|`functions.substr(str, pos[, len])`|截取字符串(函数)|
|`functions.concat_ws(sep, *cols)`|连接字符串(函数)|
| `DataFrame.randomSplit(weights, seed)`                | 拆分数据集                    |
| `DataFrame.replace(to_replace, value, subset)`        | 替换值                        |
| `DataFrame.sample(withReplacement, fraction, seed)` | 抽样                          |
| `DataFrame.sampleBy(col, fractions, seed)`          | 抽样                          |

```python
import pyspark.sql.functions as fn
spark.range(1, 7, 2).withColumn("rand", fn.rand()).show()
+---+--------------------+
| id|                rand|
+---+--------------------+
|  1|  -1.424479548864398|
|  3|-0.08662546937327156|
|  5| -0.5638136550015606|
+---+--------------------+
```

| pyspark.sql               |          集合函数   |
| :------------------------ | :----------------- |
`functions.collect_set(col)`            |行收集成数组，消除重复元素
| `functions.collect_list(col)`           |行收集成数组，具有重复项
|`functions.explode(col)`|数组分解成单列|

```python
df = spark.createDataFrame([
    (2, "Alice"), 
    (3, "Alice"), 
    (5, "Bob"), 
    (10, "Bob")
], ["age", "name"])

df2 = df.groupBy("name").agg(fn.collect_list("age").alias("age"))
df2.show()
+-----+-------+
| name|    age|
+-----+-------+
|  Bob|[5, 10]|
|Alice| [2, 3]|
+-----+-------+

df2.select("name", fn.explode("age").alias("age")).show()
+-----+---+
| name|age|
+-----+---+
|  Bob|  5|
|  Bob| 10|
|Alice|  2|
|Alice|  3|
+-----+---+
```

### Python UDFs

定义python UDF需要使用`functions.udf()`作为装饰器或包装函数 。

```python
from pyspark.sql.functions import udf

# Declare the function and create the UDF
@udf(returnType='int')  # A default, pickled Python UDF
def plus_one(s): # type: ignore[no-untyped-def]
    return s + 1

df = spark.createDataFrame([
    (29, "Michael"),
    (30, "Andy"),
    (19, "Justin")
], schema=["age", "name"])
df.select(plus_one("age")).show()
+-------------+
|plus_one(age)|
+-------------+
|           30|
|           31|
|           20|
+-------------+
```

此外，本章定义的任意UDF都可以在SQL中注册并调用：

```python 
# Declare the function and create the UDF
def plus_one_func(s): 
    return s + 1

plus_one = udf(plus_one_func, returnType='int') # type: ignore[no-untyped-def]
df.select(plus_one("age")).show()
+------------------+
|plus_one_func(age)|
+------------------+
|                30|
|                31|
|                20|
+------------------+

# Register the DataFrame as a SQL temporary view
df.createOrReplaceTempView("tableA")

spark.udf.register("plus_one", plus_one)
spark.sql("SELECT plus_one(age) FROM tableA").show()
+-------------+
|plus_one(age)|
+-------------+
|           30|
|           31|
|           20|
+-------------+
```

### Arrow Python UDFs

**Apache Arrow** 是一种内存中的列式数据格式，用于在 Spark 中高效传输 JVM 和 Python 进程之间的数据。这目前对 Python 用户使用 Pandas/NumPy 数据最有利 。默认为禁用状态，需要配置`spark.sql.execution.arrow.pyspark.enabled` 启用。

例如，当Pandas DataFrame和Spark DataFrame相互转化时，可通过设置 Arrow 进行优化。

```python
import numpy as np
import pandas as pd

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Generate a Pandas DataFrame
pdf = pd.DataFrame(np.random.rand(100, 3))

# Create a Spark DataFrame from a Pandas DataFrame using Arrow
df = spark.createDataFrame(pdf)

# Convert the Spark DataFrame back to a Pandas DataFrame using Arrow
result_pdf = df.select("*").toPandas()
```

**Arrow Batch Size**：Arrow 批处理可能会暂时导致 JVM 中的内存使用率较高。为避免出现内存不足异常，可通过设置`spark.sql.execution.arrow.maxRecordsPerBatch`来调整每个批次的最大行数。

**Arrow Python UDF** 是利用 Arrow 高效批处理的自定义函数。通过设置udf()的参数`useArrow=True`来定义Arrow Python UDF 。

下面是一个示例，演示了Arrow Python UDF 的用法：

```python
from pyspark.sql.functions import udf

@udf(returnType='int', useArrow=True)  # An Arrow Python UDF
def arrow_plus_one(s): # type: ignore[no-untyped-def]
    return s + 1

df = spark.createDataFrame([
    (29, "Michael"),
    (30, "Andy"),
    (19, "Justin")
], schema=["age", "name"])

df.select(arrow_plus_one("age")).show()
+-------------------+
|arrow_plus_one(age)|
+-------------------+
|                 30|
|                 31|
|                 20|
+-------------------+
```

此外，您还可以启用 `spark.sql.execution.pythonUDF.arrow.enabled`，在整个 SparkSession 中对 Python UDF 进行优化。与默认的序列化 Python UDF 相比，Arrow Python UDF 提供了更连贯的类型强制机制。

### Pandas UDFs

Pandas UDFs是用户定义的函数，由Spark执行，使用Arrow传输数据，支持矢量化操作。

定义一个 Pandas UDF需要使用pandas_udf() 作为装饰器或包装函数。注意，定义Pandas UDF时，需要定义数据类型提示。

以下示例演示如何创建Series to Series的 Pandas UDF。

```python
import pandas as pd
from pyspark.sql.functions import pandas_udf

# Declare the function and create the UDF
@pandas_udf('long')
def pandas_plus_one(series: pd.Series) -> pd.Series:
    # Simply plus one by using pandas Series.
    return series + 1

df = spark.createDataFrame([
    (29, "Michael"),
    (30, "Andy"),
    (19, "Justin")
], schema=["age", "name"])

# Execute function as a Spark vectorized UDF
df.select(pandas_plus_one('age')).show()
+--------------------+
|pandas_plus_one(age)|
+--------------------+
|                  30|
|                  31|
|                  20|
+--------------------+
```

聚合 UDF 也可以与 GroupedData 或Window 一起使用。此外，计算时组内所有数据将被加载到内存中。

以下示例演示如何使用 Series to Scalar UDF 通过分组计算均值和窗口操作：

```python
import pandas as pd

from pyspark.sql.functions import pandas_udf
from pyspark.sql import Window

df = spark.createDataFrame([
    (2, "Alice"), 
    (3, "Alice"), 
    (5, "Bob"), 
    (10, "Bob")
], ["age", "name"])

# Declare the function and create the UDF
@pandas_udf("double")  # type: ignore[call-overload]
def mean_udf(s: pd.Series) -> float:
    return s.mean()

df.select(mean_udf('age')).show()
+-------------+
|mean_udf(age)|
+-------------+
|          5.0|
+-------------+

df.groupby("name").agg(mean_udf('age')).show()
+-----+-------------+
| name|mean_udf(age)|
+-----+-------------+
|Alice|          2.5|
|  Bob|          7.5|
+-----+-------------+

w = Window.partitionBy('name') 
df.select('name', 'age', mean_udf('age').over(w).alias("mean_age")).show()
+-----+---+--------+
| name|age|mean_age|
+-----+---+--------+
|Alice|  2|     2.5|
|Alice|  3|     2.5|
|  Bob|  5|     7.5|
|  Bob| 10|     7.5|
+-----+---+--------+
```

### Pandas Function API

Pandas Function API 与Pandas UDF类似，使用Arrow传输数据，使用Pandas处理数据，允许矢量化操作。然而，Pandas Function API是直接作用于PySpark DataFrame而不是Column。

- GroupedData.applyInPandas()支持分组应用 UDF。
- GroupedData.cogroup().applyInPandas()支持联合分组应用UDF，它允许两个PySpark DataFrame由一个公共密钥联合分组，然后将Python函数应用于每个联合组。

映射函数需要定义以下内容：

- 定义应用于每个组的原生函数，该函数的输入和输出都是`pandas.DataFrame`
- 使用`StructType`对象或DDL模式的字符串定义输出数据类型。

请注意，在应用函数之前，组的所有数据都将加载到内存中，这可能会导致内存不足。

```python
df = spark.createDataFrame([
    (2, "Alice"), 
    (3, "Alice"), 
    (5, "Bob"), 
    (10, "Bob")
], ["age", "name"])

def subtract_mean(pdf: pd.DataFrame) -> pd.DataFrame:
    # pdf is a pandas.DataFrame
    age = pdf.age
    return pdf.assign(age=age - age.mean())

df.groupby("name").applyInPandas(subtract_mean, schema="name string, age double").show()
+-----+----+
| name| age|
+-----+----+
|Alice|-0.5|
|Alice| 0.5|
|  Bob|-2.5|
|  Bob| 2.5|
+-----+----+
```

### UDTFs

Spark 3.5 引入了 Python 用户定义表函数 （UDTF），每次返回整个表作为输出。

实现 UDTF 类并创建 UDTF：

```python
# Define the UDTF class and implement the required `eval` method.
class SquareNumbers:
    def eval(self, start: int, end: int):
        for num in range(start, end + 1):
            yield (num, num * num)

from pyspark.sql.functions import lit, udtf

# Create a UDTF using the class definition and the `udtf` function.
square_num = udtf(SquareNumbers, returnType="num: int, squared: int")

# Invoke the UDTF in PySpark.
square_num(lit(1), lit(3)).show()
+---+-------+
|num|squared|
+---+-------+
|  1|      1|
|  2|      4|
|  3|      9|
+---+-------+
```

还可以使用装饰器语法创建 UDTF：

```python
from pyspark.sql.functions import lit, udtf

# Define a UDTF using the `udtf` decorator directly on the class.
@udtf(returnType="num: int, squared: int")
class SquareNumbers:
    def eval(self, start: int, end: int):
        for num in range(start, end + 1):
            yield (num, num * num)

# Invoke the UDTF in PySpark using the SquareNumbers class directly.
SquareNumbers(lit(1), lit(3)).show()
+---+-------+
|num|squared|
+---+-------+
|  1|      1|
|  2|      4|
|  3|      9|
+---+-------+
```

创建 UDTF 时，也可以设置参数`useArrow=True` 启用 Arrow 优化。

Python UDTF 也可以注册并在 SQL 查询中使用

```python
# Register the UDTF for use in Spark SQL.
spark.udtf.register("square_numbers", SquareNumbers)

# Using the UDTF in SQL.
spark.sql("SELECT * FROM square_numbers(1, 3)").show()
+---+-------+
|num|squared|
+---+-------+
|  1|      1|
|  2|      4|
|  3|      9|
+---+-------+
```
## 统计信息

|pyspark.sql|统计信息|
|:---|:---|
|`DataFrame.describe(*cols)`|描述性统计|
|`DataFrame.summary(*statistics)`|描述性统计|
|`DataFrame.count()`|行数|
|`DataFrame.agg(*exprs)`|聚合|
|`DataFrame.approxQuantile(col, prob, relativeError)`|百分位数|
|`DataFrame.corr(col1, col2, method=None)`|相关系数|
|`DataFrame.cov(col1, col2)`|方差|
| `DataFrame.freqItems(cols, support)`    | 收集频繁条目|
| `DataFrame.observe(observation, *exprs)` | 提取统计信息 |

```python
df = spark.createDataFrame([
    (2, "Alice"), 
    (3, "Alice"), 
    (5, "Bob"), 
    (10, "Bob")
], ["age", "name"])

df.count()
# 4

df.describe().show()
+-------+------------------+-----+
|summary|               age| name|
+-------+------------------+-----+
|  count|                 4|    4|
|   mean|               5.0| NULL|
| stddev|3.5590260840104366| NULL|
|    min|                 2|Alice|
|    max|                10|  Bob|
+-------+------------------+-----+

df.summary().show()
+-------+------------------+-----+
|summary|               age| name|
+-------+------------------+-----+
|  count|                 4|    4|
|   mean|               5.0| NULL|
| stddev|3.5590260840104366| NULL|
|    min|                 2|Alice|
|    25%|                 2| NULL|
|    50%|                 3| NULL|
|    75%|                 5| NULL|
|    max|                10|  Bob|
+-------+------------------+-----+
```

```python
from pyspark.sql.functions import col, count, lit, max
from pyspark.sql import Observation

observation = Observation("my metrics")
observed_df = df.observe(observation, count(lit(1)).alias("count"), max(col("age")))
observed_df.show()
+---+-----+
|age| name|
+---+-----+
|  2|Alice|
|  3|Alice|
|  5|  Bob|
| 10|  Bob|
+---+-----+
observation.get
# {'count': 4, 'max(age)': 10}
```



## 缺失值处理

pyspark.sql|缺失值处理
:---|:---
`DataFrame.na.fill(value, subset=None)`|缺失值填充
`DataFrame.na.drop(how='any', thresh=None, subset=None)`|缺失值删除
`DataFrame.na.replace(to_teplace, value, subset=None)`|替换
| `Column.eqNullSafe(other)` | 安全处理NULL          |
| `Column.isNotNull()`       | `IS NOT NULL`  in SQL |
| `Column.isNull()`          | `IS NULL`  in SQL     |
| `functions.isnan(col)`     | `IS NaN`              |
| `functions.isnull(col)`    | `IS NULL`             |

```python
df = spark.createDataFrame([
    (None, "Michael"),
    (30, "Andy"),
    (19, "Justin")
], schema=["age", "name"])

df.na.fill(25).show()
df.na.fill({'age':25, 'name':'unknow'}).show()
+---+-------+
|age|   name|
+---+-------+
| 25|Michael|
| 30|   Andy|
| 19| Justin|
+---+-------+

df.na.drop().show()
+---+------+
|age|  name|
+---+------+
| 30|  Andy|
| 19|Justin|
+---+------+df.filter(df.age.isNotNull()).show()
+---+-------+                                                                   
|age|   name|
+---+-------+
| 30|   Andy|
| 19| Justin|
+---+-------+

df.select(
    df["age"].eqNullSafe(30),
    df["age"].eqNullSafe(None),
).show()
+------------+--------------+
|(age <=> 30)|(age <=> NULL)|
+------------+--------------+
|       false|          true|
|        true|         false|
|       false|         false|
+------------+--------------+

df.filter(df.age.isNotNull()).show()
+---+-------+                                                                   
|age|   name|
+---+-------+
| 30|   Andy|
| 19| Justin|
+---+-------+

df.select(
    df["age"].eqNullSafe(30),
    df["age"].eqNullSafe(None),
).show()
+------------+--------------+
|(age <=> 30)|(age <=> NULL)|
+------------+--------------+
|       false|          true|
|        true|         false|
|       false|         false|
+------------+--------------+
```

## 临时视图

### 临时视图

DataFrame和Spark SQL共享相同的执行引擎，因此，您可以将DataFrame注册为临时视图，并轻松运行SQL查询：

```python
df = spark.createDataFrame([
    (29, "Michael"),
    (30, "Andy"),
    (19, "Justin")
], schema=["age", "name"])

# Register the DataFrame as a SQL temporary view
df.createOrReplaceTempView("tableA")

# SQL can be run over DataFrames that have been registered as a table.
spark.sql("SELECT count(*) from tableA").show()
+--------+
|count(1)|
+--------+
|       3|
+--------+
```

这些SQL表达式可以直接用作DataFrame列：

```python
from pyspark.sql.functions import expr

df.select(expr('count(*)') > 0).show()
+--------------+
|(count(1) > 0)|
+--------------+
|          true|
+--------------+
```

### 全局临时视图

Spark SQL中的临时视图是会话范围的，如果创建它的会话终止，它将消失。如果您想拥有一个在所有会话之间共享的临时视图，并保持活动状态，直到Spark应用程序终止，您可以创建一个全局临时视图。全局临时视图与系统保留的数据库`global_temp`绑定，我们必须使用严格的字段名来引用它。

```python
# Register the DataFrame as a global temporary view
df.createOrReplaceGlobalTempView("people")

# Global temporary view is tied to a system preserved database `global_temp`
spark.sql("SELECT * FROM global_temp.people").show()
+---+-------+                                                                   
|age|   name|
+---+-------+
| 29|Michael|
| 30|   Andy|
| 19| Justin|
+---+-------+

# Global temporary view is cross-session
spark.newSession().sql("SELECT * FROM global_temp.people").show()
+---+-------+                                                                   
|age|   name|
+---+-------+
| 29|Michael|
| 30|   Andy|
| 19| Justin|
+---+-------+
```

## 缓存数据

|pyspark.sql|缓存|
|:---|:---|
|`DataFrame.cache()`|使用默认等级MEMORY_AND_DISK_DESER缓存|
|`DataFrame.persist(storageLevel)`|缓存|
|`DataFrame.unpersist()`|删除缓存|

DataFrame 同样有变换和行动两种操作，变换操作也是惰性的，不立即计算其结果。

当DataFrame在未来的行动中重复利用时，我们可以使用`persist()`或`cache()`方法将标记为持久。缓存是迭代算法和快速交互使用的关键工具。

第一次计算后，它将保存在节点的内存中。Spark的缓存是容错的如果，任何分区丢失，它将使用最初创建它的转换自动重新计算。

```python
# Persists the data in the disk by specifying the storage level.
from pyspark.storagelevel import StorageLevel
df.persist(StorageLevel.DISK_ONLY)
```

**注意：**在Python中，存储的对象将始终使用Pickle库进行序列化，因此您是否选择序列化级别并不重要。Python中的可用存储级别包括`MEMORY_ONLY`、`MEMORY_ONLY_2`、`MEMORY_AND_DISK`、`MEMORY_AND_DISK_2`、`DISK_ONLY`、`DISK_ONLY_2`和`DISK_ONLY_3`。

# Pandas API on Spark

pandas-on-Spark DataFrame和pandas DataFrame相似，提供了pandas Dataframe的几乎所有属性和方法。然而，前者是分布式计算，后者是单机计算。

[pandas_on_spark user guide](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html)

```python
# Customarily, we import pandas API on Spark as follows:
import pyspark.pandas as ps
```

## 创建 DataFrame

类似于创建 Pandas DataFrame一样创建 pandas-on-Spark DataFrame

```python
s = ps.Series([1, 3, 5, None, 6, 8])
print(s)
# 0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0

psdf = ps.DataFrame({
    'id': [1, 2, 3],
    'age': [29, 30, 19],
    'name': ['Michael', 'Andy', 'Justin']
    }, index=['a', 'b', 'c'])
print(psdf)
#    id  age     name
# a   1   29  Michael
# b   2   30     Andy
# c   3   19   Justin
```

从 Pandas DataFrame转化为 pandas-on-Spark DataFrame

```python
import pandas as pd

pdf = pd.DataFrame({
    'id': [1, 2, 3],
    'age': [29, 30, 19],
    'name': ['Michael', 'Andy', 'Justin']
    }, index=['a', 'b', 'c'])

# Now, this pandas DataFrame can be converted to a pandas-on-Spark DataFrame
psdf = ps.from_pandas(pdf) 
```

从 Spark DataFrame转化为 pandas-on-Spark DataFrame

```python
sdf = spark.createDataFrame(pdf) # Spark DataFrame
psdf = sdf.pandas_api(index_col="id")
print(psdf)
#     age     name
# id              
# 1    29  Michael
# 2    30     Andy
# 3    19   Justin
```

请注意，当从Spark DataFrame创建pandas-on-Spark DataFrame时，会创建一个新的默认索引。为了避免这种开销，请尽可能指定要用作索引的列。

| pyspark.pandas.DataFrame |                        |
| ------------------------ | ---------------------- |
| `DataFrame.to_pandas()`  | 转化为Pandas DataFrame |
| `DataFrame.to_spark()` | 转化为Spark DataFrame |
|`DataFrame.spark.frame([index_col])`     |  获取Spark DataFrame |

请注意，将pandas-on-Spark DataFrame转换为pandas DataFrame时，需要将所有数据收集到客户端机器中，数据太大时，容易引发内存错误。

## Input/Output

pandas-on-Spark 不仅支持 Pandas IO，还完全支持Spark IO：

```python
# CSV
psdf.to_csv('csvFile.csv')
ps.read_csv('csvFile.csv')
# Parquet
psdf.to_parquet('parquetFile.parquet')
ps.read_parquet('parquetFile.parquet')
# Spark IO
psdf.to_spark_io('orcFile.orc', format="orc")
ps.read_spark_io('orcFile.orc', format="orc")
```
