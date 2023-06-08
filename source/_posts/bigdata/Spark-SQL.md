---
title: 大数据手册(Spark)--PySpark SQL
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
description:
---


# SparkSession

Spark SQL用于对结构化数据进行处理，它提供了DataFrame的抽象，作为分布式平台数据查询引擎，可以在此组件上构建大数据仓库。DataFrame是一个分布式数据集，在概念上类似于传统数据库的表结构，数据被组织成命名的列，DataFrame的数据源可以是结构化的数据文件，也可以是Hive中的表或外部数据库，也还可以是现有的RDD。
DataFrame的一个主要优点是，Spark引擎一开始就构建了一个逻辑执行计划，而且执行生成的代码是基于成本优化程序确定的物理计划。与Java或者Scala相比，Python中的RDD是非常慢的，而DataFrame的引入则使性能在各种语言中都保持稳定。

在过去，你可能会使用SparkConf、SparkContext、SQLContext和HiveContext来分别执行配置、Spark环境、SQL环境和Hive环境的各种Spark查询。SparkSession现在是读取数据、处理元数据、配置会话和管理集群资源的入口。SparkSession本质上是这些环境的组合，也包括StreamingContext。

```python
from pyspark.sql import SparkSession
spark=SparkSession \
   .builder \
   .appName('test') \
   .config('master','yarn') \
   .getOrCreate()
```
Spark 交互式环境下，默认已经创建了名为 spark 的 SparkSession 对象，不需要自行创建。

# DataFrame

## DataFrame创建

**从RDD创建DataFrame**

```python
# 推断schema
from pyspark.sql import Row

lines = sc.textFile("users.txt")
parts = lines.map(lambda l: l.split(","))
data = parts.map(lambda p: Row(name=p[0],age=p[1],city=p[2]))
df=createDataFrame(data)

# 指定schema
data = parts.map(lambda p: Row(name=p[0],age=int(p[1]),city=p[2]))
df=createDataFrame(data)

# StructType指定schema
from pyspark.sql.types import *
schema = StructType([
    StructField('name',StringType(),True),
    StructField('age',LongType(),True),
    StructField('city',StringType(),True)
    ])
df=createDataFrame(parts, schema)
```
StructField 包括以下方面的内容：

- name：字段名
- dataType：数据类型
- nullable：此字段的值是否为空

**从文件系统创建DataFrame**

```python
df = spark.read.json("customer.json")
df = spark.read.load("customer.json", format="json")
df = spark.read.load("users.parquet")
df = spark.read.text("users.txt")
```

## 输出和保存

```python
df.rdd # df转化为RDD
df.toJSON() # df转化为RDD字符串
df.toPandas() # df转化为pandas

df.write.save("customer.json", format="json")
df.write.save("users.parquet")

df.write.json("users.json")
df.write.text("users.txt")
```

## 数据库读写

```python
df = spark.sql('select name,age,city from users') 
df.createOrReplaceTempView(name) # 创建临时视图
df.write.saveAsTable(name,mode='overwrite',partitionBy=None)
```
**操作hive表**：`df.write` 有两种方法操作hive表

- `saveAsTable()`
  如果hive中不存在该表，则spark会自动创建此表匹。
  如果表已存在，则匹配插入数据和原表 schema(数据格式，分区等)，只要有区别就会报错
  若是分区表可以调用`partitionBy`指定分区，使用`mode`方法调整数据插入方式：
  
   > Specifies the behavior when data or table already exists. Options include:
   >  - `overwrite`: 覆盖原始数据(包括原表的格式，注释等)
   >  - `append`: 追加数据(需要严格匹配)
   >  - `ignore`: ignore the operation (i.e. no-op).
   >  - `error` or `errorifexists`: default option, throw an exception at runtime.
  
   `df.write.partitionBy('dt').mode('append').saveAsTable('tb2')`
  
- `insertInto()`
  无关schema，只按数据的顺序插入，如果原表不存在则会报错
  对于分区表，先==开启Hive动态分区==，则不需要指定分区字段，如果有一个分区，那么默认为数据中最后一列为分区字段，有两个分区则为最后两列为分区字段，以此类推
   ```
   sqlContext.setConf("hive.exec.dynamic.partition", "true")
   sqlContext.setConf("hive.exec.dynamic.partition.mode", "nonstrict")
   df.write.insertInto('tb2')
   ```
  
- 同样也可以先==开启Hive动态分区==，用SQL语句直接运行
  `sql("insert into tb2 select * from tb1")`

## DataFrame属性和方法


DataFrame信息|说明
:---|:---
`df.show(n)` |预览前 n 行数据
`df.collect()`|列表形式返回
`df.dtypes` |列名与数据类型
`df.head(n)` |返回前 n 行数据
`df.first()` |返回第 1 行数据
`df.take(n)` |返回前 n 行数据
`df.printSchema()` |打印模式信息
`df.columns` |列名


查询语句|说明
:---|:---
`df.select(*cols)`|`SELECT` in SQL
`df.union(other)`|`UNION ALL` in SQL
`df.when(condition,value)`|`CASE WHEN` in SQL
`df.alias(*alias,**kwargs)`|`as` in SQL
`F.cast(dataType)`|数据类型转换（函数）
`F.lit(col)`|常数列（函数）


```python
from pyspark.sql import functions as F

df.select('*')
df.select('name','age') # 字段名查询
df.select(['name','age']) # 字段列表查询
df.select(df['name'],df['age']+1) # 表达式查询

df.select('name',df.mobile.alias('phone')) # 重命名列
df.select('name','age',F.lit('2020').alias('update'))  # 常数

df.select('name',
          F.when(df.age > 100,100)
           .when(df.age < 0,-1)
           .otherwise(df.age)
          ).show()
          
from pyspark.sql.types import *
df.select('name',df.age.cast('float'))
df.select('name',df.age.cast(FloatType()))

# selectExpr接口支持并行计算
expr=['count({})'.format(i) for i in df.columns]
df.selectExpr(*expr).collect()[0]
```

排序|说明
:---|:---
`df.sort(*col,**kwargs)`|排序
`df.orderBy(*col,**kwargs)`|排序(用法同sort)

```python
df.sort(df.age.desc()).show()
df.sort('age',ascending=True).show()
df.sort(desc('age'),'name').show()
df.sort(['age','name'],ascending=[0,1]).show()
```

筛选方法|说明
:---|:---
`df.filter(condition)`|筛选
`column.isin(*cols)`| `in (...)`
`column.like(pattern)`|SQL通配符匹配
`column.rlike(pattern)`|正则表达式匹配
`column.startswith(pattern)`|匹配开始
`column.endswith(pattern)`|匹配结尾
`column.substr(start,length)`|截取字符串
`column.between(lower,upper)`|`between ... and ...`


```python
df.filter("age = 22").show()
df.filter(df.age == 22).show()

df.select(df['age'] == 22).show()
df.select(df.name.isin('Bill','Elon')).show()

df.filter("name like Elon%").show()
df.filter(df.name.rlike("Musk$").show()
```

统计信息|说明
:---|:---
`df.describe()`|描述性统计
`df.count()`|行数
`df.approxQuantile(col,prob,relativeError)`|百分位数
`df.corr(col1,col2,method=None)`|相关系数

```python
# 异常值处理
bounds = {}

for col in df.columns:
    quantiles = df.approxQuantile(col,[0.25,0.75],0.05)
    # 第三个参数relativeError代表可接受的错误程度，越小精度越高
    IQR = quantiles[1] - quantiles[0]
    
    bounds[col] = [quantiles[0]-1.5*IQR, quantiles[1]+1.5*IQR]
    # bounds保存了每个特征的上下限
```

分组和聚合|说明
:---|:---
`df.groupBy(*cols)`|分组，返回GroupedData
`groupedData.count()`|计数
`groupedData.sum(*cols)`|求和
`groupedData.avg(*cols)`|平均值
`groupedData.mean(*cols)`|平均值
`groupedData.max(*cols)`|最大值
`groupedData.min(*cols)`|最小值
`groupedData.agg(*exprs)`|应用表达式
> 聚合函数还包括 countDistinct, kurtosis, skewness, stddev, sumDistinct, variance 等

```python
df.groupBy('city').count().collect()
df.groupBy(df.city).avg('age').collect()
df.groupBy('city',df.age).count().collect()

df.groupBy('city').agg({'age':'mean'}).collect() # 字典形式给出
df.groupBy('city').agg({'*':'count'}).collect() 
df.groupBy('city').agg(F.mean(df.age)).collect() 
```

去重|说明
:---|:---
`df.distinct()`|唯一值
`df.dropDuplicates(subset=None)`|删除重复项


添加、修改、删除列|说明
:---|:---
`df.withColumnRenamed(existing,new)` |重命名
`df.withColumn(colname,new)` |修改列
`df.drop(*cols)` |删除列

```python
df=df.withColumn('age',df.age+1)
df=df.drop('age')
df=df.drop(df.age)
```

缺失值处理|说明
:---|:---
`df.na.fill(value,subset=None)`|缺失值填充
`df.na.drop(how='any',thresh=None,subset=None)`|缺失值删除
`df.na.replace(to_teplace,value,subset=None)`|替换

```python
df=df.na.fill(0)
df=df.na.fill({'age':50,'name':'unknow'})

df=df.na.drop()
df=df.na.replace(['Alice','Bob'],['A','B'],'name')
```

## 分区和缓存

分区和缓存|说明
:---|:---
`df.repartition(n)`  |将df拆分为10个分区
`df.coalesce(n)`  |将df合并为n个分区
`df.cache()`|缓存



参考链接：
- Spark 编程基础 - 厦门大学 | 林子雨
- [Spark基本架构及运行原理](https://blog.csdn.net/zxc123e/article/details/79912343)

- [Spark入门介绍(菜鸟必看)](https://blog.csdn.net/Joker992/article/details/50043349)
- [Spark 修炼之道](https://blog.csdn.net/lovehuangjiaju/category_9264349.html)
- [PySpark教程 | 编程字典](http://codingdict.com/article/8880)
- [SparkSQL（Spark-1.4.0)实战系列][sparksql]
- [Machine Learning On Spark][ml]

[sparksql]: https://blog.csdn.net/lovehuangjiaju/article/details/46900585
[ml]: https://blog.csdn.net/lovehuangjiaju/article/details/48297921


# Data Types

# Column

# Built-In Functions


