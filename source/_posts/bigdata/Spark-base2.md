---
title: 大数据手册(Spark)--Spark 基础知识（二）
categories:
  - 大数据
  - Spark
tags:
  - 大数据
  - Spark
cover: /img/apache-spark-base2.png
top_img: /img/apache-spark-top-img.svg
description: false
abbrlink: 264c088
date: 2020-01-03 16:20:25
---

# Spark 初始化

## spark 交互式执行环境

```bash
spark-shell --master <master-url>  # scala
pyspark  --master <master-url> # python
```
下面介绍几种常用Spark应用程序提交方式
  - local：采用单线程运行spark，常用于本地开发测
  - local[n]：使用n线程运行spark
  - local[*]：逻辑CPU个数的线程运行
  - standalone：利用Spark自带的资源管理与调度器运行Spark集群，采用Master/Slave结构。
  - mesos ：运行在著名的Mesos资源管理框架基础之上，该集群运行模式将资源管理交给Mesos，Spark只负责进行任务调度和计算
  - yarn : 集群运行在Yarn资源管理器上，资源管理交给Yarn，Spark只负责进行任务调度和计算。


用户编写完Spark应用程序之后，需要将应用程序提交到集群中运行，提交时使用脚本spark-submit进行，spark-submit可以带多种参数

## 运行 spark 应用程序

```bash
spark-submit --master <master-url> code.py [*args]
```

## 环境初始化

默认情况下，Spark 交互式环境已经为 SparkContext 创建了名为 sc 的变量，因此创建新的环境变量将不起作用。但是，在独立spark 应用程序中，需要自行创建SparkContext 对象。
```python
from spark import SparkConf,SparkContext
conf=SparkConf()\
    .setMaster('yarn')\
    .setAppName('XXX')
spark=SparkContext(conf=conf)
```
> 一旦SparkConf对象被传递给SparkContext，它就不能被修改。

```python
conf.contains(key) # 配置中是否包含一个指定键。
conf.get(key,defaultValue=None) # 获取配置的某些键值，或者返回默认值。
conf.getAll() # 得到所有的键值对的list。
conf.set(key,value) # 设置配置属性

# 获取 SparkContext 信息
sc.version                 # 获取 SparkContext 版本
sc.pythonVer               # 获取 Python 版本
sc.master                  # 要连接的 Master URL
str(sc.sparkHome)          # Spark 在工作节点的安装路径
str(sc.sparkUser())        # 获取 SparkContext 的 Spark 用户名
sc.appName                 # 返回应用名称
sc.applicationId           # 获取应用程序ID
sc.defaultParallelism      # 返回默认并行级别
sc.defaultMinPartitions    # RDD默认最小分区数

sc.stop()  # 终止SparkContext
```

# 弹性分布式数据集 (RDD)

## RDD创建

```python
# 从并行集合创建
pairRDD=sc.parallelize([('a',7),('a',2),('b',2)]) # key-value对RDD
rdd1=sc.parallelize([1,2,3,4,5])
rdd2=sc.parallelize(range(100))
```

mapReduce|说明
:---|:---
`rdd.map(func)`|将函数应用于RDD中的每个元素并返回
`rdd.mapValues(func)`|不改变key，只对value执行map
`rdd.flatMap(func)`|先map后扁平化返回
`rdd.flatMapValues(func)`|不改变key，只对value执行flatMap
`rdd.reduce(func)`|合并RDD的元素返回
`rdd.reduceByKey(func)`|合并每个key的value
`rdd.foreach(func)`|用迭代的方法将函数应用于每个元素
`rdd.keyBy(func)`|执行函数于每个元素创建key-value对RDD

```python
>>> rdd1.map(lambda x:x+1).collect()
[2,3,4,5,6]
>>> rdd1.reduce(lambda x,y : x+y)
15
>>> rdd1.keyBy(lambda x:x%2).collect()
[(1,1),(0,2),(1,3),(0,4),(1,5)]
>>> pairRDD.mapValues(lambda x:x+1).collect()
[('a',8),('a',3),('b',3)]
>>> pairRDD.reduceByKey(lambda x,y : x+y).collect()
[('a',9),('b',2)]

>>> names=sc.parallelize(['Elon Musk','Bill Gates','Jim Green'])
>>> names.map(lambda x:x.split(' ')).collect()
[('Elon','Musk'),('Bill','Gates'),('Jim','Green')]
>>> names.flatMap(lambda x:x.split(' ')).collect()
['Elon','Musk','Bill','Gates','Jim','Green']
```

## RDD属性和方法

提取|说明
:---|:---
`rdd.collect()`|将RDD以列表形式返回
`rdd.collectAsMap()`|将RDD以字典形式返回
`rdd.take(n)`|提取前n个元素
`rdd.takeSample(replace,n,seed)`|随机提取n个元素
`rdd.first()`|提取第1名
`rdd.top(n)`|提取前n名
`rdd.keys()`|返回RDD的keys
`rdd.values()`|返回RDD的values
`rdd.isEmpty()`|检查RDD是否为空


```python
>>> pairRDD.collectAsMap()
{'a': 2,'b': 2}
>>> pairRDD.keys().collect()
['a','a','b']
>>> pairRDD.values().collect()
[7,2,2]
```


分组和聚合|说明
:---|:---
`rdd.groupBy(func)`|将RDD元素通过函数变换分组为key-iterable集
`rdd.groupByKey()`|将key-value元素集分组为key-iterable集
`rdd.aggregate(zeroValue,seqOp,combOp)`|
`rdd.aggregateByKey(zeroValue,seqOp,combOp)`|
`rdd.fold(zeroValue,func)`|
`rdd.foldByKey(zeroValue,func)`|


```python
>>> rdd1.groupBy(lambda x: x % 2).mapValues(list).collect()
[(0,[2,4]),(1,[1,3,5])]
>>> pairRDD.groupByKey().mapValues(list).collect()
[('a',[7,2]),('b',[2])]
```

| 统计                 | 说明                                             |
| :------------------- | :----------------------------------------------- |
| `rdd.count()`        | 返回RDD中的元素数                                |
| `rdd.countByKey()`   | 按key计算RDD元素数量                             |
| `rdd.countByValue()` | 按RDD元素计算数量                                |
| `rdd.sum()`          | 求和                                             |
| `rdd.mean()`         | 平均值                                           |
| `rdd.max()`          | 最大值                                           |
| `rdd.min()`          | 最小值                                           |
| `rdd.stdev()`        | 标准差                                           |
| `rdd.variance()`     | 方差                                             |
| `rdd.histograme()`   | 分箱（Bin）生成直方图                            |
| `rdd.stats()`        | 综合统计（计数、平均值、标准差、最大值和最小值） |


```python
>>> pairRDD.count()
3
>>> pairRDD.countByKey()
defaultdict(<type 'int'>,{'a':2,'b':1})
>>> pairRDD.countByValue()
defaultdict(<type 'int'>,{('b',2):1,('a',2):1,('a',7):1})
>>> rdd2.histogram(3)
([0,33,66,99],[33,33,34])
```


选择数据|说明
:---|:---
`rdd.sample(replace,frac,seed)`|抽样
`rdd.filter(func)`|筛选满足函数的元素(变换)
`rdd.distinct()`|去重

```python
>>> rdd2.sample(False,0.8,seed=42)
>>> rdd1.filter(lambda x:x%2==0).collect()
[2,4]
```

排序|说明
:---|:---
`rdd.sortBy(func,ascending=True)`|按RDD元素变换后的值排序
`rdd.sortByKey(ascending=True)`|按key排序


连接运算|说明
:---|:---
`rdd.union(other)`|并集(不去重)
`rdd.intersection(other)`|交集(去重)
`rdd.subtract(other)`|差集(不去重)
`rdd.cartesian(other)`|笛卡尔积
`rdd.subtractByKey(other)`|按key差集
`rdd.join(other)`|内连接
`rdd.leftOuterJoin(other)`|左连接
`rdd.rightOuterJoin(other)`|右连接

```python
>>> rdd1=sc.parallelize([1,1,3,5])
>>> rdd2=sc.parallelize([1,3])
>>> rdd1.union(rdd1).collect()
[1,1,3,5,1,1,3,5]
>>> rdd1.intersection(rdd2).collect()
[1,3]
>>> rdd1.subtract(rdd2).collect()
[5]
>>> rdd2.cartesian(rdd2).collect()
[(1,1),(1,3),(3,1),(3,3)]

>>> rdd1=sc.parallelize([('a',7),('a',2),('b',2)])
>>> rdd2=sc.parallelize([('b','B'),('c','C')])
>>> rdd1.subtractByKey(rdd2).collect()
[('a',7),('a',2)]
>>> rdd1.join(rdd2).collect()
[('b',(2,'B'))]
```

## 分区和缓存

持久化|说明
:---|:---
`rdd.persist()`|标记为持久化
`rdd.cache()`|等价于`rdd.persist(MEMORY_ONLY)`
`rdd.unpersist()`|释放缓存


分区|说明
:---|:---
`rdd.getNumPartitions()`|获取RDD分区数
`rdd.repartition(n)`|新建一个含n个分区的RDD
`rdd.coalesce(n)`|将RDD中的分区减至n个
`rdd.partitionBy(key,func)`|自定义分区

## 文件系统读写

```python
# 读取
rdd=sc.textFile('hdfs://file_path')  # 从hdfs集群读取
rdd=sc.textFile('file_path') 
rdd=sc.textFile('file:///local_file_path') # 从本地文件读取

# 保存
rdd.saveAsTextFile('hdfs://file_path')
rdd.saveAsTextFile('file_path') # hdfs路径
rdd.saveAsTextFile('file:///local_file_path')
```

# DataFrame

## Spark SQL

Spark SQL用于对结构化数据进行处理，它提供了DataFrame的抽象，作为分布式平台数据查询引擎，可以在此组件上构建大数据仓库。DataFrame是一个分布式数据集，在概念上类似于传统数据库的表结构，数据被组织成命名的列，DataFrame的数据源可以是结构化的数据文件，也可以是Hive中的表或外部数据库，也还可以是现有的RDD。
DataFrame的一个主要优点是，Spark引擎一开始就构建了一个逻辑执行计划，而且执行生成的代码是基于成本优化程序确定的物理计划。与Java或者Scala相比，Python中的RDD是非常慢的，而DataFrame的引入则使性能在各种语言中都保持稳定。

## 初始化

在过去，你可能会使用SparkConf、SparkContext、SQLContext和HiveContext来分别执行配置、Spark环境、SQL环境和Hive环境的各种Spark查询。SparkSession现在是读取数据、处理元数据、配置会话和管理集群资源的入口。SparkSession本质上是这些环境的组合，包括StreamingContext。

```python
from pyspark.sql import SparkSession
spark=SparkSession \
   .builder \
   .appName('test') \
   .config('master','yarn') \
   .getOrCreate()
```
Spark 交互式环境下，默认已经创建了名为 spark 的 SparkSession 对象，不需要自行创建。

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





