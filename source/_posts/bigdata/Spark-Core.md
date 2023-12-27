---
title: 大数据手册(Spark)--PySpark Core
categories:
  - 'Big Data'
  - Spark
tags:
  - 大数据
  - Spark
  - python
  - RDD 
cover: /img/apache-spark-core.png
top_img: /img/apache-spark-top-img.svg
description: Spark的环境配置及RDD
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

## SparkContext

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

# RDD

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

# 共享变量

# 广播变量

Broadcast Variables