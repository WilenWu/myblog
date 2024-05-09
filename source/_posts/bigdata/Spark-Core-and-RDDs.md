---
title: 大数据手册(Spark)--Spark Core and RDDs
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
description: Spark的核心RDD
abbrlink: 264c088
date: 2020-01-03 16:20:25
---

# Spark Core

## 独立应用程序

用户编写完Spark应用程序之后，需要将应用程序提交到集群中运行，提交时使用spark-submit命令进行：

```bash
spark-submit --master "local[2]" examples/src/main/python/pi.py
```

下面介绍几种spark常用的资源管理器 `master`

  - local：采用单线程运行spark，常用于本地开发测
  - local[n]：使用n线程运行spark
  - local[*]：逻辑CPU个数的线程运行
  - standalone：利用Spark自带的资源管理与调度器运行Spark集群，采用Master/Slave结构
  - yarn : 集群运行在Yarn资源管理器上，资源管理交给Yarn，Spark只负责进行任务调度和计算

## 使用 Spark Shell 进行交互

若要在 Python 解释器中以交互方式运行 Spark，请使用：`bin/pyspark`

```bash
pyspark  --master "local[2]"
```
## SparkContext

Spark Connect 是 Spark 3.4 中引入的一种新的客户端-服务器体系结构，用于解耦 Spark 客户端应用程序，并允许远程连接到 Spark 群集。

在 Spark 3.4 中，Spark Connect 在 PySpark 和 Scala 中的 DataFrame/Dataset API 支持。

在使用bin/pyspark命令打开Spark交互式环境后，默认情况下，Spark 已经为 SparkContext 创建了名为 sc 的变量，因此创建新的环境变量将不起作用。

但是，在提交的独立spark 应用程序中或者常规的python环境，需要自行创建SparkContext 对象连接集群。

```python
from pyspark import SparkConf,SparkContext
conf = SparkConf()\
    .setMaster('yarn')\
    .setAppName('myApp')
spark = SparkContext(conf=conf)
```
一旦SparkConf对象被传递给SparkContext，它就不能被修改。

```python
conf.contains(key) # 配置中是否包含一个指定键。
conf.get(key, defaultValue=None) # 获取配置的某些键值，或者返回默认值。
conf.getAll() # 得到所有的键值对的list。
conf.set(key, value) # 设置配置属性

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

Spark运行基本流程

弹性分布式数据集(RDD, Resilient Distributed Dataset)是Spark框架中的核心概念，它们是在多个节点上运行和操作以在集群上进行并行处理的元素。

Spark通过分析各个RDD的依赖关系生成有向无环图DAG(Directed Acyclic Graph)，通过分析各个RDD中的分区之间的依赖关系来决定如何划分Stage进行任务优化。

spark-submit提交Spark应用程序后，其执行流程如下：

1. 创建SparkContext对象，然后SparkContext会向Clutser Manager（例如Yarn、Standalone、Mesos等）申请资源。
2. 资源管理器在worker node上创建executor并分配资源（CPU、内存等)
3. SparkContext启动DAGScheduler，将提交的作业（Job）转换成若干Stage，各Stage构成DAG（Directed Acyclic Graph有向无环图），各个Stage包含若干相task，这些task的集合被称为TaskSet
4. TaskSet发送给TaskSet Scheduler，TaskSet Scheduler将Task发送给对应的Executor，同时SparkContext将应用程序代码发送到Executor，从而启动任务的执行
5. Executor执行Task，完成后释放相应的资源。
   ![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/spark/spark-submit.png)

- Job：一个Job包含多个RDD及作用于相应RDD上的各种操作构成的DAG图。
- Stages：是Job的基本调度单位(DAGScheduler)，一个Job会分解为多组Stage，每组Stage包含多组任务(Task)，称为TaskSet，代表一组关联的，相互之间没有Shuffle依赖关系(最耗费资源)的任务组成的任务集。
- Tasks：负责Stage的任务分发(TaskScheduler)，Task分发遵循基本原则：计算向数据靠拢，避免不必要的磁盘I/O开销。

在Spark里，对数据的所有操作，基本上就是围绕RDD来的，譬如创建、转换、求值等等。某种意义上来说，RDD变换操作是惰性的，因为它们不立即计算其结果，RDD的转换操作会生成新的RDD，新的RDD的数据依赖于原来的RDD的数据，每个RDD又包含多个分区。那么一段程序实际上就构造了一个由相互依赖的多个RDD组成的有向无环图(DAG)。并通过在RDD上执行行动将这个有向无环图作为一个Job提交给Spark执行。

该延迟执行会产生更多精细查询：DAGScheduler可以在查询中执行优化，包括能够避免shuffle数据。RDD支持两种类型的操作：

-   **变换**(Transformation) ：调用一个变换方法应用于RDD，不会有任何求值计算，返回一个新的RDD。
-   **行动**(Action)  ：它指示Spark执行计算并将结果返回。

> 请注意，在 Spark 2.0 之前，Spark 的主要编程接口是弹性分布式数据集 （RDD）。在 Spark 2.0 之后，RDD 被 Dataset 取代，后者具有与 RDD 类似的强类型，但在后台进行了更丰富的优化。

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

## 窄依赖与宽依赖

在前面讲的Spark编程模型当中，我们对RDD中的常用transformation与action 函数进行了讲解，我们提到RDD经过transformation操作后会生成新的RDD，前一个RDD与tranformation操作后的RDD构成了lineage关系，也即后一个RDD与前一个RDD存在一定的依赖关系，根据tranformation操作后RDD与父RDD中的分区对应关系，可以将依赖分为两种：

- **窄依赖**(narrow dependency)：变换操作后的RDD仅依赖于父RDD的固定分区，则它们是窄依赖的。
- **宽依赖**(wide dependency)：变换后的RDD的分区与父RDD所有的分区都有依赖关系（即存在shuffle过程，需要大量的节点传送数据），此时它们就是宽依赖的。

如下图所示：
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/spark/spark-dependency.png)
图中的实线空心矩形代表一个RDD，实线空心矩形中的带阴影的小矩形表示分区(partition)。从上图中可以看到， map,filter,union等transformation是窄依赖；而groupByKey是宽依赖；join操作存在两种情况，如果分区仅仅依赖于父RDD的某一分区，则是窄依赖的，否则就是宽依赖。

**优化**：fork/join

宽依赖需要进行shuffle过程，需要大量的节点传送数据，无法进行优化；而所有窄依赖则不需要进行I/O传输，可以优化执行。

当RDD触发相应的action操作后，DAGScheduler会根据程序中的transformation类型构造相应的DAG并生成相应的stage，所有窄依赖构成一个stage，而单个宽依赖会生成相应的stage。

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