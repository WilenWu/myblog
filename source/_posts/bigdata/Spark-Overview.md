---
title: 大数据手册(Spark)--Spark 简介
categories:
  - 'Big Data'
  - Spark
tags:
  - 大数据
  - Spark
cover: /img/apache-spark-overview.png
top_img: /img/apache-spark-top-img.svg
abbrlink: 32722c50
date: 2020-01-03 16:10:18
description:
---

# Spark 简介

Apache Spark 是一种用于大数据工作负载的分布式开源处理系统。它使用内存中缓存和优化的查询执行方式，可针对任何规模的数据进行快速分析查询。Apache Spark 提供了简明、一致的 Java、Scala、Python 和 R 应用程序编程接口 (API)。

Apache Spark 是专为大规模数据处理而设计的快速通用的计算引擎。Spark 拥有Hadoop MapReduce所具有的优点，但不同的是Job中间输出结果可以保存在内存中，从而不再需要读写HDFS，因此Spark能更好地适用于数据挖掘与机器学习等需要迭代的MapReduce的算法。

# Spark 安装配置 

## 准备工作

- 安装spark [运行环境 jdk][jdk]
- 如果使用python API，安装运行环境 [python2或python3][py]
- 如果使用scala语言，安装运行环境（[官网链接][scala]）
- 安装spark服务器，配置[免密登陆][sc]

[jdk]: http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
[py]: https://www.python.org/getit/
[scala]: https://www.scala-lang.org/download/
[sc]: https://blog.csdn.net/qq_41518277/article/details/80720390
[spark]: http://spark.apache.org/downloads.html

## Spark下载和安装

- [官网下载spark文件][spark]（搭建spark不需要hadoop，如果已经有Hadoop集群，可下载相应的版本）
- 解压到指定目录
```bash
tar zxvf spark-2.3.1-bin-hadoop2.7.tgz -C /usr/local/
```

## 配置spark环境变量

```bash
vi /etc/profile

# 定义spark_home并把路径加到path参数中
SPARK_HOME=/usr/local/spark-2.3.1-bin-hadoop2.7
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

# 定义Scala_home目录，不使用scala语言时无需进行配置
SCALA_HOME=/usr/local/scala-2.10
export PATH=$SCALA_HOME/bin:$PATH

#加载配置文件，可在任意位置启动pyspark,spark-shell,sparkR
source /etc/profile
```

## Spark配置文件

```bash
#--------切换到spark配置目录-------
cd /usr/local/spark-2.3.1-bin-hadoop2.7/conf/

#--------配置文件spark-enc.sh--------
mv spark-env.sh.template spark-env.sh  #重命名文件
vi spark-env.sh   
# 在文件末尾添加环境，保存并退出
export JAVA_HOME=/usr/local/jdk1.8.0_171  #指定jdk位置
export SPARK_MASTER_IP=master       #master主机IP（单机为localhost,ifconfig命令查看本机IP）
export SPARK_MASTER_PORT=7077       #master主机端口
# 使用Scala语言时配置
export SCALA_HOME=/usr/local/scala-2.10
# 已有Hadoop集群时配置
export HADOOP_HOME=/usr/hadoop/hadoop-2.7.3
export HADOOP_CONF_DIR=/usr/hadoop/hadoop-2.7.3/etc/hadoop

#--------配置文件slaves--------
mv slaves.template slaves  #重命名文件
vi slaves    
#在该文件中添加子节点的IP或主机名（worker节点），保存并退出
node01
node02

#------将配置好的spark拷贝到其他节点上------
scp -r spark-2.3.1-bin-hadoop2.7/ node01:/usr/
scp -r spark-2.3.1-bin-hadoop2.7/ node02:/usr/
```

## 启动Spark集群

spark集群配置完毕，目前是1个master，2个worker
```bash
#----------在master上启动集群------------
cd /usr/local/spark-2.3.1-bin-hadoop2.7/sbin/
bash start-all.sh     #或者bash start-master.sh + bash start-slaves.sh
#----------查看进程-----------------
jps
#----------查看集群状态----------
master任意浏览器登陆：http://master:8080/
```

## 启动Shell界面

```bash
cd /usr/local/spark-2.3.1-bin-hadoop2.7/bin/

#-------------选择一种编程环境启动-----------------
bash spark-shell    # 启动Scala shell
bash pyspark        # 启动python shell
bash sparkR         # 启动R shell
#启动时若Java版本报错，安装需要的版本即可
sudo apt-get install openjdk-8-jdk

# 启动成功会出现
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ '/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.3.1
      /_/

Using Scala version 2.10.4 (Java HotSpot(TM) 64-Bit Server VM, Java 1.7.0_71)
Type in expressions to have them evaluated.
Type :help for more information.
….
18/06/17 23:17:53 INFO BlockManagerMaster: Registered BlockManager
18/06/17 23:17:53 INFO SparkILoop: Created spark context..
Spark context available as sc.
```

## Spark集群配置免密钥登陆

```bash
#在每个节点生成公钥文件
cd ~/.ssh                      #进到当前用户的隐藏目录
ssh-keygen -t rsa              #用rsa生成密钥，一路回车
cp id_rsa.pub authorized_keys   #把公钥复制一份，并改名为authorized_keys
#这步执行完后，在当前机器执行ssh localhost可以无密码登录本机了

#每个节点的公钥文件复制到master节点
scp authorized_keys master@master：~/download/note01_keys   #重命名公钥便于整合
scp authorized_keys master@master：~/download/note01_keys
... ...

 #进入master节点，整合公钥文件分发到所有节点覆盖
cat ~/download/note01_keys >> ~/.ssh/authorized_keys
cat ~/download/note02_keys >> ~/.ssh/authorized_keys
... ...
scp ~/.ssh/authorized_keys node01@node01：~/.ssh/authorized_keys
scp ~/.ssh/authorized_keys node02@node02：~/.ssh/authorized_keys
... ...

#在每个节点更改公钥的权限
chmod 600 authorized_keys
```

> 参考链接：
> [spark-2.2.0安装和部署——Spark集群学习日记](https://blog.csdn.net/weixin_36394852/article/details/76030317)
> [Apache Spark Installation on Windows](https://sparkbyexamples.com/spark/apache-spark-installation-on-windows/)
> [How to Install PySpark on Mac](https://sparkbyexamples.com/pyspark/how-to-install-pyspark-on-mac/)


# Spark 框架

Spark 框架包括：

- Spark Core 是该平台的基础。它要负责内存管理、故障恢复、计划安排、分配与监控作业，以及和存储系统进行交互。
- Spark SQL 用于交互查询和结构化数据处理
- Spark Streaming 用于进行实时流数据的处理
- Spark MLlib 用于分布式环境下的机器学习
- Spark GraphX 用于分布式图形处理

![what-is-apache-spark](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/what-is-apache-spark.b3a3099296936df595d9a7d3610f1a77ff0749df.PNG)

# Spark 基本架构

一个完整的Spark应用程序(Application)，在提交集群运行时，它涉及到如下图所示的组件。
Spark 一般包括一个主节点（任务控制节点）和多个从节点（工作节点），每个任务(Job)会被切分成多个阶段(Stage)，每个阶段并发多线程执行，结束后返回到主节点。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/spark/spark-application.png)

- Driver Program：（主节点或任务控制节点）执行应用程序主函数并创建SparkContext对象，SparkContext配置Spark应用程序的运行环境，并负责与不同种类的集群资源管理器通信，进行资源申请、任务的分配和监控等。当Executor部分运行完毕后，Driver同时负责将SparkContext关闭。
- Cluster Manager：（集群资源管理器）指的是在集群上进行资源（CPU，内存，宽带等）调度和管理。可以使用Spark自身，Hadoop YARN，Mesos等不同的集群管理方式。
- Worker Node：从节点或工作节点。
- Executor：每个工作节点上都会驻留一个Executor进程，每个进程会派生出若干线程，每个线程都会去执行相关任务。
- Task：（任务）运行在Executor上的工作单元。


# Spark运行基本流程

RDD(Resilient Distributed Dataset)是Spark框架中的核心概念，它们是在多个节点上运行和操作以在集群上进行并行处理的元素。
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

# 弹性分布式数据集(RDD)

在Spark里，对数据的所有操作，基本上就是围绕RDD来的，譬如创建、转换、求值等等。某种意义上来说，RDD变换操作是惰性的，因为它们不立即计算其结果，RDD的转换操作会生成新的RDD，新的RDD的数据依赖于原来的RDD的数据，每个RDD又包含多个分区。那么一段程序实际上就构造了一个由相互依赖的多个RDD组成的有向无环图(DAG)。并通过在RDD上执行行动将这个有向无环图作为一个Job提交给Spark执行。
该延迟执行会产生更多精细查询：DAGScheduler可以在查询中执行优化，包括能够避免shuffle数据。

RDD支持两种类型的操作：
-   **变换**(Transformation) ：调用一个变换方法应用于RDD，不会有任何求值计算，返回一个新的RDD。
-   **行动**(Action)  ：它指示Spark执行计算并将结果返回。

**窄依赖与宽依赖**

在前面讲的Spark编程模型当中，我们对RDD中的常用transformation与action 函数进行了讲解，我们提到RDD经过transformation操作后会生成新的RDD，前一个RDD与tranformation操作后的RDD构成了lineage关系，也即后一个RDD与前一个RDD存在一定的依赖关系，根据tranformation操作后RDD与父RDD中的分区对应关系，可以将依赖分为两种：
- **窄依赖**(narrow dependency)：变换操作后的RDD仅依赖于父RDD的固定分区，则它们是窄依赖的。
- **宽依赖**(wide dependency)：变换后的RDD的分区与父RDD所有的分区都有依赖关系（即存在shuffle过程，需要大量的节点传送数据），此时它们就是宽依赖的。

如下图所示：
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/spark/spark-dependency.png)
图中的实线空心矩形代表一个RDD，实线空心矩形中的带阴影的小矩形表示分区(partition)。从上图中可以看到， map,filter,union等transformation是窄依赖；而groupByKey是宽依赖；join操作存在两种情况，如果分区仅仅依赖于父RDD的某一分区，则是窄依赖的，否则就是宽依赖。

**优化**：fork/join

宽依赖需要进行shuffle过程，需要大量的节点传送数据，无法进行优化；而所有窄依赖则不需要进行I/O传输，可以优化执行。
当RDD触发相应的action操作后，DAGScheduler会根据程序中的transformation类型构造相应的DAG并生成相应的stage，所有窄依赖构成一个stage，而单个宽依赖会生成相应的stage。




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





