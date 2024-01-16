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





