---
title: Hadoop生态概述及常见报错
date: '2021-05-04 14:28:1'
categories:
  - 'Big Data'
  - Hadoop
tags:
  - 大数据
  - hadoop
  - MapReduce
  - HDFS
cover: /img/apache-hadoop-cover.png
top_img: /img/apache-hadoop-logo.svg
description: 
abbrlink: c5386d49
emoji: white_large_square
---

# Hadoop

Apache Hadoop 是一种开源框架，用于高效存储和处理从 GB 级到 PB 级的大型数据集。利用 Hadoop，您可以将多台计算机组成集群以便更快地并行分析海量数据集，而不是使用一台大型计算机来存储和处理数据。

Hadoop 由四个主要模块组成：

- Hadoop 分布式文件系统 (HDFS)—一个在标准或低端硬件上运行的分布式文件系统。除了更高容错和原生支持大型数据集，HDFS 还提供比传统文件系统更出色的数据吞吐量。

- Yet Another Resource Negotiator (YARN)—管理与监控集群节点和资源使用情况。它会对作业和任务进行安排。

- MapReduce—一个帮助计划对数据运行并行计算的框架。该 Map 任务会提取输入数据，转换成能采用键值对形式对其进行计算的数据集。Reduce 任务会使用 Map 任务的输出来对输出进行汇总，并提供所需的结果。

- Hadoop Common—提供可在所有模块上使用的常见 Java 库。

![hadoop_cluster](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/hadoop_cluster.png)

## Hadoop 如何运作

Hadoop 让利用集群服务器中的全部存储和处理能力，针对大量数据执行分布式处理变得更简单。Hadoop 提供构建基块，然后在其上方构建其他服务和应用程序。

要收集各种格式数据的应用程序可以通过 API 操作连接到 NameNode，以便将数据放置到 Hadoop 集群当中。对于在 DataNodes 上重复的每个文件的“组块”，NameNode 会对它们的文件目录结构和位置进行追踪。要运行任务来查询数据，提供一个由众多 Map 和 Reduce 任务组成的 MapReduce 作业，而这些任务针对分散在 DataNodes 的 HDFS 中的数据运行。Map 任务在每个节点上针对提供的输入文件运行，而 Reduce 任务则会运行以汇总与整理最终的输出。

由于它的可延展性，Hadoop 生态系统多年来经历了迅猛发展。现在，Hadoop 生态系统包含众多工具和应用程序，可用来帮助收集、存储、处理、分析和管理大数据。部分最受欢迎的应用程序包括：

- **Spark**—一款常用于大数据工作负载的分布式开源处理系统。Apache Spark 利用内存中缓存和经过优化的执行方式以实现高速性能，并支持常规批处理、流式分析、机器学习、图形数据库和临时查询。

- **Presto**—一种开源的分布式 SQL 查询引擎，针对低延迟的临时数据分析进行了优化。它支持 ANSI SQL 标准，包括复杂查询、聚合、连接和窗口函数。Presto 可处理来自多个数据源（包括 Hadoop 分布式文件系统 [HDFS] 和 Amazon S3）的数据。

- **Hive**—允许用户通过 SQL 界面使用 Hadoop MapReduce，从而实现大规模分析，以及分布式和容错数据仓储。

- **HBase**—一种在 Amazon S3（使用 EMRFS）或 Hadoop 分布式文件系统 (HDFS) 顶部运行的开源、非关系、版本控制数据库。HBase 是一种可大规模扩展的分布式大数据存储，专门为随机、严格一致性地实时访问具有数十亿行和数百万列的表而定制。

- **Zeppelin**—一种可实现交互式数据探索的交互式笔记本。

![hadoop_core_components](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/hadoop_core_components.png)

# 数据仓库

数据仓库(Data Warehousing, DW) 的本质，其实就是整合多个数据源的历史数据进行细粒度的、多维的分析，帮助高层管理者或者业务分析人员做出商业战略决策或商业报表。这里面就涉及到了[数据仓库的分层方法](https://www.cnblogs.com/itboys/p/10592871.html)：
- 数据运营层：ODS（Operational Data Store）：存放原始数据，直接加载原始日志、数据，数据保存原貌不做处理。
- 数据仓库层：DW（Data Warehouse）
   数据仓库层是我们在做数据仓库时要核心设计的一层，在这里，从 ODS 层中获得的数据按照主题建立各种数据模型。DW层又细分为 DWD（Data Warehouse Detail）层、DWM（Data WareHouse Middle）层和DWS（Data WareHouse Servce）层。
   1. 数据明细层：DWD（Data Warehouse Detail）：结构与粒度与原始表保持一致，对ODS层数据进行ETL清洗
   2. 数据服务层：DWS（Data WareHouse Servce）：以DWD为基础，进行轻度汇总
- 数据应用层: ADS（Application Data Servce）：主要是提供给数据产品和数据分析使用的数据
- 维表层（Dimension）

如何搭建数据仓库
1、 分析业务需求，确定数据仓库主题
2、 构建逻辑模型：明确需求目标、维度、指标、方法、源数据等
3、 逻辑模型转换为物理模型：事实表表名，包括列名、数据类型、是否是空值以及长度等
4、 ETL过程
5、 OLAP建模，报表设计，数据展示
OLAP(Online analytical processing)，即联机分析处理，主要用于支持企业决策管理分析
ETL（Extract-Transform-Load的缩写，即数据抽取、转换、装载的过程），Kettle就是强大的ETL工具。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/Data-Warehousing.png)

# Hadoop安装配置
- [mysql-connector-java-2.0.14.tar.gz 下载地址](http://ftp.ntu.edu.tw/MySQL/Downloads/Connector-J/)
- [Apache Software 下载地址](https://downloads.apache.org/)
- [hadoop 分布式文件系统的部署](https://blog.csdn.net/weixin_54720351/article/details/116088193)
- [使用WSL配置hadoop伪分布式环境](http://www.zyiz.net/tech/detail-123110.html)
- [ssh: connect to host localhost port 22: Connection refused](https://blog.csdn.net/hxc2101/article/details/113617870)
- [linux下Source /etc/profile不生效](https://blog.csdn.net/qq_39341048/article/details/89381061?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control)


# HIVE安装配置

- [WSL Hive3.x安装与调试](https://bbs.huaweicloud.com/blogs/197920)
- [WSL安装mysql流程和坑](https://blog.csdn.net/a35100535/article/details/113250441?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161984444216780271518910%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161984444216780271518910&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-113250441.pc_search_result_hbase_insert&utm_term=wsl%E5%AE%89%E8%A3%85mysql%E6%B5%81%E7%A8%8B%E5%92%8C%E5%9D%91)
- [HIVE启动报错：Exception in thread "main" java.lang.NoSuchMethodError](https://www.cnblogs.com/jaysonteng/p/13412763.html)
- [解决Hive中文乱码](https://segmentfault.com/a/1190000021105525)
- [Hive中运行任务报错：Error during job, obtaining debugging information...](https://blog.csdn.net/qq_41428711/article/details/86169029)
- [hive shell查询时永久显示字段名或显示头(永久生效，不代表名，3种方案)](https://blog.csdn.net/myhes/article/details/90582389)
- [hive shell 方向键、退格键不能使用：使用rlwrap包装，并在用户配置文件重名民配置](https://blog.csdn.net/weixin_34050519/article/details/92353909)
- [spark连接非内置hive数仓，spark连接外部hive数仓的方法](https://www.cnblogs.com/markecc121/p/11650402.html)



