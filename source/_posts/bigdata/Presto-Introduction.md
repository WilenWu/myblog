---
title: 大数据手册(Presto)--Presto 简介
categories:
  - Big Data
  - Hadoop
tags:
  - 大数据
  - Presto
cover: /img/apache-presto.svg
top_img: /img/apache-hadoop-logo.svg
abbrlink: b33c6284
date: 2022-08-13 18:18:00
description:
---

# Presto 简介

Presto（或 PrestoDB）是一种开源的分布式 SQL 查询引擎，从头开始设计用于针对任何规模的数据进行快速分析查询。它既可支持非关系数据源，例如 Hadoop 分布式文件系统 (HDFS)、Amazon S3、Cassandra、MongoDB 和 HBase，又可支持关系数据源，例如 MySQL、PostgreSQL、Amazon Redshift、Microsoft SQL Server 和 Teradata。

Presto 可在数据的存储位置查询数据，无需将数据移动到独立的分析系统。查询执行可在纯粹基于内存的架构上平行运行，大多数结果在几秒内即可返回。您将会发现，它已被许多知名公司采用，例如 Facebook、Airbnb、Netflix、Atlassian 和 Nasdaq。

# Presto 的发展历史

Presto 最初作为 Facebook 的项目启动，针对 300PB 的数据仓库运行交互式分析查询，使用大型基于 Hadoop/HDFS 的集群构建。在构建 Presto 之前，Facebook 使用的是 2008 年创建并推出的 Apache Hive，为 Hadoop 生态系统带来熟悉的 SQL 语法。Hive 在将复杂的 Java MapReduce 作业简化成类似 SQL 的查询方面对 Hadoop 生态系统有着重大影响，同时还能够执行大规模的任务。但是，它未针对交互式查询所需的高速性能进行优化。

在 2012 年，Facebook 数据基础设施组构建了 Presto，这种交互式查询系统能够以 PB 级规模快速运行。它于 2013 年春季在全公司范围内推广。2013 年 11 月，Facebook 将 Presto 作为 Apache 软件许可证下的开源软件，任何人都可以从 Github 上下载。今天，Presto 已成为在 Hadoop 上进行交互式查询的流行选择，获得了来自 Facebook 和其他组织的大量贡献。Facebook 的 Presto 实施的使用者超过一千名员工，他们每天运行超过 30000 次查询，处理的数据达到 1PB。

# Presto 工作原理

Presto 是在 Hadoop 上运行的分布式系统，使用与经典大规模并行处理 (MPP) 数据库管理系统相似的架构。它有一个协调器节点，与多个工作线程节点同步工作。用户将其 SQL 查询提交给协调器，由其使用自定义查询和执行引擎进行解析、计划并将分布式查询计划安排到工作线程节点之间。它设计用于支持标准 ANSI SQL 语义，包括复杂查询、聚合、联接、左/右外联接、子查询、开窗函数、不重复计数和近似百分位数。

查询编译之后，Presto 将请求处理到工作线程节点之间的多个阶段中。所有处理都在内存中进行，并以流水线方式经过网络中的不同阶段，从而避免不必要的 I/O 开销。添加更多工作线程节点可提高并行能力，并加快处理速度。

为了使 Presto 可扩展到任何数据源，它的设计采用了存储抽象化，以便于轻松地构建可插入的连接器。因此，Presto 拥有大量连接器，既可用于非关系数据源，例如 Hadoop 分布式文件系统 (HDFS)、Amazon S3、Cassandra、MongoDB 和 HBase，又可用于关系源，例如 MySQL、PostgreSQL、Amazon Redshift、Microsoft SQL Server 和 Teradata。数据在其存储位置接受查询，无需将其移动到独立的分析系统中。 

# Presto 和 Hadoop

Presto 是一种开源分布式 SQL 查询引擎，设计用于对 HDFS 和其他源中的数据进行快速交互式查询。与 Hadoop/HDFS 不同，它没有自己的存储系统。因此，Presto 与 Hadoop 互补，有些机构同时使用这两种产品来解决更广泛的业务挑战。Presto 可以与 Hadoop 的任何实施一起安装，并封装在 Amazon EMR Hadoop 分发中。