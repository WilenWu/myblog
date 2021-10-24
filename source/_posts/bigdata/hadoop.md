---
title: Hadoop生态概述及常见报错
date: 2021-05-04 14:28:1
categories: [大数据]
tags: [大数据,hadoop,MapReduce,HDFS]
cover: /img/hadoop-cover-bigdata.png
top_img: /img/hadoop-logo.svg
description: Hadoop是一个开源框架来存储和处理大型数据在分布式环境中。它包含两个模块，一个是MapReduce，另外一个是Hadoop分布式文件系统（HDFS）。
---

# Hadoop

 Hadoop是一个开源框架来存储和处理大型数据在分布式环境中。它包含两个模块，一个是MapReduce，另外一个是Hadoop分布式文件系统（HDFS）。

 <!-- more -->

-  **MapReduce**：它是一种并行编程模型在大型集群普通硬件可用于处理大型结构化，半结构化和非结构化数据。
-  **HDFS**：Hadoop分布式文件系统是Hadoop的框架的一部分，用于存储和处理数据集。它提供了一个容错文件系统在普通硬件上运行。

 Hadoop生态系统包含了用于协助Hadoop的不同的子项目（工具）模块，如Sqoop, Pig 和 Hive。

-  **Sqoop**: 它是用来在HDFS和RDBMS之间来回导入和导出数据。
-  **Pig**: 它是用于开发MapReduce操作的脚本程序语言的平台。
-  **Hive**: 它是用来开发SQL类型脚本用于做MapReduce操作的平台。
  
   注：有多种方法来执行MapReduce作业：

-  传统的方法是使用Java MapReduce程序结构化，半结构化和非结构化数据。
-  针对MapReduce的脚本的方式，使用Pig来处理结构化和半结构化数据。
-  Hive查询语言（HiveQL或HQL）采用Hive为MapReduce的处理结构化数据。

![](https://gitee.com/WilenWu/images/raw/master/common/hadoop-sys.jpg)

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

![](https://gitee.com/WilenWu/images/raw/master/common/Data-Warehousing.png)

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



