---
title: 数据仓库和数据湖简介
date: '2023-12-27 21:22'
categories:
  - Big Data
  - Hadoop
tags:
  - 大数据
cover: /img/apache-hadoop-cover.png
top_img: /img/apache-hadoop-logo.svg
abbrlink: b557dff1
emoji: white_large_square
description:
---

# 数据仓库

数据仓库(Data Warehousing, DW) 的本质，其实就是整合多个数据源的历史数据进行细粒度的、多维的分析，帮助高层管理者或者业务分析人员做出商业战略决策或商业报表。这里面就涉及到了数据仓库的分层方法：
1. 数据运营层：ODS（Operational Data Store）：存放原始数据，直接加载原始日志、数据，数据保存原貌不做处理。
2. 数据仓库层：DW（Data Warehouse）是我们在做数据仓库时要核心设计的一层，在这里，从 ODS 层中获得的数据按照主题建立各种数据模型。DW层又细分为 DWD层、DWM（Data WareHouse Middle）层和DWS层。
   - 数据明细层：DWD（Data Warehouse Detail）：结构与粒度与原始表保持一致，对ODS层数据进行ETL清洗
   - 数据服务层：DWS（Data WareHouse Servce）：以DWD为基础，进行轻度汇总
3. 数据应用层: ADS（Application Data Servce）：主要是提供给数据产品和数据分析使用的数据
4. 维表层（Dimension）

# 如何搭建数据仓库

1. 分析业务需求，确定数据仓库主题
2. 构建逻辑模型：明确需求目标、维度、指标、方法、源数据等
3. 逻辑模型转换为物理模型：事实表表名，包括列名、数据类型、是否是空值以及长度等
4. ETL(Extract-Transform-Load)，即数据抽取、转换、装载的过程。Kettle就是强大的ETL工具。
5. OLAP(Online analytical processing)，即联机分析处理，主要用于支持企业决策管理分析，报表设计，数据展示。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/Data-Warehousing.png)

# 数据湖简介

数据湖(Data Lake)是一个集中式存储库，允许以任意规模存储所有结构化和非结构化数据。可以按原样存储数据，无需先对数据进行结构化处理。