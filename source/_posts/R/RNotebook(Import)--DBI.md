---
ID: a01f20a958206ca8e2e17f762a7fe048
title: R手册(Import)--DBI
tags: [R,database,DBI]
mathjax: false
copyright: true
date: 2018-05-01 17:42:54
categories: [R,Import]
sticky: false
---
# DBI: definition for communication between R and RDBMSs

DBI只使用 "front-end" API  ,通过调用”drivers”（其他包）与特定的 DBMSs (SQLite, MySQL, PostgreSQL, MonetDB, etc.) 通信 

<!-- more -->

函数|说明
:---|:---
dbConnect(drv, ...)<br>dbDisconnect(conn, ...)| 连接/断开DBMS
dbListTables(conn)| 列出远程表名称
dbListFields(conn, name)| 列出远程表字段名称
dbReadTable(conn, name)|  Reads a database table to a data frame
dbWriteTable(conn, name, value)| Writes, overwrites(覆盖) or appends(追加) a data frame to a database table
dbRemoveTable(conn, name)| 删除远程表
dbSendQuery(conn, statement)| 执行SQL语句查询

***for example:***
```r
con <- dbConnect(RSQLite::SQLite(), dbname = ":memory:")
# ":memory:"为内存数据库
```



