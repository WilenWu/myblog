---
title: 大数据手册(Flink)--Flink安装配置
categories:
  - Big Data
  - Flink
tags:
  - 大数据
  - Flink
cover: /img/Apache-Flink-header.svg
top_img: /img/Apache-logo-text.svg
abbrlink: 7ff308e2
date: '2024-01-17 23:30:00'
description:
---

> 本文默认在 zsh 终端安装配置，若使用bash终端，环境变量的配置文件相应变化。
> 若安装包下载缓慢，可复制链接到迅雷下载，亲测极速～

# 安装 flink

Flink的运行需要Java环境的支持，因此，在安装Flink之前，请先参照相关资料安装Java环境

假设已经安装好了相关大数据组件，环境变量如下

```sh
JAVA_HOME=/usr/opt/jdk
```

下载并解压安装包

```bash
tar zxvf flink-1.17.2-bin-scala_2.12.tar -C /usr/local/
```

配置环境变量  `vi ~/.zshrc`

```sh
FLINK_HOME=/usr/local/flink-1.17.2
export PATH=$FLINK_HOME/bin:$PATH
```

创建软连接，方便操作路径

```sh
ln -s /usr/local/flink-1.17.2 flink
```

# YARN运行模式

在flink任务部署在YARN机群之前要先确保已配置hadoop环境变量 `vi ~/.zshrc`

```sh
HADOOP_HOME=/usr/local/hadoop
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
export HADOOP_CLASSPATH=`hadoop classpath`
```

会话模式启动

```sh
./bin/yarn-session.sh
```





```sh
yarn-session.sh --help
```

