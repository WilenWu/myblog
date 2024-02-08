---
title: 大数据手册(Spark)--Spark安装配置
categories:
  - Big Data
  - Spark
tags:
  - 大数据
  - Spark
cover: /img/spark-install.jpg
top_img: /img/apache-spark-top-img.svg
abbrlink: d02a6da3
date: '2024-01-15 22:00:00'
description:
---

> 本文默认在 zsh 终端安装配置，若使用bash终端，环境变量的配置文件相应变化。
> 若安装包下载缓慢，可复制链接到迅雷下载，亲测极速～

# 准备工作

Spark的安装过程较为简单，在已安装好 Hadoop 的前提下，经过简单配置即可使用。

假设已经安装好了 hadoop （伪分布式）和 hive ，环境变量如下

```sh
JAVA_HOME=/usr/opt/jdk
HADOOP_HOME=/usr/local/hadoop
HIVE_HOME=/usr/local/hive
```

# 安装 spark

搭建spark不需要hadoop，如果已经有Hadoop集群，可下载相应的版本。

下载并解压安装包

```bash
tar zxvf spark-3.5.0-bin-hadoop3.tar -C /usr/local/
```

配置环境变量  `vi ~/.zshrc`

```sh
# 定义spark_home并把路径加到path参数中
SPARK_HOME=/usr/local/spark-3.5.0-bin-hadoop3
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
# 定义Scala_home目录，不使用scala语言时无需进行配置
SCALA_HOME=/usr/local/scala-2.10
export PATH=$SCALA_HOME/bin:$PATH
```

创建软连接，方便操作路径

```sh
ln -s /usr/local/spark-3.5.0-bin-hadoop3 spark
```

# Local 模式配置

要先确保已配置hadoop环境变量 `vi ~/.zshrc`

```sh
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
```

通过运行Spark自带的示例，验证Spark是否安装成功。执行时会输出非常多的运行信息，输出结果不容易找到，可以通过 grep 命令进行过滤.

```sh
cd /usr/local/spark
bin/run-example SparkPi 2>&1 | grep "Pi is"
```

配置完成后选择一种编程环境启动交互式窗口

```sh
spark-shell    # 启动Scala shell
pyspark        # 启动python shell
sparkR         # 启动R shell
```

启动成功会出现

```sh
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

可在 web UI 查看spark状态 http://localhost:8080 

# 配置pyspark包

由于pyspark包不在python包管理路径下，在本地python环境中无法加载 pyspark，我们可以通过以下方法配置：

方法一：可以在python环境变量中增加包的安装路径，也能正常加载

```sh
# 将pyspark路径加载到python环境变量中
vi ~/.zshrc
# 然后将下面的语句添加到文件末尾
export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
# 重新加载
source ~/.zshrc
```

方法二：将 `$SPARK_HOME/python/lib` 下的 py4j 和 pyspark 两个压缩包解压后放到python的包安装路径下（一般为sīte-pages文件夹），则可在一般python环境中加载 pyspark

# 连接外置hive数仓

前面连接的是spark自带的hive数仓，并且在启动目录下自动生成了metastror_data 和 spark-warehouse。用spark自带的derby来存储元数据，在启动目录下自动生成 derby临时文件。

现在连接配置好的hive环境，复制hive配置文件 `/usr/local/hive/conf/hive-site.xml` 到 spark目录`/usr/local/spark/conf/` 。这样启动spark读取conf文件的时候，就会读取hive-site这个文件下的hive数仓了。

这里还需要将 mysql-connector-java-8.0.29.jar 包放到 spark/jars下面，用来连接访问hive元数据库的jdbc客户端。

然后重新启动spark，即可使用hive元数据，并访问hive数仓。对于独立运行程序，需要使用enableHiveSupport方法来启用Hive支持。启用Hive支持后，就可以在Spark中使用Hive的元数据、表和数据源。

# Pyspark on Jupyter

编辑配置文件 `vim ~/.zshrc`

```sh
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='notebook'
```

加载完成后，则可在 jupyter 中加载 pyspark

