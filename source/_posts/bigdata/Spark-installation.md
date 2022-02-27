---
title: 大数据手册(Spark)--Spark安装配置
categories:
  - 大数据
tags:
  - 大数据
  - Spark
cover: /img/apache-spark-inst.png
top_img: /img/apache-spark-top-img.svg
abbrlink: d02a6da3
date: 2018-06-24 21:52:29
description:
---
Apache Spark 是专为大规模数据处理而设计的快速通用的计算引擎。

  Spark 是一种与 Hadoop 相似的开源集群计算环境，但是两者之间还存在一些不同之处，这些有用的不同之处使 Spark 在某些工作负载方面表现得更加优越，换句话说，Spark 启用了内存分布数据集，除了能够提供交互式查询外，它还可以优化迭代工作负载。

<!-- more -->

# 准备工作

- 安装spark [运行环境 jdk][jdk]
- 如果使用python API，安装运行环境 [python2或python3][py]
- 如果使用scala语言，安装运行环境（[官网链接][scala]）
- 安装spark服务器，配置[免密登陆][sc]

[jdk]: http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
[py]: https://www.python.org/getit/
[scala]: https://www.scala-lang.org/download/
[sc]: https://blog.csdn.net/qq_41518277/article/details/80720390
[spark]: http://spark.apache.org/downloads.html

# Spark下载和安装

- [官网下载spark文件][spark]（搭建spark不需要hadoop，如果已经有Hadoop集群，可下载相应的版本）
- 解压到指定目录
```bash
tar zxvf spark-2.3.1-bin-hadoop2.7.tgz -C /usr/local/
```
# 配置spark环境变量

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

# Spark配置文件

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

# 启动Spark集群

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

# 启动Shell界面

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

# Spark集群配置免密钥登陆

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


参考链接：[spark-2.2.0安装和部署——Spark集群学习日记](https://blog.csdn.net/weixin_36394852/article/details/76030317)



