---
title: 大数据手册(Hive)--Hive安装配置
categories:
  - Big Data
  - Hive
tags:
  - 大数据
  - hive
cover: /img/apache-hive-circ.jpg
top_img: /img/apache-hive-bg.png
abbrlink: dd512a75
date: '2024-01-15 22:00:00'
description:
---

> 本文默认在 zsh 终端安装配置，若使用bash终端，环境变量的配置文件相应变化。
> 若安装包下载缓慢，可复制链接到迅雷下载，亲测极速～

# 准备工作

在安装Hive之前首先安装好了hadoop，环境变量如下

```sh
JAVA_HOME=/usr/opt/jdk
HADOOP_HOME=/usr/local/hadoop
```

# 安装 hive

下载并解压安装包

```bash
tar zxvf apache-hive-3.1.3-bin.tar -C /usr/local/
```

配置环境变量  `vi ~/.zshrc`

```sh
HIVE_HOME=/usr/local/apache-hive-3.1.3-bin
export PATH=$HIVE_HOME/bin:$PATH
```

创建软连接，方便操作路径

```sh
ln -s /usr/local/apache-hive-3.1.3-bin hive
```

# 配置文件

在 `/usr/local/hive/conf` 目录创建配置文件 hive-site.xml，添加如下信息

```xml
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
    <property>
        <name>hive.metastore.local</name>
        <value>true</value>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionURL</name>
        <value>jdbc:mysql://localhost:3306/metastore?createDatabaseIfNotExist=true</value>
        <description>JDBC connect string for a JDBC metastore</description>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionDriverName</name>
        <value>com.mysql.jdbc.Driver</value>
        <description>Driver class name for a JDBC metastore</description>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionUserName</name>
        <value>root</value>
        <description>username to use against metastore database</description>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionPassword</name>
        <value>12345678</value>
        <description>password to use against metastore database</description>
    </property>
    <property>
        <name>hive.metastore.warehouse.dir</name>
        <value>/hive/warehouse</value>
        <description>Hive默认的hdfs工作目录，存储数据</description>
    </property>
    <property>
        <name>hive.cli.print.header</name>
        <value>true</value>
        <description>永久显示字段名</description>
    </property>
    <property>
        <name>hive.resultset.use.unique.column.names</name>
        <value>false</value>
        <description>不显示表名</description>
    </property>
</configuration>
```

# 安装并配置mysql

这里我们采用MySQL数据库保存Hive的元数据，而不是采用Hive自带的derby来存储元数据。

下载并安装MySQL。本文设置账户为root，密码为12345678.

配置环境变量  `vi ~/.zshrc`

```sh
export MYSQL_HOME=/usr/local/mysql
export PATH=$MYSQL_HOME/bin:$PATH
```

启动并登陆mysql

```sh
 service mysql start # 启动mysql服务
 mysql -u root -p  # 登陆shell界面
 
service mysql stop # 结束mysql服务
service mysql restart  # 重启mysql服务
ps -e | grep mysql # 检查是否启动
```

新建metastore数据库，用来保存hive元数据。这个数据库与hive-site.xml localhost:3306/metastore 的metastore对应。

```sql
mysql> create database metastore;
```

下载并解压 mysql-connector-java-8.0.29.tar 包，将文件 mysql-connector-java-8.0.29.jar拷贝到 /usr/local/hive/lib目录下。

# 启动 hive

启动hive之前，请先启动hadoop集群

```sh
$ /usr/local/hadoop/sbin/start-all.sh
$ hive
```

启动成功后会出现hive标识

```sh
hive> 
```

# HIVE常见报错

- [HIVE启动报错：Exception in thread "main" java.lang.NoSuchMethodError](https://www.cnblogs.com/jaysonteng/p/13412763.html)
- [Hive中运行任务报错：Error during job, obtaining debugging information...](https://blog.csdn.net/qq_41428711/article/details/86169029)
- [hive shell 方向键、退格键不能使用：使用rlwrap包装，并在用户配置文件重名民配置](https://blog.csdn.net/weixin_34050519/article/details/92353909)

## HIVE 中文乱码

Windows中Mysql的配置文件名字叫my.ini，linux 中配置文件路径为 `/usr/my.cnf`，mac中通常无配置文件。

以mac为例，先停止mysql服务

```sh
brew services stop mysql
```

查看mysql读取配置文件的默认顺序

```sh
mysqld --help --verbose | more
```

查看帮助，下翻，会看到表示配置文件默认读取顺序，如下：

```
Default options are read from the following files in the given order:
/etc/my.cnf /etc/mysql/my.cnf /usr/local/etc/my.cnf ~/.my.cnf
```

通常，这些位置上没有配置文件，所以需要创建文件，默认的样例文件 my-default.cnf 可自行百度

```sh
touch /etc/my.cnf
sudo vim /etc/my.cnf
```

在配置文件 `[mysqld]` 后添加以下内容

```
[mysqld]
init_connect='SET collation_connection = utf8_unicode_ci' 
init_connect='SET NAMES utf8'
character-set-server=utf8 
collation-server=utf8_unicode_ci
skip-character-set-client-handshake 
```

修改完成后，需要重新启动mysql服务

```sh
brew services start mysql
```

 然后，验证是否修改成功，进入mysql中，输入命令：`show variables like 'char%';`

```
+--------------------------+--------------------+
| Variable_name            | Value              |
+--------------------------+--------------------+
| character_set_client     | utf8mb4            |
| character_set_connection | utf8mb4            |
| character_set_database   | utf8mb4            |
| character_set_filesystem | binary             |
| character_set_results    | utf8mb4            |
| character_set_server     | utf8mb4            |
| character_set_system     | utf8mb3            |
| character_sets_dir       | /usr/local/mysql-8.0.35-macos13-arm64/share/charsets/ |
+--------------------------+-----------------------+
```

最后，修改mysql中hive元数据的编码：

```sql
-- 切换到元数据库
use metastore;

-- 1. 修改表字段和表注解
alter table COLUMNS_V2 modify column COMMENT varchar(256) character set utf8;
alter table TABLE_PARAMS modify column PARAM_VALUE varchar(4000) character set utf8;

-- 2. 修改分区字段注解
alter table PARTITION_PARAMS modify column PARAM_VALUE varchar(4000) character set utf8;
alter table PARTITION_KEYS modify column PKEY_COMMENT varchar(4000) character set utf8;

-- 3. 修改索引注解
alter table INDEX_PARAMS modify column PARAM_VALUE varchar(4000) character set utf8;
```

最后重启mysql服务：

```sh
brew services restart mysql
```

重启hive客户端，删掉之前的表结构，再新建表结构。
