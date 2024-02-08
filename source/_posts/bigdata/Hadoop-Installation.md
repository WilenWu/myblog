---
title: 大数据手册(Hadoop)--Hadoop安装配置
categories:
  - Big Data
  - Hadoop
tags:
  - 大数据
  - hadoop
cover: /img/apache-hadoop-cover.png
top_img: /img/apache-hadoop-logo.svg
abbrlink: 250f35c4
date: '2024-01-15 22:00:00'
description:
---

> 本文默认在 zsh 终端安装配置，若使用bash终端，环境变量的配置文件相应变化。
> 若安装包下载缓慢，可复制链接到迅雷下载，亲测极速～

# 安装Java环境

下载并解压安装包
```sh
sudo tar -zxvf ./jdk-8u391-linux-x64.tar.gz -C /usr/local/
```

配置环境变量 `vi ~/.zshrc`

```sh
# set java environment
export JAVA_HOME=/usr/opt/jdk-1.8.0_391
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=$JAVA_HOME/bin:$PATH
```

# 配置SSH免密登陆

我们需要配置成SSH无密码登陆。利用 ssh-keygen 生成密钥，并将密钥加入到授权中

```sh
cd ~/.ssh/                     # 若没有该目录，请先执行一次ssh localhost
ssh-keygen -t rsa -C username@email.com  # 会有提示，都按回车就可以
cat ./id_rsa.pub >> ./authorized_keys  # 加入授权
```

然后可以使用如下命令验证

```sh
ssh localhost
```

# 安装 Hadoop

下载并解压安装包

```sh
sudo tar -zxvf ./hadoop-3.3.6.tar.gz -C /usr/local/
```

配置环境变量  `vi ~/.zshrc`

```sh
# set hadoop environment
export HADOOP_HOME=/usr/local/hadoop-3.3.6
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib/nativ"
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH
```

创建软连接，方便操作路径

```sh
ln -s /usr/local/hadoop-3.3.6 hadoop
```

输入如下命令来检查 Hadoop 是否可用，成功则会显示 Hadoop 版本信息：

```sh
hadoop version
```

# 伪分布式环境配置

Hadoop 可以在单节点上以伪分布式的方式运行，Hadoop 进程以分离的 Java 进程来运行，节点既作为 NameNode 也作为 DataNode，同时，读取的是 HDFS 中的文件。

Hadoop 的配置文件位于 etc/hadoop/ 中，伪分布式主要需要修改2个配置文件 **core-site.xml** 和 **hdfs-site.xml** 。Hadoop的配置文件是 xml 格式，每个配置以声明 property 的 name 和 value 的方式来实现。

修改配置文件 **core-site.xml** 

```xml
<configuration>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>file:/usr/local/hadoop/tmp</value>
        <description>A base for other temporary directories.</description>
    </property>
    <property>
        <name>fs.default.name</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
```

同样的，修改配置文件 **hdfs-site.xml**：

```xml
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
        <description>hdfs备份数</description>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:/usr/local/hadoop/tmp/dfs/name</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:/usr/local/hadoop/tmp/dfs/data</value>
    </property>
</configuration>
```

Hadoop 的运行方式是由配置文件决定的（运行 Hadoop 时会读取配置文件），因此如果需要从伪分布式模式切换回非分布式模式，需要删除 core-site.xml 中的配置项。

此外，伪分布式虽然只需要配置 fs.default.name 和 dfs.replication 就可以运行（官方教程如此），不过若没有配置 hadoop.tmp.dir 参数，则默认使用的临时目录为 /tmp/hadoo-hadoop，而这个目录在重启时有可能被系统清理掉，导致必须重新执行 format 才行。所以我们进行了设置，同时也指定 dfs.namenode.name.dir 和 dfs.datanode.data.dir，否则在接下来的步骤中可能会出错。

配置 **mapped-site.xml**

```xml
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
    <description>指定mapreduce运行在yarn上</description>
  </property>
</configuration>
```

配置 **yarn-site.xml**

```xml
<configuration>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
  <property>
    <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
    <value>org.apache.hadoop.mapred.ShuffleHandler</value>
  </property>
  <property>
    <name>yarn.resourcemanager.hostname</name>
    <value>localhost</value>
  </property>
  <property>
    <name>yarn.acl.enable</name>
    <value>0</value>
  </property>
```

当java路径无法识别时，可在 **hadoop-env.sh** 文件配置

```sh
export JAVA_HOME=/usr/opt/jdk-1.8.0_391
```

当datanode或resourcemanager无法正常启动时，很可能是hostname无法正常解析，可在 /etc/hosts 文件中添加hostname。例如，本机hostname为hadoop，则添加

```sh
127.0.0.1 localhost hadoop
```

配置完成后，执行 NameNode 的格式化，相当于一个文件系统的初始化

```sh
hdfs namenode -format
```

成功的话，会看到 "successfully formatted" 的提示。

接着启动 hadoop集群

```sh
$HADOOP_HOME/sbin/start-all.sh
```

启动完成后，可以通过命令 `jps` 来判断是否成功启动，若成功启动则会列出如下进程: 

```sh
15013 ResourceManager
14822 SecondaryNameNode
30103 Jps
14583 NameNode
15112 NodeManager
14686 DataNode
```

也可在 web UI 查看 http://localhost:9870 

#  分布式集群配置免密钥登陆

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

> 参考资料：
> [Hadoop3.1.3安装教程_单机/伪分布式配置_Hadoop3.1.3/Ubuntu18.04(16.04)](https://dblab.xmu.edu.cn/blog/2441/)
