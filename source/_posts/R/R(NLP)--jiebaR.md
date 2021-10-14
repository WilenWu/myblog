---
ID: 262471c6cc5c287729e86615692f08e5
title: R手册(NLP)--jiebaR
tags: [R,NLP,可视化]
date: 2018-05-04 17:43:15
categories: [R,NLP]
---

R语言词云和中文词典包。

<!-- more -->

# jiebaR

[jiebaR](https://qinwenfeng.com/jiebaR/): for Chinese text segmentation

## Quick Start

```r
library(jiebaR)
cutter = worker()   #新建分词器
segment( "This is a good day!" , cutter )  #分词
## [1] "This" "is"   "a"    "good" "day"
```

## `worker()`：初始化分词器

**1. worker函数**
```r
worker(type = "mix",    #type指定返回词的结果类型
 dict = DICTPATH,       #dict系统词典路径
 hmm = HMMPATH,
 user = USERPATH,       #user用户词典
 idf = IDFPATH,         #IDF词典，关键词提取使用
 stop_word = STOPPATH,  #stop_word停止词库，默认空值
 qmax = 20, 
 topn = 5,              #topn关键词数
 encoding = "UTF-8",    #输入文件编码
 detect = T, 
 symbol = F,            #是否保留符号
 lines = 1e+05,         #每次读取文件的最大行数，对于大文件，实现分次读取。
 write = T,            #是否将分词结果写入文件
 output = NULL,         #输出路径
 bylines = F))          #按行输出
```

**参数**

- **type**：指定分词引擎
mp（最大概率模型）- 基于词典和词频
hmm（HMM模型）- 基于 HMM 模型，可以发现词典中没有的词
mix（混合模型）- 先用 mp 分，mp 分完调用 hmm 再来把剩余的可能成词的单字分出来。
query（索引模型）- mix 基础上，对大于一定长度的词再进行一次切分。
**tag**（标记模型）- 词性标记，基于词典的
**keywords**（关键词模型）- tf-idf 抽关键词
**simhash**（Simhash 模型） - 在关键词的基础上计算 simhash
- **user**：用户词典路径
1.用户词典，包括词、词性标记两列。用户词典中的所有词的词频均为系统词典中的最大词频 (默认，可以通过 user_weight 参数修改)。
2.可以下载使用搜狗细胞词库，`user=cidian::load_user_dict(filePath="词库路径")`
- **user_weight**：用户词典中的词的词频
{"max", "min", "median"}，默认为 “max”，系统词典中的最大值。
- **idf**：IDF词典，关键词提取时用
- **stop_word**：关键词提取使用的停止词库。分词时也可以使用，但是分词时使用的对应路径不能为默认的 jiebaR::STOPPATH。



**2. 更新分词器**
```r
cutter=worker() #初始化分词器
cutter$bylines=TRUE #分行输出（接收向量分词后，分解到list中输出）
cutter$symbol = FALSE #重设为不保留符号
```

**3. 添加新词到已存在的分词器中**
```r
new_user_word(worker, words, tags = rep("n", length(words))) 
```

## `segment()`：分词

```r
segment(code, jiebar, mod = NULL)
```
**参数：**

- code：中文文档
- jiebar：分词器
- mod：指定返回分词的结果类型(mp/hmm/mix/query)

## 标记

`tagging(code, jiebar)`分词+标记
`vector_tag(string, jiebar)`对已经分好的词进行标记
```r
words = "我爱北京天安门"
cutter = worker()
result = segment(words, cutter)

tagger = worker("tag") #标记器
vector_tag(result, tagger)  #对分好的词进行标记
#>        r        v       ns       ns 
#>     "我"     "爱"   "北京" "天安门"
```

## 关键词提取：基于系统词典词频

`keywords(code, jiebar)`提取关键词
`vector_keywords(segment, jiebar)`
```r
key = worker("keywords", topn = 1)
keywords("我爱北京天安门", key)
#>   8.9954 
#> "天安门"

# 对已经分好词的文本提取关键词
cutter = worker()
result = segment("我爱北京天安门", cutter)
vector_keywords(result, key)
#>   8.9954 
#> "天安门"
```


## Simhash 与距离

```r
simhasher = worker("simhash", topn=2)
simhash("江州市长江大桥参加了长江大桥的通车仪式",  simhasher)
distance("hello world!", "江州市长江大桥参加了长江大桥的通车仪式",  simhasher) #分词+计算距离

vector_simhash(c("今天","天气","真的","十分","不错","的","感觉"), simhasher)
vector_distance(c("今天","天气","真的","十分","不错","的","感觉"),
  c("今天","天气","真的","十分","不错","的","感觉"), simhasher) #计算距离
```

## 词频统计

`freq(x)`x为分词后的结果

# cidian

cidian: Tools for Chinese Text Segmentation Dictionaries。用来转换搜狗细胞词库
Windows 安装 RTools，设置好对应的环境变量

- **安装**
```r
library(devtools)
install_github("qinwf/cidian")
```

- **使用**
```r
decode_scel(scel = "细胞词库路径",output = "输出文件路径")
```

- **读取词典和编辑词典文件**
```r
load_user_dict(filePath = "用户词典路径", default_tag = "默认标记")## 读取用户词典
load_sys_dict(filePath = "系统词典路径")## 读取系统词典

add_user_words(dict = "load_user_dict 读取的词典", words = "UTF-8 编码文本向量", tags = "标记")## 增加用户词典词
add_sys_words(dict = "load_sys_dict 读取的词典", words = "UTF-8 编码文本向量", freq = "词频", tags = "标记")## 增加系统词典词

remove_words(dict = "load_user_dict 或 load_sys_dict 读取的词典", words = "UTF-8 编码文本向量")## 删除词典词
write_dict(dict = "load_user_dict 或 load_sys_dict 读取的词典", output = "输出路径")## 写入

(userd = load_user_dict(jiebaR::USERPATH))
userd = add_user_words(userd, enc2utf8("测试"), "v")
write_dict(userd, jiebaR::USERPATH)

(userd = load_user_dict(jiebaR::USERPATH))
userd = remove_words(userd, enc2utf8(c("测试","蓝翔")))
write_dict(userd, jiebaR::USERPATH)
(userd = load_user_dict(jiebaR::USERPATH))
```



