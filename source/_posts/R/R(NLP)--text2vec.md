---
ID: b6502175e60473c83b94636761ca6f6a
title: R手册(NLP)--text2vec
tags: [R,NLP]
date: 2018-05-28 18:44:11
categories: [R,NLP]
description: 这个 R 包提供了高性能和简洁的 API 来进行文本分析、自然语言处理
---

[**text2vec**][t2v] 这个 R 包提供了高性能和简洁的 API 来进行文本分析、自然语言处理。

[t2v]: https://cndocr.github.io/text2vec-doc-cn/index.html

<!-- more -->

# 分词器

`word_tokenizer(strings)` 英语分词器
`jieba<-jiebaR::worker() `中文分词器

```r
#RUN THIS EXAMPLE:
jieba<-jiebaR::worker()
tok_fun <-function(strings) 
  llply(strings, segment, jieba)
```

# I/O 处理(迭代器)：支持`create_<type>`函数

```r
#RUN THIS EXAMPLE:
itoken(strings, 
 preprocessor = identity,       #预处理函数集（去空格，去数字等）
 tokenizer = space_tokenizer,   #分词器
 progressbar = interactive())   #进度条
it_train <- itoken(doc, tokenizer = tok_fun)
```

# 向量化

- **创建词汇表：**(N-grams参数)

```r
#RUN THIS EXAMPLE:
stop_words <- c("在", "又", "你" )       #停止词创建
vocab <- create_vocabulary(it_train, stopwords = stop_words)
```

- **修剪词汇：**

```r
prune_vocabulary(vocabulary,  #词汇表
 term_count_min = 1L,         #最小次数
 term_count_max = Inf, 
 doc_proportion_min = 0,      #最小比例
 doc_proportion_max = 1,
 max_number_of_terms = Inf)
```

```r
#RUN THIS EXAMPLE:
pruned_vocab <- prune_vocabulary(vocab,term_count_min = 10,doc_proportion_max = 0.5,doc_proportion_min = 0.001)
```

- **词汇向量化：**`vocab_vectorizer()` , `hash_vectorizer()`

`vectorizer <- vocab_vectorizer(pruned_vocab)`

- **DTM,TCM**: (Document-Term matrices, Term co-occurence matrices)

`dtm_train <- create_dtm(it_train, vectorizer)`
`tcm_train <- create_tcm(it_train, vectorizer)`

# 主题模型

- **处理模型的统一规范**

`model$new(...)`生成一个模型对象，设置初始化参数。
`model$fit(x, ...)`拟合模型
`model$fit_transform(x, ...)`拟合模型并转换数据
`model$transform(x_new, ...)`使用已经训练好的模型转换数据

Tf-idf 转换，Global Vectors (GloVe) 词向量

- **常用模型**

`LSA$new()`潜在语义分析(LSA)
`LDA$new()`潜在Dirichlet分配模型(LDA)

- **文档相似性和距离**（不相似性）

`sim2(x, y, method)`矩阵x和y，每一行的使用指定方法的相似性
`dist2(x, y, method)`矩阵x和y，每一行的使用指定方法的距离
`psim2(x, y, method)`,  `dist2(x, y, method)`并行计算

> 参数method："cosine", "euclidean", "jaccard"
> 余弦距离，欧式距离，Jaccard距离，RelaxedWordMover’sDistance

