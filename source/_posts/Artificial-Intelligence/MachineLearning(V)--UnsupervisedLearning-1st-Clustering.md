---
title: 机器学习(V)--无监督学习(一)聚类
date: 
katex: true
categories:
  - Artificial Intelligence
  - Machine Learning
tags:
  - 机器学习
cover: /img/ML-unsupervised-learning.png
top_img: /img/artificial-intelligence.jpg
abbrlink: 8c3d002c
description:
---


# 聚类

Clustering:  Group similar data points together.

聚类分析仅根据在数据中发现的描述对象及其关系的信息，将数据对象分组。其原理是：组内的对象之间是相似的(相关的)，而不同组中的对象是不同的(不相关的)。

KMeans、模糊C均值、EM聚类、Hierarchy、Kohonen聚类、视觉聚类、Canopy、幂迭代等

KMeans超参数：数据标准化（归一化、标准化、无处理）、聚类个数、收敛容差、最大迭代次数、初始化方法（random、KMeans+）、距离度量方法（欧几里得距离、曼哈顿距离、余弦夹角、相关系数）

 距离 
 euclidean 欧几里德距离
 maximum   切比雪夫距离
 manhattan 绝对值距离 
 canberra  Lance 距离 
 minkowski 明科夫斯基距离  
 binary 二分类距离 

1. Euclidean，欧氏距离 
2. cosine，夹角余弦，机器学习中借用这一概念来衡量样本向量之间的差异。
3. jaccard，杰卡德相似系数，两个集合A和B的交集元素在A，B的并集中所占的比例，称为两个集合的杰卡德相似系数，用符号J(A,B)表示。 
4. Relaxed Word Mover's  Distance（RWMD）文本分析相似性距离 

 层次聚类法 

 single 最短距离法 
 complete  最长距离法 
 median 中间距离法 
 mcquitty  相似法
 average   类平均法  
 centroid  重心法
 ward   离差平方和法
 **划分法** 
 k-means   连续变量  
 K-modes   分类变量  
 k-prototype  混合变量  
 PAM 
 clarans   
 **密度算法**   
 DBSCAN 
 OPTICS 
 DENCLUE   











