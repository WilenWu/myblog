---
title: 大数据手册(Spark)--PySpark MLlib
categories:
  - 'Big Data'
  - Spark
tags:
  - 大数据
  - Spark
  - python
  - 机器学习
cover: /img/apache-spark-mllib.png
top_img: /img/apache-spark-top-img.svg
abbrlink: '75974533'
date: 2024-05-01 22:08:34
description: spark 机器学习库
---

Spark 专为在内存中运行的快速交互式计算而设计，使机器学习可以快速运行。

# Quick Start

MLlib(DataFrame-based) 是Spark的机器学习（ML）库。它的目标是使实用的机器学习变得可扩展和简单。它提供以下高端接口：

- ML算法：常见的学习算法，如分类、回归、聚类和协作过滤
- 特征：特征提取、转换、特征缩放和选择
- 管道：构建、评估和调整ML管道的工具
- 持久性：保存和加载算法、模型和管道
- 实用函数：线性代数、统计、数据处理等。

从Spark 2.0开始，spark.mllib软件包中基于RDD的API已进入维护模式。Spark的主要机器学习API现在是spark.ml包中基于DataFrame的API。

MLlib类似sklearn标准化了机器学习算法的API。以 iris 数据集为例，简述下建模流程：

Step 1: **加载数据集**

```python
# Load the dataset
>>> iris = spark.read.csv("file:///iris.csv", inferSchema="true", header=True)
>>> iris.show(5)                                                                  
+---------------+--------------+---------------+--------------+-------+
|SepalLength(cm)|SepalWidth(cm)|PetalLength(cm)|PetalWidth(cm)|Species|
+---------------+--------------+---------------+--------------+-------+
|            5.1|           3.5|            1.4|           0.2| setosa|
|            4.9|           3.0|            1.4|           0.2| setosa|
|            4.7|           3.2|            1.3|           0.2| setosa|
|            4.6|           3.1|            1.5|           0.2| setosa|
|            5.0|           3.6|            1.4|           0.2| setosa|
+---------------+--------------+---------------+--------------+-------+
```

Step 2: **数据准备**

标签索引化：将类别型标签数值化（可选）

```python
# Convert the categorical labels in the target column to numerical values
indexer = StringIndexer(
    inputCol="Species", 
    outputCol="label"
)
```

创建特征向量：将所有的特征整合到单一列（估计器必须）

```python
# Assemble the feature columns into a single vector column
assembler = VectorAssembler(
    inputCols=["SepalLength(cm)", "SepalWidth(cm)", "PetalLength(cm)", "PetalWidth(cm)"], 
    outputCol="features"
)
```

拆分成训练集和测试集

```python
# Split data into training and testing sets
train, test = iris.randomSplit([0.8, 0.2], seed=42)
```

Step 3: **创建估计器**

```python
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression instance. This instance is an Estimator.
classifier = LogisticRegression(
    maxIter=10, 
    regParam=0.01, 
    featuresCol="features",
    labelCol='label'
)
```

Step 4: **创建管道拟合模型**

```python
from pyspark.ml import Pipeline

# Assemble all the steps (indexing, assembling, and model building) into a pipeline.
pipeline = Pipeline(stages=[indexer, assembler, classifier])
model = pipeline.fit(train)
```

管道通过调用.fit()方法返回用于预测的PipelineModel对象。lrModel 位于管道的对应位置，可以提取并获得模型参数。

```python
lrModel = model.stages[2]
print(lrModel.coefficientMatrix)
print(lrModel.interceptVector)
```

Step 5: **模型预测**

```python
# perform predictions
predictions = model.transform(test)

# save predictions
predictions.write.mode('overwrite').saveAsTable("predictions", partitionBy=None)
```

将之前创建的测试集传递给.transform()方法获得预测。模型预测输出了几列：rawPrediction是原始值，probability是为每个类别计算出的概率，最后prediction是最终的类分配。

Step 6: **模型评估**

```python
from pyspark.ml.evaluation import MultiClassificationEvaluator

# Evaluate the model performance
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.2f}")
```

Step 7: **模型持久化**：

```python
# Save model
pipelinePath="./pipeline"
model.save(pipelinePath)

# Load the model
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load(pipelinePath)
```

# ML Pipelines

## ML 抽象类

ML管道在DataFrames之上提供了一套统一的高级API，以便更容易将多个算法合并到单个管道或工作流中。ML库提供了三个主要的抽象类：Transformer、Estimator和Pipeline。

- **DataFrame**：使用Spark SQL的DataFrame作为ML的数据集，可以容纳各种数据类型。
- **Transformer**：实现了一个方法 transform()，通常通过将一个或多个新列附加到DataFrame来转换为新的DataFrame。比如一个模型就是一个转换器，他可以把一个带有特征列的DataFrame转换为一个加上预测列的新的DataFrame。
- **Estimator**：它是学习算法或在训练数据上的训练方法的抽象概念。在Pipeline里通常被用来操作一个DataFrame数据并生成一个Transformer。评估器实现了一个fit()方法。比如随机森林算法就是一个Estimator，它可以调用fit()方法训练特征数据从而得到一个随机森林模型。
- **Pipeline**：管道的概念用来表示从转换到评估（具有一系列不同阶段）的端到端的过程，这个过程可以对输入的一些原始数据（以DataFrame形式）执行必要的数据加工（转换），最后评估统计模型，返回PipelineModel。
- **Parameter**：所有Transformer和Estimator使用统一的API来指定参数。

## Pipeline

```python
from pyspark.ml import Pipeline
Pipeline(stages=[stage1,stage2,stage3,...])
```

在Pipeline对象上执行.fit()方法时，所有阶段按照stages参数中指定的顺序执行。stages参数是转换器和评估器对象的列表。

```python
from pyspark.ml import Pipeline

# Configure an ML pipeline, which consists of three stages.
pipeline = Pipeline(stages=[indexer, assembler, classifier])

# Fit the pipeline to training dataset.
model = pipeline.fit(train)

# Make predictions on training datasett.
prediction = model.transform(train)
```

# 数据预处理

```py
import pyspark.ml.feature as ft
```

## 特征向量化

| pyspark.ml.feature |                             |
| ------------------ | --------------------------- |
| VectorAssembler    | 特征向量化Transformer       |
| VectorSlicer       | 向量特征提取切片Transformer |

`VectorAssembler` 特征向量化，将多个给定列（包括向量）组合成单个向量列。常用于生成评估器的 featuresCol参数。

```py
>>> from pyspark.ml.feature import VectorAssembler

>>> assembler = VectorAssembler(
...     inputCols=["SepalLength(cm)", "SepalWidth(cm)", "PetalLength(cm)", "PetalWidth(cm)"],
...     outputCol="features"
... )
>>> 
>>> iris = assembler.transform(iris)
>>> iris.select("features", "Species").show(5, truncate=False)
+-----------------+-------+
|features         |Species|
+-----------------+-------+
|[5.1,3.5,1.4,0.2]|setosa |
|[4.9,3.0,1.4,0.2]|setosa |
|[4.7,3.2,1.3,0.2]|setosa |
|[4.6,3.1,1.5,0.2]|setosa |
|[5.0,3.6,1.4,0.2]|setosa |
+-----------------+-------+
only showing top 5 rows
```

`VectorSlicer`是一个Transformer，它接受一个特征向量，并输出一个具有原始特征子数组的新特征向量。可以使用整数索引和字符串名称作为参数。

```py
>>> from pyspark.ml.feature import VectorSlicer

>>> slicer = VectorSlicer(inputCol="features", outputCol="selectedFeatures", indices=[1, 2])
>>> output = slicer.transform(iris)
>>> output.select("features", "selectedFeatures").show(5)
+-----------------+----------------+
|         features|selectedFeatures|
+-----------------+----------------+
|[5.1,3.5,1.4,0.2]|       [3.5,1.4]|
|[4.9,3.0,1.4,0.2]|       [3.0,1.4]|
|[4.7,3.2,1.3,0.2]|       [3.2,1.3]|
|[4.6,3.1,1.5,0.2]|       [3.1,1.5]|
|[5.0,3.6,1.4,0.2]|       [3.6,1.4]|
+-----------------+----------------+
only showing top 5 rows
```

## 特征提取

特征提取被用于将原始特征提取成机器学习算法支持的数据格式，比如文本和图像特征提取。

| pyspark.ml.feature | ML 特征                                                      |
| ------------------ | ------------------------------------------------------------ |
| CountVectorizer    | 是一个`Estimator`，从文档集合中提取词汇表并生成 `CountVectorizerModel` |
| HashingTF          | 是一个`Transformer`，它接受一组term，并将这些集合转换为固定长度的特征向量。 |
| IDF                | 是一个`Estimator`，计算给定文档集合的逆文档频率并生成`IDFModel`。 |
| Word2Vec           | 是一个`Estimator`，它接受代表文档的单词序列，并训练`Word2VecModel`。 |
| StopWordsRemover   | 是一个`Transformer`，从输入中过滤掉停止单词                  |
| NGram              | 是一个`Transformer`，将字符串的输入数组转换为n-grams.。      |
| Tokenizer          | 是一个`Transformer`，将字符串转换成小写                      |
| FeatureHasher      | 是一个`Transformer`，将一组分类或数值特征投射到指定维度的特征向量中（通常大大小于原始特征空间）。 |

```python
>>> from pyspark.ml.feature import Word2Vec
>>> 
>>> # Input data: Each row is a bag of words from a sentence or document.
>>> documentDF = spark.createDataFrame([
...     ("Hi I heard about Spark".split(" "), ),
...     ("I wish Java could use case classes".split(" "), ),
...     ("Logistic regression models are neat".split(" "), )
... ], ["text"])
>>> 
>>> # Learn a mapping from words to Vectors.
>>> word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
>>> model = word2Vec.fit(documentDF)
24/05/01 16:18:36 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
>>> 
>>> result = model.transform(documentDF)
>>> for row in result.collect():
...     text, vector = row
...     print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
... 
Text: [Hi, I, heard, about, Spark] => 
Vector: [0.012264367192983627,-0.06442034244537354,-0.007622340321540833]

Text: [I, wish, Java, could, use, case, classes] => 
Vector: [0.05160687722465289,0.025969027541577816,0.02736483487699713]

Text: [Logistic, regression, models, are, neat] => 
Vector: [-0.06564115285873413,0.02060299552977085,-0.08455150425434113]
```

## 标准化/归一化

| pyspark.ml.feature                          |                                                              |
| ------------------------------------------- | ------------------------------------------------------------ |
| StandardScaler(withMean, withStd, …)        | 是一个`Estimator`。z-scoe标准化                              |
| Normalizer(p, inputCol, outputCol)          | 是一个`Transformer`。该方法使用p范数将数据缩放为单位范数（默认为L2） |
| MaxAbsScaler(inputCol, outputCol)           | 是一个`Estimator`。将数据标准化到`[-1, 1]`范围内             |
| MinMaxScaler(min, max, inputCol, outputCol) | 是一个`Estimator`。将数据标准化到`[0, 1]`范围内              |
| RobustScaler(lower, upper, …)               | 是一个`Estimator`。根据分位数缩放数据                        |

需要先将连续变量合并成向量

```python
>>> from pyspark.ml.feature import Normalizer
>>> 
>>> # Normalize each Vector using $L^1$ norm.
>>> normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
>>> l1NormData = normalizer.transform(iris).select("normFeatures")
>>> print("Normalized using L^1 norm")
Normalized using L^1 norm
>>> l1NormData.show(5)
+--------------------+
|        normFeatures|
+--------------------+
|[0.5,0.3431372549...|
|[0.51578947368421...|
|[0.5,0.3404255319...|
|[0.48936170212765...|
|[0.49019607843137...|
+--------------------+
only showing top 5 rows
```

## 分类特征编码

| pyspark.ml.feature |                                                       |
| ------------------ | ----------------------------------------------------- |
| StringIndexer      | 将字符特征编码为索引列，可以同时编码多列。            |
| IndexToString      | 对应于StringIndexer，将标签索引列映射回原始标字符串列 |
| VectorIndexer      | 对`Vector`特征列中的分类特征索引化                    |
| OneHotEncoder      | One-hot 编码。为每个输入列返回一个编码的输出向量列    |
| ElementwiseProduct | 元素乘积                                              |

StringIndexer转换器可以把字符型特征进行编码，使其数值化。使得某些无法使用类别型特征的算法可以使用，并提高决策树等机器学习算法的效率。

```py
>>> from pyspark.ml.feature import StringIndexer
>>> 
>>> df = spark.createDataFrame(
...     [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
...     ["id", "category"])
>>> 
>>> indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
>>> indexed = indexer.fit(df).transform(df)
# Transformed string column 'category' to indexed column 'categoryIndex'.
# StringIndexer will store labels in output column metadata.
>>> indexed.show()
+---+--------+-------------+
| id|category|categoryIndex|
+---+--------+-------------+
|  0|       a|          0.0|
|  1|       b|          2.0|
|  2|       c|          1.0|
|  3|       a|          0.0|
|  4|       a|          0.0|
|  5|       c|          1.0|
+---+--------+-------------+
```

> 索引的范围从0开始，索引构建的顺序为字符标签的频率，优先编码频率较大的标签，所以出现频率最高的标签为0。

与StringIndexer相对应，IndexToString的作用是把特征索引列重新映射回原有的字符标签。

```python
from pyspark.ml.feature import IndexToString

>>> converter = IndexToString(inputCol="categoryIndex", outputCol="originalCategory")
>>> converted = converter.transform(indexed)

# Transformed indexed column 'categoryIndex' back to original string column 'originalCategory' using labels in metadata
>>> converted.select("id", "categoryIndex", "originalCategory").show()
+---+-------------+----------------+
| id|categoryIndex|originalCategory|
+---+-------------+----------------+
|  0|          0.0|               a|
|  1|          2.0|               b|
|  2|          1.0|               c|
|  3|          0.0|               a|
|  4|          0.0|               a|
|  5|          1.0|               c|
+---+-------------+----------------+
```

之前介绍的 StringIndexer 分别对单个特征进行转换，如果所有特征已经合并到特征向量features中，又想对其中某些单个分量进行处理时，ML包提供了VectorIndexer转化器来执行向量索引化。

> VectorIndexer基于不同特征的数量来识别类别型，maxCategorise参数提供一个阈值，超过阈值的将被认为是类别型，会被索引化。

```python
>>> from pyspark.ml.linalg import Vectors
>>> from pyspark.ml.feature import VectorIndexer

>>> df = spark.createDataFrame([
...     (Vectors.dense([-1.0, 0.0]),),
...     (Vectors.dense([0.0, 1.0]),), 
...     (Vectors.dense([0.0, 2.0]),)], 
...     ["features"])
>>> 
>>> indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=2)
>>> indexerModel = indexer.fit(df)
>>> 
>>> categoricalFeatures = indexerModel.categoryMaps
>>> print("Chose %d categorical features: %s" %
...       (len(categoricalFeatures), ", ".join(str(k) for k in categoricalFeatures.keys())))
Chose 1 categorical features: 0
>>> 
>>> # Create new column "indexed" with categorical values transformed to indices 
>>> indexedData = indexerModel.transform(df)
>>> indexedData.show()
+----------+---------+
|  features|  indexed|
+----------+---------+
|[-1.0,0.0]|[1.0,0.0]|
| [0.0,1.0]|[0.0,1.0]|
| [0.0,2.0]|[0.0,2.0]|
+----------+---------+
```

OneHotEncoder方法来对离散特征进行编码。但是，该方法不接受StringType列，它只能处理数值类型，需要先对特征进行索引化。

```python
>>> from pyspark.ml.feature import OneHotEncoder
>>> 
>>> df = spark.createDataFrame([
...     (0.0, 1.0),
...     (1.0, 0.0),
...     (2.0, 1.0),
...     (0.0, 2.0),
...     (0.0, 1.0),
...     (2.0, 0.0)
... ], ["categoryIndex1", "categoryIndex2"])
>>> 
>>> encoder = OneHotEncoder(inputCols=["categoryIndex1", "categoryIndex2"],
...                         outputCols=["categoryVec1", "categoryVec2"])
>>> model = encoder.fit(df)
>>> encoded = model.transform(df)
>>> encoded.show()
+--------------+--------------+-------------+-------------+
|categoryIndex1|categoryIndex2| categoryVec1| categoryVec2|
+--------------+--------------+-------------+-------------+
|           0.0|           1.0|(2,[0],[1.0])|(2,[1],[1.0])|
|           1.0|           0.0|(2,[1],[1.0])|(2,[0],[1.0])|
|           2.0|           1.0|    (2,[],[])|(2,[1],[1.0])|
|           0.0|           2.0|(2,[0],[1.0])|    (2,[],[])|
|           0.0|           1.0|(2,[0],[1.0])|(2,[1],[1.0])|
|           2.0|           0.0|    (2,[],[])|(2,[0],[1.0])|
+--------------+--------------+-------------+-------------+
```

ElementwiseProduct 输出每个输入向量与提供的“权重”向量的Hadamard积（即元素乘积）。换句话说，它用标量乘数缩放数据集的每一列。
$$
\begin{pmatrix}v_1\\ \vdots \\ v_N\end{pmatrix}\circ
\begin{pmatrix}w_1\\ \vdots \\ w_N\end{pmatrix}=
\begin{pmatrix}v_1w_1\\ \vdots \\ v_Nw_N\end{pmatrix}
$$

```py
>>> from pyspark.ml.feature import ElementwiseProduct
>>> from pyspark.ml.linalg import Vectors
>>> 
>>> # Create some vector data; also works for sparse vectors
>>> data = [(Vectors.dense([1.0, 2.0, 3.0]),), (Vectors.dense([4.0, 5.0, 6.0]),)]
>>> df = spark.createDataFrame(data, ["vector"])
>>> transformer = ElementwiseProduct(scalingVec=Vectors.dense([0.0, 1.0, 2.0]),
...                                  inputCol="vector", outputCol="transformedVector")
>>> # Batch transform the vectors to create new column:
>>> transformer.transform(df).show()
+-------------+-----------------+
|       vector|transformedVector|
+-------------+-----------------+
|[1.0,2.0,3.0]|    [0.0,2.0,6.0]|
|[4.0,5.0,6.0]|   [0.0,5.0,12.0]|
+-------------+-----------------+
```

## 连续特征离散化

| pyspark.ml.feature                           |                                                    |
| -------------------------------------------- | -------------------------------------------------- |
| Binarizer(threshold, inputCol, …)            | 给定阈值的连续特征二值化                           |
| Bucketizer(splits, inputCol, outputCol, ...) | 根据阈值列表将连续变量离散化                       |
| QuantileDiscretizer(numBuckets, ...)         | 传递一个numBuckets参数通过计算数据的近似分位数离散 |

```python
>>> values = [(0.1, 0.0), (0.4, 1.0), (1.2, 1.3), (1.5, float("nan")),
...     (float("nan"), 1.0), (float("nan"), 0.0)]
>>> df = spark.createDataFrame(values, ["values1", "values2"])
>>> bucketizer = Bucketizer(
...     splitsArray=[
...         [-float("inf"), 0.5, 1.4, float("inf")], 
...         [-float("inf"), 0.5, float("inf")]
...     ],
...     inputCols=["values1", "values2"], 
...     outputCols=["buckets1", "buckets2"]
... )
>>> bucketed = bucketizer.setHandleInvalid("keep").transform(df)
>>> bucketed.show(truncate=False)
+-------+-------+--------+--------+
|values1|values2|buckets1|buckets2|
+-------+-------+--------+--------+
|0.1    |0.0    |0.0     |0.0     |
|0.4    |1.0    |0.0     |1.0     |
|1.2    |1.3    |1.0     |1.0     |
|1.5    |NaN    |2.0     |2.0     |
|NaN    |1.0    |3.0     |1.0     |
|NaN    |0.0    |3.0     |0.0     |
+-------+-------+--------+--------+
```

`splits`：将连续特征离散化的阈值列表。对于n+1个阈值，则有n个桶。由阈值 x, y 定义的桶在 [x, y) 范围内持有值，但最后一个桶除外，它也包括y。必须显式提供-inf和inf以涵盖所有Double值。否则，指定边界之外的值将被视为错误。

## 特征构造

| pyspark.ml.feature                         |                                                              |
| ------------------------------------------ | ------------------------------------------------------------ |
| PolynomialExpansion(degree, inputCol, ...) | 多项式特征                                                   |
| PCA(k, inputCol, outputCol)                | 使用主成分分析执行数据降维                                   |
| DCT(inverse, inputCol, outputCol)          | Discrete Cosine Transform (DCT) 离散余弦变换将时间序列中的离散特征转化为周期特征 |
| Interaction(inputCols，outputCol)          | 特征交互。接受矢量或双值列，并生成一个矢量列，其中包含每个输入列中一个值的所有组合的乘积。 |
| RFormula(formula, featuresCol, ...)        | 实现根据R模型公式拟合数据集所需的转换。                      |

```python
>>> data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
...         (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
...         (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
>>> df = spark.createDataFrame(data, ["features"])

>>> pca = PCA(k=3, inputCol="features", outputCol="pca_features")
>>> model = pca.fit(df)

>>> model.explainedVariance
DenseVector([0.7944, 0.2056, 0.0])

>>> result = model.transform(df).select("pca_features")
>>> result.show(truncate=False)
+------------------------------------------------------------+
|pca_features                                                |
+------------------------------------------------------------+
|[1.6485728230883814,-4.0132827005162985,-1.0091435193998504]|
|[-4.645104331781533,-1.1167972663619048,-1.0091435193998501]|
|[-6.428880535676488,-5.337951427775359,-1.009143519399851]  |
+------------------------------------------------------------+
```

## 缺失值插补

```py
Imputer(strategy, missingValue, ...)
```

使用缺失值所在列的平均值、中位数或众数完成缺失值的插。输入列应为数字类型。目前`Imputer`不支持分类功能，并可能为包含分类功能的列创建不正确的值。

```python
>>> from pyspark.ml.feature import Imputer
>>> 
>>> df = spark.createDataFrame([
...     (1.0, float("nan")),
...     (2.0, float("nan")),
...     (float("nan"), 3.0),
...     (4.0, 4.0),
...     (5.0, 5.0)
... ], ["a", "b"])
>>> 
>>> imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
>>> model = imputer.fit(df)
>>> 
>>> model.transform(df).show()
+---+---+-----+-----+
|  a|  b|out_a|out_b|
+---+---+-----+-----+
|1.0|NaN|  1.0|  4.0|
|2.0|NaN|  2.0|  4.0|
|NaN|3.0|  3.0|  3.0|
|4.0|4.0|  4.0|  4.0|
|5.0|5.0|  5.0|  5.0|
+---+---+-----+-----+
```

## 特征选择

| pyspark.ml.feature                          |                                |
| ------------------------------------------- | ------------------------------ |
| ChiSqSelector(numTopFeatures, ...)          | 选择用于预测分类标签的分类特征 |
| VarianceThresholdSelector(featuresCol, ...) | 删除所有低方差特征             |
| UnivariateFeatureSelector(featuresCol, ...) | 单变量特征选择                 |

`UnivariateFeatureSelector`在具有分类/连续特征的分类/回归任务上选择特征。Spark根据指定的`featureType`和`labelType`参数选择要使用的评分函数。

| featureType | labelType   | score function         |
| :---------- | :---------- | :--------------------- |
| categorical | categorical | chi-squared (chi2)     |
| continuous  | categorical | ANOVATest (f_classif)  |
| continuous  | continuous  | F-value (f_regression) |

它支持五种选择模式：

- numTopFeatures 选择评分最高的固定数量的特征。
- percentile 选择评分最高的固定百分比的特征。
- fpr选择p值低于阈值的所有特征，从而控制假阳性选择率。
- fdr使用Benjamini-Hochberg程序来选择错误发现率低于阈值的所有特征。
- fwe选择p值低于阈值的所有功能。阈值按1/numFeatures缩放，从而控制family-wise的错误率。

```py
>>> from pyspark.ml.feature import UnivariateFeatureSelector
>>> from pyspark.ml.linalg import Vectors
>>> 
>>> df = spark.createDataFrame([
...     (1, Vectors.dense([1.7, 4.4, 7.6, 5.8, 9.6, 2.3]), 3.0,),
...     (2, Vectors.dense([8.8, 7.3, 5.7, 7.3, 2.2, 4.1]), 2.0,),
...     (3, Vectors.dense([1.2, 9.5, 2.5, 3.1, 8.7, 2.5]), 3.0,),
...     (4, Vectors.dense([3.7, 9.2, 6.1, 4.1, 7.5, 3.8]), 2.0,),
...     (5, Vectors.dense([8.9, 5.2, 7.8, 8.3, 5.2, 3.0]), 4.0,),
...     (6, Vectors.dense([7.9, 8.5, 9.2, 4.0, 9.4, 2.1]), 4.0,)], 
...     ["id", "features", "label"])
>>> 
>>> selector = UnivariateFeatureSelector(
...     featuresCol="features", 
...     outputCol="selectedFeatures",
...     labelCol="label", 
...     selectionMode="numTopFeatures")
>>> selector.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(1)
>>> # UnivariateFeatureSelector output with top 1 features selected using f_classif
>>> result = selector.fit(df).transform(df)
>>> result.show()
+---+--------------------+-----+----------------+
| id|            features|label|selectedFeatures|
+---+--------------------+-----+----------------+
|  1|[1.7,4.4,7.6,5.8,...|  3.0|           [2.3]|
|  2|[8.8,7.3,5.7,7.3,...|  2.0|           [4.1]|
|  3|[1.2,9.5,2.5,3.1,...|  3.0|           [2.5]|
|  4|[3.7,9.2,6.1,4.1,...|  2.0|           [3.8]|
|  5|[8.9,5.2,7.8,8.3,...|  4.0|           [3.0]|
|  6|[7.9,8.5,9.2,4.0,...|  4.0|           [2.1]|
+---+--------------------+-----+----------------+
```

## SQL转换器

SQLTransformer实现了由SQL语句定义的转换。目前，只支持如下SQL语法：

```sql
SELECT ... FROM __THIS__ ... where __THIS__
```

其中`__THIS__` 表示输入数据集的底层表。

```py
>>> from pyspark.ml.feature import SQLTransformer
>>> 
>>> df = spark.createDataFrame([
...     (0, 1.0, 3.0),
...     (2, 2.0, 5.0)
... ], ["id", "v1", "v2"])
>>> sqlTrans = SQLTransformer(
...     statement="SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
>>> sqlTrans.transform(df).show()
+---+---+---+---+----+
| id| v1| v2| v3|  v4|
+---+---+---+---+----+
|  0|1.0|3.0|4.0| 3.0|
|  2|2.0|5.0|7.0|10.0|
+---+---+---+---+----+
```

# 机器学习常用算法

## 分类和回归

| pyspark.ml.classification      |                    |
| ------------------------------ | ------------------ |
| LogisticRegression             | 逻辑回归           |
| DecisionTreeClassifier         | 决策树             |
| RandomForestClassifier         | 随机森林           |
| GBTClassifier                  | 梯度增强树（GBT）  |
| LinearSVC                      | 线性支持向量机     |
| NaiveBayes                     | 朴素贝叶斯         |
| FMClassifier                   | 分解机器学习       |
| MultilayerPerceptronClassifier | 多层感知机         |
| OneVsRest                      | 多分类简化为二分类 |

| pyspark.ml.regression       |                   |
| --------------------------- | ----------------- |
| LinearRegression            | 线性回归          |
| GeneralizedLinearRegression | 广义线性回归      |
| DecisionTreeRegressor       | 决策树            |
| RandomForestRegressor       | 随机森林          |
| GBTRegressor                | 梯度增强树（GBT） |
| AFTSurvivalRegression       | 生存回归          |
| IsotonicRegression          | 保序回归          |
| FMRegressor                 | 分解机器学习      |

```py
>>> from pyspark.ml import Pipeline
>>> from pyspark.ml.classification import DecisionTreeClassifier
>>> from pyspark.ml.feature import StringIndexer, VectorAssembler
>>> from pyspark.ml.evaluation import MulticlassClassificationEvaluator

>>> # Load the dataset.
>>> data = spark.read.csv("file:///iris.csv", inferSchema="true", header=True)

>>> # Index labels, adding metadata to the label column.
>>> # Fit on whole dataset to include all labels in index.
>>> labelIndexer = StringIndexer(inputCol="Species", outputCol="label")

>>> # Assemble the feature columns into a single vector column
>>> assembler = VectorAssembler(
...     inputCols=["SepalLength(cm)", "SepalWidth(cm)", "PetalLength(cm)", "PetalWidth(cm)"], 
...     outputCol="features"
... )

>>> # Split the data into training and test sets (30% held out for testing)
>>> trainingData, testData = data.randomSplit([0.7, 0.3])
>>> # Train a DecisionTree model.

>>> dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
>>> # Chain indexers and tree in a Pipeline
>>> pipeline = Pipeline(stages=[labelIndexer, assembler, dt])
>>> # Train model.  This also runs the indexers.
>>> model = pipeline.fit(trainingData)

>>> # Make predictions.
>>> predictions = model.transform(testData)
>>> # Select example rows to display.
>>> predictions.select("prediction", "label").show(5)
+----------+-----+
|prediction|label|
+----------+-----+
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
+----------+-----+
only showing top 5 rows

>>> # Select (prediction, true label) and compute test error
>>> evaluator = MulticlassClassificationEvaluator(
...     labelCol="label", predictionCol="prediction", metricName="accuracy")
>>> accuracy = evaluator.evaluate(predictions)
>>> print("Test Error = %g " % (1.0 - accuracy))
Test Error = 0.0425532 

>>> treeModel = model.stages[2]
>>> # summary only
>>> print(treeModel)
DecisionTreeClassificationModel: uid=DecisionTreeClassifier_912bad7cd9f2, depth=5, numNodes=15, numClasses=3, numFeatures=4
```

## 聚类

| pyspark.ml.clustering    |              |
| ------------------------ | ------------ |
| KMeans                   | k-means      |
| LDA                      | 线性判别分析 |
| BisectingKMeans          | 分层k-means  |
| GaussianMixture          | 高斯混合聚类 |
| PowerIterationClustering | PIC          |

```py
>>> from pyspark.ml.clustering import KMeans
>>> from pyspark.ml.evaluation import ClusteringEvaluator

# Loads data.
>>> data = spark.read.csv("file:///iris.txt", inferSchema="true", header=True)

>>> # Assemble the feature columns into a single vector column
>>> data = VectorAssembler(
...     inputCols=["SepalLength(cm)", "SepalWidth(cm)", "PetalLength(cm)", "PetalWidth(cm)"], 
...     outputCol="features"
... ).transform(data)

# Trains a k-means model.
>>> kmeans = KMeans().setK(3).setSeed(1)
>>> model = kmeans.fit(data)

# Make predictions
>>> predictions = model.transform(data)
>>> evaluator = ClusteringEvaluator()
>>> silhouette = evaluator.evaluate(predictions)
>>> print("Silhouette with squared euclidean distance = " + str(silhouette))
Silhouette with squared euclidean distance = 0.7342113066202739
>>> centers = model.clusterCenters()
>>> print("Cluster Centers: ")
Cluster Centers: 
>>> for center in centers:
...     print(center)
... 
[6.85384615 3.07692308 5.71538462 2.05384615]
[5.006 3.418 1.464 0.244]
[5.88360656 2.74098361 4.38852459 1.43442623]
```

## 协同过滤

协同过滤通常用于推荐系统。

| pyspark.ml.recommendation |                             |
| ------------------------- | --------------------------- |
| ALS                       | 交替最小二乘（ALS）矩阵分解 |

```py
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

lines = spark.read.text("data/mllib/als/sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
```

## 频繁模式挖掘

| pyspark.ml.fpm |                                                |
| -------------- | ---------------------------------------------- |
| FPGrowth       | 一种并行FP增长算法，用于挖掘频繁的项目集       |
| PrefixSpan     | 一个并行的PrefixSpan算法来挖掘频繁的顺序模式。 |

# 模型选择和评估

ML支持使用CrossValidator和TrainValidationSplit进行模型评估和选择。主要需要以下参数：

- Estimator：要调整的算法或Pipeline
- ParamMap：可供选择的参数网格
- Evaluator：衡量模型表现的评估器

## 模型评估

| pyspark.ml.evaluation             | Desc               | metricName                                                   |
| --------------------------------- | ------------------ | ------------------------------------------------------------ |
| RegressionEvaluator               | 回归模型评估       | areaUnderROC, areaUnderPR                                    |
| BinaryClassificationEvaluator     | 二分类模型评估     | rmse, mse, r2, mae, var                                      |
| MulticlassClassificationEvaluator | 多分类模型评估     | f1, accuracy, weightedPrecision, weightedRecall,  logLoss, … |
| MultilabelClassificationEvaluator | 多标签分类模型评估 | precisionByLabel, recallByLabel, f1MeasureByLabel            |
| ClusteringEvaluator               | 聚类模型评估       | silhouette                                                   |
| RankingEvaluator                  | 排序模型评估       | meanAveragePrecision, ndcgAtK, …                             |

```python
>>> from pyspark.ml.linalg import Vectors
>>> from pyspark.ml.evaluation import BinaryClassificationEvaluator

>>> scoreAndLabels = [
...     (Vectors.dense([0.9, 0.1]), 0.0), 
...     (Vectors.dense([0.9, 0.1]), 1.0), 
...     (Vectors.dense([0.6, 0.4]), 0.0), 
...     (Vectors.dense([0.4, 0.6]), 1.0)
... ]
>>> dataset = spark.createDataFrame(scoreAndLabels, ["raw", "label"])
>>> 
>>> evaluator = BinaryClassificationEvaluator()
>>> evaluator.setRawPredictionCol("raw")
BinaryClassificationEvaluator_13c5fd3055fb
>>> evaluator.evaluate(dataset)
0.625
>>> 
>>> evaluator.evaluate(dataset, {evaluator.metricName: "areaUnderPR"})
0.75
```

## 超参数调优

| pyspark.ml.tuning    |              |
| -------------------- | ------------ |
| ParamGridBuilder     | 构建参数网格 |
| CrossValidator       | K折交叉验证  |
| TrainValidationSplit | 单次验证     |

Spark ML 的超参数调优主要借助CrossValidator或TrainValidationSplit来实现，工作方式如下：

1. 将输入数据拆分为单独的训练集和测试集。
2. 使用ParamGridBuilder构建参数网格，根据给定评估指标，循环遍历定义的参数值列表，估计各个单独的模型，从而选定一个最佳 ParamMap。
3. 最终使用最佳ParamMap和整个数据集重新拟合Estimator。

> 默认情况下，参数网格中的参数集串行计算。

```python
>>> from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
>>> from pyspark.ml.evaluation import MulticlassClassificationEvaluator
>>> from pyspark.ml.classification import LogisticRegression

# Prepare training and test data.
>>> iris = spark.read.csv("file:///iris.csv", inferSchema="true", header=True)

# Convert the categorical labels in the target column to numerical values
>>> indexer = StringIndexer(
...     inputCol="Species", 
...     outputCol="label"
... )

>>> # Assemble the feature columns into a single vector column
>>> assembler = VectorAssembler(
...     inputCols=["SepalLength(cm)", "SepalWidth(cm)", "PetalLength(cm)", "PetalWidth(cm)"], 
...     outputCol="features"
... )

>>> train, test = iris.randomSplit([0.9, 0.1], seed=42)

>>> lr = LogisticRegression(maxIter=100)

# Assemble all the steps (indexing, assembling, and model building) into a pipeline.
>>> pipeline = Pipeline(stages=[indexer, assembler, lr])

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# CrossValidator will try all combinations of values and determine best model using
# the evaluator.
>>> paramGrid = ParamGridBuilder()\
...     .addGrid(lr.regParam, [0.1, 0.01]) \
...     .addGrid(lr.fitIntercept, [False, True])\
...     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
...     .build()

# In this case the estimator is simply the linear regression.
# A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
>>> crossval = CrossValidator(estimator=pipeline,
...                           estimatorParamMaps=paramGrid,
...                           evaluator=MulticlassClassificationEvaluator(),
...                           numFolds=3)

# Run cross-validation, and choose the best set of parameters.
>>> cvModel = crossval.fit(train)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
>>> cvModel.transform(test)\
...     .select("features", "label", "prediction")\
...     .show(5)
+-----------------+-----+----------+
|         features|label|prediction|
+-----------------+-----+----------+
|[4.8,3.4,1.6,0.2]|  1.0|       1.0|
|[4.9,3.1,1.5,0.1]|  1.0|       1.0|
|[5.4,3.4,1.5,0.4]|  1.0|       1.0|
|[5.1,3.4,1.5,0.2]|  1.0|       1.0|
|[5.1,3.8,1.6,0.2]|  1.0|       1.0|
+-----------------+-----+----------+
only showing top 5 rows
```
## 附录: ROC

PySpark 中并没有在评估模块中直接提供ROC曲线的提取方式，我们可以从以下三个渠道获得：
- MLlib (DataFrame-based)中的部分分类模型，可以通过summary属性/方法获得
- Spark 在Scala API中提供了提取ROC曲线的方式，因此我们需要从Scala模块中借用
- 自定义ROC提取函数


支持summary属性/方法的模型如下：

```python
# Returns the ROC curve, which is a Dataframe having two fields (FPR, TPR) 
# with (0.0, 0.0) prepended and (1.0, 1.0) appended to it.

roc_df = LinearSVCModel.summary().roc
roc_df = LogisticRegressionModel.summary.roc
roc_df = RandomForestClassificationModel.summary.roc
roc_df = MultilayerPerceptronClassificationModel.summary().roc
roc_df = FMClassificationModel.summary().roc
```

scala中的提取代码如下：

```scala
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

// Instantiate metrics object
val metrics = new BinaryClassificationMetrics(predictionAndLabels)

// Precision-Recall Curve
val PRC = metrics.pr

// AUPRC
val auPRC = metrics.areaUnderPR
println(s"Area under precision-recall curve = $auPRC")

// ROC Curve
val roc = metrics.roc

// AUROC
val auROC = metrics.areaUnderROC
println(s"Area under ROC = $auROC")
```

在 PySpark 中定义一个子类借用scala接口：

```python
from pyspark.mllib.evaluation import BinaryClassificationMetrics

class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)
    
    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter,
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2,
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points
    
    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        points = self._to_list(rdd)
        return zip(*points)  # return tuple(fpr, tpr)
```

定义子类后具体使用如下：

```python
from pyspark.sql import Row
def extractProbability(row, labelCol='label', probabilityCol='probability'):
    return Row(label = float(row[labelCol]), probability = float(row['probability'][1]))

pred_df = predictions.rdd.map(extractProbability)
fpr, tpr = CurveMetrics(pred_df).get_curve('roc')
```

参考sklearn自定义函数提取ROC：

```python
from pyspark.sql import Window, functions as fn
from pyspark.sql import feature as ft

def roc_curve_on_spark(dataset, labelCol='label', probabilityCol='probability'):
    """
    Returns the receiver operating characteristic (ROC) curve,
        which is a Dataframe having two fields (FPR, TPR) with
        (0.0, 0.0) prepended and (1.0, 1.0) appended to it.
    """
    
    roc = dataset.select(labelCol, probabilityCol)
    
    # window functions
    window = Window.orderBy(fn.desc(probabilityCol))
    
    # accumulate the true positives with decreasing threshold
    roc = roc.withColumn('tps', fn.sum(roc[labelCol]).over(window))
    # accumulate the false positives with decreasing threshold
    roc = roc.withColumn('fps', fn.sum(fn.lit(1) - roc[labelCol]).over(window))

    # The total number of negative samples
    numPositive = roc.tail(1)[0]['tps']
    numNegative = roc.tail(1)[0]['fps']

    roc = roc.withColumn('tpr', roc['tps'] / fn.lit(numPositive))
    roc = roc.withColumn('fpr', roc['fps'] / fn.lit(numNegative))
    
    roc = roc.dropDuplicates(subset=probabilityCol).select('fpr', 'tpr')

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    start_row = spark.createDataFrame([(0.0, 0.0)], schema=roc.schema)
    roc = start_row.unionAll(roc)
    
    return roc
```

# 实用工具

## 数理统计

| pyspark.ml.stat                                    | 统计模块                                                     |
| -------------------------------------------------- | ------------------------------------------------------------ |
| Correlation.corr(dataset, column, method)          | 计算相关系数矩阵，目前支持 Pearson and Spearman相关系数      |
| ChiSquareTest.test(dataset, featuresCol, labelCol) | 卡方检验                                                     |
| Summarizer                                         | 描述统计：可用的指标包括列最大值、最小值、均值、总和、方差、标准值和非零数，以及总计数。 |

```python
>>> from pyspark.ml.linalg import DenseMatrix, Vectors
>>> from pyspark.ml.stat import Correlation, ChiSquareTest, Summarizer
>>> dataset = [[0, Vectors.dense([1, 0, 0, -2])],
...            [0, Vectors.dense([4, 5, 0, 3])],
...            [1, Vectors.dense([6, 7, 0, 8])],
...            [1, Vectors.dense([9, 0, 0, 1])]]
>>> dataset = spark.createDataFrame(dataset, ['features'])

# Compute the correlation matrix with specified method using dataset.
>>> pearsonCorr = Correlation.corr(dataset, 'features', 'pearson').collect()[0][0]
>>> print(str(pearsonCorr).replace('nan', 'NaN'))
DenseMatrix([[ 1.        ,  0.0556...,         NaN,  0.4004...],
             [ 0.0556...,  1.        ,         NaN,  0.9135...],
             [        NaN,         NaN,  1.        ,         NaN],
             [ 0.4004...,  0.9135...,         NaN,  1.        ]])

# Perform a Pearson’s independence test using dataset.
>>> chiSqResult = ChiSquareTest.test(dataset, 'features', 'label').collect()[0]
>>> print("pValues: " + str(chiSqResult.pValues))
pValues: [0.2614641299491107,0.3678794411714422,1.0,0.2614641299491107]
>>> print("degreesOfFreedom: " + str(chiSqResult.degreesOfFreedom))
degreesOfFreedom: [3, 2, 0, 3]
>>> print("statistics: " + str(chiSqResult.statistics))
statistics: [4.0,2.0,0.0,4.0]

# create summarizer for multiple metrics "mean" and "count"
>>> summarizer = Summarizer.metrics("mean", "count")
>>> dataset.select(summarizer.summary(dataset.features)).show(truncate=False)
+--------------------------------+
|aggregate_metrics(features, 1.0)|
+--------------------------------+
|{[5.0,3.0,0.0,2.5], 4}          |
+--------------------------------+
>>> dataset.select(Summarizer.mean(dataset.features)).show(truncate=False)
+-----------------+
|mean(features)   |
+-----------------+
|[5.0,3.0,0.0,2.5]|
+-----------------+
```

# MLlib (RDD-based)

MLlib (RDD-based) 是基于RDD的原始算法的API。从Spark2.0开始，ML是主要的机器学习库，它对DataFrame进行操作。

MLlib的抽象类

-   Vector：向量（mllib.linalg.Vectors）支持dense和sparse（稠密向量和稀疏向量）。区别在与前者的每一个数值都会存储下来，后者只存储非零数值以节约空间。
-   LabeledPoint:（mllib.regression）表示带标签的数据点，包含一个特征向量与一个标签。注意，标签要通过StringIndexer转化成浮点型的。
-   Matrix：(pyspark.mllib.linalg)，支持dense和sparse（稠密矩阵和稀疏矩阵）。
-   各种Model类：每个Model都是训练算法的结果，一般都有一个predict方法可以用来对新的数据点或者数据点组成的RDD应用该模型进行预测

一般来说，大多数算法直接操作由Vector、LabledPoint或Rating组成的RDD，通常我们从外部数据读取数据后需要进行转化操作构建RDD。

```python
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_libsvm_data.txt')
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification tree model:')
print(model.toDebugString())

# Save and load model
model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")
```

 
