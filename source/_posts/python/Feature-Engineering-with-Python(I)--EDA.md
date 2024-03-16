---
title: 特征工程(I)--探索性数据分析
tags:
  - Python
categories:
  - Python
  - 'Machine Learning'
cover: /img/FeatureEngine.png
top_img: /img/sklearn-top-img.svg
abbrlink: 
description: 
date: 
---
<center><font size=10><b>特征工程</b></font></center>

有这么一句话在业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。由此可见，特征工程在机器学习中占有相当重要的地位。在实际应用当中，可以说特征工程是机器学习成功的关键。

特征工程是数据分析中最耗时间和精力的一部分工作，它不像算法和模型那样是确定的步骤，更多是工程上的经验和权衡。因此没有统一的方法。这里只是对一些常用的方法做一个总结。

特征工程包含了 Data PreProcessing（数据预处理）、Feature Extraction（特征提取）、Feature Selection（特征选择）和 Feature construction（特征构造）等子问题。

# 数据集描述

本项目使用Kaggle上的 [家庭信用违约风险数据集 (Home Credit Default Risk)](https://www.kaggle.com/competitions/home-credit-default-risk/data) ，是一个标准的机器学习分类问题。其目标是使用历史贷款的信息，以及客户的社会经济和财务信息，预测客户是否会违约。

数据集包括了8个不同的数据文件，其中：

application_{train|test}：包含每个客户社会经济信息和Home Credit贷款申请信息的主要文件。每行代表一个贷款申请，由SK_ID_CURR唯一标识。训练集30.75万数据，测试集4.87万数据。其中训练集中`TARGET=1`表示未偿还贷款。通过这两个文件，就能对这个任务做基本的数据分析和建模，也是本篇的主要内容。

数据集的描述性文件：[HomeCredit_columns_description.csv](/ipyna/HomeCredit_columns_description.csv)

# 探索性数据分析

Exploratory Data Analysis(EDA)

## 数据概览


导入必要的包

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Set global configuration.
warnings.filterwarnings('ignore')
sns.set_style('seaborn-whitegrid')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.options.display.max_colwidth = 100

SEED = 42
```

导入数据

```python
df = pd.read_csv('../datasets/Home-Credit-Default-Risk/application_train.csv')
print(df.head())
```


```python
print('Training data shape: {df.shape}', df.shape)
```


```python
# `SK_ID_CURR` is the unique id of the row.
ID_col = "SK_ID_CURR"
df[ID_col].nunique() == df.shape[0]

features = df.columns.drop([ID_col, target]).to_list() 
category_cols = []
numeric_cols = []
```
目标变量

```python
# `TARGET` is the target variable we are trying to predict (0 or 1):
# 1 = Not Repaid 
# 0 = Repaid
target = 'TARGET'
print(df[target].value_counts())
```

查看字段类型

在遇到非常多的数据的时候，我们一般先会按照数据的类型分布下手，看看不同的数据类型Field各有多少.

```python
# Number of each type of column
print(df.dtypes.value_counts())
```

```python
df.select_dtypes("int64").astype("int32", inplace=True)
df.select_dtypes("float64").astype("float32", inplace=True)
df.select_dtypes("object").astype("category", inplace=True)
```



```python
categorical_cols = df.select_dtypes(["object"]).columns.drop([ID_col, target], errors='ignore').tolist()
print("Categorical features:", categorical_cols)
numeric_cols = df.select_dtypes("number").columns.drop([ID_col, target], errors='ignore').tolist()
print("Numeric features:", numeric_cols)
boolean_cols = 
```
```python
len(categorical_cols + numeric_cols + [ID_col, target]) == len(df.columns)
```
接下来看下数据集的统计信息



## 数据相关性

data Correlation

用相关性矩阵热图表现特征与目标值之间以及两两特征之间的相关程度，对特征的处理有指导意义。

查看相关系数矩阵

```python
#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


def corrplot(df, method="pearson", annot=True, **kwargs):
    sns.clustermap(
        df.corr(method, numeric_only=True),
        vmin=-1.0,
        vmax=1.0,
        cmap="icefire",
        method="complete",
        annot=annot,
        **kwargs,
    )


corrplot(df_train, annot=None)


# Training set high correlations
df_corr = df.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
# higher than 0.1
df_corr['Correlation Coefficient'] > 0.1
```


目标变量相关性


- Continuous Features
```python
# Correlation map to see how features are correlated with SalePrice
corrmat = train.corr(numeric_only=True)
fig, axs = plt.subplots(figsize=(20, 20))
sns.heatmap(corrmat, vmax=0.9, square=True)
axs.set_title('Correlations', size=15)
plt.show()


cont_features = ['Age', 'Fare']
surv = df_train['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(cont_features):    
    # Distribution of survival in feature
    sns.distplot(df_train[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])
    sns.distplot(df_train[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])
    
    # Distribution of feature in dataset
    sns.distplot(df_train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])
    sns.distplot(df_test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])
    
    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')
    
    for j in range(2):        
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)
    
    axs[0][i].legend(loc='upper right', prop={'size': 20})
    axs[1][i].legend(loc='upper right', prop={'size': 20})
    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)

axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)
axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)
        
plt.show()


plt.figure(figsize = (10, 12))

# iterate through the sources
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    
    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)

```


- Categorical Features

```python
cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))
plt.subplots_adjust(right=1.5, top=1.25)

for i, feature in enumerate(cat_features, 1):    
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=df_train)
    
    plt.xlabel('{}'.format(feature), size=20, labelpad=15)
    plt.ylabel('Passenger Count', size=20, labelpad=15)    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})
    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)

plt.show()
```


一些分类特征和目标变量有很大的相关性，可以one-hot编码，也可以组合成新的有序分类变量。

```python
# Plots the disribution of a variable colored by value of the target
def kde_target(var_name, df):
    
    # Calculate the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])
    
    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label = 'TARGET == 1')
    
    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
    
```



## 目标变量分布

检查目标变量分布

```python
print(f"percentage of default : {df[target].mean():.2%}")
print(df[target].value_counts())
```



现实中，样本（类别）样本不平衡（class-imbalance）是一种常见的现象，一般地，做分类算法训练时，如果样本类别比例（Imbalance Ratio）（多数类vs少数类）严重不平衡时，分类算法将开始做出有利于多数类的预测。一般有以下几种方法：权重法、采样法、数据增强、损失函数、集成方法、评估指标。

|方法  |函数|python包  |
|:-|:-|:-|
| SMOTE | SMOTE |imblearn.over_sampling|
| ADASYN | ADASYN |imblearn.over_sampling|
| Bagging算法 | BalancedBaggingClassifier |imblearn.ensemble|
|Boosting算法|EasyEnsembleClassifier|imblearn.ensemble|
|损失函数|Focal Loss |self-define|

我们可以用[imbalance-learn](https://imbalanced-learn.org/stable/)这个Python库实现诸如重采样和模型集成等大多数方法。

对于分类问题的目标变量，在sklearn中设置了不同的编码函数

| sklearn.preprocessing | 预处理           |
| --------------------- | ---------------- |
| LabelEncoder          | 目标变量序数编码   |
| LabelBinarizer        | 二分类目标数值化   |
| MultiLabelBinarizer   | 多标签目标数值化 |

假设预测客户的贷款额度AMT_CREDIT

对于回归任务，



本项目使用Kaggle房价预测数据集：[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) 。训练集和测试集合中的每条记录包含79项特征，用于描述市区住宅的属性


查看目标变量分布
```python
df['SalePrice'].describe()
```

我们画出SalePrice的分布图和QQ图。
> Quantile-Quantile图是一种常用的统计图形，用来比较两个数据集之间的分布。它是由标准正态分布的分位数为横坐标，样本值为纵坐标的散点图。如果QQ图上的点在一条直线附近，则说明数据近似于正态分布，且该直线的斜率为标准差，截距为均值。

```python
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_qqplot import pplot
from scipy import stats import probplot, norm

def norm_comparison_plot(series):
	mu, sigma = stats.norm.fit(series)
	kurt, skew = series.kurt(), series.skew()
	print(f"Kurtosis: {kurt:.2f}", f"Skewness: {skew:.2f}", sep='\t')
	
	fig = plt.figure()
	#Now plot the distribution
	ax1 = fig.add_subplot(121)
	ax1.set_title('SalePrice distribution')
	ax1.set_ylabel('Frequency')
	sns.distplot(series, fit=stats.norm, ax=ax1)
	ax1.legend(f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )', loc='best')
	#Get also the QQ-plot
	ax2 = fig.add_subplot(122)
	stats.probplot(series, plot=plt)


norm_comparison_plot(df['SalePrice'])
plt.show()
```

可以看到 SalePrice 的分布呈偏态，许多回归算法都有正态分布假设，因此我们尝试对数变换，让数据接近正态分布。

```python
norm_comparison_plot(np.log1p(df['SalePrice']))
plt.show()
```

可以看到经过对数变换后，基本符合正态分布了。

sklearn.compose 中的 TransformedTargetRegressor 是专门为回归任务设置的目标变换。对于简单的变换，TransformedTargetRegressor在拟合回归模型之前变换目标变量，预测时则通过逆变换映射回原始值。

```python
reg = TransformedTargetRegressor(regressor=LinearRegression(), 
	transformer=FunctionTransformer(np.log1p))
```



参考文献：
[Home Credit Default Risk - 1 之基础篇](https://zhuanlan.zhihu.com/p/104288764)
[Home Credit Default Risk 之FeatureTools篇](https://zhuanlan.zhihu.com/p/104370111)
[Feature Engineering for House Prices](https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices) 
[Credit Fraud信用卡欺诈数据集，如何处理非平衡数据](https://blog.csdn.net/s09094031/article/details/90924284)
[Predict Future Sales 预测未来销量, Kaggle 比赛，LB 0.89896 排名6%](https://blog.csdn.net/s09094031/article/details/90347191)
[feature-engine](https://feature-engine.trainindata.com/en/latest/)将特征工程中常用的方法进行了封装
[分享关于人工智能的内容](https://blog.csdn.net/coco2d_x2014/category_8742518.html)

[特征工程系列：特征筛选的原理与实现（上）](https://cloud.tencent.com/developer/article/1469820) 7.23
[特征工程系列：特征筛选的原理与实现（下）](https://cloud.tencent.com/developer/article/1469822) 7.23
[特征工程系列：数据清洗](https://cloud.tencent.com/developer/article/1478305) 8.1
[特征工程系列：特征预处理（上）](https://cloud.tencent.com/developer/article/1483475) 8.8
[特征工程系列：特征预处理（下）](https://cloud.tencent.com/developer/article/1488245) 8.16
[特征工程系列：特征构造之概览篇](https://cloud.tencent.com/developer/article/1517948) 10.8
[特征工程系列：聚合特征构造以及转换特征构造](https://cloud.tencent.com/developer/article/1519994) 10.12
[特征工程系列：笛卡尔乘积特征构造以及遗传编程特征构造](https://cloud.tencent.com/developer/article/1521565) 10.15
[特征工程系列：GBDT特征构造以及聚类特征构造](https://cloud.tencent.com/developer/article/1530229) 10.30
[特征工程系列：时间特征构造以及时间序列特征构造](https://cloud.tencent.com/developer/article/1536537) 11.11
[特征工程系列：自动化特征构造](https://cloud.tencent.com/developer/article/1551874) 12.10
[特征工程系列：空间特征构造以及文本特征构造](https://cloud.tencent.com/developer/article/1551871) 12.10
