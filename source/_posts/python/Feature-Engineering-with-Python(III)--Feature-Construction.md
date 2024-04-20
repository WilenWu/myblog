---
title: 特征工程(III)--特征构造
tags:
  - Python
categories:
  - Python
  - 'Machine Learning'
cover: /img/FeatureEngine.png
top_img: /img/sklearn-top-img.svg
abbrlink: ca507391
description: 
date: 2024-04-10 23:30:52
---

有这么一句话在业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。由此可见，特征工程在机器学习中占有相当重要的地位。在实际应用当中，可以说特征工程是机器学习成功的关键。

特征工程是数据分析中最耗时间和精力的一部分工作，它不像算法和模型那样是确定的步骤，更多是工程上的经验和权衡。因此没有统一的方法。这里只是对一些常用的方法做一个总结。

特征工程包含了 Data PreProcessing（数据预处理）、Feature Extraction（特征提取）、Feature Selection（特征选择）和 Feature construction（特征构造）等子问题。

Jupyter Notebook 代码连接：[feature_engineering_demo_p3_feature_construction](/ipynb/feature_engineering_demo_p3_feature_construction)

导入必要的包


```python
import numpy as np
import pandas as pd
import re
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.pipeline import FeatureUnion, make_union, Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc

# Setting configuration.
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

SEED = 42
```

# 特征构造

特征构造是从现有数据创建新特征的过程。目标是构建有用的功能，帮助我们的模型了解数据集中的信息与给定目标之间的关系。


```python
df = pd.read_csv('../datasets/Home-Credit-Default-Risk/prepared_data.csv', index_col='SK_ID_CURR')
```

定义帮助节省内存的函数


```python
def convert_dtypes(df, verbose=True):
    original_memory = df.memory_usage().sum()
    df = df.apply(pd.to_numeric, errors='ignore')
    
    # Convert booleans to integers
    boolean_features = df.select_dtypes(bool).columns.tolist()
    df[boolean_features] = df[boolean_features].astype(np.int32)
     # Convert objects to category
    object_features = df.select_dtypes(object).columns.tolist()
    df[object_features] = df[object_features].astype('category')
    # Float64 to float32
    float_features = df.select_dtypes(float).columns.tolist()
    df[float_features] = df[float_features].astype(np.float32)
    # Int64 to int32
    int_features = df.select_dtypes(int).columns.tolist()
    df[int_features] = df[int_features].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    if verbose:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
    
    return df
```


```python
df = convert_dtypes(df)
X = df.drop('TARGET', axis=1) 
y = df['TARGET']
```

    Original Memory Usage: 0.37 gb.
    New Memory Usage: 0.2 gb.

```python
del df
gc.collect()
```


    0

定义数据集评估函数：


```python
def score_dataset(X, y, nfold=5):
    # Create Dataset object for lightgbm
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    
    #  Use a dictionary to set Parameters.
    params = dict(
        objective='binary',
        is_unbalance=True,
        metric='auc',
        n_estimators=500,
        verbose=0
    )
    
    # Training with 5-fold CV:
    print('Starting training...')
    eval_results = lgb.cv(
        params, 
        dtrain, 
        nfold=nfold,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        return_cvbooster=True
    )
    boosters = eval_results['cvbooster'].boosters
    # Initialize an empty dataframe to hold feature importances
    feature_importances = pd.DataFrame(index=X.columns)
    for i in range(nfold):
        feature_importances[f'cv_{i}'] = boosters[i].feature_importance()
    feature_importances['score'] = feature_importances.mean(axis=1)
    # Sort features according to importance
    feature_importances = feature_importances.sort_values('score', ascending=False)
    return eval_results, feature_importances
```

## 简单数学变换

我们可以根据业务含义，创建具有一些明显实际含义的补充特征，例如：


```python
math_features = pd.DataFrame(y)

# 贷款金额相对于收入的比率
math_features['CREDIT_INCOME_PERCENT'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL'] 

# 贷款年金占总收入比率
math_features['ANNUITY_INCOME_PERCENT'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']  

# 以月为单位的付款期限
math_features['CREDIT_TERM'] = X['AMT_ANNUITY'] / X['AMT_CREDIT'] 

#工作时间占年龄的比率
math_features['DAYS_EMPLOYED_PERCENT'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH'] 

# 该用户家庭的人均收入
math_features['INCOME_PER_PERSON'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS'] 
```

我们可以在图形中直观地探索这些新变量：


```python
plt.figure(figsize = (10, 6))
# iterate through the new features
for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
    # create a new subplot for each feature
    plt.subplot(2, 2, i + 1)
    sns.kdeplot(data=math_features, x=feature, hue='TARGET', common_norm=False)    
```


​    
![](/img/feature_engineering_with_python/construction_output_16_0.png)
​    


当然，我们不可能手动计算出所有有实际含义的数学特征。我们可以借助于 featuretools 包找到尽可能多的特征组合进行加减乘除运算。另外，还有对单变量的对数、指数、倒数、平方根、三角函数等运算。

注意：对于二元运算虽然简单，但会出现阶乘式维度爆炸。因此，我们挑选出少数型特征进行简单运算。


```python
features_to_trans = [
    'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 
    'DAYS_ID_PUBLISH', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE',
    'DAYS_REGISTRATION', 'AMT_INCOME_TOTAL', 'REGION_POPULATION_RELATIVE', 
    'ORGANIZATION_TYPE', 'anomaly_score', 'OWN_CAR_AGE', 'OCCUPATION_TYPE', 
    'HOUR_APPR_PROCESS_START', 'CODE_GENDER_M', 'NAME_EDUCATION_TYPE',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
]
```


```python
discrete_to_trans = [f for f in features_to_trans if X[f].nunique()<50]
continuous_to_trans = [f for f in features_to_trans if f not in discrete_to_trans]

print("Selected discrete features:", discrete_to_trans, sep='\n')
print("Selected continuous features:", continuous_to_trans, sep='\n')
```

    Selected discrete features:
    ['OCCUPATION_TYPE', 'HOUR_APPR_PROCESS_START', 'CODE_GENDER_M', 'NAME_EDUCATION_TYPE']
    Selected continuous features:
    ['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'DAYS_ID_PUBLISH', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION', 'AMT_INCOME_TOTAL', 'REGION_POPULATION_RELATIVE', 'ORGANIZATION_TYPE', 'anomaly_score', 'OWN_CAR_AGE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

```python
import featuretools as ft

def math_transform(
    X, y=None, 
    variables=None, 
    func=None, 
    max_depth=2,
    drop_original=True, 
    verbose=False
):
    
    """
    Apply math operators to create new features.
    Parameters
    ----------
    variables: list, default=None
        The list of input variables.
    func: List[string], default=['add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric']
        List of Transform Feature functions to apply.
    drop_original: bool, default=True
        If True, the original variables to transform will be dropped from the dataframe.
    """
    if variables is None:
        variables = X.select_dtypes('number').columns.tolist()
    df = X[variables].copy()
    if func is None:
        func = ['add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric']
    # Make an entityset and add the entity
    es = ft.EntitySet(id = 'single_table')
    es.add_dataframe(dataframe_name='df', dataframe=df, make_index=True, index='id')
    
    # Run deep feature synthesis with transformation primitives
    feature_matrix, features = ft.dfs(
        entityset = es, 
        target_dataframe_name = 'df',
        trans_primitives = func,
        max_depth = max_depth,
        verbose=verbose
    )
    new_features = feature_matrix.drop(variables, axis=1)
    new_features.index = X.index
    if drop_original:
        return new_features 
    else:
        return pd.concat([X, new_features], axis=1)
```


```python
# trans_primitives = [
#     'add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric', 'modulo_numeric',
#     'sine', 'cosine', 'tangent', 'square_root', 'natural_logarithm'
# ]

math_features = math_transform(
    X = X[continuous_to_trans].abs(),
    func = [
        'add_numeric', 'subtract_numeric','divide_numeric',
        'sine', 'square_root', 'natural_logarithm'
    ]
)
print('Math features shape', math_features.shape)
```

    Math features shape (307511, 528)

```python
math_features = convert_dtypes(math_features)
```

    Original Memory Usage: 1.3 gb.
    New Memory Usage: 0.65 gb.


## 分组统计特征衍生

分组统计特征衍生，顾名思义，就是分类特征和连续特征间的分组交互统计，这样可以得到更多有意义的特征，例如：


```python
# Group AMT_INCOME_TOTAL by NAME_INCOME_TYPE and calculate mean, max, min of loans
X.groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].agg(['mean', 'max', 'min']).head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>max</th>
      <th>min</th>
    </tr>
    <tr>
      <th>OCCUPATION_TYPE</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.048303</th>
      <td>194578.359375</td>
      <td>2214117.0</td>
      <td>27000.0</td>
    </tr>
    <tr>
      <th>0.061599</th>
      <td>182842.046875</td>
      <td>1890000.0</td>
      <td>27000.0</td>
    </tr>
    <tr>
      <th>0.062140</th>
      <td>260336.671875</td>
      <td>9000000.0</td>
      <td>27000.0</td>
    </tr>
    <tr>
      <th>0.063040</th>
      <td>172656.687500</td>
      <td>3600000.0</td>
      <td>27000.0</td>
    </tr>
    <tr>
      <th>0.063943</th>
      <td>188916.281250</td>
      <td>699750.0</td>
      <td>30600.0</td>
    </tr>
  </tbody>
</table>
</div>



常用的统计量
|||
|:--|:---|
|var/std|方差、标准差|
|mean/median|均值、中位数|
|max/min|最大值、最小值|
|skew|偏度|
|mode|众数|
|nunique|类别数|
|frequency|频数|
|count|个数|
|quantile|分位数|

> 注意：分组特征必须是离散特征，且最好是一些取值较多的离散变量，这样可以避免新特征出现大量重复取值。分组使用连续值特征时一般需要先进行离散化。

接下来我们自定义一个transformer用来处理数值类型和分类型的分组变量衍生。


```python
from itertools import product, permutations, combinations
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector

class AggFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer to aggregate features in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    ----------
    variables: list, default=None
        The list of input variables. At least one of `variables`, 
    groupby: list, default=None
        The variables to group by. 
    func: function, string, list
        List of Aggregation Feature types to apply.
        Same functionality as parameter `func` in `pandas.agg()`. 
        Build-in func: ['mode', 'kurt'， 'frequency', 'num_unique']
        Default:
        - Numeric: ['median', 'max', 'min', 'skew', 'std']
        - Category: ['mode', 'num_unique', 'frequency']
    n_bins: int, default=10
        The number of bins to produce.
    drop_original: bool, default=True
        If True, the original variables to transform will be dropped from the dataframe.
    """
    
    def __init__(self,
                 variables=None,
                 groupby=None, 
                 func=None, 
                 n_bins=20,
                 drop_original=True):
    
        self.variables = variables
        self.groupby = groupby
        self.func = func
        self.n_bins= n_bins
        self.drop_original = drop_original
    
    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas Series, default=None
            y is not needed in this transformer. You can pass y or None.
        """
        
        # check input dataframe
        # X, y = check_X_y(X, y)
        
        # Get the names and number of features in the train set.
        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]
        
        build_in_funcs = {'mode': self.mode, 
                          'kurt': self.kurt, 
                          'frequency': self.frequency,
                          'num_unique': pd.Series.nunique}
        assert self.func is not None, "Your selected funcs is None."
        self.func = [build_in_funcs.get(f, f) for f in self.func] 
        
        if self.variables is None:
            self.variables = X.columns.tolist()
        
        if self.groupby is None:
            self.groupby = X.columns.tolist()
        return self 

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            A dataframe with the statistics aggregated the selected variables.
            The columns are also renamed to keep track of features created.
        """
        X = X.copy()
        # check if class was fitted
        check_is_fitted(self)
        
        group_df = self.discretize(X[self.groupby], self.n_bins) 
        # Group by the specified variable and calculate the statistics
        n = 0
        for group_var in self.groupby:
            # Skip the grouping variable
            other_vars = [var for var in self.variables if var != group_var]
            for f in self.func:
                # Need to create new column names
                colnames = [f"{f.__name__ if callable(f) else f}({var})_by({group_var})"
                            for var in other_vars]
                X[colnames] = X[other_vars].groupby(group_df[group_var]).transform(f)
                n += len(colnames)
        print(f'Created {n} new features.')
        if self.drop_original:
            X = X.drop(self.feature_names_in_, axis=1)
        
        return X
    
    def mode(self, series):
        return series.mode(dropna=False)[0]
    
    def kurt(self, series):
        return series.kurt()
    
    def frequency(self, series):
        freq = series.value_counts(normalize=True, dropna=False)
        return series.map(freq)
    
    def discretize(self, X, bins=20):
        X = X.copy()
        numeric = X.select_dtypes('number').columns
        continuous = [col for col in numeric if X[col].nunique() >= 50]
        X[continuous] = X[continuous].apply(pd.qcut, q=bins, duplicates="drop")
        X = X.astype('category')
        return X
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        
        if input_features is None:
            feature_names_in = self.feature_names_in_
        elif len(input_features) == self.n_features_in_:
            # If the input was an array, we let the user enter the variable names.
            feature_names_in = list(input_features)
        else:
            raise ValueError(
                "The number of input_features does not match the number of "
                "features seen in the dataframe used in fit."
                ) 
        
        if self.drop_original:
            feature_names_out = []
        else:
            feature_names_out = feature_names_in
        
        func_names = [f.__name__ if callable(f) else f for f in self.func]
        for group_var in feature_names_in:
            # Skip the grouping variable
            other_vars = [var for var in self.variables if var != group_var]
            # Make new column names for the variable and stat
            colnames = [f"{f}({var})_by({group_var})" 
                        for f, var in product(func_names, other_vars)] 
            
            feature_names_out.extend(colnames)
        return feature_names_out
```

数值型特征直接计算统计量，至于分类特征，计算众数、类别数和所属类别的占比。


```python
agg_continuous_transformer = AggFeatures(
    variables = continuous_to_trans,
    func=['mean', 'std'],
    groupby = continuous_to_trans + discrete_to_trans,
    drop_original=True
)

agg_continuous_features = agg_continuous_transformer.fit_transform(X, y)

agg_discrete_transformer = AggFeatures(
    variables = discrete_to_trans,
    func=['mode', 'frequency'],
    groupby = continuous_to_trans + discrete_to_trans,
    drop_original=True
)

agg_discrete_features = agg_discrete_transformer.fit_transform(X, y)

agg_features = pd.concat([agg_continuous_features, agg_discrete_features], axis=1)
print('Aggregation features shape', agg_features.shape)
```

    Created 608 new features.
    Created 152 new features.
    Aggregation features shape (307511, 760)

```python
agg_features = convert_dtypes(agg_features)
del agg_continuous_features, agg_discrete_features
gc.collect()
```

    Original Memory Usage: 1.4 gb.
    New Memory Usage: 0.94 gb.
    
    0

## 特征交互

通过将单独的特征求笛卡尔乘积的方式来组合2个或更多个特征，从而构造出组合特征。最终获得的预测能力可能远远超过任一特征单独的预测能力。笛卡尔乘积组合特征方法一般应用于离散特征之间。


```python
def feature_interaction(X, y=None, left=None, right=None, drop_original=True):
    """
    Parameters
    ----------
    X: pandas dataframe.
    left, right: The list of interact variables. default=None
    drop_original: bool, default=True
        If True, the original variables to transform will be dropped from the dataframe.
    """
    left = X.columns if left is None else left
    right = X.columns if right is None else right
    # Make a new dataframe to hold interaction features
    X_new = pd.DataFrame(index=X.index)
    for rvar in right:
        other_vars = [lvar for lvar in left if lvar !=rvar]
        rseries = X[rvar].astype(str)
        colnames = [f"{lvar}&{rvar}" for lvar in  other_vars]
        X_new[colnames] = X[other_vars].transform(lambda s: s.astype(str) + "&" + rseries)
    if not drop_original:
        X_new = pd.concat([X, X_new], axis=1)
    return X_new

interaction_features = feature_interaction(X[discrete_to_trans])
print(interaction_features.shape)
print(interaction_features.iloc[:5, :2])
```

    (307511, 12)
               HOUR_APPR_PROCESS_START&OCCUPATION_TYPE  \
    SK_ID_CURR                                           
    100002                              10.0&0.1057877   
    100003                            11.0&0.063039534   
    100004                               9.0&0.1057877   
    100006                              17.0&0.1057877   
    100007                            11.0&0.063039534   
    
               CODE_GENDER_M&OCCUPATION_TYPE  
    SK_ID_CURR                                
    100002                     1.0&0.1057877  
    100003                   0.0&0.063039534  
    100004                     1.0&0.1057877  
    100006                     0.0&0.1057877  
    100007                   1.0&0.063039534  

```python
del interaction_features
gc.collect()
```


    0



## 多项式特征

多项式特征是 sklearn 中特征构造的最简单方法。当我们创建多项式特征时，尽量避免使用过高的度数，因为特征的数量随着度数指数级地变化，并且可能过拟合。

现在我们使用3度多项式来查看结果：


```python
from sklearn.preprocessing import PolynomialFeatures

# Make a new dataframe for polynomial features
X_poly = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)
 
# Train and transform the polynomial features
poly_features = poly_transformer.fit_transform(X_poly)
print('Polynomial features shape: ', poly_features.shape)
print('Polynomial features:', poly_transformer.get_feature_names_out()[-5:])
```

    Polynomial features shape:  (307511, 35)
    Polynomial features: ['EXT_SOURCE_2 DAYS_BIRTH^2' 'EXT_SOURCE_3^3' 'EXT_SOURCE_3^2 DAYS_BIRTH'
     'EXT_SOURCE_3 DAYS_BIRTH^2' 'DAYS_BIRTH^3']

```python
del poly_features
gc.collect()
```


    0



## 聚类分析

聚类算法在特征构造中的应用有不少，例如：利用聚类算法对文本聚类，使用聚类类标结果作为输入特征；利用聚类算法对单个数值特征进行聚类，相当于分箱；利用聚类算法对R、F、M数据进行聚类，类似RFM模型，然后再使用代表衡量客户价值的聚类类标结果作为输入特征。

当一个或多个特征具有多峰分布（有两个或两个以上清晰的峰值）时，可以使用聚类算法为每个峰值分类，并输出聚类类标结果。


```python
age_simi = X[['DAYS_BIRTH']]/-365
age_simi.columns=['age']

plt.figure(figsize=(5,4))
sns.histplot(x=age_simi['age'], bins=30)
```


    <Axes: xlabel='age', ylabel='Count'>


​    
![](/img/feature_engineering_with_python/construction_output_37_1.png)
​    


可以看到有两个峰值：40和55

一般聚类类标结果为一个数值，但实际上这个数值并没有大小之分，所以一般需要进行one-hot编码，或者创建新特征来度量样本和每个类中心的相似性（距离）。相似性度量通常使用径向基函数(RBF)来计算。

径向基函数（Radial Basis Function，简称RBF）是一种在机器学习和统计建模中常用的函数类型。它们以距离某个中心点的距离作为输入，并输出一个数值，通常表示为距离的“衰减函数”。最常用的RBF是高斯RBF，其输出值随着输入值远离固定点而呈指数衰减。高斯RBF可以使用Scikit-Learn的rbf_kernel()函数计算 $k(x,y)=\exp(-\gamma\|x-y\|^2)$，超参数 gamma 确定当x远离y时相似性度量衰减的速度。

下图显示了年龄的两个径向基函数：


```python
from sklearn.metrics.pairwise import rbf_kernel

age_simi[['simi_40', 'simi_55']] = rbf_kernel(age_simi[['age']], [[40],[55]], gamma=0.01)

fig = plt.figure(figsize=(5,4))
ax1 = fig.add_subplot(111)
sns.histplot(data=age_simi, x='age', bins=30, ax=ax1)

ax2 = ax1.twinx()
sns.lineplot(data=age_simi, x='age', y='simi_40', ci=None, ax=ax2, color='green')
sns.lineplot(data=age_simi, x='age', y='simi_55', ci=None, ax=ax2, color='orange')
```


    <Axes: xlabel='age', ylabel='simi_40'>


![](/img/feature_engineering_with_python/construction_output_39_1.png)
    


> 如果你给rbf_kernel()函数传递一个有两个特征的数组，它会测量二维距离(欧几里得)来测量相似性。

如果这个特定的特征与目标变量有很好的相关性，那么这个新特征将有很好的机会发挥作用。

接下来，我们自定义一个转换器，该转换器在fit()方法中使用KMeans聚类器来识别训练数据中的主要聚类，然后在transform()方法中使用`rbf_kernel()` 来衡量每个样本与每个聚类中心的相似程度：


```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, gamma=1.0, standardize=True, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.standardize = standardize
        self.random_state = random_state
    def fit(self, X, y=None):
        # Standardize
        if self.standardize:
            self.scaler = RobustScaler()
            X = self.scaler.fit_transform(X)
        self.kmeans = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X)
        # Coordinates of cluster centers.
        self.cluster_centers_ = self.kmeans.cluster_centers_
        return self 
    def transform(self, X):
        X = self.scaler.transform(X)
        return rbf_kernel(X, self.cluster_centers_, gamma=self.gamma)
    def get_feature_names_out(self, input_features=None):
        return [f"centroid_{i}_similarity" for i in range(self.n_clusters)]
```

现在让我们使用这个转换器：


```python
cluster = ClusterSimilarity(n_clusters=5, gamma=0.01, random_state=SEED)

cluster_similarities = cluster.fit_transform(X)
cluster_labels = cluster.kmeans.predict(X)

cluster_similarities = pd.DataFrame(
    cluster_similarities,
    columns = cluster.get_feature_names_out(),
    index = X.index
)

print(cluster_similarities.shape)
```

    (307511, 5)

```python
cluster_similarities = convert_dtypes(cluster_similarities)
```

    Original Memory Usage: 0.01 gb.
    New Memory Usage: 0.01 gb.

```python
del age_simi, cluster_labels
gc.collect()
```


    150

## featuretools

在许多情况下数据分布在多个表中，而机器学习模型必须用单个表进行训练，因此特征工程要求我们将所有数据汇总到一个表中。这些数据集，通常基于 id 值连接，我们通过计算数值特征的一些统计量来合并。featuretools 包能很方便的自动完成这些任务。

featuretools 涉及到3个概念：实体(entity)、关系(relationship)和算子(primitive)。

- 所谓的实体就是一张表或者一个dataframe，多张表的集合就叫实体集(entityset)。
- 关系就是表之间的关联键的定义。
- 而算子就是一些特征工程的函数。有应用于实体集的聚合操作(Aggregation primitives)和应用于单个实体的转换操作(Transform primitives)两种。`featuretools.list_primitives()` 方法可以列出支持的操作。

> 本项目主要是基于application表的特征工程，对其他数据集只做简单的处理


```python
import featuretools as ft

# Load other datasets
print("Loading...")
app = X.reset_index()
bureau = pd.read_csv('../datasets/Home-Credit-Default-Risk/bureau.csv')
bureau_balance = pd.read_csv('../datasets/Home-Credit-Default-Risk/bureau_balance.csv')
cash = pd.read_csv('../datasets/Home-Credit-Default-Risk/POS_CASH_balance.csv')
credit = pd.read_csv('../datasets/Home-Credit-Default-Risk/credit_card_balance.csv')
previous = pd.read_csv('../datasets/Home-Credit-Default-Risk/previous_application.csv')
installments = pd.read_csv('../datasets/Home-Credit-Default-Risk/installments_payments.csv')

# Empty entity set with id applications
es = ft.EntitySet(id = 'clients')

# Entities with a unique index
es = es.add_dataframe(dataframe_name= 'app', dataframe = app, 
index = 'SK_ID_CURR')
es = es.add_dataframe(dataframe_name= 'bureau', dataframe = bureau, 
index = 'SK_ID_BUREAU')
es = es.add_dataframe(dataframe_name= 'previous', dataframe = previous, 
index = 'SK_ID_PREV')

# Entities that do not have a unique index
es = es.add_dataframe(dataframe_name= 'bureau_balance', dataframe = bureau_balance, 
    make_index = True, index = 'bureau_balance_index')
es = es.add_dataframe(dataframe_name= 'cash', dataframe = cash, 
    make_index = True, index = 'cash_index')
es = es.add_dataframe(dataframe_name= 'installments', dataframe = installments,
    make_index = True, index = 'installments_index')
es = es.add_dataframe(dataframe_name= 'credit', dataframe = credit,
    make_index = True, index = 'credit_index')

# add Relationship
es = es.add_relationship('app', 'SK_ID_CURR', 'bureau', 'SK_ID_CURR')
es = es.add_relationship('bureau', 'SK_ID_BUREAU', 'bureau_balance', 'SK_ID_BUREAU')
es = es.add_relationship('app','SK_ID_CURR', 'previous', 'SK_ID_CURR')
es = es.add_relationship('previous', 'SK_ID_PREV', 'cash', 'SK_ID_PREV')
es = es.add_relationship('previous', 'SK_ID_PREV', 'installments', 'SK_ID_PREV')
es = es.add_relationship('previous', 'SK_ID_PREV', 'credit', 'SK_ID_PREV')

# Default primitives from featuretools
agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
trans_primitives =  ["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"]
```

    Loading...

```python
# DFS with specified primitives
feature_matrix, feature_defs = ft.dfs(
    entityset = es, 
    target_dataframe_name = 'app',
    agg_primitives=['sum', 'max', 'mean', 'min', 'count', 'mode'],
    max_depth = 2, 
    verbose = 1
)

print('After DFS, features shape: ', feature_matrix.shape)
# view last 10 features
print(feature_defs[-10:])
```

    Built 1242 features
    Elapsed: 31:13 | Progress: 100%|████████████████████████████████████████████████
    After DFS, features shape:  (307511, 1242)
    [<Feature: SUM(credit.previous.DAYS_LAST_DUE)>, <Feature: SUM(credit.previous.DAYS_LAST_DUE_1ST_VERSION)>, <Feature: SUM(credit.previous.DAYS_TERMINATION)>, <Feature: SUM(credit.previous.HOUR_APPR_PROCESS_START)>, <Feature: SUM(credit.previous.NFLAG_INSURED_ON_APPROVAL)>, <Feature: SUM(credit.previous.NFLAG_LAST_APPL_IN_DAY)>, <Feature: SUM(credit.previous.RATE_DOWN_PAYMENT)>, <Feature: SUM(credit.previous.RATE_INTEREST_PRIMARY)>, <Feature: SUM(credit.previous.RATE_INTEREST_PRIVILEGED)>, <Feature: SUM(credit.previous.SELLERPLACE_AREA)>]

```python
feature_matrix = convert_dtypes(feature_matrix)
```

    Original Memory Usage: 2.9 gb.
    New Memory Usage: 1.46 gb.

```python
feature_matrix.dtypes.value_counts()
```


    float32     1154
    int32         16
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       2
    category       2
    category       2
    category       2
    category       2
    category       2
    Name: count, dtype: int64


```python
del app, bureau, bureau_balance, cash, credit, previous, installments
gc.collect()
```


    0



## 主成分分析

由于我们新增的这些特征都是和原始特征高度相关，可以使用PCA的主成分作为新的特征来消除相关性。

由于主成分分析主要用于降维，我们将在后续特征选择部分详细介绍。

## 小结

合并之前创造的特征


```python
# Combine datasets
X_created = pd.concat(
    [feature_matrix, math_features, agg_features, cluster_similarities], 
    axis=1
)
```


```python
X_created = convert_dtypes(X_created)
X_created.dtypes.value_counts()
```

    Original Memory Usage: 3.05 gb.
    New Memory Usage: 3.05 gb.
    
    float32     2447
    int32         16
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       4
    category       2
    category       2
    category       2
    category       2
    category       2
    category       2
    Name: count, dtype: int64


```python
del X, feature_matrix, math_features, agg_features, cluster_similarities
gc.collect()
```


    0


```python
def drop_missing_data(X, threshold=0.8):
    X = X.copy()
    # Remove variables with missing more than threshold(default 20%)
    thresh = int(X.shape[0] * threshold)
    X_new = X.dropna(axis=1, thresh=thresh)
    print(f"Removed {X.shape[1]-X_new.shape[1]} variables with missing more than {1 - threshold:.1%}")
    return X_new

# Simple imputer

def impute_simply(X, threshold=0.8):
    """
    Univariate imputer for completing missing values with simple strategies.
    """
    print("Simple imputer:")
    X = X.copy()
    variables = X.columns[X.isna().mean().between(0, 1-threshold, "right")].tolist()
    features_num = X[variables].select_dtypes('number').columns.to_list()
    features_cat = X[variables].select_dtypes(exclude='number').columns.to_list()
    # Replaces missing values by the median or mode
    medians = X[features_num].median().to_dict()
    modes = X[features_cat].apply(lambda x: x.mode()[0]).to_dict()
    impute_dict = {**medians, **modes}
    X[variables] = X[variables].fillna(impute_dict)
    print(f"Transformed {len(variables)} variables with missing (threshold={threshold:.1%}).")
    print(f"And then, there are {X.isna().sum().gt(0).sum()} variables with missing.")
    return X
```


```python
# Handle missing values
X_created = X_created.replace(np.inf, np.nan).replace(-np.inf, np.nan)

X_created = drop_missing_data(X_created, threshold=0.8)
X_created = impute_simply(X_created, threshold=0.8)

print("Dataset shape:", X_created.shape)
```

    Removed 368 variables with missing more than 20.0%
    Simple imputer:
    Transformed 551 variables with missing (threshold=80.0%).
    And then, there are 0 variables with missing.
    Dataset shape: (307511, 2167)


我们继续使用LightGBM模型评估创造的新特征


```python
def score_dataset(X, y, nfold=5):
    # Create Dataset object for lightgbm
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    
    #  Use a dictionary to set Parameters.
    params = dict(
        objective='binary',
        is_unbalance=True,
        metric='auc',
        n_estimators=500,
        verbose=0
    )
    
    # Training with 5-fold CV:
    print('Starting training...')
    eval_results = lgb.cv(
        params, 
        dtrain, 
        nfold=nfold,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        return_cvbooster=True
    )
    boosters = eval_results['cvbooster'].boosters
    # Initialize an empty dataframe to hold feature importances
    feature_importances = pd.DataFrame(index=X.columns)
    for i in range(nfold):
        # Record the feature importances
        feature_importances[f'cv_{i}'] = boosters[i].feature_importance()
    feature_importances['score'] = feature_importances.mean(axis=1)
    # Sort features according to importance
    feature_importances = feature_importances.sort_values('score', ascending=False)
    return eval_results, feature_importances
```


```python
eval_results, feature_importances = score_dataset(X_created, y)
```

    Starting training...
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    Training until validation scores don't improve for 50 rounds
    [50]	cv_agg's valid auc: 0.778018 + 0.00319843
    [100]	cv_agg's valid auc: 0.783267 + 0.00307558
    [150]	cv_agg's valid auc: 0.783211 + 0.00299384
    Early stopping, best iteration is:
    [115]	cv_agg's valid auc: 0.783392 + 0.00298777


特征重要性


```python
feature_importances['score'].head(20)
```


    EXT_SOURCE_2 + EXT_SOURCE_3                            50.8
    MODE(previous.PRODUCT_COMBINATION)                     43.4
    AMT_CREDIT / AMT_ANNUITY                               42.6
    AMT_ANNUITY / AMT_CREDIT                               39.8
    MODE(installments.previous.PRODUCT_COMBINATION)        34.0
    MAX(bureau.DAYS_CREDIT)                                32.0
    DAYS_BIRTH / EXT_SOURCE_1                              29.2
    SUM(bureau.AMT_CREDIT_MAX_OVERDUE)                     27.0
    MAX(bureau.DAYS_CREDIT_ENDDATE)                        26.2
    SUM(bureau.AMT_CREDIT_SUM)                             26.0
    MEAN(previous.MIN(installments.AMT_PAYMENT))           25.4
    AMT_CREDIT - AMT_GOODS_PRICE                           25.2
    MAX(cash.previous.DAYS_LAST_DUE_1ST_VERSION)           25.0
    MEAN(bureau.AMT_CREDIT_SUM_DEBT)                       24.6
    DAYS_BIRTH - DAYS_ID_PUBLISH                           23.6
    MODE(cash.previous.PRODUCT_COMBINATION)                22.6
    MODE(previous.NAME_GOODS_CATEGORY)                     22.2
    MIN(installments.AMT_PAYMENT)                          22.0
    SUM(previous.MEAN(credit.CNT_DRAWINGS_ATM_CURRENT))    21.2
    SUM(previous.MIN(installments.AMT_PAYMENT))            20.6
    Name: score, dtype: float64

保存数据集


```python
created_data = pd.concat([
    X_created, 
    y
    ], axis=1
)

del X_created
gc.collect()

created_data.to_csv('../datasets/Home-Credit-Default-Risk/created_data.csv', index=True)

print(f'Memory Usage: {sys.getsizeof(created_data) / 1e9:.2f} gb')
```

    Memory Usage: 2.63 gb
