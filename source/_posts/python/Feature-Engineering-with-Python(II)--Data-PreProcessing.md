---
title: 特征工程(II)--数据预处理
tags:
  - Python
categories:
  - Python
  - 'Machine Learning'
cover: /img/FeatureEngine.png
top_img: /img/sklearn-top-img.svg
abbrlink: ce19bbb1
description: 
date: 2024-03-16 23:40:52
---

有这么一句话在业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。由此可见，特征工程在机器学习中占有相当重要的地位。在实际应用当中，可以说特征工程是机器学习成功的关键。

特征工程是数据分析中最耗时间和精力的一部分工作，它不像算法和模型那样是确定的步骤，更多是工程上的经验和权衡。因此没有统一的方法。这里只是对一些常用的方法做一个总结。

特征工程包含了 Data PreProcessing（数据预处理）、Feature Extraction（特征提取）、Feature Selection（特征选择）和 Feature construction（特征构造）等子问题。

Jupyter Notebook 代码连接：[feature_engineering_demo_p2_preproccessing](/ipynb/feature_engineering_demo_p2_preproccessing)

导入必要的包


```python
import numpy as np
import pandas as pd
import re
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

# 数据预处理

数据预处理是特征工程的最重要的起始步骤，需要把特征预处理成机器学习模型所能接受的形式，我们可以使用sklearn.preproccessing模块来解决大部分数据预处理问题。

本章使用两条线并行处理数据：

- 基于pandas的函数封装实现
- 基于sklearn的pipeline实现

先定义一个计时器，方便后续评估性能。


```python
def timer(func):
    import time
    import functools
    def strfdelta(tdelta, fmt):
        hours, remainder = divmod(tdelta, 3600)
        minutes, seconds = divmod(remainder, 60)
        return fmt.format(hours, minutes, seconds)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        click = time.time()
        result = func(*args, **kwargs)
        delta = strfdelta(time.time() - click, "{:.0f} hours {:.0f} minutes {:.0f} seconds")
        print(f"{func.__name__} cost time {delta}")
        return result
    return wrapper
```

## 数据清洗

数据清洗(Data cleaning)是对数据进行重新审查和校验的过程，目的在于删除重复信息、纠正存在的错误，并提供数据一致性。

### 数据去重

首先，根据某个/多个特征值构成的样本ID去重


```python
df = pd.read_csv('../datasets/Home-Credit-Default-Risk/application_train.csv')
```


```python
if df['SK_ID_CURR'].nunique() < df.shape[0]:
    df = df.drop_duplicates(subset=['SK_ID_CURR'], keep='last')
```

### 数据类型转换

字符型数字自动转成数字


```python
df.dtypes.value_counts()
```


    float64    65
    int64      41
    object     16
    Name: count, dtype: int64


```python
df.apply(pd.to_numeric,  errors='ignore').dtypes.value_counts()
```


    float64    65
    int64      41
    object     16
    Name: count, dtype: int64

有时，有些数值型特征标识的只是不同类别，其数值的大小并没有实际意义，因此我们将其转化为类别特征。  
本项目并无此类特征，以 hours_appr_process_start 为示例：


```python
# df['HOUR_APPR_PROCESS_START '] = df['HOUR_APPR_PROCESS_START'].astype(str)
```

### 错误数据清洗

接下来，我们根据业务常识，或者使用但不限于箱型图（Box-plot）发现数据中不合理的特征值进行清洗。
数据探索时，我们注意到，DAYS_BIRTH列（年龄）中的数字是负数，由于它们是相对于当前贷款申请计算的，所以我们将其转化成正数后查看分布


```python
(df['DAYS_BIRTH'] / -365).describe()
```


    count    307511.000000
    mean         43.936973
    std          11.956133
    min          20.517808
    25%          34.008219
    50%          43.150685
    75%          53.923288
    max          69.120548
    Name: DAYS_BIRTH, dtype: float64

那些年龄看起来合理，没有异常值。
接下来，我们对其他的 DAYS 特征作同样的分析


```python
for feature in ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']:
        print(f'{feature} info: ')
        print((df[feature] / -365).describe() )
```

    DAYS_BIRTH info: 
    count    307511.000000
    mean         43.936973
    std          11.956133
    min          20.517808
    25%          34.008219
    50%          43.150685
    75%          53.923288
    max          69.120548
    Name: DAYS_BIRTH, dtype: float64
    DAYS_EMPLOYED info: 
    count    307511.000000
    mean       -174.835742
    std         387.056895
    min       -1000.665753
    25%           0.791781
    50%           3.323288
    75%           7.561644
    max          49.073973
    Name: DAYS_EMPLOYED, dtype: float64
    DAYS_REGISTRATION info: 
    count    307511.000000
    mean         13.660604
    std           9.651743
    min          -0.000000
    25%           5.506849
    50%          12.339726
    75%          20.491781
    max          67.594521
    Name: DAYS_REGISTRATION, dtype: float64
    DAYS_ID_PUBLISH info: 
    count    307511.000000
    mean          8.203294
    std           4.135481
    min          -0.000000
    25%           4.712329
    50%           8.915068
    75%          11.778082
    max          19.717808
    Name: DAYS_ID_PUBLISH, dtype: float64

```python
pd.cut(df['DAYS_EMPLOYED'] / -365, bins=5).value_counts().sort_index()
```


    DAYS_EMPLOYED
    (-1001.715, -790.718]     55374
    (-790.718, -580.77]           0
    (-580.77, -370.822]           0
    (-370.822, -160.874]          0
    (-160.874, 49.074]       252137
    Name: count, dtype: int64

有超过50000个用户的DAYS_EMPLOYED在1000年上，可以猜测这只是缺失值标记。


```python
# Replace the anomalous values with nan
df_emp = df['DAYS_EMPLOYED'].where(df['DAYS_EMPLOYED'].abs()<365243, np.nan)

df_emp.plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')
```


    Text(0.5, 0, 'Days Employment')


![](/img/feature_engineering_with_python/preproccessing_output_20_1.png)
    


可以看到，数据分布基本正常了。    
同样，将其他特征的缺失值标记转换成缺失，方便后续统一处理


```python
df.map(lambda x:x=='XNA').sum().sum()/df.size
```


    0.0014761034004861135

### 布尔特征清洗


```python
for col in df.select_dtypes(exclude="number").columns:
    if df[col].nunique() == 2:
        print(df[col].value_counts())
```

    NAME_CONTRACT_TYPE
    Cash loans         278232
    Revolving loans     29279
    Name: count, dtype: int64
    FLAG_OWN_CAR
    N    202924
    Y    104587
    Name: count, dtype: int64
    FLAG_OWN_REALTY
    Y    213312
    N     94199
    Name: count, dtype: int64
    EMERGENCYSTATE_MODE
    No     159428
    Yes      2328
    Name: count, dtype: int64

```python
df[['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']].replace({'Y': 1, 'N': 0}).value_counts()
```


    FLAG_OWN_CAR  FLAG_OWN_REALTY
    0             1                  140952
    1             1                   72360
    0             0                   61972
    1             0                   32227
    Name: count, dtype: int64


```python
df['EMERGENCYSTATE_MODE'].replace({'Yes': 1, 'No': 0}).value_counts()
```


    EMERGENCYSTATE_MODE
    0.0    159428
    1.0      2328
    Name: count, dtype: int64

### 函数封装

最后，使用函数封装以上步骤：


```python
id_col = "SK_ID_CURR"
target = "TARGET"

# Data cleaning
def clean(df):
    # remove duplicates and keep last occurrences
    if df[id_col].nunique() < df.shape[0]:
        df = df.drop_duplicates(subset=[id_col], keep='last')
    
    # convert data to specified dtypes
    df = df.apply(pd.to_numeric,  errors='ignore')
    
    # transform
    for col in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EMERGENCYSTATE_MODE']:
        df[col] = df[col].replace({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0})
    
    # Replace the anomalous values with nan
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].where(df['DAYS_EMPLOYED'].abs()<365243, np.nan)
    df = df.replace('XNA', np.nan)
    
    X = df.drop([id_col, target], axis=1)
    y = df[target]
    return X, y

X, y = clean(df)
```

## 缺失值处理

特征有缺失值是非常常见的，大部分机器学习模型在拟合前需要处理缺失值（Handle Missing Values）。

| 缺失值处理方法  | 函数                   | python包       |
| --------------- | ---------------------- | -------------- |
| 统计量插补      | SimpleImputer          | sklearn.impute |
| 统计量/随机插补 | df.fillna()            | pandas         |
| 多重插补        | IterativeImputer       | sklearn.impute |
| 最近邻插补      | KNNImputer             | sklearn.impute |
| 缺失值删除      | df.dropna()            | pandas         |
| 缺失值标记      | MissingIndicator       | sklearn.impute |
| 缺失值标记      | df.isna(), df.isnull() | pandas         |

### 缺失值统计


```python
# Function to calculate missing values by column
def display_missing(df, threshold=None, verbose=True):
    missing_df = pd.DataFrame({
        "missing_number": df.isna().sum(),  # Total missing values
        "missing_rate": df.isna().mean()   # Proportion of missing values
        }, index=df.columns)
    missing_df = missing_df.query("missing_rate>0").sort_values("missing_rate", ascending=False)
    threshold = 0.25 if threshold is None else threshold
    high_missing = missing_df.query(f"missing_rate>{threshold}")
    # Print some summary information
    if verbose:
        print(f"Your selected dataframe has {missing_df.shape[0]} out of {df.shape[1]} columns that have missing values.")
        print(f"There are {high_missing.shape[0]} columns with more than {threshold:.1%} missing values.")
        print("Columns with high missing rate:", high_missing.index.tolist())
    # Return the dataframe with missing information
    if threshold is None:
        return missing_df
    else:
        return high_missing
```


```python
# Missing values statistics
print(display_missing(df).head(10))
```

    Your selected dataframe has 67 out of 122 columns that have missing values.
    There are 50 columns with more than 25.0% missing values.
    Columns with high missing rate: ['COMMONAREA_MEDI', 'COMMONAREA_AVG', 'COMMONAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAPARTMENTS_AVG', 'FONDKAPREMONT_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAPARTMENTS_AVG', 'FLOORSMIN_MODE', 'FLOORSMIN_MEDI', 'FLOORSMIN_AVG', 'YEARS_BUILD_MODE', 'YEARS_BUILD_MEDI', 'YEARS_BUILD_AVG', 'OWN_CAR_AGE', 'LANDAREA_AVG', 'LANDAREA_MEDI', 'LANDAREA_MODE', 'BASEMENTAREA_MEDI', 'BASEMENTAREA_AVG', 'BASEMENTAREA_MODE', 'EXT_SOURCE_1', 'NONLIVINGAREA_MEDI', 'NONLIVINGAREA_MODE', 'NONLIVINGAREA_AVG', 'ELEVATORS_MEDI', 'ELEVATORS_MODE', 'ELEVATORS_AVG', 'WALLSMATERIAL_MODE', 'APARTMENTS_MODE', 'APARTMENTS_MEDI', 'APARTMENTS_AVG', 'ENTRANCES_MODE', 'ENTRANCES_AVG', 'ENTRANCES_MEDI', 'LIVINGAREA_MEDI', 'LIVINGAREA_MODE', 'LIVINGAREA_AVG', 'HOUSETYPE_MODE', 'FLOORSMAX_MEDI', 'FLOORSMAX_AVG', 'FLOORSMAX_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BEGINEXPLUATATION_MODE', 'TOTALAREA_MODE', 'EMERGENCYSTATE_MODE', 'OCCUPATION_TYPE']
                              missing_number  missing_rate
    COMMONAREA_MEDI                   214865      0.698723
    COMMONAREA_AVG                    214865      0.698723
    COMMONAREA_MODE                   214865      0.698723
    NONLIVINGAPARTMENTS_MEDI          213514      0.694330
    NONLIVINGAPARTMENTS_MODE          213514      0.694330
    NONLIVINGAPARTMENTS_AVG           213514      0.694330
    FONDKAPREMONT_MODE                210295      0.683862
    LIVINGAPARTMENTS_MODE             210199      0.683550
    LIVINGAPARTMENTS_MEDI             210199      0.683550
    LIVINGAPARTMENTS_AVG              210199      0.683550


### 可视化缺失率


```python
high_missing = display_missing(X, verbose=False).head(10)

plt.figure(figsize = (8, 4))
sns.barplot(x="missing_rate", y=high_missing.index, data=high_missing)
plt.xlabel('Percent of missing values')
plt.ylabel('Features')
plt.title('Percent missing data by feature')
```


    Text(0.5, 1.0, 'Percent missing data by feature')


![](/img/feature_engineering_with_python/preproccessing_output_33_1.png)
    


### 缺失值删除 

如果某个特征的缺失值超过阈值（例如80%），那么该特征对模型的贡献就会降低，通常就可以考虑删除该特征。


```python
threshold = int(df.shape[0]*0.2)
X.dropna(axis=1, thresh=threshold).shape
```


    (307511, 120)


```python
# Remove variables with high missing rate

def drop_missing_data(X, threshold=0.8):
    X = X.copy()
    # Remove variables with missing more than threshold(default 20%)
    thresh = int(X.shape[0] * threshold)
    X_new = X.dropna(axis=1, thresh=thresh)
    print(f"Removed {X.shape[1]-X_new.shape[1]} variables with missing more than {1 - threshold:.1%}")
    return X_new
```


```python
drop_missing_data(X, threshold=0.2).shape
```

    Removed 0 variables with missing more than 80.0%
    
    (307511, 120)

### 缺失值标记

有时，对于每个含有缺失值的列，我们额外添加一列来表示该列中缺失值的位置，在某些应用中，能取得不错的效果。
继续分析之前清洗过的 DAYS_EMPLOYED 异常，我们对缺失数据进行标记，看看他们是否影响客户违约。


```python
y.groupby(X['DAYS_EMPLOYED'].isna()).mean()
```


    DAYS_EMPLOYED
    False    0.086600
    True     0.053996
    Name: TARGET, dtype: float64

发现缺失值的逾期率 5.4% 低于正常值的逾期率 8.66%，与Target的相关性很强，因此新增一列DAYS_EMPLOYED_MISSING 标记。这种处理对线性方法比较有效，而基于树的方法可以自动识别。


```python
# Adds a binary variable to flag missing observations.

from sklearn.feature_selection import chi2
def flag_missing(X, alpha=0.05):
    """
    Adds a binary variable to flag missing observations(one indicator per variable). 
    The added variables (missing indicators) are named with the original variable name plus '_missing'.
    
    Parameters:
    ----------
    alpha: float, default=0.05
        Features with p-values more than alpha are selected.
    """
    X = X.copy()
    
    # Compute chi-squared stats between each missing indicator and y.
    chi2_stats, p_values = chi2(X.isna(), y)
    # find variables for which indicator should be added.
    missing_indicator = X.loc[:, p_values > alpha]
    indicator_names = missing_indicator.columns.map(lambda x: x + "_missing")
    X[indicator_names] = missing_indicator
    print(f"Added {missing_indicator.shape[1]} missing indicators")
    return X
```


```python
flag_missing(X).shape
```

    Added 6 missing indicators
    
    (307511, 126)

### 人工插补

根据业务知识来进行人工填充。 

若变量是离散型，且不同值较少，可在编码时转换成哑变量。例如，性别变量 code_gender


```python
pd.get_dummies(X["CODE_GENDER"], dummy_na=True).head()
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
      <th>F</th>
      <th>M</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



若变量是布尔型，视情况可统一填充为零


```python
len([col for col in X if set(X[col].unique()) == {0, 1}])
```


    34

如果我们仔细观察一下字段描述，会发现很多缺失值都有迹可循，比如name_type_suite缺失，表示办理贷款的时候无人陪同，因此可以用 unknow 来填补。客户的社会关系中有30天/60天逾期及申请贷款前1小时/天/周/月/季度/年查询了多少次征信的都可填充为数字0。


```python
def impute_manually(X):
    """
    Replaces missing values by an arbitrary value
    """
    X = X.copy()
    # boolean
    boolean_features = [col for col in X if set(X[col].unique()) == {0, 1}]
    X[boolean_features] = X[boolean_features].fillna(0)
    # fill none
    features_fill_none = ["NAME_TYPE_SUITE", "OCCUPATION_TYPE"]
    X[features_fill_none] = X[features_fill_none].fillna('unknow')
    # fill 0
    features_fill_zero = [
        "OBS_30_CNT_SOCIAL_CIRCLE",  
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR"
    ]
    X[features_fill_zero] = X[features_fill_zero].fillna(0)
    return X
```


```python
impute_manually(X).isna().sum().gt(0).sum()
```


    58

### 条件平均值填充法

通过之前的相关分析，我们知道AMT_ANNUITY这个特征与AMT_CREDIT和AMT_INCOME_TOTAL有比较大的关系，所以这里用这两个特征分组后的中位数进行插补，称为条件平均值填充法（Conditional Mean Completer）。


```python
X[['AMT_CREDIT', 'AMT_INCOME_TOTAL']].corrwith(X["AMT_ANNUITY"])
```


    AMT_CREDIT          0.770138
    AMT_INCOME_TOTAL    0.191657
    dtype: float64


```python
# conditional statistic completer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import f_classif, chi2
from sklearn.feature_selection import r_regression, f_regression

def fillna_by_groups(X, threshold=(0.0, 0.8), groupby=None, k=2, min_categories=2, bins=10, verbose=False):
    """
    Replaces missing values by groups.
    
    threshold: float, default=None
        Require that percentage of non-NA values in a column to impute.
    k: int, default=2
        Number of top features to group by.
    min_categories: int, default=2
        Specifies an lower limit to the number of categories for each feature to group by.
    bins: int, default=10
    """
    print("Conditional Mean Completer:")
    lower, upper = threshold
    if 0 <= lower < upper <= 1:
        pass
    else:
        raise ValueError("threshold must be a value between 0 < x <= 1. ")
    
    X = pd.DataFrame(X.copy())
    X_bin = X.copy()
    na_size = X.isna().sum().sum()
    
    features_num = X.select_dtypes(include='number').columns.tolist()
    features_cat = X.select_dtypes(exclude='number').columns.tolist()
    X_bin[features_num] = X_bin[features_num].apply(pd.qcut, q=bins, duplicates="drop")
    X_bin[features_cat] = X_bin[features_cat].astype('category')
    X_bin = X_bin.transform(lambda x: x.cat.codes)
    X_bin = X_bin.transform(lambda x: x - x.min()) # for chi-squared to stats each non-negative feature
    
    if groupby is None:
        features_groupby = X_bin.columns.tolist()
    features_groupby = [colname for colname in features_groupby 
                        if X[colname].nunique()>=min_categories]
    
    # Estimate mutual information for a target variable.
    variables = X.columns[X.notna().mean().between(lower, upper)].tolist()
    for colname in variables:
        other_features = list(set(features_groupby) - {colname})
        if colname in features_num:
            score_func = f_regression
        elif colname in features_cat:
            score_func = chi2
        Xy = pd.concat([X_bin[other_features], X[colname]], axis=1).dropna(axis=0,how='any')
        scores, _ = score_func(Xy[other_features], Xy[colname])
        scores = pd.Series(scores, index=other_features).sort_values(ascending=False)
        vars_top_k = scores[:k].index.tolist()
        groups = [X_bin[col] for col in vars_top_k]
        if colname in features_num:
            # Replaces missing values by the mean or median
            X[colname] = X.groupby(groups)[colname].transform(lambda x:x.fillna(x.median()))
            if verbose:
                print(f"Filling the missing values in {colname} with the medians of {vars_top_k} groups.")
        elif colname in features_cat:
            # Replaces missing values by the most frequent category
            X[colname] = X[colname].groupby(groups).transform(lambda x:x.fillna(x.mode(dropna=False)[0]))
            if verbose:
                print(f"Filling the missing values in {colname} with the modes of {vars_top_k} groups.")

    fillna_size = na_size - X.isna().sum().sum()
    print(f"Filled {fillna_size} missing values ({fillna_size/na_size:.1%}).")
    print(f"Transformed {len(variables)} variables with missing (threshold = [{lower:.1%}, {upper:.1%}]).")
    print(f"And then, there are {X.isna().sum().gt(0).sum()} variables with missing.")
    return X
```


```python
fillna_by_groups(X).isna().sum().sum()
```

    Conditional Mean Completer:
    Filled 759918 missing values (8.2%).
    Transformed 50 variables with missing (threshold = [0.0%, 80.0%]).
    And then, there are 67 variables with missing.
    
    8503299

### 简单插补

对于缺失率较低（小于10%）的数值特征可以用中位数或均值插补，缺失率较低（小于10%）的离散型特征，则可以用众数插补。


```python
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
impute_simply(X).isna().sum().gt(0).sum()
```

    Simple imputer:
    Transformed 20 variables with missing (threshold=80.0%).
    And then, there are 50 variables with missing.
    
    50

### Pipeline实现

最后，总结下我们的缺失处理策略：

- 删除缺失率高于80%特征
- 添加缺失标记
- 有业务含义的进行人工插补
- 缺失率10-80%的特征多重插补或条件平均插补
- 缺失率低于10%的特征简单统计插补

**使用pandas实现**


```python
print(X.pipe(drop_missing_data, threshold=0.2) 
     .pipe(flag_missing)
     .pipe(impute_manually)
     .pipe(fillna_by_groups, threshold=(0.2, 0.8)) 
     .pipe(impute_simply, threshold=0.2) 
     .isna().sum().gt(0).sum())
```

    Removed 0 variables with missing more than 80.0%
    Added 6 missing indicators
    Conditional Mean Completer:
    Filled 726563 missing values (8.2%).
    Transformed 49 variables with missing (threshold = [20.0%, 80.0%]).
    And then, there are 61 variables with missing.
    Simple imputer:
    Transformed 61 variables with missing (threshold=20.0%).
    And then, there are 0 variables with missing.
    0


**使用sklearn实现**

先自定义几个转换器


```python
class DropMissingData(BaseEstimator, TransformerMixin):
    """
    Remove features from data.
    
    Parameters
    ----------
    threshold: float, default=None
        Require that percentage of non-NA values in a column to keep it.
    """
    def __init__(self, threshold=0.8):
        if 0 < threshold <= 1:
            self.threshold = threshold
        else:
            raise ValueError("threshold must be a value between 0 < x <= 1. ")
    
    def fit(self, X, y=None):
        """
        Find the rows for which missing data should be evaluated to decide if a
        variable should be dropped.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training data set.

        y: pandas Series, default=None
            y is not needed. You can pass None or y.
        """
        
        # check input dataframe
        # X, y = check_X_y(X, y)
        
        # Get the names and number of features in the train set (the dataframe used during fit).
        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]
        
        # Find the features to drop
        self.variables = X.columns[X.isna().mean().gt(1-self.threshold)].tolist()
        return self
    
    def transform(self, X, y=None):     
        """
        Remove variables with missing more than threshold.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------
        X_new: pandas dataframe
            The complete case dataframe for the selected variables.
        """
        # Remove variables with missing more than threshold.
        print(f"Removed {len(self.variables)} variables with missing more than {1-self.threshold:.1%}")
        return X.drop(self.variables, axis=1)
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation. In other words, returns the
        variable names of transformed dataframe.

        Parameters
        ----------
        input_features : array or list, default=None
            This parameter exits only for compatibility with the Scikit-learn pipeline.

            - If `None`, then `feature_names_in_` is used as feature names in.
            - If an array or list, then `input_features` must match `feature_names_in_`.

        Returns
        -------
        feature_names_out: list
            Transformed feature names.
        """
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
        
        # Remove features.
        feature_names_out = [var for var in feature_names_in if var not in self.variables]
        return feature_names_out
```


```python
DropMissingData(threshold=0.4).fit(X).variables
```


    ['OWN_CAR_AGE',
     'YEARS_BUILD_AVG',
     'COMMONAREA_AVG',
     'FLOORSMIN_AVG',
     'LIVINGAPARTMENTS_AVG',
     'NONLIVINGAPARTMENTS_AVG',
     'YEARS_BUILD_MODE',
     'COMMONAREA_MODE',
     'FLOORSMIN_MODE',
     'LIVINGAPARTMENTS_MODE',
     'NONLIVINGAPARTMENTS_MODE',
     'YEARS_BUILD_MEDI',
     'COMMONAREA_MEDI',
     'FLOORSMIN_MEDI',
     'LIVINGAPARTMENTS_MEDI',
     'NONLIVINGAPARTMENTS_MEDI',
     'FONDKAPREMONT_MODE']

整合到pipeline中


```python
# Transformers for missing value imputation

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import MissingIndicator

constant_imputer = FunctionTransformer(
    func=impute_manually,
    accept_sparse=True,
    feature_names_out="one-to-one")

conditional_statistic_imputer = FunctionTransformer(
    func=lambda X:fillna_by_groups(X, threshold=(0.2, 0.8)), 
    accept_sparse=True,
    feature_names_out="one-to-one")

simple_imputer = make_column_transformer(
    (SimpleImputer(strategy="median"), make_column_selector(dtype_include=np.number)),
    (SimpleImputer(strategy="most_frequent"), make_column_selector(dtype_include=object)),
    remainder="passthrough",
    verbose_feature_names_out=False,
    verbose=True)

missing_imputation =  make_pipeline(
    DropMissingData(threshold=0.2),
    constant_imputer,
    conditional_statistic_imputer,
    simple_imputer,
    verbose=True
)

# find variables for which indicator should be added.

add_missing_indicator = make_pipeline(
    MissingIndicator(features='all', sparse=False),
    SelectKBest(k=10),
    verbose=True
)

features = X.columns.tolist()
handle_missing = make_column_transformer(
    (missing_imputation, features),
    (add_missing_indicator, features),
    remainder="passthrough",
    verbose_feature_names_out=False,
    verbose=True
)
```

由于FeatureUnion在调用feature_names_out方法时会在特征名上加前缀，因此我们使用ColumnTransformer，也可以达到同样的效果


```python
X_imputed = handle_missing.fit_transform(X, y)
X_imputed = pd.DataFrame(
    X_imputed,
    columns=handle_missing.get_feature_names_out()
)

X_imputed = X_imputed.astype('category')
X_imputed = X_imputed.apply(pd.to_numeric, errors='ignore')
```

    Removed 0 variables with missing more than 80.0%
    [Pipeline] ... (step 1 of 4) Processing dropmissingdata, total=   0.2s
    [Pipeline]  (step 2 of 4) Processing functiontransformer-1, total=   0.6s
    Conditional Mean Completer:
    Filled 726563 missing values (8.2%).
    Transformed 49 variables with missing (threshold = [20.0%, 80.0%]).
    And then, there are 55 variables with missing.
    [Pipeline]  (step 3 of 4) Processing functiontransformer-2, total=  10.8s
    [ColumnTransformer]  (1 of 2) Processing simpleimputer-1, total=   2.3s
    [ColumnTransformer]  (2 of 2) Processing simpleimputer-2, total=   0.5s
    [Pipeline] . (step 4 of 4) Processing columntransformer, total=   3.2s
    [ColumnTransformer] .... (1 of 2) Processing pipeline-1, total=  14.9s
    [Pipeline] .. (step 1 of 2) Processing missingindicator, total=   1.9s
    [Pipeline] ....... (step 2 of 2) Processing selectkbest, total=   0.1s
    [ColumnTransformer] .... (2 of 2) Processing pipeline-2, total=   2.0s


确认缺失值是否已全部处理完毕：


```python
X_imputed.isna().sum().sum()
```


    0

## 特征重编码

有很多机器学习算法只能接受数值型特征的输入，不能处理离散值特征，比如线性回归，逻辑回归等线性模型，那么我们需要将离散特征重编码成数值变量。

| 方法         | 函数             | python包              |
| ------------ | ---------------- | --------------------- |
| 顺序编码     | OrdinalEncoder   | sklearn.preprocessing |
| 顺序编码     | CategoricalDtype | pandas.api.types      |
| 哑变量编码 | OneHotEncoder    | sklearn.preprocessing |
| 哑变量编码   | pd.get_dummies   | pandas                |
| 平均数编码   | MeanEncoder       |  feature_engine.encoding import  |

不同类型的离散特征有不同的编码方式。


```python
X.dtypes.value_counts()
```


    float64    67
    int64      40
    object     13
    Name: count, dtype: int64


```python
features = X.columns.tolist() 
numeric_cols = X.select_dtypes("number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
```


```python
len(features) == len(numeric_cols + categorical_cols)
```


    True


```python
int_cols = X.select_dtypes("int64").columns.tolist()
float_cols = X.select_dtypes("float64").columns.tolist()
```

### 顺序编码

多数情况下，整型变量都存储了有限的离散值。

**有序分类特征**实际上表征着潜在的排序关系，我们将这些特征的类别映射成有大小的数字，因此可以用顺序编码。   

让我们从分类特征中手动提取有序级别：


```python
# The ordinal (ordered) categorical features
# Pandas calls the categories "levels"

ordered_levels = {
    "NAME_EDUCATION_TYPE": ["Lower secondary", 
                            "Secondary / secondary special", 
                            "Incomplete higher", 
                            "Higher education"]
}
```


```python
from pandas.api.types import CategoricalDtype

def ordinal_encode(X, levels: dict = None):
    X = X.copy()
    if levels is None:
        variables = X.select_dtypes(exclude="number").columns
        X[variables] = X[variables].astype("category")
    else:
        variables = list(levels)
        dtypes = {name: CategoricalDtype(levels[name], ordered=True) for name in levels}
        X[variables] = X[variables].astype(dtypes)
    
    # The `cat.codes` attribute holds the category levels.
    # For missing values, -1 is the default code.
    X[variables] = X[variables].transform(lambda x: x.cat.codes)
    print(f'{len(variables):d} columns were ordinal encoded')
    return X
```


```python
ordinal_encode(X, ordered_levels)[list(ordered_levels)].value_counts()
```

    1 columns were ordinal encoded
    
    NAME_EDUCATION_TYPE
     1                     218391
     3                      74863
     2                      10277
     0                       3816
    -1                        164
    Name: count, dtype: int64

### 哑变量编码

**无序分类特征**对于树集成模型（tree-ensemble like XGBoost）是可用的，但对于线性模型（like Lasso or Ridge）则必须使用one-hot重编码。

现在我们来看看每个分类特征的类别数：


```python
# The nominative (unordered) categorical features
nominal_categories = [col for col in categorical_cols if col not in ordered_levels.keys()]

X[nominal_categories].nunique()
```


    NAME_CONTRACT_TYPE             2
    CODE_GENDER                    2
    NAME_TYPE_SUITE                7
    NAME_INCOME_TYPE               8
    NAME_FAMILY_STATUS             6
    NAME_HOUSING_TYPE              6
    OCCUPATION_TYPE               18
    WEEKDAY_APPR_PROCESS_START     7
    ORGANIZATION_TYPE             57
    FONDKAPREMONT_MODE             4
    HOUSETYPE_MODE                 3
    WALLSMATERIAL_MODE             7
    dtype: int64


```python
# Using pandas to encode categorical features
from pandas.api.types import CategoricalDtype

def onehot_encode(X, variables=None, dummy_na=True):
    """
    Replace the categorical variables by the binary variables.
    
    Parameters
    ----------
    X: pd.DataFrame of shape = [n_samples, n_features]
        The data to encode.
        Can be the entire dataframe, not just seleted variables.
    variables: list, default=None
        The list of categorical variables that will be encoded. 
        If None, the encoder will find and encode all variables of type object or categorical by default.
    dummy_na: boolean, default=True

    Returns
    -------
    X_new: pd.DataFrame.
        The encoded dataframe. The shape of the dataframe will be different from
        the original as it includes the dummy variables in place of the of the
        original categorical ones.
    """
    
    # pd.get_dummies automatically convert the categorical column into dummy variables
    if variables is None:
        variables = X.select_dtypes(exclude='number').columns.tolist()
        X = pd.get_dummies(X, dummy_na=True)
    else:
        X_dummy = pd.get_dummies(X[variables].astype(str), dummy_na=True)
        X = pd.concat([X, X_dummy], axis=1)
        # drop the original non-encoded variables.
        X = X.drop(variables, axis=1)
    print(f'{len(variables):d} columns were one-hot encoded')
    print(f'Dataset shape: {X.shape}')
    return X
```


```python
print(X.pipe(onehot_encode, variables=nominal_categories)
       .pipe(ordinal_encode, levels=ordered_levels)
       .dtypes.value_counts())
```

    12 columns were one-hot encoded
    Dataset shape: (307511, 254)
    1 columns were ordinal encoded
    bool       146
    float64     67
    int64       40
    int8         1
    Name: count, dtype: int64


### 平均数编码

一般情况下，针对分类特征，我们只需要使用sklearn的OneHotEncoder或OrdinalEncoder进行编码，这类简单的预处理能够满足大多数数据挖掘算法的需求。如果某一个分类特征的可能值非常多（高基数 high cardinality），那么再使用one-hot编码往往会出现维度爆炸。平均数编码（mean encoding）是一种高效的编码方式，在实际应用中，能极大提升模型的性能。

我们可以使用 feature-engine开源包实现平均数编码。[feature-engine](https://feature-engine.trainindata.com/en/latest/)将特征工程中常用的方法进行了封装。

其中变量 OCCUPATION_TYPE （职业类型）和 ORGANIZATION_TYPE类别数较多，准备使用平均数编码


```python
# replace categories by the mean value of the target for each category.

from feature_engine.encoding import MeanEncoder
mean_encoder = MeanEncoder(
    missing_values='ignore', 
    ignore_format=True, 
    unseen='ignore'
)
mean_encoder.fit_transform(X[['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']], y).head()
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
      <th>OCCUPATION_TYPE</th>
      <th>ORGANIZATION_TYPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.105788</td>
      <td>0.092996</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.063040</td>
      <td>0.059148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.105788</td>
      <td>0.069781</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.105788</td>
      <td>0.092996</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.063040</td>
      <td>0.058824</td>
    </tr>
  </tbody>
</table>
</div>



### 连续特征分箱

Binning Continuous Features

在实际的模型训练过程中，我们也经常对连续特征进行离散化处理，这样能消除特征量纲的影响，同时还能极大减少异常值的影响，增加特征的稳定性。


| 方法     | 函数             | python包              |
| -------- | ---------------- | --------------------- |
| 二值化   | Binarizer        | sklearn.preprocessing |
| 分箱     | KBinsDiscretizer | sklearn.preprocessing |
| 等频分箱 | pd.qcut          | pandas                |
| 等宽分箱 | pd.cut           | pandas                |

分箱主要分为等频分箱、等宽分箱和聚类分箱三种。等频分箱会一定程度受到异常值的影响，而等宽分箱又容易完全忽略异常值信息，从而一定程度上导致信息损失，若要更好的兼顾变量的原始分布，则可以考虑聚类分箱。所谓聚类分箱，指的是先对某连续变量进行聚类（往往是 k-Means 聚类），然后使用样本所属类别。

以年龄对还款的影响为例


```python
# Find the correlation of the positive days since birth and target
X['DAYS_BIRTH'].abs().corr(y)
```


    -0.07823930830982709

可见，客户年龄与目标意义呈负相关关系，即随着客户年龄的增长，他们往往会更经常地按时偿还贷款。我们接下来将制作一个核心密度估计图（KDE），直观地观察年龄对目标的影响。 


```python
plt.figure(figsize = (5, 3))
sns.kdeplot(x=X['DAYS_BIRTH'] / -365, hue=y, common_norm=False)
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Distribution of Ages')
```


    Text(0.5, 1.0, 'Distribution of Ages')




![](/img/feature_engineering_with_python/preproccessing_output_87_1.png)
    


如果我们把年龄分箱：


```python
# Bin the age data
age_binned = pd.cut(X['DAYS_BIRTH']/-365, bins = np.linspace(20, 70, num = 11))
age_groups  = y.groupby(age_binned).mean()

plt.figure(figsize = (8, 3))
# Graph the age bins and the average of the target as a bar plot
sns.barplot(x=age_groups.index, y=age_groups*100)
# Plot labeling
plt.xticks(rotation = 30)
plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
```


![](/img/feature_engineering_with_python/preproccessing_output_89_0.png)
    


有一个明显的趋势：年轻的申请人更有可能不偿还贷款！ 年龄最小的三个年龄组的失败率在10％以上，最老的年龄组为5％。


```python
# Using pandas
def discretize(X, variables=None, bins=10, strategy="uniform", encoding=None):
    """
    Parameters
    ----------
    bucket_labels: dict, default=None
    """
    X = X.copy()
    
    if strategy not in ["uniform", "quantile"]:
        raise ValueError("strategy takes only values 'uniform' or 'quantile'")
    if encoding not in [None, "onehot", "ordinal"]:
        raise ValueError("encoding takes only values None, 'onehot' or 'ordinal'")
    
    if variables is None:
        variables = X.select_dtypes("number").columns.tolist()
    
    if strategy == "uniform":
        X_binned = X[variables].apply(pd.cut, bins=bins, duplicates='drop')
    elif strategy == "quantile":
        X_binned = X[variables].apply(pd.qcut, q=bins, duplicates='drop')
    
    if encoding == "onehot":
        X_binned = pd.get_dummies(X_binned, dummy_na=True)
    elif encoding == "ordinal":
        X_binned = X_binned.apply(lambda x: x.cat.codes)

    X = pd.concat([X.drop(variables, axis=1), X_binned], axis=1)
    return X
```


```python
discretize(X)[['DAYS_BIRTH', 'DAYS_EMPLOYED']].head()
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
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(-11037.0, -9263.0]</td>
      <td>(-1791.2, 0.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(-18133.0, -16359.0]</td>
      <td>(-1791.2, 0.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(-19907.0, -18133.0]</td>
      <td>(-1791.2, 0.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(-19907.0, -18133.0]</td>
      <td>(-3582.4, -1791.2]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(-21681.0, -19907.0]</td>
      <td>(-3582.4, -1791.2]</td>
    </tr>
  </tbody>
</table>
</div>

sklearn.preprocessing 模块中的 KBinsDiscretizer 可以实现等频分箱、等宽分箱或聚类分箱，同时还可以对分箱后的离散特征进一步进行one-hot编码或顺序编码。


```python
from sklearn.preprocessing import KBinsDiscretizer

equal_frequency_discretiser = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
equal_width_discretiser = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
kmeans_cluster_discretiser = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
```


```python
equal_width_discretiser.fit(X[['DAYS_BIRTH', 'DAYS_EMPLOYED']].fillna(0))
for i, col in enumerate(equal_width_discretiser.get_feature_names_out()):
    print(f"{col}'s bin_edges: ")
    print(equal_width_discretiser.bin_edges_[i])
```

    DAYS_BIRTH's bin_edges: 
    [-25197.  -22162.  -20463.  -18866.  -17208.  -15740.  -14411.  -13129.7
     -11679.  -10275.   -7489. ]
    DAYS_EMPLOYED's bin_edges: 
    [-17912.  -4880.  -3239.  -2367.  -1699.  -1221.   -828.   -463.   -147.
          0.]


### Pipeline实现


```python
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# The ordinal (ordered) categorical features
# Pandas calls the categories "levels"
ordered_levels = {
    "NAME_EDUCATION_TYPE": ["Lower secondary", 
                            "Secondary / secondary special", 
                            "Incomplete higher", 
                            "Higher education"]
}

ordinal_encoder = OrdinalEncoder(
    categories=[np.array(levels) for levels in ordered_levels.values()],
    handle_unknown='use_encoded_value', 
    unknown_value=-1,
    encoded_missing_value=-1)

# replace categories by the mean value of the target for each category.
mean_encoder = MeanEncoder(
    missing_values='ignore', 
    ignore_format=True, 
    unseen='ignore')

# The nominative (unordered) categorical features
nominal_categories = [col for col in categorical_cols if col not in ordered_levels]
features_onehot = [col for col in nominal_categories if col not in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']]

onehot_encoder = OneHotEncoder(
    drop='if_binary', 
    min_frequency=0.02, 
    max_categories=20, 
    sparse_output=False,
    handle_unknown='ignore')

# Encode categorical features
categorical_encoding = make_column_transformer(
    (mean_encoder, ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']),
    (ordinal_encoder, list(ordered_levels)), 
    (onehot_encoder, features_onehot),
    remainder='passthrough', 
    verbose_feature_names_out=False,
    verbose=True
)
```


```python
X_encoded = categorical_encoding.fit_transform(X_imputed, y)
X_encoded = pd.DataFrame(
    X_encoded, 
    columns=categorical_encoding.get_feature_names_out()
)

# X_encoded = X_encoded.astype('category')
X_encoded = X_encoded.apply(pd.to_numeric, errors='ignore')
```

    [ColumnTransformer] ... (1 of 4) Processing meanencoder, total=   0.0s
    [ColumnTransformer]  (2 of 4) Processing ordinalencoder, total=   0.1s
    [ColumnTransformer] . (3 of 4) Processing onehotencoder, total=   1.7s
    [ColumnTransformer] ..... (4 of 4) Processing remainder, total=   0.0s

```python
X_encoded.dtypes.value_counts()
```


    float64    146
    bool        10
    Name: count, dtype: int64

## 异常值检测

我们在实际项目中拿到的数据往往有不少异常数据，这些异常数据很可能让我们模型有很大的偏差。异常检测的方法有很多，例如3倍标准差、箱线法的单变量标记，或者聚类、iForest和LocalOutlierFactor等无监督学习方法。

| 方法              | python模块                           |
| ----------------- | ------------------------------------ |
| 箱线图检测        | feature_engine.outliers.Winsorizer     |
| 3倍标准差原则     | feature_engine.outliers.Winsorizer  |
| 聚类检测          | self-define                          |
| One Class SVM     | sklearn.svm.OneClassSVM              |
| Elliptic Envelope | sklearn.linear_model.SGDOneClassSVM  |
| Elliptic Envelope | sklearn.covariance.EllipticEnvelope  |
| Isolation Forest  | sklearn.ensemble.IsolationForest     |
| LOF               | sklearn.neighbors.LocalOutlierFactor |

### 箱线图检测

**箱线图检测**根据四分位点判断是否异常。四分位数具有鲁棒性，不受异常值的干扰。通常认为小于 $Q_1-1.5*IQR$ 或大于 $Q_3+1.5*IQR$ 的点为离群点。 


```python
X_outlier = X[['DAYS_EMPLOYED', 'AMT_CREDIT']]

fig = plt.figure(figsize=(8,3))
for i, col in enumerate(X_outlier.columns.tolist()):
    ax = fig.add_subplot(1, 2, i+1)
    sns.boxplot(data=X_outlier, y=col, ax=ax)
```


![](/img/feature_engineering_with_python/preproccessing_output_102_0.png)
    


### 3倍标准差原则

**3倍标准差原则**：假设数据满足正态分布，通常定义偏离均值的 $3\sigma$ 之外内的点为离群点，$\mathbb P(|X-\mu|<3\sigma)=99.73\%$。如果数据不服从正态分布，也可以用远离平均值的多少倍标准差来描述。

使用 pandas 实现，并封装在 transformer 中


```python
class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps maximum and/or minimum values of a variable at automatically
    determined values.
    Works only with numerical variables. A list of variables can be indicated. 
    
    Parameters
    ----------
    method: str, 'gaussian' or 'iqr', default='iqr'
        If method='gaussian': 
            - upper limit: mean + 3 * std
            - lower limit: mean - 3 * std
        If method='iqr': 
            - upper limit: 75th quantile + 3 * IQR
            - lower limit: 25th quantile - 3 * IQR
            where IQR is the inter-quartile range: 75th quantile - 25th quantile.
    fold: int, default=3   
        You can select how far out to cap the maximum or minimum values.
    """

    def __init__(self, method='iqr', fold=3, variables=None):
        self.method = method
        self.fold = fold
        self.variables = variables

    def fit(self, X, y=None):
        """
        Learn the values that should be used to replace outliers.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas Series, default=None
            y is not needed in this transformer. You can pass y or None.
        """
        
        
        # Get the names and number of features in the train set.
        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]
        
        # find or check for numerical variables
        numeric_vars = X.select_dtypes("number").columns.tolist()
        if self.variables is None:
            self.variables = numeric_vars
        else:
            self.variables = list(set(numeric_vars) & set(self.variables))

        if self.method == "gaussian":
            mean = X[self.variables].mean()
            bias= [mean, mean]
            scale = X[self.variables].std(ddof=0)
        elif self.method == "iqr":
            Q1 = X[self.variables].quantile(q=0.25)
            Q3 = X[self.variables].quantile(q=0.75)
            bias = [Q1, Q3]
            scale = Q3 - Q1         
        
        # estimate the end values
        if (scale == 0).any():
            raise ValueError(
                f"Input columns {scale[scale == 0].index.tolist()!r}"
                f" have low variation for method {self.method!r}."
                f" Try other capping methods or drop these columns."
            )
        else:
            self.upper_limit = bias[1] + self.fold * scale
            self.lower_limit = bias[0] - self.fold * scale   

        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]

        return self # always return self!

    def transform(self, X, y=None):
        """
        Cap the variable values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """
        X = X.copy()
        
        # check if class was fitted
        check_is_fitted(self)
        
        outiers = (X[self.variables].gt(self.upper_limit) | 
                   X[self.variables].lt(self.lower_limit))
        n = outiers.sum().gt(0).sum()
        print(f"Your selected dataframe has {n} out of {outiers.shape[1]} columns that have outliers.")
        
        # replace outliers
        X[self.variables] = X[self.variables].clip(
            axis=1,
            upper=self.upper_limit,
            lower=self.lower_limit
        )
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)

        if input_features is None:
            return self.feature_names_in_
        elif len(input_features) == self.n_features_in_:
            # If the input was an array, we let the user enter the variable names.
            return list(input_features)
        else:
            raise ValueError(
                "The number of input_features does not match the number of "
                "features seen in the dataframe used in fit."
                ) 
```


```python
outlier_capper = OutlierCapper()
X_capped = outlier_capper.fit_transform(X_outlier)

fig = plt.figure(figsize=(8,3))
for i, col in enumerate(X_capped.columns.tolist()):
    ax = fig.add_subplot(1, 2, i+1)
    sns.boxplot(data=X_capped, y=col, ax=ax)
```

    Your selected dataframe has 2 out of 2 columns that have outliers.




![](/img/feature_engineering_with_python/preproccessing_output_106_1.png)
    

```python
outlier_capper = OutlierCapper(method="gaussian")
outlier_capper.fit_transform(X.select_dtypes('number')).shape
```

    Your selected dataframe has 92 out of 107 columns that have outliers.
    
    (307511, 107)

### sklearn异常检测算法

sklearn 包目前支持的异常检测算法：

- **One Class SVM**：基于 SVM (使用高斯内核) 的思想在特征空间中训练一个超球面，边界外的点即为异常值。
- **Elliptic Envelope**：假设数据满足正态分布，训练一个椭圆包络线，边界外的点则为离群点 。
- **Isolation Forest**：是一种高效的异常检测算法，它和随机森林类似，但每次分裂特征和划分点（值）时都是随机的，而不是根据信息增益或基尼指数来选择。
- **LOF**：基于密度的异常检测算法。离群点的局部密度显著低于大部分近邻点，适用于非均匀的数据集。
- **聚类检测**：常用KMeans聚类将训练样本分成若干个簇，如果某一个簇里的样本数很少，而且簇质心和其他所有的簇都很远，那么这个簇里面的样本极有可能是异常特征样本了。

### Pipeline实现

筛选出来的异常样本需要根据实际含义处理：

- 根据异常点的数量和影响，考虑是否将该条记录删除。
- 对数据做 log-scale 变换后消除异常值。
- 通过数据分箱来平滑异常值。
- 使用均值/中位数/众数来修正替代异常点，简单高效。
- 标记异常值或新增异常值得分列。
- 树模型对离群点的鲁棒性较高，可以选择忽略异常值。

我们接下来考虑对数值型变量计算iForest得分并标记异常样本。


```python
from sklearn.ensemble import IsolationForest

class CustomIsolationForest(IsolationForest, TransformerMixin):
    """
    Isolation Forest Algorithm.
    Compute the anomaly score of each sample using the IsolationForest algorithm.
    """
    def __init__(self, drop_outliers=False, **kwargs):
        super().__init__(**kwargs)
        self.drop_outliers = drop_outliers
    def transform(self, X, y=None):  
        anomaly_scores = super().decision_function(X)
        pred = super().predict(X)
        n_outiers = pred[pred == -1].size
        if self.drop_outliers:
            print(f"Remove {n_outiers} outliers from the dataset")
            return X.loc[pred == 1,:]
        else:
            # Return average anomaly score of X.
            print(f"The number of outiers: {n_outiers} ({n_outiers/X.size:.1%})")
            return anomaly_scores.reshape(-1, 1)
    def get_feature_names_out(self, input_features=None):
        if self.drop:
            return self.feature_names_in_
        else:
            return ["anomaly_score"]
```


```python
# fit the model for anomaly detection
iforest = CustomIsolationForest()
anomaly_score = pd.DataFrame(
    iforest.fit_transform(X_encoded),
    columns=["anomaly_score"]
)
anomaly_score.head()
```

    The number of outiers: 14477 (0.0%)

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
      <th>anomaly_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.061131</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.055532</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.086801</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.140955</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.111094</td>
    </tr>
  </tbody>
</table>
</div>



## 标准化/归一化

数据标准化和归一化可以提高一些算法的准确度，也能加速梯度下降收敛速度。也有不少模型不需要做标准化和归一化，主要是基于概率分布的模型，比如决策树大家族的CART，随机森林等。

- **z-score标准化**是最常见的特征预处理方式，基本所有的线性模型在拟合的时候都会做标准化。前提是假设特征服从正态分布，标准化后，其转换成均值为0标准差为1的标准正态分布。
- **max-min标准化**也称为离差标准化，预处理后使特征值映射到[0,1]之间。这种方法的问题就是如果测试集或者预测数据里的特征有小于min，或者大于max的数据，会导致max和min发生变化，需要重新计算。所以实际算法中， 除非你对特征的取值区间有需求，否则max-min标准化没有 z-score标准化好用。
- **L1/L2范数标准化**：如果我们只是为了统一量纲，那么通过L2范数整体标准化。

| sklearn.preprocessing | 说明                                  |
| --------------------- | ------------------------------------- |
| StandardScaler()      | z-score标准化                         |
| Normalizer(norm='l2') | 使用`l1`、`l2`或`max`范数归一化       |
| MinMaxScaler()        | min-max归一化                         |
| MaxAbsScaler()        | Max-abs归一化，缩放稀疏数据的推荐方法 |
| RobustScaler()        | 分位数归一化，推荐缩放有离群值的数据  |

pandas实现z-score标准化和分位数归一化在之前检测离群值函数里已有，其他标准化方法不太常用。

由于数据集中依然存在一定的离群点，我们可以用RobustScaler对数据进行标准化处理。


```python
from sklearn.preprocessing import RobustScaler
pd.DataFrame(RobustScaler().fit_transform(X[['DAYS_EMPLOYED', 'AMT_CREDIT']])).describe()
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>252137.000000</td>
      <td>307511.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.305718</td>
      <td>0.158721</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.971080</td>
      <td>0.747221</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-6.754153</td>
      <td>-0.869825</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.634136</td>
      <td>-0.452114</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.365864</td>
      <td>0.547886</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.684385</td>
      <td>6.565430</td>
    </tr>
  </tbody>
</table>
</div>



## 正态变换

### 偏度

在许多回归算法中，尤其是线性模型，常常假设数值型特征服从正态分布。我们先来计算一下各个数值特征的偏度：


```python
# Check the skew of all numerical features
skewness = X.select_dtypes('number').skew().sort_values()
skewness[abs(skewness) > 0.75].head(20)
```


    FLAG_MOBIL                     -554.536744
    FLAG_CONT_MOBILE                -23.081172
    YEARS_BEGINEXPLUATATION_MEDI    -15.573124
    YEARS_BEGINEXPLUATATION_AVG     -15.515264
    YEARS_BEGINEXPLUATATION_MODE    -14.755318
    DAYS_EMPLOYED                    -1.968316
    FLAG_EMP_PHONE                   -1.664886
    YEARS_BUILD_MODE                 -1.002305
    YEARS_BUILD_MEDI                 -0.962784
    YEARS_BUILD_AVG                  -0.962485
    FLAG_DOCUMENT_3                  -0.925725
    FLAG_OWN_REALTY                  -0.840293
    EXT_SOURCE_2                     -0.793576
    FLOORSMIN_AVG                     0.954197
    FLOORSMIN_MEDI                    0.960226
    FLOORSMIN_MODE                    0.963835
    FLAG_PHONE                        0.974083
    CNT_FAM_MEMBERS                   0.987543
    FLOORSMAX_AVG                     1.226454
    AMT_CREDIT                        1.234778
    dtype: float64

可以看到这些特征的偏度较高，因此我们尝试变换，让数据接近正态分布。

### QQ图

以AMT_CREDIT特征为例，我们画出分布图和QQ图（使用之前定义的函数）。

> Quantile-Quantile图是一种常用的统计图形，用来比较两个数据集之间的分布。它是由标准正态分布的分位数为横坐标，样本值为纵坐标的散点图。如果QQ图上的点在一条直线附近，则说明数据近似于正态分布，且该直线的斜率为标准差，截距为均值。


```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, norm

def norm_comparison_plot(series):
    series = pd.Series(series)
    mu, sigma = norm.fit(series)
    kurt, skew = series.kurt(), series.skew()
    print(f"Kurtosis: {kurt:.2f}", f"Skewness: {skew:.2f}", sep='\t')
    
    fig = plt.figure(figsize=(10, 4))
    # Now plot the distribution
    ax1 = fig.add_subplot(121)
    ax1.set_title('Distribution')
    ax1.set_ylabel('Frequency')
    sns.distplot(series, fit=norm, ax=ax1)
    ax1.legend(['dist','kde','norm'],f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )', loc='best')
    # Get also the QQ-plot
    ax2 = fig.add_subplot(122)
    probplot(series, plot=plt)
```


```python
norm_comparison_plot(X['AMT_CREDIT'])
plt.show() 
```

    Kurtosis: 1.93	Skewness: 1.23




![](/img/feature_engineering_with_python/preproccessing_output_118_1.png)
    


### 非线性变换

sklearn.preprocessing模块目前支持的非线性变换：

| 方法                | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| QuantileTransformer | 分位数变换，映射到[0,1]之间的均匀分布，或正态分布            |
| PowerTransformer    | 幂变换，将数据从任何分布映射到尽可能接近高斯分布，以稳定方差并最小化倾斜度 |

此外，最常用的是log变换。对于含有负数的特征，可以先min-max缩放到[0,1]之间后再做变换。

这里我们对AMT_CREDIT特征做Box-Cox变换
> 1964年提出的Box-Cox变换可以使得线性回归模型满足线性性、独立性、方差齐次性和正态性的同时又不丢失信息，


```python
from sklearn.preprocessing import PowerTransformer
# Box Cox Transformation of skewed features (instead of log-transformation)
norm_trans = PowerTransformer("box-cox")
```


```python
amt_credit_transformed = norm_trans.fit_transform(X['AMT_INCOME_TOTAL'].values.reshape(-1, 1))
norm_comparison_plot(amt_credit_transformed[:,0])
plt.show()
```

    Kurtosis: 0.49	Skewness: -0.01




![](/img/feature_engineering_with_python/preproccessing_output_121_1.png)
    


可以看到经过Box-Cox变换后，基本符合正态分布了。

## Baseline

至此，数据预处理已经基本完毕    


```python
X_prepared = pd.concat([X_encoded, anomaly_score], axis=1) 
print(X_prepared.shape)
print(X_prepared.dtypes.value_counts())
```

    (307511, 157)
    float64    147
    bool        10
    Name: count, dtype: int64


规范特征名


```python
X_prepared.columns = X_prepared.columns.str.replace('/','or').str.replace(' ','_').str.replace(',','_or')
```

### 交叉验证

我们可以选择模型开始训练了。我们准备选择LightGBM模型训练结果作为baseline。   

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


```python
eval_results, feature_importances = score_dataset(X_prepared, y)
```

    Starting training...
    Training until validation scores don't improve for 50 rounds
    [50]	cv_agg's valid auc: 0.754568 + 0.00339662
    [100]	cv_agg's valid auc: 0.756576 + 0.00312062
    [150]	cv_agg's valid auc: 0.756414 + 0.00303009
    Early stopping, best iteration is:
    [124]	cv_agg's valid auc: 0.756627 + 0.00292137

```python
prepared_data = pd.concat([df[id_col], X_prepared, y], axis=1)
prepared_data.to_csv('../datasets/Home-Credit-Default-Risk/prepared_data.csv', index=False)
```

### 特征重要性


```python
feature_importances['score'].head(20)
```


    EXT_SOURCE_3                  230.4
    EXT_SOURCE_1                  227.6
    AMT_CREDIT                    213.2
    EXT_SOURCE_2                  205.4
    AMT_ANNUITY                   188.0
    DAYS_BIRTH                    172.8
    AMT_GOODS_PRICE               156.2
    DAYS_ID_PUBLISH               140.0
    DAYS_EMPLOYED                 137.0
    DAYS_LAST_PHONE_CHANGE        121.6
    DAYS_REGISTRATION             103.6
    ORGANIZATION_TYPE              97.8
    AMT_INCOME_TOTAL               90.0
    REGION_POPULATION_RELATIVE     77.4
    anomaly_score                  74.2
    OWN_CAR_AGE                    63.2
    OCCUPATION_TYPE                58.8
    HOUR_APPR_PROCESS_START        48.0
    CODE_GENDER_M                  46.8
    NAME_EDUCATION_TYPE            42.6
    Name: score, dtype: float64xxxxxxxxxx clf.fit(X_prepared, y, categorical_feature='name:' + ','.join(categorical_cols))pd.Series(    clf.feature_importances_,    index=X_prepared.columns).sort_values(ascending=False).head(10)python
