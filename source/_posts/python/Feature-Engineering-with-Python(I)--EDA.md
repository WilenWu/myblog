---
title: 特征工程(I)--探索性数据分析
tags:
  - Python
categories:
  - Python
  - 'Machine Learning'
cover: /img/FeatureEngine.png
top_img: /img/sklearn-top-img.svg
abbrlink: 29bf27e3
description: 
date: 2024-03-16 23:40:52
---

有这么一句话在业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。由此可见，特征工程在机器学习中占有相当重要的地位。在实际应用当中，可以说特征工程是机器学习成功的关键。

特征工程是数据分析中最耗时间和精力的一部分工作，它不像算法和模型那样是确定的步骤，更多是工程上的经验和权衡。因此没有统一的方法。这里只是对一些常用的方法做一个总结。

特征工程包含了 Data PreProcessing（数据预处理）、Feature Extraction（特征提取）、Feature Selection（特征选择）和 Feature construction（特征构造）等子问题。

Jupyter Notebook 代码连接：[feature_engineering_demo_p1_EDA](/ipynb/feature_engineering_demo_p1_EDA)

# 数据集描述

## 项目描述

本项目使用Kaggle上的 [家庭信用违约风险数据集 (Home Credit Default Risk)](https://www.kaggle.com/competitions/home-credit-default-risk/data) ，是一个标准的机器学习分类问题。其目标是使用历史贷款的信息，以及客户的社会经济和财务信息，预测客户是否会违约。

数据由Home Credit（捷信）提供，Home Credit致力于向无银行账户的人群提供信贷，是中东欧以及亚洲领先的消费金融提供商之一。在中东欧（CEE），俄罗斯，独联体国家（CIS）和亚洲各地服务于消费者。

Home Credit有三类产品，信用卡，POS（消费贷），现金贷，三类产品的英文分别是：Revolving loan (credit card)，Consumer installment loan (Point of sales loan – POS loan)，Installment cash loan。三类产品逻辑是这样的：

1. 首先提供POS，入门级产品，类似于消费贷。只能用于消费，金额有限，风险最小，客户提供的信息也最少。
2. 然后是credit card，在本次竞赛中称为revolving loan。循环授信，主要用于消费。
3. 最后才是cash loan，用户能得到现金，风险最大。

## 数据文件

数据集包括了8个不同的数据文件，大致可以分为三大类：

- application_{train|test}：包含每个客户社会经济信息和Home Credit贷款申请信息的主要文件。每行代表一个贷款申请，由SK_ID_CURR唯一标识。训练集30.75万数据，测试集4.87万数据。其中训练集中`TARGET=1`表示未偿还贷款。通过这两个文件，就能对这个任务做基本的数据分析和建模，也是本篇的主要内容。

用户在征信机构的历史征信记录可以用来作为风险评估的参考，但是征信数据往往不全，因为这些人本身就很少有银行记录。数据集中bureau.csv和 bureau_balance.csv 对应这部分数据。

- bureau：征信机构提供的客户之前在其他金融机构的贷款申请数据。一个用户（SK_ID_CURR）可以有多笔贷款申请数据（SK_ID_BUREAU）。总计171万数据。
- bureau_balance：征信机构统计的之前每笔贷款（SK_ID_BUREAU）的每月（MONTHS_BALANCE）的还款欠款记录。共有2729万条数据。

数据集中previous_application.csv, POS_CASH_balance.csv，credit_card_balance.csv，installments_payment.csv这部分数据是来自Home Credit产品的历史使用信息。信用卡在欧洲和美国很流行，但在以上这些国家并非如此。所以数据集中信用卡数据偏少。POS只能买东西，现金贷可以得到现金。三类产品都有申请和还款记录。

- previous_application：该表是客户在申请这次贷款之前的申请记录。一个用户（SK_ID_CURR）可以有多笔历史数据（SK_ID_PREV）。共计167万条。
- POS_CASH_balance：以前每月pos流水记录。由SK_ID_PREV和 months_balance唯一标识，共计1000万条数据。
- credit_card_balance：每月信用卡账单表。由MONTHS_BALANCE和SK_ID_PREV唯一标识，合计384万条数据。
- installments_payment：分期付款表。由 SK_ID_PREV, name_instalment_version, name_instalment_number 唯一标识，共计1360万条数据。

字段描述见附录。

# 探索性数据分析

Exploratory Data Analysis(EDA)

本篇主要通过application文件，做基本的数据分析和建模，也是本篇的主要内容。

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
sns.set_style('whitegrid')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.options.display.max_colwidth = 100

SEED = 42
```


```python
df = pd.read_csv('../datasets/Home-Credit-Default-Risk/application_train.csv')
# print(df.head())
```


```python
print('Training data shape: ', df.shape)
```

    Training data shape:  (307511, 122)

```python
# `SK_ID_CURR` is the unique id of the row.
id_col = "SK_ID_CURR"
target = "TARGET"
df[id_col].nunique() == df.shape[0]
```


    True

在遇到非常多的数据的时候，我们一般先会按照数据的类型分布下手，看看不同的数据类型各有多少


```python
# Number of each type of column
print(df.dtypes.value_counts())
```

    float64    65
    int64      41
    object     16
    Name: count, dtype: int64

```python
print("\nCategorical features:")
print(df.select_dtypes(["object"]).columns.tolist())
print("\nNumerical features:")
print(df.select_dtypes("number").columns.tolist())
```


    Categorical features:
    ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
    
    Numerical features:
    ['SK_ID_CURR', 'TARGET', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']


接下来看下数据集的统计信息


```python
print(df.describe())
```

              SK_ID_CURR         TARGET   CNT_CHILDREN  AMT_INCOME_TOTAL  \
    count  307511.000000  307511.000000  307511.000000      3.075110e+05   
    mean   278180.518577       0.080729       0.417052      1.687979e+05   
    std    102790.175348       0.272419       0.722121      2.371231e+05   
    min    100002.000000       0.000000       0.000000      2.565000e+04   
    25%    189145.500000       0.000000       0.000000      1.125000e+05   
    50%    278202.000000       0.000000       0.000000      1.471500e+05   
    75%    367142.500000       0.000000       1.000000      2.025000e+05   
    max    456255.000000       1.000000      19.000000      1.170000e+08   
    
             AMT_CREDIT    AMT_ANNUITY  AMT_GOODS_PRICE  \
    count  3.075110e+05  307499.000000     3.072330e+05   
    mean   5.990260e+05   27108.573909     5.383962e+05   
    std    4.024908e+05   14493.737315     3.694465e+05   
    min    4.500000e+04    1615.500000     4.050000e+04   
    25%    2.700000e+05   16524.000000     2.385000e+05   
    50%    5.135310e+05   24903.000000     4.500000e+05   
    75%    8.086500e+05   34596.000000     6.795000e+05   
    max    4.050000e+06  258025.500000     4.050000e+06   
    
           REGION_POPULATION_RELATIVE     DAYS_BIRTH  DAYS_EMPLOYED  \
    count               307511.000000  307511.000000  307511.000000   
    mean                     0.020868  -16036.995067   63815.045904   
    std                      0.013831    4363.988632  141275.766519   
    min                      0.000290  -25229.000000  -17912.000000   
    25%                      0.010006  -19682.000000   -2760.000000   
    50%                      0.018850  -15750.000000   -1213.000000   
    75%                      0.028663  -12413.000000    -289.000000   
    max                      0.072508   -7489.000000  365243.000000   
    
           DAYS_REGISTRATION  DAYS_ID_PUBLISH    OWN_CAR_AGE     FLAG_MOBIL  \
    count      307511.000000    307511.000000  104582.000000  307511.000000   
    mean        -4986.120328     -2994.202373      12.061091       0.999997   
    std          3522.886321      1509.450419      11.944812       0.001803   
    min        -24672.000000     -7197.000000       0.000000       0.000000   
    25%         -7479.500000     -4299.000000       5.000000       1.000000   
    50%         -4504.000000     -3254.000000       9.000000       1.000000   
    75%         -2010.000000     -1720.000000      15.000000       1.000000   
    max             0.000000         0.000000      91.000000       1.000000   
    
           FLAG_EMP_PHONE  FLAG_WORK_PHONE  FLAG_CONT_MOBILE     FLAG_PHONE  \
    count   307511.000000    307511.000000     307511.000000  307511.000000   
    mean         0.819889         0.199368          0.998133       0.281066   
    std          0.384280         0.399526          0.043164       0.449521   
    min          0.000000         0.000000          0.000000       0.000000   
    25%          1.000000         0.000000          1.000000       0.000000   
    50%          1.000000         0.000000          1.000000       0.000000   
    75%          1.000000         0.000000          1.000000       1.000000   
    max          1.000000         1.000000          1.000000       1.000000   
    
              FLAG_EMAIL  CNT_FAM_MEMBERS  REGION_RATING_CLIENT  \
    count  307511.000000    307509.000000         307511.000000   
    mean        0.056720         2.152665              2.052463   
    std         0.231307         0.910682              0.509034   
    min         0.000000         1.000000              1.000000   
    25%         0.000000         2.000000              2.000000   
    50%         0.000000         2.000000              2.000000   
    75%         0.000000         3.000000              2.000000   
    max         1.000000        20.000000              3.000000   
    
           REGION_RATING_CLIENT_W_CITY  HOUR_APPR_PROCESS_START  \
    count                307511.000000            307511.000000   
    mean                      2.031521                12.063419   
    std                       0.502737                 3.265832   
    min                       1.000000                 0.000000   
    25%                       2.000000                10.000000   
    50%                       2.000000                12.000000   
    75%                       2.000000                14.000000   
    max                       3.000000                23.000000   
    
           REG_REGION_NOT_LIVE_REGION  REG_REGION_NOT_WORK_REGION  \
    count               307511.000000               307511.000000   
    mean                     0.015144                    0.050769   
    std                      0.122126                    0.219526   
    min                      0.000000                    0.000000   
    25%                      0.000000                    0.000000   
    50%                      0.000000                    0.000000   
    75%                      0.000000                    0.000000   
    max                      1.000000                    1.000000   
    
           LIVE_REGION_NOT_WORK_REGION  REG_CITY_NOT_LIVE_CITY  \
    count                307511.000000           307511.000000   
    mean                      0.040659                0.078173   
    std                       0.197499                0.268444   
    min                       0.000000                0.000000   
    25%                       0.000000                0.000000   
    50%                       0.000000                0.000000   
    75%                       0.000000                0.000000   
    max                       1.000000                1.000000   
    
           REG_CITY_NOT_WORK_CITY  LIVE_CITY_NOT_WORK_CITY   EXT_SOURCE_1  \
    count           307511.000000            307511.000000  134133.000000   
    mean                 0.230454                 0.179555       0.502130   
    std                  0.421124                 0.383817       0.211062   
    min                  0.000000                 0.000000       0.014568   
    25%                  0.000000                 0.000000       0.334007   
    50%                  0.000000                 0.000000       0.505998   
    75%                  0.000000                 0.000000       0.675053   
    max                  1.000000                 1.000000       0.962693   
    
           EXT_SOURCE_2   EXT_SOURCE_3  APARTMENTS_AVG  BASEMENTAREA_AVG  \
    count  3.068510e+05  246546.000000    151450.00000     127568.000000   
    mean   5.143927e-01       0.510853         0.11744          0.088442   
    std    1.910602e-01       0.194844         0.10824          0.082438   
    min    8.173617e-08       0.000527         0.00000          0.000000   
    25%    3.924574e-01       0.370650         0.05770          0.044200   
    50%    5.659614e-01       0.535276         0.08760          0.076300   
    75%    6.636171e-01       0.669057         0.14850          0.112200   
    max    8.549997e-01       0.896010         1.00000          1.000000   
    
           YEARS_BEGINEXPLUATATION_AVG  YEARS_BUILD_AVG  COMMONAREA_AVG  \
    count                157504.000000    103023.000000    92646.000000   
    mean                      0.977735         0.752471        0.044621   
    std                       0.059223         0.113280        0.076036   
    min                       0.000000         0.000000        0.000000   
    25%                       0.976700         0.687200        0.007800   
    50%                       0.981600         0.755200        0.021100   
    75%                       0.986600         0.823200        0.051500   
    max                       1.000000         1.000000        1.000000   
    
           ELEVATORS_AVG  ENTRANCES_AVG  FLOORSMAX_AVG  FLOORSMIN_AVG  \
    count  143620.000000  152683.000000  154491.000000   98869.000000   
    mean        0.078942       0.149725       0.226282       0.231894   
    std         0.134576       0.100049       0.144641       0.161380   
    min         0.000000       0.000000       0.000000       0.000000   
    25%         0.000000       0.069000       0.166700       0.083300   
    50%         0.000000       0.137900       0.166700       0.208300   
    75%         0.120000       0.206900       0.333300       0.375000   
    max         1.000000       1.000000       1.000000       1.000000   
    
            LANDAREA_AVG  LIVINGAPARTMENTS_AVG  LIVINGAREA_AVG  \
    count  124921.000000          97312.000000   153161.000000   
    mean        0.066333              0.100775        0.107399   
    std         0.081184              0.092576        0.110565   
    min         0.000000              0.000000        0.000000   
    25%         0.018700              0.050400        0.045300   
    50%         0.048100              0.075600        0.074500   
    75%         0.085600              0.121000        0.129900   
    max         1.000000              1.000000        1.000000   
    
           NONLIVINGAPARTMENTS_AVG  NONLIVINGAREA_AVG  APARTMENTS_MODE  \
    count             93997.000000      137829.000000    151450.000000   
    mean                  0.008809           0.028358         0.114231   
    std                   0.047732           0.069523         0.107936   
    min                   0.000000           0.000000         0.000000   
    25%                   0.000000           0.000000         0.052500   
    50%                   0.000000           0.003600         0.084000   
    75%                   0.003900           0.027700         0.143900   
    max                   1.000000           1.000000         1.000000   
    
           BASEMENTAREA_MODE  YEARS_BEGINEXPLUATATION_MODE  YEARS_BUILD_MODE  \
    count      127568.000000                 157504.000000     103023.000000   
    mean            0.087543                      0.977065          0.759637   
    std             0.084307                      0.064575          0.110111   
    min             0.000000                      0.000000          0.000000   
    25%             0.040700                      0.976700          0.699400   
    50%             0.074600                      0.981600          0.764800   
    75%             0.112400                      0.986600          0.823600   
    max             1.000000                      1.000000          1.000000   
    
           COMMONAREA_MODE  ELEVATORS_MODE  ENTRANCES_MODE  FLOORSMAX_MODE  \
    count     92646.000000   143620.000000   152683.000000   154491.000000   
    mean          0.042553        0.074490        0.145193        0.222315   
    std           0.074445        0.132256        0.100977        0.143709   
    min           0.000000        0.000000        0.000000        0.000000   
    25%           0.007200        0.000000        0.069000        0.166700   
    50%           0.019000        0.000000        0.137900        0.166700   
    75%           0.049000        0.120800        0.206900        0.333300   
    max           1.000000        1.000000        1.000000        1.000000   
    
           FLOORSMIN_MODE  LANDAREA_MODE  LIVINGAPARTMENTS_MODE  LIVINGAREA_MODE  \
    count    98869.000000  124921.000000           97312.000000    153161.000000   
    mean         0.228058       0.064958               0.105645         0.105975   
    std          0.161160       0.081750               0.097880         0.111845   
    min          0.000000       0.000000               0.000000         0.000000   
    25%          0.083300       0.016600               0.054200         0.042700   
    50%          0.208300       0.045800               0.077100         0.073100   
    75%          0.375000       0.084100               0.131300         0.125200   
    max          1.000000       1.000000               1.000000         1.000000   
    
           NONLIVINGAPARTMENTS_MODE  NONLIVINGAREA_MODE  APARTMENTS_MEDI  \
    count              93997.000000       137829.000000    151450.000000   
    mean                   0.008076            0.027022         0.117850   
    std                    0.046276            0.070254         0.109076   
    min                    0.000000            0.000000         0.000000   
    25%                    0.000000            0.000000         0.058300   
    50%                    0.000000            0.001100         0.086400   
    75%                    0.003900            0.023100         0.148900   
    max                    1.000000            1.000000         1.000000   
    
           BASEMENTAREA_MEDI  YEARS_BEGINEXPLUATATION_MEDI  YEARS_BUILD_MEDI  \
    count      127568.000000                 157504.000000     103023.000000   
    mean            0.087955                      0.977752          0.755746   
    std             0.082179                      0.059897          0.112066   
    min             0.000000                      0.000000          0.000000   
    25%             0.043700                      0.976700          0.691400   
    50%             0.075800                      0.981600          0.758500   
    75%             0.111600                      0.986600          0.825600   
    max             1.000000                      1.000000          1.000000   
    
           COMMONAREA_MEDI  ELEVATORS_MEDI  ENTRANCES_MEDI  FLOORSMAX_MEDI  \
    count     92646.000000   143620.000000   152683.000000   154491.000000   
    mean          0.044595        0.078078        0.149213        0.225897   
    std           0.076144        0.134467        0.100368        0.145067   
    min           0.000000        0.000000        0.000000        0.000000   
    25%           0.007900        0.000000        0.069000        0.166700   
    50%           0.020800        0.000000        0.137900        0.166700   
    75%           0.051300        0.120000        0.206900        0.333300   
    max           1.000000        1.000000        1.000000        1.000000   
    
           FLOORSMIN_MEDI  LANDAREA_MEDI  LIVINGAPARTMENTS_MEDI  LIVINGAREA_MEDI  \
    count    98869.000000  124921.000000           97312.000000    153161.000000   
    mean         0.231625       0.067169               0.101954         0.108607   
    std          0.161934       0.082167               0.093642         0.112260   
    min          0.000000       0.000000               0.000000         0.000000   
    25%          0.083300       0.018700               0.051300         0.045700   
    50%          0.208300       0.048700               0.076100         0.074900   
    75%          0.375000       0.086800               0.123100         0.130300   
    max          1.000000       1.000000               1.000000         1.000000   
    
           NONLIVINGAPARTMENTS_MEDI  NONLIVINGAREA_MEDI  TOTALAREA_MODE  \
    count              93997.000000       137829.000000   159080.000000   
    mean                   0.008651            0.028236        0.102547   
    std                    0.047415            0.070166        0.107462   
    min                    0.000000            0.000000        0.000000   
    25%                    0.000000            0.000000        0.041200   
    50%                    0.000000            0.003100        0.068800   
    75%                    0.003900            0.026600        0.127600   
    max                    1.000000            1.000000        1.000000   
    
           OBS_30_CNT_SOCIAL_CIRCLE  DEF_30_CNT_SOCIAL_CIRCLE  \
    count             306490.000000             306490.000000   
    mean                   1.422245                  0.143421   
    std                    2.400989                  0.446698   
    min                    0.000000                  0.000000   
    25%                    0.000000                  0.000000   
    50%                    0.000000                  0.000000   
    75%                    2.000000                  0.000000   
    max                  348.000000                 34.000000   
    
           OBS_60_CNT_SOCIAL_CIRCLE  DEF_60_CNT_SOCIAL_CIRCLE  \
    count             306490.000000             306490.000000   
    mean                   1.405292                  0.100049   
    std                    2.379803                  0.362291   
    min                    0.000000                  0.000000   
    25%                    0.000000                  0.000000   
    50%                    0.000000                  0.000000   
    75%                    2.000000                  0.000000   
    max                  344.000000                 24.000000   
    
           DAYS_LAST_PHONE_CHANGE  FLAG_DOCUMENT_2  FLAG_DOCUMENT_3  \
    count           307510.000000    307511.000000    307511.000000   
    mean              -962.858788         0.000042         0.710023   
    std                826.808487         0.006502         0.453752   
    min              -4292.000000         0.000000         0.000000   
    25%              -1570.000000         0.000000         0.000000   
    50%               -757.000000         0.000000         1.000000   
    75%               -274.000000         0.000000         1.000000   
    max                  0.000000         1.000000         1.000000   
    
           FLAG_DOCUMENT_4  FLAG_DOCUMENT_5  FLAG_DOCUMENT_6  FLAG_DOCUMENT_7  \
    count    307511.000000    307511.000000    307511.000000    307511.000000   
    mean          0.000081         0.015115         0.088055         0.000192   
    std           0.009016         0.122010         0.283376         0.013850   
    min           0.000000         0.000000         0.000000         0.000000   
    25%           0.000000         0.000000         0.000000         0.000000   
    50%           0.000000         0.000000         0.000000         0.000000   
    75%           0.000000         0.000000         0.000000         0.000000   
    max           1.000000         1.000000         1.000000         1.000000   
    
           FLAG_DOCUMENT_8  FLAG_DOCUMENT_9  FLAG_DOCUMENT_10  FLAG_DOCUMENT_11  \
    count    307511.000000    307511.000000     307511.000000     307511.000000   
    mean          0.081376         0.003896          0.000023          0.003912   
    std           0.273412         0.062295          0.004771          0.062424   
    min           0.000000         0.000000          0.000000          0.000000   
    25%           0.000000         0.000000          0.000000          0.000000   
    50%           0.000000         0.000000          0.000000          0.000000   
    75%           0.000000         0.000000          0.000000          0.000000   
    max           1.000000         1.000000          1.000000          1.000000   
    
           FLAG_DOCUMENT_12  FLAG_DOCUMENT_13  FLAG_DOCUMENT_14  FLAG_DOCUMENT_15  \
    count     307511.000000     307511.000000     307511.000000      307511.00000   
    mean           0.000007          0.003525          0.002936           0.00121   
    std            0.002550          0.059268          0.054110           0.03476   
    min            0.000000          0.000000          0.000000           0.00000   
    25%            0.000000          0.000000          0.000000           0.00000   
    50%            0.000000          0.000000          0.000000           0.00000   
    75%            0.000000          0.000000          0.000000           0.00000   
    max            1.000000          1.000000          1.000000           1.00000   
    
           FLAG_DOCUMENT_16  FLAG_DOCUMENT_17  FLAG_DOCUMENT_18  FLAG_DOCUMENT_19  \
    count     307511.000000     307511.000000     307511.000000     307511.000000   
    mean           0.009928          0.000267          0.008130          0.000595   
    std            0.099144          0.016327          0.089798          0.024387   
    min            0.000000          0.000000          0.000000          0.000000   
    25%            0.000000          0.000000          0.000000          0.000000   
    50%            0.000000          0.000000          0.000000          0.000000   
    75%            0.000000          0.000000          0.000000          0.000000   
    max            1.000000          1.000000          1.000000          1.000000   
    
           FLAG_DOCUMENT_20  FLAG_DOCUMENT_21  AMT_REQ_CREDIT_BUREAU_HOUR  \
    count     307511.000000     307511.000000               265992.000000   
    mean           0.000507          0.000335                    0.006402   
    std            0.022518          0.018299                    0.083849   
    min            0.000000          0.000000                    0.000000   
    25%            0.000000          0.000000                    0.000000   
    50%            0.000000          0.000000                    0.000000   
    75%            0.000000          0.000000                    0.000000   
    max            1.000000          1.000000                    4.000000   
    
           AMT_REQ_CREDIT_BUREAU_DAY  AMT_REQ_CREDIT_BUREAU_WEEK  \
    count              265992.000000               265992.000000   
    mean                    0.007000                    0.034362   
    std                     0.110757                    0.204685   
    min                     0.000000                    0.000000   
    25%                     0.000000                    0.000000   
    50%                     0.000000                    0.000000   
    75%                     0.000000                    0.000000   
    max                     9.000000                    8.000000   
    
           AMT_REQ_CREDIT_BUREAU_MON  AMT_REQ_CREDIT_BUREAU_QRT  \
    count              265992.000000              265992.000000   
    mean                    0.267395                   0.265474   
    std                     0.916002                   0.794056   
    min                     0.000000                   0.000000   
    25%                     0.000000                   0.000000   
    50%                     0.000000                   0.000000   
    75%                     0.000000                   0.000000   
    max                    27.000000                 261.000000   
    
           AMT_REQ_CREDIT_BUREAU_YEAR  
    count               265992.000000  
    mean                     1.899974  
    std                      1.869295  
    min                      0.000000  
    25%                      0.000000  
    50%                      1.000000  
    75%                      3.000000  
    max                     25.000000  


## 数据相关性

data Correlation

查看两两特征之间的相关程度，对特征的处理有指导意义。


```python
# The correlation matrix
corrmat = df.corr(numeric_only=True)

# Upper triangle of correlations
upper = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype('bool'))

# Absolute value correlation
correlations = upper.unstack().dropna().sort_index()

correlations.abs().sort_values(ascending=False).head(20)
```


    FLAG_EMP_PHONE                DAYS_EMPLOYED                  0.999755
    YEARS_BUILD_MEDI              YEARS_BUILD_AVG                0.998495
    OBS_60_CNT_SOCIAL_CIRCLE      OBS_30_CNT_SOCIAL_CIRCLE       0.998490
    FLOORSMIN_MEDI                FLOORSMIN_AVG                  0.997241
    FLOORSMAX_MEDI                FLOORSMAX_AVG                  0.997034
    ENTRANCES_MEDI                ENTRANCES_AVG                  0.996886
    ELEVATORS_MEDI                ELEVATORS_AVG                  0.996099
    COMMONAREA_MEDI               COMMONAREA_AVG                 0.995978
    LIVINGAREA_MEDI               LIVINGAREA_AVG                 0.995596
    APARTMENTS_MEDI               APARTMENTS_AVG                 0.995081
    BASEMENTAREA_MEDI             BASEMENTAREA_AVG               0.994317
    LIVINGAPARTMENTS_MEDI         LIVINGAPARTMENTS_AVG           0.993825
    YEARS_BEGINEXPLUATATION_MEDI  YEARS_BEGINEXPLUATATION_AVG    0.993825
    LANDAREA_MEDI                 LANDAREA_AVG                   0.991610
    NONLIVINGAPARTMENTS_MEDI      NONLIVINGAPARTMENTS_AVG        0.990768
    NONLIVINGAREA_MEDI            NONLIVINGAREA_AVG              0.990444
    YEARS_BUILD_MEDI              YEARS_BUILD_MODE               0.989463
    YEARS_BUILD_MODE              YEARS_BUILD_AVG                0.989444
    FLOORSMIN_MEDI                FLOORSMIN_MODE                 0.988406
    FLOORSMAX_MEDI                FLOORSMAX_MODE                 0.988237
    dtype: float64


```python
fig, axs = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=0.9, square=True)
axs.set_title('Correlations', size=15)
plt.show()
```


![](/img/feature_engineering_with_python/EDA_output_16_0.png)
    


目标变量相关性

**Continuous Features**


```python
# Find correlations with the target and sort
print('Most Positive Correlations:\n', correlations["TARGET"].tail(15))
print('\nMost Negative Correlations:\n', correlations["TARGET"].head(15))
```

    Most Positive Correlations:
     SK_ID_CURR   -0.002108
    dtype: float64
    
    Most Negative Correlations:
     SK_ID_CURR   -0.002108
    dtype: float64

```python
# Extract the EXT_SOURCE variables and show correlations
cont_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

fig = plt.figure(figsize=(5, 8))
plt.subplots_adjust(right=1.5)

for i, source in enumerate(cont_features):
    
    # Distribution of feature in dataset
    ax = fig.add_subplot(3, 1, i + 1)
    sns.kdeplot(x=source, data=df, 
                hue=target, 
                common_norm=False,
                fill=True, 
                ax=ax)
    
    # Label the plots
    plt.title(f'Distribution of {source} by Target Value')
    plt.xlabel(f'{source}')
    plt.ylabel('Density')
    
plt.tight_layout(h_pad = 2.5)
```


![](/img/feature_engineering_with_python/EDA_output_19_0.png)
    



```python
# Extract the EXT_SOURCE variables and show correlations
cont_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

for source in cont_features:
    # Calculate the correlation coefficient between the variable and the target
    corr = df[target].corr(df[source])
    
    # Calculate medians for repaid vs not repaid
    avg_repaid = df.loc[df[target] == 0, source].median()
    avg_not_repaid = df.loc[df[target] == 1, source].median()
      
    # print out the correlation
    print('\nThe correlation between %s and the TARGET is %0.4f' % (source, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
```


    The correlation between EXT_SOURCE_1 and the TARGET is -0.1553
    Median value for loan that was not repaid = 0.3617
    Median value for loan that was repaid =     0.5175
    
    The correlation between EXT_SOURCE_2 and the TARGET is -0.1605
    Median value for loan that was not repaid = 0.4404
    Median value for loan that was repaid =     0.5739
    
    The correlation between EXT_SOURCE_3 and the TARGET is -0.1789
    Median value for loan that was not repaid = 0.3791
    Median value for loan that was repaid =     0.5460


**Categorical Features**


```python
cat_features = df.select_dtypes(["object"]).columns

fig = plt.figure(figsize=(16, 16))
for i, feature in enumerate(cat_features):    
    ax = fig.add_subplot(4, 4, i+1)
    sns.countplot(x=feature, data=df, 
                hue=df[target].map(lambda x:str(x)), 
                fill=True, 
                ax=ax)
    
    ax.set_xlabel(feature)
    ax.set_ylabel('App Count')    
    ax.legend(loc='upper center')

plt.show()
```


![](/img/feature_engineering_with_python/EDA_output_22_0.png)
    


## 目标变量分布

对于分类问题的目标变量，在sklearn中需要编码为数值型

| sklearn.preprocessing | 预处理           |
| :-------------------- | :--------------- |
| LabelEncoder          | 目标变量序数编码 |
| LabelBinarizer        | 二分类目标数值化 |
| MultiLabelBinarizer   | 多标签目标数值化 |

检查目标变量分布


```python
# `TARGET` is the target variable we are trying to predict (0 or 1):
# 1 = Not Repaid 
# 0 = Repaid
target = 'TARGET'

print(f"percentage of default : {df[target].mean():.2%}")
print(df[target].value_counts())
```

    percentage of default : 8.07%
    TARGET
    0    282686
    1     24825
    Name: count, dtype: int64


现实中，样本（类别）样本不平衡（class-imbalance）是一种常见的现象，一般地，做分类算法训练时，如果样本类别比例（Imbalance Ratio）（多数类vs少数类）严重不平衡时，分类算法将开始做出有利于多数类的预测。一般有以下几种方法：权重法、采样法、数据增强、损失函数、集成方法、评估指标。

| 方法         | 函数                      | python包               |
| :----------- | :------------------------ | :--------------------- |
| SMOTE        | SMOTE                     | imblearn.over_sampling |
| ADASYN       | ADASYN                    | imblearn.over_sampling |
| Bagging算法  | BalancedBaggingClassifier | imblearn.ensemble      |
| Boosting算法 | EasyEnsembleClassifier    | imblearn.ensemble      |
| 损失函数     | Focal Loss                | self-define            |

我们可以用[imbalance-learn](https://imbalanced-learn.org/stable/)这个Python库实现诸如重采样和模型集成等大多数方法。

对于回归任务，假设预测目标为客户的贷款额度`AMT_CREDIT`


```python
df['AMT_CREDIT'].describe()
```


    count    3.075110e+05
    mean     5.990260e+05
    std      4.024908e+05
    min      4.500000e+04
    25%      2.700000e+05
    50%      5.135310e+05
    75%      8.086500e+05
    max      4.050000e+06
    Name: AMT_CREDIT, dtype: float64

我们画出AMT_CREDIT的分布图和QQ图。

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

可以看到 AMT_CREDIT 的分布呈偏态，许多回归算法都有正态分布假设，因此我们尝试对数变换，让数据接近正态分布。


```python
norm_comparison_plot(np.log1p(df['AMT_CREDIT']))
plt.show()
```

    Kurtosis: -0.29	Skewness: -0.34


![](/img/feature_engineering_with_python/EDA_output_31_1.png)
    


可以看到经过对数变换后，基本符合正态分布了。

sklearn.compose 中的 TransformedTargetRegressor 是专门为回归任务设置的目标变换。对于简单的变换，TransformedTargetRegressor在拟合回归模型之前变换目标变量，预测时则通过逆变换映射回原始值。


```python
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer

reg = TransformedTargetRegressor(regressor=LinearRegression(), 
      transformer=FunctionTransformer(np.log1p))
```

# 附录：数据集字段描述


| Table                 | Row                          | 中文描述                                                     |
| :-------------------- | :--------------------------- | :----------------------------------------------------------- |
| application           | SK_ID_CURR                   | 此次申请的ID                                                 |
| application           | TARGET                       | 申请人本次申请的还款风险：1-风险较高；0-风险较低             |
| application           | NAME_CONTRACT_TYPE           | 贷款类型：cash(现金)还是revolving(周转金，一次申请，多次循环提取) |
| application           | CODE_GENDER                  | 申请人性别                                                   |
| application           | FLAG_OWN_CAR                 | 申请人是否有车                                               |
| application           | FLAG_OWN_REALTY              | 申请人是否有房                                               |
| application           | CNT_CHILDREN                 | 申请人子女个数                                               |
| application           | AMT_INCOME_TOTAL             | 申请人收入状况                                               |
| application           | AMT_CREDIT                   | 此次申请的贷款金额                                           |
| application           | AMT_ANNUITY                  | 贷款年金                                                     |
| application           | AMT_GOODS_PRICE              | 如果是消费贷款，改字段表示商品的实际价格                     |
| application           | NAME_TYPE_SUITE              | 申请人此次申请的陪同人员                                     |
| application           | NAME_INCOME_TYPE             | 申请人收入类型                                               |
| application           | NAME_EDUCATION_TYPE          | 申请人受教育程度                                             |
| application           | NAME_FAMILY_STATUS           | 申请人婚姻状况                                               |
| application           | NAME_HOUSING_TYPE            | 申请人居住状况（租房，已购房，和父母一起住等）               |
| application           | REGION_POPULATION_RELATIVE   | 申请人居住地人口密度，已标准化                               |
| application           | DAYS_BIRTH                   | 申请人出生日（距离申请当日的天数，负值）                     |
| application           | DAYS_EMPLOYED                | 申请人当前工作的工作年限（距离申请当日的天数，负值）         |
| application           | DAYS_REGISTRATION            | 申请人最近一次修改注册信息的时间（距离申请当日的天数，负值） |
| application           | DAYS_ID_PUBLISH              | 申请人最近一次修改申请贷款的身份证明文件的时间（距离申请当日的天数，负值） |
| application           | OWN_CAR_AGE                  | 申请人车龄                                                   |
| application           | FLAG_MOBIL                   | 申请人是否提供个人电话（1-yes，0-no）                        |
| application           | FLAG_EMP_PHONE               | 申请人是否提供家庭电话（1-yes，0-no）                        |
| application           | FLAG_WORK_PHONE              | 申请人是否提供工作电话（1-yes，0-no）                        |
| application           | FLAG_CONT_MOBILE             | 申请人个人电话是否能拨通（1-yes，0-no）                      |
| application           | FLAG_PHONE                   | 是否提供家庭电话                                             |
| application           | FLAG_EMAIL                   | 申请人是否提供电子邮箱（1-yes，0-no）                        |
| application           | OCCUPATION_TYPE              | 申请人职务                                                   |
| application           | CNT_FAM_MEMBERS              | 家庭成员数量                                                 |
| application           | REGION_RATING_CLIENT         | 本公司对申请人居住区域的评分等级（1,2,3）                    |
| application           | REGION_RATING_CLIENT_W_CITY  | 在考虑所在城市的情况下，本公司对申请人居住区域的评分等级（1,2,3） |
| application           | WEEKDAY_APPR_PROCESS_START   | 申请人发起申请日是星期几                                     |
| application           | HOUR_APPR_PROCESS_START      | 申请人发起申请的hour                                         |
| application           | REG_REGION_NOT_LIVE_REGION   | 申请人提供的的永久地址和联系地址是否匹配（1-不匹配，2-匹配，区域级别的） |
| application           | REG_REGION_NOT_WORK_REGION   | 申请人提供的的永久地址和工作地址是否匹配（1-不匹配，2-匹配，区域级别的） |
| application           | LIVE_REGION_NOT_WORK_REGION  | 申请人提供的的联系地址和工作地址是否匹配（1-不匹配，2-匹配，区域级别的） |
| application           | REG_CITY_NOT_LIVE_CITY       | 申请人提供的的永久地址和联系地址是否匹配（1-不匹配，2-匹配，城市级别的） |
| application           | REG_CITY_NOT_WORK_CITY       | 申请人提供的的永久地址和工作地址是否匹配（1-不匹配，2-匹配，城市级别的） |
| application           | LIVE_CITY_NOT_WORK_CITY      | 申请人提供的的联系地址和工作地址是否匹配（1-不匹配，2-匹配，城市级别的） |
| application           | ORGANIZATION_TYPE            | 申请人工作所属组织类型                                       |
| application           | EXT_SOURCE_1                 | 外部数据源1的标准化评分                                      |
| application           | EXT_SOURCE_2                 | 外部数据源2的标准化评分                                      |
| application           | EXT_SOURCE_3                 | 外部数据源3的标准化评分                                      |
| application           | APARTMENTS_AVG               | 申请人居住环境各项指标的标准化评分                           |
| application           | BASEMENTAREA_AVG             | 居住信息                                                     |
| application           | YEARS_BEGINEXPLUATATION_AVG  | 居住信息                                                     |
| application           | YEARS_BUILD_AVG              | 居住信息                                                     |
| application           | COMMONAREA_AVG               | 居住信息                                                     |
| application           | ELEVATORS_AVG                | 居住信息                                                     |
| application           | ENTRANCES_AVG                | 居住信息                                                     |
| application           | FLOORSMAX_AVG                | 居住信息                                                     |
| application           | FLOORSMIN_AVG                | 居住信息                                                     |
| application           | LANDAREA_AVG                 | 居住信息                                                     |
| application           | LIVINGAPARTMENTS_AVG         | 居住信息                                                     |
| application           | LIVINGAREA_AVG               | 居住信息                                                     |
| application           | NONLIVINGAPARTMENTS_AVG      | 居住信息                                                     |
| application           | NONLIVINGAREA_AVG            | 居住信息                                                     |
| application           | APARTMENTS_MODE              | 居住信息                                                     |
| application           | BASEMENTAREA_MODE            | 居住信息                                                     |
| application           | YEARS_BEGINEXPLUATATION_MODE | 居住信息                                                     |
| application           | YEARS_BUILD_MODE             | 居住信息                                                     |
| application           | COMMONAREA_MODE              | 居住信息                                                     |
| application           | ELEVATORS_MODE               | 居住信息                                                     |
| application           | ENTRANCES_MODE               | 居住信息                                                     |
| application           | FLOORSMAX_MODE               | 居住信息                                                     |
| application           | FLOORSMIN_MODE               | 居住信息                                                     |
| application           | LANDAREA_MODE                | 居住信息                                                     |
| application           | LIVINGAPARTMENTS_MODE        | 居住信息                                                     |
| application           | LIVINGAREA_MODE              | 居住信息                                                     |
| application           | NONLIVINGAPARTMENTS_MODE     | 居住信息                                                     |
| application           | NONLIVINGAREA_MODE           | 居住信息                                                     |
| application           | APARTMENTS_MEDI              | 居住信息                                                     |
| application           | BASEMENTAREA_MEDI            | 居住信息                                                     |
| application           | YEARS_BEGINEXPLUATATION_MEDI | 居住信息                                                     |
| application           | YEARS_BUILD_MEDI             | 居住信息                                                     |
| application           | COMMONAREA_MEDI              | 居住信息                                                     |
| application           | ELEVATORS_MEDI               | 居住信息                                                     |
| application           | ENTRANCES_MEDI               | 居住信息                                                     |
| application           | FLOORSMAX_MEDI               | 居住信息                                                     |
| application           | FLOORSMIN_MEDI               | 居住信息                                                     |
| application           | LANDAREA_MEDI                | 居住信息                                                     |
| application           | LIVINGAPARTMENTS_MEDI        | 居住信息                                                     |
| application           | LIVINGAREA_MEDI              | 居住信息                                                     |
| application           | NONLIVINGAPARTMENTS_MEDI     | 居住信息                                                     |
| application           | NONLIVINGAREA_MEDI           | 居住信息                                                     |
| application           | FONDKAPREMONT_MODE           | 居住信息                                                     |
| application           | HOUSETYPE_MODE               | 居住信息                                                     |
| application           | TOTALAREA_MODE               | 居住信息                                                     |
| application           | WALLSMATERIAL_MODE           | 居住信息                                                     |
| application           | EMERGENCYSTATE_MODE          | 居住信息                                                     |
| application           | OBS_30_CNT_SOCIAL_CIRCLE     | 客户的社会关系中有30天逾期                                   |
| application           | DEF_30_CNT_SOCIAL_CIRCLE     | 客户的社会关系中有30天逾期                                   |
| application           | OBS_60_CNT_SOCIAL_CIRCLE     | 客户的社会关系中有60天逾期                                   |
| application           | DEF_60_CNT_SOCIAL_CIRCLE     | 客户的社会关系中有60天逾期                                   |
| application           | DAYS_LAST_PHONE_CHANGE       | 申请人最近一次修改手机号码的时间（距离申请当日的天数，负值） |
| application           | FLAG_DOCUMENT_2              | 申请人是否额外提供了2号文件                                  |
| application           | FLAG_DOCUMENT_3              | 申请人是否额外提供了3号文件                                  |
| application           | FLAG_DOCUMENT_4              | 申请人是否额外提供了4号文件                                  |
| application           | FLAG_DOCUMENT_5              | 申请人是否额外提供了5号文件                                  |
| application           | FLAG_DOCUMENT_6              | 申请人是否额外提供了6号文件                                  |
| application           | FLAG_DOCUMENT_7              | 申请人是否额外提供了7号文件                                  |
| application           | FLAG_DOCUMENT_8              | 申请人是否额外提供了8号文件                                  |
| application           | FLAG_DOCUMENT_9              | 申请人是否额外提供了9号文件                                  |
| application           | FLAG_DOCUMENT_10             | 申请人是否额外提供了10号文件                                 |
| application           | FLAG_DOCUMENT_11             | 申请人是否额外提供了11号文件                                 |
| application           | FLAG_DOCUMENT_12             | 申请人是否额外提供了12号文件                                 |
| application           | FLAG_DOCUMENT_13             | 申请人是否额外提供了13号文件                                 |
| application           | FLAG_DOCUMENT_14             | 申请人是否额外提供了14号文件                                 |
| application           | FLAG_DOCUMENT_15             | 申请人是否额外提供了15号文件                                 |
| application           | FLAG_DOCUMENT_16             | 申请人是否额外提供了16号文件                                 |
| application           | FLAG_DOCUMENT_17             | 申请人是否额外提供了17号文件                                 |
| application           | FLAG_DOCUMENT_18             | 申请人是否额外提供了18号文件                                 |
| application           | FLAG_DOCUMENT_19             | 申请人是否额外提供了19号文件                                 |
| application           | FLAG_DOCUMENT_20             | 申请人是否额外提供了20号文件                                 |
| application           | FLAG_DOCUMENT_21             | 申请人是否额外提供了21号文件                                 |
| application           | AMT_REQ_CREDIT_BUREAU_HOUR   | 申请人发起申请前1个小时以内，被查询征信的次数                |
| application           | AMT_REQ_CREDIT_BUREAU_DAY    | 申请人发起申请前一天以内，被查询征信的次数                   |
| application           | AMT_REQ_CREDIT_BUREAU_WEEK   | 申请人发起申请前一周以内，被查询征信的次数                   |
| application           | AMT_REQ_CREDIT_BUREAU_MON    | 申请人发起申请前一月以内，被查询征信的次数                   |
| application           | AMT_REQ_CREDIT_BUREAU_QRT    | 申请人发起申请前一个季度以内，被查询征信的次数               |
| application           | AMT_REQ_CREDIT_BUREAU_YEAR   | 申请人发起申请前一年以内，被查询征信的次数                   |
| bureau                | SK_ID_CURR                   | 此次申请的ID                                                 |
| bureau                | SK_BUREAU_ID                 | SK_BUREAU_ID                                                 |
| bureau                | CREDIT_ACTIVE                | 信用卡状态                                                   |
| bureau                | CREDIT_CURRENCY              | 信用货币类型，currency1-4，共四个特征                        |
| bureau                | DAYS_CREDIT                  | 客户在申请日前多少天申请的征信机构信用                       |
| bureau                | CREDIT_DAY_OVERDUE           | 申请贷款时客户的逾期天数                                     |
| bureau                | DAYS_CREDIT_ENDDATE          | 客户的在征信机构还有多少天的信用时间                         |
| bureau                | DAYS_ENDDATE_FACT            | 客户在征信机构关闭了多久的信用                               |
| bureau                | AMT_CREDIT_MAX_OVERDUE       | 客户到目前为止的最大逾期额度                                 |
| bureau                | CNT_CREDIT_PROLONG           | 在征信机构有几次延期                                         |
| bureau                | AMT_CREDIT_SUM               | 客户当前在信用机构的信用额度                                 |
| bureau                | AMT_CREDIT_SUM_DEBT          | 客户在信用机构的当前债务                                     |
| bureau                | AMT_CREDIT_SUM_LIMIT         | 信用卡的信用限额                                             |
| bureau                | AMT_CREDIT_SUM_OVERDUE       | 客户在信用机构的违约之和                                     |
| bureau                | CREDIT_TYPE                  | 信用机构的信用类型，车贷、房贷、信用卡、经营贷等             |
| bureau                | DAYS_CREDIT_UPDATE           | 信用机构的最近一次信息更新是多少天前                         |
| bureau                | AMT_ANNUITY                  | 每年要还的贷款额度                                           |
| bureau_balance        | SK_BUREAU_ID                 | SK_BUREAU_ID                                                 |
| bureau_balance        | MONTHS_BALANCE               | 距今的月份                                                   |
| bureau_balance        | STATUS                       | C代表closed、X是未知、0是无逾期、1是逾期在1-30天、2是逾期31-60天、3是逾期61-90天、4是逾期91-120天、5是逾期120天以上。 |
| POS_CASH_balance      | SK_ID_PREV                   | SK_ID_PREV                                                   |
| POS_CASH_balance      | SK_ID_CURR                   | SK_ID_CURR                                                   |
| POS_CASH_balance      | MONTHS_BALANCE               | 距今的月份                                                   |
| POS_CASH_balance      | CNT_INSTALMENT               | 贷款期数                                                     |
| POS_CASH_balance      | CNT_INSTALMENT_FUTURE        | 贷款剩余期数                                                 |
| POS_CASH_balance      | NAME_CONTRACT_STATUS         | 合同状态，Active、Signed、Complete                           |
| POS_CASH_balance      | SK_DPD                       | 当月逾期了多少天                                             |
| POS_CASH_balance      | SK_DPD_DEF                   | 忽略金额比较低的贷款，逾期了多少天                           |
| credit_card_balance   | SK_ID_PREV                   | SK_ID_PREV                                                   |
| credit_card_balance   | SK_ID_CURR                   | SK_ID_CURR                                                   |
| credit_card_balance   | MONTHS_BALANCE               | 距今的月份                                                   |
| credit_card_balance   | AMT_BALANCE                  | Balance during the month of previous credit                  |
| credit_card_balance   | AMT_CREDIT_LIMIT_ACTUAL      | Credit card limit during the month of the previous credit    |
| credit_card_balance   | AMT_DRAWINGS_ATM_CURRENT     | Amount drawing at ATM during the month of the previous credit |
| credit_card_balance   | AMT_DRAWINGS_CURRENT         | Amount drawing during the month of the previous credit       |
| credit_card_balance   | AMT_DRAWINGS_OTHER_CURRENT   | Amount of other drawings during the month of the previous credit |
| credit_card_balance   | AMT_DRAWINGS_POS_CURRENT     | Amount drawing or buying goods during the month of the previous credit |
| credit_card_balance   | AMT_INST_MIN_REGULARITY      | Minimal installment for this month of the previous credit    |
| credit_card_balance   | AMT_PAYMENT_CURRENT          | How much did the client pay during the month on the previous credit |
| credit_card_balance   | AMT_PAYMENT_TOTAL_CURRENT    | How much did the client pay during the month in total on the previous credit |
| credit_card_balance   | AMT_RECEIVABLE_PRINCIPAL     | Amount receivable for principal on the previous credit       |
| credit_card_balance   | AMT_RECIVABLE                | Amount receivable on the previous credit                     |
| credit_card_balance   | AMT_TOTAL_RECEIVABLE         | Total amount receivable on the previous credit               |
| credit_card_balance   | CNT_DRAWINGS_ATM_CURRENT     | Number of drawings at ATM during this month on the previous credit |
| credit_card_balance   | CNT_DRAWINGS_CURRENT         | Number of drawings during this month on the previous credit  |
| credit_card_balance   | CNT_DRAWINGS_OTHER_CURRENT   | Number of other drawings during this month on the previous credit |
| credit_card_balance   | CNT_DRAWINGS_POS_CURRENT     | Number of drawings for goods during this month on the previous credit |
| credit_card_balance   | CNT_INSTALMENT_MATURE_CUM    | Number of paid installments on the previous credit           |
| credit_card_balance   | NAME_CONTRACT_STATUS         | 合同状态，Active、Signed、Complete                           |
| credit_card_balance   | SK_DPD                       | 当月逾期了多少天                                             |
| credit_card_balance   | SK_DPD_DEF                   | 忽略金额比较低的贷款，逾期了多少天                           |
| previous_application  | SK_ID_PREV                   | SK_ID_PREV                                                   |
| previous_application  | SK_ID_CURR                   | SK_ID_CURR                                                   |
| previous_application  | NAME_CONTRACT_TYPE           | 合同类型是现金还是循环贷                                     |
| previous_application  | AMT_ANNUITY                  | 每年要还的贷款额度                                           |
| previous_application  | AMT_APPLICATION              | 之前的贷款申请了多少钱数值                                   |
| previous_application  | AMT_CREDIT                   | 客户的贷款信贷额度                                           |
| previous_application  | AMT_DOWN_PAYMENT             | 之前贷款的首付款                                             |
| previous_application  | AMT_GOODS_PRICE              | 对于消费贷来说，这个字段是要买的商品价格，从数据看每个贷款都对应某个商品，难道业务全部是消费贷吗 |
| previous_application  | WEEKDAY_APPR_PROCESS_START   | 周几申请的贷款，一般工作日多                                 |
| previous_application  | HOUR_APPR_PROCESS_START      | 几点申请的贷款                                               |
| previous_application  | FLAG_LAST_APPL_PER_CONTRACT  | 有时一个合同会被错误的提交多次申请，这个字段用来标志是不是一个贷款的最后一次申请 |
| previous_application  | NFLAG_LAST_APPL_IN_DAY       | 是不是当天的最后一次申请                                     |
| previous_application  | NFLAG_MICRO_CASH             | 是不是小微金融贷                                             |
| previous_application  | RATE_DOWN_PAYMENT            | 归一化的贷款首付比例                                         |
| previous_application  | RATE_INTEREST_PRIMARY        | 主要贷款利息的归一化值                                       |
| previous_application  | RATE_INTEREST_PRIVILEGED     | 优惠贷款利息的归一化值                                       |
| previous_application  | NAME_CASH_LOAN_PURPOSE       | 贷款用途                                                     |
| previous_application  | NAME_CONTRACT_STATUS         | 合同状态，Active、Signed、Complete                           |
| previous_application  | DAYS_DECISION                | 相对于当前贷款，上一次申请的决定是什么时候做的               |
| previous_application  | NAME_PAYMENT_TYPE            | 客户选择上一次贷款的付款方式，现金、电子支付、XNA            |
| previous_application  | CODE_REJECT_REASON           | 被拒原因，XAP、HC、LIMIT                                     |
| previous_application  | NAME_TYPE_SUITE              | 办理贷款的时候是谁跟着一起来的：孩子、家人、配偶、自己、…    |
| previous_application  | NAME_CLIENT_TYPE             | 客户是新客还是老客                                           |
| previous_application  | NAME_GOODS_CATEGORY          | 贷款是为了买什么类型的东西                                   |
| previous_application  | NAME_PORTFOLIO               | pos、cash、cards、xna                                        |
| previous_application  | NAME_PRODUCT_TYPE            | x-sell、walk-in、xna                                         |
| previous_application  | CHANNEL_TYPE                 | 获客渠道，country wide、contact center、stone、AP+（cash loan） |
| previous_application  | SELLERPLACE_AREA             | 销售区域的面积                                               |
| previous_application  | NAME_SELLER_INDUSTRY         | 卖家（应该是客户要买商品的卖家）的行业，消费电子、衣服、工业等 |
| previous_application  | CNT_PAYMENT                  | 之前贷款分为多少期还款                                       |
| previous_application  | NAME_YIELD_GROUP             | 贷款利息，small、medium、high                                |
| previous_application  | PRODUCT_COMBINATION          | 产品组合、 PRODUCT_COMBINATION、Cash X-Sell: low、Cash、POS household with interest |
| previous_application  | DAYS_FIRST_DRAWING           | 相对于当前申请日期，上一次贷款的首次发放时间                 |
| previous_application  | DAYS_FIRST_DUE               | 相对于当前申请日期，上一次贷款的首次逾期时间                 |
| previous_application  | DAYS_LAST_DUE_1ST_VERSION    | 第一次逾期                                                   |
| previous_application  | DAYS_LAST_DUE                | 最近一次逾期                                                 |
| previous_application  | DAYS_TERMINATION             | 到期日期                                                     |
| previous_application  | NFLAG_INSURED_ON_APPROVAL    | 之前的申请有没有要求保险                                     |
| installments_payments | SK_ID_PREV                   | SK_ID_PREV                                                   |
| installments_payments | SK_ID_CURR                   | SK_ID_CURR                                                   |
| installments_payments | NUM_INSTALMENT_VERSION       | 分期付款                                                     |
| installments_payments | NUM_INSTALMENT_NUMBER        | 分期付款期数                                                 |
| installments_payments | DAYS_INSTALMENT              | 相对于当前申请日期，之前的贷款应该在什么时间支付             |
| installments_payments | DAYS_ENTRY_PAYMENT           | 相对于当前申请日期，之前贷款的实际支付时间                   |
| installments_payments | AMT_INSTALMENT               | 贷款分期的约定付款金额                                       |
| installments_payments | AMT_PAYMENT                  | 分期的实际付款金额                                           |

参考文献：

[Home Credit Default Risk - 1 之基础篇](https://zhuanlan.zhihu.com/p/104288764)   
[Home Credit Default Risk 之FeatureTools篇](https://zhuanlan.zhihu.com/p/104370111)   
[Feature Engineering for House Prices](https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices)    
[Credit Fraud信用卡欺诈数据集，如何处理非平衡数据](https://blog.csdn.net/s09094031/article/details/90924284)   
[Predict Future Sales 预测未来销量, Kaggle 比赛，LB 0.89896 排名6%](https://blog.csdn.net/s09094031/article/details/90347191)   
[feature-engine](https://feature-engine.trainindata.com/en/latest/) 将特征工程中常用的方法进行了封装   
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
