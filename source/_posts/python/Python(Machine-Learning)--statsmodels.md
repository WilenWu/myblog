---
title: Python(Machine Learning)--statsmodels
categories:
  - Python
  - 'Machine Learning'
tags:
  - Python
  - 机器学习
cover: /img/statsmodels.svg
katex: true
description: 统计建模和计量经济学
abbrlink: 6c94e349
date: 2018-05-10 23:12:43
---

与scikit-learn比较，statsmodels包含经典统计学和经济计量学的算法。包括如下子模块：  

-   回归模型：线性回归，广义线性模型，健壮线性模型，线性混合效应模型等等。  
-   方差分析（ANOVA）。  
-   时间序列分析：AR，ARMA，ARIMA，VAR和其它模型。  
-   非参数方法： 核密度估计，核回归。  
-   统计模型结果可视化。  

[statsmodels](http://www.statsmodels.org/stable/index.html#table-of-contents)更关注统计推断，提供不确定估计和参数p-value。相反的，scikit-learn注重预测。  

<!-- more -->  

# 入门

## 模型拟合和描述

拟合模型`statsmodels`通常包括 3 个简单的步骤：

1. 使用模型类来描述模型
2. 使用类方法拟合模型
3. 使用汇总方法检查结果

对于 OLS，这是通过以下方式实现的：

```python  
>>> import statsmodels.api as sm  
>>> import statsmodels.formula.api as smf  
>>> import pandas  
>>> df = sm.datasets.get_rdataset("Guerry", "HistData").data  
>>> df = df.dropna()  
# step 1 Describe model,return model class  
>>> mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)  
# step 2 Fit model,return result class  
>>> res = mod.fit()  
# step 3 Summarize model  
>>> print(res.summary())  
 OLS Regression Results                              
==============================================================================  
Dep. Variable:                Lottery   R-squared:                       0.338  
Model:                            OLS   Adj. R-squared:                  0.287  
Method:                 Least Squares   F-statistic:                     6.636  
Date:                Sun, 23 Dec 2018   Prob (F-statistic):           1.07e-05  
Time:                        18:41:19   Log-Likelihood:                -375.30  
No. Observations:                  85   AIC:                             764.6  
Df Residuals:                      78   BIC:                             781.7  
Df Model:                           6                                           
Covariance Type:            nonrobust                                           
===============================================================================  
                  coef    std err          t      P>|t|      [0.025      0.975]  
-------------------------------------------------------------------------------  
Intercept      38.6517      9.456      4.087      0.000      19.826      57.478  
Region[T.E]   -15.4278      9.727     -1.586      0.117     -34.793       3.938  
Region[T.N]   -10.0170      9.260     -1.082      0.283     -28.453       8.419  
Region[T.S]    -4.5483      7.279     -0.625      0.534     -19.039       9.943  
Region[T.W]   -10.0913      7.196     -1.402      0.165     -24.418       4.235  
Literacy       -0.1858      0.210     -0.886      0.378      -0.603       0.232  
Wealth          0.4515      0.103      4.390      0.000       0.247       0.656  
==============================================================================  
Omnibus:                        3.049   Durbin-Watson:                   1.785  
Prob(Omnibus):                  0.218   Jarque-Bera (JB):                2.694  
Skew:                          -0.340   Prob(JB):                        0.260  
Kurtosis:                       2.454   Cond. No.                         371.  
==============================================================================  
  
Warnings:  
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.  
>>> res.params  # 获取模型参数  
Intercept      38.651655  
Region[T.E]   -15.427785  
Region[T.N]   -10.016961  
Region[T.S]    -4.548257  
Region[T.W]   -10.091276  
Literacy       -0.185819  
Wealth          0.451475  
dtype: float64  
>>> dir(res)    # 查看完整的属性列表  
```

## 输入输出模型  
```python  
from statsmodels.iolib import smpickle  
smpickle.save_pickle(obj, fname)	#Save the object to file via pickling.  
smpickle.load_pickle(fname)	#Load a previously saved object from file  
```

## 模型测试和绘图  
```python  
>>> #Rainbow测试线性度（零假设是关系被正确建模为线性)  
>>> sm.stats.linear_rainbow(res)  
(0.847233997615691, 0.6997965543621644)  
>>> sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'],  
      data=df, obs_labels=False) #绘制回归图  
```

![line](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/line.png)  

## 使用R型公式来拟合模型

`import statsmodels.formula.api as smf`  

| formula  | 说明  | 示例|  
| -------- | :------------ | :---------- |  
| ~  | 分隔符，左边为响应变量，右边为解释变量 |  |  
| +  | 添加变量 | y~x+y  |  
| -  | 移除变量 | y~x*z*w–x:z:w可展开为 y ~ (x + z + w)**2 |  
| - | 移除变量 | y~x-1(移除截距)  |  
| :  | 预测变量交互项 | y~x+y+x:y |  
| *  | 包含所有交互项的简洁方式| 代码y~ x * z可展开为y ~ x + z + x:z  |  
| **  | 交互项的最高次数  | 代码 y ~ (x + z + w)**2 可展开为 y ~ x + z + w + x:z + x:w + z:w |  
| C()|**处理分类变量**  | |  
| function | 数学函数 | log(y) ~ x + z + w |  

**支持R型公式的模型**  
```python  
In [15] res = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region) -1 ', data=df).fit()  
```

**不支持R型公式的模型，使用[patsy][patsy] 模块**  

[patsy]: https://patsy.readthedocs.io/en/latest/overview.html
```python  
#Using formulas with models that do not (yet) support them  
In [22]: import patsy  
In [23]: f = 'Lottery ~ Literacy * Wealth'  
In [24]: y, X = patsy.dmatrices(f, df, return_type='matrix')  
 #y被转化为patsy.DesignMatrix,x被转化为转化为numpy.ndarray  
In [26]: print(X[:5])  
[[   1.   37.   73. 2701.]  
 [   1.   51.   22. 1122.]  
 [   1.   13.   61.  793.]  
 [   1.   46.   76. 3496.]  
 [   1.   69.   83. 5727.]]  
In [27]: f = 'Lottery ~ Literacy * Wealth'  
  
In [28]: y, X = patsy.dmatrices(f, df, return_type='dataframe') #转化为pandas.dataframe  
In [30]: print(X[:5])  
   Intercept  Literacy  Wealth  Literacy:Wealth  
0        1.0      37.0    73.0           2701.0  
1        1.0      51.0    22.0           1122.0  
2        1.0      13.0    61.0            793.0  
3        1.0      46.0    76.0           3496.0  
4        1.0      69.0    83.0           5727.0  
In [31]: res=smf.OLS(y, X).fit()  
```

## statsmodels参数  

Statsmodels使用endog和exog为模型数据参数名称，作为估计器的观测变量。  

endog|exog  
:------|:------  
y|	x  
y variable	|x variable  
left hand side (LHS)	|right hand side (RHS)  
dependent variable（因变量）|	independent variable（自变量）  
regressand	|regressors  
outcome	|design  
response variable（响应变量）|explanatory variable（解释变量）  

## [模型和拟合结果的超类](https://www.statsmodels.org/stable/dev/internal.html#model-and-results-classes)

Model和Result是statsmodels所有模型和结果的父类  

**model class**  
```python  
Model(endog, exog=None, **kwargs) #建立模型  
```
Methods|desc  
:---|:---  
fit()|Fit a model to data.  
from_formula(formula, data, subset=None, drop_cols=None, *args, **kwargs)|Create a Model from a formula and dataframe.  
predict(params, exog=None, *args, **kwargs)|After a model has been fit predict returns the fitted values.  

Attributes|desc  
:---|:---  
endog_names|Names of endogenous variables  
exog_names|Names of exogenous variables  

**result class**  
```python  
Results(model, params, **kwd) #一般通过模型fit方法拟合生成  
```
Methods|desc  
:---|:---  
initialize(model, params, **kwd)	|  
predict([exog, transform])|Call self.model.predict with self.params as the first argument.  
summary()|  


# 回归分析

## [Linear Regression(线性回归)](https://www.statsmodels.org/stable/regression.html)

`import statsmodels.api as sm`  
适用于自变量X和因变量Y为线性关系，具体来说，画出散点图可以用一条直线来近似拟合。一般线性模型要求观测值之间相互独立、残差(因变量)服从正态分布、残差(因变量)方差齐性  
统计模型被假定为 $Y=Xβ+μ,  μ\sim N(0,Σ)$  

**Model Classes**

[Model Classes][Model Classes]|模型类  
:------|:------  
OLS	|一个简单的普通最小二乘模型。  
GLS|具有一般协方差结构的广义最小二乘模型  
WLS|具有对角但非标识协方差结构的回归模型  
GLSAR|具有AR（p）协方差结构的回归模型。  
yule_walker|使用Yule-Walker方程从序列X估计AR（p）参数  
QuantReg|分位数回归  
RecursiveLS|递归最小二乘法  

[Model Classes]: http://www.statsmodels.org/stable/regression.html#model-classes

Methods|desc  
:---|:---  
fit()|Full fit of the model  
from_formula(_formula_, _data_)|Create a Model from a formula and dataframe  
predict(_params_)|Return linear predicted values from a design matrix.  
score(_params_)|Evaluate the score function at a given point  


Attributes|desc  
:---|:---  
df_model|模型自由度，定义为回归矩阵的秩，如果包括常数则减1  
df_resid|剩余自由度，定义为观察数减去回归矩阵的rank  

**Results Classes**  

| Results Classes   | 结果类   |
:---|:---
RegressionResults|总结了线性回归模型的拟合
OLSResults|OLS模型的结果类
PredictionResults|	
QuantRegResults|QuantReg模型的结果实例

| Methods|desc
:---|:---
aic(), bic(), bse()...|
cov_params()|返回方差/协方差矩阵
eigenvals()|返回按降序排序的特征值
fvalue(), pvalues(), f_pvalue(), tvalues()|
f_test(r_matrix), t_test()|F检验，t检验
get_prediction()|计算预测结果
save(fname), load(fname)|保存pickle，加载（类方法）

**Examples**   
```python  
# Load modules and data  
In [1]: import numpy as np  
In [2]: import statsmodels.api as sm  
In [3]: spector_data = sm.datasets.spector.load()  
In [4]: spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)  
  
# Fit and summarize OLS model  
In [5]: mod = sm.OLS(spector_data.endog, spector_data.exog)  
In [6]: res = mod.fit()  
In [7]: res.summary()  
```

## [Generalized Linear(广义线性回归)](http://www.statsmodels.org/stable/glm.html#module-reference)  


是为了克服线性回归模型的缺点出现的，是线性回归模型的推广。首先自变量可以是离散的，也可以是连续的。离散的可以是0-1变量，也可以是多种取值的变量。广义线性模型又取消了对残差(因变量)服从正态分布的要求。残差不一定要服从正态分布，可以服从二项、泊松、负二项、正态、伽马、逆高斯等分布，这些分布被统称为指数分布族。  
与线性回归模型相比较，有以下推广：  
- 随机误差项不一定服从正态分布，可以服从二项、泊松、负二项、正态、伽马、逆高斯等分布，这些分布被统称为指数分布族。  
- 引入link函数$g(\cdot )$。因变量和自变量通过联接函数产生影响。根据不同的数据，可以自由选择不同的模型。大家比较熟悉的Logit模型就是使用Logit联接、随机误差项服从二项分布得到模型。  
  

The statistical model for each observation  ii  is assumed to be  
$Y_i ∼F_{EDM}(⋅|θ,ϕ,w_i)$ and  $μ_i=E[Y_i|x_i]=g^{-1}(x'_i β)$  

**Model Classes**  
`GLM(endog, exog, family=None)`  
> Parameters:	  
endog (array-like)  
exog (array-like)  
family (family class instance) – The default is Gaussian  

Methods|desc  
:---|:---  
fit()|Full fit of the model  
from_formula(_formula_, _data_)|Create a Model from a formula and dataframe  
predict(_params_)|Return linear predicted values from a design matrix.  
score(_params_)|Evaluate the score function at a given point  

**Results Classes**  

| Results Classes  | 结果类     |
:---|:---
GLMResults|包含GLM结果的类。
PredictionResults|

**Families**  

Families|desc
:---|:---
Family(link,variances)|单参数指数族的父类。
Binomial(link=None)|二项式指数族分布。
Gamma(link=None)|Gamma指数族分布。
Gaussian(link=None)|高斯指数族分布。
InverseGaussian(link=None)|InverseGaussian指数族。
NegativeBinomial(link=None,alpha=None)|负二项指数族。
Poisson(link=None)|泊松指数族。
Tweedie(link=None,var_power=None)|Tweedie。

**Link Functions**  
```python  
>>> sm.families.family.<familyname>.links  
```
Link Functions|desc  
:---|:---  
Link|单参数指数族的通用链接函数。  
CDFLink([DBN])|使用scipy.stats发行版的CDF  
CLogLog|互补的log-log变换  
Log|对数转换  
Logit|logit变换  
NegativeBinomial([α])|负二项式link函数  
Power([power])|指数转换  
cauchy()|Cauchy(标准Cauchy CDF)变换  
cloglog|CLogLog转换link函数。  
identity()|identity转换  
inverse_power()|逆变换  
inverse_squared()|逆平方变换  
nbinom([α])|负二项式link函数。  
probit([DBN])|probit(标准普通CDF)变换  

**Examples**   
```python  
In [1]: import statsmodels.api as sm  
In [2]: data = sm.datasets.scotland.load()  
In [3]: data.exog = sm.add_constant(data.exog)  
# Instantiate a gamma family model with the default link function.  
In [4]: gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())  
In [5]: gamma_results = gamma_model.fit()  
In [6]: print(gamma_results.summary())  
In [7]: gamma_results = gamma_model.fit()  
In [8]: gamma_results.params  
```

## [Generalized Estimating Equations(广义估计方程)](http://www.statsmodels.org/stable/gee.html)  

实际工作中一些资料由于部分观察值含有非独立或相关的信息，不能用传统的一般线性（或广义线性进行分析），故而发展出了广义估计方程。如纵向数据，重复测量数据，整群抽样设计资料，聚集性资料或是多次层次系统结构资料。  
> 纵向数据(longitudinal data)：是按照时间顺序对个体某指标进行重复测量得到的数据，同一对象的多次测量成相关性（倾向），如儿童的生长检测资料  

  

Model Classes|模型类  
:------|:------  
GEE|用广义估计方程（GEE）估计边际回归模型。  
**Results Classes**|结果类  
GEEResults|总结了使用GEE的边际回归模型的适合性。  
GEEMargins|	与GEE匹配的回归模型的估计边际效应。  
**Dependence Structures**|依赖结构  
CovStruct|分组数据的相关性和协方差结构的基类。  
Autoregressive|一阶自回归工作依赖结构。  
Exchangeable|可交换的工作依赖结构。  
GlobalOddsRatio|估算具有序数或名义数据的GEE的全局优势比。  
Independence|独立工作依赖结构。  
Nested|嵌套的工作相关结构。  

```python  
In [1]: import statsmodels.api as sm  
In [2]: import statsmodels.formula.api as smf  
In [3]: data = sm.datasets.get_rdataset('epil', package='MASS').data  
In [4]: fam = sm.families.Poisson()  
In [5]: ind = sm.cov_struct.Exchangeable()  
In [6]: mod = smf.gee("y ~ age + trt + base", "subject", data,  
   ...:               cov_struct=ind, family=fam)  
In [7]: res = mod.fit()  
In [8]: print(res.summary())  
```

## [Robust Linear Models(稳健的线性模型)](http://www.statsmodels.org/stable/rlm.html)  

稳健回归(robust regression)是将稳健估计方法用于回归模型，以拟合大部分数据存在的结构，同时可识别出潜在可能的离群点、强影响点或与模型假设相偏离的结构。当误差服从正态分布时，其估计几乎和最小二乘估计一样好，而最小二乘估计条件不满足时，其结果优于最小二乘估计。  

Model Classes|模型类  
:------|:------  
RLM|稳健的线性模型  
**Results Classes**|结果类  
RLMResults|	包含RLM结果的类  
**Norms**|  
AndrewWave|安德鲁的M估计浪潮。  
Hampe|Hampel函数用于M估计。  
HuberT|胡伯的T为M估计。  
LeastSquares|M估计的最小二乘ρ及其派生函数。  
RamsayE|Ramsay的Ea用于M估计。  
RobustNorm|用于稳健回归的规范的父类。  
TrimmedMean|用于M估计的修正均值函数。  
TukeyBiweight|Tukey的M估计的权重函数。  
estimate_location|使用self.norm和当前的尺度估计量的位置M估计量。  
**Scale**|  
Huber|Huber提出的联合估计地点和规模的建议2。  
HuberScale|Huber用于拟合鲁棒线性模型的缩放比例。  
mad|沿数组给定轴的中值绝对偏差  
hubers_scale|Huber用于拟合鲁棒线性模型的缩放比例。  

```python  
# Load modules and data  
In [1]: import statsmodels.api as sm  
In [2]: data = sm.datasets.stackloss.load()  
In [3]: data.exog = sm.add_constant(data.exog)  
  
# Fit model and print summary  
In [4]: rlm_model = sm.RLM(data.endog, data.exog,M=sm.robust.norms.HuberT())  
In [5]: rlm_results = rlm_model.fit()  
In [6]: print(rlm_results.params)  
[-41.0265   0.8294   0.9261  -0.1278]  
```

## [Linear Mixed Effects Models(线性混合效应模型)](http://www.statsmodels.org/stable/mixed_linear.html)  

参考链接：https://blog.csdn.net/sinat_26917383/article/details/51636011  

在线性模型中加入随机效应项，消了观测值之间相互独立和残差(因变量)方差齐性的要求。  


Model Classes|模型类  
:------|:------  
MixedLM|  
**Results Classes**|结果类  
MixedLMResults|  

```python  
In [1]: import statsmodels.api as sm  
In [2]: import statsmodels.formula.api as smf  
In [3]: data = sm.datasets.get_rdataset("dietox", "geepack").data  
In [4]: md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])  
In [5]: mdf = md.fit()  
In [6]: print(mdf.summary())  
```

## [Regression with Discrete Dependent Variable(离散因变量回归)](http://www.statsmodels.org/stable/discretemod.html)


Model Classes|desc  
:---|:---  
Logit(endog,exog,**kwargs)|二元logit模型  
Probit(endog,exog,**kwargs)|二元Probit模型  
MNLogit(endog,exog,**kwargs)|多项logit模型  
Poisson(endog, exog[, offset, exposure, missing])|计数数据的泊松模型  
NegativeBinomial(endog,exog [,...])|计数数据的负二项模型  
NegativeBinomialP(endog,exog [,p,offset,...])|计数数据的广义负二项(NB-P)模型  
GeneralizedPoisson(endog,exog [,p,offset,...])|计数数据的广义Poisson模型  
ZeroInflatedPoisson(endog,exog [,...])|用于计数数据的泊松零膨胀模型  
ZeroInflatedNegativeBinomialP(endog,exog [,...])|计数数据的零膨胀广义负二项模型  
ZeroInflatedGeneralizedPoisson(endog,exog)|计数数据的零膨胀广义Poisson模型  
**Results Classes**|结果类  
LogitResults（model，mlefit [，cov_type，...]）	|Logit Model的结果类  
ProbitResults（model，mlefit [，cov_type，...]）	|Probit模型的结果类  
CountResults（model，mlefit [，cov_type，...]）	|计数数据的结果类  
MultinomialResults（model，mlefit [，...]）	|多项数据的结果类  
NegativeBinomialResults（model，mlefit [，...]）	|NegativeBinomial 1和2的结果类  
GeneralizedPoissonResults（model，mlefit [，...]）	|广义泊松分析的结果类  
ZeroInflatedPoissonResults（model，mlefit [，...]）	|零膨胀泊松的结果类  
ZeroInflatedNegativeBinomialResults	|零膨胀一般化负二项式的结果类  
ZeroInflatedGeneralizedPoissonResults|	零膨胀广义泊松的结果类  

DiscreteModel是所有离散回归模型的超类。估算结果作为其中一个子类的实例返回 DiscreteResults。模型的每个类别（二进制，计数和多项）都有其自己的中级模型和结果类。这个中间类主要是为了方便实现由DiscreteModel和 定义的方法和属性DiscreteResults。  

DiscreteModel（endog，exog，** kwargs）	|抽象类用于离散选择模型。  
:------|:------  
DiscreteResults（model，mlefit [，cov_type，...]）	|离散因变量模型的结果类。  
BinaryModel（endog，exog，** kwargs）	|  
BinaryResults（model，mlefit [，cov_type，...]）	|二进制数据的结果类  
CountModel（endog，exog [，offset，exposure，...]）	|  
MultinomialModel（endog，exog，** kwargs）	|  
GenericZeroInflated（endog，exog [，...]）	|Generiz Zero计数数据的充气模型  

**Examples**  
```python  
# Load the data from Spector and Mazzeo (1980)  
In [1]: spector_data = sm.datasets.spector.load()  
In [2]: spector_data.exog = sm.add_constant(spector_data.exog)  
# Logit Model  
In [3]: logit_mod = sm.Logit(spector_data.endog, spector_data.exog)  
In [4]: logit_res = logit_mod.fit()  
Optimization terminated successfully.  
 Current function value: 0.402801  
 Iterations 7  
In [5]: print(logit_res.summary())  
```

## [Generalized Linear Mixed Effects Models(广义线性混合效应模型)](http://www.statsmodels.org/stable/mixed_glm.html)  

广义线性混合效应模型是混合效应模型的推广  

Model Classes|模型类  
:------|:------  
BinomialBayesMixedGLM|用贝叶斯方法拟合广义线性混合模型。  
PoissonBayesMixedGLM|用贝叶斯方法拟合广义线性混合模型。  
**Results Classes**|结果类  
BayesMixedGLMResults|  

## [方差分析](http://www.statsmodels.org/stable/anova.html)

方差分析(Analysis of Variance，简称ANOVA)，又称“变异数分析”，为数据分析中常见的统计模型，主要为探讨连续型（Continuous）因变量（Dependent variable）与类别型自变量（Independent variable）的关系。当自变量的因子等于或超过三个类别时，检验各类别平均值是否相等，采用方差分析。  
广义t检验中，方差相等（Equality of variance）的合并t检验（Pooled T-test）视为是方差分析的一种。t检验分析两组平均数是否相等，方差分析也采用相同的计算概念，实际上，当方差分析套用在合并t检验的分析上时，产生的F值则会等于t检验的平方项。  

总偏差平方和 $SSt = SSb + SSw$  

statsmodels包含anova_lm模型，用于使用线性OLSModel进行方差分析，和AnovaRM模型，用于重复测量方差分析（包含平衡数据方差分析）。  

Module Reference|desc  
:---|:---  
`anova_lm(*args, **kwargs)` |Anova table for one or more fitted linear models  
`AnovaRM(data, depvar, subject[, within, …])`|Repeated measures Anova using least squares regression  

```python  
In [1]: import statsmodels.api as sm  
In [2]: from statsmodels.formula.api import ols  
In [3]: moore = sm.datasets.get_rdataset("Moore", "car",  
   ...:                                  cache=True) # load data  
In [4]: data = moore.data  
In [5]: data = data.rename(columns={"partner.status":  
   ...:                             "partner_status"}) # make name pythonic  
In [6]: moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',  
   ...:                 data=data).fit()  
In [7]: table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 ANOVA DataFrame  
In [8]: print(table)  
                                              sum_sq    df          F  \  
C(fcategory, Sum)                          11.614700   2.0   0.276958     
C(partner_status, Sum)                    212.213778   1.0  10.120692     
C(fcategory, Sum):C(partner_status, Sum)  175.488928   2.0   4.184623     
Residual                                  817.763961  39.0        NaN     
  
                                            PR(>F)    
C(fcategory, Sum)                         0.759564    
C(partner_status, Sum)                    0.002874    
C(fcategory, Sum):C(partner_status, Sum)  0.022572    
Residual                                       NaN    
```

# [时间序列分析](http://www.statsmodels.org/stable/tsa.html)

时间序列分析是根据系统观测得到的时间序列数据，通过曲线拟合和参数估计来建立数学模型的理论和方法。它一般采用曲线拟合和参数估计方法（如非线性最小二乘法）进行。时间序列分析常用在国民经济宏观控制、区域综合发展规划、企业经营管理、市场潜量预测、气象预报、水文预报、地震前兆预报、农作物病虫灾害预报、环境污染控制、生态平衡、天文学和海洋学等方面。  

参考链接：  
[python时间序列分析之ARIMA](https://blog.csdn.net/u010414589/article/details/49622625)  
[AR(I)MA时间序列建模过程——步骤和python代码](https://www.jianshu.com/p/cced6617b423)  
https://www.ziiai.com/blog/638  
https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/  

```python  
from statsmodels import tsa  
```

子模块|说明  
:---|:---  
stattools|经验属性和测试，acf，pacf，granger-causality，adf单位根测试，kpss测试，bds测试，ljung-box测试等。  
ar_model|单变量自回归过程，使用条件和精确最大似然估计和条件最小二乘  
arima_mode|单变量ARMA过程，使用条件和精确最大似然估计和条件最小二乘法  
vector_ar，var|向量自回归过程（VAR）估计模型，脉冲响应分析，预测误差方差分解和数据可视化工具  
kalmanf|使用卡尔曼滤波器的ARMA和其他具有精确MLE的模型的估计类  
arma_process|具有给定参数的arma进程的属性，包括在ARMA，MA和AR表示之间进行转换的工具以及acf，pacf，频谱密度，脉冲响应函数和类似  
sandbox.tsa.fftarma|类似于arma_process但在频域工作  
tsatools|额外的辅助函数，用于创建滞后变量数组，构造趋势，趋势和类似的回归量。  
filters|过滤时间序列的辅助函数  
regime_switching|马尔可夫切换动态回归和自回归模型  

## 时间序列模型

### 常用模型

常用的时间序列模型有四种：自回归模型 $AR(p)$、移动平均模型 $MA(q)$、自回归移动平均模型 $ARMA(p,q)$、自回归差分移动平均模型 $ARIMA(p,d,q)$  
**ARMA模型**  
自回归滑动平均模型（英语：Autoregressive moving average model，简称：ARMA模型）。是研究时间序列的重要方法，由自回归模型（简称AR模型）与移动平均模型（简称MA模型）为基础“混合”构成。  
基本原理：$Y_t=\beta_1Y_{t-1}+\beta_2Y_{t-2}+..\cdots+\beta_pY_{t-p}+Z_t$  
误差项：$Z_t=\varepsilon_t+\alpha_1\varepsilon_{t-1}+\alpha_2\varepsilon_{t-2}+\cdots+\alpha_q\varepsilon_{t-q}$  
**ARIMA模型**  
ARIMA模型（英语：AutoregressiveIntegratedMovingAverage model），差分整合移动平均自回归模型，又称整合移动平均自回归模型（移动也可称作滑动），时间序列预测分析方法之一。ARIMA（p，d，q）中，AR是"自回归"，p为自回归项数；MA为"滑动平均"，q为滑动平均项数，d为使之成为平稳序列所做的差分次数（阶数）。  
当时间序列本身不是平稳的时候，如果它的增量，即一次差分，稳定在零点附近，可以将看成是平稳序列。在实际的问题中，所遇到的多数非平稳序列可以通过一次或多次差分后成为平稳时间序列，则可以建立模型。  

### ARIMA模型运用的流程

1. 根据时间序列的散点图、自相关函数和偏自相关函数图识别其平稳性。  
2. 对非平稳的时间序列数据进行平稳化处理。直到处理后的自相关函数和偏自相关函数的数值非显著非零。  
3. 根据所识别出来的特征建立相应的时间序列模型。平稳化处理后，若偏自相关函数是截尾的，而自相关函数是拖尾的，则建立AR模型；若偏自相关函数是拖尾的，而自相关函数是截尾的，则建立MA模型；若偏自相关函数和自相关函数均是拖尾的，则序列适合ARMA模型。  
4. 参数估计，检验是否具有统计意义。  
5. 假设检验，判断（诊断）残差序列是否为白噪声序列。  
6. 利用已通过检验的模型进行预测  
  

### ARMA example: Sunspots data

**导入数据并作图**  
```python  
from __future__ import print_function  
import numpy as np  
from scipy import stats  
import pandas as pd  
import matplotlib.pyplot as plt  
  
import statsmodels.api as sm  
from statsmodels.graphics.api import qqplot  
  
dta = sm.datasets.sunspots.load_pandas().data  
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))  
del dta["YEAR"]  
dta.plot(figsize=(12,8))  
plt.show()  
```
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/sunpots.png"  style="zoom:75%;" />  

**参数估计**  
```python  
fig = plt.figure(figsize=(12,8))  
ax1 = fig.add_subplot(211)  
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)  
ax2 = fig.add_subplot(212)  
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)  
```
  <img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/acf.png" alt="sunpots" width="75%" />

**拟合模型并评估**  
```python  
>>> arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit(disp=False)  
>>> print(arma_mod20.params)  
const                49.659542  
ar.L1.SUNACTIVITY     1.390656  
ar.L2.SUNACTIVITY    -0.688571  
dtype: float64  
/Users/taugspurger/sandbox/statsmodels/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency A-DEC will be used.  
  % freq, ValueWarning)  
>>> arma_mod30 = sm.tsa.ARMA(dta, (3,0)).fit(disp=False)  
/Users/taugspurger/sandbox/statsmodels/statsmodels/tsa/base/tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency A-DEC will be used.  
  % freq, ValueWarning)  
>>> print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)  
2622.636338065809 2637.5697031734 2628.606725911055  
>>> print(arma_mod30.params)  
const                49.749936  
ar.L1.SUNACTIVITY     1.300810  
ar.L2.SUNACTIVITY    -0.508093  
ar.L3.SUNACTIVITY    -0.129650  
dtype: float64  
>>> print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)  
2619.4036286964474 2638.0703350809363 2626.866613503005  
```
**假设检验**  
```python  
>>> sm.stats.durbin_watson(arma_mod30.resid.values) #D-W检验  
1.9564807635787604  
>>> fig = plt.figure(figsize=(12,8))  
>>> ax = fig.add_subplot(111)  
>>> ax = arma_mod30.resid.plot(ax=ax) #残差正态  
>>> resid = arma_mod30.resid  
>>> stats.normaltest(resid)  
NormaltestResult(statistic=49.845019661107585, pvalue=1.5006917858823576e-11)  
>>> fig = plt.figure(figsize=(12,8))  
>>> ax = fig.add_subplot(111)  
>>> fig = qqplot(resid, line='q', ax=ax, fit=True)  
```

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/arma.png" width="50%" height="50%" />  

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/qq.png" alt="qq" width="50%" height="50%"/>  

**模型预测**  

```python  
predict_sunspots = arma_mod30.predict('1990', '2012', dynamic=True)  
fig, ax = plt.subplots(figsize=(12, 8))  
ax = dta.loc['1950':].plot(ax=ax)  
fig = arma_mod30.plot_predict('1990', '2012', dynamic=True, ax=ax, plot_insample=False)  
```

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/pre.png" style="zoom:67%;" />  

## [状态空间方法](http://www.statsmodels.org/stable/statespace.html)

`statsmodels.tsa.statespace`  

### 模型意义

状态空间模型起源于平稳时间序列分析。当用于非平稳时间序列分析时需要将非平稳时间序列分解为随机游走成分(趋势)和弱平稳成分两个部分分别建模。 含有随机游走成分的时间序列又称积分时间序列，因为随机游走成分是弱平稳成分的和或积分。当一个向量值积分序列中的某些序列的线性组合变成弱平稳时就称这些序列构成了协调积分(cointegrated)过程。 非平稳时间序列的线性组合可能产生平稳时间序列这一思想可以追溯到回归分析，Granger提出的协调积分概念使这一思想得到了科学的论证。 Aoki和Cochrane等人的研究表明：很多非平稳多变量时间序列中的随机游走成分比以前人们认为的要小得多，有时甚至完全消失。[*百度百科*](https://baike.baidu.com/item/%E7%8A%B6%E6%80%81%E7%A9%BA%E9%97%B4%E6%A8%A1%E5%9E%8B/5096878)  

### 状态空间模型的建立和预测的步骤

为了避免由于状态空间模型的不可控制性而导致的错误的分解形式，当对一个单整时间序列建立状态空间分解模型并进行预测，应按下面的步骤执行：  
(1) 对相关的时间序列进行季节调整，并将季节要素序列外推；  
(2) 对季节调整后的时间序列进行单位根检验，确定单整阶数，然后在ARIMA过程中选择最接近的模型；  
(3) 求出ARIMA模型的系数；  
(4) 用ARIMA模型的系数准确表示正规状态空间模型，检验状态空间模型的可控制性；  
(5) 利用Kalman滤波公式估计状态向量，并对时间序列进行预测。  
(6) 把外推的季节要素与相应的预测值合并，就得到经济时间序列的预测结果  

## [矢量自回归](http://www.statsmodels.org/stable/vector_ar.htmls)


```python  
from statsmodels.tsa.api import VAR  
```
向量自回归（VAR）是基于数据的统计性质建立模型，VAR模型把系统中每一个内生变量作为系统中所有内生变量的滞后值的函数来构造模型，从而将单变量自回归模型推广到由多元时间序列变量组成的“向量”自回归模型。VAR模型是处理多个相关经济指标的分析与预测最容易操作的模型之一，并且在一定的条件下，多元MA和ARMA模型也可转化成VAR模型，因此近年来VAR模型受到越来越多的经济工作者的重视。  

### VAR进程(VAR processes)

VAR\(p\)建立$T \times K$多变量时间序列Y，T为观测数量，K为变量数量。  
估计时间序列与其滞后值之间关系的向量自回归过程为：  
$Y_t=A_1Y_{t-1}+\cdots+A_pY_{t-p}+u_t,\  u_t=N(0,\Sigma_u)$  
$A_i$ 是一个 K×K 系数矩阵  

### 模型拟合(Model fitting)  
`statsmodels.tsa.api`  
```python  
 # some example data  
In [1]: import numpy as np  
In [2]: import pandas  
In [3]: import statsmodels.api as sm  
In [4]: from statsmodels.tsa.api import VAR, DynamicVAR  
In [5]: mdata = sm.datasets.macrodata.load_pandas().data  
  
 # prepare the dates index  
In [6]: dates = mdata[['year', 'quarter']].astype(int).astype(str)  
In [7]: quarterly = dates["year"] + "Q" + dates["quarter"]  
In [8]: from statsmodels.tsa.base.datetools import dates_from_str  
In [9]: quarterly = dates_from_str(quarterly)  
In [10]: mdata = mdata[['realgdp','realcons','realinv']]  
In [11]: mdata.index = pandas.DatetimeIndex(quarterly)  
In [12]: data = np.log(mdata).diff().dropna()  
  
 # make a VAR model  
In [13]: model = VAR(data)  
In [14]: results = model.fit(2)  
In [15]: results.summary()  
```
> 注意：本VAR类假定通过时间序列是静止的。非静态或趋势数据通常可以通过第一差分或一些其他方法变换为静止的。对于非平稳时间序列的直接分析，标准稳定VAR（p）模型是不合适的。  


```python  
In [16]: results.plot()  
Out[16]: <Figure size 1000x1000 with 3 Axes>  
```
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/var.png" alt="pre" width="75%;"/>  

绘制时间序列自相关函数：  
```python  
In [17]: results.plot_acorr()  
Out[17]: <Figure size 1000x1000 with 9 Axes>  
```

### 滞后顺序选择(Lag order selection)

滞后顺序的选择可能是一个难题。标准分析采用可能性测试或基于信息标准的顺序选择。我们已经实现了后者，可通过VAR模型访问：  
```python  
In [18]: model.select_order(15)  
Out[18]: <statsmodels.tsa.vector_ar.var_model.LagOrderResults at 0x10c89fef0>  
# 调用fit函数时，可以传递最大滞后数和order标准以用于order选择  
In [19]: results = model.fit(maxlags=15, ic='aic')  
```

### 预测(Forecasting)

The linear predictor is the optimal h-step ahead forecast in terms of mean-squared error:  
$y_t(h)=ν+A_1y_t(h-1)+\cdots+A_py_t(h-p)$  
我们可以使用预测函数来生成此预测。请注意，我们必须为预测指定“初始值”：  

```python  
In [20]: lag_order = results.k_ar  
In [21]: results.forecast(data.values[-lag_order:], 5)  
Out[21]:   
array([[ 0.0062,  0.005 ,  0.0092],  
       [ 0.0043,  0.0034, -0.0024],  
       [ 0.0042,  0.0071, -0.0119],  
       [ 0.0056,  0.0064,  0.0015],  
       [ 0.0063,  0.0067,  0.0038]])  
In [22]: results.plot_forecast(10)  
Out[22]: <Figure size 1000x1000 with 3 Axes>  
```
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/forecast.png"  width="75%" height="75%"/>  

### 脉冲响应分析(Impulse Response Analysis)

在计量经济学研究中，脉冲响应是有意义的：它们是对其中一个变量中单位脉冲的估计响应。它们是在实践中使用$MA(\infty)$计算$VAR(p)$过程：  
$Y_t=\mu + \displaystyle\sum_{i=0}^{\infty} \Phi_i u_{t-i}$  
我们可以通过调用VARResults对象上的irf函数来执行脉冲响应分析：  
```python  
In [23]: irf = results.irf(10)  
#这些可以使用绘图函数以正交或非正交形式可视化。  
#默认情况下，渐近标准误差绘制在95％显着性水平，可由用户修改。  
In [24]: irf.plot(orth=False)  
Out[24]: <Figure size 1000x1000 with 9 Axes>  
#绘图功能非常灵活，如果需要，只能绘制感兴趣的变量  
In [25]: irf.plot(impulse='realgdp')  
Out[25]: <Figure size 1000x1000 with 3 Axes>  
```
累积效应 $\Psi_n=\sum_{i=0}^n \Phi_i$ 可以用长期运行效果绘制：  
```python  
In [26]: irf.plot_cum_effects(orth=False)  
Out[26]: <Figure size 1000x1000 with 9 Axes>  
```

### 预测误差方差分解(FEVD)  
可以使用正交化脉冲响应来分解在`i-step`预测中`k`上的分量`j`的预测误差 $\Theta_i$：  
通过`fevd`函数向前总计步数计算  
```python  
In [27]: fevd = results.fevd(5)  
In [28]: fevd.summary()  
FEVD for realgdp  
      realgdp  realcons   realinv  
0    1.000000  0.000000  0.000000  
1    0.864889  0.129253  0.005858  
2    0.816725  0.177898  0.005378  
3    0.793647  0.197590  0.008763  
4    0.777279  0.208127  0.014594  
  
FEVD for realcons  
      realgdp  realcons   realinv  
0    0.359877  0.640123  0.000000  
1    0.358767  0.635420  0.005813  
2    0.348044  0.645138  0.006817  
3    0.319913  0.653609  0.026478  
4    0.317407  0.652180  0.030414  
  
FEVD for realinv  
      realgdp  realcons   realinv  
0    0.577021  0.152783  0.270196  
1    0.488158  0.293622  0.218220  
2    0.478727  0.314398  0.206874  
3    0.477182  0.315564  0.207254  
4    0.466741  0.324135  0.209124  
```
它们也可以通过返回的FEVD对象可视化  
```python  
In [29]: results.fevd(20).plot()  
Out[29]: <Figure size 1000x1000 with 3 Axes>  
```

### 统计检验(Statistical tests)

提供了许多不同的方法来进行关于模型结果的假设检验以及模型假设的正确性(normality, whiteness / “iid-ness” of errors, etc)  

[**格兰杰因果关系(Granger causality)**](https://baike.baidu.com/item/%E6%A0%BC%E5%85%B0%E6%9D%B0%E5%9B%A0%E6%9E%9C%E5%85%B3%E7%B3%BB%E6%A3%80%E9%AA%8C/2485970)  

- 格兰杰本人在其2003年获奖演说中强调了其引用的局限性，以及“很多荒谬论文的出现”（Of course, many ridiculous papers appeared）。由于其统计学本质上是对平稳时间序列数据一种预测，仅适用于计量经济学的变量预测，不能作为检验真正因果性的判据。  
- 在时间序列情形下，两个经济变量X、Y之间的格兰杰因果关系定义为：若在包含了变量X、Y的过去信息的条件下，对变量Y的预测效果要优于只单独由Y的过去信息对Y进行的预测效果，即变量X有助于解释变量Y的将来变化，则认为变量X是引致变量Y的格兰杰原因。  
- 进行格兰杰因果关系检验的一个前提条件是时间序列必须具有平稳性，否则可能会出现虚假回归问题。因此在进行格兰杰因果关系检验之前首先应对各指标时间序列的平稳性进行单位根检验(unit root test)。常用增广的迪基—富勒检验(ADF检验)来分别对各指标序列的平稳性进行单位根检验  
  
```python  
In [30]: results.test_causality('realgdp', ['realinv', 'realcons'], kind='f')  
Out[30]: <statsmodels.tsa.vector_ar.hypothesis_test_results.CausalityTestResults at 0x10ca15978>  
```

### 动态矢量自动回归(Dynamic Vector Autoregressions)  

> 注意：要使用此功能， 必须安装pandas  

人们通常对估计时间序列数据的移动窗口回归感兴趣，以便在整个数据样本中进行预测。例如，我们可能希望生成由每个时间点估计的VAR\(p\)模型产生的一系列两步预测。  
```python  
In [31]: np.random.seed(1)  
In [32]: import pandas.util.testing as ptest  
In [33]: ptest.N = 500  
In [34]: data = ptest.makeTimeDataFrame().cumsum(0)  
In [35]: data  
Out[35]:   
                    A          B          C          D  
2000-01-03   1.624345  -1.719394  -0.153236   1.301225  
2000-01-04   1.012589  -1.662273  -2.585745   0.988833  
2000-01-05   0.484417  -2.461821  -2.077760   0.717604  
2000-01-06  -0.588551  -2.753416  -2.401793   2.580517  
2000-01-07   0.276856  -3.012398  -3.912869   1.937644  
...               ...        ...        ...        ...  
2001-11-26  29.552085  14.274036  39.222558 -13.243907  
2001-11-27  30.080964  11.996738  38.589968 -12.682989  
2001-11-28  27.843878  11.927114  38.380121 -13.604648  
2001-11-29  26.736165  12.280984  40.277282 -12.957273  
2001-11-30  26.718447  12.094029  38.895890 -11.570447  
  
[500 rows x 4 columns]  
In [36]: var = DynamicVAR(data, lag_order=2, window_type='expanding')  
```
动态模型的估计系数作为pandas.Panel对象返回 ，这可以让您轻松地按等式或按日期检查所有模型系数：  
```python  
In [37]: import datetime as dt  
In [38]: var.coefs  
Out[38]:   
<class 'pandas.core.panel.Panel'>  
Dimensions: 9 (items) x 489 (major_axis) x 4 (minor_axis)  
Items axis: L1.A to intercept  
Major_axis axis: 2000-01-18 00:00:00 to 2001-11-30 00:00:00  
Minor_axis axis: A to D  
  
 # all estimated coefficients for equation A  
In [39]: var.coefs.minor_xs('A').info()  
<class 'pandas.core.frame.DataFrame'>  
DatetimeIndex: 489 entries, 2000-01-18 to 2001-11-30  
Freq: B  
Data columns (total 9 columns):  
L1.A         489 non-null float64  
L1.B         489 non-null float64  
L1.C         489 non-null float64  
L1.D         489 non-null float64  
L2.A         489 non-null float64  
L2.B         489 non-null float64  
L2.C         489 non-null float64  
L2.D         489 non-null float64  
intercept    489 non-null float64  
dtypes: float64(9)  
memory usage: 58.2 KB  
  
 # coefficients on 11/30/2001  
In [40]: var.coefs.major_xs(dt.datetime(2001, 11, 30)).T  
Out[40]:   
                  A         B         C         D  
L1.A       0.971964  0.045960  0.003883  0.003822  
L1.B       0.043951  0.937964  0.000735  0.020823  
L1.C       0.038144  0.018260  0.977037  0.129287  
L1.D       0.038618  0.036180  0.052855  1.002657  
L2.A       0.013588 -0.046791  0.011558 -0.005300  
L2.B      -0.048885  0.041853  0.012185 -0.048732  
L2.C      -0.029426 -0.015238  0.011520 -0.119014  
L2.D      -0.049945 -0.025419 -0.045621 -0.019496  
intercept  0.113331  0.248795 -0.058837 -0.089302  
```
可以使用`forecast`函数生成前面给定步骤的动态预测，并返回pandas.DataMatrix对象：  
```python  
In [41]: var.forecast(2)  
Out[41]:   
                     A          B           C           D  
2000-01-20 -260.325888 -23.141610  104.930427 -134.489882  
2000-01-21  -52.121483 -11.566786   29.383608  -15.099109  
2000-01-24  -54.900049 -23.894858   40.470913  -19.199059  
2000-01-25   -7.493484  -4.057529    6.682707    4.301623  
2000-01-26   -6.866108  -5.065873    5.623590    0.796081  
...                ...        ...         ...         ...  
2001-11-26   31.886126  13.515527   37.618145  -11.464682  
2001-11-27   32.314633  14.237672   37.397691  -12.809727  
2001-11-28   30.896528  15.488388   38.541596  -13.129524  
2001-11-29   30.077228  15.533337   38.734096  -12.900891  
2001-11-30   30.510380  13.491615   38.088228  -12.384976  
  
[487 rows x 4 columns]  
```
可以使用plot_forecast显示预测：  
```python  
In [42]: var.plot_forecast(2)  
```

  <img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/dvar_forecast.png" alt="forecast" width="75%" height="75%"/>


# [生存分析](http://www.statsmodels.org/stable/duration.html)  

生存分析是用于分析直到一个或多个事件发生的预期持续时间的统计分支，例如生物有机体中的死亡和机械系统中的失败。本主题被称为可靠性理论和可靠性分析的工程，持续时间分析或持续时间建模在经济学和事件历史分析，在社会学。生存分析试图回答以下问题：在一定时间内存活的人口比例是多少？那些幸存下来的人会以什么样的速度死亡或失败？可以考虑多种死亡原因吗？具体情况或特征如何增加或减少生存的概率？[--*Wiki*](https://en.wikipedia.org/wiki/Survival_analysis)  


理论链接：  
[生存分析(survival analysis)](https://www.cnblogs.com/wwxbi/p/6136348.html)  
[生存分析学习笔记](https://blog.csdn.net/jaen_tail/article/details/79081954)  

`statsmodels.duration`实现了几种处理删失数据的标准方法。当数据由起始时间点和某些感兴趣事件发生的时间之间的持续时间组成时，最常使用这些方法。  

目前只处理右侧审查。当我们知道在给定时间t之后发生事件时发生了右删失，但我们不知道确切的事件时间。  

## 生存函数估计和推理  

`statsmodels.api.SurvfuncRight`类可以被用来估计可以右删失数据的生存函数。 `SurvfuncRight`实现了几个推理程序，包括生存分布分位数的置信区间，生存函数的逐点和同时置信带以及绘图程序。`duration.survdiff`函数提供了比较生存分布的检验程序。  

**Example**  
在这里，我们SurvfuncRight使用flchain研究中的数据创建一个对象 ，该数据可通过R数据集存储库获得。我们只适合女性受试者的生存分布。  
```python  
import statsmodels.api as sm  
  
data = sm.datasets.get_rdataset("flchain", "survival").data  
df = data.loc[data.sex == "F", :]  
sf = sm.SurvfuncRight(df["futime"], df["death"])  
  
# 通过调用summary方法可以看出拟合生存分布的主要特征  
sf.summary().head()  
  
#我们可以获得生存分布的分位数的点估计和置信区间。  
#由于在这项研究中只有约30％的受试者死亡，我们只能估计低于0.3概率点的分位数  
sf.quantile(0.25)  
sf.quantile_ci(0.25)  
```
要绘制单个生存函数，请调用plot方法：  
`sf.plot()`  
由于这是一个包含大量删失的大型数据集，我们可能希望不绘制删失符号：  
```python  
fig = sf.plot()  
ax = fig.get_axes()[0]  
pt = ax.get_lines()[1]  
pt.set_visible(False)  
  
#我们还可以为情节添加95％的同时置信带。通常，这些波段仅针对分布的中心部分绘制。  
fig = sf.plot()  
lcb, ucb = sf.simultaneous_cb()  
ax = fig.get_axes()[0]  
ax.fill_between(sf.surv_times, lcb, ucb, color='lightgrey')  
ax.set_xlim(365, 365*10)  
ax.set_ylim(0.7, 1)  
ax.set_ylabel("Proportion alive")  
ax.set_xlabel("Days since enrollment")  
  
#在这里，我们在同一轴上绘制两组（女性和男性）的生存函数：  
gb = data.groupby("sex")  
ax = plt.axes()  
sexes = []  
for g in gb:  
    sexes.append(g[0])  
    sf = sm.SurvfuncRight(g[1]["futime"], g[1]["death"])  
    sf.plot(ax)  
li = ax.get_lines()  
li[1].set_visible(False)  
li[3].set_visible(False)  
plt.figlegend((li[0], li[2]), sexes, "center right")  
plt.ylim(0.6, 1)  
ax.set_ylabel("Proportion alive")  
ax.set_xlabel("Days since enrollment")  
```
我们可以正式比较两个生存分布`survdiff`，它实现了几个标准的非参数程序。默认程序是`logrank`检测：  
```python  
stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex)  
  
#以下是survdiff实施的一些其他测试程序  
# Fleming-Harrington with p=1, i.e. weight by pooled survival time  
stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='fh', fh_p=1)  
# Gehan-Breslow, weight by number at risk  
stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='gb')  
# Tarone-Ware, weight by the square root of the number at risk  
stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='tw')  
```

## 回归方法  

比例风险回归模型（“Cox模型”）是用于删失数据的回归技术。它们允许以协变量的形式解释事件的时间变化，类似于线性或广义线性回归模型中所做的。这些模型以“风险比”表示协变量效应，这意味着危险（瞬时事件率）乘以给定因子，取决于协变量的值。  

**Example**  
```python  
import statsmodels.api as sm  
import statsmodels.formula.api as smf  
  
data = sm.datasets.get_rdataset("flchain", "survival").data  
del data["chapter"]  
data = data.dropna()  
data["lam"] = data["lambda"]  
data["female"] = (data["sex"] == "F").astype(int)  
data["year"] = data["sample.yr"] - min(data["sample.yr"])  
status = data["death"].values  
  
mod = smf.phreg("futime ~ 0 + age + female + creatinine + "  
                "np.sqrt(kappa) + np.sqrt(lam) + year + mgus",  
                data, status=status, ties="efron")  
rslt = mod.fit()  
print(rslt.summary())  
```

# [列联表](http://www.statsmodels.org/stable/contingency_tables.html)

Statsmodels支持各种分析列联表的方法，包括评估独立性，对称性，同质性的方法，以及处理来自分层人群的表集合的方法。  

这里描述的方法主要用于双向表。可以使用对数线性模型分析多路表。Statsmodels目前没有statsmodels.genmod.GLM用于对数线性建模的专用API，但Poisson回归可用于此目的。  

`from statsmodels import stats`  


| 类              | 说明                           |
| :-------------- | :----------------------------- |
| Table           | 双向应变表列联表               |
| Table2x2        | 可以在2x2列联表上执行分析      |
| SquareTable     | 分析方形列联表的方法           |
| StratifiedTable | 分析2×2列联表的集合            |
| mcnemar         | McNemar同质性测试              |
| cochrans_q      | Cochran对相同二项式比例的Q检验 |

statsmodels.stats.Table是使用列联表的最基本的类。我们可以Table直接从任何包含列联表单元格计数的矩形数组对象创建对象：  

```python  
In [1]: import numpy as np  
In [2]: import pandas as pd  
In [3]: import statsmodels.api as sm  
In [4]: df = sm.datasets.get_rdataset("Arthritis", "vcd").data  
In [5]: tab = pd.crosstab(df['Treatment'], df['Improved'])  
In [6]: tab = tab.loc[:, ["None", "Some", "Marked"]]  
In [7]: table = sm.stats.Table(tab)  
```

或者，我们可以传递原始数据，让Table类为我们构建单元格数组：  

```python  
In [8]: table = sm.stats.Table.from_data(df[["Treatment", "Improved"]])  
```

## Independence(独立性)  

独立性(Independence)是行和列因子独立出现的属性。联合(Association)缺乏独立性。如果联合分布是独立的，则可以将其写为行和列边缘分布的外积  

$P_ {ij} = \sum_k P_ {ij} \cdot \sum_k P_ {kj}$   

我们可以为我们观察到的数据获得最佳拟合的独立分布，然后查看识别最强烈违反独立性的特定残差  

```python  
In [9]: print(table.table_orig)  
Improved   Marked  None  Some  
Treatment                      
Placebo         7    29     7  
Treated        21    13     7  
  
In [10]: print(table.fittedvalues)  
Improved      Marked  None      Some  
Treatment                             
Placebo    14.333333  21.5  7.166667  
Treated    13.666667  20.5  6.833333  
  
In [11]: print(table.resid_pearson)  
Improved     Marked      None      Some  
Treatment                                
Placebo   -1.936992  1.617492 -0.062257  
Treated    1.983673 -1.656473  0.063758  
```

如果表的行和列是无序的（即名义变量，nominal factors），那么正式评估独立性的最常用方法是使用Pearson的$\chi^2$统计。  

```python  
In [12]: rslt = table.test_nominal_association()  
  
In [13]: print(rslt.pvalue)  
0.0014626434089526352  
  
In [14]: print(table.chi2_contribs)  
Improved     Marked      None      Some  
Treatment                                
Placebo    3.751938  2.616279  0.003876  
Treated    3.934959  2.743902  0.004065  
```

对于有序行和列因子(factors)的表，我们可以通过线性相关检验，以获得更多权重来对抗关于排序的替代假设。线性相关检验的统计量为  
$\sum_k r_i c_j T_ {ij}$  

$r_i ,c_j$是行和列分数。通常将这些分数设置为序列0,1，....   
这给出了Cochran-Armitage趋势测试。  

```python  
In [15]: rslt = table.test_ordinal_association()  
In [16]: print(rslt.pvalue)  
0.023644578093923983  
```

我们可以评估再 $r \times x$ 表中的关联，通过构建一系列$2 \times 2$表格，并计算它们的比值比(OR)。有两种方法可以做到这一点。从相邻行和列类别的本地优势比(**local odds ratios**)来构建$2 \times 2$表。  

```python  
In [17]: print(table.local_oddsratios)  
Improved     Marked      None  Some  
Treatment                            
Placebo    0.149425  2.230769   NaN  
Treated         NaN       NaN   NaN  
  
In [18]: taloc = sm.stats.Table2x2(np.asarray([[7, 29], [21, 13]]))  
In [19]: print(taloc.oddsratio)  
0.14942528735632185  
  
In [20]: taloc = sm.stats.Table2x2(np.asarray([[29, 7], [13, 7]]))  
In [21]: print(taloc.oddsratio)  
2.230769230769231  
```

也可以通过在每个可能的点上对行和列因子进行二分法的累积比值比(**cumulative odds ratios**)来构建$2 \times 2$表。  

```python  
In [22]: print(table.cumulative_oddsratios)  
Improved     Marked      None  Some  
Treatment                            
Placebo    0.185185  1.058824   NaN  
Treated         NaN       NaN   NaN  
  
In [23]: tab1 = np.asarray([[7, 29 + 7], [21, 13 + 7]])  
In [24]: tacum = sm.stats.Table2x2(tab1)  
In [25]: print(tacum.oddsratio)  
0.18518518518518517  
  
In [26]: tab1 = np.asarray([[7 + 29, 7], [21 + 13, 7]])  
In [27]: tacum = sm.stats.Table2x2(tab1)  
In [28]: print(tacum.oddsratio)  
1.0588235294117647  
```

马赛克图(mosaic plot)是一种非正式评估双向表中依赖性的图形方法。  

```python  
from statsmodels.graphics.mosaicplot import mosaic  
mosaic(data)  
```

## Symmetry and homogeneity(对称性和同质性)  

Symmetry(对称性) is the property  
 that $P_{i,j}=P_{j,i}$  for every i and j。   
 Homogeneity(同质性)是行因子和列因子的边际分布相同的特性  
meaning that  $\sum_j P_ {ij} = \sum_j P_ {ji}$ for all i  

> 注意，P (and T)必须是正方形，行和列类别必须相同，并且必须以相同的顺序出现。  

为了说明，我们加载数据集，创建列联表，并计算行和列边距(the row and column margins)。本`Table`类包含分析方法$r \times c$列联表。下面加载的数据集包含人们左眼和右眼视敏度的评估。我们首先加载数据并创建一个列联表。  

```python  
In [29]: df = sm.datasets.get_rdataset("VisualAcuity", "vcd").data  
In [30]: df = df.loc[df.gender == "female", :]  
In [31]: tab = df.set_index(['left', 'right'])  
In [32]: del tab["gender"]  
In [33]: tab = tab.unstack()  
In [34]: tab.columns = tab.columns.get_level_values(1)  
In [35]: print(tab)  
right     1     2     3    4  
left                          
1      1520   234   117   36  
2       266  1512   362   82  
3       124   432  1772  179  
4        66    78   205  492  
  
# 从列联表创建一个SquareTable对象  
In [36]: sqtab = sm.stats.SquareTable(tab)  
In [37]: row, col = sqtab.marginal_probabilities  
In [38]: print(row)  
right  
1    0.255049  
2    0.297178  
3    0.335295  
4    0.112478  
dtype: float64  
  
In [39]: print(col)  
right  
1    0.264277  
2    0.301725  
3    0.328474   
4    0.105524  
dtype: float64  
  
# 该summary方法打印对称性和均匀性检测结果  
In [40]: print(sqtab.summary())   
            Statistic P-value DF  
--------------------------------  
Symmetry       19.107   0.004  6  
Homogeneity    11.957   0.008  3  
--------------------------------  
```

## A single 2x2 table(单个2x2表)  

`sm.stats.Table2x2`类提供了几种处理单个2x2表的方法。summary方法显示表的行和列之间的若干关联度量。  

```python  
In [41]: table = np.asarray([[35, 21], [25, 58]])  
In [42]: t22 = sm.stats.Table2x2(table)  
In [43]: print(t22.summary())  
               Estimate   SE   LCB   UCB  p-value  
-------------------------------------------------  
Odds ratio        3.867       1.890 7.912   0.000  
Log odds ratio    1.352 0.365 0.636 2.068   0.000  
Risk ratio        2.075       1.411 3.051   0.000  
Log risk ratio    0.730 0.197 0.345 1.115   0.000  
-------------------------------------------------  
  
#请注意，风险比不是对称的，因此如果分析转置表，将获得不同的结果。  
In [44]: table = np.asarray([[35, 21], [25, 58]])  
In [45]: t22 = sm.stats.Table2x2(table.T)  
In [46]: print(t22.summary())  
               Estimate   SE   LCB   UCB  p-value  
-------------------------------------------------  
Odds ratio        3.867       1.890 7.912   0.000  
Log odds ratio    1.352 0.365 0.636 2.068   0.000  
Risk ratio        2.194       1.436 3.354   0.000  
Log risk ratio    0.786 0.216 0.362 1.210   0.000  
-------------------------------------------------  
```

## Stratified 2x2 tables(分层2x2表)  

当我们有一组由同样行和列因子定义的列联表时，就会发生分层。  

**案例**  

- 我们有一组2x2表，反映了中国几个地区吸烟和肺癌的联合分布。表格可能都具有共同的比值比，即使边际概率在各阶层之间变化也是如此。  
- “Breslow-Day”程序测试数据是否与常见优势比一致。它在下面显示为常数OR的测试。  
- Mantel-Haenszel程序测试这个常见优势比是否等于1。它在下面显示为OR = 1的测试。还可以估计共同的几率和风险比并获得它们的置信区间。  
- summary方法显示所有这些结果。可以从类方法和属性中获得单个结果。  

```python  
In [47]: data = sm.datasets.china_smoking.load()  
In [48]: mat = np.asarray(data.data)  
In [49]: tables = [np.reshape(x.tolist()[1:], (2, 2)) for x in mat]  
In [50]: st = sm.stats.StratifiedTable(tables)  
In [51]: print(st.summary())  
                   Estimate   LCB    UCB   
-----------------------------------------  
Pooled odds           2.174   1.984 2.383  
Pooled log odds       0.777   0.685 0.868  
Pooled risk ratio     1.519                
                                           
                 Statistic P-value   
-----------------------------------  
Test of OR=1       280.138   0.000   
Test constant OR     5.200   0.636   
                         
-----------------------  
Number of tables    8    
Min n             213    
Max n            2900    
Avg n            1052    
Total n          8419    
-----------------------  
```

# [多重插补](http://www.statsmodels.org/stable/imputation.html)


MICE模块允许大多数Statsmodels模型拟合独立和/或因变量上具有缺失值的数据集，并为拟合参数提供严格的标准误差。基本思想是将具有缺失值的每个变量视为回归中的因变量，其中一些或所有剩余变量作为其预测变量。MICE程序循环遍历这些模型，依次拟合每个模型，然后使用称为“预测平均匹配”（PMM）的过程从拟合模型确定的预测分布中生成随机抽取。这些随机抽取成为一个插补数据集的估算值。  

默认情况下，每个具有缺失变量的变量都使用线性回归建模，拟合数据集中的所有其他变量。请注意，即使插补模型是线性的，PMM过程也会保留每个变量的域。因此，例如，如果给定变量的所有观测值都是正，则变量的所有估算值将始终为正。用户还可以选择指定使用哪个模型为每个变量生成插补值。  

```python  
from statsmodels.imputation import mice  
```

| 类                                                           | 说明                             |
| :----------------------------------------------------------- | :------------------------------- |
| MICE(model_formula, model_class, data, n_skip=3, init_kwds=None, fit_kwds=None) | 用链式方程多重插补               |
| MICEData（data [，perturbation_method，k_pmm，...]）         | 包装数据集以允许MICE处理丢失数据 |


| MICEData类方法                             | 说明                                |
| ------------------------------------------ | ----------------------------------- |
| get_fitting_data                           | 返回进行插补所需的数据              |
| get_split_data(vname)                      | 返回endog和exog以获取给定变量的插补 |
| impute(vname)                              |                                     |
| impute_pmm(vname)                          | 使用预测均值匹配来估算缺失值        |
| next_sample()                              | 返回插补过程中的下一个插补数据集    |
| perturb_params(vname)                      |                                     |
| plot_bivariate(col1_name,col2_name [,...]) | 绘制两个变量的观察值和估算值        |
| plot_fit_obs(col_name[, lowess_args, …])   | 绘制拟合估计值或观察值的散点图      |
| plot_imputed_hist(col_name[, ax, …])       | 将一个变量的估算值显示为直方图      |
| plot_missing_pattern([ax, row_order, …])   | 生成显示缺失数据模式的图像          |
| set_imputer(endog_name[, formula, …])      | 指定单个变量的插补过程。            |
| update(vname)                              | 为单个变量计算缺失值。              |
| update_all([n_iter])                       | 执行指定数量的MICE迭代。            |

```python  
>>> imp = mice.MICEData(data)  
>>> fml = 'y ~ x1 + x2 + x3 + x4'  
>>> mice = mice.MICE(fml, sm.OLS, imp)  
>>> results = mice.fit(10, 10)  
>>> print(results.summary())  
                          Results: MICE  
=================================================================  
Method:                    MICE       Sample size:           1000  
Model:                     OLS        Scale                  1.00  
Dependent variable:        y          Num. imputations       10  
-----------------------------------------------------------------  
           Coef.  Std.Err.    t     P>|t|   [0.025  0.975]  FMI  
-----------------------------------------------------------------  
Intercept -0.0234   0.0318  -0.7345 0.4626 -0.0858  0.0390 0.0128  
x1         1.0305   0.0578  17.8342 0.0000  0.9172  1.1437 0.0309  
x2        -0.0134   0.0162  -0.8282 0.4076 -0.0451  0.0183 0.0236  
x3        -1.0260   0.0328 -31.2706 0.0000 -1.0903 -0.9617 0.0169  
x4        -0.0253   0.0336  -0.7520 0.4521 -0.0911  0.0406 0.0269  
=================================================================  
  
#获得一系列拟合分析模型，无需合并即可获得摘要：  
>>> imp = mice.MICEData(data)  
>>> fml = 'y ~ x1 + x2 + x3 + x4'  
>>> mice = mice.MICE(fml, sm.OLS, imp)  
>>> results = []  
>>> for k in range(10):  
>>>     x = mice.next_sample()  
>>>     results.append(x)  
```


# [多变量统计](https://www.statsmodels.org/stable/multivariate.html)  


多变量统计是一种统计方法，当数据种包含不止一个结果变量时，同时进行观察和分析。多变量统计的应用是多变量分析。  

多变量统计涉及了解每种不同形式的多变量分析的不同目的和背景，以及它们如何相互关联。多变量统计对特定问题的实际应用可能涉及几种类型的单变量和多变量分析，以便理解变量之间的关系及其与所研究问题的相关性。*[---Wiki](https://en.wikipedia.org/wiki/Multivariate_statistics)*  

`statsmodels.multivariate`  

## 主成分分析(Principal Component Analysis)  

主成分分析是设法将原来众多具有一定相关性（比如P个指标），重新组合成一组新的互相无关的综合指标来代替原来的指标。  

主成分分析，是考察多个变量间相关性一种多元统计方法，研究如何通过少数几个主成分来揭示多个变量间的内部结构，即从原始变量中导出少数几个主成分，使它们尽可能多地保留原始变量的信息，且彼此间互不相关.通常数学上的处理就是将原来P个指标作线性组合，作为新的综合指标。  

```python  
PCA(data, ncomp=None, standardize=True, demean=True, normalize=True, ...)  
pca(...)  
```

**例子**  

```python  
>>> import numpy as np  
>>> from statsmodels.multivariate.pca import PCA  
>>> x = np.random.randn(100)[:, None]  
>>> x = x + np.random.randn(100, 100)  
>>> pc = PCA(x)  
```

> 注意，主成分是使用SVD计算的，因此从不构造相关矩阵，除非method ='eig'。  

PCA使用数据的协方差矩阵  

```python  
>>> pc = PCA(x, standardize=False)  
```

使用NIPALS将返回的因子数限制为1  

```python  
>>> pc = PCA(x, ncomp=1, method='nipals')  
>>> pc.factors.shape  
(100, 1)  
```

## 因子分析(Factor Analysis)  

因子分析法是从研究变量内部相关的依赖关系出发，把一些具有错综复杂关系的变量归结为少数几个综合因子的一种多变量统计分析方法。它的基本思想是将观测变量进行分类，将相关性较高，即联系比较紧密的分在同一类中，而不同类变量之间的相关性则较低，那么每一类变量实际上就代表了一个基本结构，即公共因子（隐性变量, latent variable, latent factor）。对于所研究的问题就是试图用最少个数的不可测的所谓公共因子的线性函数与特殊因子之和来描述原来观测的每一分量。  

因子分析的方法有两类。一类是探索性因子分析法，另一类是验证性因子分析。探索性因子分析不事先假定因子与测度项之间的关系，而让数据“自己说话”。主成分分析和共因子分析是其中的典型方法。验证性因子分析假定因子与测度项的关系是部分知道的，即哪个测度项对应于哪个因子，虽然我们尚且不知道具体的系数。  

```  
Factor(endog=None, n_factor=1, corr=None,...)  #因子分析  
FactorResults(factor) #因子分析结果类  
```

**因子旋转(Factor Rotation)**  
`statsmodels.multivariate.factor_rotation`  

| 类              | 说明                               |
| --------------- | ---------------------------------- |
| rotate_factors  | 矩阵正交和倾斜旋转                 |
| target_rotation | 朝向目标矩阵的正交旋转，即最小化： |
| procrustes      | 分析解决Procrustes问题             |
| promax          | 执行矩阵的promax旋转               |

```python  
>>> A = np.random.randn(8,2)  
>>> L, T = rotate_factors(A,'varimax')  
>>> np.allclose(L,A.dot(T))  
>>> L, T = rotate_factors(A,'orthomax',0.5)  
>>> np.allclose(L,A.dot(T))  
>>> L, T = rotate_factors(A,'quartimin',0.5)  
>>> np.allclose(L,A.dot(np.linalg.inv(T.T)))  
```

## 典型相关(Canonical Correlation)  

典型相关分析（canonical correlation analysis），是对互协方差矩阵的一种理解，是利用综合变量对之间的相关关系来反映两组指标之间的整体相关性的多元统计分析方法。它的基本原理是：为了从总体上把握两组指标之间的相关关系，分别在两组变量中提取有代表性的两个综合变量U1和V1（分别为两个变量组中各变量的线性组合），利用这两个综合变量之间的相关关系来反映两组指标之间的整体相关性。  

`CanCorr(endog, exog,...)  # 使用单因素分解的典型相关分析`  

## 多元方差分析(MANOVA)  

在统计学中，多元方差分析（MANOVA）是一种比较多变量样本均值的程序 。作为一个多变量过程，它在有两个或多个因变量时使用，并且通常后面是分别涉及各个因变量的显着性检验。  

MANOVA是单变量方差分析（ANOVA）的推广形式，尽管与单变量ANOVA不同，它使用结果变量之间的协方差来检验平均差异的统计显着性。其中，在单变量方差分析中出现平方和的情况下，在多变量方差分析中出现某些正定矩阵。对角线条目是出现在单变量ANOVA中的相同种类的平方和，非对角线条目则是相应的乘积和。在关于误差分布的正态假设下，由于误差导致的平方和对应部分服从Wishart分布。  

`MANOVA(endog, exog,...)`  

## 多元线性模型(MultivariateOLS)  

`_MultivariateOLS`是一个功能有限的模型类。目前它支持多变量假设检验，并用作MANOVA的后端  

# 可视化

## 拟合图

Goodness of Fit Plots| 拟合图  
:------|:------  
gofplots.qqplot|QQ图。  
gofplots.qqline|绘制一个qqplot的参考线  
gofplots.qqplot_2samples|两个样本分位数的QQ图  
gofplots.ProbPlot|自定义QQ图，PP图或概率图  

 

```python  
>>> import statsmodels.api as sm  
>>> from matplotlib import pyplot as plt  
>>> import scipy.stats as stats  
>>> data = sm.datasets.longley.load()  
>>> data.exog = sm.add_constant(data.exog)  
>>> mod_fit = sm.OLS(data.endog, data.exog).fit()  
>>> res = mod_fit.resid # residuals  
>>> fig = sm.qqplot(res, stats.t, fit=True, line='45')  
>>> plt.show()  
```
![qq](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/fit.png)  


## 箱线图  

Boxplots|箱线图  
:------|:------  
boxplots.violinplot|在数据序列中为每个数据集制作小提琴图  
boxplots.beanplot|在数据序列中创建每个数据集的bean图  

```python  
>>> data = sm.datasets.anes96.load_pandas()  
>>> party_ID = np.arange(7)  
>>> labels = ["Strong Democrat", "Weak Democrat", "Independent-Democrat",  
...           "Independent-Indpendent", "Independent-Republican",  
...           "Weak Republican", "Strong Republican"]  
>>> plt.rcParams['figure.subplot.bottom'] = 0.23  # keep labels visible  
>>> age = [data.exog['age'][data.endog == id] for id in party_ID]  
>>> fig = plt.figure()  
>>> ax = fig.add_subplot(111)  
>>> sm.graphics.violinplot(age, ax=ax, labels=labels,  
...                        plot_opts={'cutoff_val':5, 'cutoff_type':'abs',  
...                                   'label_fontsize':'small',  
...                                   'label_rotation':30})  
>>> ax.set_xlabel("Party identification of respondent.")  
>>> ax.set_ylabel("Age")  
>>> plt.show()  
```
![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/box.png)  

## 相关图  

Correlation Plots|相关图  
:------|:------  
correlation.plot_corr|相关图  
correlation.plot_corr_grid|创建相关图的网格  
plot_grids.scatter_ellipse|用置信度椭圆创建一个散点图网格  

```python  
>>> import numpy as np  
>>> import matplotlib.pyplot as plt  
>>> import statsmodels.graphics.api as smg  
>>> hie_data = sm.datasets.randhie.load_pandas()  
>>> corr_matrix = np.corrcoef(hie_data.data.T)  
>>> smg.plot_corr(corr_matrix, xnames=hie_data.names)  
>>> plt.show()  
```
  ![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/corr.png)

## 函数图  
Functional Plots|函数图  
:------|:------  
functional.hdrboxplot|高密度区域箱线图  
functional.fboxplot|绘制函数箱线图  
functional.rainbowplot|为一组曲线创建一个彩虹图  
functional.banddepth|计算一组函数曲线的带深度  

```python  
>>> import matplotlib.pyplot as plt  
>>> import statsmodels.api as sm  
>>> data = sm.datasets.elnino.load()  
>>> fig = plt.figure()  
>>> ax = fig.add_subplot(111)  
>>> res = sm.graphics.hdrboxplot(data.raw_data[:, 1:],  
...                              labels=data.raw_data[:, 0].astype(int),  
...                              ax=ax)  
>>> ax.set_xlabel("Month of the year")  
>>> ax.set_ylabel("Sea surface temperature (C)")  
>>> ax.set_xticks(np.arange(13, step=3) - 1)  
>>> ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])  
>>> ax.set_xlim([-0.2, 11.2])  
>>> plt.show()  
```
![fp](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/statsmodels/fun.png)  

## 回归图  
Regression Plots|回归图  
:------|:------  
regressionplots.plot_fit|Plot fit against one regressor  
regressionplots.plot_regress_exog|针对一个回归模型绘制回归结果。  
regressionplots.plot_partregress|绘制对于单个回归模型的部分回归。  
regressionplots.plot_ccpr|将CCPR与一位回归模型对比。  
regressionplots.abline_plot|绘制斜线  
regressionplots.influence_plot|回归影响  
regressionplots.plot_leverage_resid2|Plots leverage statistics vs  


## 时间序列图  
Time Series Plots|时间序列图  
:------|:------  
tsaplots.plot_acf|绘制自相关函数  
tsaplots.plot_pacf|绘制部分自相关函数  
tsaplots.month_plot|每月数据的季节性  
tsaplots.quarter_plot|季度数据的季节性  


## 其他  
Other Plots|其他  
:------|:------  
factorplots.interaction_plot|对每个因子水平的交互作用图  
mosaicplot.mosaic|马赛克图  
agreement.mean_diff_plot|Tukey’s Mean Difference Plot  

