---
ID: ab8ec05998a2fb5954b82d3f3d89a886
title: R手册(Time Series)--forecast and prophet
tags: [R,时间序列]
date: 2018-05-04 17:02:36
categories: [R,时间序列]
---

时间序列分析(Time-Series Analysis)是指将原来的数据分解为四部分来看——趋势、周期、时期和不稳定因素， 然后综合这些因素， 提出预测。

<!-- more -->

# forecast : for Time Series and Linear Models

## 时间序列分析

`tsclean(x, replace.missing = TRUE, lambda = NULL) `识别和替换异常值和缺失值（lambda给出Box-Cox变换参数的数值）
`ndiffs, nsdiffs`固定系列所需的差异数
`seasonal(object)`提取季节分量
`trendcycle(object)`提取趋势周期分量
`remainder(object)`提取余数分量
`findfrequency`查找时间序列的主频
`ma(x, order, centre = TRUE)`计算更平滑的移动平均

## 模型

`arfima`: FitARFIMAmodel
`Arima, auto.arima` : FitARIMAmodel
`ets(y,model=”ZZZ”)`指数预测模型
`baggedETS, bats, tbats`: FitbaggedETS/BATS/TBATSmodel
`nnetar`神经网络时间序列预测

**forecastHybrid**: 组合模型

```r
hybridModel(y, models = "aefnst",                         #模型组a(auto.arima),e(ets),f(thetam),n(nnetar),s(stlm),t(tbats)
  weights = c("equal", "insample.errors", "cv.errors"),    #模型加权方法
  parallel = FALSE,                                        #是否并行运算
  num.cores = 2L)                                          #并行内核数
```

```r
example: 
hybridModel(wineind, models = "aet", weights = "equal")%>%
  forecast(hm1, h = 48)%>%plot()
```

## 预测

`forecast(object, h = ifelse(frequency(object) > 1, 2*frequency(object), 10),level=c(80,95))`

> 参数：
> h：预测数
> level：置信区间

## ggplot2扩展

`Acf, Pacf, Ccf, taperedacf, taperedpacf`自相关和相关函数估计
`autoplot(object, …)`通用制图函数

## 模型评估

`checkresiduals(object, lag, df = NULL, test, plot = TRUE, ...)`检查残差
`accuracy(f, x)`准确率
`CV, CVar, dsCV`交叉验证
`dm.test`Diebold-Mariano测试的预测准确性

`example: WWWusage %>%auto.arima %>%forecast(h=20) %>%autoplot()`

# prophet

模型组成：Y(t)=Trend(t)+Seasonal(t)+Holiday(t)+Irregular(t)

## 构建模型

```r
prophet(df = df,              #data.frame:ds(date type)+ y,cap+floor指定饱和最大值和最小值
  growth = "linear",           #linearorlogistictrend
  changepoints = NULL,          #包含潜在变化点的日期向量
  n.changepoints = 25,          #潜在变化点数
  changepoint_prior_scale=0.05,   #调整trend灵活性
  yearly.seasonality = "auto",     #适合年度seasonality;'auto',TRUE,orFALSE
  weekly.seasonality = "auto",     #适合周度
  holidays = NULL,                 #data.frame:holiday(character)+ds(datetype),lower_window+upper_window(可选，指定假日周围的天数)
  seasonality.prior.scale = 10,     #调整季节性模型的强度
  holidays.prior.scale = 10,        #调整假期组件模型的强度
  mcmc.samples = 0, 
  interval.width = 0.8,             #trend间隔不确定性
  uncertainty.samples = 1000,       #season的不确定习性
  fit = TRUE)
```

## 模型预测

```r
furture<-make_future_dataframe(m,           #Prophet model object|
  periods,                                  #要预测的数量
  freq = "d",                               #day','week','month','quarter',or'year'
  include_history = TRUE)                   #历史日期是否包含在预测中
```

`predict(object,df = NULL)`

> 参数：
> object：Prophet modelo bject 
> df：NULL or future

## 可视化

`plot(x, fcst)`
`prophet_plot_components(m, fcst)` ggplot2组件，将预测细分为趋势，每周季节性和年度季节

## 交叉验证

`cross_validation(m, horizon ,initial,units = 'days')`

> 参数horizon, initial, units：初始日期，截止日期，间隔



