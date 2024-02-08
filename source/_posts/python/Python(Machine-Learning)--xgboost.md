---
title: Python手册(Machine Learning)--XGBoost
tags:
  - Python
  - 机器学习
categories:
  - Python
  - Machine Learning
cover: /img/XGBoost-cover.svg
top_img: /img/XGBoost-cover.svg
abbrlink: c46d5dae
date: 2024-01-25 22:15:00
description:
---



# Overview

eta = 0.1
max_depth = 8
num_round = 500
nthread = 16
tree_method = exact
min_child_weight = 100

# Scikit-Learn API

```python
from xgboost import XGBClassifier
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
# create model instance
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
```

# 增量学习

# XGBoost with PySpark

从1.7.0版本开始，xgboost封装了pyspark API，因此不需要纠结spark版本对应的jar包 xgboost4j 和 xgboost4j-spark 的下载问题了，也不需要下载调度jar包 sparkxgb.zip。

distributed_learning_demo


```python
import xgboost as xgb	
from sklearn.metrics import roc_curve, recall_score,accuracy_score,roc_auc_score from sklearn.model_selection import train_test_split import matplotlib.pyplot as plt import seaborn as sns import pandas as pd import re
import numpy as np 

pd.set_option('display.max_columns',None) pd.set_option('display.max_rows',None) pd.set_option('maxcolwidth',2000)
pd.set_option('display.float_format', lambda x: '%.5f' %x) 
data=pd.read_csv(./data cleaned.csv') 
with open('./columns.txt',mode='r') as f: cols=f.read() cols=cols.split(",') print(cols) ien(cols)
data=data.loc[:.cols]
x_train,X_test, y_train, y_test=train_test_split(data.drop('label',axis=1),data['label'],test size=0.4,random_state=123)
x_test,X_valid, y_test, y_valid=train_test_split(X_test, y_test,test_size=0.5,random_state=123) dtrain=xgb.DMatrix(X_train,label=y_train) dvalid=xgb.DMatrix(X_valid,label=y_valid) dtest=xgb.DMatrix(X_test,label=y_test) print(y_train.value_counts()) print(y_test.value_counts())
params={
'booster':'gbtree',
'objective':'binary:logistic','device':'cpu','eval_metric':'auc','nthread':4,
'learning_rate':0.1,'max_depth':6,'subsample':0.5,
'colsample_bytree':0.7,'reg_alpha':0.001,'reg_lambda':0.001 evals_result={}
clf=xgb.train(params,dtrain,num_boost_round=500,early_stopping_rounds=5,evals[(dvalid,'eval')],evals_result=evals_result,verbose_eval=10) imp=pd.Series(clf.get_fscore()) imp.sort_values(ascending=False) plt.figure()
xgb.plot_importance(clf,max_num_features=20) plt.show()
imp=pd.DataFrame(imp,columns=['score'])
imp.sort_values('score’,ascending=True,inplace=True) imp['pct']=imp['score']/imp['score'].sum() imp['pct_cumsum']=imp['pct'].cumsum()
imp_top=imp.query(pct_cumsum>=0.1').sort_values('score',ascending=False) print(imp_top)
# ##模型评估
y_prob_train=clf.predict(dtrain) y_prob_test=clf.predict(dtest)
fpr,tpr,thresholds=roc_curve(y_test,y_prob_test) auc=roc_auc_score(y_test,y_prob_test) idx=(tpr-fpr).argmax() ks=(tpr-fpr).max()
best_threshold=thresholds[idx] plt.figure(figsize=(6,6))
plt.plot(fpr, tpr,label=f'auc=') 
plt.plot((0,1), (0,1).'--')
plt.plot((fpr[idx],fpr[idxJ), (fpr[idx],tpr[idx]),'--', label=f'score=\nks=') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
pit.title(ROC curve') plt.legend() plt.show()
y_pred_train=y_prob_train.copy0 y_pred_test=y_prob_test.copy0
y_pred_train[y_prob_train>=best_threshold]=1 y_pred_train[y_prob_train<best_threshold]=0 y_pred_test[y_prob_test>=best_threshold]=1 y_pred_test[y_prob_test<best_threshold]=0 print(f"train auc:, test auc:") print(f"train acc:, test acc:") print(f"train recall:, test recall:")#迭代曲线
plt.figure(figsize=(6,6))
plt.plot(evals_result['eval']['auc']) plt.xlabel('Iterations') plt.ylabel('auc')
plt.title('Metric during training') plt.show()
##正负样本得分分布
ro = y_prob_train[y_train ==0] r1= y_prob_train[y_train==1]
fig = plt.figure(facecolor ='white',figsize =(20,5)) ax1 =fig.add_subplot(1,2,1)
sns.distplot(r0,kde_kws = {"label":"y"},color = 'r',ax=ax1)
sns.distplot(r1,kde_kws = {"label":"pred"},color = 'g',ax=ax1) I ax1.set_title(label ='Frequency',loc ='center') ax2=fig.add_subplot(1,2,2)
sns.kdeplot(r0,cumulative=True,label ="y",color ='r',ax=ax2) sns.kdeplot(r1,cumulative=True,label ="pred",color='g',ax=ax2) ax2.set_title(label='cumulative',loc ='center')
###模型稳定性
fig=plt.figure(figsize=(15,5)) ax1=fig.add_subplot(121)
sns.kdeplot(y_prob_train,ax=ax1) sns.kdeplot(y_prob_test,ax=ax1) ax1.set_xscale('log')
ax2=fig.add_subplot(122)
sns.kdeplot(y_prob_train,ax=ax2) sns.kdeplot(y_prob_test,ax=ax2) 
plt.show0
def get_psi(act,pre,n=10): act=pd.Series(act) pre=pd.Series(pre)
step=(act.max0-act.min0)/n
cuts=[-np.inf]+[act.min()+i*step for i in range(1,n)]+[np.inf) act_bins=pd.cut(act,cuts).value_counts0 pre_bins=pd.cut(pre,cuts).value_counts()
psi=pd.DataFrame({'act':act_bins,'pre':pre_bins ) psi[['act_pct','pre_act']]=psi/psi.sum0)
psi['psi']=(psi['act_pct']-psi['pre_act'])*np.log(psi['act_pct']/psi['pre_act']) return psi,psi['psi'].sum()
psi_df,psi=get_psi(y_prob_train,y_prob_test) print(psi_df,psi,sep='\n')
psi_df.plot(y=['act_pct', pre_act'],kind='bar')# clf.save_model('xgb.json')# clf.load_model('xgb.json')###观察重点特征的分布 for col in imp_top.index:
df=pd.DataFrame({'x':X_train[col],'label':y_train}) df[col] = pd.qcut(df['x'],6,duplicates='drop')
df_pivot=df.pivot_table(index=col,columns='label',values='label',aggfunc='count'), df_pivot.plot(kind='barh')
#-	----xgb增量	
#!/usr/bin/env python# coding: utf-8
import xgboost as xgb
from sklearn.metrics import roc_curve,recall_score,accuracy_score,roc_auc_score from sklearn.model_selection import train_test_split import matplotlib.pyplot as plt import pandas as pd import re
import numpy as np
reader=pd.read_csv("./data_cleaned.csv',chunksize=500000) data=reader.get_chunk(100000)
x_test,y_test =data.drop("label',axis=1),data['label'] dtest=xgb.DMatrix(X_test,label=y_test) params={
'booster':'gbtree',
'objective':'binary:logistic,'device':'cpu',
'eval_metric':'auc',
'nthread':4,
'learning_rate':0.1,'max depth':6,'subsample':0.5,
'colsample bytree':0.7,'reg_alpha':0.001,'reg_lambda':0.001 evals_result={} clf=None
for i,chunk in enumerate(reader):
x_train,y_train= chunk.drop('label',axis=1),chunk['label'] dtrain=xgb.DMatrix(X_train,label=y_train)
print('-+-'*15+f'chunk {i+1}, size = {len(y_train)}'+'-+-'*15) clf=xgb.train(params,dtrain,
num_boost round=10,
#	early_stopping_rounds=5,	
evals=[(dtest,'test')], xgb_model=clf,
evals_result=evals_result, verbose_eval=10)
if i==0:
params['process_type']='update' # Set `process_type`to`default` if you wan params['updater']='refresh'
params['refresh_leaf]=True#True:更新叶节点，False:更新树的权重 clf.best_ntree_limit n=0
for i in clf.get_dump():	I	
n+=1 print(n)
imp=pd.Series(clf.get_fscore()) imp.sort_values(ascending=False) clf=None
params['process_type']='default' for i,chunk in enumerate(reader):
X_train,y_train=chunk.drop('label',axis=1),chunk['label'] dtrain=xgb.DMatrix(X_train,label=y_train)
print('-+-'*15+f'chunk {i+1}, size = (len(y_train)}'+'-+-*15) clf=xgb.train(params,dtrain,
num_boost_round=2, early_stopping_rounds=5, evals=[(dtest,'test')], 
xgb model=clf,
evals result=evals result, verbose_eval=10)
clf.best_ntree limit n=0
for i in clf.get_dump0: n+=1 print(n)
imp=pd.Series(clf.get_fscore()) imp.sort_values(ascending=False)

# ------------------xgb on spark#!/usr/bin/env python# coding: utf-8
from pyspark.conf import SparkConf from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import DoubleType, FloatType,IntegerType, Stri import pyspark.ml.feature as ft import pyspark.sql.functions as F from pyspark.sql import Row
from sklearn.metrics import roc_curve, recall_score,accuracy_score,roc from sklearn.model_selection import train_test_split import matplotlib.pyplot as plt import xgboost as xgb import sparkxgb import pandas as pd import numpy as np import re import sys
# 启动spark会话
spark = SparkSession\.builder\
.config("spark.driver.memory","25g")\
.config("spark.driver.memoryOverhead","25g")\.config("spark.driver.cores","5")\
.config("spark.executor.memory","22g")\
.config("spark.executor.memoryOverhead"."22g")\.config("spark.executor.cores","5")\
.config("spark.driver.maxResultSize","30g")\
.config("spark.dynamicAllocation.enabled","false")\.appName("data")\
enableHiveSupport().getOrCreate(0
sc = spark.sparkContext
start_date, end_date, end_date2='20231101,20231101,20231102' with open('./get_data.sql",mode='r') as f: sql_expr=f.read()
sql_expr = sql_expr.format(start_date = start_date, end_date =end_date, print(sql_expr)
df=spark.sql(sql_expr) df.columns
data=df.sampleBy('label',fractions={0: 0.002, 1: 1.0},seed=123).drop('BECIF data.persist()
data.groupBy('label').count0.show0 data.dtypes
str_cols=[i for i,j in data.dtypes if j=='string'] str_cols
for col in str_cols:
if col not in ['MOB','M_LEVEL','MAX AMT_CHANNEL','MAX_AMT_DATE', data=data.withColumn(col,data[col].cast('float'))
data.groupby('MOB').count().show( data=data.withColumn('MOB',F.regexp_replace('MOB','\\*',")) I
data=data.withColumn('MOB',F.when(data['MOB']==",'10').otherwise(data[ data=data.withColumn('MOB',data['MOB'].cast('float')) data.groupby('MOB').count().show() def isna(data,thresh=-1): na_dict={} n=data.count()
for col in data.columns:
pct=data.filter(F.isnull(col)).count()/n if pct>thresh:
na_dict[col]=pct return na_dict
dropna_dict=isna(data,0.1) dropna_dict
cols=[i for i in data.columns if i not in dropna_dict.keys0] data_clean=data.select(cols).na.drop(thresh=int(len(cols)*0.9)) data_clean.persist() data_clean.columns
data_clean.write.saveAsTable
('tbl_kk.CCM_INST_MDL_XJ_CUST_SAMPLE',mode='overwrite',partitionBy='DT)

cats=[i for i jin data_clean.dtypes if j=='string']
cols =[col for col in data_clean.columns if not col.endswith('_FLAG') and col not in ('DT','label')] fill_values=data_clean.select(*[F.mean(col).alias(col) for col in cols]) fill_values=fill_values.toPandas().astype('double').T.to_dict()[0] fillvalues
result=data_clean.fillna(fill_values).fillna(0) isna(result)
train_df,test_df=result.randomSplit([0.7,0.3],seed=123)
train_df=ft.Stringlndexer(inputCol='label',outputCol='target).fit(train_df).transform(train_df) featuresCreator=ft.VectorAssembler( inputCols=result.columns[:-2], outputCol='features'
train_df= featuresCreator.transform(train df)# xgb_classifier=sparkxgb.XGBoostClassifier(
#	featuresCol='features',	
#	labelCol='label',	
#	subsample=0.5,	
#	.colsampleBytree=0.7,	
evalMetric='auc', maxDepth=6, nthread=1,
#	numEarlyStoppingRounds=5,	
#	numRound=200,	
numWorkers=4,
objective ='binary:logistic', eta=0.3,
alpha=0.001,
#	gamma=0.001	
# model=xgbclassifier.fit(train df)
from pyspark.ml.classification import GBTClassifier import pyspark.ml.evaluation as ev gbdt=GBTClassifier(
featuresCol='features', labelCol='target', maxDepth=6, maxlter =200, stepSize=0.01,
subsamplingRate=0.7,
featureSubsetStrategy='sqrt' model=gbdt.fit(train_df)
test_df=featuresCreator.transform(test_df)
evaluator=ev.BinaryClassificationEvaluator()
recall=evaluator.setMetricName('areaUnderPR').evaluate(pred df) auc=evaluator.setMetricName('areaUnderROC').evaluate(pred_df) print(fauc:, recall: ')
pred_df.select( 'label', probability','prediction').show() def extractProbability(row):
_, prob=row['probability'].toArray0)
return Row(label=row['label'], pred=row['prediction'], prob=float(prob))
predictions=pred_df.select( 'label"','probability',' prediction').rdd.map(extractProbability).toDF predictions.show0
imp=pd.DataFrame(model.featurelmportances.toArray0,index=featuresCreato
```

# XGBoost with Dask



