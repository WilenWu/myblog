---
ID: bc07efdb6832af4578bc9968bb8c5787
title: R手册(Common)--R语言基础包
tags: [R]
date: 2018-04-30 17:22:51
categories: [R,Common]
cover: /img/r-report.jpg
top_img: 
---

R语言基础包：`base, stats`

<!-- more -->

# 环境设置

**系统函数**

函数|说明
:-----|:-----
options(...)|显示或设置当前选项(digits,encoding,...)
memory.limit(size=NA)|内存信息
source()/sink()|加载/输出R脚本
quit()/q()|退出R

**packages操作函数**

函数|说明
:-----|:-----
install.package(“pkg”)|安装包
devtools::install_github(“pkg”)|安装github开发版包
library(pkg, warn.conflicts = TRUE)|加载包
detach("package:pkg")|移除加载的包
(.packages())|展示加载的包
`installr::updateR()`|更新R（复制包）<br>或者将老版本的R包复制到新版本后，`update.packages()`
help(package=”pkg”)|展示包的信息
update.packages()|更新包
remove.packages(pkgs, lib)|卸载包
if(!require("installr"))<br>install.packages('installr') <br>library("installr")|安装并加载

# 输入输出

函数|说明
:-----|:-----
edit(x),fix(x),pico(x)|输入
print(...),cat(...)|输出
save(objects,file=”.RData”)|保存
load(file)|
format(x, digits)|有效数字
format(x, format = "")|日期格式
iconv(x，encoding)|字符串编码转换

# 文件操作

函数|说明
:-----|:-----
system.file()|获取系统文件路径
getwd(),setwd()|获得/设置工作路径
dir.create()|创建新目录
list.files(path,pattern=NULL,<br>  all.files = FALSE,<br>  full.names = FALSE)|path列出文件名,pattern筛选文件<br>all.files是否子文件夹下的全部文件<br>full.names是否输出文件名包含路径全称
choose.dir(default = "",caption)|交互式选择文件夹
choose.files(default = "",caption,multi = TRUE, filters = Filters)|交互式选择文件，multi:复选,filters:文件过滤向量
menu(choices, graphics = FALSE, title = NULL)|菜单交互函数
winDialog(type ="yesno",message) |type in c("ok", "okcancel", "yesno", "yesnocancel")

# 进度条

函数|说明
:-----|:-----
txtProgressBar(min=0,max=1,initial=0,char ="=",width=NA,style=1)|初始化文本进度条<br>参数: 进度最小值,最大值,初始值,文本,宽度,格式(1,2:支持换行,3:进度值)
setTxtProgressBar(pb, value)|加载进度条,value进度值
winProgressBar(title,label,min, max, initial, width)|win进度条
setWinProgressBar(pb, value, title = NULL, label = NULL)|
close(con,...)|关闭进度条

# 数据创建

函数|说明
:-----|:-----
c(…); list(…)|向量
matrix(data, nrow, ncol, byrow = FALSE,dimnames=list(rownames,colnames))|矩阵
array(data, dim, dimnames)|数组
data.frame(col1,col2,col3,..., row.names)|数据框
factor(x, levels, labels, ordered) |`sex<-factor(sex,levels=c(1,2),labels=c("Male","Female"))`
seq(from = 1, to = 1, by)|等差数列
rep(x,…)<br>`rep(c("character","integer","numberic"),c(4,1,2))`|times每个元素重复的次数<br>length.out期望长度,each

# 数据选取及数据信息

函数|说明
:-----|:-----
str(),summary()|概要
x[-(1:n)]<br>x['rows','cols']|索引和切片
x$name<br>x[x>3]<br> x[x %in% c(...)]|索引和切片
head(x,n),tail(x,n)|
dim(x), dimnames(x), names(x)|信息
nrow(),ncol(x), length(x)|行数，列数，长度
which.max(), which.min(), which()|返回索引
cut(x,breaks,labels = NULL,right,ordered)|返回切割而后的factor<br>breaks切割点向量或切割间隔数,right是否右开区间,ordered是否有序

# 列联表

函数|说明
:-----|:-----
table(var1, var2,...,useNA="no")|useNA in c("no", "ifany", "always")
prop.table(x, margin)|小数比例的形式
xtabs(formula=~., data)|formula:var~var+var+...<br>左边为要透视的值(默认sum函数,左边缺失时计数)，右边为列联表的维度
ftable(table) |将多维table对象转为二维table(data.frame)
 xtabs(mpg+cyl~am+gear,mtcars)%>%ftable()|

# 内置常量

函数|说明
:-----|:-----
NA,NULL|缺失值
Inf/-Inf|无穷
pi|
as.roman(x)|罗马数字
month.abb, month.name|月份
letters, LETTERS|大小写字母

# 数学

函数|说明
:-----|:-----
`ˆ, %%, %/%` |求幂(`**`)，求余数(mod)，整除
sign()|提取正负(1,0,-1)
log(x, base),exp(x)|
round(x, digits = 0)|四舍五入
signif(x, digits = 6)|有效数字
trunc(x)|取整
ceiling(x)|天花板
floor(x)|地板
max,min,median,mean(x,trim=0,na.rm=FALSE)|
range,IQR|极差,四分位距
quantile(x, probs = seq(0, 1, 0.25))|百分位数
abs,sqrt|
sum,prod|
rank(x, na.last = TRUE,ties.method = c("average", "first", "last", "random", "max", "min"))|na.last：NA排名处置方式
diff|差分
cum_sum|返回向量x[i]=sum{x[1]:x[i]}
cum_prod|
scale(x, center = TRUE, scale = TRUE)|
combn(x, m, FUN = NULL)|x中m个元素的所有(统计)组合

> Matrix包中的函数使得R可以处理高密度矩阵或稀疏矩阵。可以高效的访问BLAS（Basic Linear Algebra Subroutines）、Lapack（密集矩阵） 、TAUCS（稀疏矩阵）和UMFPACK（稀疏 矩阵） 。 

# 矩阵运算

运算符或函数|描 述
:---|:---
`+ - * / ^`|分别是逐个元素的加、减、乘、除和幂运算 
A %*% B |矩阵乘法 
A %o% B |外积：AB
cbind(A, B, ...) |横向合并矩阵或向量 
chol(A) |A的Choleski分解。若`R <- chol(A)`，那么chol(A)包含上三角因子，即R'R=A 
colMeans(A) |返回A的列均值组成的向量 
crossprod(A) |返回A'A 
crossprod(A, B) |返回A'B 
colSums(A) |返回A的列总和组成的向量 
diag(A) |返回主对角元素组成的向量 
diag(x) |用x中元素作为主对角元素创建对角矩阵 
diag(k) |如果k是标量，就创建k × k的单位矩阵 
eigen(A) |A的特征值和特征向量。若y <- eigen(A)，那么：<br> `y$val`是A的特征值<br>`y$vec`是A的特征向量 
ginv(A) |A的Moore-Penrose广义逆（需要MASS包） 
qr(A) |A的QR分解。若 `y <- qr(A)`，那么： `y$qr`的上三角是分解结果，下三角是分解的信息 <br> `y$rank`是A的秩 <br>`y$qraux`是Q的附加信息向量 <br>`y$pivot`是所使用的主元素选择策略
 rbind(A, B, ...) |纵向合并矩阵或向量 
 rowMeans(A) |返回A的行均值组成的向量 
 rowSums(A) |返回A的行总和组成的向量 
solve(A, b) |求解方程b = Ax中的向量x 
svd(A) |A的奇异值分解。若 `y <- svd(A)`，那么： <br>`y$d`是A的奇异值组成的向量 <br>`y$u`是矩阵且每一列都是A的左奇异向量 <br> `y$v`是矩阵且每一列都是A的右奇异向量 
t(A) |A的转置

# 模型

函数|说明
:-----|:-----
var,sd,cov|
cor(x, y = NULL, use = "everything",method = "pearson")|data%>%cor%>%symnum(相关系数矩阵可视化)<br>method in c('person','spearman','kendall')
cor.test|相关性检验
anova|单因素方差分析
ancova|单因素协方差分析
lm/glm|线性回归
aov|方差分析模型

# 其他函数

函数|说明
:-----|:-----
Sys.sleep(seconds)|
system.time(expr)|
expression(expr)|创建表达式
eval(expr)|在指定的环境下运行表达式



