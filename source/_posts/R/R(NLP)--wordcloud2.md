---
title: R(NLP)--wordcloud2
tags:
  - R
  - NLP
  - 可视化
categories:
  - R
  - NLP
cover: /img/love-cloud.png
top_img: /img/r-top-img.svg
description: R interface to wordcloud for data visualization
abbrlink: 28aaa9da
date: 2018-05-28 18:41:43
---

**wordcloud2** :R interface to wordcloud for data visualization

Wordcloud2主要包括两个函数：
1. `wordcloud2`：用HTML5提供传统的wordcloud
2. `letterCloud`：用选定的单词（字母）提供wordcloud

<!-- more -->

# wordcloud2函数

```r
wordcloud2(data, size = 1, minSize = 0, gridSize =  0,
    fontFamily = 'Segoe UI', fontWeight = 'bold',
    color = 'random-dark', backgroundColor = "white",
    minRotation = -pi/4, maxRotation = pi/4, shuffle = TRUE,
    rotateRatio = 0.4, shape = 'circle', ellipticity = 0.65,
    widgetsize = NULL, figPath = NULL, hoverFunction = NULL)
```

**参数**

- data：包含每列中的word和freq的数据帧，按照word出现的顺序由内向外画图（可以按照freq降序美化wordcloud）。
- size：字体大小，默认为1。较大的大小意味着较大的单词。
- fontFamily：要使用的字体。
- fontWeight：字体重量，例如`normal, bold or 600`
- color：文本的颜色，可以使用关键字`random-dark`和`random-light`。也支持颜色矢量。
- minSize：字幕的字符串
- backgroundColor：背景的颜色。
- gridSize：用于标记画布可用性的网格大小，网格大小越大，单词之间的差距越大。
- minRotation：文本应该旋转的最小旋转（以rad为单位）。
- maxRotation：文本应旋转的最大旋转（以rad为单位）。
- rotateRatio：单词旋转的概率。将数字设置为1以始终旋转。
- shape：绘制“云”的形状。 ‘circle’ (default), ‘cardioid’ (心形'，苹果或心形曲线，最知名的极坐标方程), ‘diamond’ (菱形), ‘triangle-forward’(三角形前移), ‘triangle’(三角形), ‘pentagon’(五角形), and ‘star’。
- ellipticity：平坦度
- figPath：画布路径
- widgetsize：小部件的大小

**示例**

```r
head(demoFreq)
##          word freq
## oil       oil   85
## said     said   73
## prices prices   48
## opec     opec   42
## mln       mln   31
## the       the   26
wordcloud2(demoFreq, color = "random-light", backgroundColor = "grey") #颜色和背景
wordcloud2(demoFreq, size = 1,shape = 'star') #形状
wordcloud2(demoFreq, size = 2, minRotation = -pi/6, maxRotation = -pi/6,rotateRatio = 1) #旋转
wordcloud2(demoFreqC, size = 2, fontFamily = "微软雅黑",
           color = "random-light", backgroundColor = "grey") #支持中文
#自定义画布
figPath = system.file("examples/t.png",package = "wordcloud2")
wordcloud2(demoFreq, figPath = figPath, size = 1.5,color = "skyblue") 
```

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/wordcloud/Wordcloud1.png"  style="zoom:50%;" /><img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/wordcloud/Wordcloud2.png" style="zoom:80%;" /><img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/wordcloud/Wordcloud3.png" style="zoom:50%;" />
<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/wordcloud/Wordcloud4.png"  style="zoom:50%;" /><img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/wordcloud/Wordcloud5.png" style="zoom: 67%;" />

# letterCloud函数

```r
letterCloud(data, word, wordSize = 0, letterFont = NULL, ...)
```
**参数**

- data：包含每列中的word和freq的数据帧
- word：一个单词，为wordcloud创造形状。
- wordSize：单词大小的参数，默认为2。
- letterFont：字母的字体
- ...

```r
letterCloud(demoFreq, word = "WORDCLOUD2", wordSize = 1)
```

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/wordcloud/Wordcloud6.png" style="zoom:150%;" />


# shiny支持

See Example:
```r
if(require(shiny)){
  library(wordcloud2)
   # Global variables can go here
   n <- 1

   # Define the UI
   ui <- bootstrapPage(
      numericInput('size', 'Size of wordcloud', n),
     wordcloud2Output('wordcloud2')
   )


   # Define the server code
   server <- function(input, output) {
      output$wordcloud2 <- renderWordcloud2({
      # wordcloud2(demoFreqC, size=input$size)
        wordcloud2(demoFreq, size=input$size)
      })
   }
   # Return a Shiny app object
   # Sys.setlocale("LC_CTYPE","chs") #if you use Chinese character
   ## Do not Run!
   shinyApp(ui = ui, server = server)
   }
```



