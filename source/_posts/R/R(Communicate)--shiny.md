---
ID: 0fe5dfb16482659f111fe9e47fbb1386
title: R手册(Communicate)--shiny
tags: [R,可视化]
date: 2018-05-23 23:44:24
categories: [R,Communicate]
---
Shiny是来自RStudio的软件包，它使得用R构建交互式Web应用程序变得非常简单。

<!-- more -->

# Build an App

A **Shiny app** is a web page (**UI**) connected to a
computer running a live R session (**Server**)

```R
library(shiny)
ui <- fluidPage(
 numericInput(inputId = "n",  
 "Sample size", value = 25),  
 plotOutput(outputId = "hist") 
 )
server <- function(input, output) {
 output$hist <- renderPlot({   
 hist(rnorm(input$n)) })    
 }
shinyApp(ui = ui, server = server)
```

1. **Define UI ----(支持HTML标签语法)**
-  `ui <- fluidPage()`定义HTML用户界面
- 添加input到ui ，用`<type>Input`系列函数
- 添加output到ui，用`<type>Output`系列函数

2. **Define server logic**
- `server <- function(input, output){}`定义R对象和UI组件的连接方式
- 用`input$<inputId>`或`output$<outputId> `提供数据给UI组件
- `render<type>`赋值给`output$<id>`输出给UI output 组件

3. **Run the app**
- combines ui and server
- Run `shinyApp(ui = ui, server = server)`

4. **Save**
保存成一个**app.R** 或者拆分成 **ui.R** 和**serve.R**（shinyApp不需要保存）

5. **Share your APP**
- 注册免费或者专业账户 http://shinyapps.io
- 在RStudio IDE点击**Publish** 按钮或者运行代码：
`rsconnect::deployApp("<path to directory>")`

# Inputs

使用`input$<inputId>`获取input当前value

input|说明
:------|:------
actionButton(inputId, label, icon,…)|按钮
submitButton(text, icon)|提交按钮
radioButtons(inputId, label,choices, selected, inline)|单选按钮
checkboxGroupInput(inputId, label,choices, selected, inline) |复选框组合
checkboxInput(inputId, label,value) |单个复选框
dateInput(inputId, label, value, min,max, format, <br>startview, weekstart,language)|日期单选
dateRangeInput(inputId, label,start, end, min, max, format,<br>startview, weekstart, language,separator)|日期复选
numericInput(inputId, label, value,min, max, step)|数字输入文本框
passwordInput(inputId, label,value)|密码输入文本框
textInput(inputId, label, value)|文本输入框
selectInput(inputId, label, choices,selected, multiple, <br>selectize, width,size)| (also selectizeInput())下拉列表
sliderInput(inputId, label, min, max,value, step, <br>round, format, locale,ticks, animate, width, sep, pre,post)|滚动条
actionLink(inputId, label, icon, …)|链接
fileInput(inputId, label, multiple,accept)|文件选择

# Outputs

` render<type>()` and `<type>Output()` functions work together to add R output to the UI

` render<type>()`| `<type>Output()`|说明
:------|:------|:------
DT::renderDataTable(expr, options,callback, escape, env, quoted)|dataTableOutput(outputId, icon, …)|dataTable
renderImage(expr, env, quoted,deleteFile)|imageOutput(outputId, width, height,<br>click, dblclick, hover, hoverDelay, inline,<br>hoverDelayType, brush, clickId, hoverId)|图片
renderPlot(expr, width, height, res, …,env, quoted, func)|plotOutput(outputId, width, height, click,<br>dblclick, hover, hoverDelay, inline,<br>hoverDelayType, brush, clickId, hoverId)|作图
renderPrint(expr, env, quoted, func,width)|verbatimTextOutput(outputId)|打印
renderTable(expr,…, env, quoted, func)|tableOutput(outputId)|表格输出
renderText(expr, env, quoted, func)|textOutput(outputId, container, inline)|文本标签输出
renderUI(expr, env, quoted, func)|uiOutput(outputId, inline, container, …)<br>htmlOutput(outputId, inline, container, …)|

# Layouts(布局)

合并多个元素为单一元素

panel|说明
:------|:------
absolutePanel()|
conditionalPanel()|
fixedPanel()|
headerPanel()|
inputPanel()|输入面板
mainPanel()|主面板
navlistPanel()|
sidebarPanel()|工具栏面板
tabPanel()|
tabsetPanel()|
titlePanel()|标题面板
wellPanel()|

```r
wellPanel(
  dateInput("a", ""),
  submitButton()
)
```

**UI布局**
```r
# fluidRow()
ui <- fluidPage(
 fluidRow(column(width = 4),column(width = 2, offset = 3)),
 fluidRow(column(width = 12))
)
```
![](https://gitee.com/WilenWu/images/raw/master/r-shiny/shiny_fluid.png)

```r
# flowLayout
ui <- fluidPage(
 flowLayout( # object 1,
 # object 2, # object 3 )
)
```

![](https://gitee.com/WilenWu/images/raw/master/r-shiny/shiny_flow.png)

```r
# sidebarLayout
ui <- fluidPage(
 sidebarLayout(
 sidebarPanel(),  # 侧边栏
 mainPanel()      # 主面板
 )
)
```
![](https://gitee.com/WilenWu/images/raw/master/r-shiny/shiny_panel.png)
```r
# splitLayout
ui <- fluidPage(
 splitLayout( # object 1,
 # object 2
 )
)
```
![](https://gitee.com/WilenWu/images/raw/master/r-shiny/shiny_split.png)
```r
# verticalLayout
ui <- fluidPage(
 verticalLayout( # object 1,
 # object 2,
 # object 3
 )
)
```
![](https://gitee.com/WilenWu/images/raw/master/r-shiny/shiny_vertical.png)
```r
# tabPanel
ui <- fluidPage( tabsetPanel(
 tabPanel("tab 1", "contents"),
 tabPanel("tab 2", "contents"),
 tabPanel("tab 3", "contents")))
ui <- fluidPage( navlistPanel(
 tabPanel("tab 1", "contents"),
 tabPanel("tab 2", "contents"),
 tabPanel("tab 3", "contents")))
ui <- navbarPage(title = "Page",
 tabPanel("tab 1", "contents"),
 tabPanel("tab 2", "contents"),
 tabPanel("tab 3", "contents"))
```
![](https://gitee.com/WilenWu/images/raw/master/r-shiny/shiny_tabs.png)


# Reactivity(反应)

Reactivity|说明
:------|:------
`input$<inputId>` functions<br>reactiveValues(…)|input接收的值被储存在`input$<inputId>`中
`render<type>()` functions<br>`output$<outputId>`|将结果保存到`output$<outputId>`
isolate(expr)|抑制反应
observeEvent(eventExpr, handlerExpr,...)|触发主观代码
reactive(x, env, quoted,label, domain)|模块化反应机制
eventReactive(eventExpr,valueExpr,...)|延迟反应

