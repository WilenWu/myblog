---
ID: 5314cb2b58df7fcc17603b67989dc8ab
title: R手册(Import)--for csv, xlsx and json
tags: [R,tidyverse]
date: 2018-05-01 17:40:30
categories: [R,Import]
cover: /img/readr.png
top_img: 
---

R语言导入常用文件的包

- readr: 可读取文本型文件
- readxl: 读取excel文件
- openxlsx: 可读取、写入和编辑excel文件
- jsonlite: 读取json文件

<!-- more -->

# readr: Read Rectangular Text Data

## Save Data

保存类型|函数
:------|:------
逗号分隔文件<br>*base::write.csv()*|write_csv(x, path, na = "NA", append = FALSE,col_names = !append)
任意符号分隔|write_delim(x, path, delim = " ", na = "NA",append = FALSE, col_names = !append)
CSV for excel|write_excel_csv(x, path, na = "NA", append =FALSE, col_names = !append)
String to file|write_file(x, path, append = FALSE)
String vector to file, one element per line|write_lines(x,path, na = "NA", append = FALSE)
RDS文件|write_rds(x, path, compress = c("none", "gz","bz2","xz"), ...)
Tab分隔|write_tsv(x, path, na = "NA", append = FALSE,col_names = !append)

## Read Tabular Data

```r
read_*(file, col_names = TRUE, col_types = NULL, locale = default_locale(), na = c("", "NA"),
  quoted_na = TRUE, comment = "", trim_ws = TRUE, skip = 0, n_max = Inf, guess_max = min(1000,n_max), 
  progress = interactive())
# read_* 函数共享这些参数
```

规整类型|函数（示例）
:------|:------
Comma Delimited Files(逗号分隔)|read_csv("file.csv")
Semi-colon Delimited Files(分号分隔)|read_csv2("file2.csv")
Files with Any Delimiter(任意分隔符)|read_delim("file.txt", delim = "\|")
Fixed Width Files(固定宽度)|read_fwf("file.fwf", col_positions = c(1, 3, 5))
Tab Delimited Files(Tab分隔符`\t`)|read_tsv("file.tsv") Also read_table().

> **常用参数**
> col_names: logic or string vector, 列名
> skip: int, 跳过第n行
> na: 指定用于表示文件中缺失值的值
> comment = "#": 删除以#开头的所有行
> progress = show_progress(): 进度条：("none"/"text"/"tk"/"win")  

```r
write_file("a,b,c\n1,2,3\n4,5,NA","file.csv")
f <- "file.csv"           #Example file

read_csv(f, col_names = FALSE)     #No header
read_csv(f, col_names = c("x", "y", "z"))   #Provide header
read_csv(f, skip = 1)      #Skip lines
read_csv(f, n_max = 1)     #Read in a subset
read_csv(f, na = c("1", "."))  #Missing Values
```

## Read Non-Tabular Data

非规整类型|函数
:------|:------
read_file(file, locale = default_locale())|读入单独的字符串中
read_lines(file, skip = 0, n_max = -1L, na = character(),<br>locale = default_locale(), progress = interactive())|每行读入一个字符串
read_file_raw(file)|读入原始向量中
read_lines_raw(file, skip = 0, n_max = -1L,progress = interactive())|每行读入一个原始向量
read_log(file, col_names = FALSE, col_types = NULL, <br>skip = 0, n_max = -1, progress = interactive())|读取Apache日志文件

## Data types

`readr` 函数猜测每一列适合的数据类型并且转化，但不会将字符串类型转化为factor
col_types参数对列数据类型的解析（示例）：

```r
col_types=cols(
  age = col_integer(),
  sex = col_character(),
  earn = col_double()
)
```

1. 用`problems()`函数去判断
`x <- read_csv("file.csv"); problems(x)`

2. 可以`col_ function`函数引导解析，也可以字符格式读入，然后用`parse_ function` 函数解析
col_ function|parse_ function
------|------
col_guess() *the default* |parse_guess()
col_character()|parse_character()
col_double(), col_euro_double()|parse_double()
col_datetime(format = "")<br> col_date(format = ""), col_time(format = "")|parse_datetime()<br>parse_date() and parse_time()
col_factor(levels, ordered = FALSE)|parse_factor()
col_integer()|parse_integer()
col_logical()|parse_logical()
col_number(), col_numeric()|parse_number()
col_skip()|不解析
```r
x <- read_csv("file.csv", col_types = cols(
  A = col_double(),
  B = col_logical(),
  C = col_factor()))

x$A <- parse_number(x$A)
```

# readxl: for excel

```r
# 读取excel文件
read_excel(path, sheet = NULL, range = NULL, col_names = TRUE,
  col_types = NULL, na = "", trim_ws = TRUE, skip = 0, n_max = Inf,
  guess_max = min(1000, n_max))
read_xls()
read_xlsx()

# 列出excel文件中的所有sheet
excel_sheets(path) 
```

**读取单元格的指定范围**：以`A1：D10`样式

`cellranger::cell_limits()`
`cellranger::cell_rows()`
`cellranger::cell_cols()`
`cellranger::anchored()`

```r
read_excel(path, range = "B3:D6")
read_excel(path, range = cell_rows(3:6))
read_excel(path, range = cell_cols("C:D"))
read_excel(path, range = cell_cols(2))
read_excel(path, range = cell_limits(c(4, 3), c(NA, NA)))
```

# openxlsx: xlsx reading, writing and editing

openxlsx包括读写及编辑excel文件，集成极为丰富的函数。

```r
read.xlsx(xlsxFile, sheet = 1, startRow = 1, colNames = TRUE,
  rowNames = FALSE, detectDates = FALSE, skipEmptyRows = TRUE,
  skipEmptyCols = TRUE, rows = NULL, cols = NULL, check.names = FALSE,
  namedRegion = NULL, na.strings = "NA", fillMergedCells = FALSE)`

write.xlsx(x, file, asTable = FALSE, ...)
```

# jsonlite: for json

1. 读入/写出JSON
`read_json(path, simplifyVector = FALSE, ...)`
`write_json(x, path, ...)`
2. JSON和R对象之间转换
`fromJSON(txt)`
`toJSON(x)`



