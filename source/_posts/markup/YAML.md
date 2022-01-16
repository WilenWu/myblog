---
title:  YAML 基础语法
date: 2021-12-03 23:12:27
updated:
tags: [标记语言, YAML]
categories: [标记语言]
sticky: 2
cover: /img/yaml-tutorial.png
top_img: /img/yaml-cover.jpg
description:
---

**YAML** （发音 /ˈjæməl/ ）是 **Y**AML **A**in't a **M**arkup **L**anguage（YAML不是一种标记语言）的递归缩写。在开发的这种语言时，YAML 的意思其实是："Yet Another Markup Language"（仍是一种标记语言），但为了强调这种语言以数据做为中心，而不是以标记语言为重点，而用反向缩略语重命名。

YAML 是一个可读性高，用来表达资料序列化的格式。特别适合用来表达或编辑数据结构、各种配置文件、倾印调试内容、文件大纲。

<!-- more -->

YAML 官网 [The Official YAML Web Site](https://yaml.org/) 。本文介绍 YAML 的基础语法，可以在[JS-YAML](https://nodeca.github.io/js-yaml/) 进行JavaScript解析。

# 基本语法

- 大小写敏感。
- 使用缩进表示层级关系。
- 缩进不允许使用 Tab 键，只允许空格。
- 缩进的空格数不重要，只要相同层级的元素左对齐即可。
- 文件中的字符串不需要使用引号标识。如果字符串之中包含空格或特殊字符，则需要放在引号之中（单引号和双引号均可）。
- `#` 表示注释，只支持单行注释。
- 一个文件中可以包含多个文档的内容。文档间用 `---`  表示一份内容的开始，也可以使用 `...` 表示一份内容的结束（非必需）。如果只是单个文档，分隔符 `---` 可省略。
- YAML在使用逗号及冒号时，后面都必须接一个空白字符，所以可以在字符串或数值中自由加入分隔符号而不需引号。
- YAML 文件一般以`.yml `为扩展名。

```yaml
# Ranking of 1998 home runs
---
- Mark McGwire
- Sammy Sosa
- Ken Griffey

# Team ranking
---
- Chicago Cubs
- St Louis Cardinals
```

# 数据类型

YAML 数据结构有很多种，但它们都可以用三个基本的数据类型表示

- 映射 (mappings)：键值对的集合，也称为哈希（hashes）、字典（dictionaries）
- 序列 (sequences)：一组按次序排列的值，也称为序列（arrays）、列表（lists）
- 标量 (scalars)：单个的、不可再分的值

YAML提供缩进区块（block ）以及内置（inline）两种格式，来表示映射和序列。

## 序列

习惯上序列比较常用区块格式（block format）表示，也就是用短杠加空格（`-`）来标记每个条目

```yaml
- value1
- value2
- value3
```

同时也支持行内置格式（inline format），用方括号` [] `包裹，并用逗号加空格分隔

```yaml
[value1, value2, value3]
```

YAML 支持多维序列，用缩进表示层级关系

```yaml
-
  - value1
  - value2
-
  - value3
  - value4
```

内置形式为

```yaml
[[value1, value2], [value3, value4]]
```

## 映射

映射使用冒号加空格来标记每个键值对

```yaml
key: value
```

映射区块形式（block format）使用缩进格式的层级关系分隔

```yaml
key: 
    child-key1: value1
    child-key2: value2
```

内置形式（inline format）在大括号中使用逗号加空格分隔 `key: value` 对

```yaml
key: {child-key1: value1, child-key2: value2} 
```

使用问号加空格声明一个复杂映射，允许你使用多个词汇（序列）来组成键，配合冒号加空格生成值

```yaml
? - Detroit Tigers
  - Chicago cubs
: - 2001-07-23

? [ New York Yankees, Atlanta Braves ]
: [ 2001-07-02, 2001-08-12, 2001-08-14 ]
```

内置形式为

```yaml
[complex-key1, complex-key2]: [complex-value1, complex-value2]
```

序列和映射可以构成**复合结构**

```yaml
# 映射中使用序列
languages: 
  - Ruby
  - Perl
  - Python 
  
# 映射中使用映射
websites:
  YAML: yaml.org 
  Ruby: ruby-lang.org 
  Python: python.org 
  Perl: use.perl.org

# 序列中使用映射
-
  name: Mark McGwire
  hr:   65
  avg:  0.278
-
  name: Sammy Sosa
  hr:   63
  avg:  0.288

# 序列中使用序列
- [name        , hr, avg  ]
- [Mark McGwire, 65, 0.278]
- [Sammy Sosa  , 63, 0.288]

# 紧凑的嵌套映射
- item    : Super Hoop
  quantity: 1
- item    : Basketball
  quantity: 4
- item    : Big Shoes
  quantity: 1
```

## 标量

标量是最基本的，不可再分的值，包括：

- **字符串(str)**：默认不使用引号包裹。如果字符串之中包含空格或特殊字符，需要放在引号之中（单引号和双引号均可）。

  ```yaml
  unicode: "Sosa did fine.\u263A"
  control: "\b1998\t1999\t2000\n"
  hex esc: "\x0d\x0a is \r\n"
  
  single: '"Howdy!" he cried.'
  quoted: ' # Not a ''comment''.'
  tie-fighter: '|\-*-/|'
  ```

- **布尔值(bool)**：

  ```yaml
  - [true, True, TRUE]
  - [false, False, FALSE]
  ```

- **整数(int)**：支持二进制、八进制和十六进制

  ```yaml
  canonical: 12345
  decimal: +12345
  octal: 0o14
  hexadecimal: 0xC
  ```

- **浮点数(float)**：支持科学计数法

  ```yaml
  canonical: 1.23015e+3
  exponential: 12.3015e+02
  fixed: 1230.15
  negative infinity: -.inf
  not a number: .nan
  ```

- **空值(null)**：`~/null/Null` 都表示空值，不指定值默认也是空值

  ```yaml
  empty:
  canonical: ~
  english: null
  English: Null
  ~: null key
  ```

- **时间戳(timestamp)**：使用 ISO 8601格式，最后使用`+/-`代表时区

  ```yaml
  canonical: 2001-12-15T02:59:43.1Z
  iso8601: 2001-12-14t21:59:43.10-05:00
  spaced: 2001-12-14 21:59:43.10 -5
  date: 2002-12-14
  ```

- YAML 会自动判定标量类型。但有时候也可以用两个感叹号显示标注类型

```yaml
not-date: !!str 2002-04-28

picture: !!binary |
 R0lGODlhDAAMAIQAAP//9/X
 17unp5WZmZgAAAOfn515eXv
 Pz7Y6OjuDg4J+fn5OTk6enp
 56enmleECcgggoBADs=
```

## 集合

```yaml
# Sets are represented as a
# Mapping where each key is
# associated with a null value
--- !!set
? Mark McGwire
? Sammy Sosa
? Ken Griffey
```

## 有序映射

```yaml
# Ordered maps are represented as
# A sequence of mappings, with
# each mapping having one key
--- !!omap
- Mark McGwire: 65
- Sammy Sosa: 63
- Ken Griffey: 58
```

# 区块字符

字符串可以写成多行，默认每行开头的缩进和行末空白会被去除，换行符会被转为空格。

```yaml
demo: 
  Mark McGwire's
  year was crippled
    by a knee injury.
```

转换为 JavaScript

```javascript
{ demo: 'Mark McGwire\'s year was crippled by a knee injury.' }
```

YAML 还有两种语法可以书写多行文字（multi-line strings），一种为保留换行，另一种为折叠换行。

- **保留换行(Newlines preserved)**：使用 `|` 字符，默认每行开头的缩进（以首行为基准）和行末空白会被去除，而额外的缩进会保留。
- **折叠换行(Newlines folded)**：使用 `>` 字符，和保留换行不同的是，只有空白行才视为换行，原本的换行字符会被转换成空白字符，而行首缩进会被去除。
-  `|` 字符或 `>` 字符后跟 `+` 可以保留文字块末尾的换行符，跟`-`表示删除文字块末尾的换行符号。

```yaml
name: Mark McGwire
accomplishment: >
  Mark set a major league
  home run record in 1998.
stats: |
  65 Home Runs
  0.278 Batting Average
  
with plus: |+
  This scalar
  spans many lines.
  
with minus: |-
  What a year!
  
```

转换为 JavaScript

```javascript
{ name: 'Mark McGwire',
  accomplishment: 'Mark set a major league home run record in 1998.\n',
  stats: '65 Home Runs\n0.278 Batting Average\n',
  'with plus': 'This scalar\nspans many lines.\n\n',
  'with minus': 'What a year!' }
```

# 引用

为了维持文件的简洁，并避免资料输入的错误，YAML提供了锚点标记 (`& `)、锚点引用 (`*`) 和序列合并 (`<<`) 。
重复的内容在YAML中可以使用 `&` 来定义锚点和别名，使用 `*` 来引用锚点。合并只有序列中可以使用，使用 `<<` 可以将键值自锚点标记复制到当前序列中。

下面是一个例子:

```yaml
merge:
  # 定义并命名了4各锚点 (anchor)
  - &CENTER { x: 1, y: 2 }
  - &LEFT
    x: 0
    y: 2
  - &BIG { r: 10 }
  - &SMALL
    r: 1 

  # 以下序列中的映射都是相等的：

  - # 明确的键值对
    x: 1
    y: 2
    r: 10
    label: nothing

  - # 合并一个键值对
    << : *CENTER
    r: 10
    label: center

  - # 合并多个键值对
    << : [*CENTER, *BIG] 
    label: center/big

  - # 覆盖
    << :  
      - *LEFT
      - *BIG
      - *SMALL 
    x: 1
    label: left/big/small
```

转换为 JavaScript

```javascript
{ merge: 
   [ { x: 1, y: 2 },
     { x: 0, y: 2 },
     { r: 10 },
     { r: 1 },
     { x: 1, y: 2, r: 10, label: 'nothing' },
     { x: 1, y: 2, r: 10, label: 'center' },
     { x: 1, y: 2, r: 10, label: 'center/big' },
     { x: 1, y: 2, r: 10, label: 'left/big/small' } 
   ] }
```

# 示例

以下是YAML的两个完整示例。第一个是样本发票，第二个是示例日志文件。

**例 2.27 发票**

```yaml
--- !<tag:clarkevans.com,2002:invoice>
invoice: 34843
date   : 2001-01-23
bill-to: &id001
  given  : Chris
  family : Dumars
  address:
    lines: |
      458 Walkman Dr.
      Suite #292
    city    : Royal Oak
    state   : MI
    postal  : 48046
ship-to: *id001
product:
- sku         : BL394D
  quantity    : 4
  description : Basketball
  price       : 450.00
- sku         : BL4438H
  quantity    : 1
  description : Super Hoop
  price       : 2392.00
tax  : 251.42
total: 4443.52
comments:
  Late afternoon is best.
  Backup contact is Nancy
  Billsmer @ 338-4338.
```

**例 2.28 日志文件**

```yaml
---
Time: 2001-11-23 15:01:42 -5
User: ed
Warning:
  This is an error message
  for the log file
---
Time: 2001-11-23 15:02:31 -5
User: ed
Warning:
  A slightly different error
  message.
---
Date: 2001-11-23 15:03:17 -5
User: ed
Fatal:
  Unknown variable "bar"
Stack:
- file: TopClass.py
  line: 23
  code: |
    x = MoreObject("345\n")
- file: MoreClass.py
  line: 58
  code: |-
    foo = bar
```

