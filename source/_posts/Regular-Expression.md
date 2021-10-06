---
ID: 168b7402946f2e7d5650790be4ca8226
title: 正则表达式
tags: 正则表达式
mathjax: false
copyright: true
date: 2018-05-01 18:26:12
categories: Regular Expression
sticky: false
---
正则表达式是对字符串（包括普通字符（例如，a 到 z 之间的字母）和特殊字符（称为“元字符”））操作的一种逻辑公式，就是用事先定义好的一些特定字符、及这些特定字符的组合，组成一个“规则字符串”，这个“规则字符串”用来表达对字符串的一种过滤逻辑。正则表达式是一种文本模式，该模式描述在搜索文本时要匹配的一个或多个字符串。

Cheat Sheet：[Basic Regular Expressions in R ](https://rstudio.com/wp-content/uploads/2016/09/RegExCheatsheet.pdf)

<!-- more -->

# 正则表达式

**普通字符**：普通字符包括没有显式指定为元字符的所有可打印和不可打印字符。可直接匹配自身。
**元字符**：元字符加斜杠\匹配自身(在R中斜杠为`\\`，匹配`+`应写为`\\+`

## 非打印字符

非打印字符也可以是正则表达式的组成部分。下表列出了表示非打印字符的转义序列

| 字符 | 描述                                                         |
| ---- | :----------------------------------------------------------- |
| \cx  | 匹配由x指明的控制字符。例如， \cM 匹配一个 Control-M 或回车符。<br />x 的值必须为 A-Z 或 a-z 之一。否则，将 c 视为一个原义的 'c' 字符。 |
| \f   | 换页符                                                       |
| \n   | 换行符                                                       |
| \r   | 回车符                                                       |
| \t   | 制表符                                                       |
| \v   | 垂直制表符                                                   |
| \s   | 空白字符，等价于`[ \f\n\r\t\v]`                              |
| \S   | 非空白字符，等价于`[^ \f\n\r\t\v]`                           |

## 打印字符

打印字符|描述|等价
---|---|---
`.`(dot)|匹配除换行符、制表符之外的任何单个字符|`[^\n\r]`
\d|匹配一个数字|`[0-9]`
\D|匹配一个非数字|`[^0-9]`
\w|匹配字母、数字、下划线|`[A-Za-z0-9_]`
\W|匹配非字母、数字、下划线|`[^A-Za-z0-9_]`

## 限定符

限定符|描述|等价
---|---|---
*|零次或多次|{0,}
+|一次或多次|{1,}
?|零次或一次|{0,1}
{n}|匹配确定的n次|
{n,}|至少匹配n次|
{n,m}|匹配>=n且<=m次|

{% note warning %}  限定符都是贪婪匹配，在之后放置 ?，可转换为最小匹配（如\*?） {% endnote %}

## 定位符

定位符|描述|示例
---|---|---
$|匹配结尾位置|`(.csv)$`
^|匹配开始位置|`^(Windows)`
\b|匹配单词边界位置|`er\b` 可以匹配`never` 中的 `er`，但不能匹配 `verb` 中的 `er`
\B|匹配非单词边界位置|`er\B` 能匹配 `verb` 中的 `er`，但不能匹配 `never` 中的 `er`

## 选择

选择|描述|举例
---|---|---
`(pattern)`|匹配pattern并获取结果
`(?:pattern)`|匹配 pattern 但不获取匹配结果|`Win(?:7\|10)`相当于`Win7\|Win10`简略版
`(?=pattern)`|正向肯定预查（look ahead positive assert)|`Win(?=XP)`能匹配`WinXP`中的`Win`，但不能匹配`Win10`中的`Win`
`(?!pattern)`|正向否定预查(negative assert)|`Win(?!XP)`能匹配"Win10"中的`Win`，但不能匹配`WinXP`中的`Win`
`(?<=pattern)`|反向(look behind)肯定预查| `(?<=7)Win`能匹配`7Win`中的`Win`，但不能匹配`10Win`中的`Win`
`(?<!pattern)`|反向否定预查| `(?<!7)Win`能匹配`10Win`中的`Win`，但不能匹配`7Win`中的`Win`

## 字符集合

| 字符集合 | 描述                                                         |
| -------- | ------------------------------------------------------------ |
| x \| y   | 或，优先级最低                                               |
| [xyz]    | 字符集合，匹配任意一个                                       |
| [a-z]    | 小写字母                                                     |
| [0-9]    | 数字                                                         |
| [^xyz]   | 匹配不在集合里的任意字符                                     |
| [^a-z]   | 除了小写字母                                                 |
| [A-Z]    | 大写字母                                                     |
| \num     | 匹配 num，其中 num 是一个正整数。对所获取的匹配的引用。<br />例如，'(.)\1' 匹配两个连续的相同字符。 |

# R语言通用字符集

| 字符集合   | 描述                                           |
| :--------- | :--------------------------------------------- |
| [:punct:]  | punctuation                                    |
| [:alpha:]  | letters.                                       |
| [:lower:]  | lowercase letters.                             |
| [:upper:]  | upperclass letters.                            |
| [:digit:]  | digits.                                        |
| [:xdigit:] | hex digits.                                    |
| [:alnum:]  | letters and numbers.                           |
| [:cntrl:]  | control characters.                            |
| [:graph:]  | letters, numbers, and punctuation.             |
| [:print:]  | letters, numbers, punctuation, and whitespace. |
| [:space:]  | space characters (basically equivalent to \s). |
| [:blank:]  | space and tab.                                 |



