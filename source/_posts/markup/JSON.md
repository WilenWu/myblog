---
title: JSON 介绍
date: 2021-11-23 21:12:17
tags: [标记语言,JSON]
categories: [标记语言]
description: False
cover: /img/file-json-wide.svg
top_img: /img/json-logo-wide.svg
---

# JSON 简介

[JSON](http://www.json.org.cn/) 指的是 JavaScript 对象表示法（**J**ava**S**cript **O**bject **N**otation），是一种轻量级的文本数据交换格式。它使得人们很容易的进行阅读和编写。同时也方便了机器进行解析和生成。它是基于 Javascript 语法来描述数据对象，但是 JSON 采用完全独立于编程语言的文本格式来存储和表示数据。简洁和清晰的层次结构使得 JSON 成为理想的数据交换语言。 

JSON基于两种结构：主要由对象 `{ }` 和数组 `[ ]` 组成

# 基本数据类型

- **对象 (object)**：若干无序的键-值对 (key-value pairs) 集合，其中键只能是字符串，值必须是有效的 JSON 数据类型。建议但不强制要求对象中的键是独一无二的。键-值对之间使用逗号分隔，键与值之间用冒号`:`分割。对象以花括号`{}` 包围：

  ```json
  {key1 : value1, key2 : value2, ... }
  ```

  取值使用 `objcet.key`

- **数组 (array)**：值 (value) 的有序集合。每个值可以为任意类型，值之间用逗号`,`分割。数组以中括号`[]` 包围：

  ```json
  [value1, value2, ...]
  ```

  利用索引取值 `array[index]`，索引从 0 开始

- **字符串 (string)**：是由双引号包围的任意数量Unicode字符的集合，使用反斜线转义。

- **数值 (number)**：十进制数，不能有前导0，可以为负数，可以有小数部分。还可以用`e`或者`E`表示科学计数法。不能包含非数，如NaN。不区分整数与浮点数。
- **布尔值 (bool)**：`true/false` 
- **空值 (null)** 

# 示例

```json
{
     "firstName": "John",
     "lastName": "Smith",
     "sex": "male",
     "age": 25,
     "cars":[ "Porsche", "BMW", "Volvo" ],
  "middlename":null,
     "address": 
     {
         "streetAddress": "21 2nd Street",
         "city": "New York",
         "state": "NY",
         "postalCode": "10021"
     },
     "phoneNumber": 
     [
         {
           "type": "home",
           "number": "212 555-1234"
         },
         {
           "type": "fax",
           "number": "646 555-4567"
         }
     ]
 }
```

# 与 JavaScript 的交互

- 通过 `JSON.stringify()` 把 JavaScript 对象转换为字符串。
- 通过 `JSON.parse()` 解析数据，这些数据会成为 JavaScript 对象。

所有主流浏览器和最新的 ECMAScript (JavaScript) 标准都包含 `JSON.parse()` 和 `JSON.stringify()` 函数
