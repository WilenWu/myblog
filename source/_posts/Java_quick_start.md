---
title: 30分钟速成Java
categories:
  - General
tags:
  - Java
cover: 
top_img: 
abbrlink: 8f3475e8
date: 
description:
---

Java 程存由注释、import 语句和类声明组成。

## 注释

```java
/* comm */
// 单行注释
/* 头注释
 * 头注释
 */
```



## import 语句

将系统定义的类或自定义的类组织到包 (package) 中，以方便使用。然后，用 import 语句引入包中的类。

```java
import package_name.class_name; 
import package_name.*; 
```

# 类

## 类和对象

类声明

```java
class Student{
  ... ...
}
```

程序中必须有一个主类，主类名必须和文件名相同。当类为主类时，必须定义一个为 main 的方法，Java 运行时，从主类的 main 方法开始运行。

方法声明

```java
修饰符 返回数据类型 方法名 (参数){
  ... ... 
}
```

对象

```java
类名 对象名序列; // 对象声明
对象名 = new 类名(参数); // 对象生成
```

> 一般类名以大写字母开头，对象名以小写字母开头。

也可将对象声明和对象生成联合使用

```java
类名 对象名 = new 类名(参数);
```

向对象发送消息

```java
对象名.方法名(参数);
```

```java
import javax.swing.*;
public class MyJava {
    public static void main(String[] args) {
        JOptionPane.showMessageDialog(null,"Hello world!");
        System.exit(0);
    }
}
```

## 自定义类

构造函数

```java
class Student{
    private String name;
    public Student{
        name = "Alice";
    }
    public String getName(){
        return name;
    }
    public void setName(String name){
        return name;
    }
}
```



# 标准包

## GUI

在图形用户界面 (GUI) 中，大体有两种窗体：框架窗体 (JFrame) 和 对话窗体 (JDialog)。

```java
import javax.swing.*;

JFrame window = JFrame();
window.setVisible(true);
```

用于输入/输出的 JOptionPane 类

```java
JOptionPane.showMessageDialog(null,"Hello world!");
```

第一个参数null表示没有 JFrame 对象，则对话显示在屏幕中央，若传递一个 JFrame 对象，对话在框架中央。

```java
JOptionPane.showMessageDialog(window, "Hello world!");
```

<img src="../../../../../Downloads/JOptionPane.showMessageDialog.png" style="zoom:50%;" />

```java
String name = JOptionPane.showInputDialog(null,"What is your name?");
```

若单机 cancel ，则任何字符都被忽略，并返回 null；若无输入，单机ok，返回空字符串。 

<img src="../../../../../Downloads/JOptionPane.showInputDialog.png" style="zoom:50%;" />

## 字符

双引号分隔的字符序列为String 常量

```java
String message = "Hello world!"
```

常用方法

| 方法         | 说明                                                         | 示例                       |
| ------------ | ------------------------------------------------------------ | -------------------------- |
| .substring() | 子字符串                                                     | `message.substring(0, 4)`  |
| .length()    | 字符长度                                                     | `message.length()`         |
| .indexOf()   | 获取子字符串的位置。若不存在，返回-1；若有多组，返回第一个。 | `message.indexOf(“world”)` |
|              |                                                              |                            |

若 `+` 两边都是数值，则为运算符，否则为字符连接符，也可直接连接数值
```java
"My age is " + 18
```

## 数值

Java 有6种数值数据类型：byte short int long float double

运算符

显示转换

```java
(float) 1 / 3; 
```

常量保留字 final

```java
final double PI = 3.14
```

常量名常用大写表示

字符型数字转换

| 包裹类  | 转换方法    |
| ------- | ----------- |
| Integer | parseInt    |
| Float   | parseFloat  |
| Long    | parseLong   |
| Double  | parseDouble |

```java
int num = Integer.parseInt("42")
```

## 标准输入/输出

`System.out` 指向一个预先生成的 `PrintStream` 对象，用于将文本输出到标准输出窗口

```java
System.out.print("Hello");
System.out.println("Hello"); // 换行输出
```

`System.in` 是  `InputStream`  类的实例，使用read方法时一次只能输入1字节，`System.in` 需要与 `java.util.Scanner` 对象联用。

nextInt, nextFloat … 等方法输入数值

next 方法输入字符串

```java 
Scanner scanner = new Scanner(System.in);
System.out.print("Enter: ");
String quote = scanner.next();
System.out.println("You entered: " + quote);
```

## Math

Math 常量有 PI 和 E 两个

Math.random()



## 日期

Date.toString 方法将时间的格式转换为字符串显示

```java
import java.util.*;
Date today = new Date();
```

javabook.Clock.pause() 方法时间延迟