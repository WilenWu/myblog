---
title: Java速成入门
categories:
  - General
tags:
  - Java
cover: /img/java_cover.png
top_img: /img/java_top_img.jpg
abbrlink: db6e5578
date: 2024-09-17 22:55:05
description:
---

# 基本语法

[Java 官方文档](https://www.w3cschool.cn/java/dict)

## 主函数

以下我们通过一个简单的实例来展示 Java 编程。首先创建文件 **Hello.java**

```java
import javax.swing.*;
public class Hello {
    public static void main(String[] args) {
        JOptionPane.showMessageDialog(null,"Hello world!");
    }
}
```

每个 Java源文件必须有一个主类，且**主类名必须和文件名相同**。当类为主类时，必须定义一个名为 `main` 的方法，称为主函数。主函数的格式是固定的

```java
public static void main(String[] args) {
    \\ ... 
}
```

所有的 Java 程序从主函数 `main` 开始执行。

Java 是大小写敏感的，且每一行语句必须以`;`结束。

## 编译运行

Java源码本质上是一个文本文件，我们需要先用`javac`把`Hello.java`编译成字节码文件`Hello.class`，然后，用`java`命令执行这个字节码文件。

```shell
$ javac Hello.java
$ java Hello
```

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/run_java.jpg" style="zoom:80%;" />

## 环境变量

`-cp` 是 `-classpath` 的简写，是JVM用到的一个环境变量，它用来指示JVM搜索`class`的路径和顺序。例如

```bash
java -cp .:~/Project/bin:/bin/com.example.Hello
```

在`demo.jar`文件里去搜索类 `com.example.Hello`

```bash
java -cp ./demo.jar com.example.Hello
```

如果没有设置系统环境变量，也没有传入`-cp` 参数，那么JVM默认的`classpath`为`.`，即当前目录。

## 注释

Java 注释主要有三种类型：

单行注释

```java
// comment
```

多行注释

```java
/*
Comment (line 1)
Comment (line 2)
...
*/
```

文档注释

```java
/**
 * This is a documentation comment.
 * This is my first Java program.
 * This will print 'Hello World' as the output
 * This is an example of multi-line comments.
*/
```

# 数据类型

## 基本数据类型

Java是一种强类型语言，这意味着每个变量都必须声明其类型。基本数据类型包括

- 整数类型：byte，short，int，long
- 浮点数类型：float，double
- 字符类型：char
- 布尔类型：boolean

每个变量必须先声明类型后使用

```java
int x, y; // 变量声明（允许同时声明多个变量）
x = 1; // 变量生成
```

也可将变量声明和生成联合使用

```java
int age = 25;
float salary = 2500.50f;
boolean isEmployed = true;
char gender = 'M';
```

注意：对于`float`类型，需要加上`f`后缀。布尔类型`boolean`只有`true`和`false`两个值。`char`类型使用单引号`'`，且仅有一个字符。

定义变量的时候，如果加上`final`修饰符，这个变量就变成了常量：

```java
final double PI = 3.14;
```

为了和变量区分开来，常量名通常**全部大写**。

## 字符串

字符串是 Java 一个内置的类，使用双引号表示

```java
String message = "Hello world!";
```

实际上字符串在`String`内部是通过一个`char[]`数组表示的。

## 数组

**数组**用于存储一组类型相同的变量，所有元素初始化为默认值，整型都是`0`，浮点型是`0.0`，布尔型是`false`。

```java
int[] array = new int[5];
```

可以使用索引访问或修改数组元素，索引从`0`开始。

```java
array[1] = 79;
```

也可以在定义数组时直接指定初始化的元素

```java
int[] array = new int[] {68, 79, 91, 85, 62};
```

还可以进一步简写为：

```java
int[] array = {68, 79, 91, 85, 62};
```

可以用 `length` 属性获取数组大小，数组一旦创建后，大小就不可改变。

```java
System.out.println(array.length); // 5
```

**多维数组**

```java
int[][] ns = {
    { 1, 2, 3, 4 },
    { 5, 6 },
    { 7, 8, 9 }
};
```

访问二维数组的某个元素需要使用`array[row][col]`。

**打印数组**：直接打印数组变量，得到的是数组在JVM中的引用地址

```java
System.out.println(array); // [I@28a418fc
```

因此，常用`for`循环来遍历数组

```java
for (int i=0; i<array.length; i++) {
    System.out.println(array[i]);
}
```

`for each`循环可以直接遍历数组的每个元素；

```java
for (int element : array) {
    System.out.println(element);
}
```

和`for`循环相比，`for each`循环的变量不再是计数器，而是直接对应到数组的每个元素。

Java标准库也提供了`Arrays.toString()`，可以快速打印数组内容：

```java
import java.util.Arrays;

System.out.println(Arrays.toString(array)); // [68, 79, 91, 85, 62]
```

## 运算符

### 数学运算符

| Operator |  Description   |
| :------: | :------------: |
|    +     |    Addition    |
|    -     |  Subtraction   |
|    *     | Multiplication |
|    /     |    Division    |
|    %     |    Modulus     |
|    ++    |   Increment    |
|    --    |   Decrement    |

不过整数运算只能得到结果的整数部分

```java
float x = 1 / 3; // 0.0
```

也可强制转换类型

```java
float x = (float) 1 / 3; // 0.33333334
```

 `+` 也可以连接字符串和其他数据类型

```java
String message = "My age is " + 18;
```

### 关系运算符

| Operator |       Description        |
| :------: | :----------------------: |
|    ==    |         equal to         |
|    !=    |       not equal to       |
|    >     |       greater than       |
|    <     |        less than         |
|    >=    | greater than or equal to |
|    <=    |  (less than or equal to  |

注意：比较两个浮点数是否相等不能直接用 `==` 运算符，而应该判断差值小于某个极小值。比较两个字符串是否相同时，必须使用`equals()`方法。

### 逻辑运算符

| Operator | Description |
| :------: | :---------: |
|    &&    | logical and |
|   \|\|   | logical or  |
|    !     | logical not |

### 赋值运算符

| Operator |                 Description                 |
| :------: | :-----------------------------------------: |
|    =     | C = A + B will assign value of A + B into C |
|    +=    |      C += A is equivalent to C = C + A      |
|    -=    |      C -= A is equivalent to C = C − A      |
|    *=    |      C *= A is equivalent to C = C * A      |
|    /=    |      C /= A is equivalent to C = C / A      |
|    %=    |      C %= A is equivalent to C = C % A      |
|   <<=    |        C <<= 2 is same as C = C << 2        |
|   >>=    |        C >>= 2 is same as C = C >> 2        |
|    &=    |         C &= 2 is same as C = C & 2         |
|    ^=    |         C ^= 2 is same as C = C ^ 2         |
|   \|=    |        C \|= 2 is same as C = C \| 2        |

### 三元运算符

条件运算符也被称为三元运算符。该运算符有3个操作数，并且需要判断布尔表达式的值。该运算符的主要是决定哪个值应该赋值给变量。

```java
variable x = (expression) ? value_if_true : value_if_false
```

```java
int n = -100;
int x = n >= 0 ? n : -n;
```

### instanceof

该运算符用于判断对象所属的类

```java
String name = "James";
boolean result = name instanceof String; 
```

# 流程控制

## 条件语句

```java
if (age > 60) {
    System.out.println("Elder");
} else if (age > 18) {
    System.out.println("Adult");
} else {
    System.out.println("Minor");
}
```

## 循环语句

### while循环

```java
int sum = 0;
int n = 1;
while (n <= 100) {
    sum = sum + n;
    n ++;
}
System.out.println(sum); // 5050
```

### do-while 循环

```java
int sum = 0;
int n = 1;
do {
    sum = sum + n;
    n ++;
} while (n <= 100);
System.out.println(sum);  // 5050
```

### for循环

`for`循环会先初始化计数器，然后，在每次循环前检测循环条件，在每次循环后更新计数器。

```java
for (int i = 0; i <= 100; i++) {
    sum = sum + i;
}
System.out.println(sum);
```

### 跳出循环

`break` 主要用在循环语句或者 switch 语句中，用来跳出整个语句块。

```java
int sum = 0;
for (int i=1; ; i++) {
    sum = sum + i;
    if (i == 100) {
        break;
    }
}
System.out.println(sum);
```

`continue`语句用于提前结束本次循环

```java
int sum = 0;
for (int i=1; i<=10; i++) {
    System.out.println("begin i = " + i);
    if (i % 2 == 0) {
        continue; 
    }
    sum = sum + i;
    System.out.println("end i = " + i);
}
System.out.println(sum); // 25
```

## 分支语句

```java
switch (gender) {
    case 'M':
        System.out.println("Male");
        break;
    case 'F':
        System.out.println("Female");
        break;
    default:
        System.out.println("Other");
}
```

使用`switch`时，注意`case`语句并没有花括号`{}`，如果遗漏了`break`，后续语句将全部执行，直到遇到`break`语句。当没有匹配到任何`case`时，执行`default`。

# 面向对象

Java是一种面向对象（OOP）的编程语言，类和对象是其核心概念。

## 类与对象

例如，创建一个 Animal 类 

```java
class Animal {
    public int age;
    
    void sound() {
        System.out.println("Animal sound");
    }
}
```

类内封装了多个属性和方法。定义类就是定义了一种数据类型，对象就是这种数据类型的实例。

在Java中，使用关键字 new 来创建一个新的对象

```java
Animal animal; // 对象声明
animal = new Animal(); // 对象生成
```

也可将对象声明和对象生成联合使用

```java
Animal animal = new Animal();
```

一般类名以大写字母开头，对象名以小写字母开头。类的方法名以小写字母开头。

## 修饰符

Java 支持 4 种不同**访问控制修饰符**

| 修饰符      | 作用域                 |
| :---------- | :--------------------- |
| `public`    | 对所有类开放           |
| `protected` | 同一个包的类或子类访问 |
| `default`   | 同一个包的类访问       |
| `private`   | 仅限类内访问           |

如果在类、变量、方法或构造函数的定义中没有指定任何访问修饰符，那么它们就默认具有默认访问修饰符。默认访问修饰符的访问级别是包级别（package-level），即只能被同一包中的其他类访问。

## 方法

在方法内部，可以使用一个隐含的变量 `this`，它始终指向当前实例。如果没有命名冲突，可以省略`this`。方法返回值通过`return`语句实现，如果没有返回值，返回类型设置为`void`，可以省略`return`。

**构造方法**：实例在创建时通过 `new` 操作符会调用其对应的构造方法，构造方法用于初始化实例。构造方法的名称就是类名。和普通方法相比，构造方法没有返回值（也没有`void`），调用构造方法，必须用`new`操作符。

```java
class Animal {
    int age;
    Animal() {
        this.age = 0;
    }
    Animal(int age) {
        this.age = age;
    }
}
```

若没有定义构造方法，编译器会自动创建一个默认的无参数构造方法；可以定义多个构造方法，编译器根据参数自动判断。

**可变参数**：相当于数组类型

```java
class Group {
    private String[] names;

    public void setNames(String... names) {
        this.names = names;
    }
}
```

## 继承

继承是通过继承父类的属性和方法来创建新类。通过继承，可以实现代码的复用和扩展。Java使用`extends`关键字来实现继承，且只支持单继承，但可以实现多接口。

```java
class Animal {
    void eat() {
        System.out.println("Eating");
    }
}

class Dog extends Animal {
    void bark() {
        System.out.println("Barking");
    }
}

public class Main {
    public static void main(String[] args) {
        Dog dog = new Dog();
        dog.eat();
        dog.bark();
    }
}
```

子类引用父类的字段时，可以用 super 关键字。

## 多态

多态是指同一操作在不同对象上具有不同的行为。通过多态，可以提高代码的灵活性和扩展性。

```java
class Animal {
    void sound() {
        System.out.println("Animal sound");
    }
}

class Dog extends Animal {
    void sound() {
        System.out.println("Barking");
    }
}

class Cat extends Animal {
    void sound() {
        System.out.println("Meowing");
    }
}

public class Main {
    public static void main(String[] args) {
        Animal a1 = new Dog();
        Animal a2 = new Cat();
        a1.sound();  // Barking
        a2.sound();  // Meowing
    }
}
```

## 静态

在类中用 static 关键字声明的方法和属性，属于类而不是实例，在加载时被初始化，而且只初始化一次。调用实例方法必须通过一个实例变量，而调用静态方法则不需要实例变量，通过类名就可以调用。

```java
class Animal {
    public static int number;
    public static void setNumber(int value) {
        number = value;
    }
}
```

# 包管理

## 创建包

为了更好地组织类，Java 提供了包机制，用于区别类名的命名空间。从概念上讲，我们可以将包看作是计算机上的不同文件夹。所有Java文件对应的目录层次要和包的层次一致。

 在定义类的时候，我们需要在第一行使用关键字 `package` 声明这个类的所属包，例如

```java 
package animals;

public class Cat{
    // ...
}
```

上面的代码就表示 `Cat` 类属于 `animals` 包，也就意味着 Cat.java 文件必须保存在 animals/ 文件夹中。包的作用是把不同的 java 程序分类保存，更方便的被其他 java 程序调用。

包名通常使用小写的字母来命名，避免与类、接口名字的冲突。

如下我们在 `animals` 包定义了3个类

```java
package animals;

public class Animal {   
    public void eat() {
        System.out.println("Eating");
    }
}
```

```java
package animals;

public class Cat extends Animal {
    public void sound() {
        System.out.println("Barking");
    }
}
```

```java
package animals;

public class Dog extends Animal {
    public void bark() {
        System.out.println("Barking");
    }
}
```

包可以是多层结构，用`.`隔开，例如 `person.students`

```java
package person.students;

public class Tom{
     public static void speak(String name){
         System.out.println("hello" + name);
     }
 }
```

最终的目录结构如下

```
├─ animals
│  ├─ Animal.java
│  ├─ Cat.java
|  └─ Dog.java
├─ person
|  └─ students
|     └─ Tom.java
└─ Hello.java
```

## 导入包

**import** 关键字用于引入其他包中的类、接口或静态成员，它允许你在代码中直接使用其他包中的类，而不需要完整地指定类的包名。

例如，下面的命令将会载入 animals/ 路径下的 Dog 类和 person/students 路径下的 Tom 类。

```java
import animals.Dog;
import person.students.Tom;;

public class Hello {
    public static void main(String[] args) {
        Dog dog = new Dog();
        dog.bark();
        Tom.speak("Alice");
    }
}
```

在写`import` 的时候，可以使用通配符 `*` 来导入指定包下面的所有类。package 语句应该在源文件的首行，import 语句放在 package 语句和类定义之间。

JDK的核心类使用`java.lang`包，编译器会自动导入。

## Jar 包

`.class`文件是JVM的最小可执行文件，而一个大型程序需要编写很多类，并生成一堆`.class`文件，散落在各层目录中，肯定不便于管理。而 jar 包相当于`class`文件的目录，是一个zip格式的压缩文件，方便下载和使用。

上节Java 项目的源文件目录结构如下，其中 Hello 是主类，会调用 animals包和 students包中的类。

```
├─ animals
│  ├─ Animal.java
│  ├─ Cat.java
|  └─ Dog.java
├─ person
|  └─ students
|     └─ Tom.java
└─ Hello.java
```

第一步：用以下命令编译源文件

```shell
javac Hello.java -d target 
```

此时被主类调用的文件同时被编译，并都放到target文件夹下。

第二步：在target目录下下添加主清单文件 META-INF/MANIFEST.MF 定义 Main-Class: Hello

```shell
cd target
touch META-INF/MANIFEST.MF
```

```
Manifest-Version: 1.0
Created-By: 1.8.0_391 (Oracle Corporation)
Main-Class: Hello 

```

注意 Main-Class 冒号后面有一个空格，整个文件最后有一行空行。如果未定义 Main-Class ，生成的 hello.jar 包运行时会报错：缺少主清单属性。

最终生成的target目录结构如下

```
target
├─ animals
│  ├─ Animal.class
│  ├─ Cat.class
|  └─ Dog.class
├─ person
|  └─ students
|     └─ Tom.class
├─ Hello.class
└─ META-INF
   └─ MANIFEST.MF
```

第三步：将target 文件夹下的所有文件都打进 jar 包

```shell
jar -cvfm hello.jar *
```

其中，参数 c 表示要创建一个新的 jar 包，v 表示创建的过程中在控制台输出创建过程的一些信息，f 表示给生成的 jar 包命名，m 表示要定义MANIFEST文件。* 表示把当前目录下所有文件都打在jar包里。

第四步：运行jar包

```shell
java -jar hello.jar 
```

# 标准包

## 集合操作

Java的集合工具包 `java.util` 提供了多种数据结构，如列表（List）、集合（Set）和映射（Map）。这些数据结构用于存储和操作数据。

`ArrayList` 是支持可以根据需求动态生长的数组，初始长度默认为 10。

```java
import java.util.ArrayList; 

ArrayList<String> list = new ArrayList<>();
list.add("Apple");
list.add("Banana");

System.out.println(list);
```

| 常用方法                  | 功能                                             |
| :------------------------ | :----------------------------------------------- |
| `size()`                  | 返回ArrayList的长度                              |
| `add(Integer val)`        | 在ArrayList尾部插入一个元素                      |
| `add(int idx, Integer e)` | 在ArrayList指定位置插入一个元素                  |
| `get(int idx)`            | 返回ArrayList中第 idx 位置的值，若越界则抛出异常 |
| `set(int idx, Integer e)` | 修改ArrayList中第 idx 位置的值                   |

`Set` 是保持容器中的元素不重复的一种数据结构

```java
import java.util.HashSet;

HashSet<String> set = new HashSet<>();
set.add("Apple");
set.add("Banana");
set.add("Apple");

System.out.println(set);
```

| 常用方法                  | 功能                            |
| :------------------------ | :------------------------------ |
| `size()`                  | 返回Set的长度                   |
| `add()`                   | 插入一个元素进Set               |
| `contains()`              | 判断 Set 中是否有元素 val       |
| `addAll(Collection e)`    | 将一个容器里的所有元素添加进Set |
| `retainAll(Collection e)` | 将Set改为两个容器内相同的元素   |
| `removeAll(Collection e)` | 将Set中与 e 相同的元素删除      |

`Map` 是维护键值对 `<Key, Value>` 的一种数据结构，其中 `Key` 唯一

```java
import java.util.HashMap;

HashMap<String, Integer> map = new HashMap<>();        
map.put("Apple", 1);        
map.put("Banana", 2); 

System.out.println(map);
```

| 常用方法                          | 功能                               |
| :-------------------------------- | :--------------------------------- |
| `put(Integer key, Integer value)` | 插入一个元素进Map                  |
| `size()`                          | 返回Map的长度                      |
| `containsKey(Integer val)`        | 判断 Map中是否有元素 key 为 val    |
| `get(Integer key)`                | 将Map中对应的 key 的 value 返回    |
| `keySet`                          | 将Map中所有元素的 key 作为集合返回 |

`Arrays` 是 `java.util` 中对数组操作的一个工具类。方法均为静态方法，可使用类名直接调用。

| 常用方法                | 功能                       |
| ----------------------- | -------------------------- |
| `Arrays.toString()`     | 快速打印数组内容           |
| `Arrays.deepToString()` | 快速打印多维数组           |
| `Arrays.sort()`         | 对数组进行排序             |
| `Arrays.fill()`         | 对数组指定区间赋值         |
| `Arrays.binarySearch()` | 对数组连续区间进行二分搜索 |

```java
import java.util.Arrays;

System.out.println(array); // [I@28a418fc
System.out.println(Arrays.toString(array)); // [68, 79, 91, 85, 62]
```

`Collections` 是 `java.util` 中对集合操作的一个工具类。方法均为静态方法，可使用类名直接调用。

| 常用方法                              | 功能                         |
| ------------------------------------- | ---------------------------- |
| `Collections.sort()`                  | 对集合进行排序               |
| `Collections.binarySearch(list, key)` | 对集合中指定区间进行二分搜索 |
| `Collections.swap(list, i, j)`        | 交换集合中指定二个位置的元素 |

## GUI

在标准图形用户界面 (GUI) 中，大体有两种窗体：框架窗体 (JFrame) 和 对话窗体 (JDialog)

```java
import javax.swing.*;
```

 **JOptionPane 类输出**

```java
JOptionPane.showMessageDialog(null,"Hello world!");
```

第一个参数null表示没有 JFrame 对象，则对话显示在屏幕中央，若传递一个 JFrame 对象，对话在框架中央。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/JOptionPane.showMessageDialog.png" style="zoom:50%;" />

 **JOptionPane 类输入**

```java
String name = JOptionPane.showInputDialog(null,"What is your name?");
```

若单击 cancel ，则任何字符都被忽略，并返回 null；若无输入，单击ok，返回空字符串。 

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/JOptionPane.showInputDialog.png" style="zoom:50%;" />

**JOptionPane 类确认对话框**

```java
int selection = JOptionPane.showConfirmDialog(null,"Yes or No?");
```

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/common/JOptionPane.showConfirmDialog.png" alt="JOptionPane.showConfirmDialog" style="zoom:50%;" />

**JFileChooser 类选择文件**

```java
JFileChooser chooser = new JFileChooser("/path/to/file");
chooser.showDialog(null, "Compile");
```

获得文件目录

```java
File directory = chooser.getCurrentDirectory();
```

## 命令行参数

用户输入的命令行参数直接传递给主类的`main`方法：参数类型是`String[]`数组

```java
public class Terminal {
    public static void main(String[] args) {
        for (String arg : args) {
            System.out.println(arg);
        }
    }
}
```

## Number & Math

一般地，当需要使用数字的时候，我们通常使用内置数据类型。然而，在实际开发过程中，我们经常会遇到需要使用对象，而不是内置数据类型的情形。因此 Java 为八个基本类型提供了对应的包装类，两者之间的对应关系如下：

| 基本数据类型 | 包装数据类型 |
| :----------: | :----------: |
|    `byte`    |    `Byte`    |
|   `short`    |   `Short`    |
|  `boolean`   |  `Boolean`   |
|    `char`    | `Character`  |
|    `int`     |  `Integer`   |
|    `long`    |    `Long`    |
|   `float`    |   `Float`    |
|   `double`   |   `Double`   |

包装类型实默认值是 `null`，且实例后才能使用，只能使用 `equals()` 方法比较是否相等。

Java 的 Math 包含了用于执行基本数学运算的属性和方法，如初等指数、对数、平方根和三角函数。Math 的方法都被定义为 static 形式，通过 Math 类可以在主函数中直接调用。

Math.random()

## 日期时间

java.util 包提供了 Date 类来封装当前的日期和时间。使用Date.toString 方法将时间的格式转换为字符串显示。

```java
Date date = new Date();
System.out.println(date.toString());
```

`java.text.SimpleDateFormat `允许自定义日期时间格式

```java
SimpleDateFormat ft = new SimpleDateFormat ("yyyy-MM-dd hh:mm:ss");
System.out.println("Current Date: " + ft.format(date)
```

## 输入输出

### 标准IO

**输出**：

`System.out` 指向一个预先生成的 `PrintStream` 对象，用于将文本输出到标准输出窗口。`println` 是print line的缩写，表示输出并换行。如果输出后不想换行，可以用`print()`：

```java
System.out.print("Hello");
System.out.println("Hello"); // 换行输出
```

**输入**：`System.in` 是  `InputStream`  类的实例，使用read方法时一次只能输入1字节。可以通过 `Scanner` 类来处理命令行输入

```java
import java.util.Scanner;

Scanner scan = new Scanner(System.in); // System.in 是输入流
System.out.print("Enter: ");
String quote = scan.next();
System.out.println("You entered: " + quote);
```

其中 `Scanner.next()`  方法用于输入字符串，另外还有 nextInt, nextFloat, nextDouble 等方法用于输入数值。

### 格式化输出

可以对变量进行格式化输出。

| 符号 |    意义    |
| :--: | :--------: |
| `%f` |  浮点类型  |
| `%s` | 字符串类型 |
| `%d` |  整数类型  |
| `%c` |  字符类型  |

## 文件IO

**得到当前目录路径**

```java
String current = System.getProperty("user.dir");
```

`java.io` 用于处理文件和流。常见的类包括File、FileReader、FileWriter、BufferedReader、BufferedWriter等。

```java
import java.io.*;
```

```java
// Write to a file
FileWriter writer = new FileWriter("output.txt");
writer.write("Hello, World!");
writer.close();
```

```java
// Read from a file
FileReader reader = new FileReader("output.txt");
BufferedReader bufferedReader = new BufferedReader(reader);

String line;
while ((line = bufferedReader.readLine()) != null) {
    System.out.println(line);
}
reader.close();
```

