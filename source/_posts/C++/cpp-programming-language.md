---
title: C++ 快速入门
categories:
  - C++
  - Basics
tags:
  - C++
cover: /img/cpp-introduction.png
top_img: /img/cpp-top-img.svg
abbrlink: 3f4b6fbd
date: 2025-03-03 22:16:01
description: 
---

C语言是一种通用的、过程式的编程语言，广泛用于系统软件和应用程序开发。C++ 进一步扩充和完善了 C 语言，是一种面向对象的程序设计语言。

https://en.cppreference.com/w/
https://www.studycpp.cn

# 基础知识

## 程序结构

以下代码是最简单的C\+\+程序之一，它将帮助我们理解C\+\+程序的基本语法结构。

```cpp
// Header file for input output functions
#include <iostream>
using namespace std;

// main function: where the execution of C++ program begins
int main() {
    // This statement prints "Hello World"
    cout << "Hello World" << endl;
    return 0;
}
```

- C++ 程序按照代码的书写顺序执行，其中`main()`函数是每个程序的入口点，且只有一个`main`函数。
- 语句块是一组使用大括号`{}` 括起来的按逻辑连接的语句。
- 每个语句必须以分号结束。
- 上例中 `#include<iostream>` 是一个预处理器指令，告诉编译器引入C++标准输入/输出库 `iostream` 。
- `using namespace std` 将 std 命名空间的实体导入到程序中，它基本上是定义所有内置函数的空间。

## 编译和运行

<img title="" src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/Compilation-Process-in-C.png" alt="compilation process in c" width="278" data-align="center">

常用的编译器

- GCC 全称GNU Compiler Collection，是由 GUN 项目开发的编译器套件。包含了 C、C++、Objective-C、Fortran、Ada和Go等多种高级编程语言的编译器。
- Clang 用于编译C、C++、Objective-C和Objective-C++的编译器前段。采用LLVM为后端，一般MacOS下使用较多。
- MSVC Microsoft开发的 C和C++编译器，Windows系统使用较多。

大多数的 C++ 编译器并不在乎源文件的扩展名，一般默认使用 **.cpp**。

```shell
g++ helloworld.cpp -o hello
```

## 注释

C++有两种注释类型：单行注释 `//` 和多行注释 `/*...*/`

```c
#include <iostream>
using namespace std;

int main() 
{
    // This is a single line comment
    /* This is a multi-line comment
     which can span multiple lines */
    cout << "Hi"; // After line comment here
    return 0;
}
```

# 数据类型与变量

## 数据类型

C++ 是一门静态类型语言。这意味着任何变量都有一个相关联的类型，并且该类型在编译时是可知的。

- **基本数据类型**：`int`，`float`，`double`,  `char`, `bool`, `void`
- **派生数据类型**：数组(array)、指针(pointer)、引用(reference)、函数(function)、结构体(`struct`)、共用体(`union`)、枚举(`enum`)、类(`class`) 
- **数据类型修饰符**：`short`, `long`, `signed`, `unsigned`

不同的数据类型也有不同的范围，这些范围可能因编译器而异。数据类型修饰符可作为前缀用于修改已有数据类型可以存储的数据大小或范围。以下是**32位GCC编译器**上的范围列表以及内存要求和格式指定符。

| Data Type                | Memory (bytes) | Range                  | Format Specifier |
|:------------------------:|:--------------:|:----------------------:|:----------------:|
| `short int`              | 2              | -2^15^ to 2^15^-1      | `%hd`            |
| `unsigned short int`     | 2              | 0 to 2^16^-1           | `%hu`            |
| `unsigned int`           | 4              | 0 to 2^32^-1           | `%u`             |
| `int`                    | 4              | -2^31^ to 2^31^-1      | `%d`             |
| `long int`               | 4              | -2^31^ to 2^31^-1      | `%ld`            |
| `unsigned long int`      | 4              | 0 to 2^32^-1           | `%lu`            |
| `long long int`          | 8              | -(2^63^) to 2^63^-1    | `%lld`           |
| `unsigned long long int` | 8              | 0 to 2^64^             | `%llu`           |
| `float`                  | 4              | 1.2E-38 to 3.4E+38     | `%f`             |
| `double`                 | 8              | 1.7E-308 to 1.7E+308   | `%lf`            |
| `long double`            | 16             | 3.4E-4932 to 1.1E+4932 | `%Lf`            |
| `signed char`            | 1              | -128 to 127            | `%c`             |
| `char`                   | 1              | 0 to 255               | `%c`             |

## 定义变量

当变量被定义时，它就会分配内存。分配的内存量取决于变量打算存储的数据类型。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/variable-memory-management-in-cpp.png" style="zoom:80%;" />

变量本质是一块内存区域的别名，用来存储程序运行时需要的数据。变量在使用前必须声明，声明时指定数据类型，例如

```cpp
int age; // Defining a variable
age = 25; // Initialize the variable
```

也可同时声明和赋值变量

```cpp
// Defining and initializing a variable
int age = 25;
char gender = 'F';
float f = 2.5;
double pi = 3.14;
bool isTrue = true; // or false
```

注意，`char`类型使用单引号，用于存储单个字符。

C/C++ 允许一次创建多个变量

```cpp
char a = 'A', b;
```

C++11 新增 `auto` 关键字，用于自动推断数据类型

```cpp
auto age = 18;
```

## 常量和宏

使用`const`关键字声明常量

```cpp
const int MAX_SIZE = 100;
```

也可以使用宏定义`#define`来表示常量

```cpp
#define PI 3.14
```

为了和变量区分开来，常量名通常**全部大写**。

## 命名空间

为防止命名冲突，C++引入了命名空间。以关键字 `namespace` 开头，后跟命名空间名称。

```cpp
#include <iostream>
using namespace std;
// first name space
namespace first
{
  void func(){cout << "Inside first space" << endl;}
}
// second name space
namespace second
{
  void func(){cout << "Inside second space" << endl;}
}
```

> 注意，大括号后没有分号。

为了调用带有命名空间的函数或变量，需要用域解析操作符 `::` 来指明要使用的命名空间

```cpp
int main (){
    // Calls function from first name space.
      first :: func();
    // Calls function from second name space.
      second :: func();

    return 0;
}
```

还可以采用 `using` 关键字引入整个命名空间，这样在使用命名空间时就可以不用加上前缀，如

```cpp
using std;
```

`using` 指令也可以用来指定命名空间中的特定项目：

```cpp
using std::cout;
```

随后的代码中，在使用 `cout` 时就可以不用加上命名空间名称作为前缀。

**不连续的命名空间**：可以创建两个具有相同名称的命名空间块。第二个命名空间块实际上只不过是第一个命名空间的延续。因此，我们可以将一个命名空间的各个组成部分分散在多个文件中，例如  `std` 命名空间，C++标准库中的函数和类通常都位于 `std`命名空间中。

# 运算符

在C++中，运算符根据其执行的操作类型分为6种类型。

## 算术运算符

算术运算符用于执行算术或数学运算。

| Operator | Description | Syntax  |
|:--------:|:-----------:|:-------:|
| `+`      | Plus        | `a + b` |
| `–`      | Minus       | `a – b` |
| `*`      | Multiply    | `a * b` |
| `/`      | Divide      | `a / b` |
| `%`      | Modulus     | `a % b` |
| `++`     | Increment   | `a++`   |
| `--`     | Decrement   | `a--`   |

`a++` 与 `++a` 都是增量运算符，但是，两者都略有不同：`a++` 在使用 `a` 之后才自增它的值，而 `++a` 会在使用 `a` 之前自增它的值。递减运算符也会发生类似的情况。

## 关系运算符

关系运算符用于比较两个数的值，返回 0 (false) 或 1 (true)

| Operator | Description              | Syntax   |
|:--------:|:------------------------:|:--------:|
| `<`      | Less than                | `a < b`  |
| `>`      | Greater than             | `a > b`  |
| `<=`     | Less than or equal to    | `a <= b` |
| `>=`     | Greater than or equal to | `a >= b` |
| `==`     | Equal to                 | `a == b` |
| `!=`     | Not equal to             | `a != b` |

浮点数的精度是有限的，因此在比较两个浮点数是否相等时，应该使用一个小的误差范围来判断，而不是直接使用 `==` 操作符。

## 逻辑运算符

逻辑运算符用于组合两个或两个以上条件，结果返回一个布尔值。

| Operator | Description | Syntax     |
|:--------:|:-----------:|:----------:|
| `&&`     | Logical AND | `a && b`   |
| `\|\|`   | Logical OR  | `a \|\| b` |
| `!`      | Logical NOT | `!a`       |

## 按位运算符

| Operator | Description              | Syntax   |
|:--------:|:------------------------:|:--------:|
| `&`      | Bitwise AND              | `a & b`  |
| `\|`     | Bitwise OR               | `a \| b` |
| `^`      | Bitwise XOR              | `a ^ b`  |
| `~`      | Bitwise First Complement | `~a`     |
| `<<`     | Bitwise Leftshift        | `a << b` |
| `>>`     | Bitwise Rightshilft      | `a >> b` |

注意：只有char和int数据类型可以与Bitwise运算符一起使用。

## 赋值运算符

| Operator | Description         | Syntax   |
|:--------:|:-------------------:|:--------:|
| `=`      | Simple Assignment   | `a = b`  |
| `+=`     | Plus and assign     | `a += b` |
| `-=`     | Minus and assign    | `a -= b` |
| `*=`     | Multiply and assign | `a *= b` |
| `/=`     | Divide and assign   | `a /= b` |
| `%=`     | Modulus and assign  | `a %= b` |

## 三元运算符

条件运算符也被称为三元运算符

```cpp
(expression) ? value_if_true : value_if_false;
```

`?` 运算符首先检查给定的条件，如果条件为真，则执行第一个表达式，否则执行第二个表达式。它是C++中if-else条件的替代方案。

```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 3, b = 4;

    // Conditional Operator
    int result = (a < b) ? b : a;
    cout << "The greatest number is " << result;

    return 0;
}
```

## sizeof

`sizeof` 是一个运算符，它以字节为单位计算变量或者数据类型的大小。

```cpp
#include <stdio.h>

int main(void) {
  int age = 37;
  printf("%ld\n", sizeof(age));
  printf("%ld", sizeof(int));
}
```

## 静态转换

注意，整数运算只能得到结果的整数部分

```c
int a = 1, b = 3;
cout << 1 / 3; // 0
```

因此需要强制转换类型

```c
cout << static_cast<float>(a) / b; // 0.3333333
cout << (float)a / b; // 0.3333333
```

静态转换不进行任何运行时类型检查，因此可能会导致运行时错误。

# 标准输入输出

### cstdio

`<cstdio>` 是 C++ 标准库中的一个头文件，它包含了 C 语言标准 I/O 库的 C++ 封装，主要用于文件的输入和输出操作。我们通常会使用 `scanf `和 `printf` 函数进行标准输入输出。

```cpp
#include <stdio.h>

int main() {
    int age;
    printf("Enter your age: ");

    // Reads an integer
    scanf("%d", &age);  

    // Prints the age
    printf("Age is: %d\n", age);  
    return 0;
}
```

`printf()` 格式控制符的完整形式如下：

```cpp
%[flag][width][.precision]type
```

type 表示输出类型，width 表示最小输出宽度，当输出结果的宽度不足时，默认会在左边补齐空格。precision 表示输出精度，也就是小数的位数。用于整数时，precision 表示最小输出宽度，整数的宽度不足时会在左边补 0，用于字符串时，precision 表示最大输出宽度。

### iostream

`<iostream>`库是 C++ 标准库中用于输入输出操作的头文件。其中定义了几个常用的流类和操作符： 

- `std::cin` 标准输入流
- `std::cout` 标准输出流
- `std::cerr` 非缓冲标准错误流
- `std::clog` 缓冲标准日志流。

同时，重载的输入输出运算符  `<<` 和 `>>`  可以自行分析所处理的数据类型，无需像使用 `scanf` 和 `printf` 函数那样给出格式控制字符串。

```cpp
#include <iostream>
using namespace std;

int main() {
    int age;
    cout << "Enter your age:";  // Output a label
    cin >> age;  // Taking input from user and store it in variable

    if (cin.fail()) {
        cerr << "Invalid input!" << endl;
    } else {
        cout << "Age entered: " << age << endl;  // Output the entered age
    }

    return 0;
}
```

其中 `endl` 用于在行末添加一个换行符 `\n` 。

注意，在使用 `cin` 将文本作为输入时，一旦遇到空格、制表符或换行符就会停止读取输入。使用标准库 `<string>` 中的 `getline`函数可以读取包含空格的整行输入。

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    string fullName;
    cout << "Enter your full name: ";
    getline(cin, fullName);
    cout << "Hello, " << fullName << "!" << endl;

    return 0;
}
```

# 流程控制

## 条件语句

### if-else

```cpp
#include <iostream>
using namespace std;

int main() {
    int age = 18;

    if (age < 13) {
        cout << "child";
    }
    else if (age >= 1 and age <= 18) {
        cout << "Growing stage";
    }
    else {
        cout << "adult";
    }
    return 0;
}
```

### switch-case

在C++中，当主要根据变量或表达式的值来评估多种情况时，就会使用 switch-case

```cpp
#include <iostream>
using namespace std; 

int main() 
{ 
    string gender = "Male"; 

    switch (gender) { 
    case "Male": 
        cout << "Gender is Male"; 
        break; 
    case "Female": 
        cout << "Gender is Female"; 
        break; 
    default: 
        cout << "Value can be Male or Female"; 
    } 
    return 0; 
}
```

> 注意`case`语句并没有花括号`{}`，如果遗漏了`break`，后续语句将全部执行，直到遇到`break`语句。

## 循环语句

### while

```cpp
int sum = 0;
int n = 1;
while (n <= 100) {
    sum = sum + n;
    n ++;
}
cout << sum; // 5050
```

### do-while

```cpp
int sum = 0;
int n = 1;
do {
    sum = sum + n;
    n ++;
} while (n <= 100);
cout << sum;  // 5050
```

### for

`for` 循环会先初始化计数器，然后，在每次循环前检测循环条件，在每次循环后更新计数器。

```cpp
for (int i = 0; i <= 100; i++) {
    sum = sum + i;
}
cout << sum;
```

还有一种基于范围的 `for` 循环，不需要条件和更新语句。它只能用于可迭代的对象，如向量、集合等。

```cpp
vector<int> v { 1, 2, 3, 4, 5};

// Using range based for loop to print vector
for (auto i: v) {
    cout << i << " ";
}
```

无限循环

```cpp
// This is an infinite for loop as the condition expression is blank
for (;;) {
    cout << "This loop will run forever.\n";
}
```

## 跳转语句

### break

`break` 语句用于完全终止循环或 `switch` 语句

```cpp
#include <iostream>
using namespace std;

int main()
{
    for (int i = 0; i < 5; i++) {
        // if i become 3 then break the loop and move to next statement out of loop
        if (i == 3) {
            break;
        }
        cout << i << endl;
    }
    // next statements
    return 0;
}
```

### continue

`continue` 语句用于跳过当前迭代并继续下一个迭代

```cpp
#include <iostream>
using namespace std;

int main()
{
    for (int i = 0; i < 5; i++) {
        // if i become 3 then skip the rest body of loop and move next iteration
        if (i == 3) {
            continue;
        }
        cout << i << endl;
    }
    return 0;
}
```

# 指针和引用

## 指针

在讲解指针之前，先来了解下内存（memory）的概念。内存是最重要的硬件之一，是用来存储数据的。内存的最小单位是字节（byte，1byte=8bit），每个字节都有一个唯一的地址（类似储物柜的编号），程序可以通过这个地址访问内存中的数据。

指针（Pointer）是一个存储内存地址的变量，它所存储的值是内存中某个位置的地址。使用指针的主要原因是操作方便、效率高。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/pointer-in-c.png" style="zoom:67%;" />

我们可以使用**引用运算符** `&` 获取内存中该变量的地址，将该地址赋给一个指针变量。声明指针变量的语法不同，需要在其名称前加上星号 `*` ，指针变量的类型是其指向的的变量类型。

```cpp
int age;
int* ptr = &age;  // 0x7ffeef7dcb9c
```

指针的**解引用运算符** `*` 可以获取和修改该地址指向的变量的值。注意，解引用运算符和声明指针时前面加的星号含义是不同的。

```cpp
#include <bits/stdc++.h>
using namespace std;
int main(){
    int var = 20;

    // declare pointer variable
    int* ptr;

    // note that data type of ptr and var must be same
    ptr = &var;

    // assign the address of a variable to a pointer
    cout << "Value at ptr = " << ptr << "\n";
    cout << "Value at var = " << var << "\n";
    cout << "Value at *ptr = " << *ptr << "\n";

    return 0;
}
```

**Output**

```
Value at ptr = 0x7ffe454c08cc
Value at var = 20
Value at *ptr = 20
```

## 指针运算

指针的 `+/-` 运算相当于移动指针，常用在数组中。

```c
int a[10], *ptr;
ptr = &a[0];
ptr += 2; 
```

<img title="" src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/Pointer-Addition.webp" width="449" data-align="center">

两指针的差值为指针相隔元素的个数。

## 指针的指针

由于指针也是变量，也有存储地址，同样也可以定义另一个指针指向该指针，称为指针的指针。

```cpp
int **ptr2 = &ptr;
printf("**ptr = %d", *ptr); // **ptr = 20
```

<img title="" src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/double-pointers-in-c.webp" width="539" data-align="center">

## 空指针

在指针变量声明的时候，如果没有确切的地址可以赋值，建议赋值为 NULL 。NULL 指针是一个定义在标准库中的值为零的常量。

```cpp
int *ptr1 = 0;
int *ptr2 = NULL;
```

在大多数的操作系统上，程序不允许访问地址为 0 的内存，因为该内存是操作系统保留的。如需检查一个空指针，可以使用 if 语句：

```cpp
if(ptr)
if(!ptr)
```

## 引用

引用（Reference）是 C++ 相对于C语言的又一个扩充。引用可以看做是同一份内存的别名。引用的声明方式类似于指针，在变量名前加上 `&` 。引用必须在定义的同时初始化，并且以后也不能再引用其它数据。

```cpp
#include <iostream>
using namespace std;

int main()
{
    int x = 10;

    // ref is a reference to x.
    int& ref = x;

    // Value of x is now changed to 20
    ref = 20;
    cout << "x = " << x << '\n';

    // Value of x is now changed to 30
    x = 30;
    cout << "ref = " << ref << '\n';

    return 0;
}
```

注意，引用不需要运算符 `*` 即可访问值。它们可以像普通变量一样使用。仅在声明时需要 `&` 运算符。

我们在循环中使用引用来修改所有元素

```cpp
#include <iostream>
#include <vector>

using namespace std;

// Driver code
int main()
{
    vector<int> vect{ 10, 20, 30, 40 };

    // We can modify elements if we use reference
    for (int& x : vect) {
        x = x + 5;
    }

    return 0;
}
```

# 派生数据类型

## 数组

### 声明和初始化

在 C++ 中，数组是一种数据结构，用于将相同数据类型存储在连续的内存位置。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/c-array-declaration.png"  width="407">

在 C++ 中，我们只需先指定数据类型，然后指定数组的名称及其大小即可声明数组。不允许使用变量定义数组大小，数组一旦创建后，大小就不可改变。

你可以在声明数组之后为其赋值

```cpp
int arr[5];

for (int i = 0; i < 5; i++) {
  arr[i] = i + 1;
}
```

也可以在声明数组的时候进行初始化

```cpp
int arr[5] = {1, 2, 3, 4, 5};
int arr[] = {1, 2, 3, 4, 5};
```

如果我们已经用值初始化了数组，但没有声明数组的长度，则数组的长度等于大括号内的元素数量。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/C-array-initialization.png" width="409">

我们也可以初始化部分数组，其余元素为默认值：整型都是`0`，浮点型是`0.0`，布尔型是`false`。

```c
int arr[5] = {2, 4, 8, 12, 16};
int arr[5] = {2, 4, 4}; 
int arr[] = {2, 4, 8, 12, 16};
```

在 C++ 中，我们没有像 Java 中那样的 length 函数来查找数组大小，但我们可以使用 `sizeof`  运算符计算数组的大小。

```cpp
// Length of an array
int n = sizeof(arr) / sizeof(arr[0]);
```

### 数组与指针

数组可以通过索引来访问、修改元素，索引从`0`开始。

```cpp
arr[0] = 1;
```

在 C++ 中，数组和指针彼此密切相关。数组名其实是一个指针常量，它存储的是数组中首个元素的地址，即 `arr` 和 `&arr[0]` 是等价的。因此，可以像普通指针一样使用数组名。

<img title="" src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/random_access_in_array.png" width="538" data-align="center">

```cpp
// C++ Program to Illustrate that Array Name is a Pointer
// that Points to First Element of the Array
#include <iostream>
using namespace std;

int main()
{
    // Defining an array
    int arr[] = { 1, 2, 3, 4 };

    // Define a pointer
    int* ptr = arr;

    // Printing address of the arrary using array name
    cout << "Memory address of arr: " << &arr << endl;

    // Printing address of the array using ptr
    cout << "Memory address of arr: " << ptr << endl;

    return 0;
}
```

**Output**

```
Memory address of arr: 0x7fff2f2cabb0
Memory address of arr: 0x7fff2f2cabb0
```

上例中，我们能够将 `arr` 分配给 `ptr`，因为 `arr` 也是一个指针。之后，我们使用引用运算符 `&` 打印 `arr` 的内存地址，并打印存储在指针 `ptr` 中的地址，我们可以看到 `arr` 和 `ptr`，它们都存储相同的内存地址。

现在，我们可以仅使用数组名称访问数组的元素，即 `*(arr + i)` 等价于 `arr[i]`

```cpp
int arr[] = {2, 4, 8, 12, 16};
cout << "Second element: " << *(arr + 1) << endl; 
```

### 多维数组

使用最广泛的多维数组是 2D 数组和 3D 数组。这些数组通常以行和列的形式表示。

<img title="" src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/dimensions_of_array.png" width="482" data-align="center">

```cpp
int arr[3][4] = {
    { 1, 2, 3, 4 },
    { 5, 6 },
    { 7, 8, 9 }
};

int a[2][4] = {1, 2, 3, 4, 5, 6, 7, 8}
```

访问二维数组的某个元素需要使用行和列两个索引： `array[row][col]`。

二维数组由多个一位数组组成，数组名 `arr` 代表二维数组首地址，也代表第0行首地址，`arr + 1` 代表第1行首地址，依次类推。一般 `arr[i] + j` 代表第`i`行第`j`列元素地址，即 `&arr[i][j]`。

## 字符串

C 语言中字符串并不是一种基本数据，实际上是一个以 `\0` 结尾的字符数组。以下是 C 中定义的字符串的内存表示：

<img title="" src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/strings-and-pointers.webp" alt="" width="428" data-align="center">

可以像初始化一个普通的数组那样初始化一个字符串，或者使用更加方便的字符串常量表示

```cpp
char str[] = { 'H', 'e', 'l', 'l', 'o', '\0' };
char* str = "Hello";
```

> 字符串使用**双引号**表示时，编译器会在初始化时自动追加一个 `\0`

C++ 除了可以使用C风格的字符串，还可以使用标准库 `<string>` 中的 `std::string` 类。它是对 C 风格字符串的封装，提供了更安全、更易用的字符串操作功能。

```cpp
#include <string>
using namespace std;

string str = "Hello, world!";
string str("Hello, world!");
```

string 字符串也可以像C风格的字符串一样按照下标来访问其中的每一个字符，起始下标仍是从 0 开始。

```cpp
cout << "First character: " << str[0];  
```

支持使用 `+` 或 `+=` 运算符来直接拼接 string 字符串

```cpp
string str1 = "Hello, ";
string str2 = "World!";
string result = str1 + str2;
```

## 结构体

### 定义和使用

结构体是一种自定义的数据类型，用于创建复杂的数据结构，他可以包含多个不同类型的成员。

```cpp
struct [tag] { 
    member-list;
    member-list;
    member-list;
    ...
} [variable-list];
```

可以先定义结构体类型，再声明结构体变量；

```cpp
struct Student {
    int age;
    char name[8], sex;
    float score[3];
};

struct Student Alice, Bob;
```

也可以直接声明变量，同时省略结构体名

```cpp
struct {
    int age;
    char name[8], sex;
    float score[3];
} Alice, Bob;
```

可以在声明的时候初始化一个结构体：

```cpp
struct Student Alice = { 18, "Alice", 'F' };
```

结构体通过点语法来访问和修改成员：

```cpp
printf("%s, age %d", Alice.name, Alice.age);
```

### 内嵌结构体

```c
struct Student {
    int age;
    char name[8], sex;
    struct {
        int year, month, day;
    } birthday;
    float score[3];
} Alice, Bob;
```

初始化

```c
// age, name, sex, year, month, day, score[0], score[1], score[2]
Alice = {16, 'Alice', 'F', 2024, 6, 16, 83.6, 88.5, 90}
```

### 结构体指针

像原始类型一样，我们可以有指向结构体的指针。

```cpp
struct Student* ptr = &Alice;
```

结构体指针必须使用 `->` 运算符访问结构体的成员：

```cpp
ptr->age = 20;
```

### 结构体数组

与其他原始数据类型一样，我们可以创建一个结构体数组。

```c
struct Student stu[5];
```

## 枚举

枚举 （Enumerated ） 是用户定义的数据类型，可以为其分配一些有限的值。这些值由用户在声明时定义。

```cpp
#include <iostream> 
using namespace std; 

int main() 
{ 
    // Defining enum Gender 
    enum Gender { MALE, FEMALE }; 

    // Creating Gender type variable 
    Gender gender = MALE; 

    switch (gender) { 
    case MALE: 
        cout << "Gender is Male"; 
        break; 
    case FEMALE: 
        cout << "Gender is Female"; 
        break; 
    default: 
        cout << "Value can be Male or Female"; 
    } 
    return 0; 
}
```

在 C/C++ 中，将枚举值作为 `int` 连续值来处理。默认情况下，第一个名称的值为 0，第二个名称的值为 1，第三个名称的值为 2，以此类推。但是，您也可以给名称赋予一个特殊的值。

## 类型别名

C/C++ 语言允许使用 `typeof` 关键字对数据类型赋予一个新名字

```cpp
typedef char[] STRING;
STRING name = "Alice";
```

使用 `typedef`，我们可以简化处理结构体时的代码。声明变量的时候不需要 struct 关键字

```cpp
typedef struct {
        int year, month, day;
} DATE;

DATE birthday = { 2024, 6, 16 };
```

C++11 引入 using 关键字为现有类型定义别名

```cpp
using char[] = STRING;
```

# 函数

函数允许用户将程序划分为多个模块，每个模块执行特定任务。

## 定义和调用

函数定义包括返回类型、函数名、参数列表和函数体。

```cpp
#include <iostream>
using namespace std;

int max(int x, int y)
{
    if (x > y)
        return x;
    else
        return y;
}

// main function that doesn't receive any parameter and returns integer
int main()
{
    int a = 10, b = 20;

    // Calling above function to find max of 'a' and 'b'
    int m = max(a, b);

    cout << "m is " << m;
    return 0;
}
```

上例中 `int` 为函数返回值类型，如无返回值，以 `void` 类型表示，函数返回值用 `return` 显示给出。如无返回值，我们仍然可以使用 `return` 语句终止函数。

## 前向声明

由于 C++ 程序按照代码的书写顺序执行，因此在函数调用之前必须先声明。函数声明告诉编译器参数的数量、参数的数据类型以及返回函数的类型。

```cpp
void hello();
int main(){
    hello(); 
}
void hello(){
    printf("Hello world!\n")
}
```

在函数声明的时候可以只写参数类型，省略参数名

```cpp
int sum(int a, int b);  // Function declaration with parameter names
int max(int , int);     // Function declaration without parameter names
```

## 参数传递

函数在定义时预期接收的参数称为**形式参数**，函数调用时实际传入的参数称为**实际参数**。C++ 支持三种参数传递方法：

- **值传递**：函数调用的时候会把实参的值拷贝一份传给函数内部，并不会影响到函数外部。C++默认方法。
- **指针传递**：函数调用的时候会把实参的地址传递给函数，在函数内，该地址用于访问调用中要用到的实际参数。这意味着，修改形式参数会影响实际参数。
- **引用传递**：函数调用的时候传递实参的引用。在函数内，对引用的操作会直接作用于实际参数。

```cpp
#include <stdio.h>

// Pass by Value
void swap1(int a, int b){
    int temp = a;
    a = b;
    b = temp;
}

// Pass by Pointer
void swap2(int* a, int* b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Pass by Reference
void swap3(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int main(){
    int x = 10, y = 20;
    printf("Before swap: x = %d, y = %d\n", x, y);

    swap1(x, y); // a = 20, b = 10
    printf("Pass by Value: x = %d, y = %d\n", x, y);

    swap2(&x, &y); // *a = 20, *b = 10
    printf("Pass by Pointer: x = %d, y = %d\n", x, y); 

    swap3(x, y); // a = 10, b = 20
    printf("Pass by Reference: x = %d, y = %d\n", x, y); 
};
```

**Output**

```
Before swap: x = 10, y = 20
Pass by Value: x = 10, y = 20
Pass by Pointer: x = 20, y = 10
Pass by Reference: x = 10, y = 20
```

在 C++ 中，数组名实质上是指针，数组参数在传递的函数中都被视为指针。有两种数组参数传递方式：

- 将数组名称作为指针参数传递， 如 `int* arr`
- 函数使用简单的数组声明接受数组，如 `int arr[]`

```cpp
#include <iostream>
using namespace std;

void printMinSized(int arr[5]);
void printMinUnsized(int arr[]);
void printMinPointer(int* arr);

// driver code
int main()
{
    int ar[5] = { 30, 10, 20, 40, 50 };
    printMin(ar); // passing array to function
}

// passing array as a sized array argument
void printMinUnsized(int arr[5], int n)
{
    int min = arr[0];
    for (int i = 0; i < n; i++) {
        if (min > arr[i]) {
            min = arr[i];
        }
    }
    cout << "Minimum element is: " << min << "\n";
}


// passing array as an unsized array argument
void printMinUnsized(int arr[], int n)
{
    int min = arr[0];
    for (int i = 0; i < n; i++) {
        if (min > *(arr + i)) {
            min = *(arr + i);
        }
    }
    cout << "Minimum element is: " << min << "\n";
}


// Passing array as a pointer argument
void printMinPointer(int* ptr, int n)
{
    int min = ptr[0];
    for (int i = 0; i < n; i++) {
        if (min > ptr[i]) {
            min = ptr[i];
        }
    }
    cout << "Minimum element is: " << min << "\n";
}
```

## 默认参数

在C++ 中，定义函数时可以给形参指定一个默认的值，默认参数只能放在形参列表的最后。

```cpp
#include <iostream>
using namespace std;

// Function with default height 'h' argument
double calcArea(double l, double h = 10.0) {
    return l * h;
}

int main() {

      // Uses default height
    cout << calcArea(5) << endl;

      // Uses custom height
    cout << calcArea(5, 7);
    return 0;
}
```

如果函数是单独声明和定义的，则参数的默认值必须在声明中，且声明默认参数后，无法在函数定义中修改它们。

```cpp
// Declaration with default argument
void func(int x = 10);

// Definition without default argument
void func(int x) {
    cout << "Value: " << x << endl;
}
```

## 命令行参数

在 C++ 程序中，命令行参数是使用 `main()` 函数参数来接收的。

```cpp
int main (int argc, char* argv[])
```

其中，`argc` 是指传入参数的个数，`argv[]` 是一个指针数组，指向传递给程序的每个参数。

```cpp
#include <stdio.h>

int main(int argc, char* argv[]) {

    // Printing the coundt of arguments
      printf("The value of argc is %d\n", argc);

    // Prining each argument
    for (int i = 0; i < argc; i++) {
        printf("%s \n", argv[i]);
    }

    return 0;
}
```

应当指出的是，`argv[0]` 存储程序的名称，如果没有提供任何参数，`argc` 将为 1。多个命令行参数之间用空格分隔，但是如果参数本身带有空格，那么传递参数的时候应把参数放置在双引号或单引号内部。

## 返回指针的函数

```cpp
#include <iostream>
using namespace std;

int* createArray(int size) {
    int* arr = new int[size]; // Dynamically allocate memory for an array
    for (int i = 0; i < size; ++i) {
        arr[i] = i * 10; // Initialize array elements
    }
    return arr; // Return the pointer to the array
}

int main() {
    int size = 10;
    int* myArray = createArray(size); // Function returns a pointer to the array

    // Print the array
    for (int i = 0; i < size; ++i) {
        cout << myArray[i] << " ";
    }
    cout << endl;

    delete[] myArray; // Don't forget to free the allocated memory
    return 0;
}
```

## 递归函数

函数直接或间接调用自身，直到满足给定条件。

```c
// calculate the sum of first N natural numbers using recursion
#include <stdio.h>

int nSum(int n)
{
    // base condition to terminate the recursion when N = 0
    if (n == 0) {
        return 0;
    }

    // recursive case / recursive call
    int res = n + nSum(n - 1);

    return res;
}

int main()
{
    int n = 5;

    // calling the function
    int sum = nSum(n);

    printf("Sum of First %d Natural Numbers: %d", n, sum);
    return 0;
}
```

## 函数指针

在C语言中，函数其实也可以看作一种数据类型，函数的类型其实就是它的返回值和参数列表。我们可以定义一个函数指针

```cpp
void (*ptr)(int*, int*);
ptr = swap;
```

这里的 `ptr` 就是一个函数指针， `*ptr` 就代表该函数。

```cpp
*ptr(&x, &y);
```

函数指针的应用也是非常广泛的，比如在实现一个回调函数的时候，就可以把函数指针作为参数传递给另一个函数，然后在这个函数里使用函数指针来使用函数，这种方式可以让代码更加灵活。

这里，我们有 c++ 示例，用于访问数组中的元素

```cpp
#include <iostream>
using namespace std;

// Function declarations
int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

int main() {
    // Declare and initialize an array of function pointers
    int (*funcArray[3])(int, int) = { add, subtract, multiply };

    // Variables to use as function parameters
    int x = 2, y = 3;

    // Access and call the functions using the array of function pointers
    cout << "Add: " << funcArray[0](x, y) << endl;       // Calls add(10, 5)
    cout << "Subtract: " << funcArray[1](x, y) << endl;  // Calls subtract(10, 5)
    cout << "Multiply: " << funcArray[2](x, y) << endl;  // Calls multiply(10, 5)

    return 0;
}
```

## 函数重载

C++ 允许多个函数拥有相同的名字，只要它们的参数列表不同就可以，这就是函数的重载（Function Overloading）。调用时根据实参和形参的匹配选择最佳函数

```cpp
#include <iostream>
using namespace std;

int add(int a, int b) { return a + b; }
int add(int a, int b, int c) { return a + b + c; }

int main(void)
{
    cout << add(10, 20) << endl;
    cout << add(12, 20, 23);
    return 0;
}
```

## 模板函数

C++ 支持将数据类型作为参数传递，这样我们就不需要为不同的数据类型编写相同的代码。这个函数就称为函数模板（Function Template）。

```cpp
template <typename T>
```

`template` 和 `typename` 是定义函数模板的关键字，它后面尖括号里的 `T` 是类型占位符。`typename T` 告诉编译器：字母 T 将在接下来的函数里代表一种不确定的数据类型。

```cpp
#include <iostream>
using namespace std;

// One function works for all data types. This would work
// even for user defined types if operator '>' is overloaded
template <typename T> 
T myMax(T x, T y)
{
    return (x > y) ? x : y;
}

int main(){
    // Call myMax for int
    cout << myMax<int>(3, 7) << endl;
    // call myMax for double
    cout << myMax<double>(3.0, 7.0) << endl;
    // call myMax for char
    cout << myMax<char>('g', 'e') << endl;

    return 0;
}
```

就像普通参数一样，我们可以将多个数据类型作为参数传递给模板，也可以为模板指定默认参数

```cpp
// C++ Program to implement use of template
#include <iostream>
using namespace std;

template <class T, class U = char> 
void func(T x, U y) { cout << "Function Called" << endl; }

int main(){
    // This will call func<char, char>
    func<char>('a', 'b');
}
```

## lambda 表达式

C++ 11 引入了 lambda 表达式。Lambda 表达式可以像对象一样使用，比如可以将它们赋给变量和作为参数传递，还可以像函数一样对其求值。

```cpp
[capture] (parameters) -> return-type{body}
```

通常，lambda 表达式中的 retur-type 由编译器本身计算，我们不需要显式指定它。

```cpp
[](int a, int b){ return (a < b) ? b : a ; }
```

`[]` 方括号用于向编译器表明当前是一个 lambda 表达式，其不能被省略。在方括号内部，可以注明当前 lambda 函数体中可以使用的外部变量。

| Syntax    | Description              |
| --------- | ------------------------ |
| `[]`      | 空方括号表示不导入任何外部变量          |
| `[=]`     | 表示以值传递的方式导入所有外部变量        |
| `[&]`     | 表示以引用传递的方式导入所有外部变量       |
| `[x, &y]` | `x` 以传值方式导入，`y` 以引用方式导入  |
| `[&, x]`  | `x` 以值传递方式导入，其余变量以引用方式导入 |
| `[=, &x]` | `x` 以引用方式导入，其余变量以值传递方式导入 |

小括号可以接收外部传递的多个参数，和普通函数不同的是，如果不需要传递参数，可以连同 `()` 小括号一起省略可变参数

# 动态内存管理

在C/C++中内存可分为两种类型：

- **栈内存**：（stack）一般用来存储局部变量和函数的参数，它的分配和释放由编译器自动完成；
- **堆内存**：（heap）堆内存比栈内存要大得多，但是它的分配和释放需要手动完成。是所有程序共同拥有的自由内存。

<img title="" src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/Memory-Layout-of-C-Program.webp" alt="Memory-Layout-of-C-Program" width="520" data-align="center">

C标准库 `<cstdlib>` 为内存的分配和管理提供了四个函数

| Function                                    | Description                           |
| ------------------------------------------- | ------------------------------------- |
| `void* calloc(int num, int size)`           | 分配一个 num 个元素的数组，每个元素的大小为 size 字节，返回指针 |
| `void free(void *address)`                  | 释放 address 所指向的内存块                    |
| `void* malloc(int num)`                     | 配一个 num 字节的内存块，返回指针                   |
| `void* realloc(void *address, int newsize)` | 重新分配内存，把内存扩展到 newsize，返回指针            |

> 注意：void\* 类型表示未确定类型的指针。C/C++ 规定 void\* 类型可以通过类型转换强制转换为任何其它类型的指针。
> 在堆区分配内存，必须手动释放，否则只能等到程序运行结束由操作系统回收。

```cpp
int* ptr = (int*)malloc(sizeof(int) * 10);
free(ptr);
```

C++ 又新增了两个关键字 `new` 和`delete` 来更加简单的分配内存。`new` 操作符会根据后面的数据类型来推断所需空间的大小。`delete` 用来释放内存。

```cpp
int* ptr = new int;
delete ptr;
```

如果希望使用一组连续的内存，可以使用 `new` 来分配，使用 `delete[]` 来释放内存。

```cpp
int* ptr = new int[10];
delete[] ptr;
```

在实际开发中，`new` 和 `delete` 往往成对出现，以保证及时删除不再使用的对象，防止无用内存堆积。

# 面向对象

C++ 是一种功能强大的高级编程语言，它在 C 语言的基础上增加了面向对象编程的特性。

## 类与对象

类（class）是一种用户定义的数据类型，它包含自己的属性和方法，可以通过创建该类的实例来访问和使用它们。

```cpp
#include <iostream>
#include <string>

using namespace std;

// Define a class named 'Person'
class Person {
  public:
    // Data members
    string name;
    int age;

    // Member Functions
    void introduce();

    // Parameterized Constructor
    Person(string name, int age) {
      this->name = name;
      this->age = age;
    }
};

void Person::introduce() {
    cout << "Hi, my name is " << name << " and I am "
         << age << " years old." << endl;
}
```

上例中 `Person`是类名称，类名的首字母一般大写。`{ }`内部是类所包含的属性和方法，它们统称为类的成员（Member）。定义类外面的方法必须使用域解析符 `::`指明当前函数所属的类。

创建对象以后，可以使用点号`.`来访问属性和方法。

```cpp
int main()
{
    // Create an object of the Person class
    Person person("Alice", "18");

    // accessing data members
    cout << "Name: " << person.name << endl;

    // Call the introduce member method
    person.introduce();

    return 0;
}
```

在 C++ 中，指向类的指针与指向结构的指针类似

```cpp
int main()
{
    Person* ptr = &person;

    // accessing data members
    cout << "Name: " << ptr->name << endl;

    // Call the introduce member method
    ptr->introduce();

    return 0;
}
```

## 封装

C++ 通过访问修饰符 `public, protected, private` 来控制类属性和方法的访问权限。

- `public`：类的所有成员都是公开的，可以在任何地方访问。
- `protected`：类成员可以被类及其子类访问。
- `private`：类成员只能在类的内部访问，默认修饰符。

```cpp
#include <iostream>
using namespace std;

class Circle {

    // private member
    float area;
    float radius;

public:
    void getRadius()
    {
        cout << "Enter radius\n";
        cin >> radius;
    }
    void findArea()
    {
        area = 3.14 * radius * radius;
        cout << "Area of circle=" << area;
    }
};
int main()
{
    // creating instance(object) of class
    Circle cir;
    cir.getRadius(); 
    cir.findArea(); 
}
```

## 构造函数

构造函数（Constructor）是类的一种特殊的成员函数，它会在每次创建类的新对象时执行。构造函数的名称与类的名称是完全相同的，并且不会返回任何类型，并且不需要 `void` 声明。一个类可以有多个重载的构造函数。如果用户没有定义，编译器会自动生成一个默认的构造函数。

```cpp
#include <iostream>
#include <string.h>
using namespace std;

class Person {
    string name;
    int age;
  public:
    // Declaration of parameterized constructor
    Person(string, int);
    void display();
};

// Parameterized constructor outside class
Person::Person(string name, int age) {
    this->name = name;
    this->age = age;
}
```

## 析构函数

析构函数（Destructor）是类的一种特殊的成员函数，它会在每次删除所创建的对象时执行。析构函数的名称与类的名称是完全相同的，只是在前面加了个波浪号（`~`）作为前缀，它不会返回任何值，也不能带有任何参数。如果用户没有定义，编译器会自动生成一个默认的析构函数。

C++ 中的 `new` 和 `delete` 分别用来分配和释放内存，它们与C语言中 `malloc()`、`free()` 最大的一个不同之处在于：用 `new` 分配内存时会调用构造函数，用 `delete` 释放内存时会调用析构函数。

```cpp
#include <iostream>
using namespace std;

class Person {
public:
    // User-Defined Constructor
    Person() { cout << "\n Constructor executed"; }

    // User-Defined Destructor
    ~Person() { cout << "\nDestructor executed"; }
};
```

## this 指针

`this` 是 C++ 类中的隐式形参，它是指向当前对象的指针，通过它可以访问当前对象的所有成员，包括 `public, protected, private` 权限的。

```cpp
class Person {
  public:
    string name;
    int age;

    // Parameterized Constructor
    void setValue(string name, int age) {
      this->name = name;
      this->age = age;
    }
};
```

注意，`this` 是一个指针，要用 `->` 来访问属性和方法。本例中类方法的参数和属性重名，只能通过 `this` 区分。

## 友元函数

友元函数不是类的成员函数，通过在类内使用关键字 `friend` 来声明，可以访问当前类中的所有成员，包括 `public, protected, private` 权限的。

```cpp
#include <iostream>

class Person {
private:
    int age;

public:
    friend void displayAge(Person& person);
};

void displayAge(Person& person) {
    std::cout << "Age: " << person.age << std::endl;
}  
```

## 继承

继承（Inheritance）允许一个派生类继承基类的属性和方法。 

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/CPP/inheritance-660x454.png" alt="Inheritance in C++ with Example" width="379" data-align="center">

在C++中，有3种继承模式：`public, protected, private`  ，用来指明基类成员在派生类中的最高访问权限。

```cpp
#include <iostream>
using namespace std;

// base class
class Animal {
public:
    void sleep(){};
};

// sub class
class Dog : public Animal {
public:
    void sound() {
        cout << "Dog barks" << endl;
    }
};
```

C++ 中基类的构造函数不能被继承，编译器会自动按照继承顺序调用基类的默认构造函数，这意味着将首先调用基类构造函数，然后调用派生类构造函数。基类的参数化构造函数则必须在派生类构造函数的初始化列表中显式调用。

```cpp
class Parent {
public:
    // base class constructor
    Parent() { cout << "Inside base class" << endl; }
};

// sub class
class Child : public Parent {
public:
    // sub class constructor
    Child() { cout << "Inside sub class" << endl; }
};
```

**Output**

```
Inside base class
Inside sub class
```

## 函数覆盖

在C++中，派生类会覆盖基类中相同类型的属性和方法。

```cpp
// C++ program for function overriding
#include <iostream>
using namespace std;

//  base class declaration.
class Animal {
public:
    string name = "animal";
    void sound(){
        cout << "Animal makes a sound" << endl;
    }
};

// inheriting Animal class.
class Dog : public Animal {
public:
    string name = "dog";
    // Override the sound method
    void sound(){
        cout << "Dog barks" << endl;
    }
};

int main(){
    Dog dog = Dog(); 
    cout << dog.name << endl;
    dog.sound();

    return 0;
}
```

## 运算符重载

运算符重载（Operator Overloading）允许改变运算符的行为，以适应用户定义的类型。`operator` 关键字专门用于定义重载运算符的函数。虽然运算符重载所实现的功能完全可以用函数替代，但运算符重载使得程序的书写更加人性化，易于阅读。

```cpp
#include <stdio.h>

class Complex {
private:
    int real, imag;

public:
    Complex(int r = 0, int i = 0)
    {
        real = r;
        imag = i;
    }

    // This is automatically called when '+' is used with between two Complex objects
    Complex operator+(Complex const& obj)
    {
        Complex res;
        res.real = real + obj.real;
        res.imag = imag + obj.imag;
        return res;
    }
    void print() { printf("%d + i%d\n", real, imag); }
};

// Driver code
int main()
{
    Complex c1(10, 5), c2(2, 4);

    // An example call to "operator+"
    Complex c3 = c1 + c2;
    c3.print();
}
```

## 虚函数

虚函数允许我们使用基类的指针或引用调用任何派生类的方法，甚至可以在不知道派生类对象类型的情况下调用。虚拟函数是使用关键字 `virtual` 在基类中声明的方法，并在派生类中覆盖。

```cpp
// C++ program for virtual function overriding
#include <iostream>
using namespace std;

class Base {
public:
    string attr = "base";
    virtual void print(){ cout << "print base class" << endl; }
    void show() { cout << "show base class" << endl; }
};

class Derived : public Base {
public:
    string attr = "derived";
    // print () is already virtual function in
    // derived class, we could also declared as
    // virtual void print () explicitly
    void print() { cout << "print derived class" << endl; }

    void show() { cout << "show derived class" << endl; }
};

// Driver code
int main()
{
    Base* ptr; 
    Derived d;

    // Point base class pointer to derived class object
    ptr = &d;

    // Virtual function, binded at
    // runtime (Runtime polymorphism)
    ptr->print();

    // Non-virtual function, binded
    // at compile time
    ptr->show();

    // Accessing base class attributes
    cout << "Base class attribute: " << ptr->attr << endl;

    return 0;
}
```

**Output**

```cpp
print derived class
show base class
Base class attribute: base
```

## 类模板

正如我们定义函数模板一样，我们也可以定义类模板。类模板在类定义独立于数据类型的内容时非常有用。

```cpp
// C++ Program to implement
// template Array class
#include <iostream>
using namespace std;

template <typename T> 
class Array {
private:
    T* ptr;
    int size;

public:
    Array(T arr[], int s);
    void print();
};

template <typename T> Array<T>::Array(T arr[], int s)
{
    ptr = new T[s];
    size = s;
    for (int i = 0; i < size; i++)
        ptr[i] = arr[i];
}

template <typename T> void Array<T>::print()
{
    for (int i = 0; i < size; i++)
        cout << " " << *(ptr + i);
    cout << endl;
}

int main()
{
    int arr[5] = { 1, 2, 3, 4, 5 };
    Array<int> a(arr, 5);
    a.print();
    return 0;
}
```

# 异常处理

C++ 异常处理涉及到三个关键字：**try、catch、throw**。

```cpp
try {         
     // Code that might throw an exception
     throw SomeExceptionType("Error message");
 } 
catch( ExceptionName e1 )  {   
     // catch block catches the exception that is thrown from try block
 } 
catch( ExceptionName e2 )  {   
     // catch block catches the exception that is thrown from try block
 } 
 catch( ... )  {   
     // catch all exception
 } 
```

当异常发生时，会立即终止当前函数并开始查找匹配的 catch 块来处理引发的异常。C++ 语言本身以及标准库中的函数抛出的异常，都是 exception 的子类，称为标准异常（Standard Exception）。

```cpp
#include <iostream>
#include <stdexcept>
using namespace std;

int main(){
    // try block
    try {
        int numerator = 10;
        int denominator = 0;
        int res;

        // check if denominator is 0 then throw runtime error.
        if (denominator == 0) {
            throw runtime_error(
                "Division by zero not allowed!");
        }

        // calculate result if no exception occurs
        res = numerator / denominator;
        // printing result after division
        cout << "Result after division: " << res << endl;
    }
    // catch block to catch the thrown exception
    catch (const exception& e) {
        // print the exception
        cout << "Exception " << e.what() << endl;
    }

    return 0;
}
```

**Output**

```
Exception Division by zero not allowed!
```

# 文件和流

## cstdio

C 头文件 `<cstdio>` 包含了一个专门用来处理文件的数据类型 `FILE`。每当程序执行文件打开操作，其返回值是一个 `FILE` 类型的指针，所有关于文件的操作都要通过此指针来进行。

```cpp
FILE* fopen (filename, mode)
```

```cpp
// C program to Open a File,
// Write in it, And Close the File
#include <cstdio>
#include <cstring>

int main() {
    // Open file in write mode
    FILE* file = fopen("output.txt", "w");  

    if (file == NULL) {
        printf("The file is not opened. The program will exit now\n");
        return 1;
    }

    fprintf(file, "Hello, this is a test file.\n");
    printf("Data written successfully.\n");

    // Close the file
    fclose(file);  
    return 0;
}
```

## fstream

在 C++ 中，`<fstream>` 是标准库中用于文件输入输出操作的类。它提供了一种方便的方式来读写文件。

```cpp
// C++ program to Open a File,
// Write in it, And Close the File

#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    // Creation of fstream class object
    fstream fio;
    string line;

    // by default openmode = ios::in|ios::out mode
    // Automatically overwrites the content of file
    fio.open("sample.txt", ios::trunc | ios::out | ios::in);

    // Execute a loop If file successfully Opened
    while (fio) {

        // Read a Line from standard input
        getline(cin, line);

        // Press -1 to exit
        if (line == "-1")
            break;

        // Write line in file
        fio << line << endl;
    }

    // Execute a loop until EOF (End of File)
    // point read pointer at beginning of file
    fio.seekg(0, ios::beg);

    while (fio) {

        // Read a Line from File
        getline(fio, line);

        // Print line in Console
        cout << line << endl;
    }

    // Close the file
    fio.close();

    return 0;
}
```

上例中 `getline(fio, line)` 函数用于从文件流中读取整行字符串。EOF (End of File) 是在 `iostream` 类中定义的一个整型常量，值为 -1。

打开文件的常见的模式有：

- `std::ios::in`：以输入模式打开文件。
- `std::ios::out`：以输出模式打开文件。
- `std::ios::app`：以追加模式打开文件。
- `std::ios::ate`：打开文件并定位到文件末尾。
- `std::ios::trunc`：打开文件并截断文件，即清空文件内容。

# 预处理指令

预处理器是在实际编译开始之前处理源代码的程序。所有预处理的指令，必须独占一行，并以 `#` 开始，**末尾不加分号**。

## 宏定义

`#define` 预处理指令用于创建符号常量，该符号常量通常称为宏。关键字 `#define` 和 `#undef` 用于在 C 中创建和删除宏。

```cpp
#define PI 3.14
#define PR printf("\n") 
```

我们还可以将参数传递给宏。这些宏的工作方式类似于函数。

```cpp
#define AREA(r) (PI * (r) * (r))
```

参数加括号是因为宏定义只是简单的替换。

ANSI C定义了许多宏，它们的名字的前后有两个下划线作为标识，例如：

- `__LINE__` 代表源代码文件中的当前行
- `__FILE__` 代表文件的名字
- `__DATE__` 表示编译日期，格式为 `mmm dd yyyy`
- `__TIME__` 表示编译时间，格式为 `hh:mm:ss`

## 头文件

一个项目往往包含多个源码文件，通常的做法是创建一个 **头文件**，然后使用预处理器指令 `#include` 加载进入当前文件。头文件用于放置对应源文件里面函数声明等内容，不包括函数定义和实现。

有两种形式：

```cpp
#include <stdio.h>
#include "myfile.h"
```

尖括号 `<>` 是用来引入标准库中的头文件，双引号是用来引入用户自定义的头文件。

C++头文件不是以 `.h` 做扩展名，C语言中的标准头文件如 `math.h, stdio.h` 在C++中被命名为 `cmath, cstdio`。

例如，我们想在主程序中引入一个函数 `add`

```cpp
// add.cpp
int add(iint x, int y);

int add(int x, int y) {
    return x + y;
}
```

需要创建一个和源文件同名的头文件 `add.h` ，里边只放入函数声明

```cpp
// add.h
int add(int x, int y);
```

然后就可以在主文件引入使用了

```cpp
// main.cpp
#include <iostream>
#include "add.h"

int main() {
    std::cout << "The sum of 3 and 4 is: " << add(3, 4); 
    return 0;
}
```

编译多个文件组成的程序时，需要在命令行中列出它们：

```sh
g++ -o main main.cpp add.cpp
```

如果编译的文件分别位于不同的目录下，编译时可以通过 `-I` 选项来指明头文件搜索路径

```sh
g++ -o main -I/source/includes main.cpp
```

## 条件编译

头文件里面还可以加载其他头文件，因此有可能产生重复加载，这将产生错误。为了防止这种情况，标准的做法是每个头文件都包含头文件包含。预处理指令包括：`#if, #ifdef, #ifndef, else, #elif, #endif`

```cpp
#ifndef HEADER_FILE_H
#define HEADER_FILE_H
// the entire header file file
#endif
```

这时，当头文件被包含时，预处理器会检查 HEADER_FILE_H 是否已经被定义过。如果该头文件之前已经被包含了，那么预处理器会跳过文件的整个内容。

所有的头文件都应该有头文件保护。但根据惯例，它被设置为头文件的完整文件名，以大写字母键入，使用下划线表示空格或标点。标准库头文件也使用头文件保护。

现代编译器使用更简单的 `#pragma`  请求编译器保护头文件

```cpp
#pragma once 
// your code here
```

目前对 `#pragma once` 的支持是相当普遍的，由于不是由C++标准定义的，因此一些编译器可能不会实现它。

# 模块管理

## 模块

从 C++20 开始，C++ 引入了模块（Modules），并在 C++23 中进一步完善了对标准库模块的支持。模块提供了一种更高效、更安全的方式来导入标准库。模块只编译一次，后续导入时直接使用编译好的二进制接口。

```cpp
import std;

int main() {
    std::cout << "Hello, C++23 Modules!\n";
    return 0;
}
```

或者导入标准库的特定部分：

```cpp
import std.core;
import std.iostream;
```

目前，主流编译器对 C++ 模块的支持正在逐步完善。以下是一些编译器启用 C++23 标准和模块支持的方法：

- GCC：`g++ -std=c++23 -fmodules-ts -o program main.cpp`
- Clang：`clang++ -std=c++23 -fmodules -o program main.cpp`
- MSVC（Visual Studio）：`cl /std:c++23 /experimental:module /EHsc /Fe:program main.cpp`

## static

正常情况下，当前文件内部的全局变量，可以被其他文件使用。有时候，不希望发生这种情况，而是希望某个变量只局限在当前文件内部使用，不要被其他文件引用。这时可以在声明变量的时候，使用 `static`关键字，使得该变量变成当前文件的私有变量。

```cpp
// Variable with internal linkage
static int animals = 8;
```