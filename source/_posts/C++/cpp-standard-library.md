---
title: C++ 标准库
categories:
  - C++
  - Basics
tags:
  - C++
cover: /img/cpp-introduction.png
top_img: /img/cpp-top-img.svg
abbrlink: e88bb280
date: 2025-03-04 12:04:01
description: 
---

# Containers

## 简介

容器是用来存储数据的序列，每个容器都作为模板类实现。不同的容器提供了不同的存储方式和访问模式。一般分为 4 种类型：

- **序列容器**：是指那些在容器中按顺序存放元素的容器类。
  - `std::array` 固定大小连续数组，支持快速随机访问
  
  - `std::vector` 动态连续数组，支持快速随机访问
  
  - `std::deque` 双端队列，支持向前或向后插入
  
  - `std::list` 双向链表，支持向前或向后插入，但不支持随机访问
  
  - `std::forward_list`  单向链表
  
- **关联容器**：会在新元素插入时自动排序，可实现快速搜索
  - `std::set`：是一种用于存放唯一元素的集合，按元素排序
  
  - `std::multiset`：是一种允许重复元素的集合
  
  - `std::map`：键值对的集合。按键排序，且键是唯一的
  
  - `std::multimap`：键值对的集合，允许重复的键
  
- **无序关联容器**：实现可快速搜索的未排序（哈希）数据结构
  
  - `std::unordered_set`：无序集合
  
  - `std::unordered_multiset`：无序多重集合
  
  - `std::unordered_map`：无序映射
  
  - `std::unordered_multimap`：无序多重映射

- **容器适配器**：用于将一种容器或适配成另一种容器
  - `std::stack` 
  - `std::queue`
  - `std::priority_queue` 

## 序列容器

C++11 标准引入了 `<array>` 头文件，它提供了一种固定大小的数组容器，与 C 语言中的数组相比，具有更好的类型安全和内存管理特性。

```cpp
#include <array>

std::array<typename, size> arr_name;
```

vector 是一种连续存储动态数组，可以在运行时根据需要动态调整大小。自动管理内存，不需要手动分配和释放。

```cpp
#include <vector>

std::vector<typename> vec_name;
```

vector 可通过多种方式进行初始化：

```cpp
// Creating an empty vector
vector<int> v1;

// Creating a vector of 5 elements from initializer list
vector<int> v2 = {1, 4, 2, 3, 5};

// Creating a vector of 5 elements with default value
vector<int> v3(5, 9);
```

deque 全称是 double-ended queue ，是允许在两端进行插入和删除操作，是stack和queue 的结合体。

与 vector 不同，list 将数据存储在非连续内存中。容器中的每个元素都通过指针指向前一个元素和后一个元素，称为双向链表。链表只提供了从第一个元素或最后一个元素访问容器的方法——即不支持元素的随机访问。如果你需要访问中间位置的元素，你必须遍历链表以找到你需要访问的元素。链表的优势在于它支持快速插入元素。

forward_list 也像list一样以顺序方式存储数据，但不同之处在于forward_list仅存储序列中下一个元素的位置。它实现了单向链表。

以下是 `std::vector` 的一些常用成员函数：

| Function                    | Description           |
|:--------------------------- |:--------------------- |
| `at(size_t pos)`            | 返回指定位置的元素，带边界检查       |
| `operator[]`                | 返回指定位置的元素，不带边界检查      |
| `front()`                   | 返回第一个元素               |
| `back()`                    | 返回最后一个元素              |
| `data()`                    | 返回指向底层数组的指针           |
| `size()`                    | 返回当前元素数量              |
| `capacity()`                | 返回当前分配的容量             |
| `reserve(size_t n)`         | 预留至少 `n` 个元素的存储空间     |
| `resize(size_t n)`          | 将元素数量调整为 `n`          |
| `clear()`                   | 清空所有元素                |
| `insert(iterator pos, val)` | 在指定位置插入元素             |
| `erase(iterator pos)`       | 删除指定位置的元素             |
| `push_back(const T& val)`   | 在末尾添加元素               |
| `pop_back()`                | 删除末尾元素                |
| `begin()` / `end()`         | 返回指向向量的第一个/最后一个元素的迭代器 |

```cpp
include<iostream>
#include<vector> 
using namespace std;

int main() {
    // Initializing the array elements
    vector<char> vec = {'a', 'c', 'f', 'd', 'z'};

    // Printing number of vector elements
    cout << "The number of vector elements is : ";
    cout << vec.size() << endl;

    // Printing vector elements
    for (int element : vec) {
        cout << element << " ";
    }
    cout << endl;

    cout << "Printing 3th elements using at() : ";
    cout << arr.at(2) << " " << endl; }
    cout << "Printing 3th elements using operator[] : ";
    cout << arr[2] << " " << endl; }

    try {
        cout << vec.at(7) << std::endl;  // error
    } catch (const out_of_range& e) {
        cout << "Exception: " << e.what() << endl;
    }

    // Printing first element of vector 
    cout << "First element of vector is : ";
    cout << "First element: " << vec.front() << endl;

    // Printing last element of vector 
    cout << "Last element of vector is : ";
    cout << "Last element: " << vec.back() << endl;

    // Inserting 'z' at the back
    vec.push_back('z'); 

    // Inserting 'c' at index 1
    vec.insert(vec.begin() + 1, 'c');

    // Find the element 'f'
    auto it = find(vec.begin(), vec.end(), 'f');
    cout << *it << endl;

    // Deleting element 'f'
    vec.erase(find(vec.begin(), vec.end(), 'f'));

    return 0;
}
```

## 关联容器

`set` 是一种用于存储唯一元素的有序集合，底层使用红黑树实现。

`map` 是一种有序的键值对容器，这使得它非常适合需要快速查找、插入和删除的场景。

## 容器适配器

`stack`是一种后进先出（LIFO, Last In First Out）的数据结构，容器内的元素是线性排列的，但只允许在一端（栈顶）进行添加和移除操作。这种数据结构非常适合于需要"最后添加的元素最先被移除"的场景。

`queue` 是一种先进先出（FIFO, First In First Out）的数据结构，它允许在一端添加元素（称为队尾），并在另一端移除元素（称为队首）。常用于广度优先搜索、任务调度等场景。

# Iterators

迭代器用于遍历容器中的元素，允许以统一的方式访问容器中的元素，而不用关心容器的内部实现细节。STL 提供了多种类型的迭代器，包括随机访问迭代器、双向迭代器、前向迭代器和输入输出迭代器等。

迭代器可以被形象地看做是一个指向容器元素的指针，每个容器都提供了四个基本的成员函数，用于返回迭代器：

- `begin()`：返回一个指向容器起始元素的迭代器；
- `end()`：返回一个指向容器末尾的迭代器（其前一个元素为容器中最后一个元素）；
- `cbegin()` ：返回一个指向容器起始元素的const迭代器；
- `cend()` ：返回一个指向容器末尾的const迭代器（其前一个元素为容器中最后一个元素）。

同时迭代器具有一系列被重载的运算符：

| Operator      | Description                                                |
| ------------- | ---------------------------------------------------------- |
| `*`           | 解引用运算符，可以返回迭代器当前所指的元素                 |
| `++`          | 将迭代器移动到下一个元素                                   |
| `==` and `!=` | 基本的比较运算符，用于比较两个迭代器所指的元素是否相同。   |
| `=`           | 为迭代器赋值一个新的位置（一般来说是容器的起点或末尾元素） |

所有的容器都提供了（至少）两种类型的迭代器：

- `container::iterator` ：可以读写的迭代器
- `container::const_iterator`： 只读迭代器

迭代器的遍历语法通常如下：

```cpp
#include <array>
#include <iostream>
using namespace std;

int main(){
    // declaration of array container
    array<int, 5> myarray{ 1, 2, 3, 4, 5 };

    // using begin() to print array
    for (auto it = myarray.begin(); it != myarray.end(); ++it) {
        cout << *it << ' '; // print the value of the element it points to
    }
    cout << endl;

    return 0;
}
```

```cpp
#include <iostream>
#include <map>
#include <string>

int main()
{
    std::map<int, std::string> mymap;
    mymap.insert(std::make_pair(4, "apple"));
    mymap.insert(std::make_pair(2, "orange"));
    mymap.insert(std::make_pair(1, "banana"));
    mymap.insert(std::make_pair(3, "grapes"));
    mymap.insert(std::make_pair(6, "mango"));
    mymap.insert(std::make_pair(5, "peach"));

    auto it{ mymap.cbegin() }; // declare a const iterator and assign to start of vector
    while (it != mymap.cend()) // while it hasn't reach the end
    {
        std::cout << it->first << '=' << it->second << ' '; //  the value of the element it points to
        ++it; // and iterate to the next element
    }

    std::cout << endl;
}
```

**Output：**

```
1=banana 2=orange 3=grapes 4=apple 5=peach 6=mango
```

# Algorithms

C++ 标准库中的 `<algorithm>` 头文件提供了一组用于操作容器（如数组、向量、列表等）的算法。这些算法包括排序、搜索、复制、比较等，它们是编写高效、可重用代码的重要工具。

注意，这些算法都被实现为使用迭代器进行操作的函数。这意味着每个算法都只需要实现一次，就可以配合所有提供迭代器的容器使用。

大多数 `<algorithm>` 中的函数都遵循以下基本语法：

```
func(first, last, ...);
```

这里的 `first` 和 `last` 分别是指向容器开始和结束的迭代器。

| Function     | Description                                            |
| ------------ | ------------------------------------------------------ |
| sort         | 对容器中的元素进行排序                                 |
| partial_sort | 对部分区间排序，前 n 个元素为有序                      |
| reverse      | 反转范围内元素的顺序                                   |
| rotate       | 旋转某个范围内的元素，使特定元素成为第一个元素         |
| is_sorted    | 检查区域中的元素是否按非降序排序                       |
| copy         | 复制特定范围的元素                                     |
| fill         | 为范围内的所有元素分配指定的值                         |
| for_each     | 对区间内的每个元素执行操作                             |
| transform    | 将函数应用于区间中的每个元素，并将结果存储在另一个区间 |
| replace      | 将范围内出现的所有特定值替换为新值                     |
| merge        | 将两个有序区间合并到一个有序区间                       |
| swap         | 交换两个变量的值                                       |
| remove       | 从范围中删除具有指定值的所有元素，但不减小容器大小     |
| unique       | 从范围中删除连续的重复元素。                           |
|              |                                                        |
|              |                                                        |
| count        | 计算范围内给定元素的出现次数                           |
| find         | 返回范围中第一次出现的元素的迭代器                     |
|              |                                                        |

注意 `std::sort()` 不能配合链表使用，链表提供了自己的 `sort()` 成员函数，它的效率要比泛型的排序高的多。

```cpp
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
    vector<int> vec = { 5, 10, 15 };
  
    sort(vec.begin(), vec.end());
    
    for_each(first, last, [](int& x) { x += 1; });

    return 0;
}
```

C++ 标准库中的 `<numeric>` 头文件提供了一组用于数值计算的函数模板，这些函数可以对容器中的元素进行各种数值操作，如求和、乘积、最小值、最大值等。

| Function            | Description                      |
| ------------------- | -------------------------------- |
| max_element         | 查找给定范围内的最大元素         |
| min_element         | 查找给定范围内的最小元素         |
| accumulate          | 计算容器中所有元素的总和         |
| partial_sum         | 计算容器中元素的部分和           |
| inner_product       | 计算两个容器中对应元素乘积的总和 |
| adjacent_difference | 计算容器中相邻元素的差值         |
| gcd                 | 计算两个整数的最大公约数         |
| lcm                 | 计算两个整数的最小公倍数         |

```cpp
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

int main() {

    std::vector<int> v = {3, 1, 4, 1, 5, 9};

    // Defining range as whole array
    auto first = vec.begin();
    auto last = vec.end();

    // 计算最小值和最大值
    int min = *min_element(first, last);
    int max = *max_element(first, last);

    // Use accumulate to find the sum of elements in the vector
    int sum = accumulate(first, last, 0);

    // 计算平均值
    double avg = static_cast<double>(sum) / vec.size();

    // 输出结果
    cout << "Min: " << min << endl;
    cout << "Max: " << max << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << avg << endl;

    return 0;
}
```



# Functor

函数对象是可以像函数一样调用的对象，可以用于算法中的各种操作。

`<functional>` 头文件中定义了一些常用的函数对象：

- `std::function`：一个通用的多态函数封装器。
- `std::bind`：用于绑定函数的参数。
- `std::plus`、`std::minus`、`std::multiplies`、`std::divides`、`std::modulus`：基本的算术操作。
- `std::equal_to`、`std::not_equal_to`、`std::greater`、`std::less`、`std::greater_equal`、`std::less_equal`：比较操作。
- `std::unary_negate`、`std::binary_negate`：逻辑否定操作。
- `std::logical_and`、`std::logical_or`、`std::logical_not`：逻辑操作。

```cpp
// C++ program to demonstrate working of functors.
#include <iostream>
using namespace std;

// A Functor
class increment
{
private:
    int num;
public:
    increment(int n) : num(n) { }

    // This operator overloading enables calling
    // operator function () on objects of increment
    int operator () (int arr_num) const {
        return num + arr_num;
    }
};

// Driver code
int main()
{
    int arr[] = {1, 2, 3, 4, 5};
    int n = sizeof(arr)/sizeof(arr[0]);
    int to_add = 5;

    transform(arr, arr+n, arr, increment(to_add));

    for (int i=0; i<n; i++)
        cout << arr[i] << " ";
}
```

`std::function` 是一个模板类，可以存储、调用和复制任何可调用对象，比如函数、lambda 表达式或函数对象。

```cpp
#include <iostream>
#include <functional>

void greet() {
 std::cout << "Hello, World!" << std::endl;
}

int main() {
 std::function<void()> f = greet; // 使用函数
 f(); // 输出: Hello, World!

std::function<void()> lambda = []() {
 std::cout << "Hello, Lambda!" << std::endl;
 };
 lambda(); // 输出: Hello, Lambda!

return 0;
}
```

`std::bind` 允许我们创建一个可调用对象，它在调用时会将给定的参数绑定到一个函数或函数对象。

```cpp
\#include <iostream>
\#include <functional>

int add(int a, int b) {
 return a + b;
}

int main() {
 auto bound_add = std::bind(add, 5, std::placeholders::_1);
 std::cout << bound_add(10) << std::endl; // 输出: 15

return 0;
}
```

在这个例子中，`std::placeholders::_1` 是一个占位符，它在调用 `bound_add` 时会被实际的参数替换。

# 字符串

C++的标准库 `<cstring>` 中有大量的函数用来操作C风格的字符串

| Function Name    | Description                                                      |
|:---------------- |:---------------------------------------------------------------- |
| `strcpy(s1, s2)` | 复制字符串 s2 到字符串 s1                                                 |
| `strcat(s1, s2)` | 连接字符串 s2 到字符串 s1 的末尾，连接字符串也可以用 `+` 号                             |
| `strlen(s1)`     | 返回字符串 s1 的长度                                                     |
| `strcmp(s1, s2)` | 如果 s1 和 s2 是相同的，则返回 0；<br>如果 s1\<s2 则返回值小于 0；如果 s1\>s2 则返回值大于 0。 |
| `strstr(s1, s2)` | 返回一个指针，指向字符串 s1 中字符串 s2 第一次出现的位置。                                |
| `strlwr(s1)`     | 转换字符串成小写                                                         |
| `strupr(s1)`     | 转换字符串成大写                                                         |

C++ 除了可以使用C风格的字符串，还可以使用标准库 `<string>` 中的 `std::string` 类。它是对 C 风格字符串的封装，提供了更安全、更易用的字符串操作功能。以下是一些常用的成员函数：

| Function Name         | Description                |
|:--------------------- |:-------------------------- |
| `size()`              | 返回字符串的长度（字符数）。             |
| `length()`            | 与 `size()` 相同，返回字符串的长度。    |
| `empty()`             | 判断字符串是否为空。                 |
| `operator[]`          | 通过索引访问字符串中的字符。             |
| `at()`                | 访问字符串中指定位置的字符（带边界检查）。      |
| `substr()`            | 返回从指定位置开始的子字符串。            |
| `find()`              | 查找子字符串在字符串中的位置。            |
| `rfind()`             | 从字符串末尾开始查找子字符串的位置。         |
| `replace()`           | 替换字符串中的部分内容。               |
| `append()`            | 在字符串末尾添加内容。                |
| `insert()`            | 在指定位置插入内容。                 |
| `erase()`             | 删除指定位置的字符或子字符串。            |
| `clear()`             | 清空字符串。                     |
| `c_str()`             | 返回 C 风格的字符串（以 null 结尾）。    |
| `data()`              | 返回指向字符数据的指针（C++11 及之后的版本）。 |
| `compare()`           | 比较两个字符串。                   |
| `find_first_of()`     | 查找第一个匹配任意字符的位置。            |
| `find_last_of()`      | 查找最后一个匹配任意字符的位置。           |
| `find_first_not_of()` | 查找第一个不匹配任意字符的位置。           |
| `find_last_not_of()`  | 查找最后一个不匹配任意字符的位置。          |

`at()` 是 string 类的一个成员函数，它会根据下标来返回字符串的一个字符。与`[]`不同，`at()` 会检查下标是否越界，如果越界就抛出一个异常；而`[]`不做检查，不管下标是多少都会照常访问。

```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "Hello, World!";

    // size()
    std::cout << "Length: " << str.size() << std::endl;
    // empty()
    std::cout << "Is empty? " << (str.empty() ? "Yes" : "No") << std::endl;
    // operator[]
    std::cout << "First character: " << str[0] << std::endl;
    // at()
    std::cout << "Character at position 7: " << str.at(7) << std::endl;
    // substr()
    std::string sub = str.substr(7, 5);
    std::cout << "Substring from position 7 with length 5: " << sub << std::endl;
    // replace()
    str.replace(pos, 5, "C++");
    std::cout << "Modified string: " << str << std::endl;
    // append()
    str.append(" How are you?");
    std::cout << "Appended string: " << str << std::endl;
    // c_str()
    str = "Hello, C++!";
    const char* cstr = str.c_str();
    std::cout << "C-style string: " << cstr << std::endl;

    return 0;
}
```

# 时间和日期

C++ 继承了 C 语言用于日期和时间操作的库 `<ctime>` 。与时间相关的常用数据类型：

- `time_t`：表示时间的类型，通常是一个长整型。
- `struct tm`：一个结构体，用于表示时间的各个部分

```cpp
struct tm {
    int tm_sec;  // seconds, range 0 to 59
    int tm_min;  // minutes, range 0 to 59
    int tm_hour;  // hours, range 0 to 23
    int tm_mday;  // day of the month, range 1 to 31
    int tm_mon;  // month, range 0 to 11
    int tm_year;  // The number of years since 1900
    int tm_wday;  // day of the week, range 0 to 6
    int tm_yday;  // day in the year, range 0 to 365
    int tm_isdst;  // daylight saving time
}
```

下面是 `<ctime>` 中 中关于日期和时间的重要函数

| Function                          | Explanation                             |
| --------------------------------- | --------------------------------------- |
| `time_t time(NULL)`               | 获取当前时间，自 1970 年 1 月 1 日以来经过的秒数          |
| `struct tm* localtime(time_t*)`   | 将 `time_t` 类型的时间转换为 `tm` 结构体            |
| `tm* gmtime (time_t*)`            | 将 `time_t` 类型的时间转换为协调世界时（UTC）的 `tm` 结构体 |
| `char* ctime(time_t*)`            | 返回 `Www Mmm dd hh:mm:ss yyyy`格式的字符串指针   |
| `char* asctime(struct tm*)`       | 返回 `Www Mmm dd hh:mm:ss yyyy`格式的字符串指针   |
| `time_t mktime(struct tm*)`       | 将 `tm` 结构体转换为 `time_t` 类型的秒数            |
| `double difftime(time_t, time_t)` | 返回相差的秒数                                 |
| `size_t strftime()`               | 格式化日期和时间为指定的格式                          |

用于打印系统日期和时间的程序

```cpp
#include <iostream>
#include <ctime>
int main()
{
    time_t t = time(0);
    std::cout << ctime(&t) << std::endl;
    return 0;
}
```

**Output**

```cpp
Fri Feb 28 12:51:02 2025
```

C++11 引入了 `<chrono>` 库，这是一个用于处理时间和日期的库。它提供了一套丰富的工具来测量时间间隔、执行时间点的计算以及处理日期和时间。

```cpp
#include <chrono>

auto now = std::chrono::system_clock::now();
auto duration = std::chrono::seconds(5);
auto future_time = now + duration;
```

# 多线程并行

C++ 提供了强大的多线程支持，特别是在 C++11 标准及其之后，通过 `<thread>` 标准库使得多线程编程变得更加简单和安全。它包括以下几个关键组件：

- `std::thread`：用于创建和管理线程。
- `std::this_thread`：提供了一些静态成员函数，用于操作当前线程。
- `std::thread::id`：线程的唯一标识符。
- std::mutex 用于同步对共享资源的访问
- `std::unique_lock` 锁管理器，用于自动管理锁的生命周期。
- `std::condition_variable` 用于线程间的等待和通知。
- `std::future` 和 `std::promise` 用于线程间的结果传递。
- `std::async` 用于启动异步任务，并返回一个 `std::future`。

要创建一个线程，你需要实例化 `std::thread` 类，并传递一个可调用对象（函数、lambda 表达式或对象的成员函数）作为参数。

创建 `std::thread` 对象后，线程会立即开始执行，你可以调用 `join()` 方法来等待线程完成。`join()` 方法会阻塞当前线程，直到被调用的线程完成执行。

当线程执行完毕后，你可以使用 `detach()` 方法来分离线程

```cpp
#include <iostream>
#include <thread>
#include <chrono>

// 简单的函数，在线程中执行
void print_message(const std::string& message, int delay) {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    std::cout << message << std::endl;
}

int main() {
    // 创建两个线程，执行 print_message 函数
    std::thread t1(print_message, "Hello from thread 1", 1000);
    std::thread t2(print_message, "Hello from thread 2", 500);

    // 等待线程 t1 完成
    if (t1.joinable()) {
        t1.join();
    }

    // 等待线程 t2 完成
    if (t2.joinable()) {
        t2.join();
    }

    std::cout << "Main thread finished." << std::endl;

    return 0;
}
```



# 智能指针

智能指针是 `<memory>` 头文件中的核心内容。它们是 C++11 引入的特性，用于自动管理动态分配的内存。智能指针的主要类型有：

- `std::unique_ptr`：独占所有权的智能指针，同一时间只能有一个 `unique_ptr` 指向特定内存。
- `std::shared_ptr`：共享所有权的智能指针，多个 `shared_ptr` 可以指向同一内存，内存在最后一个 `shared_ptr` 被销毁时释放。
- `std::weak_ptr`：弱引用智能指针，用于与 `shared_ptr` 配合使用，避免循环引用导致的内存泄漏。

智能指针是一个带有析构函数和重载运算符（ `*, ->`）的包装类，当对象被销毁时，它也会释放内存。

```cpp
#include <iostream>
#include <memory> // for std::unique_ptr

class Resource
{
public:
    Resource() { std::cout << "Resource acquired\n"; }
    ~Resource() { std::cout << "Resource destroyed\n"; }
    friend std::ostream& operator<<(std::ostream& out, const Resource &res)
    {
        out << "I am a resource";
        return out;
    }
};

int main()
{
    std::unique_ptr<Resource> res{ new Resource{} };

    if (res) // use implicit cast to bool to ensure res contains a Resource
        std::cout << *res << '\n'; // print the Resource that res is owning

    return 0;
}
```

# 实用库

`<cstdio>` 输入输出标准库

`<cmath>` 是 C++ 标准库中的一个头文件，它提供了许多基本的数学函数。

`<regex>` 头文件提供了正则表达式的功能

`<cstdio>` 是 C++ 标准库中的一个头文件，它包含了 C 语言标准 I/O 库的 C++ 封装，主要用于文件的输入和输出操作。

`<cstdio>` 库定义了一组用于执行输入和输出操作的函数，这些函数可以用于读写文件和控制台。

`<utility>` 头文件包含了一些实用的工具类和函数，这些工具类和函数在编写高效、可读性强的代码时非常有用。

- `pair`：一个包含两个元素的容器，通常用于存储和返回两个相关联的值。
- `make_pair`：一个函数模板，用于创建 `pair` 对象。
- `swap`：一个函数模板，用于交换两个对象的值。
- `forward` 和 `move`：用于完美转发和移动语义的函数模板。

标准库中的 `<random>` 头文件提供了一组用于生成随机数的工具，涵盖了从简单的均匀分布到复杂的离散分布，

标准库中的 `locale` 模块提供了一种方式，允许程序根据用户的区域设置来处理文本数据，如数字、日期和时间的格式化，以及字符串的比较和排序。这使得编写国际化应用程序变得更加容易。

`<climits>` 是 C++ 标准库中的一个头文件，提供了与整数类型相关的限制和特性。它定义了一组常量，描述了各种整数类型（如 `char`、`int`、`long` 等）的最小值、最大值和其他相关属性。 

`<cfloat>` 是 C++ 标准库中的一个头文件，用于定义浮点数相关的宏和常量。

`std::numbers` 是 C++20 中引入的一个标准库模块，主要用于提供一组常用的数学常量。

`std::numbers`位于 `<numbers>` 头文件中，并且包含了很多数学常量，涵盖了圆周率、自然对数的底数、黄金比例等常见常数。

`<cstdlib>` 是 C++ 标准库中的一个头文件，提供了各种通用工具函数，包括内存分配、进程控制、环境查询、排序和搜索、数学转换、伪随机数生成等。

`cstdlib` 中包含了许多有用的函数，以下是一些常用的函数及其简要说明：

1. `exit(int status)`: 终止程序执行，并返回一个状态码。
2. `system(const char* command)`: 执行一个命令行字符串。
3. `malloc(size_t size)`: 分配指定大小的内存。
4. `free(void* ptr)`: 释放之前分配的内存。
5. `atoi(const char* str)`: 将字符串转换为整数。
6. `atof(const char* str)`: 将字符串转换为浮点数。
7. `rand()`: 生成一个随机数。
8. `srand(unsigned int seed)`: 设置随机数生成器的种子。

