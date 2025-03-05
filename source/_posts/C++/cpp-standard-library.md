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
  
  - `std::array` 是一种固定大小的数组容器，与 C 语言中的数组相比，具有更好的类型安全和内存管理特性。支持快速随机访问
  - `std::vector` 是一种连续存储动态数组，可以在运行时根据需要动态调整大小。自动管理内存，不需要手动分配和释放。支持快速随机访问
  - `std::deque` 全称是 double-ended queue ，是一个动态数组，它提供了快速的随机访问能力，同时允许在两端进行高效的插入和删除操作
  - `std::list`  将数据存储在非连续内存中。容器中的每个元素都通过指针指向前一个元素和后一个元素，称为双向链表。链表只提供了从第一个元素或最后一个元素访问容器的方法——即不支持元素的随机访问。
  - `std::forward_list`  也像 `list` 一样以顺序方式存储数据，但不同之处在于`forward_list` 仅存储序列中下一个元素的位置。它实现了单向链表

- **关联容器**：会在新元素插入时自动排序，可实现快速搜索
  
  - `std::set`：是一种用于存储唯一元素的有序集合，底层使用红黑树实现
  - `std::multiset`：是一种允许重复元素的有序集合
  - `std::map`：是一种有序的键值对容器，按键排序，且键是唯一的。这使得它非常适合需要快速查找、插入和删除的场景
  - `std::multimap`：键值对的有序集合，允许重复的键

- **无序关联容器**：实现可快速搜索的未排序（哈希）数据结构
  
  - `std::unordered_set`：无序集合
  - `std::unordered_multiset`：无序多重集合
  - `std::unordered_map`：无序映射
  - `std::unordered_multimap`：无序多重映射

- **容器适配器**：用于将一种容器或适配成另一种容器
  
  - `std::stack`  是一种后进先出的数据结构，容器内的元素是线性排列的，但只允许在一端（栈顶）进行添加和移除操作
  - `std::queue` 是一种先进先出的数据结构，它允许在一端添加元素（称为队尾），并在另一端移除元素（称为队首）。
  - `std::priority_queue` 

## vector

vector 是一种连续存储动态数组，可以在运行时根据需要动态调整大小。自动管理内存，不需要手动分配和释放。

```cpp
#include <vector>

std::vector<type> myVec;
```

vector 可通过多种方式进行初始化

| Constructor        | Description                |
| ------------------ | -------------------------- |
| `vector()`         | 创建一个空的容器                   |
| `vector(n)`        | 创建一个包含 `n` 个默认值元素的容器       |
| `vector(n, value)` | 创建一个包含 `n` 个值为 `value` 的容器 |
| `vector(il)`       | 使用初始化列表 `il` 构造容器          |

```cpp
// Creating an empty vector
vector<int> v;

// Creating a vector of 5 elements from initializer list
vector<int> v = {1, 4, 2, 3, 5};

// Creating a vector of size 5 where each element initialized to 9
vector<int> v(5, 9);

// Initialize the std::vector v by arr
int arr[] = {11, 23, 45, 89};
int n = sizeof(arr) / sizeof(arr[0]);
vector<int> v = {arr, arr + n};

// Initialize the vector v2 from vector v1 
vector<int> v1 = {11, 23, 45, 89};
vector<int> v2(v1.begin(), v1.end());
```

以下是 `std::vector` 的一些常用成员函数：

| Function                                                                                 | Description           |
|:---------------------------------------------------------------------------------------- |:--------------------- |
| `operator[pos]`                                                                          | 返回指定位置的元素，不带边界检查      |
| `at(pos)`                                                                                | 返回指定位置的元素，带边界检查       |
| `front()`                                                                                | 返回第一个元素               |
| `back()`                                                                                 | 返回最后一个元素              |
| `data()`                                                                                 | 返回指向底层数组的指针           |
| `size()`                                                                                 | 返回当前元素数量              |
| `capacity()`                                                                             | 返回当前分配的容量             |
| `reserve(n)`                                                                             | 预留至少 `n` 个元素的存储空间     |
| `resize(n)`                                                                              | 更改向量的大小               |
| `empty()`                                                                                | 检查向量是否为空              |
| `clear()`                                                                                | 清空所有元素                |
| `insert(pos_iter, val)`<br>`insert(pos_iter, n, val)`<br>`insert(pos_iter, first, last)` | 在指定位置插入元素             |
| `erase(pos_iter)`<br>`erase(first, last)`                                                | 删除指定位置或范围的元素          |
| `push_back(val)`                                                                         | 在末尾添加元素               |
| `pop_back()`                                                                             | 删除末尾元素                |
| `begin()` / `end()`                                                                      | 返回指向向量的第一个/最后一个元素的迭代器 |

```cpp
#include<iostream>
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

    // Accessing and printing values
    cout << "Acessing value at index 3 using at() : ";
    cout << vec.at(2) << " " << endl; 
    cout << "Acessing value at index 3 using operator[] : ";
    cout << vec[2] << " " << endl; 
    cout << "Acessing value at 3 using iterator : "
    cout << *(vec.begin() + 2) << " " << endl; 

    // Accessing value at invalid index 10
    try {
        cout << vec.at(10) << std::endl;  
    } catch (const out_of_range& e) {
        cout << "Exception: " << e.what() << endl;
    }

    // Modify the element at index 2
    vec[2] = 'F';
    vec.at(2) = 'F';
    *(vec.begin() + 2) = 'F';

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

## list

`list` 将数据存储在非连续内存中。容器中的每个元素都通过指针指向前一个元素和后一个元素，称为双向链表。链表只提供了从第一个元素或最后一个元素访问容器的方法——即不支持元素的随机访问。

```cpp
#include <list>

std::list<type> myList;
```

list 和 vector一样，可通过多种方式进行初始化。

以下是 `std::list` 的一些常用的成员函数：

| Function                                                                                 | Description   |
| ---------------------------------------------------------------------------------------- | ------------- |
| `push_back(val)`                                                                         | 在链表末尾添加元素     |
| `push_front(val)`                                                                        | 在链表头部添加元素     |
| `pop_back()`                                                                             | 删除链表末尾的元素     |
| `pop_front()`                                                                            | 删除链表头部的元素     |
| `insert(pos_iter, val)`<br>`insert(pos_iter, n, val)`<br>`insert(pos_iter, first, last)` | 在指定位置插入元素     |
| `erase(pos_iter)`<br>`erase(first, last)`                                                | 删除指定位置或范围的元素  |
| `clear()`                                                                                | 清空所有元素        |
| `size()`                                                                                 | 返回链表中的元素数量    |
| `empty()`                                                                                | 检查链表是否为空      |
| `front()`                                                                                | 返回链表第一个元素     |
| `back()`                                                                                 | 返回链表最后一个元素    |
| `remove(val)`                                                                            | 删除所有等于指定值的元素  |
| `sort()`                                                                                 | 对链表中的元素进行排序   |
| `merge(other)`                                                                           | 合并另一个已排序的链表   |
| `reverse()`                                                                              | 反转链表          |
| `begin()` / `end()`                                                                      | 返回容器的起始/结束迭代器 |

```cpp
#include <iostream>
#include <list>
using namespace std;

int main() {
    std::list<int> lst = {10, 20, 30};

    // Inserting an element at the end
    lst.push_back(5);

    // Inserting an element at the beginning
    lst.push_front(1);

    // Inserting an element at a specific position
    auto it = lst.begin();
    advance(it, 2);
    lst.insert(it, 4);

    // Deleting last element
    lst.pop_back();

    // Deleting first element
    lst.pop_front();

    return 0;
}
```

合并两个已排序的链表

```cpp
#include <iostream>
#include <list>
using namespace std;

int main() {
    // declaring the lists 
    // initially sorted 
    list<int> list1 = { 10, 20, 30 }; 
    list<int> list2 = { 40, 50, 60 }; 

    // merge operation 
    list2.merge(list1); 

    return 0;
}
```

## set

`set` 是一种用于存储唯一元素的有序集合，特别适合需要快速查找、插入和删除操作的场景。

```cpp
#include <set>

std::set<type> mySet;
std::set<type, comp> mySet;
```

以下是 `std::list` 的一些常用的成员函数：

| Function                                                            | Description        |
| ------------------------------------------------------------------- | ------------------ |
| `insert(val)`<br>`insert(first, last)`<br>`insert({val1,val2,...})` | 插入元素               |
| `emplace(val)`                                                      | 将新元素插入容器中          |
| `erase(val)`<br>`erase(pos_iter)`<br>`erase(first, last)`           | 删除指定位置或范围的元素       |
| `clear()`                                                           | 清空所有元素             |
| `size()`                                                            | 返回容器中的元素数量         |
| `empty()`                                                           | 检查容器是否为空           |
| `find(key)`                                                         | 查找集合中的元素           |
| `contains(key)`                                                     | 查看元素是否存在           |
| `count(key)`                                                        | 返回指定元素的计数          |
| `merge(other)`                                                      | 将一个 Set 合并到另一个 Set |
| `equal_range(key)`                                                  |                    |
| `upper_bound(key)`                                                  | 大于key的第一个元素的迭代器    |
| `lower_bound(key)`                                                  | 小于key的第一个元素的迭代器    |
| `begin()` / `end()`                                                 | 返回容器的起始/结束迭代器      |

我们不能像 vector 中那样按索引访问集合的元素。在 set 中，我们必须分别递增或递减从 `begin()` 或  `end()` 方法获取的迭代器，才能按位置访问元素。也可以在 `next()`  或 `advance()` 函数的帮助下完成。

```cpp
#include <iostream>
#include <set>
using namespace std;

int main() {
    set<int> s = {1, 4, 2, 3, 5};

    // Accessing first element
    auto it1 = s.begin();

    // Accessing third element
    auto it2 = next(it1, 2);

    cout << *it1 << " " << *it2;
    return 0;
}
```

元素的插入

```cpp
#include <iostream>
#include <set>
#include <vector>

int main() {
    // Create an empty set
    std::set<int> st;

    // Insert elements into set
    st.insert(5);
    st.emplace(3);
    st.insert(5);

    // Insting the multiple values
    st.insert({12, 45, 11, 78, 9}); 

    // Inserting values of vector to set
    std::vector<int> vec = {1, 2, 3};
    st.insert(vec.begin(), vec.end());
}
```

查找指定值是否存在

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    set<int> s = {1, 3, 5, 7, 9};
    int x = 5;

    // Check if the element exists using find()
    if (s.find(x) != s.end()) {
        cout << "Element found";
    } else {
        cout << "Element not found";
    }

    // Check if the element exists using count()
    if (s.count(x) > 0) {
        cout << "Element found";
    } else {
        cout << "Element not found";
    }

    // Check if element exists using contains()
    if (s.contains(x)) {
        cout << "Element found";
    } else {
        cout << "Element not found";
    }

    return 0;
}
```

## map

`map` 是一种有序的键值对容器，容器中的元素是按照键的顺序自动排序的，这使得它非常适合需要快速查找和有序数据的场景。

```cpp
#include <map>

std::map<key_type, value_type, comp> myMap;
```

声明和初始化 map 的不同方法

```cpp
#include <map>
using namespace std;

int main() {
    // Inializing std::map using initializer list
    map<int, char> m = {{1, 'a'},
                        {3, 'b'},
                        {2, 'c'}};

    
    // Create and initialize the map with vector
    vector<pair<int, char>> v = {{1, 'a'},
                                 {3, 'b'}, 
                                 {2, 'c'}};

    map<int, char> m(v.begin(), v.end());

    for (auto& p : m)
        cout << p.first << " " << p.second << "\n";
}
```

以下是 `std::map` 的一些常用成员函数：

| Function                                                                         | Description                              |
|:-------------------------------------------------------------------------------- |:---------------------------------------- |
| `operator[key]`                                                                  | 返回或修改指定key的元素                            |
| `at(key)`                                                                        | 返回或修改指定key的元素，带边界检查                      |
| `size()`                                                                         | 返回当前元素数量                                 |
| `insert({key, val})`<br>`insert(first, last)`<br>`insert({{k1,v1}, {k2,v2}...})` | 在容器中插入元素                                 |
| `insert_or_assign(key, value)`                                                   | 如果给定的键不存在，它会将其插入，但如果给定的键已经存在，则它只会更新给定键的值 |
| `empty()`                                                                        | 检查容器是否为空                                 |
| `clear()`                                                                        | 清空所有元素                                   |
| `erase(key)`<br>`erase(pos_iter)`<br>`erase(first, last)`                        | 删除指定元素                                   |
| `count(key)`                                                                     | 返回与特定key匹配的元素数                           |
| `find(key)`                                                                      | 返回 map 中具有key值的元素的迭代器                    |
| `contains(key)`                                                                  | 检查key是否存在                                |
| `begin()` / `end()`                                                              | 返回指向容器的第一个/最后一个元素的迭代器                    |

 可以使用 `[]` 运算符或 `insert()` 方法将元素插入到 map 中 。如果具有给定键的元素已存在，则 `insert()` 方法会跳过插入，但 `[]` 运算符会将关联值更新为新值。

```cpp
#include <map>
using namespace std;

int main() {
    // creating a map
    map<int, string> myMap = { { 1, "One" }, { 2, "Two" } };

    // Updating an existing element's value
    myMap[2] = "Three";

    // Inserting a key value pair
    myMap.insert({3, 'Three'});

    // Insert key which is already present
    myMap.insert_or_assign(2, 'd');

    // Intializing two maps
    map<int, string> map1{ { 10, "Mike" }, { 20, "John" } };
    map<int, string> map2{ { 30, "Alice" }, { 40, "Bob" } };

    // Concatenate both the maps
    map1.insert(map2.begin(), map2.end());

    return 0;
}
```

可以使用运算符 `[]` 中的相应键来访问 Map 元素。 如果键存在，它将返回关联的值，但如果键不存在，它将使用给定的键和默认值创建一个新元素。为了避免这种情况，我们还可以使用 `at()` 方法来访问元素。

```cpp
#include <map>
using namespace std;

int main() {
    map<int, string> m = {{1, "One"},
             {2, "Two"}, {3, "Three"}};

    // Accessing elements
    cout << m[1] << endl;
    cout << m.at(2);

    return 0;
}
```

## stack

`stack` 遵循 （LIFO, Last In First Out） 的插入和删除顺序。这意味着首先删除最近插入的元素，最后删除第一个插入的元素。这是通过仅在堆栈的一端（通常称为堆栈顶部）插入和删除元素来完成的。

<img title="" src="https://media.geeksforgeeks.org/wp-content/uploads/20240606180844/Push-Operation-in-Stack-(1).webp" alt="" width="391" data-align="center">

以下是 `std::stack` 的一些常用成员函数：

| Function  | Description     |
|:--------- |:--------------- |
| `push()`  | 在栈顶添加一个元素       |
| `pop()`   | 移除栈顶元素          |
| `top()`   | 返回栈顶元素的引用，但不移除它 |
| `empty()` | 检查容器是否为空        |
| `size()`  | 返回容器中元素数量       |

```cpp
#include <stack>
#include <iostream>
using namespace std;

int main() {
    stack<int> st;
    st.push(5);
    st.push(11);
    st.push(9);

      // Top element before pop
    cout << st.top() << endl;

      // Popping the top element
      st.pop();
      cout << st.top();
    return 0;
}
```

注意：`stack` 不提供直接访问栈中元素的方法，新元素只能通过使用 `push()` 方法插入到stack的顶部，只能通过 `top()` 访问stack顶部元素。

# Iterators

迭代器可以被形象地看做是一个指向容器元素的指针，允许以统一的方式访问容器中的元素，而不用关心容器的内部实现细节。STL 提供了多种类型的迭代器，包括随机访问迭代器、双向迭代器、前向迭代器和输入输出迭代器等。

每个容器都提供了四个基本的成员函数，用于返回迭代器：

- `begin()`：返回一个指向容器起始元素的迭代器；
- `end()`：返回一个指向容器末尾的迭代器（其前一个元素为容器中最后一个元素）；
- `cbegin()` ：返回一个指向容器起始元素的const迭代器；
- `cend()` ：返回一个指向容器末尾的const迭代器（其前一个元素为容器中最后一个元素）。

同时迭代器具有一系列被重载的运算符：

| Operator      | Description                   |
| ------------- | ----------------------------- |
| `*`           | 解引用运算符，可以返回迭代器当前所指的元素         |
| `++`          | 将迭代器移动到下一个元素                  |
| `==` and `!=` | 基本的比较运算符，用于比较两个迭代器所指的元素是否相同。  |
| `=`           | 为迭代器赋值一个新的位置（一般来说是容器的起点或末尾元素） |

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

# algorithm

C++ 中的 `<algorithm>` 标准库提供了一组用于操作容器（如数组、向量、列表等）的算法。这些算法包括排序、搜索、复制、比较等，它们是编写高效、可重用代码的重要工具。

大多数 `<algorithm>` 中的函数都遵循以下基本语法：

```
func(first, last, ...);
```

这里的 `first` 和 `last` 分别是指向容器开始和结束的迭代器。

> 注意，这些算法都被实现为使用迭代器进行操作的函数。这意味着每个算法都只需要实现一次，就可以配合所有提供迭代器的容器使用。

| Function      | Description                 |
| ------------- | --------------------------- |
| `sort          | 对容器中的元素进行排序                 |
| `partial_sort  | 对部分区间排序，前 n 个元素为有序          |
| `reverse       | 反转范围内元素的顺序                  |
| `rotate        | 旋转某个范围内的元素，使特定元素成为第一个元素     |
| `is_sorted     | 检查区域中的元素是否按非降序排序            |
| `copy`          | 复制特定范围的元素                   |
| `fill`          | 为范围内的所有元素分配指定的值             |
| `for_each`      | 对区间内的每个元素执行操作               |
| `transform`     | 将函数应用于区间中的每个元素，并将结果存储在另一个区间 |
| `replace`       | 将范围内出现的所有特定值替换为新值           |
| `merge`         | 将两个有序区间合并到一个有序区间            |
| `swap`          | 交换两个变量的值                    |
| `remove`        | 从范围中删除具有指定值的所有元素，但不减小容器大小   |
| `unique`        | 从范围中删除连续的重复元素。              |
| `count`         | 计算范围内给定元素的出现次数              |
| `find`          | 返回范围中第一次出现的元素的迭代器           |
| `max_element` | 查找给定范围内的最大元素                |
| `min_element` | 查找给定范围内的最小元素                |

注意 `std::sort()` 不能配合链表使用，链表提供了自己的 `sort()` 成员函数，它的效率要比泛型的排序高的多。

```cpp
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
    vector<int> vec = { 5, 10, 15 };

    sort(vec.begin(), vec.end());

    for_each(first, last, [](int& x) { x += 1; });

    // Max element in vector
    int max = *max_element(vec.begin(), vec.end());

    return 0;
}
```

# numeric

C++ 标准库中的 `<numeric>` 头文件提供了一组用于数值计算的函数模板，这些函数可以对容器中的元素进行各种数值操作。

| Function              | Description      |
| --------------------- | ---------------- |
| `accumulate`          | 计算容器中所有元素的总和     |
| `partial_sum`         | 计算容器中元素的累积和      |
| `inner_product`       | 计算两个容器中对应元素乘积的总和 |
| `adjacent_difference` | 计算容器中相邻元素的差值     |
| `gcd`                 | 计算两个整数的最大公约数     |
| `lcm`                 | 计算两个整数的最小公倍数     |
| `iota`                | 填充范围内的序列值        |

```cpp
#include <numeric>
#include <vector>
using namespace std;

int main() {

    std::vector<int> v = {3, 1, 4, 1, 5, 9};

    // Defining range as whole array
    auto first = vec.begin();
    auto last = vec.end();

    // Use accumulate to find the sum of elements in the vector
    int sum = accumulate(first, last, 0);

    return 0;
}
```

使用 `std::iota` 填充范围内的序列值

```cpp
#include <iostream>
#include <numeric> 
#include <vector>
using namespace std;

int main() {
    vector<int> v(5);

    // Using std::iota() to initialize vector v with 11
    iota(v.begin(), v.end(), 11);

    for (auto i : v)
        cout << i << " ";
    return 0;
}
```

**Output:**

```cpp
11 12 13 14 15 
```

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

`<regex>` 头文件提供了正则表达式的功能

| Function Name     | Description        |
| ----------------- | ------------------ |
| `regex`           | 表示一个正则表达式对象        |
| `regex_match`     | 检查整个字符串是否与正则表达式匹配  |
| `regex_search`    | 在字符串中搜索与正则表达式匹配的部分 |
| `regex_replace`   | 替换字符串中与正则表达式匹配的部分  |
| `sregex_iterator` | 迭代器，用于遍历所有匹配项      |

```cpp
#include <iostream>
#include <string>
#include <regex>

int main() {
    std::string text = "Hello, World!";
    std::regex pattern("World");
    std::string replacement = "Universe";

    std::string result = std::regex_replace(text, pattern, replacement);

    std::cout << "Original: " << text << std::endl;
    std::cout << "Modified: " << result << std::endl;

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
- `std::mutex` 用于同步对共享资源的访问
- `std::unique_lock` 锁管理器，用于自动管理锁的生命周期。
- `std::condition_variable` 用于线程间的等待和通知。
- `std::future` 和 `std::promise` 用于线程间的结果传递。
- `std::async` 用于启动异步任务，并返回一个 `std::future`。

要创建一个线程，需要实例化 `std::thread` 类，并传递一个可调用对象（函数、lambda 表达式或对象的成员函数）作为参数。

创建 `std::thread` 对象后，线程会立即开始执行，你可以调用 `std::thread::join()` 方法来等待线程完成。`join()` 方法会阻塞当前线程，直到被调用的线程完成执行。

当线程执行完毕后，你可以使用 `detach()` 方法来分离线程

```cpp
// C++ program to demonstrate multithreading using three different callables.
#include <iostream>
#include <thread>
using namespace std;

// A dummy function
void foo(int Z)
{
    for (int i = 0; i < Z; i++) {
        cout << "Thread using function pointer as callable\n";
    }
}

// A callable object
class thread_obj {
public:
    void operator()(int x){
        for (int i = 0; i < x; i++)
            cout << "Thread using function object as callable\n";
    }
};

// class definition
class Base {
public:
    // non-static member function
    void foo(){
        cout << "Thread using non-static member function "
                "as callable"
             << endl;
    }
    // static member function
    static void foo1(){
        cout << "Thread using static member function as callable\n";
    }
};

// Driver code
int main(){
    cout << "Threads 1 and 2 and 3 operating independently\n";

    // This thread is launched by using function pointer as callable
    thread th1(foo, 3);

    // This thread is launched by using function object as callable
    thread th2(thread_obj(), 3);

    // Define a Lambda Expression
    auto f = [](int x) {
        for (int i = 0; i < x; i++)
            cout << "Thread using lambda expression as callable\n";
    };

    // This thread is launched by using lambda expression as callable
    thread th3(f, 3);

    // object of Base Class
    Base b;
    thread th4(&Base::foo, &b);
    thread th5(&Base::foo1);

    // Wait for the threads to finish
    // Wait for thread t1 to finish
    th1.join();

    // Wait for thread t2 to finish
    th2.join();

    // Wait for thread t3 to finish
    th3.join();

    // Wait for thread t4 to finish
    th4.join();

    // Wait for thread t5 to finish
    th5.join();

    return 0;
}
```

**Output:**

```cpp
Threads 1 and 2 and 3 operating independently
Thread using function pointer as callable
Thread using function pointer as callable
Thread using function pointer as callable
Thread using lambda expression as callable
Thread using lambda expression as callable
Thread using lambda expression as callable
Thread using static member function as callable
Thread using non-static member function as callableThread using function object as callable
Thread using function object as callable
Thread using function object as callable
```

注意：要编译支持 `std::thread` 的程序，请使用 `g++ -std=c++11 -pthread`。

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

`<cstdio>` 是C 语言标准 I/O 库的 C++ 封装，提供了标准输入输出函数

`<cmath>` 提供了许多基本的数学函数

`<random>` 提供了一组用于生成随机数的工具

`<numbers>` 是 C++20 中引入的一个标准库模块，包含了很多数学常量，涵盖了圆周率、自然对数的底数、黄金比例等常见常数

`<climits>` 和 `<cfloat>` 定义了一组常量，描述了各种数字类型的最小值、最大值和其他相关属性

`<utility>` 包含了一些实用的工具类和函数

- `pair`：一个包含两个元素的容器，通常用于存储和返回两个相关联的值。
- `make_pair`：一个函数模板，用于创建 `pair` 对象。
- `swap`：一个函数模板，用于交换两个对象的值。
- `forward` 和 `move`：用于完美转发和移动语义的函数模板。

`<locale>` 允许程序根据用户的区域设置来处理文本数据，如数字、日期和时间的格式化，以及字符串的比较和排序。这使得编写国际化应用程序变得更加容易。

`<cstdlib>` 提供了各种通用工具函数，包括内存分配、进程控制、环境查询、排序和搜索、数学转换、伪随机数生成等。

- `exit(int status)`: 终止程序执行，并返回一个状态码。
- `system(const char* command)`: 执行一个命令行字符串。
- `malloc(size_t size)`: 分配指定大小的内存。
- `free(void* ptr)`: 释放之前分配的内存。
- `atoi(const char* str)`: 将字符串转换为整数。
- `atof(const char* str)`: 将字符串转换为浮点数。
- `rand()`: 生成一个随机数。
- `srand(unsigned int seed)`: 设置随机数生成器的种子。

`<tuple>` 类模板 `std::tuple` 是一个固定大小的异构值集合，它是 `std::pair`的泛化。`std::tuple`可以存储任意数量的任意类型的成员变量。
- `make_tuple` 创建由参数类型定义的类型的 tuple 对象
- `tie` 创建 tuple 左值引用或将元组解包到单个对象中
- `forward_as_tuple` 创建 转发引用 的 tuple
- `tuple_cat` 通过连接任意数量的元组来创建 tuple
- `get()` 元组访问指定元素
