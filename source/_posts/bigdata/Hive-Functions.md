---
title: 大数据手册(Hive)--HiveQL(Functions)
categories:
  - Big Data
  - Hive
tags:
  - 大数据
  - hive
  - SQL
cover: /img/apache-hive-functions.png
top_img: /img/apache-hive-bg.png
description: Hive 常用函数
abbrlink: f18b723f
date: 2018-07-03 17:57:36
---

# 内置函数

参考资料：[Hive 官方文档](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF#LanguageManualUDF-Built-inFunctions)

```sql
SHOW FUNCTIONS [LIKE "pattern"]; -- 列出用户定义和内置函数
DESCRIBE FUNCTION function_name;   -- 查看函数信息
DESCRIBE FUNCTION EXTENDED function_name; 
```

## 数学函数

| Return Type   | Name (Signature)              | Description                                         |
| ------------- | ----------------------------- | --------------------------------------------------- |
| BIGINT        | round(a)                      | 四舍五入取整                                        |
| DOUBLE        | round(a, d)                   | 四舍五入，保留d位小数                               |
| BIGINT        | floor(a)                      | 向下取整                                            |
| BIGINT        | ceil(a), ceiling(a)           | 向上取整                                            |
| DOUBLE        | rand(), rand(seed)            | 返回0到1间的随机数                                  |
| DOUBLE        | exp(a),                       | $e^a$                                               |
| DOUBLE        | ln(a)                         | $\ln a$                                             |
| DOUBLE        | log10(a)                      | $\log_{10}a$                                        |
| DOUBLE        | log2(a)                       | $\log_{2}a$                                         |
| DOUBLE        | log(b, a)                     | $\log_{b}a$                                         |
| DOUBLE        | pow(a, p)                     | $a^p$                                               |
| DOUBLE        | sqrt(a)                       | $\sqrt{a}$                                          |
| STRING        | bin(a)                        | 返回二进制数字对应的字段                            |
| STRING        | hex(a), hex(a), hex(BINARY a) | 返回十六进制数字对应的字段                          |
| BINARY        | unhex(a)                      | hex的逆方法                                         |
| DOUBLE        | abs(a)                        | 绝对值                                              |
| INT or DOUBLE | pmod(a, b), pmod(a, b)        | $a \mod b$                                          |
| DOUBLE        | sin(a)                        | 正弦值                                              |
| DOUBLE        | asin(a)                       | 反正弦值                                            |
| DOUBLE        | cos(a)                        | 余弦值                                              |
| DOUBLE        | acos(a)                       | 反余弦值                                            |
| DOUBLE        | tan(a)                        | 正切值                                              |
| DOUBLE        | atan(a)                       | 反正切值                                            |
| DOUBLE        | degrees(a)                    | 将弧度值转换角度值                                  |
| DOUBLE        | radians(a)                    | 将角度值转换成弧度值                                |
| INT or DOUBLE | positive(a), positive(a)      | 返回a                                               |
| INT or DOUBLE | negative(a), negative(a)      | 返回-a                                              |
| DOUBLE or INT | sign(a)                       | 如果a是正数则返回1.0，是负数则返回-1.0，否则返回0.0 |
| DOUBLE        | e()                           | 数学常数e                                           |
| DOUBLE        | pi()                          | 数学常数$\pi$                                       |
| BIGINT        | factorial(a)                  | 求a的阶乘                                           |
| DOUBLE        | cbrt(a)                       | 求a的立方根                                         |
| TYPE          | greatest(T v1, T v2, ...)     | 求最大值                                            |
| TYPE          | least(T v1, T v2, ...)        | 求最小值                                            |

## 集合函数

| Return Type | Name(Signature)                            | Description                                                  |
| ----------- | ------------------------------------------ | ------------------------------------------------------------ |
| int         | `size(Map)`                                | 返回map的长度                                                |
| int         | `size(Array)`                              | 返回数组的长度                                               |
| boolean     | map_contains_key(map, key)                 | map的所有键是否包含key                                       |
| array       | map_keys(Map)                              | 返回map中的所有key                                           |
| array       | map_values(Map)                            | 返回map中的所有value                                         |
| boolean     | `array_contains(Array, value)`             | 数组中是否包含value                                          |
| array       | `sort_array(Array)`                        | 对数组进行排序并返回                                         |
| string      | `concat_ws(SEP, array)`                    | Array中的元素拼接                                            |
| array       | sentences(str, lang, locale)               | 字符串str将被转换成单词数组                                  |
| array       | split(str, pat)                            | 按照正则表达式pat来分割字符串str                             |
| map         | str_to_map(text[, delimiter1, delimiter2]) | 将字符串str按照指定分隔符转换成Map，第一个参数是需要转换字符串，第二个参数是键值对之间的分隔符，默认为逗号；第三个参数是键值之间的分隔符，默认为"=" |
| ARRAY       | collect_set(col)                           | 返回一组消除了重复元素的对象                                 |
| ARRAY       | collect_list(col)                          | 返回具有重复项的对象列表                                     |

```sql
SELECT str_to_map('a:1,b:2,c:3', ',', ':');
+-----------------------------+
|str_to_map(a:1,b:2,c:3, ,, :)|
+-----------------------------+
|         {a -> 1, b -> 2, ...|
+-----------------------------+
```

## 类型转换函数

| Return Type | Name(Signature)        | Description          |
| ----------- | ---------------------- | -------------------- |
| binary      | binary(string\|binary) | 转换成二进制         |
| type        | `cast(expr as <type>)` | 将expr转换成type类型 |

## 日期函数

| Return Type     | Name(Signature)                                 | Description                                                  |
| --------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| string          | from_unixtime(unixtime[, format])               | 将Unix时间戳 (1970-01-0100:00:00 UTC 为起始秒) 转化为时间字符 |
| bigint          | unix_timestamp()                                | 获取本地时区下的时间戳                                       |
| bigint          | unix_timestamp(date)                            | 将格式为 yyyy-MM-dd HH:mm:ss 的时间字符串转换成时间戳        |
| bigint          | unix_timestamp(date, fmt)                       | 将指定时间字符串格式字符串转换成Unix时间戳                   |
| string<br> date | to_date(timestamp) <br>to_date(date_str[, fmt]) | 返回时间字符串的日期部分 <br> spark中可以将 fmt 格式的字符串转化为日期 |
| timestamp       | to_timestamp(timestamp_str[, fmt])              | spark中可以将 fmt 格式的字符串转化为时间戳                   |
| int             | year(date)                                      | 年份部分                                                     |
| int             | quarter(date/timestamp/string)                  | 季度部分                                                     |
| int             | month(date)                                     | 月份部分                                                     |
| int             | dayofyear(date)                                 | the day of year                                              |
| int             | dayofweek(date)                                 | the day of week                                              |
| int             | dayofmonth(date)                                | the day of month                                             |
| int             | day(date)                                       | 天                                                           |
| int             | hour(date)                                      | 小时                                                         |
| int             | minute(date)                                    | 分钟                                                         |
| int             | second(date)                                    | 秒                                                           |
| int             | weekofyear(date)                                | 一年中的第几个周内                                           |
| int             | extract(field FROM source)                      | 提取日期组件                                                 |
| int             | datediff(enddate, startdate)                    | 相差的天数                                                   |
| string          | date_add(startdate, days)                       | 从开始时间startdate加上days                                  |
| string          | date_sub(startdate, days)                       | 从开始时间startdate减去days                                  |
| timestamp       | from_utc_timestamp(timestamp, timezone)         | 如果给定的时间戳并非UTC，则将其转化成指定的时区下时间戳      |
| timestamp       | to_utc_timestamp(timestamp, timezone)           | 如果给定的时间戳指定的时区下时间戳，则将其转化成UTC下的时间戳 |
| date            | current_date                                    | 返回当前时间日期                                             |
| timestamp       | current_timestamp                               | 返回当前时间戳                                               |
| string          | add_months(start_date, num_months)              | 返回当前时间下再增加num_months个月的日期                     |
| string          | last_day(date)                                  | 返回这个月的最后一天的日期，忽略时分秒部分（HH:mm:ss）       |
| string          | next_day(start_date, day_of_week)               | 返回当前时间的下一个星期X所对应的日期                        |
| string          | trunc(date, format)                             | 返回时间的最开始年份或月份。注意所支持的格式为MONTH/MON/MM, YEAR/YYYY/YY |
| double          | months_between(date1, date2)                    | 返回date1与date2之间相差的月份                               |
| string          | date_format(date/timestamp/ts, fmt)             | 按指定[Format][dt]返回日期字符                               |

[dt]: https://docs.oracle.com/javase/7/docs/api/java/text/SimpleDateFormat.htm

 ```sql
select extract(month from "2016-10-20") -- results in 10.
select extract(hour from "2016-10-20 05:06:07") -- results in 5.
select extract(dayofweek from "2016-10-20 05:06:07") -- results in 5.
select extract(month from interval '1-3' year to month) -- results in 3.
select extract(minute from interval '3 12:20:30' day to second) -- results in 20
SELECT to_date('2016-12-31', 'yyyy-MM-dd') -- results in 2016-12-31
 ```

## 条件函数

| Return Type | Name(Signature)                                            | Description                                      |
| ----------- | ---------------------------------------------------------- | ------------------------------------------------ |
| TYPE        | if(boolean testCondition, T valueTrue, T valueFalseOrNull) | 二分支语句                                       |
| TYPE        | nvl(T value, T default_value)                              | 返回第一个不是NULL的参数                         |
| TYPE        | COALESCE(T v1, T v2, ...)                                  | 返回第一个不是NULL的参数                         |
| TYPE        | CASE a WHEN b THEN c [WHEN d THEN e]* [ELSE f] END         | 多分支语句                                       |
| TYPE        | CASE WHEN a THEN b [WHEN c THEN d]* [ELSE e] END           | 多分支语句                                       |
| boolean     | isnull( a )                                                | 是否NULL                                         |
| boolean     | isnotnull ( a )                                            | 是否非NULL                                       |
| TYPE        | nullif( a, b )                                             | 如果 a=b，则返回 NULL；否则返回 a                |
| void        | assert_true(boolean condition)                             | 如果 condition 不为真，则抛出异常，否则返回 null |

## 字符函数

| Return Type | Name(Signature)                                      | Description                                                  |
| ----------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| string      | A \|\| B                                             | 字符串连接（从HIve 2.2开始支持）                             |
| string      | concat(A, B...)                                      | 字符串连接                                                   |
| string      | concat_ws(SEP, A, B...)                              | 指定分隔符拼接                                               |
| string      | `concat_ws(SEP, array<string>)`                      | Array中的元素拼接                                            |
| string      | elt(N,str1, str2, ...)                               | 返回索引号处的字符串。例如 elt(2,'hello','world') 返回 'world'。如果 N 小于 1 或大于参数数量，则返回 NULL。 |
| int         | field(val T,val1 T,val2 T,val3 T,...)                | 返回 val1,val2,val3,... 参数列表中 val 的索引，如果未找到则返回 0。例如field ('world','say','hello','world') 返回 3。 |
| int         | find_in_set(str, strList)                            | 返回以逗号分隔的字符串中str出现的位置，如果参数str为逗号或查找失败将返回0 |
| string      | format_number(number x, d)                           | 数字转字符串                                                 |
| string      | get_json_object(json_string, path)                   |                                                              |
| boolean     | in_file(str, filename)                               | 在文件中查找字符串                                           |
| int         | instr(str, substr)                                   | 查找子字符串substr出现的位置，如果查找失败将返回0            |
| int         | length(A)                                            | 字符串的长度                                                 |
| int         | locate(substr, str[, pos])                           | 查找字符串str中的pos位置后字符串substr第一次出现的位置       |
| string      | lower(A) <br />lcase(A)                              | 小写                                                         |
| string      | lpad(str, len, pad)                                  | 在左端填充字符串 pad，长度为len                              |
| string      | ltrim(A)                                             | 去掉左边空格                                                 |
| string      | parse_url(url, part [, key])                         | 从URL返回指定的部分，part的有效值包括 HOST, PATH, QUERY, REF, PROTOCOL, AUTHORITY, FILE, and USERINFO |
| tuple       | parse_url_tuple(url, p1, p2, ...)                    | 同时提取URL的多个部分                                        |
| string      | regexp_extract(subject, pattern, index)              | 抽取字符串subject中符合正则表达式pattern的第index个部分的子字符串 |
| string      | regexp_replace(INITIAL_STRING, PATTERN, REPLACEMENT) | 按照Java正则表达式PATTERN将字符串INTIAL_STRING中符合条件的部分成REPLACEMENT所指定的字符串 |
| string      | repeat(str, n)                                       | 重复输出n次字符串str                                         |
| string      | reverse(A)                                           | 反转字符串                                                   |
| string      | rpad(str, len, pad)                                  | 在右端填充字符串 pad，长度为len                              |
| string      | rtrim(A)                                             | 去掉右边空格                                                 |
| array       | sentences(str, lang, locale)                         | 字符串str将被转换成单词数组                                  |
| string      | space(n)                                             | 返回n个空格                                                  |
| array       | split(str, pat)                                      | 按照正则表达式pat来分割字符串str                             |
| map         | str_to_map(text[, delimiter1, delimiter2])           | 将字符串str按照指定分隔符转换成Map，第一个参数是需要转换字符串，第二个参数是键值对之间的分隔符，默认为逗号;第三个参数是键值之间的分隔符，默认为"=" |
| string      | substr(A, start) <br />substring(A, start)           | 提取子字符串                                                 |
| string      | substr(A, start, len) <br />substring(A, start, len) | 提取长度为len的子字符串                                      |
| string      | substring_index(A, delim, count)                     | 截取第count分隔符之前的字符串，如count为正则从左边开始截取，如果为负则从右边开始截取 |
| string      | translate(input, from, to)                           | 字符串替换                                                   |
| string      | trim(A)                                              | 去掉两边空格                                                 |
| string      | upper(A) <br />ucase(A)                              | 大写                                                         |
| string      | initcap(A)                                           | 首字母大写                                                   |
| int         | levenshtein(A, B)                                    | 计算两个字符串之间的差异大小                                 |

```sql
hive> select parse_url('http://facebook.com/path1/p.php?k1=v1&k2=v2#Ref1', 'HOST');
facebook.com
hive> select parse_url('http://facebook.com/path1/p.php?k1=v1&k2=v2#Ref1', 'QUERY', 'k1');
v1
hive> SELECT b.* FROM src LATERAL VIEW parse_url_tuple(fullurl, 'HOST', 'QUERY:k1', 'QUERY:k2') b as host, query_k1, query_k2;
```

```sql
-- json 示例
                               json
+----+
{"store":
  {"fruit":\[{"weight":8,"type":"apple"},{"weight":9,"type":"pear"}],
   "bicycle":{"price":19.95,"color":"red"}
  },
 "email":"amy@only_for_json_udf_test.net",
 "owner":"amy"
}
+----+

hive> SELECT get_json_object(src_json.json, '$.owner') FROM src_json;
amy
 
hive> SELECT get_json_object(src_json.json, '$.store.fruit\[0]') FROM src_json;
{"weight":8,"type":"apple"}
```

## 数据掩码函数

| Return Type | Name(Signature)                       | Description                                                  |
| :---------- | :------------------------------------ | :----------------------------------------------------------- |
| string      | mask(str[, upper[, lower[, number]]]) | 返回 str 的掩码版本。默认情况下，大写字母转换为“X”，小写字母转换为“x”，数字转换为“n”。 |
| string      | mask_first_n(str[, n])                | 返回前 n 个值被屏蔽的掩码版本。                              |
| string      | mask_last_n(str[, n])                 | 返回后 n 个值被屏蔽的掩码版本。                              |
| string      | mask_show_first_n(str[, n])           | 返回前 n 个值未被屏蔽的掩码版本。                            |
| string      | mask_show_last_n(str[, n])            | 返回后 n 个值未被屏蔽的掩码版本。                            |
| string      | mask_hash(str)                        | 返回hash掩码                                                 |

## Misc. Functions

| Return Type | Name(Signature)                                     | Description                           |
| :---------- | :-------------------------------------------------- | :------------------------------------ |
| int         | hash(a1[, a2...])                                   | 返回参数的哈希值                      |
| string      | current_user()                                      | 返回当前用户名                        |
| string      | logged_in_user()                                    | 从会话状态返回当前用户名              |
| string      | current_database()                                  | 返回当前数据库名称                    |
| string      | md5(string/binary)                                  | 返回MD5编码                           |
| string      | sha1(string/binary)<br />sha(string/binary)         | 计算SHA-1 摘要                        |
| bigint      | crc32(string/binary)                                |                                       |
| string      | sha2(string/binary, int)                            | 计算SHA-2 系列摘要                    |
| binary      | aes_encrypt(input string/binary, key string/binary) | 使用 AES 加密                         |
| binary      | aes_decrypt(input binary, key string/binary)        | 使用 AES 解密                         |
| string      | version()                                           | 返回 Hive 版本                        |
| bigint      | surrogate_key([write_id_bits, task_id_bits])        | 将数据输入表格时，自动为行生成数字 ID |

## 聚合函数

| Return Type               | Name(Signature)                                   | Description                                                  |
| ------------------------- | ------------------------------------------------- | ------------------------------------------------------------ |
| BIGINT                    | count(*)                                          | 统计总行数，包括含有NULL值的行                               |
| BIGINT                    | count([DISTINCT ] expr, ...)                      | 统计提供非NULL的expr表达式值的行数                           |
| DOUBLE                    | sum(col)                                          | 求和                                                         |
| DOUBLE                    | sum(DISTINCT col)                                 | 去重后求和                                                   |
| DOUBLE                    | avg(col)                                          | 平均值                                                       |
| DOUBLE                    | avg(DISTINCT col)                                 | 去重后平均值                                                 |
| DOUBLE                    | min(col)                                          | 最小值                                                       |
| DOUBLE                    | max(col)                                          | 最大值                                                       |
| DOUBLE                    | variance(col)<br />var_pop(col)                   | 方差                                                         |
| DOUBLE                    | var_samp(col)                                     | 样本方差                                                     |
| DOUBLE                    | stddev_pop(col)                                   | 标准偏差                                                     |
| DOUBLE                    | stddev_samp(col)                                  | 样本标准偏差                                                 |
| DOUBLE                    | covar_pop(col1, col2)                             | 协方差                                                       |
| DOUBLE                    | covar_samp(col1, col2)                            | 样本协方差                                                   |
| DOUBLE                    | corr(col1, col2)                                  | 相关系数                                                     |
| DOUBLE                    | percentile(col, p)                                | 返回col的p分位数（不适用于浮点类型）                         |
| ARRAY                     | percentile(col, array(p1 [, p2]...))              | 与上面相同，接收并返回数组                                   |
| DOUBLE                    | percentile_approx(col, p [, B])                   | 返回col的近似p分位数（包括浮点类型），B 参数控制近似精度。较高的值会产生更好的近似值，默认值为 10,000。当 col 中不同值的数量小于 B 时，这会给出精确的百分位值。 |
| ARRAY                     | percentile_approx(col, array(p1 [, p2]...) [, B]) | 与上面相同，接收并返回数组                                   |
| `array<struct {'x','y'}>` | histogram_numeric(col, b)                         | 使用 b 个非均匀间隔的 bin 计算组中数字列的直方图。输出是一个大小为 b 的双值 (x,y) 坐标数组，表示 bin 中心和高度 |
| ARRAY                     | collect_set(col)                                  | 行收集成数组，消除重复元素                                   |
| ARRAY                     | collect_list(col)                                 | 行收集成数组，具有重复项                                     |
| INTEGER                   | ntile(INTEGER x)                                  |                                                              |

```sql
hive> create table as student
    > select id, course, score
    > from VALUES
    > ('001', 'Chinese', 87 ),
    > ('001', 'Math'   , 87 ),
    > ('001', 'English', 92 ),
    > ('002', 'Chinese', 89 ),
    > ('002', 'Math'   , 95 ),
    > ('002', 'English', 93 ),
    > ('003', 'Chinese', 93 ),
    > ('003', 'Math'   , 82 ),
    > ('003', 'English', 87 ),
    > ('004', 'Chinese', 86 ),
    > ('004', 'Math'   , 86 ),
    > ('004', 'English', 100 )
    >  as tab(id, course, score);
hive> select id,collect_list(score) as score 
	> from student
	> group by id;
id	score
001	[90,95,80]
002	[70,65,83]
... ...
```

## 表生成函数

| Return Type | Name(Signature)                  | Description                                                  |
| ----------- | -------------------------------- | ------------------------------------------------------------ |
| N rows      | explode(ARRAY a)                 | 将数组a分解为单列，每行对应数组中的每个元素                  |
| N rows      | explode(MAP m)                   | 将映射m分解为两列，每行对应每个key-value对                   |
| N rows      | posexplode(ARRAY a)              | 与explode类似，不同的是还返回一列各元素在数组中的位置        |
| N rows      | stack(n, v_1, v_2, ..., v_k)     | 将k列转换为n行，每行有k/n个字段                              |
| tuple       | json_tuple(jsonStr, k1, k2, ...) | 从一个JSON字符串中获取多个键并作为一个元组返回，与get_json_object不同的是此函数能一次获取多个键值 |
| N rows      | `inline(ARRAY<STRUCT>)`          | 将结构数组分解为多行，数组中每个结构体一行                   |

```sql
-- 创建示例表
create table student as 
select id, Chinese, English, Math
from VALUES 
('001', 87, 92, 87 ),
('002', 89, 93, 95 ),
('003', 93, 87, 82 ),
('004', 86, 100, 86 )
as tab(id, Chinese, English, Math);
```

```sql
-- 数组分解成单列，并聚合运算
hive> SELECT a.id, sum(tf.score) FROM 
    > (select id,array(chinese,math,english) as sc from student) a
    > LATERAL VIEW explode(sc) tf AS score GROUP BY a.id; 
id      _c1
001     266
002     277
003     262
004     272

-- 映射分解成两列，重命名时逗号分隔（union语句替代品）
hive> select id,course,score from
    > (select id,map('chinese',chinese,'math',math,'english',english) as dict 
       from student) a
    > LATERAL VIEW explode(dict) tf AS course,score; 
id      course  score
001     chinese 87
001     math    87
001     english 92
... ...
```

```sql
-- inline (array of structs)
select inline(array(
    struct('A',10,date '2015-01-01'),
    struct('B',20,date '2016-02-02')
));
select inline(array(
    struct('A',10,date '2015-01-01'),
    struct('B',20,date '2016-02-02'))) as (col1,col2,col3);
select tf.* 
from (select 0) t 
lateral view inline(array(
    struct('A',10,date '2015-01-01'),
    struct('B',20,date '2016-02-02'))) tf;
select tf.* 
from (select 0) t 
lateral view inline(array(
    struct('A',10,date '2015-01-01'),
    struct('B',20,date '2016-02-02'))) tf as col1,col2,col3;
```

| col1 | col2 | col3       |
| ---- | ---- | ---------- |
| A    | 10   | 2015-01-01 |
| B    | 20   | 2016-02-02 |


# 自定义函数

Hive 中有三种 UDF

1. 用户定义函数(user-defined function,UDF)：作用于单个数据行，并且产生一个数据行作为输出。大多数函数都属于这一类（比如数学函数和字符串函数）。 
2. 用户定义聚合函数（user-defined aggregate function ,UDAF ）：接受多个输入数据行，并产生一个输出数据行。例如聚合函数。
3. 用户定义表生成函数（user-defined table-generating function,UDTF ）：作用于单个数据行，并且产生一个表（多个数据行）作为输出。

## 临时函数

**创建临时函数**

```sql
CREATE TEMPORARY FUNCTION function_name AS class_name;
```

此语句允许您创建一个由 class_name 实现的函数 （UDF）。只要会话持续，您就可以在 Hive 查询中使用此函数。

**删除临时函数**

```sql
DROP TEMPORARY FUNCTION [IF EXISTS] function_name;
```

## 永久函数

在 Hive 0.13 或更高版本中，函数可以注册到 Metastore，因此可以在查询中引用它们，而无需在每个会话中创建临时函数。

**创建函数**

```sql
CREATE FUNCTION [db_name.]function_name AS class_name
  [USING JAR|FILE|ARCHIVE 'file_uri' [, JAR|FILE|ARCHIVE 'file_uri'] ];
```

此语句允许您创建一个由 class_name 实现的函数。可以使用 USING 子句指定需要添加到环境中的 jar、文件或档案；当 Hive 会话第一次引用该函数时，这些资源将被添加到环境中，就像已发出 ADD JAR/FILE一样。如果 Hive 未处于本地模式，则资源位置必须是非本地 URI，例如 HDFS 位置。

该函数将被添加到指定的数据库中，或在创建该函数时添加到当前数据库中。可以通过完全限定函数名称 (db_name.function_name) 来引用该函数，如果该函数在当前数据库中，则可以不加限定地引用该函数。

**删除函数**

```sql
DROP FUNCTION [IF EXISTS] function_name;
```

**重新加载**

```sql
RELOAD (FUNCTIONS|FUNCTION);
```

在 HiveServer2 或 HiveCLI 已打开的会话中使用 RELOAD FUNCTIONS 将允许它获取对不同 HiveCLI 会话可能已完成的永久函数的任何更改。

# 窗口函数和分析函数

## OVER 子句

OVER 子句支持函数定义查询结果集内的窗口，函数计算窗口中的值。OVER 子句支持标准聚合函数 (SUM, AVG, MAX, MIN, COUNT) 、窗口函数（Window Function）和分析函数（Analytics Function）。

**OVER 子句语法**

```sql
func(args) OVER([PARTITION BY expr, ...] [ORDER BY expr [ASC | DESC], ...] [window_clause])
```

- 当 ORDER BY 指定时缺少 WINDOW 子句，WINDOW 规范默认为 `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`
- 当 ORDER BY 和 WINDOW 子句都缺失时，WINDOW 规范默认为 `ROW BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`
- 窗口函数和分析函数使用 OVER 子句时不支持 WINDOW 子句。

**WINDOW 子句语法**

```sql
window_clause: 
(ROWS | RANGE) BETWEEN frame_bound AND frame_bound 
```

- WINDOW 子句对窗口进行进一步的分区，有两种范围限定方式：一种是使用ROWS子句，通过指定当前行之前或之后的固定数目的行来限制分区中的行数；另一种是RANGE子句，按照排序列的当前值，根据值的范围来确定。
- WINDOW 规范：

| frame_bound         | 说明                         |
| ------------------- | ---------------------------- |
| UNBOUNDED PRECEDING | 第一行                       |
| [num] PRECEDING     | 在当前行前面第num行，默认为1 |
| CURRENT ROW         | 当前行                       |
| [num] FOLLOWING     | 在当前行后面第num行，默认为1 |
| UNBOUNDED FOLLOWING | 最后一行                     |

**示例**

OVER 子句支持标准聚合函数：能够在同一行中返回原始行的同时返回聚合值。等价于原表 GROUP BY 聚合后再 JOIN 原表。以购物表 order_detail 为例

```sql
hive> SELECT * FROM order_detail;
user_id user_type sales
001   new    1
002   new    1
003   new    2
004   new    3
005   new    5
006   new    5
007   new    6
008   old    1
009   old    2
010   old    3
```

**OVER() 子句无参数**：`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`，先累计汇总，然后匹配源数据。

```sql
hive> SELECT *, SUM(sales) OVER() FROM order_detail;
user_id user_type       sales   sum_window_0
010   old     3       29.0
009   old     2       29.0
008   old     1       29.0
007   new     6       29.0
006   new     5       29.0
005   new     5       29.0
004   new     3       29.0
003   new     2       29.0
002   new     1       29.0
001   new     1       29.0
```

**只有PARTITION BY 参数**：`ROW BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`，按照指定列分区汇总

```sql
hive> SELECT *, SUM(sales) OVER(PARTITION BY user_type) FROM order_detail;
user_id user_type       sales   sum_window_0
007  new     6       23.0
006  new     5       23.0
005  new     5       23.0
004  new     3       23.0
003  new     2       23.0
002  new     1       23.0
001  new     1       23.0
010  old     3       6.0
009  old     2       6.0
008  old     1       6.0
```

**只有ORDER BY 参数**：`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`，按照指定列进行排序，然后累计汇总。若第 n 行的值唯一，返回累计到第 n 行的值即可。若第 n 行的值不唯一，累计到所有和第 n 行值相同的行，返回累计值。

```sql
hive> SELECT *, SUM(sales) OVER(ORDER BY sales) FROM order_detail;
user_id user_type       sales   sum_window_0
008    old     1       3.0
002    new     1       3.0
001    new     1       3.0
009    old     2       7.0
003    new     2       7.0
010    old     3       13.0
004    new     3       13.0
006    new     5       23.0
005    new     5       23.0
007    new     6       29.0
```

**PARTITION BY ... ORDER BY**：`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`，按照指定列分区，在分区内排序然后累计汇总。

```sql
hive> SELECT *, SUM(sales) OVER(PARTITION BY user_type ORDER BY sales) FROM order_detail;
user_id user_type       sales   sum_window_0
002  new     1       2.0
001  new     1       2.0
003  new     2       4.0
004  new     3       7.0
006  new     5       17.0
005  new     5       17.0
007  new     6       23.0
008  old     1       1.0
009  old     2       3.0
010  old     3       6.0
```

**WINDOW 子句**

`OVER`一个查询中可以有多个 子句。单个 `OVER`子句仅适用于前一个函数调用。

```sql
hive> SELECT *, 
	> sum(sales) OVER(ORDER BY sales asc),
	> -- 当前行及之前所有行
	> SUM(sales) OVER(ORDER BY sales ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
	> -- 当前行+往前3行
	> SUM(sales) OVER(ORDER BY sales ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
	> FROM order_detail;
user_id user_type       sales   sum_window_0    sum_window_1    sum_window_2
008    old     1       3.0     1.0     1.0
002    new     1       3.0     2.0     2.0
001    new     1       3.0     3.0     3.0
009    old     2       7.0     5.0     5.0
003    new     2       7.0     7.0     6.0
010    old     3       13.0    10.0    8.0
004    new     3       13.0    13.0    10.0
006    new     5       23.0    18.0    13.0
005    new     5       23.0    23.0    16.0
007    new     6       29.0    29.0    19.0
```

```sql
-- 滚动平均值
SELECT sale_date, 
	AVG(sales) OVER(ORDER BY sales ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
FROM order_detail;
```

## 窗口函数

| 偏移取值                     | 说明                                                         |
| :--------------------------- | :----------------------------------------------------------- |
| FIRST_VALUE(expr, skip_null) | 第一个值，skip_null设置是否跳过空值，默认false               |
| LAST_VALUE(expr, skip_null)  | 最后一个值，skip_null设置是否跳过空值，默认false             |
| LEAD(expr, n, default)       | 返回往下第n行值，默认为1。超出当前窗口时返回default，默认为NULL |
| LAG(expr, n, default)        | 返回往上第n行值，默认为1。超出当前窗口时返回default，默认为NULL |

## 分析函数

| 数据排名       | 说明                                                  |
| :------------- | :---------------------------------------------------- |
| ROW_NUMBER()   | 行序号                                                |
| RANK()         | 排名，排名相等则下一个序号间断                        |
| DENSE_RANK()   | 排名，排名相等则下一个序号不间断                      |
| PERCENT_RANK() | 百分比排名，(分组内当前行的RANK值-1)/(分组内总行数-1) |

```sql
hive> SELECT 
    user_id,user_type,sales,
    RANK() over (partition by user_type order by sales desc) as r,
    ROW_NUMBER() over (partition by user_type order by sales desc) as rn,
    DENSE_RANK() over (partition by user_type order by sales desc) as dr
FROM
    order_detail; 
user_id user_type       sales   r       rn      dr
007  new     6       1       1       1
006  new     5       2       2       2
005  new     5       2       3       2
004  new     3       4       4       3
003  new     2       5       5       4
002  new     1       6       6       5
001  new     1       6       7       5
010  old     3       1       1       1
009  old     2       2       2       2
008  old     1       3       3       3
```

| 累计分布    | 说明                              |
| :---------- | :-------------------------------- |
| CUME_DIST() | 小于等于当前值的行数/分组内总行数 |
| NTILE(n)    | 切成n个桶，返回当前切片号         |

```sql
hive> SELECT 
    user_id,user_type,sales,
	NTILE(2) OVER(ORDER BY sales),
	NTILE(3) OVER(ORDER BY sales)
FROM 
    order_detail;
user_id user_type       sales   NTILE_window_0  NTILE_window_1
008   old     1       1       1
002   new     1       1       1
001   new     1       1       1
009   old     2       1       1
003   new     2       1       2
010   old     3       2       2
004   new     3       2       2
006   new     5       2       3
005   new     5       2       3
007   new     6       2       3
```