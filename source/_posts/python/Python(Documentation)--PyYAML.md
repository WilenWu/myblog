---
title: Python(Documentation)--PyYAML
date: 2021-12-20 18:52:12
updated:
tags: [Python, 标记语言, YAML]
categories: 
  - [Python, 文档]
  - [标记语言]
cover: /img/PyYAML.jpg
top_img: /img/yaml-cover.jpg
description: 
---

YAML 是一个可读性高，用来表达资料序列化的格式。特别适合用来表达或编辑数据结构、各种配置文件、倾印调试内容、文件大纲。PyYAML是Python的YAML解析器和发射器。

想了解更多内容，请参考 [PyYAML 官方文档](https://pyyaml.org/wiki/PyYAMLDocumentation)。

<!-- more -->

![](https://gitee.com/WilenWu/images/raw/master/common/yaml-overview.svg)

# 安装和加载

使用 pip 安装方法

```shell
pip install PyYAML
```

一般情况下，只需加载 yaml 即可

```python
import yaml
```

# 导入 YAML

`yaml.load` 将 YAML 文档转换为 Python 对象。`yaml.load` 接受字节字符串、Unicode 字符串、打开的二进制文件对象或打开的文本文件对象。

```python
yaml.load(stream, Loader = Loader)
```

```python
>>> import yaml
>>> stream = """
... - Hesperiidae
... - Papilionidae
... - Apatelodidae
... - Epiplemidae
... """
>>> yaml.load(stream, Loader=yaml.CLoader)
['Hesperiidae', 'Papilionidae', 'Apatelodidae', 'Epiplemidae']
```

Python 通过 open 方式读取YAML文件，再通过 `yaml.load` 将数据转化为列表或字典。
我们先创建一个 YAML 文件 `stream.yaml` 

```yaml
# stream.yaml
name: Vorlin Laruknuzum
sex: Male
class: Priest
title: Acolyte
hp: [32, 71]
sp: [1, 13]
gold: 423
inventory:
- a Holy Book of Prayers (Words of Wisdom)
- an Azure Potion of Cure Light Wounds
- a Silver Wand of Wonder
```

将输入 stream 对象加载为 Python 对象

```python
>>> with open("stream.yaml", mode = "r") as f:
...   stream = f.read()
...
>>> yaml.load(stream, yaml.FullLoader)
{'name': 'Vorlin Laruknuzum', 'sex': 'Male', 'class': 'Priest', 'title': 'Acolyte', 'hp': [32, 71], 'sp': [1, 13], 'gold': 423, 'inventory': ['a Holy Book of Prayers (Words of Wisdom)', 'an Azure Potion of Cure Light Wounds', 'a Silver Wand of Wonder']}
```

> 如果 yaml 中有中文，需要使用 `str.encode('utf-8')`或打开文件时指定 `encoding='utf-8'`

如果字符串或 YAML 文件包含多个文档，则可以使用 `yaml.load_all()` 来解析全部的文档，再从中读取对象中的数据。

```python
yaml.load_all(stream, Loader = Loader)
```

先创建一个多文档的YAML文件 `documents.yaml`

```yaml
# documents.yaml
name: The Set of Gauntlets 'Pauraegen'
description: >
    A set of handgear with sparks that crackle
    across its knuckleguards.
---
name: The Set of Gauntlets 'Paurnen'
description: >
  A set of gauntlets that gives off a foul,
  acrid odour yet remains untarnished.
---
name: The Set of Gauntlets 'Paurnimmen'
description: >
  A set of handgear, freezing with unnatural cold.
```

`yaml.load_all` 方法会生成一个迭代器

```python
>>> with open("documents.yaml", mode = "r") as f:
...   docs = f.read()
...
>>> for doc in yaml.load_all(docs, Loader=yaml.FullLoader):
...   print(doc)
...
{'name': "The Set of Gauntlets 'Pauraegen'", 'description': 'A set of handgear with sparks that crackle across its knuckleguards.\n'}      
{'name': "The Set of Gauntlets 'Paurnen'", 'description': 'A set of gauntlets that gives off a foul, acrid odour yet remains untarnished.\n'}
{'name': "The Set of Gauntlets 'Paurnimmen'", 'description': 'A set of handgear, freezing with unnatural cold.\n'}
```

PyYAML允许构造任何类型的Python对象

```python
>>> yaml.load("""
... none: [~, null]
... bool: [true, false, on, off]
... int: 42
... float: 3.14159
... list: [LITE, RES_ACID, SUS_DEXT]
... dict: {hp: 13, sp: 5}
... """, Loader=yaml.FullLoader)

{'none': [None, None], 'int': 42, 'float': 3.1415899999999999,
'list': ['LITE', 'RES_ACID', 'SUS_DEXT'], 'dict': {'hp': 13, 'sp': 5},
'bool': [True, False, True, False]}
```

{% note warning %}
**注意**：由于 `yaml.load()` 能通过加载yaml文件构建任何类型的python对象，若加载的yaml文件来源不可信，则可能产生注入攻击的风险。
{% endnote %}

如果您不信任输入 YAML 流，则推荐使用 `yaml.safe_load()`，只能识别标准的 YAML 标记，不能构造任意 Python 对象。同样多文档推荐使用 `yaml.safe_load_all()`。

# 转储 YAML

`yaml.dump` 方法接受 Python 对象并生成 YAML stream

```python
dump(data, stream=None)
```

```python
>>> import yaml
>>> data = {'name': 'Silenthand Olleander',
...         'race': 'Human',
...         'traits': ['ONE_HAND', 'ONE_EYE']
...         }
>>>
>>> print(yaml.dump(data))
name: Silenthand Olleander
race: Human
traits:
- ONE_HAND
- ONE_EYE
```

`yaml.dump` 接受第二个可选参数，该参数必须是打开的文本文件或二进制文件。在这种情况下，`yaml.dump` 会将生成的 YAML 文档写入文件。否则，返回生成的文档。

```python
>>> with open('document.yaml', 'w') as stream:
...   yaml.dump(data, stream)
```


如果需要将多个 YAML 文档转储到单个流，请使用 `yaml.dump_all` 。要序列化为 YAML 文档的 Python 对象可以是列表或生成器，第二个可选参数是打开的文件。

```python
>>> print(yaml.dump([1,2,3], explicit_start=True))
---
- 1
- 2
- 3

>>> print(yaml.dump_all([1,2,3], explicit_start=True))
--- 1
--- 2
--- 3
...
```

`safe_dump(data, stream=None)` 将给定的 Python 对象序列化，仅生成标准 YAML 标记，不能表示任意 Python 对象。
同样 `safe_dump_all(data, stream=None) `也可以将多个 Python 对象转为yaml，推荐使用`yaml.safe_dump_all(data, stream=None)`。



