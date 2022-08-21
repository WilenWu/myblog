---
title: NexT主题标签插件的使用
categories:
  - Blog
tags:
  - Hexo
cover: /img/hexo-next-theme.png
top_img: /img/hexo-top-img.png
noticeOutdate: true
description: false
abbrlink: 7a9ce795
date: 2019-09-11 14:10:48
---

{% note warning%} 
以下的写法，只适用于NexT主题，用在其它主题上不会有效果，甚至可能会报错。使用前请留意
{% endnote %}

<!--more-->

## 文本居中引用

此标签将生成一个带上下分割线的引用，同时引用内文本将自动居中。 文本居中时，多行文本若长度不等，视觉上会显得不对称，因此建议在引用单行文本的场景下使用。 例如作为文章开篇引用 或者 结束语之前的总结引用

使用方式

- HTML方式：直接在 Markdown 文件中编写 HTML 来调用，给 `img` 添加属性 `class="blockquote-center"` 即可。
- 标签方式：使用 `centerquote` 或者 简写 `cq`。

```
# HTML方式
<blockquote class="blockquote-center">blah blah blah</blockquote>

# 标签方式
{% centerquote %}blah blah blah{% endcenterquote %}
```

## 突破容器宽度限制的图片

当使用此标签引用图片时，图片将自动扩大 26%，并突破文章容器的宽度。 此标签使用于需要突出显示的图片, 图片的扩大与容器的偏差从视觉上提升图片的吸引力。 此标签有两种调用方式（详细参看底下示例）：

使用方式

- HTML方式：直接在 Markdown 文件中编写 HTML 来调用，为 `img` 添加属性 `class="full-image"`即可。
- 标签方式：使用 `fullimage` 或者 简写 `fi`， 并传递图片地址、 `alt` 和 `title` 属性即可。 属性之间以逗号分隔。

```
# HTML方式:
<img src="/image-url" class="full-image" />

# 标签 方式
{% fullimage /image-url, alt, title %}
```

## Note 标签

**使用方式**

```
{% note [class] [no-icon] %} 
content (support inline tags too.io).
{% endnote %}
```

| 名称    | 用法                                                         |
| :------ | :----------------------------------------------------------- |
| class   | 【可选】标识，不同的标识有不同的配色<br/>（ default / primary / success / info / warning / danger ） |
| no-icon | 【可选】不显示 icon                                          |

效果如下
{% note default %} default {% endnote %}
{% note default no-icon %} default no-icon {% endnote %}
{% note primary %} primary  {% endnote %}
{% note success %} success {% endnote %}
{% note info %} info  {% endnote %}
{% note warning %} warning {% endnote %}
{% note danger %} danger{% endnote %}

## Tabs 选项卡

使用方式

```
{% tabs Unique name, [index] %}
<!-- tab [caption] [@icon] -->
**This is Tab 1.**
<!-- endtab -->

<!-- tab Solution 2 -->
**This is Tab 2.**
<!-- endtab -->

<!-- tab Solution 3 @paw -->
**This is Tab 3.**
<!-- endtab -->
{% endtabs %}
```

> Unique name：选项卡唯一名字
> [index]：活动卡索引号
> [caption]：标签标题
> [@icon]：FontAwesome图标名称

{% tabs Unique name, 1 %}
<!-- tab caption @github -->
**This is Tab 1.**
<!-- endtab -->

<!-- tab Solution 2 -->
**This is Tab 2.**
<!-- endtab -->

<!-- tab Solution 3 @paw -->
**This is Tab 3.**
<!-- endtab -->
{% endtabs %}


## Label 标签

使用方法

```
{% label [class]@Text %}
```

> [class] : default | primary | success | info | warning | danger.

效果如下

Will you choose {% label default default %}, {% label primary purple%}, {% label success green%}, {% label info blue%}, {% label warning orange%} or {% label danger red%} ?

## Video 标签

```
{% video https://example.com/sample.mp4 %}
```

## Button 标签

使用 `button` 或者 简写 `btn`

```
{% button url, text, icon [class], [title] %}
```

> url：绝对或相对路径
> text, icon：按钮文字或FontAwesome图标
> [class]：FontAwesome类：fa-fw | fa-lg | fa-2x | fa-3x | fa-4x | fa-5x
> [title]：鼠标悬停时的工具提示

{% btn #,home, home %}  


## 流程图

```markdown
{% mermaid type%}
{% endmermaid %}
```

> type: 请访问 https://github.com/knsv/mermaid 以获取更多信息

```sh
{% mermaid sequenceDiagram %}
Alice ->> Bob: Hello Bob, how are you?
Bob-->>John: How about you John?
Bob--x Alice: I am good thanks!
Bob-x John: I am good thanks!
Note right of John: Bob thinks a long<br/>long time, so long that the text does not fit on a row.
Bob-->>Alice: Checking with John...
Alice->>John: Yes... John, how are you?
{% endmermaid %}
```

{% mermaid %} 
sequenceDiagram 
Alice ->> Bob: Hello Bob, how are you?
Bob-->>John: How about you John?
Bob--x Alice: I am good thanks!
Bob-x John: I am good thanks!
Note right of John: Bob thinks a long<br/>long time, so long that the text does not fit on a row.
Bob-->>Alice: Checking with John...
Alice->>John: Yes... John, how are you?
{% endmermaid %}