#!/usr/bin/python
# -*- coding: utf-8 -*- 

import os
import re
import datetime
import yaml 

#----------------------------- 读取 guide 文件
guide_url = './source/user-guide/index.md'
with open(guide_url , mode = 'r', encoding = 'utf-8') as f:
    guide = f.read()

guide = guide.split('\n')

# 解析 guide 文件内容
links = {} # link_id -> link, title_num, link_num
link_pattern = [
    '^ *- +\:(.+?)\: *\[(.+?)\]\[(.+?)\]\:? *(.*)',  # - :emoji: [title][link_id]: desc
    '^ *- +\:(.+?)\: *\[(.+?)\]\((.+?)\)\:? *(.*)', # - :emoji: [title](url): desc
    '^ *- +\:(.+?)\: *([^:]+)\:? *(.*)' # - :emoji: title: desc
]

cats = []

for i in range(len(guide)):
    line = guide[i].rstrip()
    # 一级标题
    if re.search('^# .+$', line): 
        name = re.search('# +<font.+?>(.+?)</font>', line).group(1).strip()
        cats.append({'name':name, 'level':'#', 'line_num': i})
    # 二级标题
    elif re.search('^## .+$', line):
        name = re.search('# +<font.+?>(.+?)</font>', line).group(1).strip()
        cats.append({'name':name, 'level':'##', 'line_num': i})
    #- :emoji: [title][link_id]: desc
    elif re.search(link_pattern[0], line):
        emoji = re.search(link_pattern[0], line).group(1).strip()
        title = re.search(link_pattern[0], line).group(2).strip()
        link_id = re.search(link_pattern[0], line).group(3).strip()
        desc = re.search(link_pattern[0], line).group(4).strip()
        links[link_id] = {'emoji': emoji,'title': title,'desc': desc,'line_num': i}
    #- :emoji: [title](link_id): desc
    elif re.search(link_pattern[1], line):
        emoji = re.search(link_pattern[1], line).group(1).strip()
        title = re.search(link_pattern[1], line).group(2).strip()
        url = re.search(link_pattern[1], line).group(3).strip()
        desc = re.search(link_pattern[1], line).group(4).strip()
        link_id = f'line_num_{i}'
        links[link_id] = {'emoji': emoji,'title': title,'desc': desc,'line_num': i}
        links[link_id]['url'] = url
    #- :emoji: title: desc
    elif re.search(link_pattern[2], line):
        emoji = re.search(link_pattern[2], line).group(1).strip()
        title = re.search(link_pattern[2], line).group(2).strip()
        desc = re.search(link_pattern[2], line).group(3).strip()
        link_id = f'line_num_{i}'
        links[link_id] = {'emoji': emoji,'title': title,'desc': desc,'line_num': i}
    # 链接 [link_id]: permalink
    elif re.search('^\[.+?\]\: +.*$', line):
        link_id = re.search('\[(.+?)\]\: +(.*)', line).group(1).strip()
        url = re.search('\[(.+?)\]\: +(.*)', line).group(2).strip()
        links[link_id]['url'] = url
        links[link_id]['url_num'] = i 
# 规范化
for link in links.values():
    emoji = link["emoji"]
    title = f'[{link["title"]}]' if link.get("url") else link["title"]
    url = f'({link["url"]})' if link.get("url") else ''
    desc = f': {link["desc"]}' if link.get("desc") else ''
    guide[link['line_num']]=f'- :{emoji}: {title}{url}{desc}'
    if link.get('url_num'):
        guide[link['url_num']]=None 

# 标题规范化
for cat in cats:
    level = cat['level']
    color = 'red' if level=='#' else 'green'
    name = cat['name']
    guide[cat['line_num']]=f'{level} <font color="{color}">{name}</font>'

guide = [line for line in guide if line is not None]
#------------------------------------- 重写 guide
with open('./source/user-guide/_index.md','w',encoding='utf-8') as f:
    f.writelines('\n'.join(guide))
