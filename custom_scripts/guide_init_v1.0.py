#!/usr/bin/python
# -*- coding: utf-8 -*- 

import os
import re
import datetime
import yaml 

#----------------------------- 读取 toc 文件
toc_url = './source/user-guide/index.md'
with open(toc_url , mode = 'r', encoding = 'utf-8') as f:
    toc = f.read()

yaml_pattern = '(--- *\n)(.+?)(--- *\n)'
toc_yaml = re.search(yaml_pattern, toc, re.S).group()
toc = re.sub(yaml_pattern, '', toc, flags = re.S, count = 1).split('\n')

# 解析 toc 文件内容
title_map = {}  # title -> link_id
links = {} # link_id -> link, title_num, link_num
for i in range(len(toc)):
    line = toc[i].rstrip()
    # 列表 - :emoji: [title][link_id]: desc
    if re.search('^ *- +\:.+?\: *\[.+?\]\[.+?\]', line):
        title = re.search('\[(.+?)\]\[(.+?)\]', line).group(1)
        link_id = re.search('\[(.+?)\]\[(.+?)\]', line).group(2)
        box = re.search('^ *- +\:.+?\:', line).group()
        desc = re.search('^ *- +\:.+?\: *\[.+?\]\[.+?\](.*)', line).group(1)
        links[link_id] = {'title': title, 'title_num': i, 'box': box, 'desc': desc}
        title_map[title] = link_id
    # 链接 [link_id]: permalink
    elif re.search('^\[.+?\]\: +.*$', line):
        link_id = re.search('\[(.+?)\]\: +(.*)', line).group(1)
        permalink = re.search('\[(.+?)\]\: +(.*)', line).group(2)
        links[link_id]['permalink'] = permalink
        links[link_id]['link_num'] = i 

#---------------------------- 读取文章 front_matter
posts_dir = './source/_posts/'
for url, dirs, files in os.walk(posts_dir,topdown=False):
    for post_name in files: 
        post_url = os.path.join(url, post_name) 
        with open(post_url, mode = 'r', encoding = 'utf-8') as f:
            post = f.read()
        yaml_pattern = '(--- *\n)(.+?)(--- *\n)'
        front_matter = re.search(yaml_pattern, post, re.S).group(2)
        front_matter = yaml.load(front_matter.encode('utf-8'), yaml.FullLoader)
        # 简化 title
        title = front_matter['title']
        title = re.sub('R ?手册\(.+?\)--', '', title)
        title = re.sub('\(ggplot2 extensions\)', '', title)
        title = re.sub('Python ?手册\(.+?\)--', '', title)
        title = re.sub('大数据手册 ?\(.+?\)--', '', title)
        abbrlink = front_matter['abbrlink']
        if title_map.get(title):
            title_num = links[title_map[title]]['title_num']
            link_num = links[title_map[title]]['link_num']
            box = links[title_map[title]]['box']
            desc = links[title_map[title]].get('desc','').strip()
            print(title,desc,sep='\t')
        else:
            continue
        # 生成默认的 link_id
        link_id = re.sub('R\(.+?\)--', 'r_', post_name)
        link_id = re.sub('Python\(.+?\)--', 'py_', link_id)
        link_id = re.sub('\.md', '', link_id)
        link_id = re.sub('-', '_', link_id)
        # 刷新 title_num
        toc[title_num] = f'{box} [{title}][{link_id}]{desc}'
        # 刷新 link_num
        permalink = f'/posts/{abbrlink}/'
        toc[link_num] = f'[{link_id}]: {permalink}'

#------------------------------------- 重写 toc
new_toc = toc_yaml + '\n'.join(toc)
with open(toc_url,'w',encoding='utf-8') as f:
    f.writelines(new_toc)
