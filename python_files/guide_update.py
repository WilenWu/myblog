#!/usr/bin/python
# -*- coding: utf-8 -*- 

import os
import re
import sys 
import argparse
import datetime
import yaml 
import numpy as np 

#------------------------------ 参数获取
parser = argparse.ArgumentParser(description='更新目录文件')
parser.add_argument('-p','--path', default='./source/_posts/',
                    help='posts文件路径')
parser.add_argument('-o','--output', default='./source/user-guide/index.md',
                    help='目录文件路径')
parser.add_argument('--permalink', default='posts/:abbrlink/',
                    help='文章永久链接模板')

args = parser.parse_args()
print(f'开始更新 {args.output}')

#----------------------------- 读取 guide 文件
# guide_url = './source/user-guide/index.md'
guide_url = args.output
with open(guide_url , mode = 'r', encoding = 'utf-8') as f:
    raw = f.read()

guide = raw.split('\n')

# 解析 guide 文件内容
links = {}  # 存储行链接
link_pattern = [
    '^ *- +\:(.+?)\: *\[(.+?)\]\[(.+?)\]\:? *(.*)',  # - :emoji: [title][link_id]: desc
    '^ *- +\:(.+?)\: *\[(.+?)\]\((.+?)\)\:? *(.*)',  # - :emoji: [title](url): desc
    '^ *- +\:(.+?)\: *([^:]+)\:? *(.*)' # - :emoji: title: desc
]
cats = {} # 匹配分类最大行
link_num_map = {} # 匹配已有的链接行号

for i, line in enumerate(guide):
    line = line.rstrip()
    # 一级标题
    if re.search('^# .+$', line): 
        h1 = re.search('# +(.+) *', line).group(1).strip()
        h1 = h1.replace('手册','')
        cat_name = h1 
        cats[cat_name] = {'level':'#', 'line_num_max': i}
    # 二级标题
    elif re.search('^## .+$', line):
        h2 = re.search('## +(.+) *', line).group(1).strip()
        cat_name = ', '.join([h1,h2])
        cats[cat_name] = { 'level':'##', 'line_num_max': i}
    #- :emoji: [title](link_id): desc
    elif re.search(link_pattern[1], line):
        emoji = re.search(link_pattern[1], line).group(1).strip()
        title = re.search(link_pattern[1], line).group(2).strip()
        url = re.search(link_pattern[1], line).group(3).strip()
        desc = re.search(link_pattern[1], line).group(4).strip()
        if re.search('^/posts/[a-zA-Z0-9]+/$', url):
            link_id = re.search('/posts/([a-zA-Z0-9]+)/',url).group(1).strip()
        else:
            link_id = f'line_num_{i}'
        links[link_id] = {'emoji': emoji,'title': title,'desc': desc,'url': url}
        # 本分类
        links[link_id]['cat_name'] = cat_name
        cats[cat_name]['line_num_max'] = i 
        # 行号码
        link_num_map[link_id] = i 
    #- :emoji: title: desc
    elif re.search(link_pattern[2], line):
        emoji = re.search(link_pattern[2], line).group(1).strip()
        title = re.search(link_pattern[2], line).group(2).strip()
        desc = re.search(link_pattern[2], line).group(3).strip()
        link_id = f'line_num_{i}'
        links[link_id] = {'emoji': emoji,'title': title,'desc': desc,'url': None}
        # 本分类
        links[link_id]['cat_name'] = cat_name
        cats[cat_name]['line_num_max'] = i 
        # 行号码
        link_num_map[link_id] = i 

#---------------------------- 读取文章 front_matter
# posts_dir = './source/_posts/'
posts_dir = args.path
posts = {} # 存储文章
for addr, dirs, files in os.walk(posts_dir,topdown=False):
    for post_name in files: 
        if not post_name.endswith('.md'):
            continue
        post_url = os.path.join(addr, post_name) 
        with open(post_url, mode = 'r', encoding = 'utf-8') as f:
            post = f.read()
        yaml_pattern = '(--- *\n)(.+?)(--- *\n)'
        try:
            front_matter = re.search(yaml_pattern, post, re.S).group(2)
        except:
            continue
        front_matter = yaml.load(front_matter.encode('utf-8'), yaml.FullLoader)
        # 简化 title
        title = front_matter['title']
        title = re.sub('R ?手册\(.+?\)--', '', title)
        title = re.sub('\(ggplot2 extensions\)', '', title)
        title = re.sub('Python ?手册\(.+?\)--', '', title)
        title = re.sub('大数据手册 ?\(.+?\)--', '', title)
        # 读取 front_matter 信息
        abbrlink = front_matter['abbrlink']
        url = f'/posts/{abbrlink}/'
        cat_name = ', '.join(front_matter['categories'][:2])
        desc = front_matter.get('description')
        desc = desc if desc else ''
        emoji = front_matter.get('emoji')
        emoji = emoji if emoji else 'ballot_box_with_check'
        # append
        posts[abbrlink] = {
                 'emoji': emoji,
                 'title': title,
                 'desc': desc,
                 'url': url,
                 'cat_name': cat_name
                 }

# --------------------------- 添加新文章链接
# 匹配分类最大行
to_do_list = {}
for link_id,post_yaml in posts.items():
    cat_name = post_yaml['cat_name']
    if not cats.get(cat_name):
        print(f'请添加类别: <{cat_name}>来自于文章<{title}>')
        continue
    to_do_list[link_id] = cats[cat_name]['line_num_max'] 

# 倒叙更新
to_do_list = sorted(to_do_list.items(),key=lambda x:x[1],reverse=True)

for link_id,cat_num_max in to_do_list:
    emoji = posts[link_id]["emoji"]
    title = posts[link_id]["title"]
    url   = posts[link_id]["url"]
    desc  = posts[link_id]["desc"]
    desc = f': {desc}' if desc else ''
    content = f'- :{emoji}: [{title}]({url}){desc}'
    if link_id not in links.keys(): # 添加新链接
        line_num = cat_num_max + 1
        guide.insert(line_num, content)
        print(f'添加链接: {content}')
    elif posts[link_id] != links[link_id]:  # 更新已有链接
        line_num = link_num_map[link_id]
        print(f'更新链接: {guide[line_num]} ===> \n\t  {content}')
        if posts[link_id]['cat_name'] != links[link_id]['cat_name']:
            print(f'请手动更新分类: {links[link_id]["cat_name"]} ===> {posts[link_id]["cat_name"]}')
        guide[line_num] = content
        

#------------------------------------- 重写 guide
new_guide = '\n'.join(guide)
# guide_url = './source/user-guide/_index.md'
if raw == new_guide:
    print('「用户指南」页面无内容更新!')
else:
    with open(guide_url,'w',encoding='utf-8') as f:
        f.writelines(new_guide)
