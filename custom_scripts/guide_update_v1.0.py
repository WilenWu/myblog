#!/usr/bin/python
# -*- coding: utf-8 -*- 

import os
import re
import argparse
import datetime
import yaml 

# 初始化
print('hexo clean')
os.system('hexo clean')

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

#----------------------------- 读取 toc 文件
# toc_url = './source/user-guide/index.md'
toc_url = args.output
with open(toc_url , mode = 'r', encoding = 'utf-8') as f:
    toc = f.read()

yaml_pattern = '(--- *\n)(.+?)(--- *\n)'
toc_yaml = re.search(yaml_pattern, toc, re.S).group()
toc = re.sub(yaml_pattern, '', toc, flags = re.S, count = 1).split('\n')

# 解析 toc 文件内容
links = {}  # link_id -> link, title_num, link_num
spans = {}
h1 = h2 = None
for i in range(len(toc)):
    line = toc[i].rstrip()
    # 一级标题
    if re.search('^# .+$', line): 
        h1 = re.search('# +<font.+?>(.+?)</font>', line).group(1)
        spans[h1] = {'title_num_max': i, 'link_num_max': i}
    # 二级标题
    elif re.search('^## .+$', line):
        h2 = re.search('## +<font.+?>(.+?)</font>', line).group(1)
        spans[h1][h2] = {'title_num_max': i, 'link_num_max': i} 
    # 列表 - :emoji: [title][link_id]: desc
    elif re.search('^ *- +\:.+?\: *\[.+?\]\[.+?\]', line):
        title = re.search('\[(.+?)\]\[(.+?)\]', line).group(1)
        link_id = re.search('\[(.+?)\]\[(.+?)\]', line).group(2)
        box = re.search('^ *- +\:.+?\:', line).group()
        desc = re.search('^ *- +\:.+?\: *\[.+?\]\[.+?\](.*)', line).group(1)
        links[link_id] = {'title': title, 'title_num': i, 'box': box, 'desc': desc}
        # 分类
        spans[h1]['title_num_max'] = i 
        if spans[h1].get(h2):
            spans[h1][h2]['title_num_max'] = i
    # 链接 [link_id]: permalink
    elif re.search('^\[.+?\]\: +.*$', line):
        link_id = re.search('\[(.+?)\]\: +(.*)', line).group(1)
        permalink = re.search('\[(.+?)\]\: +(.*)', line).group(2)
        links[link_id]['permalink'] = permalink
        links[link_id]['link_num'] = i 
        # 分类
        spans[h1]['link_num_max'] = i 
        if spans[h1].get(h2):
            spans[h1][h2]['link_num_max'] = i

#---------------------------- 读取文章 front_matter
# posts_dir = './source/_posts/'
posts_dir = args.path
tot = n = 0
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
        # 读取 front_matter 信息
        abbrlink = front_matter['abbrlink']
        permalink = f'/posts/{abbrlink}/'
        cat = front_matter['categories']
        desc = front_matter.get('description')
        desc = desc if desc else ''
        # 生成默认的 link_id
        link_id = re.sub('R\(.+?\)--', 'r_', post_name)
        link_id = re.sub('Python\(.+?\)--', 'py_', link_id)
        link_id = re.sub('\.md', '', link_id)
        link_id = re.sub('-', '_', link_id)
        # 统计link_id覆盖量
        tot = tot + 1 
        n = n + 1 if links.get(link_id) else n 
        # 添加新文章链接
        if links.get(link_id) is None:
            if len(cat)>1:
                h1,h2,*_ = cat
                title_num = spans[h1][h2]['title_num_max'] + 1
                link_num = spans[h1][h2]['link_num_max'] + 2
            else:
                h1 = cat[0]
                title_num = spans[h1]['title_num_max'] + 1
                link_num = spans[h1]['link_num_max'] + 2
            content = f'- :ballot_box_with_check: [{title}][{link_id}]: {desc}'
            toc.insert(title_num, content)
            toc.insert(link_num, f'[{link_id}]: {permalink}')
            print(f'添加链接: {title}\t{permalink}')
        # 更新链接
        elif links.get(link_id)['permalink'] != permalink:
            link_num = links.get(link_id)['link_num']
            toc[link_num] = f'[{link_id}]: {permalink}'
            print(f'更新链接: {title}\t{permalink}')
        # 更新 title
        elif links.get(link_id)['title'] != title:
            title_num = links.get(link_id)['title_num']
            box = links.get(link_id)['box']
            desc = links.get(link_id).get('desc','').strip()
            toc[title_num] = f'{box} [{title}][{link_id}]{desc}'
            print(f'更新 title: {title}')

print(f'覆盖情况: {n}/{tot}')
#------------------------------------- 重写 toc
new_toc = toc_yaml + '\n'.join(toc)
with open(toc_url,'w',encoding='utf-8') as f:
    f.writelines(new_toc)
