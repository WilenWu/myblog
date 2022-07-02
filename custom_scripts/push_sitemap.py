#!/usr/bin/python
# -*- coding: utf-8 -*- 

import os
import sys 
import argparse

#------------------------------ 参数获取
parser = argparse.ArgumentParser(description='更新目录文件')
parser.add_argument('-c','--config', default='./_config.yml',
                    help='_config文件路径')
parser.add_argument('-p','--public', default='./public/',
                    help='public文件路径')

args = parser.parse_args()

#------------------------------- 修改config
print("临时修改网站配置文件：www.tinylei.tech ==> wilenwu.gitee.io")
config_url = args.config
with open(config_url , mode = 'r+', encoding = 'utf-8') as f:
    config = f.read()
    config_new = config.replace('www.tinylei.tech', 'wilenwu.gitee.io')
    config_new = config_new.replace('git@github.com:WilenWu/wilenwu.github.io.git', 'git@gitee.com:WilenWu/wilenwu.git')
    f.write(config_new)

config_new = config.replace('www.tinylei.tech', 'wilenwu.gitee.io')
with open(config_url,'w',encoding='utf-8') as f:
    f.write(config_new)

#------------------------------- 修改SEO文件
files = ['atom.xml', 'baidu_urls.txt', 'baidusitemap.xml', 'sitemap.xml']
print("更新" + ', '.join(urls))
for url in files:
    url = args.public + url
    with open(url , mode = 'r+', encoding = 'utf-8') as f:
        site = f.read()
        site_new = site.replace('www.tinylei.tech', 'wilenwu.gitee.io')
        f.write(site_new)

# 推送
os.system('hexo d')

#--------------------------------恢复配置文件
print("恢复配置文件")
with open(config_url,'w',encoding='utf-8') as f:
    f.write(config)