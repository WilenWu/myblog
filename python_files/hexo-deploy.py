#!/usr/bin/python
# -*- coding: utf-8 -*- 

import os
import sys 
import argparse

#------------------------------- 修改config
config_url = './_config.yml'

gitee_url = 'wilenwu.gitee.io'
site_url = 'www.tinylei.tech'

gitee_push = 'git@gitee.com:WilenWu/wilenwu.git'
github_push = 'git@github.com:WilenWu/wilenwu.github.io.git'

def switch(old_url, new_url, old_push, new_push):
    with open(config_url , mode = 'r', encoding = 'utf-8') as f:
        config = f.read()
    config_new = config.replace(old_url, new_url).replace(old_push, new_push)
    with open(config_url , mode = 'w', encoding = 'utf-8') as f:
        f.write(config_new)

switch(gitee_url, site_url, gitee_push, github_push)
print('推送到 github')
os.system('hexo clean && hexo g -d')

print("临时修改网站配置文件：www.tinylei.tech ==> wilenwu.gitee.io")
switch(site_url, gitee_url, github_push, gitee_push)
print('推送到 gitee')
os.system('hexo clean && hexo g -d')
