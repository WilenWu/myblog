---
title: Git 快速参考指南
categories:
  - General
tags:
  - git
cover: /img/git-for-beginners.png
top_img: /img/git-toolkit.png
abbrlink: f48f437a
date: 2022-06-20 23:36:00
description:
---

# Git 起步

## 版本控制系统

Git 是一个开源的分布式版本控制系统（Distributed Version Control System）。Git 和其它版本控制系统（包括 Subversion 和近似工具）的主要差别在于 Git 对待数据的方式。 在 Git 中，每当你提交更新或保存项目状态时，它基本上就会对当时的全部文件创建一个快照并保存这个快照的索引。 为了效率，如果文件没有修改，Git 不再重新存储该文件，而是只保留一个链接指向之前存储的文件。 Git 对待数据更像是一个 **快照流**。

![Git 存储项目随时间改变的快照](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/git-snapshots.png)

## 安装和配置

Git 有多种使用方式。 你可以使用原生的命令行模式（[Git](https://git-scm.com/)），也可以使用 GUI 模式（例如 [<img src="/img/github-desktop-icon.svg" style="zoom: 30%;" /> GitHub Desktop](https://desktop.github.com/)），这些 GUI 软件也能提供多种功能。 

```shell
$ git --version  # 检查Git版本
```

安装完 Git 之后，要做的第一件事就是设置你的用户名和邮件地址，因为git每次`commit`都会记录他们。

```shell
$ git config --global user.name "John Doe"
$ git config --global user.email "johndoe@example.com"
```

查看配置信息

```shell
$ git config --list
user.name=John Doe
user.email=johndoe@example.com
...
```

你可以通过输入 `git config <key>` 来检查 Git 的某一项配置

```shell
$ git config user.name
John Doe
```

# 基本操作

## 工作流程

Git 的基本工作流程如下：

1. 在本地工作区中添加、修改项目文件。
2. 将更改的部分添加（`git add`）到暂存区。一般存放在 `.git/index` 文件中。
3. 提交更新（`git commit`），将快照**永久性**存储到版本库。本地仓库中有一个隐藏的 `.git` 文件夹，即是 Git 的版本库。
4. 将版本推送（`git push`）到远程仓库。

因此，Git 本地仓库拥有三个工作区域：**工作区（Working Directory）、暂存区（Stage / Index）以及版本库（Git Directory / Repository）**。如果再加上远程仓库（Remote Directory）就可以分为四个工作区域。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/git-command.jpg)

## 创建仓库

通常有两种方法创建本地仓库

- 在本地文件夹初始化一个新的仓库

    ```shell
    $ git init
    ```

    该命令执行完后会在当前目录生成一个 `.git` 目录，所有 Git 需要的数据和资源都存放在这个目录中。

- 从服务器**克隆**一个已存在的 Git 仓库，包含所有的文件、分支和提交(commits)。

    ```shell
    $ git clone <repo-url>
    ```

    比如，要克隆 Git 的链接库 `libgit2`，可以用下面的命令：

    ```shell
    $ git clone https://github.com/libgit2/libgit2
    ```

    Git 支持多种数据传输协议。 上面的例子使用的是 `https://` 协议，不过你也可以使用 `git://` 协议或者使用 SSH 传输协议，比如 `user@server:path/to/repo.git` 。 

## 检查当前文件状态

git本地文件可能处于三种状态： **已提交（committed）**、**已修改（modified）** 和 **已暂存（staged）**。

- 已修改表示修改了文件，但还没保存到暂存区。
- 已暂存表示对已修改文件保存到暂存区，使之包含在下次提交的快照中。
- 已提交表示数据已经安全地保存在本地版本库。

我们可以用 `git status` 命令查看文件状态

```shell
$ echo 'My Project' > README.md
$ git status
On branch master
No commits yet
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        README.md

nothing added to commit but untracked files present (use "git add" to track)
```

`git status` 命令的输出十分详细，通常我们使用 `-s` 或`--short`参数来获得简短的输出结果

```shell
$ git status -s
?? README.md
```

新添加的未跟踪文件前面有 `??` 标记，新添加到暂存区中的文件前面有 `A` 标记，修改过的文件前面有 `M` 标记。

## 暂存已修改文件

使用命令 `git add` 将该文件添加到暂存区（或称为索引区），以备下次提交。

```shell
$ git add <files>       # 添加一个或多个文件
$ git add <dir>         # 添加指定目录到暂存区，包括子目录
$ git add .             # 添加当前目录下的所有文件到暂存区
```

例如，添加 README.md 文件到暂存区

```shell
$ git add README.md
$ git status
On branch master
No commits yet
Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   README.md
```

## 提交暂存文件

前面章节我们使用 `git add` 命令将内容写入暂存区，然后再运行命令`git commit` 将暂存区内容添加到本地仓库中。

```shell
$ git commit -m "<message>"
```

参数 message 是一些版本注释信息（必须）。

```shell
$ git commit -m "first commit"
[master (root-commit) a34cbd8] first commit
 1 file changed, 1 insertion(+)
 create mode 100644 README.md
$ git status
On branch master
nothing to commit, working tree clean
```

其中 a34cbd8 为自动生成的 commitID （版本号），Git的 commitID 不是1，2，3 ... 递增的数字，而是一个SHA1计算出来的一个非常大的数字，用十六进制表示。

每提交一个新版本，实际上Git 用暂存区域的文件创建一个新的 commitID，并把它们自动串成一条时间线。然后把当前分支指向新的提交节点。

在 Git 中，有一个名为 `HEAD` 的特殊指针，它是一个指向当前分支的指针（可以将 `HEAD` 想象为当前分支的别名）。它总是指向该分支上的最后一次提交。 这表示 HEAD 将是下一次提交的父结点。 通常，理解 HEAD 的最简方式，就是将它看做 **该分支上的最后一次提交** 的快照。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/git-commit.png"  />有时候我们提交完了才发现漏掉了几个文件没有添加，或者提交信息写错了。 此时，可以运行带有 `--amend` 选项的提交命令来重新提交：

```shell
$ git commit --amend
```

这个命令会将暂存区中的文件提交，完全用一个新的提交 **替换** 掉旧有的最后一次提交

例如，你提交后发现忘记了暂存某些需要的修改，可以像下面这样操作：

```shell
$ git commit -m 'initial commit'
$ git add forgotten_file
$ git commit --amend
```

另外，git 也可只提交暂存区的指定文件

```shell
$ git commit [files] -m "<message>"
```

## 移除文件

`git rm` 是用来从工作区，或者暂存区移除文件的命令 

```shell
$ git rm <file>
```

例如，从暂存区和工作区中删除 PROJECTS.md 文件

```shell
$ git rm PROJECTS.md
rm 'PROJECTS.md'
```

如果要删除之前修改过或已经放到暂存区的文件，则必须加上参数 `-f`

```shell
$ git rm -f <file>
```

使用 `--cached` 选项来只移除暂存区域的文件但是保留工作区的文件

```shell
$ git rm --cached <file>
```

## 移动文件

`git mv `命令是一个便利命令，用于移动或重命名一个文件

```shell
$ git mv file_from file_to
```

```shell
$ git mv README.md README
$ git status
On branch master
Your branch is up-to-date with 'origin/master'.
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

    renamed:    README.md -> README
```

其实，运行 `git mv` 就相当于运行了下面三条命令：

```shell
$ mv README.md README
$ git rm README.md
$ git add README
```

如果新文件名已经存在，可以使用 `-f` 参数强制覆盖

```shell
$ git mv -f file_from file_to
```

## 比较修改内容

有许多种方法查看内容变动，下面是一些示例

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/git-diff.png)



使用 `git diff` 默认比较工作区与暂存区的差异

```shell
$ echo hello >> README.md
$ git diff
warning: LF will be replaced by CRLF in README.md.
The file will have its original line endings in your working directory
diff --git a/README.md b/README.md
index 56266d3..2349c28 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 My Project
+hello
```

可以添加 `--staged` 或 `--cached` 参数比较暂存区与最后一次提交的文件差异
```shell
$ git add README.md
warning: LF will be replaced by CRLF in README.md.
The file will have its original line endings in your working directory
$ git diff --staged
diff --git a/README.md b/README.md
index 56266d3..2349c28 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 My Project
+hello
```

或者比较两个分支之间的差异
```shell
$ git diff [first-branch] [second-branch]
```

比较当前版本和工作区的差异

```shell
$ git diff HEAD
```

## 查看提交记录

在使用 Git 提交了若干更新之后，又或者克隆了某个项目，想回顾下提交历史，我们可以使用 `git log` 命令查看。

```shell
$ git log
commit a34cbd8df2bab332f9af0ae3083f09e9ced194a7 (HEAD -> master)
Author: your_name <your_email@youremail.com>
Date:   Tue May 10 23:35:12 2022 +0800

    first commit
```

我们可以用 `--oneline` 参数来查看历史记录的简洁的版本，单行输出且commitID更简短（仍然是唯一的）

```shell
$ git log --oneline
a34cbd8 (HEAD -> master) first commit
```

如果只想查找指定用户的提交日志可以使用命令

```shell
$ git log --author=<username>
```

## 回退版本

Git 的 `reset` 和 `checkout` 命令用来回退版本。 在初遇的 Git 命令中，这两个是最让人困惑的。 

`git reset` 命令语法格式如下：

```shell
$ git reset [--soft | --mixed | --hard] [commit]
```

-   `--soft` 只移动 `HEAD` 指向的分支，其余都保持不变。它本质上是撤销了上一次 `git commit` 命令。
-   `--mixed` 为默认参数，会移动 `HEAD` 指向的分支，同时回退暂存区，但工作区文件内容保持不变。本质上是回滚到了所有 `git add` 和 `git commit` 的命令执行之前
-   `--hard` 参数进一步将工作区回到上一次版本。此时，你撤销了最后的提交、`git add` 和 `git commit` 命令以及工作目录中的所有工作。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/reset-hard.png" style="zoom:80%;" />

其中，要回退的版本号 commit 可以使用  `git log` 指令查看，对于已经删除的提交记录可以使用 `git reflog` 查看。如果没有给出提交点的版本号，那么默认用`HEAD`。`HEAD`指向的版本就是当前版本，`HEAD~` 指向上一个版本。

```shell
$ git reset HEAD~            # 回退所有内容到上一个版本  
$ git reset HEAD~ hello.php  # 回退 hello.php 文件的版本到上一个版本  
$ git reset 052e             # 回退到指定版本
```

运行 `git checkout [branch]` 与运行 `git reset --hard [branch]` 非常相似，不过有两点重要的区别。

-   首先不同于 `reset --hard`，`checkout` 对工作目录是安全的，它会通过检查来确保不会将已更改的文件弄丢。 其实它还更聪明一些。它会在工作目录中先试着简单合并一下，这样所有 *还未修改过的* 文件都会被更新。 而 `reset --hard` 则会不做检查就全面地替换所有东西。
-   第二个重要的区别是 `checkout` 如何更新 HEAD。 `reset` 会移动 HEAD 分支的指向，而 `checkout` 只会移动 HEAD 自身来指向另一个分支。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/reset-checkout.png" style="zoom:80%;" />

## 忽略文件

有些时候我们不想把某些文件纳入版本控制中， 通常都是那些自动生成的文件，比如日志文件，或者编译过程中创建的临时文件等。 在这种情况下，我们可以创建一个名为 `.gitignore` 的文件，列出要忽略的文件。

文件 `.gitignore` 的格式规范如下：

-   所有空行或者以 `#` 开头的行都会被 Git 忽略。
-   可以使用标准的 glob 模式匹配，它会递归地应用在整个工作区中。
-   匹配模式可以以（`/`）开头防止递归。
-   匹配模式可以以（`/`）结尾指定目录。
-   要忽略指定模式以外的文件或目录，可以在模式前加上叹号（`!`）取反。

 来看一个实际的例子

```shell
*.a  # 忽略所有的 .a 文件
!lib.a  # 但跟踪所有的 lib.a，即便你在前面忽略了 .a 文件
/TODO  # 只忽略当前目录下的 TODO 文件，而不忽略 subdir/TODO
build/  # 忽略任何目录下名为 build 的文件夹
doc/*.txt  # 忽略 doc/notes.txt，但不忽略 doc/server/arch.txt
doc/**/*.pdf  # 忽略 doc/ 目录及其所有子目录下的 .pdf 文件
```

# 分支管理

## 分支简介

几乎所有的版本控制系统都以某种形式支持分支。 使用分支意味着你可以把你的工作从开发主线上分离开来，以免影响开发主线。

![趋于稳定分支的工作流](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/lr-branches-2.png)

在实际开发中，我们应该按照几个基本原则进行分支管理：

-    `master` 分支上保留完全稳定的代码——有可能仅仅是已经发布或即将发布的代码。
-   `dev`分支是从`master`创建的分支，被用来做后续开发或者测试稳定性——这些分支不必保持绝对稳定，但是一旦达到稳定状态，它们就可以被合并入 `master` 分支了。
-   `topic`分支是一种短期分支，它被用来实现单一特性或其相关工作。
-   软件开发中，bug就像家常便饭一样。有了bug就需要修复，在Git中，每个bug都可以通过一个新的临时分支来修复，修复后，合并分支，然后将临时分支删除。

Git把每次提交串成一条时间线，这条时间线就是一个分支。Git 的默认分支名字是 `master`，Git用`master`指向最新的提交。每次提交，`master`分支都会向前移动一步，这样，随着你不断提交，`master`分支的线也越来越长。

>   Git 的 `master` 分支并不是一个特殊分支。 它就跟其它分支完全没有区别。 之所以几乎每一个仓库都有 master 分支，是因为 `git init` 命令默认创建它，并且大多数人都懒得去改动它。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/branch-and-history.png" style="zoom:80%;" />

## 列出分支

`git branch` 命令实际上是某种程度上的分支管理工具。 它可以列出你所有的分支、创建新分支、删除分支及重命名分支。

```shell
$ git branch
* master
  dev
```

## 创建分支

Git 是怎么创建新分支的呢？ 很简单，它只是为你创建了一个可以移动的新的指针。

```shell
$ git branch <branch-name>
```

比如，创建一个 testing 分支， 你需要使用 `git branch` 命令：

```shell
$ git branch testing
```

这会在当前所在的提交对象上创建一个指针

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/head-to-master.png" alt="两个指向相同提交历史的分支" style="zoom:80%;" />

## 切换分支

那么，Git 又是怎么知道当前在哪一个分支上呢？ 也很简单，它有一个名为 `HEAD` 的特殊指针。在 Git 中，它是一个指向当前分支的指针（可以将 `HEAD` 想象为当前分支的别名）。

要切换到一个已存在的分支，可以使用命令

```shell
$ git checkout <branch-name>
```

例如，换到新创建的 `testing` 分支，不妨再提交一次

```shell
$ git checkout testing
$ vim test.rb
$ git commit -a -m 'made a change'
```

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/advance-testing.png)

这样 `HEAD` 就指向 `testing` 分支，并随着提交操作自动向前移动。你的 `testing` 分支向前移动了，但是 `master` 分支却没有，它仍然指向运行 `git checkout` 时所指的对象。 

当分支不存在时，创建并切换到新分支有两种方法

```shell
$ git checkout -b <branch-name>
$ git switch -c <branch-name>
```

例如，创建并切换到 dev 分支

```shell
$ git switch -c dev
Switched to branch 'dev'
```

## 合并分支

合并分支默认是把当前提交（如下图 ed489）和另一个提交（33104）以及他们最近的共同祖先（b325c）进行一次三方合并。合并的结果是生成一个新的快照（并提交）。

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/git-merge.png)

`git merge` 工具用来合并一个或者多个分支到当前分支。 然后它将当前分支指针移动到合并结果上。

```shell
$ git merge <branch-name>
```

譬如，你要修复一个紧急问题，我们先来建立一个 `hotfix` 分支，并在该分支上工作直到问题解决。

```shell
$ git checkout -b hotfix
Switched to a new branch 'hotfix'
$ vim index.html
$ git commit -a -m 'fixed the broken email address'
[hotfix 1fb7853] fixed the broken email address
 1 file changed, 2 insertions(+)
```

然后将 `hotfix` 分支合并回你的 `master` 分支来部署到线上。

```shell
$ git checkout master
$ git merge hotfix
Updating f42c576..3a0874c
Fast-forward
 index.html | 2 ++
 1 file changed, 2 insertions(+)
```

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/basic-branching-5.png" style="zoom:80%;" />

当你试图合并两个分支时， 如果顺着一个分支走下去能够到达另一个分支，那么 Git 在合并两者的时候， 只会简单的将指针向前推进（指针右移），因为这种情况下的合并操作没有需要解决的分歧——这就叫做快进（fast-forward）。

如果你在两个不同的分支中，对同一个文件的同一个部分进行了不同的修改，Git 在合并它们的时候就会产生合并冲突。这时候就需要你修改这些文件来手动合并这些冲突（conflicts），并且改完之后，需要将它们标记为合并成功。步骤如下：

1.   手动处理冲突的文件

2.   将解决完冲突的文件加入暂存区（`git add`）

3.   将更新提交到仓库（`git commit`）

在合并改动之前，你可以使用如下命令预览差异

```shell
$ git diff <source_branch> <target_branch>
```

出现冲突的文件会包含一些特殊区段，看起来像下面这个样子：

```html
<<<<<<< HEAD:index.html
<div id="footer">contact : email.support@github.com</div>
=======
<div id="footer">
 please contact us at support@github.com
</div>
>>>>>>> iss53:index.html
```

其中标记 `<<<<<<<` , `=======` , 和 `>>>>>>>` 表示冲突的开始分支，分割线和结束分支。 在你解决了所有文件里的冲突之后，对每个文件使用 `git add` 命令来将其标记为冲突已解决。 一旦暂存这些原本有冲突的文件，Git 就会将它们标记为冲突已解决。

## 删除分支

不能删除当前分支，只能删除其他分支

```shell
$ git branch [ -d | -D ] <branch-name>
```

`-D` 参数用于强制删除

```shell
$ git branch -d hotfix
Deleted branch hotfix (3a0874c).
```

## 变基

在 Git 中整合来自不同分支的修改主要有两种方法：`merge` 以及 `rebase`。在 Git 中， 你可以使用 `rebase` 命令将提交到某一分支上的所有修改都移至另一分支上，这种操作就叫做 **变基（rebase）**。

在这个例子中，你可以检出 `experiment` 分支，然后将它变基到 `master` 分支上：

```shell
$ git checkout experiment
$ git rebase master
First, rewinding head to replay your work on top of it...
Applying: added staged command
```

![将C4中的修改变基到C3上](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/basic-rebase-3.png)

现在回到 `master` 分支，进行一次快进合并。

```shell
$ git checkout master
$ git merge experiment
```

![master分支的快进合并。](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/basic-rebase-4.png)

这两种整合方法的最终结果没有任何区别，但是变基使得提交历史更加整洁。 一般我们这样做的目的是为了确保在向远程分支推送时能保持提交历史的整洁。

## 储藏

有时，当你在一个分支上修改过文件后， 需要切换到另一个分支做一点别的事情。但是，你不想仅仅因为过会儿回到这一点而为做了一半的工作创建一次提交。此时，`git stash` 命令将未完成的修改保存到一个栈上， 而你可以在任何时候重新应用这些改动（甚至在不同的分支上）。

运行 `git status`，可以看到有改动的状态：

```shell
$ git status
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	modified:   index.html

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   lib/simplegit.rb
```

现在想要切换分支，但是还不想要提交之前的工作，所以贮藏修改。

```shell
$ git stash
Saved working directory and index state \
  "WIP on master: 049d078 added the index file"
HEAD is now at 049d078 added the index file
(To restore them type "git stash apply")
```

可以看到工作目录是干净的了：

```shell
$ git status
# On branch master
nothing to commit, working directory clean
```

此时，你可以切换分支并在其他地方工作；你的修改被存储在栈上。 要查看贮藏的东西，可以使用 `git stash list`：

```shell
$ git stash list
stash@{0}: WIP on master: 049d078 added the index file
stash@{1}: WIP on master: c264051 Revert "added file_size"
stash@{2}: WIP on master: 21d80a5 added number to log
```

将你刚刚贮藏的工作重新应用：`git stash apply`。如果不指定一个贮藏，Git 认为指定的是最近的贮藏：

```shell
$ git stash apply
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   index.html
	modified:   lib/simplegit.rb

no changes added to commit (use "git add" and/or "git commit -a")
```

# 远程仓库

在 Git 中没有多少访问网络的命令，几乎所以的命令都是在操作本地的数据库。 当你想要分享你的工作，或者从其他地方拉取变更时，这有几个处理远程仓库的命令。

## 查看远程仓库

如果想查看你已经配置的远程仓库服务器，可以运行 `git remote` 命令

```shell
$ git remote
origin
```

origin 为远程仓库别名。你也可以使用参数 `-v`，显示读写远程仓库的别名和对应的 URL。

```shell
$ git remote -v
origin	https://github.com/schacon/ticgit (fetch)
origin	https://github.com/schacon/ticgit (push)
```

## 添加远程仓库

可以使用以下命令添加一个新的远程仓库，同时指定一个方便使用的简写：

```shell
$ git remote add <remote-alias> <server-url>
```

其中 `remote-alias` 是远程仓库的别名（默认别名是`origin` ），可以用来代替整个 URL。

```shell
$ git remote
origin
$ git remote add pb https://github.com/paulboone/ticgit
$ git remote -v
origin	https://github.com/schacon/ticgit (fetch)
origin	https://github.com/schacon/ticgit (push)
pb	https://github.com/paulboone/ticgit (fetch)
pb	https://github.com/paulboone/ticgit (push)
```

如果想同时添加 Github 和 Gitee 的远程仓库关联，则可以指定不同的别名，例如

```shell
$ git remote add github git@github.com:tianqixin/runoob-git-test.git
$ git remote add gitee git@gitee.com:imnoob/runoob-test.git
```

这两个远程库的名字不同。这样一来，我们的本地库就可以同时与多个远程库互相同步

```shell
$ git push github master
$ git push gitee master
```

## 推送远程仓库

`git push` 命令用来与另一个仓库通信，计算你本地数据库与远程仓库的差异，然后将差异推送到另一个仓库中。 它需要有另一个仓库的写权限，因此这通常是需要验证的。

```shell
$ git push [-f] <remote-alias> [branch-name]
```

如果本地版本与远程版本有差异，则可以使用 `-f` 参数强制推送。

例如，推送本地的 master 分支到远程仓库 origin

```shell
$ git push origin master
```

## 抓取远程仓库更新

`git fetch` 命令与一个远程的仓库交互，并且将远程仓库中有但是在当前仓库的没有的所有信息拉取下来然后存储在你本地数据库中

```shell
$ git fetch <remote-alias> [branch-name]
```

![](https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/git/remote-branches-3.png)

例如，从名为 `pb` 的远程上拉取 `master` 分支到本地分支 `pb/master` 中

```shell
$ git fetch pb
remote: Counting objects: 43, done.
remote: Compressing objects: 100% (36/36), done.
remote: Total 43 (delta 10), reused 31 (delta 5)
Unpacking objects: 100% (43/43), done.
From https://github.com/paulboone/ticgit
 * [new branch]      master     -> pb/master
 * [new branch]      ticgit     -> pb/ticgit
```

`git pull` 命令基本上就是 `git fetch` 和 `git merge` 命令的组合体，Git 从你指定的远程仓库中抓取内容，然后马上尝试将其合并进你所在的分支中。

```shell
$ git pull <remote-alias> [branch-name]
```

远程分支也是分支，所以合并时冲突的解决方式也和解决本地分支冲突相同，在此不再赘述。

## 删除远程仓库连接

```shell
$ git remote rm [remote-alias]
```

比如删除`pb`

```shell
$ git remote rm pb
```

此处的删除其实是解除了本地和远程的绑定关系，并不是物理上删除了远程库。远程库本身并没有任何改动。

# Git 标签

如果你达到一个重要的阶段，并希望永远记住那个特别的提交快照，你可以使用 `git tag` 给它打上标签。 比较有代表性的是人们会使用这个功能来标记发布结点（ `v1.0` 、 `v2.0` 等等）。 

## 列出标签

列出已有标签

```shell
$ git tag
v1.0
v2.0
```

这个命令以字母顺序列出标签，但是它们显示的顺序并不重要。

## 创建标签

Git 支持两种标签：轻量标签（lightweight）与附注标签（annotated）。轻量标签很像某个特定提交的引用。而附注标签是存储在 Git 数据库中的一个完整对象。

```shell
git tag -a <tagname> -m [message]
```

例如，创建一个带注解的标签

```shell
$ git tag -a v1.4 -m "my version 1.4"
$ git tag
v0.1
v1.3
v1.4
```

通过使用 `git show <tagname>` 命令可以看到标签信息和与之对应的提交信息

```shell
$ git show v1.4
tag v1.4
Tagger: Ben Straub <ben@straub.cc>
Date:   Sat May 3 20:19:12 2014 -0700

my version 1.4

commit ca82a6dff817ec66f44342007202690a93763949
Author: Scott Chacon <schacon@gee-mail.com>
Date:   Mon Mar 17 21:52:11 2008 -0700

    changed the version number
```

我们也可以给指定版本追加标签

```shell
$ git tag -a <tagname> <commitID>
```

## 共享标签

默认情况下，`git push` 命令并不会传送标签到远程仓库。 在创建完标签后你必须显式地推送标签到远程仓库

```shell
$ git push origin <tagname>
```

```shell
$ git push origin v1.5
Counting objects: 14, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (12/12), done.
Writing objects: 100% (14/14), 2.05 KiB | 0 bytes/s, done.
Total 14 (delta 3), reused 0 (delta 0)
To git@github.com:schacon/simplegit.git
 * [new tag]         v1.5 -> v1.5
```

如果想要一次性推送很多标签，也可以使用带有 `--tags` 选项的 `git push` 命令。 这将会把所有不在远程仓库上的标签全部传送到那里。

```shell
$ git push origin --tags
Counting objects: 1, done.
Writing objects: 100% (1/1), 160 bytes | 0 bytes/s, done.
Total 1 (delta 0), reused 0 (delta 0)
To git@github.com:schacon/simplegit.git
 * [new tag]         v1.4 -> v1.4
 * [new tag]         v1.4-lw -> v1.4-lw
```

## 删除标签

要删除掉你本地仓库上的标签，可以使用命令

```shell
$ git tag -d <tagname> 
```

例如，可以使用以下命令删除一个轻量标签

```shell
$ git tag -d v1.4-lw
Deleted tag 'v1.4-lw' (was e7d5add)
```

注意上述命令并不会从任何远程仓库中移除这个标签，你必须更新你的远程仓库。有两种方式

第一种变体是：

```shell
$ git push <remote> :refs/tags/<tagname>
```

上面这种操作的含义是，将冒号前面的空值推送到远程标签名，从而高效地删除它。

```shell
$ git push origin :refs/tags/v1.4-lw
To /git@github.com:schacon/simplegit.git
 - [deleted]         v1.4-lw
```

第二种更直观的删除远程标签的方式是：

```shell
$ git push origin --delete <tagname>
```



> 参考资料：
> [Git 官方文档](https://git-scm.com/book/zh/v2)
> [GitHub Cheat Sheet](https://training.github.com/downloads/zh_CN/github-git-cheat-sheet/)
> [Git 教程|廖雪峰](https://www.liaoxuefeng.com/wiki/896043488029600/900002180232448)
> [狂神聊Git](https://mp.weixin.qq.com/s/Bf7uVhGiu47uOELjmC5uXQ)
> [图解 Git](https://www.runoob.com/w3cnote/git-graphical.html)