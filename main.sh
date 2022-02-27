hexo clean 
python ./worker/update_toc.py 
if [ $1 == "s" ]
then
    hexo s 
elif [ $1 == "d" ]
    hexo d -g 
    git add
    git commit
fi
