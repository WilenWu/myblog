// header配色
var paint = function(data) {
    if(data.header_color){
      data.content = data.content.replace(/\n *# +(.+) *\n/g, '\n# <font color="red">$1</font>\n');
      data.content = data.content.replace(/\n *## +(.+) *\n/g, '\n## <font color="green">$1</font>\n');
    }
    return data;
  }
hexo.extend.filter.register('before_post_render', paint);