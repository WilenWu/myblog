// 更新 user-guide 页面
// 注册函数：在子窗口调用shell命令
var shell_exec = function(){  
  var child_process = require('child_process');
  var iconv = require('iconv-lite'); // 解决中文乱码
  var encoding = 'cp936';  // 解决中文乱码
  var binaryEncoding = 'binary';  // 解决中文乱码
  const command = 'python ./custom_scripts/guide_update.py';
  child_process.exec(command, { encoding: binaryEncoding }, function(error,stdout,stderr){
    if(error){
        console.info(stderr);
    } else {
        console.log(iconv.decode(Buffer.from(stdout, binaryEncoding), encoding));
    }
  });
}
hexo.extend.filter.register('before_exit', shell_exec);  // 运行时间选在abbrlink初次生成后

// header配色
var paint = function(data) {
  if(data.header_color){
    data.content = data.content.replace(/\n *# +(.+) *\n/g, '\n# <font color="red">$1</font>\n');
    data.content = data.content.replace(/\n *## +(.+) *\n/g, '\n## <font color="green">$1</font>\n');
  }
  return data;
}
hexo.extend.filter.register('before_post_render', paint);

