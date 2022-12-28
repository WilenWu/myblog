// 更新 user-guide 页面
// 注册函数：在子窗口调用shell命令
var shell_exec = function(){  
  var child_process = require('child_process');
  var iconv = require('iconv-lite'); // 解决中文乱码
  var encoding = 'cp936';  // 解决中文乱码
  var binaryEncoding = 'binary';  // 解决中文乱码
  const command = 'python ./python_files/guide_update.py';
  child_process.exec(command, { encoding: binaryEncoding }, function(error,stdout,stderr){
    if(error){
        console.info(stderr);
    } else {
        console.log(iconv.decode(Buffer.from(stdout, binaryEncoding), encoding));
    }
  });
}
hexo.extend.filter.register('before_exit', shell_exec);  // 运行时间选在abbrlink初次生成后


