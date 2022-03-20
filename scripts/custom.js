hexo.extend.filter.register('before_exit', function(){  // 运行时间选在abbrlink初次生成后
  var child_process = require('child_process');
  var iconv = require('iconv-lite'); // 解决中文乱码
  var encoding = 'cp936';  // 解决中文乱码
  var binaryEncoding = 'binary';  // 解决中文乱码
  child_process.exec('python ./custom_scripts/guide_update.py', { encoding: binaryEncoding }, function(error,stdout,stderr){
      if(error){
          console.info(stderr);
      } else {
          console.log(iconv.decode(Buffer.from(stdout, binaryEncoding), encoding));
      }
  });
});
