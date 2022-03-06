// hexo.extend.filter.register('before_post_render', function(data){
//   var child_process = require('child_process');
//   child_process.exec('python ./custom_scripts/guide_update_v1.0.py', function(error,stdout,stderr){
//       if(error){
//           console.info(stderr);
//       } else {
//           console.log(stdout);
//       }
//   });
// });