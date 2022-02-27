var child_process = require('child_process');
child_process.exec('python ./worker/update_toc.py', function(error,stdout,stderr){
    if(error){
        console.info(stderr);
    } else {
        console.log(stdout);
    }
});