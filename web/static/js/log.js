var res;
function querylog() {
    //显示弹窗的主界面
   
    // var task_object=object.parentNode.parentNode;
    var logmsg="";
    var Taskid = storage['Taskid'];
    console.log(Taskid);
    $.ajax({
      type: 'get',
      url:'/Task/QueryLog',
      async: 'true',
      data: {Taskid:Taskid},
      success: function(results) {
        res = results;
        if (results.code==1003){
            $(".pop_text").html("日志未输出！")
        }else{
            for (let temp in results.Log){
              logmsg += temp+"\n"+results.Log[temp];
              console.log(results.Log[temp][0].substr(0,19),results.Log[temp][results.Log[temp].length-1].substr(0,19));
              dateDIf(results.Log[temp][0].substr(0,19), results.Log[temp][results.Log[temp].length-1].substr(0,19))
              // console.log(results.Log[temp]);
            };
            $(".pop_text").html(logmsg.replace(/\n,/g,"<br />"));
            // console.log(typeof logmsg);
        }
      }
    });
  };

time_start = storage['Taskid'].substr(0,4)+"-"+storage['Taskid'].substr(4,2)+"-"+storage['Taskid'].substr(6,2)+" "+storage['Taskid'].substr(9,2)+":"+storage['Taskid'].substr(11,2);
$('.text_16').html(time_start);
var clk = self.setInterval("querylog()", 1000);
var res_btn = document.getElementsById("res_btn");
// res_btn.onclick=function(){window.open('/index_results?tid='+storage['Taskid'], '_self')};
//   $(".text_22").click(window.open('/index_results?tid='+storage['Taskid'], '_self'))

function dateDIf(start_time, end_time) {
  var date = new Date(end_time)-new Date(start_time);
  var days    = date / 1000 / 60 / 60 / 24;
  var daysRound   = Math.floor(days);
  var hours    = date/ 1000 / 60 / 60 - (24 * daysRound);
  var hoursRound   = Math.floor(hours);
  var minutes   = date / 1000 /60 - (24 * 60 * daysRound) - (60 * hoursRound);
  var minutesRound  = Math.floor(minutes);
  var seconds   = date/ 1000 - (24 * 60 * 60 * daysRound) - (60 * 60 * hoursRound) - (60 * minutesRound);
  var secondsRound  = Math.floor(seconds); //粗略计时，到分
  var time = daysRound+"天"+hoursRound +"时"+minutesRound+"分"+secondsRound+"秒";
  $('.text_18').html(time);
}