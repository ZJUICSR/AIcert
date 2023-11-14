// 初始化参数
var storage = window.localStorage;  //只保存taskid、功能选择、数据集名和模型名
// 其余参数
var dataset_params = {
  "data_id": 0,
  "upload_flag": 0,
  "mean": [],
  "std": [],
  "num_class": 10,
  "bounds":[]
};
var model_params = {
  "pretrained": 1,
  "path": "",
  "model_id": 0,
  "upload_flag": 0
};
var methods = [];
var functions_params = {}; //构造成dict还是list？
// 初始化功能checked参数为false
$('#function_adv_analysis, #function_robust, #multi_defense_params, #auto_attack_params').prop("checked", false);

// 任务中心获取任务列表
function create_table(results) {
  Num_of_task = results["Number"];
  TaskInfo = results["TaskList"];
  var TaskList = Object.keys(TaskInfo);
  // console.log(TaskList);
  $('#Num_of_task').text(Num_of_task);
  // 创建表格
  var rows=Num_of_task;
  var cols=8;
  var $table=$("<table class=\"task_table\"><tbody class='task_tbody'></tbody></table>");
  // $table.append($("<tr><th>序号</th><th>任务编号</th><th>任务状态</th><th>创建时间</th><th>任务信息</th><th colspan=\"3\">任务操作</th></tr>"))
  for(var r=0;r<rows;r++){
    reverse_order = TaskList[rows-r-1];
    var $tr = $("<tr id=\""+reverse_order+"\"></tr>");
    for(var c=0;c<cols;c++){
      switch (c){
        case 0: //序号
          var $td = $("<td>"+(r+1)+"</td>")
          break
        case 1: //任务编号
          task_order = reverse_order.substr(0,16)
          var $td = $("<td>T"+task_order+"</td>")
          break
        case 2: //任务状态
          task_state = TaskInfo[reverse_order]["state"]
          if (task_state==0) {
            state_info = "未开始"
          } else if (task_state==1) {
            state_info = "执行中"
          } else if (task_state==2) {
            state_info = "执行成功"
          } else if (task_state==3) {
            state_info = "执行失败"
          }
          var $td=$("<td>"+state_info+"</td>")
          break
        case 3: //创建时间
          task_time = TaskInfo[reverse_order]["createtime"]
          if (task_time.length==12) {
            time_info = task_time.substr(0,4)+"-"+task_time.substr(4,2)+"-"+task_time.substr(6,2)+" "+task_time.substr(8,2)+":"+task_time.substr(10,2)+""
          }
          var $td=$("<td>"+time_info+"</td>")
          break
        case 4: //任务信息
          var func = Object.keys(TaskInfo[reverse_order]["function"])
          if(func[0]){
            if(TaskInfo[reverse_order]["function"][func[0]]["type"]=="AdvAttack"){
              var $td=$("<td>数据集："+TaskInfo[reverse_order]["dataset"]+"；模型："+TaskInfo[reverse_order]["model"]+"；执行功能："+TaskInfo[reverse_order]["function"][func[0]]["attackmethod"]+"</td>")
            }else if(TaskInfo[reverse_order]["function"][func[0]]["type"]=="model_evaluate"){
              var $td=$("<td>数据集："+TaskInfo[reverse_order]["function"][func[0]]["dataset"]+"；模型："+TaskInfo[reverse_order]["function"][func[0]]["model"]+"；执行功能：模型公平性评估</td>")
            }else if(TaskInfo[reverse_order]["function"][func[0]]["type"]=="model_debias"){
              var $td=$("<td>数据集："+TaskInfo[reverse_order]["function"][func[0]]["dataset"]+"；模型："+TaskInfo[reverse_order]["function"][func[0]]["model"]+"；模型训练优化算法："+TaskInfo[reverse_order]["function"][func[0]]["algorithmname"]+"；执行功能：模型公平性提升</td>")
            }else if(TaskInfo[reverse_order]["function"][func[0]]["type"]=="data_debias"){
              var $td=$("<td>数据集："+TaskInfo[reverse_order]["function"][func[0]]["dataset"]+"；数据集优化算法："+TaskInfo[reverse_order]["function"][func[0]]["datamethod"]+"；执行功能：数据集公平性提升</td>")
            }else if(TaskInfo[reverse_order]["function"][func[0]]["type"]=="date_evaluate"){
              var $td=$("<td>数据集："+TaskInfo[reverse_order]["function"][func[0]]["dataset"]+"；执行功能：数据集公平性评估</td>")
            }else if(TaskInfo[reverse_order]["function"][func[0]]["type"]=="formal_verification"){
              var $td=$("<td>数据集："+TaskInfo[reverse_order]["function"][func[0]]["dataset"]+"；模型："+TaskInfo[reverse_order]["function"][func[0]]["model"]+"；执行功能：形式化验证</td>")
            }
          }
          else { //实际中不存在数据集和模型未选择情况
            var $td=$("<td>执行功能：无</td>")
          }
          break
        case 5: // 评估结果
          var $td=$("<td><button onclick=\"queryresult(this)\">评估结果</button></td>")
          break
        case 6: // 日志
          var $td=$("<td><button onclick=\"querylog(this)\">日志</button></td>")
          break      
        case 7: // 删除
          var $td=$("<td><button onclick=\"delTr(this)\">删除</button></td>")
          break
      }
      $tr.append($td);
    }
    $table.append($tr);
  }
  $('#task_table').append($table);
};

function delTr(object) {
var task_object=object.parentNode.parentNode;
Taskid = task_object.id;
console.log(Taskid);
$.ajax({
  type:'delete',
  url:'/Task/DeleteTask',
  async: false,
  data: {"Taskid":Taskid},
  success: function(results) {
    console.log(results);
    open_task_center();
  }
  });
};

function queryresult(object) {
  var task_object=object.parentNode.parentNode;
  Taskid = task_object.id;
  open_task_results(Taskid);
};

function closswindow(){
  $('.pop_con').animate({'top':0,'opacity':0},function(){
    //隐藏弹窗的主界面
    $('.pop_main').hide()
});

};

function querylog(object) {
  //显示弹窗的主界面
 
  var task_object=object.parentNode.parentNode;
  var logmsg="";
  Taskid = task_object.id;
  console.log(Taskid);
  $.ajax({
    type: 'get',
    url:'/Task/QueryLog',
    async: 'true',
    data: {Taskid:Taskid},
    success: function(results) {
      $('.pop_main').show();
      //设置animate动画初始值
      $('.pop_con').css({'top':0,'opacity':0});
      $('.pop_con').animate({'top':'50%','opacity':1});
      for (let temp in results.Log){
        logmsg += temp+"\n"+results.Log[temp]
      }
      
      $(".pop_text").html(logmsg.replace(/\n,/g,"<br />"))
      console.log(results);
      // alert("query log here!");
    }
  });
};

function querytask() {
  $.ajax({
    type: 'get',
    dataType: 'JSON',
    url:'/Task/QueryTask',
    async: 'true',
    data: {},
    success: function(results) {
      // console.log(results);
      create_table(results);
    },
    error: function (jqXHR) {
      console.error(jqXHR);
    }
  });
}

setTimeout(querytask(), 10);


// 页面跳转
function open_function_introduction() {
  window.open('/index_function_introduction', '_self');
};

function open_task_center() {
  // var window_url = window.open('/index_task_center', '_self');
  window.open('/index_task_center', '_self');
  // var window_url = '/index_task_center';
  
};

function open_params_1() {
  storage.clear();
  window.open('/index_params_1', '_self');
};

function open_params_2() {
  window.open('/index_params_2', '_self');
}

function jingqingqidai() {
  alert("开发中，敬请期待！");
}

function open_task_results(tid) {
  console.log("open_task_results",tid);
  window.open('/index_results?tid='+tid, '_self');
  // 预留获取任务结果
}

// 数据集和参数设置相关js
// dic需转成json格式储存到缓存中
function select_mnist() {
    var t = document.getElementById('dataset_mnist');
    document.getElementById('dataset_choose').textContent=t.textContent;
    $('#div1').show();
    $('.dataset_display').each(function(index, ele) {
      $(ele).attr('src', './static/imgs/mnist'+index+'.jpg')
      // console.log( $(ele).attr('src'));
    }) ;
    // document.getElementById('dataset_display').setAttribute("display", "show");
    storage["dataset_name"] = t.textContent;
};

function select_cifar10() {
    var t = document.getElementById('dataset_cifar10');
    document.getElementById('dataset_choose').textContent=t.textContent;
    $('#div1').show();
    $('.dataset_display').each(function(index, ele) {
      $(ele).attr('src', './static/imgs/cifar10'+index+'.jpg')
      // console.log( $(ele).attr('src'));
    }) ;
    storage["dataset_name"] = t.textContent;
};

var Timer;
var drawing;
var curOfs = 10;
var ofs;
var resnet18 = ['pool(5,5,64)', 'conv(3, 3, 64)', 'conv(3, 3, 64)', 'conv(3, 3, 64)', 'conv(3, 3, 64)',
    'conv(3, 3, 128)', 'conv(3, 3, 128)', 'conv(3, 3, 128)', 'conv(3, 3, 128)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 256)',
    'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'pool(5,5,512)', 'fullyconn(5,5,1000)'];
var resnet34 = ['pool(5,5,64)', 'conv(3, 3, 64)', 'conv(3, 3, 64)', 'conv(3, 3, 64)', 'conv(3, 3, 64)', 'conv(3, 3, 64)', 'conv(3, 3, 64)', 'conv(3, 3, 64)',
    'conv(3, 3, 128)', 'conv(3, 3, 128)', 'conv(3, 3, 128)', 'conv(3, 3, 128)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 256)',
    'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'pool(5,5,512)', 'fullyconn(5,5,1000)'];
var resnet50 = ['pool(5,5,64)', 'conv(3, 3, 64)', 'conv(3, 3, 64)', 'conv(3, 3, 256)', 'conv(3, 3, 64)', 'conv(3, 3, 64)', 'conv(3, 3, 256)', 'conv(3, 3, 64)', , 'conv(3, 3, 64)', 'conv(3, 3, 256)',
    'conv(3, 3, 128)', 'conv(3, 3, 128)', 'conv(3, 3, 512)', 'conv(3, 3, 128)', 'conv(3, 3, 128)', 'conv(3, 3, 512)', 'conv(3, 3, 128)', 'conv(3, 3, 128)',
    'conv(3, 3, 512)', 'conv(3, 3, 128)', 'conv(3, 3, 128)', 'conv(3, 3, 512)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 1024)',
    'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 1024)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 1024)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 1024)',
    'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 1024)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 1024)',
    'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 2048)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 2048)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 2048)', 'pool(5,5,512)', 'fullyconn(5,5,1000)'];
var vgg11 = ['conv(3, 3, 64)', 'pool(5,5,64)', 'conv(3, 3, 128)', 'pool(5,5,128)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'pool(5,5,256)',
    'conv(3, 3, 512)', 'conv(3, 3, 512)', 'pool(5,5,512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'pool(5,5,512)', 'fullyconn(5,5,10)', 'softmax(2,2,3)'];
var vgg16 = ['conv(3, 3, 64)', 'conv(3, 3, 64)', 'pool(5,5,64)', 'conv(3, 3, 128)', 'conv(3, 3, 128)', 'pool(5,5,128)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'pool(5,5,256)',
    'conv(3, 3, 512)', 'conv(3, 3, 512)', 'pool(5,5,512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'pool(5,5,512)', 'fullyconn(5,5,10)', 'softmax(2,2,3)'];
var vgg19 = ['conv(3, 3, 64)', 'pool(5,5,64)', 'conv(3, 3, 128)', 'pool(5,5,128)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'conv(3, 3, 256)', 'pool(5,5,256)',
    'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'pool(5,5,512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'conv(3, 3, 512)', 'pool(5,5,512)', 'fullyconn(5,5,10)', 'softmax(2,2,3)'];
var model = new Array();
var curmodel = resnet34;
model["resnet18"] = resnet18;
model["resnet34"] = resnet34;
model["resnet50"] = resnet50;
model["vgg11"] = vgg11;
model["vgg16"] = vgg16;
model["vgg19"] = vgg19;

$(window).on('load', function () {
  try{
    drawing = new convnetdraw.drawing("net");
    drawing.draw(curmodel);
  }catch(err){};  
    

});

function show_model(modelname){
    curmodel = model[modelname];
    curOfs = 10;
    drawing.draw(curmodel, curOfs);
};

$('#net').mousedown(function (e) {
    var firstX = e.pageX;
    $(document).on('mousemove.drag', function (m) {
        ofs = m.pageX - firstX + curOfs;
        drawing.draw(curmodel, ofs);
    }).on('mouseup', function (n) {
        $(document).off('mousemove.drag');
        curOfs = ofs;
    })
});

  $('#control').mousedown(function (e) {
      $(document).on('mousemove.drag', function (m) {
          drawing.draw(curmodel, curOfs);
      }).on('mouseup', function (e) {
          $(document).off('mousemove.drag');
      })

  });

function select_vgg16() {
    var t = document.getElementById('model_vgg');
    document.getElementById('model_choose').textContent=t.textContent;
    document.getElementById('model_img').setAttribute("src", "./static/imgs/vgg16.png");
    storage["model_name"] = t.textContent;
};

function select_resnet34() {
    var t = document.getElementById('model_resnet');
    document.getElementById('model_choose').textContent=t.textContent;
    document.getElementById('model_img').setAttribute("src","./static/imgs/resnet34.png");
    storage["model_name"] = t.textContent;
};

// 数据集和模型确认选择
function check_params_and_next_step() {
    if (document.getElementById('model_choose').textContent && document.getElementById('model_choose').textContent) {
        window.open('/index_params_2', '_self')
    }
    else {
        alert("请确认数据集和模型已选择！")
    }
};

// 功能按钮切换
$(function(){
  // 对抗攻击评估
  $("#function_adv_analysis").click(function() {
    if($(this).prop("checked")==false && $('#function_robust').prop("checked")==false) {
      console.log("对抗评估和鲁棒性评估都未选中时，对抗攻击参数");
      $(this).prop("checked", true);
      $(this).toggleClass('switchOff');
      $('#adv_analysis_params').prop("checked", true);
      $('#adv_analysis_params').toggleClass('switchOff');
      $('#FGSM_algorithm, #RFGSM_algorithm, #FFGSM_algorithm, #BIM_algorithm, #PGD_algorithm').prop("disabled", false);
      console.log($(this).prop("checked"), $('#adv_analysis_params').prop("checked"));
    } else if($(this).prop("checked")==false && $('#function_robust').prop("checked")==true) {
      console.log("对抗评估未选中，鲁棒性评估选中时，对抗攻击参数");
      $(this).prop("checked", true);
      $(this).toggleClass('switchOff');
      console.log($(this).prop("checked"), $('#adv_analysis_params').prop("checked"));
    } else if($(this).prop("checked")==true && $('#function_robust').prop("checked")==true) {
      console.log("对抗评估和鲁棒性评估选中时，对抗攻击参数");
      $(this).prop("checked", false);
      $(this).toggleClass('switchOff');
      console.log($(this).prop("checked"), $('#adv_analysis_params').prop("checked"));
    } else if($(this).prop("checked")==true && $('#function_robust').prop("checked")==false){
      console.log("对抗评估选中，鲁棒性评估未选中时，对抗攻击参数");
      $(this).prop("checked", false);
      $(this).toggleClass('switchOff');
      $('#adv_analysis_params').prop("checked", false);
      $('#adv_analysis_params').toggleClass('switchOff');
      $('#FGSM_algorithm, #RFGSM_algorithm, #FFGSM_algorithm, #BIM_algorithm, #PGD_algorithm').prop("disabled", true);
      console.log($(this).prop("checked"), $('#adv_analysis_params').prop("checked"));
    }}); 


    $("#function_robust").click(function() {
      if($(this).prop("checked")==false && $('#function_adv_analysis').prop("checked")==false) {
        console.log("对抗评估和鲁棒性评估都未选中时，对抗攻击参数");
        $(this).prop("checked", true);
        $(this).toggleClass('switchOff');
        $('#adv_analysis_params').prop("checked", true);
        $('#adv_analysis_params').toggleClass('switchOff');
        $('#FGSM_algorithm, #RFGSM_algorithm, #FFGSM_algorithm, #BIM_algorithm, #PGD_algorithm').prop("disabled", false);
        console.log($(this).prop("checked"), $('#adv_analysis_params').prop("checked"));
      } else if($(this).prop("checked")==false && $('#function_adv_analysis').prop("checked")==true) {
        console.log("对抗评估未选中，鲁棒性评估选中时，对抗攻击参数");
        $(this).prop("checked", true);
        $(this).toggleClass('switchOff');
        console.log($(this).prop("checked"), $('#adv_analysis_params').prop("checked"));
      } else if($(this).prop("checked")==true && $('#function_adv_analysis').prop("checked")==true) {
        console.log("对抗评估和鲁棒性评估都选中时，对抗攻击参数");
        $(this).prop("checked", false);
        $(this).toggleClass('switchOff');
        console.log($(this).prop("checked"), $('#adv_analysis_params').prop("checked"));
      } else if($(this).prop("checked")==true && $('#function_adv_analysis').prop("checked")==false) {
        console.log("对抗评估选中，鲁棒性评估未选中时，对抗攻击参数");
        $(this).prop("checked", false);
        $(this).toggleClass('switchOff');
        $('#adv_analysis_params').prop("checked", false);
        $('#adv_analysis_params').toggleClass('switchOff');
        $('#FGSM_algorithm, #RFGSM_algorithm, #FFGSM_algorithm, #BIM_algorithm, #PGD_algorithm').prop("disabled", true);
        console.log($(this).prop("checked"), $('#adv_analysis_params').prop("checked"));
      }}); 

  // 鲁棒性增强评估之群智化防御,鲁棒性评估选中，才可进行自动攻击检测
  $("#multi_defense_params").click(function () {
    if($("#function_robust").prop('checked')) {
      $(this).toggleClass('switchOff');
      if($(this).prop('checked')) {
        $(this).prop('checked', false);
      } else {
        $(this).prop('checked', true);
      }
    };  
    console.log("multi_defense_params is checked:"+$(this).prop('checked'));
  });

  // 鲁棒性增强评估之自动攻击检测,鲁棒性评估选中，才可进行自动攻击检测
  $("#auto_attack_params").click(function () {
    if($("#function_robust").prop('checked')) {
      $(this).toggleClass('switchOff');
      if($(this).prop('checked')) {
        $(this).prop('checked', false);
      } else {
        $(this).prop('checked', true);
      }
    };  
    console.log("auto_attack is checked:"+$(this).prop('checked'));
  });
});

// 创建任务，获取tid和各功能
function create_task() {
  // 获取Taskid
  $.ajax({
    type: 'POST',
    dataType: 'JSON',
    url:'/Task/CreateTask',
    async: false,
    // 目前无论是否选中功能都会有tid产生
    data: {"AttackAndDefenseTask":$('#function_adv_analysis').prop('checked')},
    success: function(results) {
      storage["Taskid"] = results["Taskid"];
      storage["IsAdvAttack"] = $('#function_adv_analysis').prop('checked');
      storage["IsAdvTrain"] = $('#function_robust').prop('checked');
      storage["IsEnsembleDefense"] = $('#multi_defense_params').prop('checked');
      storage["IsPACADetect"] = $('#auto_attack_params').prop('checked');
      dataset_params["name"] = storage["dataset_name"];
      model_params["name"] = storage["model_name"];
      console.log("storage所有参数设置完成！");
      console.log(storage);
      
    },
    error: function (jqXHR) {
      console.error(jqXHR);
    }
  });
  get_adv_algo_params();
  adv_attack();
};

function get_adv_algo_params() {
  console.log('print function params here!'); 
  var INPUT = $('#algorithms_params_input').find('input:checkbox:checked');
  methods = [];
  functions_params = {};
  // console.log(INPUT);
  for(i=0; i<INPUT.length; i++) {
    // 参数配对
    alg = INPUT[i].nextSibling.nodeValue;
    target = $(INPUT[i]).nextAll();
    pars = {};
    num_params = target.children('div').length;
    for(num=0;num<num_params;num++) {
      pars[target.children('div')[num].textContent] = target.children('input')[num].value;
    };
    functions_params[$.trim(alg)] = pars;
    // storage.setItem($.trim(alg), JSON.stringify(pars));
    methods.push($.trim(alg));
  };
  // storage.setItem("Method", JSON.stringify(methods));
  console.log(functions_params);
  // console.log(methods)
};

function to_bool_value(value) {
  return Number(value);
}

function adv_attack() {
  adv_data = {
    "IsAdvAttack": Number($('#function_adv_analysis').prop('checked')),
    "IsAdvTrain": Number($('#function_robust').prop('checked')),
    "IsEnsembleDefense": Number($('#multi_defense_params').prop('checked')),
    "IsPACADetect": Number( $('#auto_attack_params').prop('checked')),
    "Taskid": storage["Taskid"],
    // "Taskid":"20221208_1620_73bc35e",
    "Dataset": dataset_params,
    "Model": model_params,
    "Method": methods
  };  
  $.each(functions_params, function(index, value) { adv_data[index] = value;});
  console.log(adv_data);
  // console.log(JSON.stringify(storage));
  $.ajax({
    type: 'POST',
    dataType: 'application/json',
    url:'/Attack/AdvAttack',
    async: 'true',
    data: JSON.stringify(adv_data),
    // data: JSON.stringify(test),
    success: function(results) {
      alert("成功返回对抗攻击结果");
      console.log(results);
    },
    error: function (jqXHR) {
      console.error(jqXHR);
    }
  })
};

function click_pop() {
  var flag = confirm("请确保您的选择符合您的评估需要，一经提交不能返回修改。");
  if(flag) {
    create_task();
    window.open('/index_evaluate', '_self')
  }};

