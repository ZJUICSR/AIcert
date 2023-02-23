function open_function_introduction() {
  window.open('/index_function_introduction', '_self');
};

var clk;
var mydata;
var funclist=[];
var taskinfo;
var taskid;
var storage = window.localStorage; 


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
var curmodel = resnet18;
model["resnet18"] = resnet18;
model["resnet34"] = resnet34;
model["resnet50"] = resnet50;
model["vgg11"] = vgg11;
model["vgg16"] = vgg16;
model["vgg19"] = vgg19;
// drawing = new convnetdraw.drawing("net");

$(window).on('load', function () {
  select_cifar10();
  drawing = new convnetdraw.drawing("net");
  // drawing.draw(curmodel);
  show_model('vgg16')

});

function show_model(modelname){
    var t = document.getElementById('model_vgg');
    document.getElementById('model_choose').textContent=t.textContent;
    storage["model_name"] = t.textContent;
    curmodel = model[modelname];
    curOfs = 10;
    drawing.draw(curmodel, curOfs);
    $(".lwcontainer").show();
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


// var echart_for_robust = echarts.init(document.getElementById("results_robust"));
// var echart_for_scatter_bef = echarts.init(document.getElementById("results_scatter_bef"));
// var echart_for_scatter_aft = echarts.init(document.getElementById("results_scatter_aft"));
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

// 初始化功能checked参数为false
$('#adv_analysis_params, #multi_defense_params, #auto_attack_params').prop("checked", false);

// 鲁棒性直方图
async function draw_robust_echart(ID,data) {
    var conseva = document.getElementById(ID);
    var myChartcons = echarts.init(conseva);
    window.addEventListener("resize", function () {
        myChartcons.resize()});
    var sourcelist = [];
    sourcelist[0] = [""];
    var serieslist = [];
    var robustmethod="";
    if (data.AdvAttack.AdvAttack){
      sourcelist[0].push("原模型准确率");
      i = 1;
      for ( var key in data.AdvAttack.AdvAttack.atk_acc){
        sourcelist[i]=[];
        sourcelist[i].push(key);
        sourcelist[i].push(data.AdvAttack.AdvAttack.atk_acc[key]);
        i++;
      }
    };
    if(data.AdvAttack.AdvTrain){
      sourcelist[0].push("对抗训练后的准确率");
      robustmethod+="对抗训练 "
      i = 1;
      for ( var key in data.AdvAttack.AdvTrain.def_acc){
        if (sourcelist[i].length==0){
          sourcelist[i]=[];
          sourcelist[i].push(key);
        }
        sourcelist[i].push(data.AdvAttack.AdvTrain.def_acc[key]);
        i++;
      }
    };
    if(data.AdvAttack.EnsembleDefense){
      sourcelist[0].push("群智化防御");
      robustmethod+="群智化防御";
      i = 1;
      for ( var key in data.AdvAttack.EnsembleDefense.ens_acc){
        if (sourcelist[i].length==0){
          sourcelist[i]=[];
          sourcelist[i].push(key);
        }
        sourcelist[i].push(data.AdvAttack.EnsembleDefense.ens_acc[key]);
        i++;
      }
    };
    
    if(data.AdvAttack.PACA){
      sourcelist[0].push("PACA自动化攻击检测");
      robustmethod += " PACA自动化攻击检测";
      i = 1;
      for (var key in data.AdvAttack.PACA){
        if (sourcelist[i].length==0){
          sourcelist[i]=[];
          sourcelist[i].push(key);
        }
        sourcelist[i].push(data.AdvAttack.PACA[key]*100);
        i++;
    }};
    for(var i =1;i < sourcelist[0].length;i++){
        serieslist.push(
          {type: 'bar'}
        )
      };
    var option;
    option = {
      title: {
        text: '鲁棒性评估',
        textStyle:{
          fontSize:18,
        },
        left:"center",
        textAlign: 'left',
        y:'30',
          },
      legend: {
        right:20,
        top:50,
          },
      grid: {
        left: '10%',
        right: '10%',
        bottom: '5%',
        top:'15%',
        containLabel: true
          },
      tooltip: {},
      dataset: {
        source: []
      },
      xAxis: { 
        type: 'category', 
        // boundaryGap: false
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: '{value} %'
        }
      },
      series: []
    };
    if (sourcelist.length > 1){
      option.dataset.source=sourcelist;
      option.series=serieslist;
    //   echart_for_robust.hideLoading();
      msg = "模型"+localStorage["model_name"]+"经过"+robustmethod+"等增强措施后，不同对抗攻击下的准确率直方图如下";
      $("#robust_conclusions").html(msg);
      option && myChartcons.setOption(option);
      robustbar = 1
    conseva.style.display="inline-block";
    }
    
  };
  
  //样本置信度分布图
function draw_scatter(ID,bendata,advdata,title){
    var conseva = document.getElementById(ID);
var myChartcons = echarts.init(conseva);
window.addEventListener("resize", function () {
    myChartcons.resize()});
  var option = {
    title: {
        text: title,
        textStyle: { fontSize: 18, color: '#000' },
        top: "20px",
        left:"center"
    },
    toolbox: {
        left: "right",
        feature: {
            dataZoom: {}
        },
        top: "20px"
    },
    legend: {
        left: "center",
        itemWidth: 14, itemHeight: 5, itemGap: 10,
        bottom: '10px',
        textStyle: { fontSize: 15, color: '#000' }
    },
    tooltip: {
        formatter: function (params) {
            return (
                params.seriesName +
                '样本:<br/>' +
                params.value[0] +
                '，<br/>' +
                params.value[1]
            );
        }
    },
    xAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#000' } },
        splitLine: { lineStyle: { color: '#57617B' } },
        axisLabel: {
            fontSize: 15,
            color: "#000",
            // rgba(4, 249, 208, 1)

            textBorderColor: "#000",
            // rgba(139, 238, 103, 0.98)
            textBorderWidth: 0.1,
            textBorderType: "solid"
        }
    },
    yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#000' } },
        splitLine: { lineStyle: { color: '#57617B' } },
        axisLabel: {
            fontSize: 15,
            color: "#000",
            // rgba(4, 249, 208, 1)
            textBorderColor: "#000",
            // rgba(139, 238, 103, 0.98)
            textBorderWidth: 0.1,
            textBorderType: "solid"
        }
    },
    series: [
        {
            name: "正常样本",
            type: "scatter",
            symbolSize: 3.5,
            data: bendata,
            itemStyle: { normal: { color: 'green' } }
        },
        {
            name: "对抗样本",
            type: "scatter",
            symbolSize: 3.5,
            data: advdata,
            itemStyle: { normal: { color: 'red' } }
        }
    ]
};
option && myChartcons.setOption(option);
conseva.style.display="inline-block";
};

//鲁棒性加固结果呈现
function drawrobust(){

      // 鲁棒性直方图
      try{
        draw_robust_echart("results_robust",mydata.result);
      }catch(err){}
      //鲁棒训练后样本置信度分布图
      try{
        var method = [];
        var sctDataAdv_tmp = [];
        var sctDataAdv2_tmp = [];
        var sctDataBen_tmp = [];
        var sctDataBen2_tmp = [];
        for(temp in mydata.result.AdvAttack.AdvAttack.atk_acc){
          method.push(temp);
        };
        for (i = 0; i < 200; i++) {
          sctDataBen_tmp.push(mydata.result.AdvAttack.AdvAttack[method[0]].normal_scatter[i]);
          sctDataBen2_tmp.push(mydata.result.AdvAttack.AdvAttack[method[0]].robust_scatter[i]);
        }
        for (i = 200; i < 400; i++) {
            sctDataAdv_tmp.push(mydata.result.AdvAttack.AdvAttack[method[0]].normal_scatter[i]);
            sctDataAdv2_tmp.push(mydata.result.AdvAttack.AdvAttack[method[0]].robust_scatter[i]);
        }
        draw_scatter("results_scatter_bef",sctDataBen_tmp,sctDataAdv_tmp,"鲁棒训练前样本置信度分布图");
        draw_scatter("results_scatter_aft",sctDataBen2_tmp,sctDataAdv2_tmp,"鲁棒训练后样本置信度分布图");
      }catch(err){}
  };
// 获取数据
function getData(){
    $.getJSON('/output/Resultdata', { Taskid: taskid }, function (data) {
      mydata = data;
      // console.log(mydata);
  });
}
var robustbar = 0;
// 获取任务信息
function gettaskinfo(){
$.getJSON('/Task/QuerySingleTask', { Taskid: taskid }, function (data) {
    var namelist = [];
    taskinfo = data.result;
    for(temp in data.result.function){
    for( var i =0 ;i <data.result.function[temp].name.length;i++){
        namelist.push(data.result.function[temp].name[i]);
        
    }
    }
    time_start = data.result.createtime;
    // $(".text_20").html(time_start.substr(0,4)+"年"+time_start.substr(4,2)+"月"+time_start.substr(6,2)+"日 "+time_start.substr(8,2)+":"+time_start.substr(10,2));
    // $(".text_24").html(data.result.dataset);
    // $(".text_26").html(data.result.model);
    var cur = "";
    funclist = new Set(namelist);
    for(let temp of new Set(funclist)){
    switch(temp){
        case "AdvAttack":
        cur += " "+"对抗攻击评估";
        break;
        case "model_evaluate":
        cur += " "+"模型公平性评估";
        break;
        case "model_debias":
        cur += " "+"模型公平性提升";
        break;
        case "data_debias":
        cur += " "+"数据公平性提升";
        $(".text-wrapper_9").hide();
        break;
        case "date_evaluate":
        cur += " "+"数据公平性评估";
        $(".text-wrapper_9").hide();
        break;
        case "AdvTrain":
        cur += " "+"对抗训练";
        break;
        case "EnsembleDefense":
        cur += " "+"群智化防御";
        break;
        case "PACA":
        cur += " "+"PACA自动化攻击监测";
        break;
    }
    
    }
    // $(".text_22").html(cur);
    
    // console.log(funclist);
});
}
function dataupdate(){
    getData();
    gettaskinfo();
    if (funclist.size>0){
        for (let key of funclist){
            // if(key=="AdvAttack"){
            // $("#adv_div").show();
            // echart_for_adv_analysis.showLoading(loading_opt);
            // try{
            //     drawAtk(mydata.result);
            // }catch(err){};  
            // }else if(key=="model_debias" ){
            // try{
            //     drawmodel_debias(mydata.result);
            // }catch(err){};
            // }else if(key=="model_evaluate"){
            // try{
            //     drawmodel_evaluate(mydata.result);
            // }catch(err){};
            // }else if(key=="data_debias"){
            // try{
            //     drawdata_debias(mydata.result);
            // }catch(err){};
            // }else if(key=="date_evaluate"){
            // try{
            //     drawdate_evaluate(mydata.result);
            // }catch(err){};
            // }else 
            if(key=="AdvTrain"||key=="EnsembleDefense"||key=="PACA"){
            try{
              if (robustbar ==0){
                drawrobust();
                
              }
                
            }catch(err){};
            }
        }
        try{
            stopTimer(mydata);
        }catch(err){}
    }
    
    
    }
// 关闭循环获取
function stopTimer(data) {
if (data.stop) {
    
    $(".loading").hide();
    $("#robust_div").show();
    $("#scatter").show();
    $(".post-content1")[0].style.height="100%";
    // $(".robust_div").show();
    clearInterval(clk);
}
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
  $('.params_select').css('height','580px');
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

function open_task_center() {
  // var window_url = window.open('/index_task_center', '_self');
  window.open('/index_task_center', '_self');
  // var window_url = '/index_task_center';
  
};
$(function(){
  // 对抗攻击评估
  $("#adv_analysis_params").click(function() {
    if($(this).prop("checked")==false) {
      $(this).prop("checked", true);
      $(this).toggleClass('switchOff');
    } else if($(this).prop("checked")==true){
      $(this).prop("checked", false);
      $(this).toggleClass('switchOff');
    }}); 

  $("#auto_attack_params").click(function() {
    if($(this).prop("checked")==false) {
      $(this).prop("checked", true);
      $(this).toggleClass('switchOff');
    } else if($(this).prop("checked")==true){
      $(this).prop("checked", false);
      $(this).toggleClass('switchOff');
    }}); 
  
  $("#multi_defense_params").click(function() {
    if($(this).prop("checked")==false) {
      $(this).prop("checked", true);
      $(this).toggleClass('switchOff');
    } else if($(this).prop("checked")==true){
      $(this).prop("checked", false);
      $(this).toggleClass('switchOff');
    }});      
});


function get_adv_algo_params() {
  console.log('print function params here!'); 
  var INPUT = $('#algorithms_params_input').find('input:checkbox:checked');
  methods = [];
  functions_params = {};
  console.log(INPUT);
  for(i=0; i<INPUT.length; i++) {
    // 参数配对
    alg = INPUT[i].nextElementSibling.textContent;
    target = $(INPUT[i]).nextAll();
    pars = {};
    num_params = target.children('div').length;
    for(num=0;num<num_params;num++) {
      pars[$(target.children('div')[num]).attr('name')] = target.children('input')[num].value;
    };
    functions_params[$.trim(alg)] = pars;
    // storage.setItem($.trim(alg), JSON.stringify(pars));
    methods.push($.trim(alg));
  };
  // storage.setItem("Method", JSON.stringify(methods));
  console.log(functions_params);
  // console.log(methods)
};

function adv_attack() {
  adv_data = {
    "IsAdvAttack": 2,
    "IsAdvTrain": Number($('#adv_analysis_params').prop('checked')),
    "IsEnsembleDefense": Number($('#multi_defense_params').prop('checked')),
    "IsPACADetect": Number( $('#auto_attack_params').prop('checked')),
    "Taskid": storage["Taskid"],
    // "Taskid":"20221208_1620_73bc35e",
    "Dataset": dataset_params,
    "Model": model_params,
    "Method": methods
  };  
  $.each(functions_params, function(index, value) { adv_data[index] = value;});
  // console.log(adv_data);
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

//创建任务
function create_task() {
// 获取Taskid
  $.ajax({
      type: 'POST',
      dataType: 'JSON',
      url:'/Task/CreateTask',
      async: false,
      // 目前无论是否选中功能都会有tid产生
      data: {"AttackAndDefenseTask":$('#adv_analysis_params').prop('checked')},
      success: function(results) {
      storage["Taskid"] = results["Taskid"];
      taskid = results["Taskid"];
      storage["IsAdvAttack"] = 1;
      storage["IsAdvTrain"] = $('#adv_analysis_params').prop('checked');
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
//   dataupdate()
  clk = self.setInterval("dataupdate()", 5000);
};

function click_pop() {
  robustbar = 0;
  var flag = confirm("请确保您的选择符合您的评估需要，一经提交不能返回修改。");
  if(flag) {
        $("#scatter").hide();
      $("#robust_div").hide();
      $(".result_body1").show();
      $(".post-content1")[0].style.height="160px";
      $(".loading").show();
      create_task();
}};