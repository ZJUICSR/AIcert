
// 对抗攻击评估echarts

var taskid = $('#taskid').text();
var mydata;
var funclist=[];
var taskinfo;
var robustbar = 0;
$("#adv_div").hide();
$("#robust_div").hide();
$("#scatter").hide();
$("#adv_conclusions").html("结论生成中！")
$("#robust_conclusions").html("结论生成中！")
$(".loading").show();
update();
var clk = self.setInterval("update()", 5000);

// 获取数据
function getData(){
  $.getJSON('/output/Resultdata', { Taskid: taskid }, function (data) {
    mydata = data;
    console.log(mydata);
});
}
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
    $(".text_20").html(time_start.substr(0,4)+"年"+time_start.substr(4,2)+"月"+time_start.substr(6,2)+"日 "+time_start.substr(8,2)+":"+time_start.substr(10,2));
    $(".text_24").html(data.result.dataset);
    $(".text_26").html(data.result.model);
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
        case "formal_verification":
          cur += " "+"形式化验证";
          break;
      }
      
    }
    $(".text_22").html(cur);
    
    console.log(funclist);
});
}
var echart_for_adv_analysis = echarts.init(document.getElementById('results_adv_analysis'));
var echart_for_robust = echarts.init(document.getElementById("results_robust"));
var echart_for_scatter_bef = echarts.init(document.getElementById("results_scatter_bef"));
var echart_for_scatter_aft = echarts.init(document.getElementById("results_scatter_aft"));
//初始加载
setTimeout(function () {
  echart_for_adv_analysis.resize();
  echart_for_robust.resize();
},3000);

var loading_opt = {
  text: 'loading',
  // color: 'rgba(4, 249, 208, 1)',
  color:"rgb(105,211,48)",
  textColor: '#000',
  maskColor: 'rgba(255, 0, 255, 0.0)'
};


// 对抗攻击曲线图
function drawAtk(data){
  var method= [];
  var serieslist = [];
  var xlable = [];
  var maxadv;
  var maxacc = 0;
  for(temp in data.AdvAttack.AdvAttack.atk_acc){
    method.push(temp);
  };
  for(var i=0;i < data.AdvAttack.AdvAttack[method[0]].var_eps.length;i++){
    xlable.push(data.AdvAttack.AdvAttack[method[0]].var_eps[i].toFixed(2));
  };
  for(var i=0 ;i < method.length;i++){
    var tempdata = [];
    for (var j=0;j < data.AdvAttack.AdvAttack[method[i]].var_asr.length;j++){
      tempdata.push(data.AdvAttack.AdvAttack[method[i]].var_asr[j].toFixed(2));
    };
    if(parseFloat(data.AdvAttack.AdvAttack[method[i]].var_asr[j-1])>parseFloat(maxacc)){
      maxacc = data.AdvAttack.AdvAttack[method[i]].var_asr[j-1];
      maxadv = method[i];
    };
    serieslist.push({
      "name":method[i],
      "type":"line",
      "data":tempdata,
      "smooth": true
    })

  }
  var option;
  option = {
    title: {
      
      text: '对抗攻击评估',
      textStyle:{
        fontSize:18,
      },
      left:"center",
      textAlign: 'left',
      y:'30',
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      right:20,
      top:50,
      data: []
    },
    grid: {
      left: '10%',
      right: '10%',
      bottom: '5%',
      top:'15%',
      containLabel: true
    },

    xAxis: {
      type: 'category',

      boundaryGap: false,
      data: []
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value} %'
      }
    },
    series: []
  };
  option.series=serieslist;
  option.xAxis.data=xlable;
  option.legend.data=method;
  echart_for_adv_analysis.hideLoading();
  msg = "模型"+localStorage["model_name"]+"在"+method[0]+"等"+method.length+"种对抗攻击算法下，不同扰动噪声eps的攻击成功率如下图所示"+"<br />"+"其中在eps为"+xlable[xlable.length-1]+"时"+maxadv+"攻击成功率最高，为"+maxacc+"%;";
  $("#adv_conclusions").html(msg);
  option && echart_for_adv_analysis.setOption(option);
}

// 鲁棒性直方图
async function draw_robust_echart(data) {
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
  for(var i =0;i < sourcelist[0].length;i++){
    serieslist.push(
      {type: 'bar'}
    )
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
    echart_for_robust.hideLoading();
    msg = "模型"+localStorage["model_name"]+"经过"+robustmethod+"等增强措施后，不同对抗攻击下的准确率直方图如下";
    $("#robust_conclusions").html(msg);
    option && echart_for_robust.setOption(option);
    robustbar = 1;
  }
  
};

//样本置信度分布图
function draw_scatter(bendata,advdata,title,scatterDom){
  option = {
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
scatterDom.hideLoading();
option && scatterDom.setOption(option);
};
//公平性评估大
function drawconseva1(ID,value){
  var conseva = document.getElementById(ID);
  var myChartcons = echarts.init(conseva);
  window.addEventListener("resize", function () {
    myChartcons.resize()});
  var option;
  // var value = mydata.Consistency.toFixed(2);
  console.log(value);
  option = {
    series: [
      {
        type: 'gauge',
        minInterval:0.02,
        min:0,
        max:1,
        axisLine: {
          lineStyle: {
            width: 15,
            color: [
              [0.3, '#fd666d'],
              [0.7, '#37a2da'],
              [1, '#67e0e3']
            ]
          }
        },
        grid: {
          left: "1%",
          right: "1%",
          show: true
        },
        pointer: {
          itemStyle: {
            color: 'auto'
          }
        },
        axisTick: {
          distance: -10,
          length: 15,
          lineStyle: {
            color: '#fff',
            width: 2
          }
        },
        splitLine: {
          distance: -20,
          length: 30,
          lineStyle: {
            color: '#fff',
            width: 2
          }
        },
        axisLabel: {
          color: 'auto',
          distance: 20,
          fontSize: 15
        },
        detail: {
          valueAnimation: true,
          formatter: 'Consistency\n{value}',
          color: 'auto',
          fontSize:20
        },
        data: [
          {
            value: value
          }
        ]
      }
    ]
  };

  option && myChartcons.setOption(option);
  conseva.style.display="inline-block"
}
//个体公平性评估小
function drawconseva(ID,value,titlename){
  var conseva = document.getElementById(ID);
  var myChartcons = echarts.init(conseva);
  window.addEventListener("resize", function () {
    myChartcons.resize()});
  var option;
  // var value = mydata.Consistency.toFixed(2);
  console.log(value);
  option = {
    title:{
      // show:true,
      text:titlename,
      textStyle:{
        color:"rgba(0, 0, 0, 1)",
        fontSize:18,
      },
x:'center',
y:280,
},
    series: [
      {
        type: 'gauge',
        minInterval:0.02,
        min:0,
        max:1,
        radius:100,
        axisLine: {
          lineStyle: {
            width: 10,
            color: [
              [0.3, '#fd666d'],
              [0.7, '#37a2da'],
              [1, '#67e0e3']
            ]
          }
        },
        grid: {
          left: "1%",
          right: "1%",
          show: true
        },
        pointer: {
          itemStyle: {
            color: 'auto'
          }
        },
        axisTick: {
          distance: -5,
          length: 5,
          lineStyle: {
            color: '#fff',
            width: 2
          }
        },
        splitLine: {
          distance: -5,
          length: 15,
          lineStyle: {
            color: '#fff',
            width: 2
          }
        },
        axisLabel: {
          color: 'auto',
          distance: 10,
          fontSize: 13
        },
        detail: {
          valueAnimation: true,
          formatter: 'Consistency\n{value}',
          color: 'auto',
          fontSize:15
        },
        data: [
          {
            value: value
          }
        ]
      }
    ]
  };

  option && myChartcons.setOption(option);
  conseva.style.display="inline-block";
};
// 数据占比 属性1
function drawclass1pro(ID,mydata,classname,dataname){
  // data = getData();
  var setEchartHW= {
    width:300,
    height: 280
};
  var chartDom = document.getElementById(ID,null,setEchartHW);
  var myChart = echarts.init(chartDom);
  window.addEventListener("resize", function () {
    myChart.resize()});
  var option;
  var data_s = [];
  var label = [];
  var temp = {value:0,name:null};
   for (var key in mydata){
       data_s.push({
           name:key,
           value:mydata[key].toFixed(2)
       })
       label.push(key)
   }

  // const data = genData(50);
  option = {
    title: {
      text: '敏感属性：'+classname,
      textStyle:{
          color:'rgba(0, 0, 0, 1)'
      },
      left: 'center',
      top: 280,
    },
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b} : {c} ({d}%)'
    },
    grid: {
      top:"1%",

      left:10,

      right:"1%",

      bottom:250
  },
    legend: {
      textStyle:{
          color:'rgba(0, 0, 0, 1)'
      },
      type: 'scroll',
      orient: 'vertical',
      right: 10,
      // left:30,
      top: 20,
      bottom: 10,
      data: label
    },
    color:["rgb(25,182,172)","rgb(122,193,73)"],
    series: [
      {
        name: dataname,
        type: 'pie',
        radius: '55%',
        center: ['40%', '50%'],
        data: data_s,
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  };
  option && myChart.setOption(option);
  chartDom.style.display="inline-block";
}
//敏感属性1直方图 difference id=class1Difference
function drawbar(ID,data,label,name){
  var bar1 = document.getElementById(ID);
  var myChartbar1 = echarts.init(bar1);
  myChartbar1.clear();
  window.addEventListener("resize", function () {
    myChartbar1.resize()});
  var option;
  option = {
    title:{
      text:name,
      textStyle:{
        fontSize:18,
        color:"rgba(0, 0, 0, 1)"
      },
      x:'center',
      y:'bottom'
    },
    xAxis: {
      type: 'category',
      data: label,
      axisLabel:{
        show:true,
        textStyle:{
          color: "rgba(0, 0, 0, 1)",
          fontSize:14
        }
      },
      axisLine:{
        onZero:true,
        lineStyle:{
          color:'rgba(0, 0, 0, 1)'
        }
      }
    },
    yAxis: {
      type: 'value',
      min:0,
      max:1,
      interval:0.2,
      axisLabel:{
        show:true,
        textStyle:{
          color: "rgba(0, 0, 0, 1)",
          fontSize:14
        },
        // margin: 10px,
      },
      axisLine:{
        show:false,
        lineStyle:{
          color:'rgba(0, 0, 0, 1)'
        }
      },
      splitLine: {
        show: true,
        lineStyle:{
            type:'dashed'
        }
      }
    },
    series: [
      {
        data: data,
        type: 'bar',
        showBackground: true,
        backgroundStyle: {
          color: 'rgba(180, 180, 180, 0.2)'
        },
        itemStyle:{
          normal:{
            color:'#91c7ae',
            label:{
              show:true,
              position:"top",
              color:"rgba(0, 0, 0, 1)"
            }
          }
        }
      }
    ]
  };
  option && myChartbar1.setOption(option);
  bar1.style.display="block";
}
//敏感属性直方图 提升
function drawbarimproved(ID,data,data2,label,name){
  var conseva = document.getElementById(ID);
  var myChartcons = echarts.init(conseva);
  myChartcons.clear();
  window.addEventListener("resize", function () {
    myChartcons.resize()});
  var option;
  option = {
    title:{
      text:name,
      textStyle:{
        fontSize:18,
        // align:center,
        
        color:"rgba(0, 0, 0, 1)"
      },
      x:'center',
      y:'bottom'
    },
    xAxis: {
      type: 'category',
      data: label,
      axisLabel:{
        show:true,
        textStyle:{
          color: "rgba(0, 0, 0, 1)",
          fontSize:14
        }
      },
      axisLine:{
        onZero:true,
        lineStyle:{
          color:'rgba(0, 0, 0, 1)'
        }
      }
    },
    yAxis: {
      type: 'value',
      min:-0.1,
      max:1.1,
      interval:0.2,
      axisLabel:{
        show:true,
        textStyle:{
          color: "rgba(0, 0, 0, 1)",
          fontSize:14
        },
        // margin: 10px,
      },
      axisLine:{
        show:false,
        lineStyle:{
          color:'rgba(0, 0, 0, 1)'
        }
      },
      splitLine: {
        show: true,
        lineStyle:{
            type:'dashed'
        }
      }
    },
    legend: {
      data: ['origin', 'improved'],
      textStyle:{
        color:"rgba(0, 0, 0, 1)",
        fontSize:14,
        // right:"100px"
        // x:"right",
        // y:"top",
        // padding:[20,200,300,10]
      },
      x:"right"
    },
    series: [
      {
        name:"origin",
        data: data,
        type: 'bar',
        showBackground: true,
        backgroundStyle: {
          color: 'rgba(180, 180, 180, 0.2)'
        },
        itemStyle:{
          normal:{
            color:'#91c7ae',
            label:{
              show:true,
              position:"top",
              color:"rgba(0, 0, 0, 1)"
            }
          }
        }
      },
      {
        name:"improved",
        data: data2,
        type: 'bar',
        showBackground: true,
        backgroundStyle: {
          color: 'rgba(180, 180, 180, 0.2)'
        },
        itemStyle:{
          normal:{
            color:'rgb(61,163,216)',
            label:{
              show:true,
              position:"top",
              color:"rgba(0, 0, 0, 1)"
            }
          }
        }
      }
    ]
  };
  option && myChartcons.setOption(option);
  conseva.style.display="inline-block";
}
//鲁棒性加固结果呈现
function drawrobust(){
  $("#robust_div").show();
    echart_for_robust.showLoading(loading_opt);
    $("#scatter").show();
    echart_for_scatter_bef.showLoading(loading_opt);
    echart_for_scatter_aft.showLoading(loading_opt);
    // 鲁棒性直方图
    try{
      draw_robust_echart(mydata.result);
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
      draw_scatter(sctDataBen_tmp,sctDataAdv_tmp,"鲁棒训练前样本置信度分布图",echart_for_scatter_bef);
      draw_scatter(sctDataBen2_tmp,sctDataAdv2_tmp,"鲁棒训练后样本置信度分布图",echart_for_scatter_aft);
    }catch(err){}
};
//模型公平性提升结果呈现
function drawmodel_debias(data){
  res = data.model_debias;
  drawconseva("original",res.Consistency[0].toFixed(2),"original");
  drawconseva("improved",res.Consistency[1].toFixed(2),"improved");
  $("#consdebias").show();
  id = 1;
  dataname = taskinfo.dataset;
  for( var key in res.Proportion){
    drawclass1pro("class"+id+"pro",res.Proportion[key],key,dataname);
    id++;
  }
  $(".dataproportion").show();
  namelist = ["Favorable Rate Difference","Favorable Rate Ratio"];
  oriclass1 = [];
  oriclass2=[];
  impclass1 = [];
  impclass2 = [];
  labels = [];
  classname =[];
  for(var key in res){
    if (key == "Consistency"|| key == "Proportion" || key=="stop"){
      continue;
    }
    else{
      labels.push(key);
      if (oriclass1.length == 0){
        classname.push(Object.keys(res[key][0])[0]) ;
        classname.push(Object.keys(res[key][0])[1]) ;
      }
      oriclass1.push(res[key][0][classname[0]].toFixed(2));
      oriclass2.push(res[key][0][classname[1]].toFixed(2));
      impclass1.push(res[key][1][classname[0]].toFixed(2));
      impclass2.push(res[key][1][classname[1]].toFixed(2));
    }
    drawbarimproved("class1table",oriclass1,impclass1,labels);
    var h4class = document.getElementById("class1");
    h4class.innerHTML = "敏感属性："+classname[0];
    drawbarimproved("class2table",oriclass2,impclass2,labels);
    var h4class = document.getElementById("class2");
    h4class.innerHTML = "敏感属性："+classname[1];
  };
  $(".bar_model_debias").show();
  $(".bar").show();
};
//模型公平性评估结果呈现
function drawmodel_evaluate(data){
  res = data.model_evaluate;
  //得分图
  drawconseva1("conseva",res.Consistency.toFixed(2));
  $("#rdeva").show();
  //饼图
  id = 1;
  dataname = taskinfo.dataset;
  for( var key in res.Proportion){
    drawclass1pro("class"+id+"pro",res.Proportion[key],key,dataname);
    id++;
  }
  $(".dataproportion").show();
  //直方图
  class1 = [];
  class2=[];
  labels = [];
  classname =[];
  for(var key in res){
    if (key == "Consistency"|| key == "Proportion" || key == "stop"){
      continue;
    }
    else{
      labels.push(key);
      if (class1.length == 0){
        classname.push(Object.keys(res[key])[0]) ;
        classname.push(Object.keys(res[key])[1]) ;
      }
      class1.push(res[key][classname[0]].toFixed(2));
      class2.push(res[key][classname[1]].toFixed(2));
    }
    drawbar("class1table",class1,labels);
    var h4class = document.getElementById("class1");
    h4class.innerHTML = "敏感属性："+classname[0];
    drawbar("class2table",class2,labels);
    var h4class = document.getElementById("class2");
    h4class.innerHTML = "敏感属性："+classname[1];
  };
  $(".bar_model_eva").show();
  $(".bar").show();
};
//数据公平性提升结果呈现
function drawdata_debias(data){
  res = data.data_debias;
  drawconseva("original",res.Consistency[0].toFixed(2),"original");
  drawconseva("improved",res.Consistency[1].toFixed(2),"improved");
  
  $("#consdebias").show();
  id = 1;
  dataname = taskinfo.dataset;
  for( var key in res.Proportion){
    drawclass1pro("class"+id+"pro",res.Proportion[key],key,dataname);
    id++;
  }
  $(".dataproportion").show();
  namelist = ["Favorable Rate Difference","Favorable Rate Ratio"];
  
  for(var key=0;key<namelist.length;key++){
    oriclass1 = [];
    oriclass2=[];
    impclass1 = [];
    impclass2 = [];
    labels = [];
    classname =[];
    for(var temp in res[namelist[key]]){
      labels.push(temp);
      if (oriclass1.length == 0){
        classname.push(Object.keys(res[namelist[key]][temp][0])[0]) ;
        classname.push(Object.keys(res[namelist[key]][temp][0])[1]) ;
      }
      oriclass1.push(res[namelist[key]][temp][0][classname[0]].toFixed(2));
      oriclass2.push(res[namelist[key]][temp][0][classname[1]].toFixed(2));
      impclass1.push(res[namelist[key]][temp][1][classname[0]].toFixed(2));
      impclass2.push(res[namelist[key]][temp][1][classname[1]].toFixed(2));
    }

    if(key == 0){
      drawbarimproved("class1Difference",oriclass1,impclass1,labels,namelist[key]);
      var h4class = document.getElementById("class1");
      h4class.innerHTML = "敏感属性："+classname[0];
      drawbarimproved("class2Difference",oriclass2,impclass2,labels,namelist[key]);
      var h4class = document.getElementById("class2");
      h4class.innerHTML = "敏感属性："+classname[1];
    }
    else{
      drawbarimproved("class1Ratio",oriclass1,impclass1,labels,namelist[key]);
      var h4class = document.getElementById("class1");
      h4class.innerHTML = "敏感属性："+classname[0];
      drawbarimproved("class2Ratio",oriclass2,impclass2,labels,namelist[key]);
      var h4class = document.getElementById("class2");
      h4class.innerHTML = "敏感属性："+classname[1];
    }
  }
  $(".bar_data_debias").show();
  $(".bar").show();
};
//数据公平性评估结果呈现
function drawdate_evaluate(data){
  res = data.date_evaluate;
  drawconseva1("conseva",res.Consistency.toFixed(2));
        
  $("#rdeva").show();
  //饼图
  id = 1;
  dataname = taskinfo.dataset;
  for( var key in res.Proportion){
    drawclass1pro("class"+id+"pro",res.Proportion[key],key,dataname);
    id++;
  }
  $(".dataproportion").show();
  //直方图
  namelist = ["Favorable Rate Difference","Favorable Rate Ratio"];
  
  for(var key=0;key<namelist.length;key++){
    class1 = [];
    class2=[];
    labels = [];
    classname =[];
    for(var temp in res[namelist[key]]){
      labels.push(temp);
      if (class1.length == 0){
        classname.push(Object.keys(res[namelist[key]][temp])[0]) ;
        classname.push(Object.keys(res[namelist[key]][temp])[1]) ;
      }
      class1.push(res[namelist[key]][temp][classname[0]].toFixed(2));
      class2.push(res[namelist[key]][temp][classname[1]].toFixed(2));
    }

    if(key == 0){
      drawbar("class1Difference",class1,labels,namelist[key]);
      var h4class = document.getElementById("class1");
      h4class.innerHTML = "敏感属性："+classname[0];
      drawbar("class2Difference",class2,labels,namelist[key]);
      var h4class = document.getElementById("class2");
      h4class.innerHTML = "敏感属性："+classname[1];
    }
    else{
      drawbar("class1Ratio",class1,labels,namelist[key]);
      var h4class = document.getElementById("class1");
      h4class.innerHTML = "敏感属性："+classname[0];
      drawbar("class2Ratio",class2,labels,namelist[key]);
      var h4class = document.getElementById("class2");
      h4class.innerHTML = "敏感属性："+classname[1];
    }
  }
  $(".bar_data_eva").show();
  $(".bar").show();

};
function draw(data) {
  console.log("同步 绘制");
  var l1 = data.output_param.FGSM;
  var l2 = data.output_param.LiRPA;
  var epsDom = document.getElementById('eps_line');
  var epsChart = echarts.init(epsDom);
  var eps_option = {

      grid: {
          top: 30
      },
      tooltip: { trigger: 'axis', axisPointer: { lineStyle: { color: '#000' } } },
      legend: {
          icon: 'rect',
          itemWidth: 14, itemHeight: 5, itemGap: 10,
          right: '10px', top: '0px',
          textStyle: { fontSize: 12, color: '#000' }
      },

      xAxis: {
          type: 'category',
          name: "扰动值",
          data: l1.eps,

          axisLine: { lineStyle: { color: '#000' } },
          splitLine: { lineStyle: { color: '#57617B' } },
          axisLabel: {
              fontSize: 12,
              color: "#000",
              fontWeight: "bolder",
              textBorderColor: "#000",
              textBorderWidth: 0.5,
              textBorderType: "solid"
          },
          axisTick: {
              alignWithLable: true
          },
      },
      yAxis: {
          type: 'value',
          name: "成功率",
          axisLine: { lineStyle: { color: '#000' } },
          splitLine: { lineStyle: { color: '#57617B' } },
          axisLabel: {
              formatter: '{value}',
              fontSize: 16,
              color: "#000",
              fontWeight: "bolder",
              textBorderColor: "#000",
              textBorderWidth: 0.5,
              textBorderType: "solid"
          }
      },
      series: [
          {
              name: l1.name,
              data: l1.rates,
              smooth: false, lineStyle: { normal: { width: 2 } },
              animationDuration: 2500,
              areaStyle: {
                  normal: {
                      color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                          offset: 0,
                          color: 'rgba(218,57,20,0.3)'
                      }, {
                          offset: 0.8,
                          color: 'rgba(218,57,20,0)'
                      }], false),
                      shadowColor: 'rgba(0, 0, 0, 0.1)',
                      shadowBlur: 10
                  }
              },
              itemStyle: { normal: { color: '#DA3914' } },
              type: 'line',
          },
          {
              name: l2.name,
              data: l2.rates,
              type: 'line',
              smooth: true, lineStyle: { normal: { width: 2 } },
              animationDuration: 2500,
              areaStyle: {
                  normal: {
                      color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                          offset: 0,
                          color: 'rgba(3, 194, 236, 0.3)'
                      }, {
                          offset: 0.8,
                          color: 'rgba(3, 194, 236, 0)'
                      }], false),
                      shadowColor: 'rgba(0, 0, 0, 0.1)',
                      shadowBlur: 10
                  }
              },
              itemStyle: { normal: { color: '#03C2EC' } },
          }
      ]
  };
  eps_option && epsChart.setOption(eps_option);
  $("#formal").show();
  sleep(1800).then(() => {
      $.magnificPopup.open({
  items: {
      src: '#popResultDiv',
      type: 'inline',
      removalDelay: 300,
      mainClass: 'mfp-fade'
  },
  callbacks: {
      open: function () {

      },
      close: function () {
      }
  }
});
})

}

function draw_formal_verification(data){

  draw(data.formal_verification);
  

};
// 数据更新
function update(){
  getData();
  gettaskinfo();
  // if (mydata ){
  //   if (mydata.stop==1 && mydata.result.AdvAttack.length==0 && mydata.result.fairness.length==0){
  //   $("#adv_div").html("<h3 style='text-align:center'>结果未生成</h3>")}
  // };
  for (let key of funclist){
    if(key=="AdvAttack"){
      $("#adv_div").show();
      echart_for_adv_analysis.showLoading(loading_opt);
      try{
        drawAtk(mydata.result);
      }catch(err){};  
    }else if(key=="model_debias" ){
      try{
        drawmodel_debias(mydata.result);
      }catch(err){};
    }else if(key=="model_evaluate"){
      try{
        drawmodel_evaluate(mydata.result);
      }catch(err){};
    }else if(key=="data_debias"){
      try{
        drawdata_debias(mydata.result);
      }catch(err){};
    }else if(key=="date_evaluate"){
      try{
        drawdate_evaluate(mydata.result);
      }catch(err){};
    }else if(key=="AdvTrain"||key=="EnsembleDefense"||key=="PACA"){
      try{
        if(robustbar==0){
          drawrobust();
        }
      }catch(err){};
    }else if(key=="formal_verification"){
      try{
        draw_formal_verification(mydata.result);
      }catch(err){};
    }
  }
  try{
    stopTimer(mydata);
  }catch(err){}
}
// 关闭循环获取
function stopTimer(data) {
  if (data.stop) {
      $(".loading").hide();
      clearInterval(clk);
  }
}

function download_pdf() {
  if (confirm("您确认下载该pdf文件吗？") ){
    document.body.scrollTop = document.documentElement.scrollTop = 0;
    // 输出pdf尺寸为download_page大小
    var pdfX = parseInt($('#download_page').css('width'))+10;
    var pdfY = parseInt($('#download_page').css('height'))+50;
    var pdf = new jsPDF('p','pt', [pdfX, pdfY]);
    // 设置打印比例 越大打印越小
    pdf.internal.scaleFactor = 1;
    var options = {
      pagesplit: false, //设置是否自动分页
     "background": '#FFFFFF'   //如果导出的pdf为黑色背景，需要将导出的html模块内容背景 设置成白色。
    };
    // 所有结果放在div id=download_page 下
    var printHtml = $('#download_page').get(0);
    pdf.addHTML(printHtml, 15, 15, options,function() {
      pdf.save($('#taskid').text()+'.pdf');
    });
  }
}
