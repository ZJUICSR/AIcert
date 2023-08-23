var clk;
var mydata;
var funclist=[];
var taskinfo;
var taskid;
var storage = window.localStorage; 
// 获取数据
function getData(){
    $.getJSON('/output/Resultdata', { Taskid: taskid }, function (data) {
      mydata = data;
      console.log(mydata);
  });
}

$(window).on('load', function () {
    try{
        draw_eva_method_echart("eva_method_echart");
        draw_eva_dataset_echart("eva_dataset_echart")
    }catch(err){};  
  });

function open_function_introduction() {
    window.open('/index_function_introduction', '_self');
  };

// 文件上传
$(".file").on("change","input[type='file']",function(){
    var filePath=$(this).val();
    var arr=filePath.split('\\');
    var fileName=arr[arr.length-1];
    alert('您上传的文件'+fileName+'，格式有误！')
    // if(filePath.indexOf("jpg")!=-1 || filePath.indexOf("png")!=-1){
    //     // $(".fileerrorTip1").html("").hide();
    //     var arr=filePath.split('\\');
    //     var fileName=arr[arr.length-1];
    //     // $(".showFileName1").html(fileName);
    // }else{
    //     // $(".showFileName1").html("");
    //     // $(".fileerrorTip1").html("您未上传文件，或者您上传文件类型有误！").show();
    //     return false
    // }
})

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
    }
    
    }
    $(".text_22").html(cur);
    
    console.log(funclist);
});
}
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
        color:" #6F8BA4"
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
bar1.style.display="inline-block";
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
//模型公平性提升结果呈现
function drawmodel_debias(data){
    $("#rdeva").hide();
    $(".bar_data_debias").hide();
    $(".bar_data_eva").hide();
    $(".bar_model_eva").hide();
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
$("#class1Difference").hide();
    $("#class1Ratio").hide();
    $("#class2Difference").hide();
    $("#class2Ratio").hide();
$(".bar_model_debias").show();
$(".bar").show();
};
//模型公平性评估结果呈现
function drawmodel_evaluate(data){
    $("#consdebias").hide();
    $(".bar_data_debias").hide();
    $(".bar_data_eva").hide();
    $(".bar_model_debias").hide();
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
$("#class1Difference").hide();
$("#class1Ratio").hide();
$("#class2Difference").hide();
$("#class2Ratio").hide();
$(".bar_model_eva").show();
$(".bar").show();
};
//数据公平性提升结果呈现
function drawdata_debias(data){
    $("#rdeva").hide();
    $(".bar_model_eva").hide();
    $(".bar_data_eva").hide();
    $(".bar_model_debias").hide();
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
$("#class1table").hide();
$("#class2table").hide();
$(".bar_data_debias").show();
$(".bar").show();
};
//数据公平性评估结果呈现
function drawdate_evaluate(data){
    $("#consdebias").hide();
    $(".bar_data_debias").hide();
    $(".bar_model_debias").hide();
    $(".bar_model_eva").hide();
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
$("#class1table").hide();
$("#class2table").hide();
$(".bar_data_eva").show();
$(".bar").show();

};
// 数据更新
function update(){
getData();
gettaskinfo();
if (funclist.size>0){
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
            drawrobust();
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
    $(".post-content1")[0].style.height="100%";
    $(".result_body").show();
    clearInterval(clk);
}
}
// 提交主任务
function create_task1() {
// 获取Taskid
$.ajax({
    type: 'POST',
    dataType: 'JSON',
    url:'/Task/CreateTask',
    async: false,
    // 目前无论是否选中功能都会有tid产生
    data: {"AttackAndDefenseTask":1},
    success: function(results) {
    storage["Taskid"] = results["Taskid"];
    },
    error: function (jqXHR) {
    console.error(jqXHR);
    }
});
};
// 提交数据公平性评估
dataeva = function(){
    funclist=[];
    
$(".result_body1").show();
$(".post-content1")[0].style.height="160px";
$(".result_body").hide();

dataname = $('#dataname option:selected').val();
console.log("参数确认：")
console.log(dataname);
if(dataname == "" ){
    alert("请确认输入选项不为空！"); return;
}
$(".bar_data_debias").hide();
$(".loading").show();
$(".fail").hide();
create_task1();
taskid = storage["Taskid"];

data = {
    dataname: dataname,
    tid:storage["Taskid"]
}

$.ajax({
    type: "POST",
    cache: false,
    data: data,
    url: "/DataFairnessEvaluate",
    dataType: "json",
    success: function (res) {
        console.log(res);
        clk = self.setInterval("update()", 5000);
    },
    error: function (jqXHR) {
        console.log(jqXHR);
        $(".fail").show();
    }
});
}
//提交数据公平性提升
datadebias = function(){
    funclist=[];
    $(".result_body1").show();
    $(".post-content1")[0].style.height="160px";
    $(".result_body").hide();
    
    dataname = $('#dataname option:selected').val();
    datamethod = $('#datamethod option:selected').val();
    console.log("参数确认：")
    console.log(dataname);
    if(dataname == "" ){
        alert("请确认数据集不为空！"); return;
    }
    if(datamethod == "" ){
      alert("请确认数据集优化算法不为空！"); return;
    }
    $(".bar_data_eva").hide();
    $(".loading").show();
    $(".fail").hide();
    $("#rdeva").hide();
    create_task1();
    taskid = storage["Taskid"];
    data = {
      dataname: dataname,
      datamethod:datamethod,
      tid:storage["Taskid"]
    }
  
    $.ajax({
        type: "POST",
        cache: false,
        async : false,
        data: data,
        url: "/DataFairnessDebias",
        dataType: "json",
        success: function (res) {
          console.log("结果打印");
          console.log(res);
          clk = self.setInterval("update()", 5000);
      },
        error: function (jqXHR) {
  
          $(".fail").show();
  
        }
    });
}
//提交函数:模型公平性评估
modeleva = function(){
    funclist=[];
    $(".result_body1").show();
    $(".post-content1")[0].style.height="160px";
    $(".result_body").hide();
    $(".bar_model_eva").show();
    $(".bar_model_debias").hide();
    
    $(".fail").hide();
    dataname = $('#model_dataname option:selected').val();
    modelname = $('#modelname option:selected').val();
  
    console.log("参数确认：")
    console.log(dataname);
    if(dataname == "" ){
        alert("请确认数据集不为空！"); return;
    }
    if(modelname == "" ){
      alert("请确认模型结构不为空！"); return;
    }
    $(".loading").show();
  
    // 点击功能按钮后结果隐藏
    var imgdiv = $(".result");
    for (var i = 0;i<imgdiv.length;i++){
        imgdiv[i].style.display="none";
    };
    $("#consdebias").hide();
    create_task1();
    taskid = storage["Taskid"];
    data = {
      dataname: dataname,
      modelname: modelname,
      tid:storage["Taskid"]
    }
  
    $.ajax({
        type: "POST",
        cache: false,
        data: data,
        url: "/ModelFairnessEvaluate",
        dataType: "json",
        success: function (res) {
          console.log(res);
          clk = self.setInterval("update()", 5000);
        },
        error: function (jqXHR) {
          $(".fail").show();
          console.log(jqXHR);
        }
    });
}
//提交函数:公平性提升
modeldebias = function(){
    funclist=[];
    $(".result_body1").show();
    $(".post-content1")[0].style.height="160px";
    $(".result_body").hide();
    $(".bar_model_debias").show();
    $(".bar_model_eva").hide();
    $(".fail").hide();
    dataname = $('#model_dataname option:selected').val();
    algorithmname = $('#algorithmname option:selected').val();
    modelname = $('#modelname option:selected').val();
    
    console.log("参数确认：")
    console.log(dataname);
    if(dataname == "" ){
        alert("请确认数据集不为空！"); return;
    }
    if(algorithmname == "" ){
      alert("请确认训练优化算法不为空！"); return;
    }
    if(modelname == "" ){
      alert("请确认模型结构不为空！"); return;
    }
    
    $(".loading").show();
    $("#rdeva").hide();
    create_task1();
    taskid = storage["Taskid"];
    $("#rdeva").hide();
    data = {
      dataname: dataname,
      algorithmname:algorithmname,
      modelname:modelname,
      tid:storage["Taskid"]
    }
  
    $.ajax({
        type: "POST",
        cache: false,
        async : false,
        data: data,
        url: "/ModelFairnessDebias",
        dataType: "json",
        success: function (res) {
          console.log("结果打印");
          console.log(res);
          clk = self.setInterval("update()", 5000);
      },
        error: function (jqXHR) {
          console.log("错误打印");
          console.log(jqXHR.responseText);
          $(".fail").show();
        }
    });
  }
function open_task_center() {
    // var window_url = window.open('/index_task_center', '_self');
    window.open('/index_task_center', '_self');
    // var window_url = '/index_task_center';
    
  };