// 初始化设置
var indexLog = 0;
var Eof = false;
var tid = 0;
var dataset;
var storage = window.localStorage; 
// $("#pg").hide();
$(".result_body").hide();
$(".loading").hide();
$(".fail").hide();

var imgdiv = $(".result");
  for (var i = 0;i<imgdiv.length;i++){
    imgdiv[i].style.display="none";
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

//上传文件
$(".file").on("change", "input[type='file']", function () {
    var filePath = $(this).val();
    if (filePath.indexOf("csv") != -1 || filePath.indexOf("CSV") != -1) { //上传数据集格式
        $(".fileerrorTip").html("").hide();
        var arr = filePath.split('\\');
        var fileName = arr[arr.length - 1];
        $(".showFileName").html(fileName);
        data = {
            filepath: fileName
        }

    } else {
        $(".showFileName").html("");
        $(".fileerrorTip").html("您未上传文件，或者您上传文件类型有误！").show();
        document.getElementById('file_input').value = "";
        return false
    }
})

function upload_dataset() {
    var filePath;
    if (document.getElementById("dataset").checked) {
        $.ajax({
            type: "POST",
            cache: false,
            data: data,
            url: "/DataCollection",
            dataType: "json",
            success: function (res) {
                $("#pg").hide();
                $("#loadingText").hide();
            },
            error: function (jqXHR) {
                $("#pg").hide();
                $("#loadingText").hide();
                console.error(jqXHR);
            }
    })}
    else{
        var data = new FormData;
        data.append("newfile", document.getElementById("file_input").files[0]);
        console.log("文件zip");
        // alert("要上传文件啦！")
        $.ajax({
            url: "/uploadfile",
            type: "POST",
            dataType: "JSON",
            data: data,
            contentType: false,
            processData: false,
            success: function (res) {
                alert("执行进入上传啦！")
                console.log("执行进入上传啦！");
                $("#pg").hide();
                $("#loadingText").hide();
            },
            error: function (jqXHR) {
                alert("上传文件失败！")
                console.log("上传文件失败");
                $("#pg").hide();
                $("#loadingText").hide();
                console.error(jqXHR);
            }
        })}
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

//提交函数:公平性评估
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
dataeva = function(){
  $(".result_body").hide();
  $(".bar_data_debias").hide();
  $(".loading").show();
  $(".fail").hide();
  dataname = $('#dataname option:selected').val();
  console.log("参数确认：")
  console.log(dataname);
  if(dataname == "" ){
      alert("请确认输入选项不为空！"); return;
  }

  $(".result_body").show();
  $(".loading").show();
  $("#consdebias").hide();
  create_task1();

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
        //得分图
        drawconseva1("conseva",res.Consistency.toFixed(2));
        
        $("#rdeva").show();
        //饼图
        id = 1;
        for( var key in res.Proportion){
          drawclass1pro("class"+id+"pro",res.Proportion[key],key,dataname);
          id++;
        }
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
        // result显示
        $(".loading").hide();
        $(".result_body").show();
        $(".bar_data_eva").show();
        var imgdiv = $(".result");
        for (var i = 0;i<imgdiv.length;i++){
            imgdiv[i].style.display="inline-block";
        };
      },
      error: function (jqXHR) {
        console.log(jqXHR);
        $(".fail").show();
      }
  });
}

//提交函数:公平性提升
datadebias = function(){
  
  $(".result_body").hide();
  $(".bar_data_eva").hide();
  $(".loading").show();
  $(".fail").hide();
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
  $(".result_body").show();
  $("#rdeva").hide();
  create_task1();
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
        // curtime=new Date();
        // cur = cur + curtime.toLocaleString()+" 图表绘制中"+"<br/>";
        // $('#logDiv').html(cur);
        console.log("结果打印");
        console.log(res);
        drawconseva("original",res.Consistency[0].toFixed(2),"original");
        drawconseva("improved",res.Consistency[1].toFixed(2),"improved");
        
        $("#consdebias").show();
        id = 1;
        for( var key in res.Proportion){
          drawclass1pro("class"+id+"pro",res.Proportion[key],key,dataname);
          id++;
        }
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
        // result显示
        var imgdiv = $(".result");
        for (var i = 0;i<imgdiv.length;i++){
            imgdiv[i].style.display="inline-block";
        };
        $(".bar_data_debias").show();
        $(".loading").hide();
        $(".result_body").show();

    },
      error: function (jqXHR) {

        $(".fail").show();

      }
  });
  // dataLoading();
  // logLoading();
}