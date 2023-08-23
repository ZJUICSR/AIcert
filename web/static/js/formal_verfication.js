var mydata;
var funclist=[];
var taskinfo;
var taskid;
var storage = window.localStorage; 

function open_function_introduction() {
    window.open('/index_function_introduction', '_self');
};

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
function testInput(id){
    var input=document.getElementById(id).value;
    if(id == "dataset_size"){
        console.log(input);
        var inputValueFilter=/^[1-2][0-9][0-9]$|^[1-9]$|^[1-9][0-9]$/;
        
        if(!inputValueFilter.test(input)){alert("请正确输入数据集规模！");document.getElementById(id).value = ""; ;}
    }
    if(id == "move_times"){
        console.log(input);
        var inputValueFilter=/^[1-6]$/;
        
        if(!inputValueFilter.test(input)){alert("请正确输入扰动次数！");document.getElementById(id).value = ""; ;}
    }
    if(id == "max_move"){
        console.log(input);
        var inputValueFilter=/^[0]\.[\d]+$/;
        
        if(!inputValueFilter.test(input)){alert("请正确输入最大扰动值！");document.getElementById(id).value = ""; ;}
    }
    if(id == "min_move"){
        console.log(input);
        console.log("##");
        
        var inputValueFilter=/^[0]\.[\d]+$/;
        if(document.getElementById("max_move").value == ""){alert("请先输入最大扰动值！");document.getElementById(id).value = "";}
        else if(!inputValueFilter.test(input) || parseFloat(document.getElementById("max_move").value)<parseFloat(document.getElementById("min_move").value)){alert("请正确输入最小扰动值！");document.getElementById(id).value = "";}
    }
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
                fontSize: 12,
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
function create_task2() {
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

async function upload() {
    var dataset = $('#selectDataset option:selected').val();
    var model = $('#selectModel option:selected').val();
    var datasetSize = $('#dataset_size').val();
    var maxMove = document.getElementById('max_move').value
    var minMove = document.getElementById('min_move').value
    var moveTimes = document.getElementById('move_times').value
    console.log("参数确认：")
    console.log(dataset, model, datasetSize, maxMove, minMove, moveTimes);
    if(dataset == "" || model == "" || datasetSize == "" || maxMove == "" || minMove == "" || moveTimes == ""){
        alert("请确认输入选项不为空！"); return;
    }
    $(".result_body1").show();
    $(".post-content1")[0].style.height="160px";
    $(".result_body").hide();
    $(".loading").show();

    max = 100;
    create_task2();
    taskid = storage["Taskid"]
    data = {
        dataset: dataset,
        model: model,
        size: datasetSize,
        up_eps: maxMove,
        down_eps: minMove,
        steps: moveTimes,
        type: "forward",
        tid:taskid
    }

    $.ajax({
        type: "POST",
        cache: false,
        data: data,
        url: "/FormalVerification",
        dataType: "json",
        success: function (res) {
            clk = self.setInterval("update()", 5000);
        },
        error: function (jqXHR) {
            console.error(jqXHR);
        }
    });
    
    // logLoading();
}
