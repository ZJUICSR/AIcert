// 评估算法树状图
function draw_eva_method_echart(ID){
    var eva_method_echart = document.getElementById(ID);
    var myChartcons = echarts.init(eva_method_echart);
    window.addEventListener("resize", function () {
    myChartcons.resize()});
    var data={
      "name":"公平性评估",
      'itemStyle':{"color":"#F56C6C","borderColor":"white"},
      "children":
      [{
        "name":"数据集",
        "children":[
          {"name":"xxx",
          "value":1
          },
          {"name":"xxx",
          "value":1},
          {"name":"xxx",
          "value":1},
          {"name":"xxx",
          "value":1
          }
        ],
        'itemStyle':{"color":"#E6A23C","borderColor":"#fff"},
      },
      {
        "name":"模型",
        'itemStyle':{"color":"#67C23A","borderColor":"#FFF"},
        "children":[
        {"name":"DI","value":1},
        {"name":"DP","value":1},
        {"name":"DP_norm","value":1},
        {"name":"OMd","value":1},
        {"name":"OMr","value":1},
        {"name":"FPn","value":1},
        {"name":"FPd","value":1},
        {"name":"FPr","value":1},
        {"name":"TPn","value":1},
        {"name":"TPd","value":1},
        {"name":"TPr","value":1},
        {"name":"FNn","value":1},
        {"name":"FNd","value":1},
        {"name":"FNr","value":1},
        {"name":"TNn","value":1},
        {"name":"TNd","value":1},
        {"name":"TNr","value":1},
        {"name":"FOd","value":1},
        {"name":"FOr","value":1},
        {"name":"FDd","value":1},
        {"name":"FDr","value":1},
        {"name":"PRd","value":1},
        {"name":"F1d","value":1},
        {"name":"PE","value":1},
        {"name":"EOD","value":1},
        {"name":"PP","value":1},
        {"name":"TEE","value":1},
        {"name":"PED","value":1},
        {"name":"FPR","value":1},
        {"name":"TPR","value":1},
        {"name":"FNR","value":1},
        {"name":"TNR","value":1}
        ]
      }
      ]}
    for (var i =0;i < data.children[1].children.length;i++){
      data.children[1].children[i].itemStyle={"color":"#67C23A","borderColor":"#fff"};
    }
    for (var i =0;i < data.children[0].children.length;i++){
      data.children[0].children[i].itemStyle={"color":"#E6A23C","borderColor":"#fff"};

    }
    data.children.forEach(function (datum, index) {
      index % 2 === 0 && (datum.collapsed = true);
    });

    var option;
    option={
        tooltip: {
            trigger: 'item',
            triggerOn: 'mousemove',
            extraCssText: "width:250px;white-space:pre-wrap;text-align:left",
            formatter:function(arg){
              console.log(arg);
              let info = {
                "PED":"算法介绍",
                "公平性评估":"包含数据集公平性评估和模型公平性评估",
                "数据集":"包含xx，xx，xx，xx四种算法",
                "模型":"包含30种评估算法",
                "DI":"",
                "DP":"",
                "DP_norm":"",
                "OMd":"",
                "OMr":"",
                "FPn":"",
                "FPd":"",
                "FPr":"",
                "TPn":"",
                "TPd":"",
                "TPr":"",
                "FNn":"",
                "FNd":"",
                "FNr":"",
                "TNn":"",
                "TNd":"",
                "TNr":"",
                "FOd":"",
                "FOr":"",
                "FDd":"",
                "FDr":"",
                "PRd":"",
                "F1d":"",
                "PE":"",
                "EOD":"",
                "PP":"",
                "TEE":"",
                "PED":"",
                "FPR":"",
                "TPR":"",
                "FNR":"",
                "TNR":"",
              }
              if(arg.data.name=="数据集"||arg.data.name=="模型"){
                return arg.data.name + '公平性评估算法:' + info[arg.data.name];
              }else if(info[arg.data.name]){
                return 0;
              }else{
                return arg.data.name + '算法:' + info[arg.data.name];
              }
              
            }
          },
          series: {
            type: 'sunburst',
            data: [data],
            radius: [10, '90%'],
            itemStyle: {
              borderRadius: 7,
              borderWidth: 2
            },
            label: {
              rotate: -25,
              show: true,
              extraCssText:"white-space:pre-wrap;"
            }
          }
    }
    option && myChartcons.setOption(option);
};


function draw_eva_dataset_echart(ID){
  var eva_method_echart = document.getElementById(ID);
  var myChartcons = echarts.init(eva_method_echart);
  window.addEventListener("resize", function () {
    myChartcons.resize()});
  var option={
    tooltip: {
      trigger: "item",
      triggerOn: "mousemove",
      extraCssText: "width:250px;white-space:pre-wrap;text-align:left",
			formatter: function(arg) {
        var info ={
          "Compas":"预测刑事被告重新犯罪的可能性。数据包含COMPAS算法在对佛罗里达州布劳沃德县的10,000多名刑事被告进行评分时使用的变量，以及他们在判决后2年内的结果",
          "Adult":"根据人口普查数据预测收入是否超过5万美元/年。Barry Becker从1994年人口普查数据库中提取制作该数据集",
          "German":"预测个人的信用风险。原始数据集包含1000个条目和20个分类/符号属性，由Hofmann教授制作。在此数据集中，每个条目代表一个从银行获得信贷的人。根据属性集，每个人都被分类为良好或不良信用风险"
        }
        return arg.data.name + '数据集:' + info[arg.data.name]
      }
    },
    series: [
    {
      visualMin: -100,    //决定颜色的最小值
      visualMax: 100,     //决定颜色的最大值
      visualDimension: 1, //决定颜色看 value 的哪一个值, 1=> 第2个值
      levels: [
      {                                           //第一层
        color: ['#942e38', '#aaa', '#269f3c'],  //颜色
        colorMappingBy: 'value',                //用value来决定颜色
        itemStyle: {
            borderWidth: 2,
            borderColor: '#fff',
            gapWidth: 1
        }
      }],
      type: 'treemap',
      data: [
        {
          name: 'Adult',
          value: [10,10],
          },
          {
            name: 'German',
            value: [10,40]
          },
          {
            name: 'Compas',
            value: [10,70]
          }
        ]
      }
    ]
  }
  option && myChartcons.setOption(option);
}