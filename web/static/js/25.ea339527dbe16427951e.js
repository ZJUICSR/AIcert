webpackJsonp([25],{"5C0S":function(t,a,e){"use strict";Object.defineProperty(a,"__esModule",{value:!0});var s=e("R45V"),i=e("83tA"),l=e("T9rv"),o=e("UI/F"),n=e("Cvks"),r=e.n(n),c=e("2b9l"),m=e.n(c),d={template:'\n        <svg t="1680138013828" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4354" width="128" height="128"><path d="M534.869333 490.496a1403.306667 1403.306667 0 0 0 50.858667-25.813333c16.042667-8.618667 29.013333-15.061333 38.570667-19.029334 9.557333-3.925333 17.066667-6.058667 22.869333-6.058666 9.557333 0 17.749333 3.2 24.917333 10.026666 6.826667 6.826667 10.581333 15.061333 10.581334 25.088 0 5.76-1.706667 11.818667-5.12 17.92-3.413333 6.101333-7.168 10.069333-10.922667 11.861334-35.157333 14.677333-74.410667 25.429333-116.736 31.872 7.850667 7.168 17.066667 17.237333 28.330667 29.781333 11.264 12.544 17.066667 18.986667 17.749333 20.053333 4.096 6.101333 9.898667 13.653333 17.408 22.613334 7.509333 8.96 12.629333 15.786667 15.36 20.778666 2.730667 5.034667 4.437333 11.093333 4.437333 18.304a33.706667 33.706667 0 0 1-9.898666 24.021334 33.834667 33.834667 0 0 1-25.6 10.410666c-10.24 0-22.186667-8.618667-35.157334-25.472-12.970667-16.512-30.037333-46.933333-50.517333-91.050666-20.821333 39.424-34.816 65.962667-41.642667 78.506666-7.168 12.544-13.994667 22.186667-20.48 28.672a30.976 30.976 0 0 1-22.528 9.685334 32.256 32.256 0 0 1-25.258666-11.093334 35.413333 35.413333 0 0 1-9.898667-23.68c0-7.893333 1.365333-13.653333 4.096-17.578666 25.258667-35.84 51.541333-67.413333 78.848-93.568a756.650667 756.650667 0 0 1-61.44-12.544 383.061333 383.061333 0 0 1-57.685333-20.48c-3.413333-1.749333-6.485333-5.717333-9.557334-11.818667a30.208 30.208 0 0 1-5.12-16.853333 32.426667 32.426667 0 0 1 10.581334-25.088 33.152 33.152 0 0 1 24.234666-10.026667c6.485333 0 14.677333 2.133333 24.576 6.101333 9.898667 4.266667 22.186667 10.026667 37.546667 18.261334 15.36 7.893333 32.426667 16.853333 51.882667 26.538666-3.413333-18.261333-6.485333-39.082667-8.874667-62.378666-2.389333-23.296-3.413333-39.424-3.413333-48.042667 0-10.752 3.072-19.712 9.557333-27.264A30.677333 30.677333 0 0 1 512.341333 341.333333c9.898667 0 18.090667 3.925333 24.576 11.477334 6.485333 7.893333 9.557333 17.92 9.557334 30.464 0 3.584-0.682667 10.410667-1.365334 20.48-0.682667 10.368-2.389333 22.570667-4.096 36.906666-2.048 14.677333-4.096 31.146667-6.144 49.834667z" fill="#FF3838" p-id="4355"></path></svg>\n        '},v={template:'\n        <a-icon :component="selectSvg" />\n    ',data:function(){return{selectSvg:d}}},u={name:"coverage_layer",components:{navmodule:s.a,func_introduce:i.a,showLog:l.a,resultDialog:o.a,selectIcon:v},data:function(){return{radioStyle:{display:"block",lineHeight:"30px"},datasetChoice:"CIFAR10",MNIST_imgs:[{imgUrl:e("wlDy"),name:"mnist0"},{imgUrl:e("KdR5"),name:"mnist1"},{imgUrl:e("2LXz"),name:"mnist2"},{imgUrl:e("3rtV"),name:"mnist3"},{imgUrl:e("Ppgw"),name:"mnist4"},{imgUrl:e("P9ea"),name:"mnist5"},{imgUrl:e("ec3M"),name:"mnist6"},{imgUrl:e("wAAx"),name:"mnist7"},{imgUrl:e("30PV"),name:"mnist8"},{imgUrl:e("v3v+"),name:"mnist9"}],CIFAR10_imgs:[{imgUrl:e("HiaR"),name:"mnist0"},{imgUrl:e("DJt5"),name:"mnist1"},{imgUrl:e("S99w"),name:"mnist2"},{imgUrl:e("J598"),name:"mnist3"},{imgUrl:e("/pRs"),name:"mnist4"},{imgUrl:e("YuLH"),name:"mnist5"},{imgUrl:e("Nvyw"),name:"mnist6"},{imgUrl:e("lB35"),name:"mnist7"},{imgUrl:e("dKp5"),name:"mnist8"},{imgUrl:e("NgyD"),name:"mnist9"}],modelChoice:"VGG11",thresHold:"",imageNumber:"",buttonBGColor:{background:"#0B55F4",color:"#FFFFFF"},disStatus:!1,logflag:!1,percent:10,logtext:[],funcDesText:{name:"标准化单元测试",imgpath:r.a,bgimg:m.a,destext:"多测试准则的标准化AI模型单元测试方法",backinfo:"评估模型训练效果时，由于测试数据有限，容易出现模型行中的某些行为无法被测试到的情况；平台开发标准化单元测试模块，提出制定多测试准则的标准化AI模型单元测试方法，全面评估模型训练效果。",highlight:["测试准则5种，满足模型鲁棒性评估与测试数据充分性评估需求","多粒度神经元覆盖准则（包含单神经元和神经层覆盖测试准则）与重要神经元覆盖准则从不同角度评估测试数据集的充分性","敏感神经元测试准则用于评估模型鲁棒性、逻辑神经元测试准则用于评估模型安全性"]},isShowPublish:!1,result:{},mark:0,res_tmp:{},tid:"",stid:"",clk:"",logclk:""}},watch:{isShowPublish:{immediate:!0,handler:function(t){t?this.noScroll():this.canScroll()}}},created:function(){document.title="标准化单元测试"},methods:{closeDialog:function(){this.isShowPublish=!1},onDatasetChoiceChange:function(t){console.log("radio checked",t.target.value),"MNIST"==t.target.value?this.modelChoice="LeNet5":this.modelChoice="VGG11"},onModelChoiceChange:function(t){console.log("radio checked",t.target.value),"LeNet5"==t.target.value?this.datasetChoice="MNIST":this.datasetChoice="CIFAR10"},onThresholdChange:function(t){""!=t.target.value&&(this.thresHold=t.target.value,console.log("Threshold: ",this.thresHold))},onImagesNumberChange:function(t){""!=t.target.value&&(console.log("ImagesNumber: ",t.target.value),this.imageNumber=t.target.value)},downloadGeneration:function(){if(confirm("您确认下载生成测试样本？")){alert("开发中，敬请期待！")}},exportResult:function(){if(confirm("您确认下载该pdf文件吗？")){document.body.scrollTop=document.documentElement.scrollTop=0;var t=document.getElementById("download_page"),a={margin:[10,20,10,20],filename:this.tid+".pdf",image:{type:"jpeg",quality:1},html2canvas:{scale:5},jsPDF:{unit:"mm",format:"a4",orientation:"portrait"}};html2pdf().from(t).set(a).save()}},resultPro:function(t){this.result.img_list=t.CoverageLayer.coverage_test_yz.coverage_layer;for(var a=0;a<this.result.img_list.length;a++)this.result.img_list[a].coverage=parseInt(100*this.result.img_list[a].coverage);this.play()},autoPlay:function(){this.mark++,this.mark==this.result.img_list.length&&(this.mark=0)},play:function(){setInterval(this.autoPlay,2e3)},getData:function(){var t=this;t.$axios.get("/output/Resultdata",{params:{Taskid:t.tid}}).then(function(a){console.log("dataget:",a),t.res_tmp=a})},getLog:function(){var t=this;t.percent<99&&(t.percent+=1),t.logflag=!0,t.$axios.get("/Task/QueryLog",{params:{Taskid:t.tid}}).then(function(a){t.logtext=a.data.Log[t.stid]})},stopTimer:function(){this.res_tmp.data.stop&&(this.logflag=!1,clearInterval(this.clk),clearInterval(this.logclk),this.isShowPublish=!0,this.resultPro(this.res_tmp.data.result))},update:function(){this.getData();try{this.stopTimer()}catch(t){}},changeSelectPage:function(){},dataEvaClick:function(){if(""!=this.thresHold)if(""!=this.imageNumber){var t=this;this.$axios.post("/Task/CreateTask",{AttackAndDefenseTask:0}).then(function(a){t.tid=a.data.Taskid;var e={dataset:t.datasetChoice,model:t.modelChoice,k:t.thresHold,N:t.imageNumber,tid:t.tid};t.$axios.post("/UnitTest/CoverageLayerParamSet",e).then(function(a){t.logflag=!0,t.stid=a.data.stid,t.logclk=self.setInterval(t.getLog,500),t.clk=self.setInterval(t.update,500)}).catch(function(t){console.log(t)})}).catch(function(t){console.log(t)})}else this.$message.warning("请输入测试图片数量！",3);else this.$message.warning("请输入神经元激活阈值！",3)}}},h={render:function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",[e("a-layout",[e("a-layout-header",[e("navmodule")],1),t._v(" "),e("a-layout-content",[e("func_introduce",{attrs:{funcDesText:t.funcDesText}}),t._v(" "),e("div",{staticClass:"paramCon"},[e("h2",{staticClass:"subTitle",staticStyle:{"margin-top":"-96px"}},[t._v("参数配置")]),t._v(" "),e("div",{staticClass:"labelSelection"},[e("router-link",{attrs:{to:"/coverage_neural"}},[e("button",{staticClass:"labelunselected"},[t._v("单神经元覆盖准则")])]),t._v(" "),e("router-link",{attrs:{to:"/coverage_layer"}},[e("button",{staticClass:"labelselected"},[t._v("神经元层覆盖准则")])]),t._v(" "),e("router-link",{attrs:{to:"/coverage_importance"}},[e("button",{staticClass:"labelunselected"},[t._v("重要神经元覆盖准则")])]),t._v(" "),e("router-link",{attrs:{to:"/deepsst"}},[e("button",{staticClass:"labelunselected"},[t._v("敏感神经元测试准则")])]),t._v(" "),e("router-link",{attrs:{to:"/deeplogic"}},[e("button",{staticClass:"labelunselected"},[t._v("逻辑神经元测试准则")])])],1),t._v(" "),e("div",{staticClass:"funcParam"},[e("div",{staticClass:"paramTitle"},[e("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),e("h3",[t._v(t._s(t.funcDesText.name))]),t._v(" "),e("a-button",{staticClass:"DataEva",style:t.buttonBGColor,attrs:{disabled:t.disStatus},on:{click:t.dataEvaClick}},[e("a-icon",{attrs:{type:"security-scan"}}),t._v("\n                       评估\n                   ")],1)],1),t._v(" "),e("a-divider"),t._v(" "),e("div",{staticClass:"inputdiv"},[e("div",{staticClass:"datasetSelected"},[e("p",{staticClass:"mainParamNameNotop"},[t._v("请选择数据集")]),t._v(" "),e("a-radio-group",{on:{change:t.onDatasetChoiceChange},model:{value:t.datasetChoice,callback:function(a){t.datasetChoice=a},expression:"datasetChoice"}},[e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"CIFAR10"}},[t._v("\n                                   CIFAR10\n                               ")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[e("span",[t._v("CIFAR10数据集：")]),t._v("是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图 片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[t._v("图例：")]),t._v(" "),e("div",{staticClass:"demoData"},t._l(t.CIFAR10_imgs,function(t,a){return e("div",{key:a},[e("img",{attrs:{src:t.imgUrl}})])}),0)],1),t._v(" "),e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"MNIST"}},[t._v("\n                                   MNIST\n                               ")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[e("span",[t._v("MNIST数据集：")]),t._v("是一个手写体数字的图片数据集，该数据集来由美国国家标准与技术研究所（National Institute of Standards and Technology (NIST)）发起整理，一共统计了来自250个不同的人手写数字图片，其中50%是高中生，50%来自人口普查局的工作人员。该数据集的收集目的是希望通过算法，实现对手写数字的识别。")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[t._v("图例：")]),t._v(" "),e("div",{staticClass:"demoData"},t._l(t.MNIST_imgs,function(t,a){return e("div",{key:a},[e("img",{attrs:{src:t.imgUrl}})])}),0)],1)])],1),t._v(" "),e("div",{staticClass:"modelSelected"},[e("p",{staticClass:"mainParamName"},[t._v("请选择模型")]),t._v(" "),e("a-radio-group",{on:{change:t.onModelChoiceChange},model:{value:t.modelChoice,callback:function(a){t.modelChoice=a},expression:"modelChoice"}},[e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"VGG11"}},[t._v("VGG11")]),t._v(" "),e("a-radio",{style:t.radioStyle,attrs:{value:"VGG13"}},[t._v("VGG13")]),t._v(" "),e("a-radio",{style:t.radioStyle,attrs:{value:"VGG19"}},[t._v("VGG19")]),t._v(" "),e("a-radio",{style:t.radioStyle,attrs:{value:"ResNet18"}},[t._v("ResNet18")]),t._v(" "),e("a-radio",{style:t.radioStyle,attrs:{value:"ResNet34"}},[t._v("ResNet34")]),t._v(" "),e("a-radio",{style:t.radioStyle,attrs:{value:"LeNet5"}},[t._v("LeNet5")])],1)])],1),t._v(" "),e("div",{staticClass:"thresholdSet"},[e("p",{staticClass:"mainParamName"},[t._v("请输入神经元激活阈值")]),t._v(" "),e("a-input",{staticClass:"paramsInput",attrs:{id:"param_runtimes",placeholder:"神经元激活值应该排在该层神经元的前多少位，范围[0,1]，默认值0.1"},on:{change:t.onThresholdChange}})],1),t._v(" "),e("div",{staticClass:"imagesTested"},[e("p",{staticClass:"mainParamName"},[t._v("请输入测试图片数量")]),t._v(" "),e("a-input",{staticClass:"paramsInput",attrs:{id:"param_runtimes",placeholder:"请输入测试图片数量，范围是[1,10000]"},on:{change:t.onImagesNumberChange}})],1)])],1)]),t._v(" "),t.logflag?e("div",[e("showLog",{attrs:{percent:t.percent,logtext:t.logtext}})],1):t._e(),t._v(" "),e("resultDialog",{directives:[{name:"show",rawName:"v-show",value:t.isShowPublish,expression:"isShowPublish"}],attrs:{isShow:t.isShowPublish},on:{"on-close":t.closeDialog}},[e("div",{attrs:{slot:"header"},slot:"header"},[e("div",{staticClass:"dialog_title"},[e("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),e("h1",[t._v("神经层覆盖准则")])])]),t._v(" "),e("div",{staticClass:"dialog_publish_main",attrs:{slot:"main",id:"download_page"},slot:"main"},[e("div",{staticClass:"result_div"},[e("div",{staticClass:"conclusion_info"},[e("p",{staticClass:"result_annotation"},[t._v("数据集："+t._s(t.datasetChoice))]),t._v(" "),e("p",{staticClass:"result_annotation"},[t._v("模型："+t._s(t.modelChoice))]),t._v(" "),e("p",{staticClass:"result_annotation"},[t._v("神经元激活阈值："+t._s(t.thresHold))]),t._v(" "),e("p",{staticClass:"result_annotation"},[t._v("测试图片数量："+t._s(t.imageNumber))])]),t._v(" "),e("div",{staticClass:"main_top_echarts_con_title"},[t._v("神经层覆盖测试准则")]),t._v(" "),e("div",{attrs:{id:"rdeva"}},[e("div",{staticClass:"box"},t._l(t.result.img_list,function(a,s){return e("div",{directives:[{name:"show",rawName:"v-show",value:s==t.mark,expression:"index==mark"}],key:s},[e("img",{staticClass:"graph_show",attrs:{src:a.imgUrl,alt:""}}),t._v(" "),e("p",[t._v("当前覆盖率："+t._s(a.coverage)+"%")])])}),0),t._v(" "),e("div",{staticClass:"conclusion"},[e("p",{staticClass:"result_text"},[t._v("理论上经过充分测试的模型覆盖率应该接近100%，如果覆盖率小于80%，则模型存在安全隐患的可能性较大。由于深度网络参数过多，图片里进行压缩显示，每个圆点代表多个神经元，圆点的深度代表对应神经元被激活的比例，深蓝色为全部激活。对于超大模型，只显示前20层的激活情况，但覆盖率数值对应整个模型。")])])])]),t._v(" "),e("div",[e("button",{staticClass:"exportResultBtn",on:{click:t.exportResult}},[e("a-icon",{attrs:{type:"upload"}}),t._v("导出报告内容")],1)])])])],1),t._v(" "),e("a-layout-footer")],1)],1)},staticRenderFns:[]};var g=e("VU/8")(u,h,!1,function(t){e("h8r7")},"data-v-3654f24a",null);a.default=g.exports},h8r7:function(t,a){}});