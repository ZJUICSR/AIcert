webpackJsonp([11],{Q4ey:function(t,e,a){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var s=a("gRE1"),o=a.n(s),i=a("mvHQ"),n=a.n(i),l=a("R45V"),c=a("83tA"),r=a("T9rv"),d=a("UI/F"),v={name:"popDialog",props:{isShow:{type:Boolean,default:!1,required:!0},widNum:{type:Number,default:50},leftSite:{type:Number,default:25.2},topDistance:{type:Number,default:10},pdt:{type:Number,default:22},pdb:{type:Number,default:47}},methods:{closeMyself:function(){this.$emit("on-close")},_stopPropagation:function(t){t.stopPropagation()}}},h={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"dialog"},[t.isShow?a("div",{staticClass:"dialog-cover back",on:{click:t.closeMyself}}):t._e(),t._v(" "),a("transition",{attrs:{name:"drop"}},[t.isShow?a("div",{staticClass:"dialog-content",on:{click:function(e){return e.stopPropagation(),t._stopPropagation(e)}}},[a("div",{staticClass:"dialog_head back"},[a("div",{staticClass:"close_button"},[a("a-icon",{staticClass:"closebutton",staticStyle:{"font-size":"20px",color:"#6C7385"},attrs:{type:"close"},on:{click:t.closeMyself}})],1),t._v(" "),t._t("header",function(){return[t._v("结果报告")]})],2),t._v(" "),a("div",{staticClass:"dialog_main"},[t._t("main",function(){return[t._v("弹窗内容")]})],2)]):t._e()])],1)},staticRenderFns:[]};var u=a("VU/8")(v,h,!1,function(t){a("ZtNU")},"data-v-7d3b153f",null).exports,m={name:"onlineProcess",props:{isShow:{type:Boolean,default:!1,required:!0},widNum:{type:Number,default:50},leftSite:{type:Number,default:25.2},topDistance:{type:Number,default:10},pdt:{type:Number,default:22},pdb:{type:Number,default:47}},methods:{closeMyself:function(){this.$emit("on-close")},_stopPropagation:function(t){t.stopPropagation()}}},p={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"dialog"},[t.isShow?a("div",{staticClass:"dialog-cover back",on:{click:t.closeMyself}}):t._e(),t._v(" "),a("transition",{attrs:{name:"drop"}},[t.isShow?a("div",{staticClass:"dialog-content",on:{click:function(e){return e.stopPropagation(),t._stopPropagation(e)}}},[a("div",{staticClass:"dialog_head back"},[a("div",{staticClass:"close_button"},[a("a-icon",{staticClass:"closebutton",staticStyle:{"font-size":"20px",color:"#6C7385"},attrs:{type:"close"},on:{click:t.closeMyself}})],1),t._v(" "),t._t("header",function(){return[t._v("结果报告")]})],2),t._v(" "),a("div",{staticClass:"dialog_main"},[t._t("main",function(){return[t._v("弹窗内容")]})],2)]):t._e()])],1)},staticRenderFns:[]};var _=a("VU/8")(m,p,!1,function(t){a("r9Kz")},"data-v-a34c85ea",null).exports,g=a("FaAE"),C=a("juNB"),f=a.n(C),w=a("2b9l"),b=a.n(w),y={template:'\n       <svg t="1680138013828" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4354" width="128" height="128"><path d="M534.869333 490.496a1403.306667 1403.306667 0 0 0 50.858667-25.813333c16.042667-8.618667 29.013333-15.061333 38.570667-19.029334 9.557333-3.925333 17.066667-6.058667 22.869333-6.058666 9.557333 0 17.749333 3.2 24.917333 10.026666 6.826667 6.826667 10.581333 15.061333 10.581334 25.088 0 5.76-1.706667 11.818667-5.12 17.92-3.413333 6.101333-7.168 10.069333-10.922667 11.861334-35.157333 14.677333-74.410667 25.429333-116.736 31.872 7.850667 7.168 17.066667 17.237333 28.330667 29.781333 11.264 12.544 17.066667 18.986667 17.749333 20.053333 4.096 6.101333 9.898667 13.653333 17.408 22.613334 7.509333 8.96 12.629333 15.786667 15.36 20.778666 2.730667 5.034667 4.437333 11.093333 4.437333 18.304a33.706667 33.706667 0 0 1-9.898666 24.021334 33.834667 33.834667 0 0 1-25.6 10.410666c-10.24 0-22.186667-8.618667-35.157334-25.472-12.970667-16.512-30.037333-46.933333-50.517333-91.050666-20.821333 39.424-34.816 65.962667-41.642667 78.506666-7.168 12.544-13.994667 22.186667-20.48 28.672a30.976 30.976 0 0 1-22.528 9.685334 32.256 32.256 0 0 1-25.258666-11.093334 35.413333 35.413333 0 0 1-9.898667-23.68c0-7.893333 1.365333-13.653333 4.096-17.578666 25.258667-35.84 51.541333-67.413333 78.848-93.568a756.650667 756.650667 0 0 1-61.44-12.544 383.061333 383.061333 0 0 1-57.685333-20.48c-3.413333-1.749333-6.485333-5.717333-9.557334-11.818667a30.208 30.208 0 0 1-5.12-16.853333 32.426667 32.426667 0 0 1 10.581334-25.088 33.152 33.152 0 0 1 24.234666-10.026667c6.485333 0 14.677333 2.133333 24.576 6.101333 9.898667 4.266667 22.186667 10.026667 37.546667 18.261334 15.36 7.893333 32.426667 16.853333 51.882667 26.538666-3.413333-18.261333-6.485333-39.082667-8.874667-62.378666-2.389333-23.296-3.413333-39.424-3.413333-48.042667 0-10.752 3.072-19.712 9.557333-27.264A30.677333 30.677333 0 0 1 512.341333 341.333333c9.898667 0 18.090667 3.925333 24.576 11.477334 6.485333 7.893333 9.557333 17.92 9.557334 30.464 0 3.584-0.682667 10.410667-1.365334 20.48-0.682667 10.368-2.389333 22.570667-4.096 36.906666-2.048 14.677333-4.096 31.146667-6.144 49.834667z" fill="#FF3838" p-id="4355"></path></svg>\n       '},D={template:'\n           <a-icon :component="selectSvg" />\n       ',data:function(){return{selectSvg:y}}},S={name:"modularDevelop",components:{navmodule:l.a,func_introduce:c.a,showLog:r.a,resultDialog:d.a,selectIcon:D,drawAcc_or_loss:g.b,popDialog:u,onlineProcess:_},data:function(){return{htmlTitle:"模型模块化开发",radioStyle:{display:"block",lineHeight:"30px",width:"100%"},MNIST_imgs:[{imgUrl:a("wlDy"),name:"mnist0"},{imgUrl:a("KdR5"),name:"mnist1"},{imgUrl:a("2LXz"),name:"mnist2"},{imgUrl:a("3rtV"),name:"mnist3"},{imgUrl:a("Ppgw"),name:"mnist4"},{imgUrl:a("P9ea"),name:"mnist5"},{imgUrl:a("ec3M"),name:"mnist6"},{imgUrl:a("wAAx"),name:"mnist7"},{imgUrl:a("30PV"),name:"mnist8"},{imgUrl:a("v3v+"),name:"mnist9"}],CIFAR10_imgs:[{imgUrl:a("HiaR"),name:"mnist0"},{imgUrl:a("DJt5"),name:"mnist1"},{imgUrl:a("S99w"),name:"mnist2"},{imgUrl:a("J598"),name:"mnist3"},{imgUrl:a("/pRs"),name:"mnist4"},{imgUrl:a("YuLH"),name:"mnist5"},{imgUrl:a("Nvyw"),name:"mnist6"},{imgUrl:a("lB35"),name:"mnist7"},{imgUrl:a("dKp5"),name:"mnist8"},{imgUrl:a("NgyD"),name:"mnist9"}],searchChoice:"DREAM",initChoice:"Normal",searchTimes:"4",trainTimes:"4",modelChoice:"VGG",datasetChoice:"CIFAR10",buttonBGColor:{background:"#0B55F4",color:"#FFFFFF"},disStatus:!1,logflag:!1,percent:10,logtext:[],funcDesText:{name:"AI模型模块化开发",imgpath:f.a,bgimg:b.a,destext:"高效率选择并自动化构造AI模型",backinfo:"为提升AI系统开发效率，辅助开发者设计并获取高性能AI模型，基于模块化的思想，为给定任务选择合适的模型架构与模型以及训练中的详细超参数，功能进行组件式构建、自动化训练并保存模型，并可视化显示模型的各项性能指标。",highlight:["支持构建深度学习模型5种，模型参数超过100万","实现高效通用化开发，覆盖开发框架3种：PyTorch、TensorFlow、PaddlePaddle","具备组件式系统建模、动态数据收集、功能结构验证能力，经AI模型模块化开发构造的模型开发效率提升≥30%"]},isShowPublish:!1,isShowPopDownload:!1,isShowPopInfer:!1,result:{},res_tmp:{},tid:"",stidlist:"",clk:"",logclk:""}},watch:{isShowPublish:{immediate:!0,handler:function(t){t?this.noScroll():this.canScroll()}}},created:function(){document.title="AI模型模块化开发"},methods:{closeDialog:function(){this.isShowPublish=!1},PopDownloadShow:function(){this.isShowPopDownload=!0},closePopDownloadDialog:function(){this.isShowPopDownload=!1},PopInferenceShow:function(){this.isShowPopInfer=!0},closePopInferenceDialog:function(){this.isShowPopInfer=!1},onSearchChoiceChange:function(t){console.log("radio checked",t.target.value)},onInitChoiceChange:function(t){console.log("radio checked",t.target.value)},onSearchTimesChange:function(t){""!=t.target.value&&(console.log("input search times: ",t.target.value),this.searchTimes=t.target.value),console.log(this.searchTimes)},onModelChoiceChange:function(t){console.log("radio checked",t.target.value)},onTrainTimesChange:function(t){""!=t.target.value&&(console.log("input search times: ",t.target.value),this.trainTimes=t.target.value),console.log(this.trainTimes)},onDatasetChoiceChange:function(t){console.log("radio checked",t.target.value)},downloadGeneration:function(t){if(confirm("您确认下载模型？")){var e=this.result[t.target.value],a=new FormData;a.append("file",e),a.append("type","dictionary"),this.post_file=a;this.$axios.post("/Task/DownloadData",this.post_file,{headers:{"Content-Type":"multipart/form-data"},responseType:"blob"}).then(function(t){console.log(t);var e=new Blob([t.data],{type:"application/zip"});if("download"in document.createElement("a")){var a=document.createElement("a");a.style.display="none",a.href=window.URL.createObjectURL(e),a.download="Generation_Download",a.click(),URL.revokeObjectURL(a.href),document.body.removeChild(a)}else navigator.msSaveBlob(e,"Generation_Download")}).catch(function(t){console.log(t)})}},onlineInference:function(){alert("在线推理接口开发中！")},exportResult:function(){if(confirm("您确认下载该pdf文件吗？")){var t=document.getElementById("download_page"),e={margin:[10,20,10,20],filename:this.tid+".pdf",image:{type:"jpeg",quality:1},html2canvas:{scale:5},jsPDF:{unit:"mm",format:"a4",orientation:"portrait"}};html2pdf().from(t).set(e).save()}},resultPro:function(t){var e=t.ModularDevelop.best_history.accuracy,a=t.ModularDevelop.best_history.loss,s=t.ModularDevelop.best_history.val_accuracy,o=t.ModularDevelop.best_history.val_loss;Object(g.b)("acc_echart",[e,s],["训练准确率","验证准确率"]),Object(g.b)("loss_echart",[a,o],["训练损失","验证损失"]),this.result.PyTorch=t.ModularDevelop.target_torch,this.result.Tensorflow=t.ModularDevelop.target_tensorflow,this.result.PaddlePaddle=t.ModularDevelop.target_paddle},getData:function(){var t=this;t.$axios.get("/output/Resultdata",{params:{Taskid:t.tid}}).then(function(e){console.log("dataget:",e),t.res_tmp=e})},getLog:function(){var t=this;t.percent<99&&(t.percent+=1),t.$axios.get("/Task/QueryLog",{params:{Taskid:t.tid}}).then(function(e){if("{}"==n()(t.stidlist))t.logtext=[o()(e.data.Log).slice(-1)[0]];else for(var a in t.logtext=[],t.stidlist)t.logtext.push(e.data.Log[t.stidlist[a]])})},stopTimer:function(){this.res_tmp.data.stop&&(this.percent=100,this.logflag=!1,clearInterval(this.clk),clearInterval(this.logclk),this.isShowPublish=!0,this.resultPro(this.res_tmp.data.result))},update:function(){this.getData();try{this.stopTimer()}catch(t){}},dataEvaClick:function(){var t=this;if(""==this.trainTimes|""==this.searchTimes)return this.$message.warning("请输入搜索迭代轮数和训练轮数",3),0;var e=this;this.$axios.post("/Task/CreateTask",{AttackAndDefenseTask:0}).then(function(a){e.tid=a.data.Taskid;var s={dataset:e.datasetChoice,model:e.modelChoice,epoch:e.trainTimes,tuner:e.searchChoice,init:e.initChoice,iternum:t.searchTimes,tid:e.tid};console.log(s),e.$axios.post("/MDTest/ModularDevelopParamSet",s).then(function(t){e.logflag=!0,e.stidlist={ModularDevelop:t.data.stid},e.logclk=self.setInterval(e.getLog,3e3),e.clk=self.setInterval(e.update,3e3)}).catch(function(t){console.log(t)})}).catch(function(t){console.log(t)})}}},P={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",[a("a-layout",[a("a-layout-header",[a("navmodule")],1),t._v(" "),a("a-layout-content",[a("func_introduce",{attrs:{funcDesText:t.funcDesText}}),t._v(" "),a("div",{staticClass:"paramCon"},[a("h2",{staticClass:"subTitle",staticStyle:{"margin-top":"-96px"}},[t._v("参数配置")]),t._v(" "),a("div",{staticClass:"funcParam"},[a("div",{staticClass:"paramTitle"},[a("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),a("h3",[t._v(t._s(t.funcDesText.name))]),t._v(" "),a("a-button",{staticClass:"DataEva",style:t.buttonBGColor,attrs:{disabled:t.disStatus},on:{click:t.dataEvaClick}},[a("a-icon",{attrs:{type:"security-scan"}}),t._v("\n                       评估\n                   ")],1)],1),t._v(" "),a("a-divider"),t._v(" "),a("div",{staticClass:"inputdiv"},[a("div",{staticClass:"modelSelected"},[a("p",{staticClass:"mainParamNameNotop"},[t._v("请选择模型搜索方法")]),t._v(" "),a("a-radio-group",{on:{change:t.onSearchChoiceChange},model:{value:t.searchChoice,callback:function(e){t.searchChoice=e},expression:"searchChoice"}},[a("div",{staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"DREAM"}},[t._v("DREAM")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("使用基于反馈的AutoML模型搜索方法， 高效率搜索并构建AI模型")])],1)])],1),t._v(" "),"DREAM"==t.searchChoice?a("div",{staticClass:"modelSelected"},[a("p",{staticClass:"mainParamName"},[t._v("请选择初始化方式")]),t._v(" "),a("a-radio-group",{on:{change:t.onInitChoiceChange},model:{value:t.initChoice,callback:function(e){t.initChoice=e},expression:"initChoice"}},[a("div",{staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"Normal"}},[t._v("Normal")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("默认初始搜索范围")])],1),t._v(" "),a("div",{staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"Large"}},[t._v("Large")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("初始搜索范围较大")])],1),t._v(" "),a("div",{staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"Random"}},[t._v("Random")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("随机选取初始搜索范围")])],1)])],1):t._e(),t._v(" "),a("div",{staticClass:"runtimesSelected"},[a("p",{staticClass:"mainParamName"},[t._v("请输入搜索迭代轮数")]),t._v(" "),a("a-input",{staticClass:"paramsInput",attrs:{id:"param_searchtimes",placeholder:"模型搜索次数，输入范围[3,10]，建议值4"},on:{change:t.onSearchTimesChange}})],1),t._v(" "),a("div",{staticClass:"modelSelected"},[a("p",{staticClass:"mainParamName"},[t._v("请选择构建模型类型")]),t._v(" "),a("a-radio-group",{on:{change:t.onModelChoiceChange},model:{value:t.modelChoice,callback:function(e){t.modelChoice=e},expression:"modelChoice"}},[a("div",{staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"VGG"}},[t._v("VGG")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("包含VGG16、VGG19等")])],1),t._v(" "),a("div",{staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"ResNet"}},[t._v("ResNet")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("包含ResNet50、ResNet101，ResNet 152等")])],1),t._v(" "),a("div",{directives:[{name:"show",rawName:"v-show",value:"DREAM"==t.searchChoice,expression:"searchChoice == 'DREAM'"}],staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"vanilla"}},[t._v("CNN")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("普通卷积神经网络")])],1),t._v(" "),a("div",{directives:[{name:"show",rawName:"v-show",value:"DeepAlchemy"==t.searchChoice,expression:"searchChoice == 'DeepAlchemy'"}],staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"MobileNet"}},[t._v("MobileNet")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("MobileNet网络模型")])],1)])],1),t._v(" "),a("div",{staticClass:"runtimesSelected"},[a("p",{staticClass:"mainParamName"},[t._v("请输入每次搜索的训练轮数")]),t._v(" "),a("a-input",{staticClass:"paramsInput",attrs:{id:"param_traintimes",placeholder:"每次搜索迭代中训练次数，输入范围[3,10]，建议值4"},on:{change:t.onTrainTimesChange}})],1),t._v(" "),a("div",{staticClass:"datasetSelected"},[a("p",{staticClass:"mainParamName"},[t._v("请选择数据集")]),t._v(" "),a("a-radio-group",{on:{change:t.onDatasetChoiceChange},model:{value:t.datasetChoice,callback:function(e){t.datasetChoice=e},expression:"datasetChoice"}},[a("div",{staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"CIFAR10"}},[t._v("\n                                    CIFAR10\n                                ")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[a("span",[t._v("CIFAR10数据集：")]),t._v("是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图 片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("图例：")]),t._v(" "),a("div",{staticClass:"demoData"},t._l(t.CIFAR10_imgs,function(t,e){return a("div",{key:e},[a("img",{attrs:{src:t.imgUrl}})])}),0)],1)])],1)])],1)]),t._v(" "),t.logflag?a("div",[a("showLog",{attrs:{percent:t.percent,logtext:t.logtext}})],1):t._e(),t._v(" "),a("resultDialog",{directives:[{name:"show",rawName:"v-show",value:t.isShowPublish,expression:"isShowPublish"}],ref:"report_pdf",attrs:{isShow:t.isShowPublish},on:{"on-close":t.closeDialog}},[a("div",{attrs:{slot:"header"},slot:"header"},[a("div",{staticClass:"dialog_title"},[a("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),a("h1",[t._v("AI模型模块化开发")])])]),t._v(" "),a("div",{staticClass:"dialog_publish_main",attrs:{slot:"main"},slot:"main"},[a("div",{staticClass:"result_div"},[a("div",{staticClass:"conclusion_info"},[a("p",{staticClass:"result_annotation"},[t._v("搜索方法："+t._s(t.searchChoice))]),t._v(" "),"DREAM"==t.searchChoice?a("p",{staticClass:"result_annotation"},[t._v("初始化方式："+t._s(t.initChoice))]):t._e(),t._v(" "),a("p",{staticClass:"result_annotation"},[t._v("搜索迭代轮数："+t._s(t.searchTimes))]),t._v(" "),a("p",{staticClass:"result_annotation"},[t._v("构建模型类型："+t._s(t.modelChoice))]),t._v(" "),a("p",{staticClass:"result_annotation"},[t._v("每次搜索训练轮数："+t._s(t.trainTimes))]),t._v(" "),a("p",{staticClass:"result_annotation"},[t._v("数据集："+t._s(t.datasetChoice))])]),t._v(" "),a("div",{staticClass:"result_row"},[a("div",{staticClass:"echart_title"},[a("div",{staticClass:"main_top_echarts_con_title"},[t._v("准确率曲线")]),t._v(" "),a("div",{staticClass:"box",attrs:{id:"acc_echart"}})]),t._v(" "),a("div",{staticClass:"echart_title"},[a("div",{staticClass:"main_top_echarts_con_title"},[t._v("损失曲线")]),t._v(" "),a("div",{staticClass:"box",attrs:{id:"loss_echart"}})])]),t._v(" "),a("div",{attrs:{id:"rdeva"}},[a("div",{staticClass:"conclusion"},[a("p",{staticClass:"result_text"},[t._v("X轴代表每个方法迭代搜索的次数，左右两图纵轴分别是历史最优的准确率 "+t._s()+"和损失函数数值 "+t._s())]),t._v(" "),a("p",{staticClass:"result_text"},[t._v("两张图展示了各个搜索方法在不断迭代中逐渐找到准确率更高、损失更低的模型及参数等模块化的组合，最终为给定任务构造和开发一个性能优秀的模型。")])])])]),t._v(" "),a("div",{staticClass:"button_group"},[a("button",{staticClass:"downloadGenerationBtn",on:{click:t.PopDownloadShow}},[a("a-icon",{attrs:{type:"download"}}),t._v("下载模型")],1),t._v(" "),a("a-button",{staticStyle:{width:"160px",height:"50px","margin-bottom":"30px","margin-top":"10px","font-size":"18px",color:"white","background-color":"rgb(46, 56, 245)","border-radius":"8px"},on:{click:function(e){return t.getPdf()}}},[t._v("\n                    导出报告内容\n                    ")]),t._v(" "),a("button",{staticClass:"downloadGenerationBtn",on:{click:t.PopInferenceShow}},[t._v("在线推理")])],1),t._v(" "),a("popDialog",{directives:[{name:"show",rawName:"v-show",value:t.isShowPopDownload,expression:"isShowPopDownload"}],attrs:{isShow:t.isShowPopDownload},on:{"on-close":t.closePopDownloadDialog}},[a("div",{staticClass:"pop_text",attrs:{slot:"header"},slot:"header"},[t._v("下载模型")]),t._v(" "),a("div",{attrs:{slot:"main"},slot:"main"},[a("p",{staticClass:"pop_content"},[t._v("请选择模型框架类型")]),t._v(" "),a("div",{staticClass:"pop_button_group"},[a("button",{staticClass:"pop_button",attrs:{value:"PyTorch"},on:{click:t.downloadGeneration}},[a("a-icon",{attrs:{type:"download"}}),t._v("PyTorch")],1),t._v(" "),a("button",{staticClass:"pop_button",attrs:{value:"TensorFlow"},on:{click:t.downloadGeneration}},[a("a-icon",{attrs:{type:"download"}}),t._v("TensorFlow")],1),t._v(" "),a("button",{staticClass:"pop_button",attrs:{value:"PaddlePaddle"},on:{click:t.downloadGeneration}},[a("a-icon",{attrs:{type:"download"}}),t._v("PaddlePaddle")],1)])])]),t._v(" "),a("onlineProcess",{directives:[{name:"show",rawName:"v-show",value:t.isShowPopInfer,expression:"isShowPopInfer"}],attrs:{isShow:t.isShowPopInfer},on:{"on-close":t.closePopInferenceDialog}},[a("div",{attrs:{slot:"header"},slot:"header"},[a("p",{staticClass:"pop_text"},[t._v("模型推理")]),t._v(" "),a("div",{staticClass:"pop_button_group_"},[a("button",{staticClass:"pop_button",on:{click:t.onlineInference}},[a("a-icon",{attrs:{type:"upload"}}),t._v("上传图片")],1),t._v(" "),a("button",{staticClass:"pop_button",on:{click:t.onlineInference}},[a("a-icon",{attrs:{type:"file-image",theme:"twoTone"}}),t._v("添加图片到数据集")],1)])]),t._v(" "),a("div",{attrs:{slot:"main"},slot:"main"},[a("div",{staticClass:"pop_button_group_ inferclass"},[a("button",{staticClass:"upload_buuton",on:{click:t.onlineInference}},[t._v("+")]),t._v(" "),a("button",{staticClass:"upload_buuton"},[t._v("预测结果"+t._s())])])])])],1)])],1),t._v(" "),a("a-layout-footer")],1)],1)},staticRenderFns:[]};var x=a("VU/8")(S,P,!1,function(t){a("mSNm")},"data-v-4ef9be8a",null);e.default=x.exports},ZtNU:function(t,e){},juNB:function(t,e,a){t.exports=a.p+"static/img/modularDevelopIcon.a2968e0.png"},mSNm:function(t,e){},r9Kz:function(t,e){}});