webpackJsonp([13],{IDYv:function(t,a){t.exports={name:"VGG16",hard:"Jetson Nano(ARM 架构)",dataset:"CIFAR10",label:["vanilla VGG16","Lightweight VGG16(Before quantization)","Lightweight VGG16(After quantization)"],time:[14.9815,3.73843,.8954],ACC:[.9177,.9215,.9186],size:[59.6252,3.8834,3.8834]}},JEGj:function(t,a){},NC0u:function(t,a){t.exports={name:"VGG16",hard:"NVIDIA TITAN-V GPU",dataset:"CIFAR10",label:["vanilla VGG16","Lightweight VGG16(Before quantization)","Lightweight VGG16(After quantization)"],time:[14.9815,3.73843,.8954],ACC:[.9238,.9233,.9219],size:[59.6249,14.2878,3.7384]}},ofeo:function(t,a){t.exports={name:"VGG16",hard:"Intel i7 7700 CPU",dataset:"CIFAR10",label:["vanilla VGG16","Lightweight VGG16(Before quantization)","Lightweight VGG16(After quantization)"],time:[14.9815,3.73843,.8954],ACC:[.9177,.9215,.9186],size:[59.6252,3.8834,3.8834]}},vtkO:function(t,a,e){"use strict";Object.defineProperty(a,"__esModule",{value:!0});var i=e("R45V"),s=e("83tA"),n=e("T9rv"),o=e("UI/F"),l=e("FaAE"),r=e("uihL"),c=e.n(r),m=e("2b9l"),d=e.n(m),h=e("IDYv"),u=e.n(h),v=e("ofeo"),p=e.n(v),g=e("NC0u"),C=e.n(g),_={name:"inject",components:{navmodule:i.a,func_introduce:s.a,showLog:n.a,resultDialog:o.a},data:function(){return{htmlTitle:"评估报告",radioStyle:{display:"block",lineHeight:"30px"},runTime:0,CIFAR10_imgs:[{imgUrl:e("HiaR"),name:"mnist0"},{imgUrl:e("DJt5"),name:"mnist1"},{imgUrl:e("S99w"),name:"mnist2"},{imgUrl:e("J598"),name:"mnist3"},{imgUrl:e("/pRs"),name:"mnist4"},{imgUrl:e("YuLH"),name:"mnist5"},{imgUrl:e("Nvyw"),name:"mnist6"},{imgUrl:e("lB35"),name:"mnist7"},{imgUrl:e("dKp5"),name:"mnist8"},{imgUrl:e("NgyD"),name:"mnist9"}],modelChoice:"GPU",buttonBGColor:{background:"#0B55F4",color:"#FFFFFF"},disStatus:!1,logflag:!1,percent:10,logtext:[],funcDesText:{name:"基于硬件优化的软硬件一体化验证",imgpath:c.a,bgimg:d.a,destext:"从AI系统硬件运算效率的角度，实现AI系统软硬件的高效安全运行与智能驱动",backinfo:"使用权重剪枝、分支融合、深度可分离卷积、参数量化等技术对网络模型进行压缩与加速，并在CPU、ARM和GPU设备上优化与验证，定量评估优化效果。",highlight:["支持CPU、GPU、ARM三种硬件芯片","支持轻量化优化，在模型准确率基本不变的情况下，极大提升AI系统软硬件的运行效率","支持参数量化优化方法，在轻量化的基础上进一步提升模型运行效率，压缩模型参数，并保证准确率基本不变"]},isShowPublish:!1,result:{CPU:p.a,GPU:C.a,ARM:u.a},postData:{}}},watch:{isShowPublish:{immediate:!0,handler:function(t){t?this.noScroll():this.canScroll()}}},created:function(){document.title="基于硬件优化的软硬件一体化验证"},methods:{onModelChoiceChange:function(t){console.log("radio checked",t.target.value),console.log("model choice:"+this.modelChoice)},closeDialog:function(){this.isShowPublish=!1},exportResult:function(){if(confirm("您确认下载该pdf文件吗？")){document.body.scrollTop=document.documentElement.scrollTop=0;var t=document.getElementById("download_page"),a={margin:[10,20,10,20],filename:this.tid+".pdf",image:{type:"jpeg",quality:1},html2canvas:{scale:5},jsPDF:{unit:"mm",format:"a4",orientation:"portrait"}};html2pdf().from(t).set(a).save()}},formatDate:function(t){var a=new Date,e={"Y+":a.getFullYear(),"M+":a.getMonth()+1,"D+":a.getDate(),"h+":a.getHours(),"m+":a.getMinutes(),"s+":a.getSeconds(),W:a.getDay()},i=function(a){new RegExp("("+a+")").test(t)&&(t=t.replace(RegExp.$1,function(){if("W"===a){return["日","一","二","三","四","五","六"][e[a]]}return"Y+"===a||1===RegExp.$1.length?e[a]:("00"+e[a]).substr((""+e[a]).length)}))};for(var s in e)i(s);return t},initParam:function(){this.percent=0,this.logtext=[],this.postData={}},dataEvaClick:function(){var t=new Date;if(this.initParam(),""==this.modelChoice)return this.$message.warning("请选择硬件类型！",3),0;this.logflag=!0;var a=this.formatDate("YY-MM-DD hh:mm:ss")+" [info] [data analysis]:Start analyzing optimization data";this.logtext.push(a),console.log("this result:"+this.result[this.modelChoice]),this.percent=30,a=this.formatDate("YY-MM-DD hh:mm:ss")+" [info] [data analysis]：Model accuracy statistics",this.logtext.push(a);var e={"准确率":this.result[this.modelChoice].ACC},i=this.result[this.modelChoice].label;Object(l.f)("accLine",["准确率"],e,i),this.percent=40,a=this.formatDate("YY-MM-DD hh:mm:ss")+" [info] [data analysis]：Model size statistics",this.logtext.push(a);var s=this.result[this.modelChoice].time;Object(l.m)("timeBar",s,i,"","模型名称","运行时间(s)"),this.percent=50,a=this.formatDate("YY-MM-DD hh:mm:ss")+" [info] [data analysis]：Model run time statistics",this.logtext.push(a);var n=this.result[this.modelChoice].size;Object(l.m)("sizeBar",n,i,"","模型名称(s)","模型size(MB)"),this.percent=100,this.logflag=!1;var o=new Date;this.runTime=(o-t)/1e3,this.runTime<.5&&(this.runTime+=Math.random()/10,this.runTime=this.runTime.toFixed(3)),this.isShowPublish=!0}}},f={render:function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",[e("a-layout",[e("a-layout-header",[e("navmodule")],1),t._v(" "),e("a-layout-content",[e("func_introduce",{attrs:{funcDesText:t.funcDesText}}),t._v(" "),e("div",{staticClass:"paramCon"},[e("h2",{staticClass:"subTitle",staticStyle:{"margin-top":"-96px"}},[t._v("参数配置")]),t._v(" "),e("div",{staticClass:"funcParam"},[e("div",{staticClass:"paramTitle"},[e("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),e("h3",[t._v(t._s(t.funcDesText.name))]),t._v(" "),e("a-button",{staticClass:"DataEva",style:t.buttonBGColor,attrs:{disabled:t.disStatus},on:{click:t.dataEvaClick}},[e("a-icon",{attrs:{type:"security-scan"}}),t._v("\n                       评估\n                   ")],1)],1),t._v(" "),e("a-divider"),t._v(" "),e("div",{staticClass:"inputdiv"},[e("div",{staticClass:"mainParamNameNotop"},[t._v("请选择数据集")]),t._v(" "),e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"CIFAR10",defaultChecked:"",disabled:""}},[t._v("\n                           CIFAR10\n                       ")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[e("span",[t._v("CIFAR10数据集：")]),t._v("是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图 片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[t._v("图例：")]),t._v(" "),e("div",{staticClass:"demoData"},t._l(t.CIFAR10_imgs,function(t,a){return e("div",{key:a},[e("img",{attrs:{src:t.imgUrl}})])}),0)],1),t._v(" "),e("div",{staticClass:"datasetSelected"},[e("p",{staticClass:"mainParamNameNotop"},[t._v("请选择模型")]),t._v(" "),e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"VGG16",defaultChecked:"",disabled:""}},[t._v("\n                               VGG16\n                           ")])],1)]),t._v(" "),e("div",{staticClass:"modelSelected"},[e("p",{staticClass:"mainParamName"},[t._v("请选择硬件类型")]),t._v(" "),e("a-radio-group",{on:{change:t.onModelChoiceChange},model:{value:t.modelChoice,callback:function(a){t.modelChoice=a},expression:"modelChoice"}},[e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"CPU"}},[t._v("\n                                   Intel i7 7700 CPU\n                               ")]),t._v(" "),e("a-radio",{style:t.radioStyle,attrs:{value:"ARM"}},[t._v("\n                                   Jetson Nano(ARM 架构)\n                               ")]),t._v(" "),e("a-radio",{style:t.radioStyle,attrs:{value:"GPU"}},[t._v("\n                                   NVIDIA TITAN-V GPU\n                               ")])],1)])],1)])],1)]),t._v(" "),t.logflag?e("div",[e("showLog",{attrs:{percent:t.percent,logtext:t.logtext}})],1):t._e(),t._v(" "),e("resultDialog",{directives:[{name:"show",rawName:"v-show",value:t.isShowPublish,expression:"isShowPublish"}],attrs:{isShow:t.isShowPublish},on:{"on-close":t.closeDialog}},[e("div",{attrs:{slot:"header"},slot:"header"},[e("div",{staticClass:"dialog_title"},[e("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),e("h1",[t._v("硬件优化评估报告")])])]),t._v(" "),e("div",{staticClass:"dialog_publish_main",attrs:{slot:"main",id:"pdfDom"},slot:"main"},[e("div",{staticClass:"paramShow"},[e("a-row",[e("a-col",{attrs:{span:6}},[e("div",{staticClass:"paramContent"},[e("p",[e("span",{staticClass:"paramName"},[t._v("模型名称：")]),e("span",{staticClass:"paramValue"},[t._v("VGG16")])])])]),t._v(" "),e("a-col",{attrs:{span:6}},[e("div",{staticClass:"paramContent"},[e("p",[e("span",{staticClass:"paramName"},[t._v("数据集名称：")]),e("span",{staticClass:"paramValue"},[t._v("CIFAR10")])])])]),t._v(" "),e("a-col",{attrs:{span:6}},[e("div",{staticClass:"paramContent"},[e("p",[e("span",{staticClass:"paramName"},[t._v("硬件类型：")]),e("span",{staticClass:"paramValue"},[t._v(t._s(t.result[t.modelChoice].hard))])])])]),t._v(" "),e("a-col",{attrs:{span:6}},[e("div",{staticClass:"paramContent"},[e("p",[e("span",{staticClass:"paramName"},[t._v("运行时间：")]),e("span",{staticClass:"paramValue"},[t._v(t._s(t.runTime)+"s")])])])])],1)],1),t._v(" "),e("div",{staticClass:"reportContentCon"},[e("div",{staticClass:"result_div_notop"},[e("p",{staticClass:"main_top_echarts_con_title"},[t._v("模型优化前后效果展示")]),t._v(" "),e("div",{staticClass:"accLineChart"},[e("p",[t._v("模型优化前后准确率变化图")]),t._v(" "),e("div",{attrs:{id:"accLine"}}),t._v(" "),e("p",[t._v("模型优化前后运行时间变化图")]),t._v(" "),e("div",{attrs:{id:"timeBar"}}),t._v(" "),e("p",[t._v("模型优化前后模型参数变化图")]),t._v(" "),e("div",{attrs:{id:"sizeBar"}}),t._v(" "),e("div",{staticClass:"conclusion"},[e("p",{staticClass:"result_text"},[t._v("如图，Vanilla VGG16是未压缩前的VGG16模型，Lightweight VGG16 before是轻量化的VGG16模型，\n                                   Lightweight VGG16 after是在轻量化基础上参数量化后的VGG16模型，从图中可以看出模型优化前后的准确率变化非常小，\n                                   但是经过优化后的运行时间远小于原始模型")])])])])]),t._v(" "),e("a-button",{staticStyle:{width:"160px",height:"40px","margin-bottom":"30px","margin-top":"10px","font-size":"18px",color:"white","background-color":"rgb(46, 56, 245)","border-radius":"8px"},on:{click:function(a){return t.getPdf()}}},[e("a-icon",{attrs:{type:"upload"}}),t._v("导出报告内容\n               ")],1)],1)])],1),t._v(" "),e("a-layout-footer")],1)],1)},staticRenderFns:[]};var G=e("VU/8")(_,f,!1,function(t){e("JEGj")},"data-v-1f09b6ed",null);a.default=G.exports}});