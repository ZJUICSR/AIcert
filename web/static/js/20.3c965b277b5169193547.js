webpackJsonp([20],{"3d0V":function(t,e,a){t.exports=a.p+"static/img/sideIcon.879b18c.png"},MJz0:function(t,e){},jmja:function(t,e,a){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var s=a("gRE1"),i=a.n(s),o=a("mvHQ"),n=a.n(o),c=a("R45V"),d=a("83tA"),l=a("T9rv"),h=a("UI/F"),r=a("FaAE"),u=a("3d0V"),m=a.n(u),p=a("2b9l"),v=a.n(p),C={name:"inject",components:{navmodule:c.a,func_introduce:d.a,showLog:l.a,resultDialog:h.a,drawside:r.u,drawTtest:r.n,drawALLPA:r.c},data:function(){return{htmlTitle:"侧信道分析",radioStyle:{display:"block",lineHeight:"30px"},buttonBGColor:{background:"#0B55F4",color:"#FFFFFF"},dataChoice:"",dataChoiceList:{trace1:{name:"Trace1",value:"trace1",id:"(022.108.112.-9)elmotracegaus_cpa_-9.trs",disindex:!1},trace2:{name:"Trace2",value:"trace2",id:"(022.112.116.-9)elmotracegaus_hpa_-9.trs",disindex:!1},trace3:{name:"Trace3",value:"trace3",id:"(2122.108.112.47)elmotracegaus_cpa_47.trs",disindex:!1},trace4:{name:"Trace4",value:"trace4",id:"(2122.108.112.47)elmotracegaus_hpa_47.trs",disindex:!1},trace5:{name:"Trace5",value:"trace5",id:"(7522.108.112.2)elmotracegaus_cpa_2.trs",disindex:!1},trace6:{name:"Trace6",value:"trace6",id:"(7522.112.116.2)elmotracegaus_hpa_2.trs",disindex:!1}},methodChoice:"",methodChoiceList:{CPA:{name:"CPA",value:"cpa",disindex:!1,des:"CPA方法描述描述"},DPA:{name:"DPA",value:"dpa",disindex:!1,des:"DPA方法描述"},HPA:{name:"HPA",value:"hpa",disindex:!1,des:"HPA方法描述"},"T-test":{name:"T-test",value:"ttest",disindex:!1,des:"T-test方法描述"},"X2-test":{name:"X²-Test",value:"x2test",disindex:!1,des:"X²-Test方法描述"}},disStatus:!1,logflag:!1,percent:10,logtext:[],funcDesText:{name:"侧信道分析",imgpath:m.a,bgimg:v.a,destext:"使用模型运行时的功耗/电磁数据，评估硬件性能",backinfo:"采集目标模型在目标平台上运行产生的功耗/电磁信息，形成曲线数据。输入功耗/电磁曲线数据文件，分别进行不同的侧信道攻击",highlight:["支持对目标模型使用T-test、X^2-test分析进行泄漏检测，根据阈值判断是否存在泄漏","对目标模型进行DPA分析及HPA分析，并得到相关性系数，输出相关性曲线图展示攻击结果及不同能耗分析攻击方法的特点","内置不同trs文件，支持上传用户自己定义数据"]},isShowPublish:!1,result:{text1:"",text2:"",text3:""},res_tmp:{},tid:"",stidlist:{},clk:null,logclk:null}},watch:{isShowPublish:{immediate:!0,handler:function(t){t?this.noScroll():this.canScroll()}}},created:function(){document.title="侧信道分析"},methods:{onDataChoiceChange:function(t){this.dataChoice=t.target.value},onMethodChoiceChange:function(t){console.log("radio checked",t.target.value)},methodFilter:function(t){["trace1"].includes(t)?(this.methodChoiceList.CPA.disindex=!1,this.methodChoiceList.DPA.disindex=!1,this.methodChoiceList.HPA.disindex=!0,this.methodChoiceList["T-test"].disindex=!1,this.methodChoiceList["X2-test"].disindex=!1):["trace5"].includes(t)?(this.methodChoiceList.CPA.disindex=!1,this.methodChoiceList.DPA.disindex=!0,this.methodChoiceList.HPA.disindex=!0,this.methodChoiceList["T-test"].disindex=!1,this.methodChoiceList["X2-test"].disindex=!1):["trace2","trace4","trace6"].includes(t)?(this.methodChoiceList.CPA.disindex=!0,this.methodChoiceList.DPA.disindex=!0,this.methodChoiceList.HPA.disindex=!1,this.methodChoiceList["T-test"].disindex=!0,this.methodChoiceList["X2-test"].disindex=!0):"trace3"==t&&(this.methodChoiceList.CPA.disindex=!1,this.methodChoiceList.DPA.disindex=!1,this.methodChoiceList.HPA.disindex=!0,this.methodChoiceList["T-test"].disindex=!1,this.methodChoiceList["X2-test"].disindex=!0)},closeDialog:function(){this.isShowPublish=!1},exportResult:function(){if(confirm("您确认下载该pdf文件吗？")){document.body.scrollTop=document.documentElement.scrollTop=0;var t=document.getElementById("download_page"),e={margin:[10,20,10,20],filename:this.tid+".pdf",image:{type:"jpeg",quality:1},html2canvas:{scale:5},jsPDF:{unit:"mm",format:"a4",orientation:"portrait"}};html2pdf().from(t).set(e).save()}},resultPro:function(t){if("ttest"in(t=t.result.side))return Object(r.n)("score",t.ttest.X,t.ttest.Y[0]),void(Math.max.apply(Math,t.ttest.Y[0])>4.5||Math.min.apply(Math,t.ttest.Y[0])<-4.5?(this.result.text1="",this.result.text2="有"):(this.result.text1="不",this.result.text2="无"));if("x2test"in t)return Object(r.n)("score",t.x2test.X,t.x2test.Y[0]),void(Math.max.apply(Math,t.x2test.Y[0])>4.5||Math.min.apply(Math,t.x2test.Y[0])<-4.5?(this.result.text1="",this.result.text2="有"):(this.result.text1="不",this.result.text2="无"));if("HPA"==this.methodChoice){for(var e=[],a=t.hpa.X,s=[],i=0;i<t.hpa.Y.length;i++){e.push("错误值");var o={symbol:"none"};o.name=e[i],o.type="line",o.data=t.hpa.Y[i],s.push(o)}s.push({symbol:"none",name:"错误值",type:"line",data:t.hpa.false}),e.push("错误值"),s.push({symbol:"none",name:"正确值",type:"line",data:t.hpa.true}),e.push("正确值"),Object(r.c)("score",e,a,s),Math.max(Math.max.apply(Math,t.hpa.Y),Math.max.apply(Math,t.hpa.false))<Math.min.apply(Math,t.hpa.true)?this.result.text3="正确值相关性与错误值相关性差值大且稳定，说明侧信道攻击成功":this.result.text3="正确值相关性与错误值相关性差值小或不稳定，说明侧信道攻击失败"}if("SPA"==this.methodChoice&&Object(r.c)("score",t.spa.X,t.spa.Y,t.spa.false,t.spa.true),"DPA"==this.methodChoice){var n=[],c=t.dpa.X,d=[];for(i=0;i<t.dpa.Y.length;i++){n.push("错误值");var l={symbol:"none"};l.name=n[i],l.type="line",l.data=t.dpa.Y[i],d.push(l)}d.push({symbol:"none",name:"错误值",type:"line",data:t.dpa.false}),n.push("错误值"),d.push({symbol:"none",name:"正确值",type:"line",data:t.dpa.true}),n.push("正确值"),Object(r.c)("score",n,c,d)}if("CPA"==this.methodChoice){var h=[],u=t.cpa.X,m=[];for(i=0;i<t.cpa.Y.length;i++){h.push("错误值");var p={symbol:"none"};p.name=h[i],p.type="line",p.data=t.cpa.Y[i],m.push(p)}m.push({symbol:"none",name:"错误值",type:"line",data:t.cpa.false}),h.push("错误值"),m.push({symbol:"none",name:"正确值",type:"line",data:t.cpa.true}),h.push("正确值"),Object(r.c)("score",h,u,m)}},getData:function(){var t=this;t.$axios.get("/output/Resultdata",{params:{Taskid:t.tid}}).then(function(e){console.log("dataget:",e),t.res_tmp=e})},getLog:function(){var t=this;t.percent<99&&(t.percent+=1),t.$axios.get("/Task/QueryLog",{params:{Taskid:t.tid}}).then(function(e){if("{}"==n()(t.stidlist))t.logtext=[i()(e.data.Log).slice(-1)[0]];else for(var a in t.logtext=[],t.stidlist)t.logtext.push(e.data.Log[t.stidlist[a]])})},stopTimer:function(){1==this.res_tmp.data.stop&&this.tid==this.res_tmp.data.result.tid&&(this.percent=100,window.clearInterval(this.clk),window.clearInterval(this.logclk),this.isShowPublish=!0,this.resultPro(this.res_tmp.data))},update:function(){this.getData();try{this.stopTimer()}catch(t){}},dataEvaClick:function(){if(window.clearInterval(this.clk),window.clearInterval(this.logclk),""==this.dataChoice|""==this.methodChoice)return this.$message.warning("请选择能耗文件与分析方法！",3),0;var t=this;t.res_tmp={},t.$axios.post("/Task/CreateTask",{AttackAndDefenseTask:0}).then(function(e){t.tid=e.data.Taskid;var a={trs_file:t.dataChoiceList[t.dataChoice].id,methods:t.methodChoiceList[t.methodChoice].value,tid:t.tid};t.$axios.post("/SideAnalysis",a).then(function(e){t.logflag=!0,t.stidlist={SideAnalysis:e.data.stid},t.logclk=window.setInterval(t.getLog,300),t.clk=window.setInterval(t.update,300)}).catch(function(t){console.log(t)})}).catch(function(t){console.log(t)})}}},_={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",[a("a-layout",[a("a-layout-header",[a("navmodule")],1),t._v(" "),a("a-layout-content",[a("func_introduce",{attrs:{funcDesText:t.funcDesText}}),t._v(" "),a("div",{staticClass:"paramCon"},[a("h2",{staticClass:"subTitle",staticStyle:{"margin-top":"-96px"}},[t._v("参数配置")]),t._v(" "),a("div",{staticClass:"funcParam"},[a("div",{staticClass:"paramTitle"},[a("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),a("h3",[t._v(t._s(t.funcDesText.name))]),t._v(" "),a("a-button",{staticClass:"DataEva",style:t.buttonBGColor,attrs:{disabled:t.disStatus},on:{click:t.dataEvaClick}},[a("a-icon",{attrs:{type:"security-scan"}}),t._v("\n                       评估\n                   ")],1)],1),t._v(" "),a("a-divider"),t._v(" "),a("div",{staticClass:"inputdiv"},[a("div",{staticClass:"datasetSelected"},[a("div",{staticClass:"selectWithupload"},[a("p",{staticClass:"mainParamNameNotop"},[t._v("请选择能耗文件")]),t._v(" "),a("a-button",{staticClass:"uploadDatasetBtn",attrs:{name:"table"}},[a("a-icon",{staticStyle:{color:"#0B55F4"},attrs:{type:"upload"}}),t._v("上传数据\n                           ")],1)],1),t._v(" "),a("a-radio-group",{on:{change:t.onDataChoiceChange},model:{value:t.dataChoice,callback:function(e){t.dataChoice=e},expression:"dataChoice"}},t._l(t.dataChoiceList,function(e,s){return a("div",{key:s,staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:s,disabled:t.methodFilter(t.dataChoice)}},[t._v(t._s(e.name)+".trs")])],1)}),0)],1),t._v(" "),a("div",{staticClass:"modelSelected"},[a("p",{staticClass:"mainParamName"},[t._v("请选择能耗分析方法")]),t._v(" "),a("a-radio-group",{on:{change:t.onMethodChoiceChange},model:{value:t.methodChoice,callback:function(e){t.methodChoice=e},expression:"methodChoice"}},t._l(t.methodChoiceList,function(e,s){return a("div",{key:s,staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:s,disabled:e.disindex}},[t._v(" "+t._s(e.name))])],1)}),0)],1)])],1)]),t._v(" "),t.logflag?a("div",[a("showLog",{attrs:{percent:t.percent,logtext:t.logtext}})],1):t._e(),t._v(" "),a("resultDialog",{directives:[{name:"show",rawName:"v-show",value:t.isShowPublish,expression:"isShowPublish"}],ref:"report_pdf",attrs:{isShow:t.isShowPublish},on:{"on-close":t.closeDialog}},[a("div",{attrs:{slot:"header"},slot:"header"},[a("div",{staticClass:"dialog_title"},[a("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),a("h1",[t._v("侧信道分析结果报告")])])]),t._v(" "),a("div",{staticClass:"dialog_publish_main",attrs:{slot:"main",id:"pdfDom"},slot:"main"},[a("div",{staticClass:"paramShow"},[a("a-row",[a("a-col",{attrs:{span:6}},[a("div",{staticClass:"paramContent"},[a("p",[a("span",{staticClass:"paramName"},[t._v("能耗文件：")]),a("span",{staticClass:"paramValue"},[t._v(t._s(t.dataChoice))])])])]),t._v(" "),a("a-col",{attrs:{span:6}},[a("div",{staticClass:"paramContent"},[a("p",[a("span",{staticClass:"paramName"},[t._v("能耗分析方法：")]),a("span",{staticClass:"paramValue"},[t._v(t._s(t.methodChoice))])])])]),t._v(" "),a("a-col",{attrs:{span:6}},[a("div",{staticClass:"paramContent"})]),t._v(" "),a("a-col",{attrs:{span:6}},[a("div",{staticClass:"paramContent"})])],1)],1),t._v(" "),a("div",{staticClass:"reportContentCon"},[a("div",{staticClass:"result_div_notop"},[a("p",{staticClass:"main_top_echarts_con_title"},[t._v("能耗分析结果")]),t._v(" "),a("div",{staticClass:"accLineChart"},[a("div",{staticClass:"g_score_content",attrs:{id:"score"}}),t._v(" "),a("div",{staticClass:"conclusion"},[a("p",{directives:[{name:"show",rawName:"v-show",value:"HPA"!=t.methodChoice,expression:"methodChoice!='HPA'"}],staticClass:"result_text"},[t._v(t._s(t.result.text1)+"存在阈值>4.5的值，说明"+t._s(t.result.text2)+"泄露")]),t._v(" "),a("p",{directives:[{name:"show",rawName:"v-show",value:"HPA"==t.methodChoice,expression:"methodChoice=='HPA'"}],staticClass:"result_text"},[t._v(t._s(t.result.text3))])])])]),t._v(" "),a("div",{directives:[{name:"show",rawName:"v-show",value:"SPA"==t.methodChoice,expression:"methodChoice=='SPA'"}],staticClass:"result_div_notop"},[a("p",{staticClass:"main_top_echarts_con_title"},[t._v("原模型示意图")]),t._v(" "),a("div",{staticClass:"g_score_content",attrs:{id:"ori_network"}}),t._v(" "),a("p",{staticClass:"main_top_echarts_con_title"},[t._v("模型恢复示意图")]),t._v(" "),a("div",{staticClass:"g_score_content",attrs:{id:"res_network"}}),t._v(" "),a("div",{staticClass:"conclusion"},[a("p",{staticClass:"result_text"},[t._v("根据SPA攻击的相关性系数曲线特征，可以直接区分出模型各层结构。")])])])]),t._v(" "),a("a-button",{staticStyle:{width:"160px",height:"50px","margin-bottom":"30px","margin-top":"10px","font-size":"18px",color:"white","background-color":"rgb(46, 56, 245)","border-radius":"8px"},on:{click:function(e){return t.getPdf()}}},[a("a-icon",{attrs:{type:"upload"}}),t._v("导出报告内容\n               ")],1)],1)])],1),t._v(" "),a("a-layout-footer")],1)],1)},staticRenderFns:[]};var x=a("VU/8")(C,_,!1,function(t){a("MJz0")},"data-v-fcb60aee",null);e.default=x.exports}});