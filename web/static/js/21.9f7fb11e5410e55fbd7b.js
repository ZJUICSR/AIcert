webpackJsonp([21],{"/jQh":function(t,a){},"3rtc":function(t,a,s){t.exports=s.p+"static/img/abnormal_table_demo.4ec3162.png"},rffR:function(t,a,s){"use strict";Object.defineProperty(a,"__esModule",{value:!0});var e=s("gRE1"),i=s.n(e),l=s("mvHQ"),o=s.n(l),n=s("R45V"),r=s("83tA"),_=s("T9rv"),c=s("UI/F"),d=s("7GXC"),u=s.n(d),p=s("2b9l"),m=s.n(p),v={template:'\n        <svg t="1680138013828" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4354" width="128" height="128"><path d="M534.869333 490.496a1403.306667 1403.306667 0 0 0 50.858667-25.813333c16.042667-8.618667 29.013333-15.061333 38.570667-19.029334 9.557333-3.925333 17.066667-6.058667 22.869333-6.058666 9.557333 0 17.749333 3.2 24.917333 10.026666 6.826667 6.826667 10.581333 15.061333 10.581334 25.088 0 5.76-1.706667 11.818667-5.12 17.92-3.413333 6.101333-7.168 10.069333-10.922667 11.861334-35.157333 14.677333-74.410667 25.429333-116.736 31.872 7.850667 7.168 17.066667 17.237333 28.330667 29.781333 11.264 12.544 17.066667 18.986667 17.749333 20.053333 4.096 6.101333 9.898667 13.653333 17.408 22.613334 7.509333 8.96 12.629333 15.786667 15.36 20.778666 2.730667 5.034667 4.437333 11.093333 4.437333 18.304a33.706667 33.706667 0 0 1-9.898666 24.021334 33.834667 33.834667 0 0 1-25.6 10.410666c-10.24 0-22.186667-8.618667-35.157334-25.472-12.970667-16.512-30.037333-46.933333-50.517333-91.050666-20.821333 39.424-34.816 65.962667-41.642667 78.506666-7.168 12.544-13.994667 22.186667-20.48 28.672a30.976 30.976 0 0 1-22.528 9.685334 32.256 32.256 0 0 1-25.258666-11.093334 35.413333 35.413333 0 0 1-9.898667-23.68c0-7.893333 1.365333-13.653333 4.096-17.578666 25.258667-35.84 51.541333-67.413333 78.848-93.568a756.650667 756.650667 0 0 1-61.44-12.544 383.061333 383.061333 0 0 1-57.685333-20.48c-3.413333-1.749333-6.485333-5.717333-9.557334-11.818667a30.208 30.208 0 0 1-5.12-16.853333 32.426667 32.426667 0 0 1 10.581334-25.088 33.152 33.152 0 0 1 24.234666-10.026667c6.485333 0 14.677333 2.133333 24.576 6.101333 9.898667 4.266667 22.186667 10.026667 37.546667 18.261334 15.36 7.893333 32.426667 16.853333 51.882667 26.538666-3.413333-18.261333-6.485333-39.082667-8.874667-62.378666-2.389333-23.296-3.413333-39.424-3.413333-48.042667 0-10.752 3.072-19.712 9.557333-27.264A30.677333 30.677333 0 0 1 512.341333 341.333333c9.898667 0 18.090667 3.925333 24.576 11.477334 6.485333 7.893333 9.557333 17.92 9.557334 30.464 0 3.584-0.682667 10.410667-1.365334 20.48-0.682667 10.368-2.389333 22.570667-4.096 36.906666-2.048 14.677333-4.096 31.146667-6.144 49.834667z" fill="#FF3838" p-id="4355"></path></svg>\n        '},h={template:'\n        <a-icon :component="selectSvg" />\n    ',data:function(){return{selectSvg:v}}},g={name:"dataClean",components:{navmodule:n.a,func_introduce:r.a,showLog:_.a,resultDialog:c.a,selectIcon:h},data:function(){return{htmlTitle:"异常数据检测",radioStyle:{display:"block",lineHeight:"30px",width:"100%"},datasetChoice:"table",upload_flag:0,upload_flag_table:0,upload_flag_img:0,upload_path:"",upload_path_table:"",upload_path_img:"",post_file:{},upload_success:"",upload_fail:"",MNIST_imgs:[{imgUrl:s("wlDy"),name:"mnist0"},{imgUrl:s("KdR5"),name:"mnist1"},{imgUrl:s("2LXz"),name:"mnist2"},{imgUrl:s("3rtV"),name:"mnist3"},{imgUrl:s("Ppgw"),name:"mnist4"},{imgUrl:s("P9ea"),name:"mnist5"},{imgUrl:s("ec3M"),name:"mnist6"},{imgUrl:s("wAAx"),name:"mnist7"},{imgUrl:s("30PV"),name:"mnist8"},{imgUrl:s("v3v+"),name:"mnist9"}],CIFAR10_imgs:[{imgUrl:s("HiaR"),name:"mnist0"},{imgUrl:s("DJt5"),name:"mnist1"},{imgUrl:s("S99w"),name:"mnist2"},{imgUrl:s("J598"),name:"mnist3"},{imgUrl:s("/pRs"),name:"mnist4"},{imgUrl:s("YuLH"),name:"mnist5"},{imgUrl:s("Nvyw"),name:"mnist6"},{imgUrl:s("lB35"),name:"mnist7"},{imgUrl:s("dKp5"),name:"mnist8"},{imgUrl:s("NgyD"),name:"mnist9"}],buttonBGColor:{background:"#0B55F4",color:"#FFFFFF"},disStatus:!1,logflag:!1,percent:10,logtext:[],funcDesText:{name:"异常数据检测",imgpath:u.a,bgimg:m.a,destext:"自动化数据清洗与量化评估",backinfo:"通过分析AI系统常见异常数据来源及类型，基于置信学习等多种技术修复异常数据，并丢弃不可修复的异常数据，从而实现自动化的数据集清洗及量化评估，生成异常检测效果分析报告及清洁的数据集",highlight:["检测AI系统多模态异常数据，数据类型≥3种","采用离群值检测、编码检测与标点标准化处理、置信学习方法修复","支持自定义数据集上传与数据集异常检测，能够有效提升数据可用性和安全性"]},isShowPublish:!1,result:{},res_tmp:{},tid:"",stidlist:"",clk:null,logclk:null}},watch:{isShowPublish:{immediate:!0,handler:function(t){t?this.noScroll():this.canScroll()}}},created:function(){document.title="异常数据检测"},methods:{closeDialog:function(){this.isShowPublish=!1},onDatasetChoiceChange:function(t){console.log("radio checked",t.target.value)},UpFile:function(t){var a=this;a.noScroll();var s=t.target.files[0],e=new FormData;e.append("file",s),e.append("type",t.target.name),a.post_file=e;a.$axios.post("/Task/UploadData",a.post_file,{headers:{"Content-Type":"multipart/form-data"}}).then(function(t){a.upload_flag=1,a.upload_path=t.data.save_dir,a.upload_path_img=a.upload_path,a.upload_path_table=a.upload_path,a.upload_success="文件上传成功，存放位置为"+a.upload_path}).catch(function(t){console.log(t)})},exportResult:function(){if(confirm("您确认下载该pdf文件吗？")){var t=document.getElementById("download_page"),a={margin:[10,20,10,20],filename:this.tid+".pdf",image:{type:"jpeg",quality:1},html2canvas:{scale:5},jsPDF:{unit:"mm",format:"a4",orientation:"portrait"}};html2pdf().from(t).set(a).save()}},resultPro:function(t){if(this.result={},this.result.score=parseInt(100*t.DataClean.fix_rate),this.result.dataset_or_format=this.datasetChoice,"table"==this.result.dataset_or_format){var a=t.DataClean.result_origin.split("output"),s=t.DataClean.result_clean.split("output");this.result.table_origin="static/output"+a[1],this.result.table_fix="static/output"+s[1]}else if("MNIST"==this.result.dataset_or_format|"CIFAR10"==this.result.dataset_or_format){this.result.num_images=t.DataClean.num_images,this.result.num_detect=t.DataClean.num_detect;var e=t.DataClean.result.split("output");this.result.demo_img="static/output"+e[1]}else this.result.text_origin=t.DataClean.before,this.result.text_clean=t.DataClean.after},getData:function(){var t=this;t.$axios.get("/output/Resultdata",{params:{Taskid:t.tid}}).then(function(a){console.log("dataget:",a),t.res_tmp=a})},getLog:function(){var t=this;t.percent<99&&(t.percent+=1),t.$axios.get("/Task/QueryLog",{params:{Taskid:t.tid}}).then(function(a){if("{}"==o()(t.stidlist))t.logtext=[i()(a.data.Log).slice(-1)[0]];else for(var s in t.logtext=[],t.stidlist)t.logtext.push(a.data.Log[t.stidlist[s]])})},stopTimer:function(){1==this.res_tmp.data.stop&&this.tid==this.res_tmp.data.result.tid&&(this.percent=100,this.logflag=!1,window.clearInterval(this.clk),window.clearInterval(this.logclk),this.isShowPublish=!0,this.resultPro(this.res_tmp.data.result))},update:function(){this.getData();try{this.stopTimer()}catch(t){}},dataUploadButton:function(t){switch(this.upload_flag=1,t.target.name){case"table":this.upload_flag_table=1;break;case"img":this.upload_flag_img=1}this.noScroll()},CancelUpload:function(){this.upload_flag=0,this.upload_path="",this.upload_flag_img=0,this.upload_path_img="",this.upload_flag_table=0,this.upload_path_table="",this.canScroll()},ConfirmUpload:function(){this.upload_flag=1,this.upload_flag_img=0,this.upload_path_img="",this.upload_flag_table=0,this.upload_path_table="",this.canScroll()},dataEvaClick:function(){var t=this;t.res_tmp={},t.$axios.post("/Task/CreateTask",{AttackAndDefenseTask:0}).then(function(a){t.tid=a.data.Taskid;var s={dataset:t.datasetChoice,upload_flag:t.upload_flag,upload_path:t.upload_path,tid:t.tid};t.$axios.post("/DataClean/DataCleanParamSet",s).then(function(a){t.logflag=!0,t.stidlist={dataClean:a.data.stid},t.logclk=window.setInterval(t.getLog,300),t.clk=window.setInterval(t.update,300)}).catch(function(t){console.log(t)})}).catch(function(t){console.log(t)})}}},f={render:function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",[e("a-layout",[e("a-layout-header",[e("navmodule")],1),t._v(" "),e("a-layout-content",[e("func_introduce",{attrs:{funcDesText:t.funcDesText}}),t._v(" "),e("div",{staticClass:"paramCon"},[e("h2",{staticClass:"subTitle",staticStyle:{"margin-top":"-96px"}},[t._v("参数配置")]),t._v(" "),e("div",{staticClass:"funcParam"},[e("div",{staticClass:"paramTitle"},[e("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),e("h3",[t._v(t._s(t.funcDesText.name))]),t._v(" "),e("a-button",{staticClass:"DataEva",style:t.buttonBGColor,attrs:{disabled:t.disStatus},on:{click:t.dataEvaClick}},[e("a-icon",{attrs:{type:"security-scan"}}),t._v("\n                       评估\n                   ")],1)],1),t._v(" "),e("a-divider"),t._v(" "),e("div",{staticClass:"inputdiv"},[e("div",{staticClass:"datasetSelected"},[e("div",{staticClass:"SelectWithUpload"},[e("p",{staticClass:"mainParamName"},[t._v("低维数据")]),t._v(" "),e("a-button",{staticClass:"uploadDatasetBtn",attrs:{name:"table"},on:{click:t.dataUploadButton}},[e("a-icon",{staticStyle:{color:"#0B55F4"},attrs:{type:"upload"}}),t._v("上传数据")],1)],1),t._v(" "),e("a-radio-group",{on:{change:t.onDatasetChoiceChange},model:{value:t.datasetChoice,callback:function(a){t.datasetChoice=a},expression:"datasetChoice"}},[e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"table"}},[t._v("\n                                   abnormal_table.npz\n                               ")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[e("span",[t._v("低维数据：")]),t._v("指数字形式的、特征维数较少的数据。检测离群点或异常值是数据挖掘的核心问题之一，数据的爆发式增长要求我们能够及时筛选出其中存在问题的数据并予以剔除。")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[t._v("示例：随机生成正态化的二维样本，模拟存在离群异常数据的低维数据，并进行检测。")]),t._v(" "),e("div",{staticClass:"demoData"},[e("img",{staticClass:"onedemo",attrs:{src:s("3rtc")}})])],1)]),t._v(" "),e("p",{staticClass:"mainParamName"},[t._v("文本数据")]),t._v(" "),e("a-radio-group",{on:{change:t.onDatasetChoiceChange},model:{value:t.datasetChoice,callback:function(a){t.datasetChoice=a},expression:"datasetChoice"}},[e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"txt_encode"}},[t._v("\n                                   THUCNews格式异常文本\n                               ")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[e("span",[t._v("THUCNews：")]),t._v("由清华大学自然语言处理实验室推出的根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），并划分出14个分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[t._v("示例：检测文本中的错误编码，并进行清洗")]),t._v(" "),e("div",{staticClass:"demoData"},[e("p",[t._v("\n                                       Ħ��������GPON��FTTH�б�EPON��������\n                                   ������ �ߣ�³����\n                                   ����2009�꣬�ڹ��ڹ��ͭ�˵Ļ��������£�Ħ������Я���ڹ����г����Ѿ���÷ḻ��Ӫ�����GPON�����������\n                                   ")])])],1),t._v(" "),e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"txt_format"}},[t._v("\n                                   THUCNews待标点清洗文本\n                               ")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[e("span",[t._v("THUCNews：")]),t._v("由清华大学自然语言处理实验室推出的根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），并划分出14个分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[t._v("示例：将文本数据的标点符号进行清洗，便于后续处理")]),t._v(" "),e("div",{staticClass:"demoData"},[e("p",[t._v("\n                                       昨日上海天然橡胶期货价格再度大幅上扬，在收盘前1小时，大量场外资金涌入，主力1003合约强劲飙升很快升穿21000\n                                       元/吨整数关口，终盘报收于21,400元/吨，上涨2.27%，较前一日结算价上涨475元/吨，成交量为736,816手，持仓量为225,046 手。当日整体市场增仓3.4万余手。\n\n                                   ")])])],1)]),t._v(" "),e("div",{staticClass:"SelectWithUpload"},[e("p",{staticClass:"mainParamName"},[t._v("图像数据")]),t._v(" "),e("a-button",{staticClass:"uploadDatasetBtn",attrs:{name:"img"},on:{click:t.dataUploadButton}},[e("a-icon",{staticStyle:{color:"#0B55F4"},attrs:{type:"upload"}}),t._v("上传数据")],1)],1),t._v(" "),e("div",{directives:[{name:"show",rawName:"v-show",value:t.upload_flag_img,expression:"upload_flag_img"}],staticClass:"UploadPage"},[e("div",{staticClass:"UploadPagetitle"},[e("a-icon",{staticStyle:{color:"#0B55F4","font-size":"16px"},attrs:{type:"exclamation-circle",theme:"filled"}}),t._v(" "),e("p",{staticClass:"uploadtiletext"},[t._v("请选择上传的数据类型")])],1),t._v(" "),e("div",{staticClass:"UploadPageButton"},[e("div",{staticClass:"upload_type"},[e("input",{staticStyle:{visibility:"hidden"},attrs:{type:"file",id:"uFile",name:"MNIST",accept:".gz"},on:{change:function(a){return t.UpFile(a)}}}),t._v(" "),e("label",{staticClass:"UploadDataTyleButton",attrs:{for:"uFile"}},[e("a-icon",{attrs:{type:"plus"}}),t._v(" "),e("p",{staticClass:"buttontitle"},[t._v("MNIST")]),e("p",{staticClass:"buttontext"},[t._v("需处理成MNIST数据集.gz格式")])],1)]),t._v(" "),e("div",{staticClass:"upload_type"},[e("input",{staticStyle:{visibility:"hidden"},attrs:{type:"file",id:"uFile_",name:"CIFAR10",accept:".gz"},on:{change:function(a){return t.UpFile(a)}}}),t._v(" "),e("label",{staticClass:"UploadDataTyleButton",attrs:{for:"uFile_"}},[e("a-icon",{attrs:{type:"plus"}}),t._v(" "),e("p",{staticClass:"buttontitle"},[t._v("CIFAR10")]),e("p",{staticClass:"buttontext"},[t._v("需处理成CIFAR10数据集.gz格式")])],1)])]),t._v(" "),e("div",{directives:[{name:"show",rawName:"v-show",value:t.upload_flag_img&&""!=t.upload_path_img,expression:"upload_flag_img && upload_path_img!=''"}],staticStyle:{margin:"1% 7%"}},[t.upload_flag_img?e("p",{staticClass:"buttontitle"},[t._v(t._s(t.upload_success))]):t._e()]),t._v(" "),e("div",{staticClass:"UploadPageCC"},[e("button",{staticClass:"CancelButton",on:{click:t.CancelUpload}},[t._v("取消")]),t._v(" "),e("button",{staticClass:"ConfirmButton",on:{click:t.ConfirmUpload}},[t._v("确认")])])]),t._v(" "),e("div",{directives:[{name:"show",rawName:"v-show",value:t.upload_flag_table,expression:"upload_flag_table"}],staticClass:"UploadPage"},[e("div",{staticClass:"UploadPagetitle"},[e("a-icon",{staticStyle:{color:"#0B55F4","font-size":"16px"},attrs:{type:"exclamation-circle",theme:"filled"}}),t._v(" "),e("p",{staticClass:"uploadtiletext"},[t._v("请选择上传的数据类型")])],1),t._v(" "),e("div",{staticClass:"UploadPageButton"},[e("div",{staticClass:"upload_type"},[e("input",{staticStyle:{visibility:"hidden"},attrs:{type:"file",id:"uFilet",name:"Table",accept:".npz"},on:{change:function(a){return t.UpFile(a)}}}),t._v(" "),e("label",{staticClass:"UploadDataTyleButton",attrs:{for:"uFilet"}},[e("a-icon",{attrs:{type:"plus"}}),t._v(" "),e("p",{staticClass:"buttontitle"},[t._v("表格数据")]),e("p",{staticClass:"buttontext"},[t._v("需处理成表格数据.npz格式")])],1)])]),t._v(" "),e("div",{directives:[{name:"show",rawName:"v-show",value:t.upload_flag_table&&""!=t.upload_path_table,expression:"upload_flag_table && upload_path_table!=''"}],staticStyle:{margin:"1% 7%"}},[t.upload_flag_table?e("p",{staticClass:"buttontitle"},[t._v(t._s(t.upload_success))]):t._e()]),t._v(" "),e("div",{staticClass:"UploadPageCC"},[e("button",{staticClass:"CancelButton",on:{click:t.CancelUpload}},[t._v("取消")]),t._v(" "),e("button",{staticClass:"ConfirmButton",on:{click:t.ConfirmUpload}},[t._v("确认")])])]),t._v(" "),e("a-radio-group",{on:{change:t.onDatasetChoiceChange},model:{value:t.datasetChoice,callback:function(a){t.datasetChoice=a},expression:"datasetChoice"}},[e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"CIFAR10"}},[t._v("\n                                   CIFAR10\n                               ")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[e("span",[t._v("CIFAR10数据集：")]),t._v("是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图 片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[t._v("图例：")]),t._v(" "),e("div",{staticClass:"demoData"},t._l(t.CIFAR10_imgs,function(t,a){return e("div",{key:a},[e("img",{attrs:{src:t.imgUrl}})])}),0)],1),t._v(" "),e("div",{staticClass:"matchedDes"},[e("a-radio",{style:t.radioStyle,attrs:{value:"MNIST"}},[t._v("\n                                   MNIST\n                               ")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[e("span",[t._v("MNIST数据集：")]),t._v("是一个手写体数字的图片数据集，该数据集来由美国国家标准与技术研究所（National Institute of Standards and Technology, NIST）发起整理，一共统计了来自250个不同的人手写数字图片，其中50%是高中生，50%来自人口普查局的工作人员。该数据集的收集目的是希望通过算法，实现对手写数字的识别。")]),t._v(" "),e("p",{staticClass:"matchedMethodText"},[t._v("图例：")]),t._v(" "),e("div",{staticClass:"demoData"},t._l(t.MNIST_imgs,function(t,a){return e("div",{key:a},[e("img",{attrs:{src:t.imgUrl}})])}),0)],1)])],1)])],1)]),t._v(" "),t.logflag?e("div",[e("showLog",{attrs:{percent:t.percent,logtext:t.logtext}})],1):t._e(),t._v(" "),e("resultDialog",{directives:[{name:"show",rawName:"v-show",value:t.isShowPublish,expression:"isShowPublish"}],ref:"report_pdf",attrs:{isShow:t.isShowPublish},on:{"on-close":t.closeDialog}},[e("div",{attrs:{slot:"header"},slot:"header"},[e("div",{staticClass:"dialog_title"},[e("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),e("h1",[t._v("异常数据检测结果报告")])])]),t._v(" "),e("div",{staticClass:"dialog_publish_main",attrs:{slot:"main",id:"pdfDom"},slot:"main"},[e("div",{staticClass:"result_div"},[e("div",{staticClass:"conclusion_info"},["table"==t.result.dataset_or_format?e("p",{staticClass:"result_annotation"},[t._v("检测类型：低维数据")]):t._e(),t._v(" "),"txt_encode"==t.result.dataset_or_format|"txt_format"==t.result.dataset_or_format?e("p",{staticClass:"result_annotation"},[t._v("检测类型：文本数据")]):t._e(),t._v(" "),"MNIST"==t.result.dataset_or_format|"CIFAR10"==t.result.dataset_or_format?e("p",{staticClass:"result_annotation"},[t._v("检测类型：图像数据")]):t._e(),t._v(" "),"txt_encode"==t.result.dataset_or_format|"txt_format"==t.result.dataset_or_format?e("p",{staticClass:"result_annotation"},[t._v("文本格式："+t._s(t.result.dataset_or_format))]):e("p",{staticClass:"result_annotation"},[t._v("数据集："+t._s(t.result.dataset_or_format))])]),t._v(" "),e("div",{staticClass:"g_score_content"},[e("div",{staticClass:"scorebg"},[e("div",{staticClass:"main_top_echarts_con_title"},[t._v("修复率")]),t._v(" "),e("p",{staticClass:"g_score"},[t._v(" "+t._s(t.result.score)+"%")])]),t._v(" "),"MNIST"==t.result.dataset_or_format|"CIFAR10"==t.result.dataset_or_format?e("div",{staticClass:"conclusion"},[e("p",{staticClass:"result_annotation"},[t._v("对"+t._s(t.result.num_images)+"张图片进行检测，检测到"+t._s(t.result.num_detect)+"张异常数据，使用Cleanlab算法对异常数据进行修复，修复率为"+t._s(t.result.score)+"%")])]):t._e()])]),t._v(" "),e("div",{staticClass:"result_div"},[e("div",{staticClass:"echart_title"},["table"==t.result.dataset_or_format?e("div",{staticClass:"main_top_echarts_con_title"},[t._v("低维数据修复结果")]):t._e(),t._v(" "),"txt_encode"==t.result.dataset_or_format|"txt_format"==t.result.dataset_or_format?e("div",{staticClass:"main_top_echarts_con_title"},[t._v("文本数据清洗结果")]):t._e(),t._v(" "),"MNIST"==t.result.dataset_or_format|"CIFAR10"==t.result.dataset_or_format?e("div",{staticClass:"main_top_echarts_con_title"},[t._v("检出标签错误示例")]):t._e()]),t._v(" "),e("div",{attrs:{id:"rdeva"}},["table"==t.result.dataset_or_format?e("div",{staticClass:"table_class_result"},[e("img",{attrs:{src:t.result.table_origin}}),t._v(" "),e("img",{attrs:{src:t.result.table_fix}})]):t._e(),t._v(" "),"txt_encode"==t.result.dataset_or_format|"txt_format"==t.result.dataset_or_format?e("div",{staticClass:"text_class_result"},[e("p",[t._v(t._s(t.result.text_origin))]),t._v(" "),e("p",[t._v(t._s(t.result.text_clean))])]):t._e(),t._v(" "),"MNIST"==t.result.dataset_or_format|"CIFAR10"==t.result.dataset_or_format?e("div",{staticClass:"image_class_result"},[e("img",{attrs:{src:t.result.demo_img}})]):t._e(),t._v(" "),e("div",{staticClass:"conclusion"},["table"==t.result.dataset_or_format?e("p",{staticClass:"result_text"},[t._v("图为低维数据清洗结果，图中黑点为真实离群样本点，黄点为真实良性样本点，绿线展示分类决策边界，在绿色范围内的即为清洗后的样本点。")]):t._e(),t._v(" "),"txt_encode"==t.result.dataset_or_format|"txt_format"==t.result.dataset_or_format?e("p",{staticClass:"result_text"},[t._v("左侧展示的是清洗前的文本数据，右侧展示的是清洗后的文本数据。")]):t._e(),t._v(" "),"MNIST"==t.result.dataset_or_format|"CIFAR10"==t.result.dataset_or_format?e("p",{staticClass:"result_text"},[t._v("图示为检测出的标签错误样本，其中蓝色框展示该样本原始标签及程序判别其可信度，灰色框展示该样本在数据集内的编号，绿色框内展示清洗后标签及可信度。")]):t._e()])])]),t._v(" "),e("a-button",{staticStyle:{width:"160px",height:"40px","margin-bottom":"30px","margin-top":"10px","font-size":"18px",color:"white","background-color":"rgb(46, 56, 245)","border-radius":"8px"},on:{click:function(a){return t.getPdf()}}},[t._v("\n                   导出报告内容\n               ")])],1)])],1),t._v(" "),e("a-layout-footer")],1)],1)},staticRenderFns:[]};var C=s("VU/8")(g,f,!1,function(t){s("/jQh")},"data-v-7d526769",null);a.default=C.exports}});