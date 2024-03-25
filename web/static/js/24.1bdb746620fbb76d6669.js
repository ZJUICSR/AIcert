webpackJsonp([24],{FSTF:function(t,e,a){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var s=a("mvHQ"),i=a.n(s),o=a("gRE1"),n=a.n(o),l=a("fZjL"),r=a.n(l),c=a("R45V"),d=a("83tA"),m=a("T9rv"),h=a("UI/F"),v=a("FaAE"),u=a("IDe4"),_=a.n(u),g=a("2b9l"),p=a.n(g),C={template:'\n        <svg t="1680138013828" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4354" width="128" height="128"><path d="M534.869333 490.496a1403.306667 1403.306667 0 0 0 50.858667-25.813333c16.042667-8.618667 29.013333-15.061333 38.570667-19.029334 9.557333-3.925333 17.066667-6.058667 22.869333-6.058666 9.557333 0 17.749333 3.2 24.917333 10.026666 6.826667 6.826667 10.581333 15.061333 10.581334 25.088 0 5.76-1.706667 11.818667-5.12 17.92-3.413333 6.101333-7.168 10.069333-10.922667 11.861334-35.157333 14.677333-74.410667 25.429333-116.736 31.872 7.850667 7.168 17.066667 17.237333 28.330667 29.781333 11.264 12.544 17.066667 18.986667 17.749333 20.053333 4.096 6.101333 9.898667 13.653333 17.408 22.613334 7.509333 8.96 12.629333 15.786667 15.36 20.778666 2.730667 5.034667 4.437333 11.093333 4.437333 18.304a33.706667 33.706667 0 0 1-9.898666 24.021334 33.834667 33.834667 0 0 1-25.6 10.410666c-10.24 0-22.186667-8.618667-35.157334-25.472-12.970667-16.512-30.037333-46.933333-50.517333-91.050666-20.821333 39.424-34.816 65.962667-41.642667 78.506666-7.168 12.544-13.994667 22.186667-20.48 28.672a30.976 30.976 0 0 1-22.528 9.685334 32.256 32.256 0 0 1-25.258666-11.093334 35.413333 35.413333 0 0 1-9.898667-23.68c0-7.893333 1.365333-13.653333 4.096-17.578666 25.258667-35.84 51.541333-67.413333 78.848-93.568a756.650667 756.650667 0 0 1-61.44-12.544 383.061333 383.061333 0 0 1-57.685333-20.48c-3.413333-1.749333-6.485333-5.717333-9.557334-11.818667a30.208 30.208 0 0 1-5.12-16.853333 32.426667 32.426667 0 0 1 10.581334-25.088 33.152 33.152 0 0 1 24.234666-10.026667c6.485333 0 14.677333 2.133333 24.576 6.101333 9.898667 4.266667 22.186667 10.026667 37.546667 18.261334 15.36 7.893333 32.426667 16.853333 51.882667 26.538666-3.413333-18.261333-6.485333-39.082667-8.874667-62.378666-2.389333-23.296-3.413333-39.424-3.413333-48.042667 0-10.752 3.072-19.712 9.557333-27.264A30.677333 30.677333 0 0 1 512.341333 341.333333c9.898667 0 18.090667 3.925333 24.576 11.477334 6.485333 7.893333 9.557333 17.92 9.557334 30.464 0 3.584-0.682667 10.410667-1.365334 20.48-0.682667 10.368-2.389333 22.570667-4.096 36.906666-2.048 14.677333-4.096 31.146667-6.144 49.834667z" fill="#FF3838" p-id="4355"></path></svg>\n        '},f={template:'\n        <a-icon :component="selectSvg" />\n    ',data:function(){return{selectSvg:C}}},S={name:"coverage_layer",components:{navmodule:c.a,func_introduce:d.a,showLog:m.a,resultDialog:h.a,selectIcon:f,DrawRobustBar:v.a},data:function(){return{htmlTitle:"CNN对抗训练",methodHoverIndex:-1,methodDescription:"",radioStyle:{display:"block",lineHeight:"30px"},datasetChoice:"CIFAR10",MNIST_imgs:[{imgUrl:a("wlDy"),name:"mnist0"},{imgUrl:a("KdR5"),name:"mnist1"},{imgUrl:a("2LXz"),name:"mnist2"},{imgUrl:a("3rtV"),name:"mnist3"},{imgUrl:a("Ppgw"),name:"mnist4"},{imgUrl:a("P9ea"),name:"mnist5"},{imgUrl:a("ec3M"),name:"mnist6"},{imgUrl:a("wAAx"),name:"mnist7"},{imgUrl:a("30PV"),name:"mnist8"},{imgUrl:a("v3v+"),name:"mnist9"}],CIFAR10_imgs:[{imgUrl:a("HiaR"),name:"mnist0"},{imgUrl:a("DJt5"),name:"mnist1"},{imgUrl:a("S99w"),name:"mnist2"},{imgUrl:a("J598"),name:"mnist3"},{imgUrl:a("/pRs"),name:"mnist4"},{imgUrl:a("YuLH"),name:"mnist5"},{imgUrl:a("Nvyw"),name:"mnist6"},{imgUrl:a("lB35"),name:"mnist7"},{imgUrl:a("dKp5"),name:"mnist8"},{imgUrl:a("NgyD"),name:"mnist9"}],modelChoice:"ResNet18",advChoice:"FGSM",advTrainMethod:["FGSM","FFGSM","RFGSM","MIFGSM","BIM","PGDL1","PGDL2","DIFGSM","C&W","TPGD"],selectedMethod:[],selectedAttributes:"",showmethodInfo:[[{name:"FGSM",description:"FGSM算法:快速梯度符号法是一种简单而有效的生成对抗样本的方法，其工作方式如下：在给定输入数据后，利用已训练的模型输出预测并计算损失函数的梯度，然后使用梯度的符号来创建使损失最大化的新数据"},{name:"FFGSM",description:"FFGSM算法：在使用FGSM攻击算法前加入随机初始化的扰动，经过实验发现基于FFGSM的对抗训练拥有高效性"},{name:"RFGSM",description:"RFGSM算法：R+FGSM在FGSM中加入随机的步骤, 是一个在白盒设置下高效的能替代迭代攻击的方法"},{name:"MIFGSM",description:"MIFGSM算法：momentum iterative FGSM是一种使用momentum迭代梯度的方法，该方法在迭代梯度对抗攻击(如BIM)的基础上，累计每次梯度方向的速度向量作为momentum，每次对抗扰动不再直接使用梯度方向，转而采用momentum方向，从而稳定更新方向并避免局部极值，更好提高攻击迁移性"}],[{name:"BIM",description:"BIM算法：Basic Iterative MethodBIM迭代式FGSM是对FGSM的改进方法，主要的改进有两点，其一是FGSM方法是一步完成的，而BIM方法通过多次迭代来寻找对抗样本；其次，为了避免迭代过程中出现超出有效值的情况出现，使用了一个修建方法严格限制像素值的范围"},{name:"PGDL1",description:"PGD算法：Projected Gradient DescentPGD投影梯度下降法是FGSM的迭代版本，该方法思路和BIM基本相同，不同之处在于该方法在迭代过程中使用范数投影的方法来约束非法数据，并且相对于BIM有一个随机的开始噪声"},{name:"PGDL2",description:"PGDL2算法：Projected Gradient DescentPGD投影梯度下降法是FGSM的迭代版本，该方法思路和BIM基本相同，不同之处在于该方法在迭代过程中使用范数投影的方法来约束非法数据，并且相对于BIM有一个随机的开始噪声"}],[{name:"DIFGSM",description:"DIFGSM算法：Diverse Inputs Iterative Fast Gradient Sign Method,通过创建多样的输入模式提高对抗样本的迁移性。做法是对输入的原图像以p的概率加上随机且可导的变换(transformation)，使用梯度的方法最大化模型对变换后的原图像的损失函数值从而得到对抗图像"},{name:"C&W",description:"C&W算法：该方法的出发点是攻击比较有名的对抗样本防御方法-防御蒸馏(就防御蒸馏方法而言，它在基本的L-BFGS，FGSM攻击方法上表现本身就比较差)。对于寻找对抗样本过程中目标函数的设置将会极大的影响对抗样本的攻击效果，为此，通过目标函数的设定，在零范数，二范数和无穷范数的限制下分别设计了三种不同的寻找对抗样本的目标函数，这三种方法均可以绕过防御蒸馏的防御"},{name:"TPGD",description:"TPGD算法：基于KL-Divergence loss的pgd攻击"}]],buttonBGColor:{background:"#0B55F4",color:"#FFFFFF"},disStatus:!1,logflag:!1,percent:10,logtext:[],funcDesText:{name:"模型鲁棒性训练",imgpath:_.a,bgimg:p.a,destext:"提升模型在对抗样本攻击下的鲁棒性",backinfo:"对抗攻击对于模型危害巨大，轻则造成模型失效，重则影响人工智能安全性。通过可认证鲁棒训练、对抗训练等方式来对AI模型进行安全加固，提升模型在对抗样本攻击下的鲁棒性。",highlight:["鲁棒性训练方法5种，满足多任务类型模型的鲁棒性提升需求；","面向GCN的可认证鲁棒训练，能够有效提升图神经网络模型的鲁棒性；","面向CNN的对抗训练、基于特征散射的鲁棒性训练、基于异常感知的鲁棒性训练以及基于随机平滑的鲁棒性训练，能够有效提升卷积神经网络的鲁棒性。"]},isShowPublish:!1,result:{},res_tmp:{},tid:"",stidlist:"",clk:"",logclk:""}},watch:{isShowPublish:{immediate:!0,handler:function(t){t?this.noScroll():this.canScroll()}}},created:function(){document.title="模型鲁棒性训练"},methods:{closeDialog:function(){this.isShowPublish=!1},onDatasetChoiceChange:function(t){console.log("radio checked",t.target.value)},onModelChoiceChange:function(t){console.log("radio checked",t.target.value)},onAdvChoiceChange:function(t){console.log("radio checked",t)},handleBlur:function(){console.log("blur")},handleFocus:function(){console.log("focus")},changeMethods:function(t,e){var a=document.getElementById("button"+t+e);""==a.style.color?(this.methodHoverIndex=t,this.methodDescription=this.showmethodInfo[t][e].description,a.style.color="#0B55F4",a.style.borderColor="#C8DCFB",a.style.background="#E7F0FD",this.selectedMethod.push(this.showmethodInfo[t][e].name)):(this.methodHoverIndex=-1,this.methodDescription="",a.style.color="",a.style.borderColor="#C8DCFB",a.style.background="#F2F4F9",a.blur(),this.selectedMethod.splice(this.selectedMethod.indexOf(this.showmethodInfo[t][e].name),1))},exportResult:function(){if(confirm("您确认下载该pdf文件吗？")){var t=document.getElementById("download_page"),e={margin:[10,20,10,20],filename:this.tid+".pdf",image:{type:"jpeg",quality:1},html2canvas:{scale:5},jsPDF:{unit:"mm",format:"a4",orientation:"portrait"}};html2pdf().from(t).set(e).save()}},resultPro:function(t){var e=t.CNN_AT.Normal,a=t.CNN_AT.Enhance,s=["Normal Training","Robust Training"],i=r()(e.atk_acc),o=n()(e.atk_acc),l=n()(a.atk_acc),c=n()(e.atk_asr),d=n()(a.atk_asr);Object(v.a)("adv_robust_result1",s,i,o,l),Object(v.a)("adv_robust_result2",s,i,c,d)},getData:function(){var t=this;t.$axios.get("/output/Resultdata",{params:{Taskid:t.tid}}).then(function(e){console.log("dataget:",e),t.res_tmp=e})},getLog:function(){var t=this;t.percent<99&&(t.percent+=1),t.$axios.get("/Task/QueryLog",{params:{Taskid:t.tid}}).then(function(e){if("{}"==i()(t.stidlist))t.logtext=[n()(e.data.Log).slice(-1)[0]];else for(var a in t.logtext=[],t.stidlist)t.logtext.push(e.data.Log[t.stidlist[a]])})},stopTimer:function(){1==this.res_tmp.data.stop&&this.tid==this.res_tmp.data.result.tid&&(this.percent=100,this.logflag=!1,clearInterval(this.clk),clearInterval(this.logclk),this.isShowPublish=!0,this.resultPro(this.res_tmp.data.result))},update:function(){this.getData();try{this.stopTimer()}catch(t){}},changeSelectPage:function(){},dataEvaClick:function(){var t=this;0!=t.selectedMethod.length?this.$axios.post("/Task/CreateTask",{AttackAndDefenseTask:0}).then(function(e){t.tid=e.data.Taskid;var a={dataset:t.datasetChoice,modelname:t.modelChoice,attackmethod:t.advChoice,evaluate_methods:t.selectedMethod,tid:t.tid};t.$axios.post("/Defense/AdvTraining_CNNAT",a).then(function(e){t.logflag=!0,t.stidlist={CNN_AT:e.data.stid},t.logclk=self.setInterval(t.getLog,3e3),t.clk=self.setInterval(t.update,3e3)}).catch(function(t){console.log(t)})}).catch(function(t){console.log(t)}):t.$message.warning("请至少选择一项对抗攻击方法！",3)}}},b={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",[a("a-layout",[a("a-layout-header",[a("navmodule")],1),t._v(" "),a("a-layout-content",[a("func_introduce",{attrs:{funcDesText:t.funcDesText}}),t._v(" "),a("div",{staticClass:"paramCon"},[a("h2",{staticClass:"subTitle",staticStyle:{"margin-top":"-96px"}},[t._v("参数配置")]),t._v(" "),a("div",{staticClass:"labelSelection"},[a("router-link",{attrs:{to:"/robust_advTraining"}},[a("button",{staticClass:"labelselected"},[t._v("对抗鲁棒训练")])]),t._v(" "),a("router-link",{attrs:{to:"/gcn_robustTraining"}},[a("button",{staticClass:"labelunselected"},[t._v("可认证鲁棒性训练")])]),t._v(" "),a("router-link",{attrs:{to:"/featurescatter_robustTraining"}},[a("button",{staticClass:"labelunselected"},[t._v("特征散射鲁棒性训练")])]),t._v(" "),a("router-link",{attrs:{to:"/seat_robustTraining"}},[a("button",{staticClass:"labelunselected"},[t._v("自我整合鲁棒性训练")])]),t._v(" "),a("router-link",{attrs:{to:"/smoothing_robustTraining"}},[a("button",{staticClass:"labelunselected"},[t._v("关键参数微调鲁棒性训练")])])],1),t._v(" "),a("div",{staticClass:"funcParam"},[a("div",{staticClass:"paramTitle"},[a("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),a("h3",[t._v(t._s(t.funcDesText.name))]),t._v(" "),a("a-button",{staticClass:"DataEva",style:t.buttonBGColor,attrs:{disabled:t.disStatus},on:{click:t.dataEvaClick}},[a("a-icon",{attrs:{type:"security-scan"}}),t._v("\n                       评估\n                   ")],1)],1),t._v(" "),a("a-divider"),t._v(" "),a("div",{staticClass:"inputdiv"},[a("div",{staticClass:"datasetSelected"},[a("p",{staticClass:"mainParamNameNotop"},[t._v("请选择数据集")]),t._v(" "),a("a-radio-group",{on:{change:t.onDatasetChoiceChange},model:{value:t.datasetChoice,callback:function(e){t.datasetChoice=e},expression:"datasetChoice"}},[a("div",{staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"CIFAR10"}},[t._v("\n                                   CIFAR10\n                               ")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[a("span",[t._v("CIFAR10数据集：")]),t._v("是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图 片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("图例：")]),t._v(" "),a("div",{staticClass:"demoData"},t._l(t.CIFAR10_imgs,function(t,e){return a("div",{key:e},[a("img",{attrs:{src:t.imgUrl}})])}),0)],1),t._v(" "),a("div",{staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"MNIST"}},[t._v("\n                                   MNIST\n                               ")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[a("span",[t._v("MNIST数据集：")]),t._v("是一个手写体数字的图片数据集，该数据集来由美国国家标准与技术研究所（National Institute of Standards and Technology (NIST)）发起整理，一共统计了来自250个不同的人手写数字图片，其中50%是高中生，50%来自人口普查局的工作人员。该数据集的收集目的是希望通过算法，实现对手写数字的识别。")]),t._v(" "),a("p",{staticClass:"matchedMethodText"},[t._v("图例：")]),t._v(" "),a("div",{staticClass:"demoData"},t._l(t.MNIST_imgs,function(t,e){return a("div",{key:e},[a("img",{attrs:{src:t.imgUrl}})])}),0)],1)])],1),t._v(" "),a("div",{staticClass:"modelSelected"},[a("p",{staticClass:"mainParamName"},[t._v("请选择模型")]),t._v(" "),a("a-radio-group",{on:{change:t.onModelChoiceChange},model:{value:t.modelChoice,callback:function(e){t.modelChoice=e},expression:"modelChoice"}},[a("div",{staticClass:"matchedDes"},[a("a-radio",{style:t.radioStyle,attrs:{value:"ResNet18"}},[t._v("ResNet18")]),t._v(" "),a("a-radio",{style:t.radioStyle,attrs:{value:"ResNet34"}},[t._v("ResNet34")]),t._v(" "),a("a-radio",{style:t.radioStyle,attrs:{value:"ResNet50"}},[t._v("ResNet50")])],1)])],1),t._v(" "),a("div",{staticClass:"modelSelected"},[a("p",{staticClass:"mainParamName"},[t._v("请选择对抗训练方法")]),t._v(" "),a("a-select",{staticStyle:{width:"1104px"},on:{focus:t.handleFocus,blur:t.handleBlur,change:t.onAdvChoiceChange},model:{value:t.advChoice,callback:function(e){t.advChoice=e},expression:"advChoice"}},t._l(t.advTrainMethod,function(e){return a("a-select-option",{attrs:{value:e}},[t._v("\n                           "+t._s(e)+"\n                           ")])}),1)],1),t._v(" "),a("div",{staticClass:"thresholdSet"},[a("p",{staticClass:"mainParamName"},[t._v("请输入选择攻击方法（可多选）")]),t._v(" "),t._l(t.showmethodInfo,function(e,s){return a("div",{key:s,staticStyle:{"margin-bottom":"16px"}},[a("a-row",{staticStyle:{height:"50px"},attrs:{gutter:16,type:"flex"}},t._l(e,function(i,o){return a("a-col",{key:o,staticClass:"denfenseMethod",attrs:{flex:24/e.length}},[a("a-button",{attrs:{id:"button"+s+o},on:{click:function(e){return t.changeMethods(s,o)}}},[t._v(t._s(i.name))])],1)}),1),t._v(" "),t.methodHoverIndex==s&&""!==t.methodDescription?a("div",{staticStyle:{padding:"14px 24px",margin:"16px auto"}},[t._v(" "+t._s(t.methodDescription)+" ")]):t._e()],1)})],2)])],1)]),t._v(" "),t.logflag?a("div",[a("showLog",{attrs:{percent:t.percent,logtext:t.logtext}})],1):t._e(),t._v(" "),a("resultDialog",{directives:[{name:"show",rawName:"v-show",value:t.isShowPublish,expression:"isShowPublish"}],ref:"report_pdf",attrs:{isShow:t.isShowPublish},on:{"on-close":t.closeDialog}},[a("div",{attrs:{slot:"header"},slot:"header"},[a("div",{staticClass:"dialog_title"},[a("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),a("h1",[t._v("CNN对抗训练")])])]),t._v(" "),a("div",{staticClass:"dialog_publish_main",attrs:{slot:"main",id:"pdfDom"},slot:"main"},[a("div",{staticClass:"result_div"},[a("div",{staticClass:"conclusion_info"},[a("p",{staticClass:"result_annotation"},[t._v("数据集："+t._s(t.datasetChoice))]),t._v(" "),a("p",{staticClass:"result_annotation"},[t._v("模型："+t._s(t.modelChoice))]),t._v(" "),a("p",{staticClass:"result_annotation"},[t._v("对抗训练方法："+t._s(t.advChoice))]),t._v(" "),a("p",{staticClass:"result_annotation"},[t._v("攻击方法：")]),t._v(" "),a("div",{staticClass:"result_annotation",staticStyle:{"word-wrap":"break-word",display:"flex","flex-direction":"row","flex-wrap":"nowrap","justify-content":"flex-start","align-items":"center",gap:"10px"}},t._l(t.selectedMethod,function(e,s){return a("p",{key:s},[t._v(t._s(e))])}),0)]),t._v(" "),a("div",{staticClass:"main_top_echarts_con_title"},[t._v("模型对抗训练效果")]),t._v(" "),a("p",{staticClass:"echart_title"},[t._v("训练前后模型受攻击分类准确率")]),t._v(" "),a("div",{attrs:{id:"rdeva"}},[a("div",{staticStyle:{width:"1000px",height:"500px"},attrs:{id:"adv_robust_result1"}})]),t._v(" "),a("p",{staticClass:"echart_title"},[t._v("训练前后模型攻击成功率")]),t._v(" "),a("div",{attrs:{id:"rdeva"}},[a("div",{staticStyle:{width:"1000px",height:"500px"},attrs:{id:"adv_robust_result2"}}),t._v(" "),a("div",{staticClass:"conclusion"},[a("p",{staticClass:"result_text"},[t._v(t._s(t.modelChoice)+"模型、"+t._s(t.datasetChoice)+"数据集，用"+t._s(t.advChoice)+"对抗训练方法对模型鲁棒性进行提升。")])])])]),t._v(" "),a("a-button",{staticStyle:{width:"160px",height:"50px","margin-bottom":"30px","margin-top":"10px","font-size":"18px",color:"white","background-color":"rgb(46, 56, 245)","border-radius":"8px"},on:{click:function(e){return t.getPdf()}}},[t._v("\n                 导出报告内容\n               ")])],1)])],1),t._v(" "),a("a-layout-footer")],1)],1)},staticRenderFns:[]};var M=a("VU/8")(S,b,!1,function(t){a("gqWq")},"data-v-cfe8eb3e",null);e.default=M.exports},gqWq:function(t,e){}});