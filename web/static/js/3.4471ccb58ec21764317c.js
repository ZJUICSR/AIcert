webpackJsonp([3],{"/ASi":function(t,e){},"2XIm":function(t,e,s){t.exports=s.p+"static/img/ImageNet5.043c15a.png"},"95eC":function(t,e,s){t.exports=s.p+"static/img/ImageNet1.d779c91.png"},"9pcm":function(t,e,s){t.exports=s.p+"static/img/ImageNet2.5601c8c.png"},GUbY:function(t,e,s){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var i={};s.d(i,"parse",function(){return M});var a=s("gRE1"),n=s.n(a),o=s("mvHQ"),r=s.n(o),d=s("R45V"),l=s("83tA"),c=s("T9rv"),h=s("UI/F"),u=s("oh/m"),p=s.n(u),m=s("2b9l"),v=s.n(m),g=s("XLwt"),f=s("/gxq");function M(t){var e;"string"==typeof t?e=(new DOMParser).parseFromString(t,"text/xml"):e=t;if(!e||e.getElementsByTagName("parsererror").length)return null;var s=y(e,"gexf");if(!s)return null;for(var i,a=y(s,"graph"),n=(i=y(a,"attributes"))?f.H(A(i,"attribute"),function(t){return{id:x(t,"id"),title:x(t,"title"),type:x(t,"type")}}):[],o={},r=0;r<n.length;r++)o[n[r].id]=n[r];return{nodes:function(t,e){return t?f.H(A(t,"node"),function(t){var s=x(t,"id"),i=x(t,"label"),a={id:s,name:i,itemStyle:{normal:{}}},n=y(t,"viz:size"),o=y(t,"viz:position"),r=y(t,"viz:color"),d=y(t,"attvalues");if(n&&(a.symbolSize=parseFloat(x(n,"value"))),o&&(a.x=parseFloat(x(o,"x")),a.y=parseFloat(x(o,"y"))),r&&(a.itemStyle.normal.color="rgb("+[0|x(r,"r"),0|x(r,"g"),0|x(r,"b")].join(",")+")"),d){var l=A(d,"attvalue");a.attributes={};for(var c=0;c<l.length;c++){var h=l[c],u=x(h,"for"),p=x(h,"value"),m=e[u];if(m){switch(m.type){case"integer":case"long":p=parseInt(p,10);break;case"float":case"double":p=parseFloat(p);break;case"boolean":p="true"===p.toLowerCase()}a.attributes[u]=p}}}return a}):[]}(y(a,"nodes"),o),links:function(t){return t?f.H(A(t,"edge"),function(t){var e=x(t,"id"),s=x(t,"label"),i=x(t,"source"),a=x(t,"target"),n={id:e,name:s,source:i,target:a,lineStyle:{normal:{}}},o=n.lineStyle.normal,r=y(t,"viz:thickness"),d=y(t,"viz:color");return r&&(o.width=parseFloat(r.getAttribute("value"))),d&&(o.color="rgb("+[0|x(d,"r"),0|x(d,"g"),0|x(d,"b")].join(",")+")"),n}):[]}(y(a,"edges"))}}function x(t,e){return t.getAttribute(e)}function y(t,e){for(var s=t.firstChild;s;){if(1===s.nodeType&&s.nodeName.toLowerCase()===e.toLowerCase())return s;s=s.nextSibling}return null}function A(t,e){for(var s=t.firstChild,i=[];s;)s.nodeName.toLowerCase()===e.toLowerCase()&&i.push(s),s=s.nextSibling;return i}function C(t,e){var s=(t.length-1)*e+1,i=Math.floor(s),a=+t[i-1],n=s-i;return n?a+n*(t[i]-a):a}g.a&&(g.a.version="1.0.0",g.a.gexf=i,g.a.prepareBoxplotData=function(t,e){for(var s,i=[],a=[],n=[],o=(e=e||{}).boundIQR,r="none"===o||0===o,d=0;d<t.length;d++){n.push(d+"");var l=((s=t[d].slice()).sort(function(t,e){return t-e}),s),c=C(l,.25),h=C(l,.5),u=C(l,.75),p=l[0],m=l[l.length-1],v=(null==o?1.5:o)*(u-c),g=r?p:Math.max(p,c-v),f=r?m:Math.min(m,u+v);i.push([g,c,h,u,f]);for(var M=0;M<l.length;M++){var x=l[M];if(x<g||x>f){var y=[d,x];"vertical"===e.layout&&y.reverse(),a.push(y)}}}return{boxData:i,outliers:a,axisData:n}});var b=s("qo+b"),_={name:"DropSelect",props:{Id:{type:String,default:Math.random().toString()},items:{type:Array,default:function(){return["a","b","c","d","e"]}}},data:function(){return{message:"请选择",showNum:"0",selected:""}},mounted:function(){},methods:{nameClick:function(){this.showNum=1},ulClick:function(){this.showNum=0},changeClick:function(t){this.message="当前展示攻击方法："+t,this.selected=t,this.$emit("SelectClick",t)}}},k={render:function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",[i("div",{staticClass:"team-list"},[i("div",{staticClass:"team-name",on:{click:function(e){return t.nameClick()}}},[i("div",[t._v(t._s(t.message))]),t._v(" "),i("img",{directives:[{name:"show",rawName:"v-show",value:0==t.showNum,expression:"showNum==0"}],staticClass:"team-xiala",attrs:{src:s("v/51")}}),t._v(" "),i("img",{directives:[{name:"show",rawName:"v-show",value:1==t.showNum,expression:"showNum==1"}],staticClass:"team-shouqi",attrs:{src:s("nTcO")}})]),t._v(" "),i("ul",{directives:[{name:"show",rawName:"v-show",value:1==t.showNum,expression:"showNum==1"}],staticClass:"team-form",on:{click:function(e){return t.ulClick()}}},t._l(t.items,function(e,s){return i("li",{key:s,on:{click:function(s){return t.changeClick(e)}}},[t._v(t._s(e))])}),0)])])},staticRenderFns:[]};var I=s("VU/8")(_,k,!1,function(t){s("JIup")},"data-v-9f1fe2ce",null).exports,S={name:"resultDialog",components:{PictureTable:b.a,DropSelect:I},props:{isShow:{type:Boolean,default:!1,required:!0},result:{},postData:{},widNum:{type:Number,default:50},leftSite:{type:Number,default:25.2},topDistance:{type:Number,default:10},pdt:{type:Number,default:22},pdb:{type:Number,default:47}},data:function(){return{htmlTitle:"AdvExplainDefenseReport",echart_init:!1,selectedPictureIndex:0,allPictures:{},selectedMethod:"",allDimReduction:{},attackMethods:[],selectPicList:[],instanceResultList:[],instanceLayerResultList:[],DimReducitonResult:[],tid:"",stidlist:{},cellHeight:"200px",chartHeight:220,advCellWidth:[.2,.2,.2,.2,.2],postdata:{},dimCellWidth:[.13,.29,.29,.29]}},methods:{defenseShow:function(){return(arguments.length>0&&void 0!==arguments[0]?arguments[0]:"[]").join("、")},closeMyself:function(){this.$emit("on-close")},_stopPropagation:function(t){t.stopPropagation()},selectPicture:function(t){this.selectedPictureIndex=parseInt(t)},changeCellHeight:function(t){this.cellHeight=t},splitArr:function(t){for(var e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:1,s=0,i=[];s<t.length;)i.push(t.slice(s,s+e)),s+=e;return i},selectAdvMethod:function(t){this.selectedMethod=t,this.instanceLayerResultList=this.getInstanceLayerResult(this.selectedPictureIndex,this.selectedMethod)},getInstanceResult:function(t){var e=[],s="./static/output/"+this.tid+"/"+this.stidlist.feature+"/",i=["解释方法","原始样本"],a=[];1==this.postdata.ExMethods.length?this.advCellWidth=[.2,.4,.4]:2==this.postdata.ExMethods.length?this.advCellWidth=[.25,.25,.25,.25]:this.advCellWidth=[.2,.2,.2,.2,.2],this.postdata.ExMethods.indexOf("lrp")>-1&&(i.push(["LRP","图中像素点表示原始图像对应位置每个像素对分类决策的贡献度，颜色越深表示贡献越大，蓝色表示反向贡献度，红色表示正向贡献。"]),a.push("lrp")),this.postdata.ExMethods.indexOf("gradcam")>-1&&(i.push(["Grad-CAM","图中热力图覆盖目标为模型预测时重点关注的目标，采用jet颜色域，从蓝色到红色表示重要度逐渐变高。"]),a.push("gradcam")),this.postdata.ExMethods.indexOf("integrated_grad")>-1&&(i.push(["IG","积分梯度法（Integrated Gradients，IG）通过对模型梯度做积分来归因目标样本中对分类起到积极作用的特征，并通过灰度图进行展示。其中白色区域表示产生积极影响的像素，该像素对模型分类的影响作用越大，相应区域的白色越亮。"]),a.push("ig")),e.push(i);var n=this.allPictures.nor,o=["正常样本 分类标签:\n"+n.class_name[t],[s+n.nor_imgs[t],"pic"]];for(var r in a)o.push([s+n[a[r]][t],"pic"]);e.push(o);var d=this.allPictures.adv_methods;for(var l in d){var c=this.allPictures[d[l]],h=[d[l]+" 分类标签:\n"+c.class_name[t],[s+c.adv_imgs[t],"pic"]];for(var u in a)h.push([s+c[a[u]][t],"pic"]);e.push(h)}return e},getInstanceLayerResult:function(t,e){var s=[],i="./static/output/"+this.tid+"/"+this.stidlist.feature+"/";s.push(["卷积层","原始样本结果","对抗样本结果","相似性指数"]);var a=this.allLayerPictures["img_"+t].nor,n=this.allLayerPictures["img_"+t][e],o=this.allLayerPictures.value[e]["img_"+t],r=0;for(var d in a)s.push([d,[i+a[d],"pic"],[i+n[d],"pic"],[o[r],"text"]]),r++;return s},handleSelectPicture:function(t){var e=this.selectPicList[t[0][0]][t[0][1]][0],s=parseInt(e.split("_").pop().replace(".png",""));this.selectPicture(s),this.instanceResultList=this.getInstanceResult(this.selectedPictureIndex),this.instanceLayerResultList=this.getInstanceLayerResult(this.selectedPictureIndex,this.selectedMethod)},getDimReducitonResult:function(){var t=[],e=["对抗攻击方法"],s=[];for(var i in 1==this.postdata.VisMethods.length?this.dimCellWidth=[.15,.85]:2==this.postdata.VisMethods.length&&(this.dimCellWidth=[.14,.43,.43]),this.postdata.VisMethods.indexOf("pca")>-1?(e.push("PCA"),s.push({name:"pca",type:"scatterChart"})):this.chartHeight+=20,this.postdata.VisMethods.indexOf("svm")>-1?(e.push("SVM"),s.push({name:"svm",type:"HistogramChart"})):this.chartHeight+=20,this.postdata.VisMethods.indexOf("tsne")>-1?(e.push("t-SNE"),s.push({name:"tsne",type:"scatterChart"})):this.chartHeight+=20,t.push(e),this.attackMethods){var a=[this.attackMethods[i]];for(var n in s)a.push([this.allDimReduction[this.attackMethods[i]][s[n].name],s[n].type]);t.push(a)}return console.log("dimlist:",t),t},setBoxchartsOptions:function(t,e,s){return{title:[{text:s+"的肯德尔相似系数",left:"center",textStyle:{color:"#333"}}],tooltip:{trigger:"item",axisPointer:{type:"shadow"}},grid:{left:"10%",right:"10%",bottom:"15%"},xAxis:{type:"category",boundaryGap:!0,name:"对抗攻击方法",nameGap:30,data:e,textStyle:{fontSize:16},nameTextStyle:{color:"#fff"},axisLabel:{color:"fff",formatter:"{value}"},splitArea:{show:!1},splitLine:{show:!1}},yAxis:{type:"value",name:"相似性分数（肯德尔相关系数）",splitArea:{show:!0}},series:[{name:"数据信息",type:"boxplot",data:t.boxData,itemStlye:{color:"fff"},datasetIndex:1},{name:"异常点",type:"scatter",data:t.outliers,datasetIndex:2}]}},webupdate:function(){if(this.postdata.ExMethods.indexOf("lrp")>-1){for(var t=[],e=0;e<this.attackMethods.length;e++)t.push(this.allPictures.kendalltau[this.attackMethods[e]].lrp);var s=this.setBoxchartsOptions(g.a.prepareBoxplotData(t),this.attackMethods,"LRP");setTimeout(function(){var t=g.b(document.getElementById("myChart1"));window.addEventListener("resize",function(){t.resize()}),s&&t.setOption(s)},500)}if(this.postdata.ExMethods.indexOf("gradcam")>-1){for(var i=[],a=0;a<this.attackMethods.length;a++)i.push(this.allPictures.kendalltau[this.attackMethods[a]].gradcam);var n=this.setBoxchartsOptions(g.a.prepareBoxplotData(i),this.attackMethods,"Grad-CAM");setTimeout(function(){var t=g.b(document.getElementById("myChart2"));window.addEventListener("resize",function(){t.resize()}),n&&t.setOption(n)},500)}if(this.postdata.ExMethods.indexOf("integrated_grad")>-1){for(var o=[],r=0;r<this.attackMethods.length;r++)o.push(this.allPictures.kendalltau[this.attackMethods[r]].ig);var d=this.setBoxchartsOptions(g.a.prepareBoxplotData(o),this.attackMethods,"IG");setTimeout(function(){var t=g.b(document.getElementById("myChart3"));window.addEventListener("resize",function(){t.resize()}),d&&t.setOption(d)},500)}}},watch:{result:function(t,e){if("tid"in t){if(this.tid=t.tid,this.stidlist=t.stidlist,this.postdata=this.postData,"attack_attrbution_analysis"in this.result){this.allPictures=this.result.attack_attrbution_analysis.adv_ex,this.attackMethods=this.result.attack_attrbution_analysis.adv_ex.adv_methods;for(var s=[],i=0;i<this.result.attack_attrbution_analysis.adv_ex.nor.nor_imgs.length;i++){var a=["./static/output/"+this.tid+"/"+this.stidlist.feature+"/"+this.result.attack_attrbution_analysis.adv_ex.nor.nor_imgs[i],"pic"];s.push(a)}this.selectedMethod=this.attackMethods[0],this.selectPicList=this.splitArr(s,10),this.instanceResultList=this.getInstanceResult(0),"layer_ex"in this.result.attack_attrbution_analysis&&(this.allLayerPictures=this.result.attack_attrbution_analysis.layer_ex,this.instanceLayerResultList=this.getInstanceLayerResult(0,this.selectedMethod))}"attack_dim_reduciton"in this.result&&(this.allDimReduction=this.result.attack_dim_reduciton,this.DimReducitonResult=this.getDimReducitonResult())}}},mounted:function(){this.webupdate()},updated:function(){this.webupdate()}},w={render:function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"dialog"},[t.isShow?i("div",{staticClass:"dialog-cover back",on:{click:t.closeMyself}}):t._e(),t._v(" "),i("transition",{attrs:{name:"drop"}},[t.isShow?i("div",{staticClass:"dialog-content",on:{click:function(e){return e.stopPropagation(),t._stopPropagation(e)}}},[i("div",{staticClass:"dialog_head back"},[i("div",{staticClass:"close_button"},[i("a-icon",{staticClass:"closebutton",staticStyle:{"font-size":"20px",color:"#6C7385"},attrs:{type:"close"},on:{click:t.closeMyself}})],1),t._v(" "),t._t("header",function(){return[i("div",{staticClass:"dialog-title"},[i("img",{staticStyle:{width:"50px",height:"50px"},attrs:{src:s("oh/m")}}),t._v("攻击机理分析报告")])]})],2),t._v(" "),i("div",{staticClass:"dialog_main",attrs:{id:"pdfDom"}},[t._t("main",function(){return[Object.keys(t.postdata).length>0?i("div",{staticStyle:{background:"var(--gray-7, #F2F4F9)",width:"100%",padding:"24px"}},[i("a-row",[i("a-col",{attrs:{span:2}},[i("div",{staticClass:"grid-content-name",staticStyle:{color:"#6C7385"}},[t._v("数据集:")])]),t._v(" "),i("a-col",{attrs:{span:3}},[i("div",{staticClass:"grid-content-value"},[t._v(t._s(t.postdata.DatasetParam.name))])]),t._v(" "),i("a-col",{attrs:{span:5}},[i("div",{staticClass:"grid-content-name",staticStyle:{color:"#6C7385"}},[t._v("特征归因可视化分析方法:")])]),t._v(" "),i("a-col",{attrs:{span:4}},[t.postdata.ExMethods.length>0?i("div",{staticClass:"grid-content-value"},[t._v(t._s(t.defenseShow(t.postdata.ExMethods)))]):i("div",{staticClass:"grid-content-value"},[t._v("未选")])]),t._v(" "),i("a-col",{attrs:{span:2}},[i("div",{staticClass:"grid-content-name",staticStyle:{color:"#6C7385"}},[t._v("攻击方法:")])]),t._v(" "),i("a-col",{attrs:{span:8}},[i("div",{staticClass:"grid-content-value"},[t._v(t._s(t.defenseShow(t.postdata.AdvMethods)))])])],1),t._v(" "),i("a-row",[i("a-col",{attrs:{span:2}},[i("div",{staticClass:"grid-content-name",staticStyle:{color:"#6C7385"}},[t._v("模型:")])]),t._v(" "),i("a-col",{attrs:{span:3}},[i("div",{staticClass:"grid-content-value"},[t._v(t._s(t.postdata.ModelParam.name))])]),t._v(" "),i("a-col",{attrs:{span:5}},[i("div",{staticClass:"grid-content-name",staticStyle:{color:"#6C7385"}},[t._v("数据分布降维分析方法:")])]),t._v(" "),i("a-col",{attrs:{span:4}},["VisMethods"in t.postdata?i("div",{staticClass:"grid-content-value"},[t._v(t._s(t.defenseShow(t.postdata.VisMethods)))]):i("div",{staticClass:"grid-content-value"},[t._v("未选")])]),t._v(" "),t.postdata.Use_layer_explain?i("a-col",{attrs:{span:2}},[i("div",{staticClass:"grid-content-name",staticStyle:{color:"#6C7385"}},[t._v("模型内部解释算法:")])]):t._e(),t._v(" "),t.postdata.Use_layer_explain?i("a-col",{attrs:{span:8}},[i("div",{staticClass:"grid-content-value"},[t._v("Guided-backpropagation")])]):t._e()],1)],1):t._e(),t._v(" "),"ExMethods"in t.postdata?i("div",{staticClass:"result-title"},[t._v("对抗图像解释")]):t._e(),t._v(" "),"ExMethods"in t.postdata?i("div",{staticClass:"selectContent"},[t._v("请选择展示的图片")]):t._e(),t._v(" "),i("div",{staticStyle:{width:"960px"}},[i("PictureTable",{key:"pictable0",staticClass:"center-horizon",staticStyle:{height:"100%",width:"960px","margin-bottom":"20px"},attrs:{"table-id":"table0",header:!1,headerRow:!1,"have-border":!0,content:t.selectPicList,"single-output":!0,cellWidth:[.1,.1,.1,.1,.1,.1,.1,.1,.1,.1],selectedPicutre:["table0_0_0"]},on:{pictureSelect:t.handleSelectPicture}})],1),t._v(" "),"ExMethods"in t.postdata?i("div",{staticStyle:{width:"960px"}},[i("PictureTable",{key:"pictable1",staticClass:"center-horizon",staticStyle:{height:"100%",width:"960px","margin-bottom":"20px"},attrs:{"table-id":"table1",header:!0,headerRow:!0,"have-border":!0,content:t.instanceResultList,"single-output":!0,cellWidth:t.advCellWidth,cellHeight:t.cellHeight}})],1):t._e(),t._v(" "),"ExMethods"in t.postdata?i("div",[i("div",{staticClass:"result-subtitle"},[t._v("对抗解释图与正常解释图之间的相似度分数 ")]),t._v(" "),t.postdata.ExMethods.indexOf("lrp")>-1?i("div",{staticClass:"echart",attrs:{id:"myChart1"}}):t._e(),t._v(" "),t.postdata.ExMethods.indexOf("gradcam")>-1?i("div",{staticClass:"echart",attrs:{id:"myChart2"}}):t._e(),t._v(" "),t.postdata.ExMethods.indexOf("integrated_grad")>-1?i("div",{staticClass:"echart",attrs:{id:"myChart3"}}):t._e()]):t._e(),t._v(" "),"ExMethods"in t.postdata?i("div",{staticClass:"describe",staticStyle:{width:"960px","line-height":"30px"}},[t._v("\n                    通过观察可视化后的结果，可以看到对抗样本通常会显著改变模型关注的特征区域（比如无法关注在目标物体上），\n                    说明模型关注区域发生了偏移；此外通过计算相关系数量化这种差异的程度，相关系数为0-1的实数，越大说明结果越相关（即差异越小），可以看到对抗样本影响下，\n                    相关指数基本维持在0.6附近，说明对抗样本使模型对特征的关注区域发生了较大偏移。\n                  ")]):t._e(),t._v(" "),i("div",["ExMethods"in t.postdata&&t.postdata.Use_layer_explain?i("div",{staticClass:"result-subtitle"},[t._v("模型内部卷积层解释 ")]):t._e(),t._v(" "),"ExMethods"in t.postdata&&t.postdata.Use_layer_explain?i("DropSelect",{attrs:{Id:"select0",message:"当前展示攻击方法："+t.attackMethods[0],items:t.attackMethods},on:{SelectClick:t.selectAdvMethod}}):t._e(),t._v(" "),i("div",{staticStyle:{width:"960px"}},["ExMethods"in t.postdata&&t.postdata.Use_layer_explain?i("PictureTable",{key:"pictable2",staticClass:"center-horizon",staticStyle:{height:"100%",width:"960px","margin-bottom":"20px"},attrs:{"table-id":"table2",header:!0,headerRow:!0,"have-border":!0,content:t.instanceLayerResultList,"single-output":!0,cellWidth:[.21,.29,.29,.21],cellHeight:t.cellHeight}}):t._e(),t._v(" "),"ExMethods"in t.postdata&&t.postdata.Use_layer_explain?i("div",{staticClass:"describe",staticStyle:{width:"960px","line-height":"30px"}},[t._v("通过对神经网络逐层提取到的特征进行可视化观察，可以发现对抗噪声从模型浅层就开始破坏图像纹理特征，并随着传播深入最终导致模型分类错误。\n                      同时通过计算得到的差异性指数（越大说明对抗扰动造成的特征破坏程度越大〉也可以证明这一点")]):t._e()],1)],1),t._v(" "),"VisMethods"in t.postdata?i("div",[i("div",{staticClass:"result-subtitle"},[t._v("数据特征分布降维解释 ")]),t._v(" "),i("div",{staticStyle:{width:"960px"}},[i("PictureTable",{key:"pictable3",staticClass:"center-horizon",staticStyle:{height:"100%",width:"960px","margin-bottom":"20px"},attrs:{"table-id":"table3",header:!0,headerRow:!0,"have-border":!0,content:t.DimReducitonResult,"single-output":!0,cellWidth:t.dimCellWidth,cellHeight:t.chartHeight+"px"}}),t._v(" "),i("div",{staticClass:"describe",staticStyle:{width:"960px","line-height":"30px"}},[t._v("通过观察降维分布，可以观察到正常样本（图中蓝色区域）和对抗样本（图中红色区域）在空间分布中存在较大差异，\n                      说明对抗样本尽管在人眼看来难以区分，但在数字域中是明显可分的，进一步支撑了对抗样本可通过维度差异进行检测过滤的可行性。")])],1)]):t._e(),t._v(" "),i("a-button",{staticStyle:{width:"160px",height:"40px","margin-bottom":"30px","margin-top":"10px","font-size":"18px",color:"white","background-color":"rgb(46, 56, 245)","border-radius":"8px"},on:{click:function(e){return t.getPdf()}}},[t._v("\n                    导出报告内容\n                  ")])]})],2)]):t._e()])],1)},staticRenderFns:[]};var D=s("VU/8")(S,w,!1,function(t){s("/ASi")},"data-v-26e40032",null).exports,F=s("hcOA"),E=s("0zDd"),P=s("7wpf"),L={name:"robustFormalVerification",components:{navmodule:d.a,func_introduce:l.a,showLog:c.a,resultDialog:h.a,DataSetCard:F.a,ModelCard:E.a,MethodCard:P.a,ExMethodEval:D},data:function(){return{radioStyle:{display:"block",lineHeight:"30px"},layerchecked:!1,buttonBGColor:{background:"#0B55F4",color:"#FFFFFF"},disStatus:!1,logflag:!1,percent:10,logtext:[],funcDesText:{name:"攻击机理分析",imgpath:p.a,bgimg:v.a,destext:"通过解释算法对样本进行分析，可视化展示对抗样本攻击效果",backinfo:"通过研究各类对抗性攻击的生成与作用机理，即其误导模型的内在原因，并建立易于人类理解的解释方法，为安全性验证与防御策略构建等工作提供理论指导。",highlight:["特征归因可视化：通过解释算法计算模型在正常样本和对抗样本上的显著图，并做可视化标注处理及展示；进一步通过相关系数量化显著区域的差异性，以此说明对抗噪声通过影响模型关注的特征进而影响分类决策","数据分布降维可视化：将对抗样本和正常样本同时输入到降维方法中，将其同时映射到同一平面中并做可视化处理，进而分析其空间分布的差异，以此说明对抗样本在分布上与正常样本差距较大，可以通过一些分布检测算法进行过滤","模型内部特征分析可视化：根据模型输出层来分析模型内部（各个隐藏层）上特征所贡献的大小说明对抗噪声通过影响模型关注的特征进而影响分类决策"]},result:{},tid:"",stidlist:{},clk:"",logclk:"",dataSetInfo:[{name:"ImageNet",description:"是ILSVRC竞赛使用的是数据集，由斯坦福大学李飞飞教授主导，包含了超过1400万张全尺寸的有标记图片，大约有22000个类别的数据。",classname:["数字0","数字1","数字2","数字3","数字4","数字5","数字6","数字7","数字8","数字9"],pictureSrcs:[[s("95eC"),s("9pcm"),s("S5lj"),s("Xe0k"),s("2XIm"),s("Sisf"),s("HL68"),s("wiub"),s("HGwa"),s("95eC")]]},{name:"CIFAR10",classname:["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"],description:"是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图 片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。",pictureSrcs:[[s("HiaR"),s("DJt5"),s("S99w"),s("J598"),s("/pRs"),s("YuLH"),s("Nvyw"),s("lB35"),s("dKp5"),s("NgyD")]]},{name:"MNIST",description:"是一个手写体数字的图片数据集，该数据集来由美国国家标准与技术研究所（National Institute of Standards and Technology (NIST)）发起整理，一共统计了来自250个不同的人手写数字图片，其中50%是高中生，50%来自人口普查局的工作人员。该数据集的收集目的是希望通过算法，实现对手写数字的识别。",classname:["数字0","数字1","数字2","数字3","数字4","数字5","数字6","数字7","数字8","数字9"],pictureSrcs:[[s("wlDy"),s("KdR5"),s("2LXz"),s("3rtV"),s("Ppgw"),s("P9ea"),s("ec3M"),s("wAAx"),s("30PV"),s("v3v+")]]}],selectedDataset:0,modelInfo:[{name:"VGG",subset:["vgg11","vgg13","vgg16","vgg19"]},{name:"ResNet",subset:["ResNet18","ResNet34","ResNet50","ResNet101","ResNet152"]},{name:"DenseNet",subset:["DenseNet121","DenseNet161","DenseNet169","DenseNet201"]}],selectedModel:0,subModel:0,methodInfo:[[{name:"FGSM",description:"Fast Gradient Sign MethodFGSM快速梯度符号法是一种简单而有效的生成对抗样本的方法，其工作方式如下：在给定输入数据后，利用已训练的模型输出预测并计算损失函数的梯度，然后使用梯度的符号来创建使损失最大化的新数据。"},{name:"RFGSM",description:"Random Fast Gradient Sign MethodR-FGSM随机快速梯度符号法是FGSM的一种变体，在应用FGSM产生的对抗扰动之前，给在输入样本中增加一个小的随机扰动，这样有助于避免梯度Mask的防御策略。"},{name:"FFGSM",description:"FGSM in fast adversarial trainingFFGSM快速FGSM是FGSM的一种变体，在应用FGSM产生的对抗扰动前，给在输入样本中增加一个小的随机扰动。与R-FGSM不同的是，扰动以均匀分布代替高斯分布。"},{name:"MIFGSM",description:"Momentum Iterative Fast Gradient Sign MethodMI-FGSM基于动量的迭代FGSM是在I-FGSM的基础上，通过将动量项整合到攻击的迭代过程中，使计算结果能摆脱局部最优，并且增加了更新方向的稳定性。"}],[{name:"DIFGSM",description:"Diverse Inputs Iterative Fast Gradient Sign Method(DI2-FGSM)输入多样的迭代FGSM是在I-FGSM的基础上，让输入图片随机进行数据增强，即在样本输入模型之前，以一定概率p对其进行随机resize，padding等操作。该方法增强了攻击的鲁棒性，使得黑盒攻击成功率显著增加。"},{name:"BIM",description:"Basic Iterative MethodBIM迭代式FGSM是FGSM方法的变体，其工作原理为：每轮迭代在上一步算得的对抗样本基础上，各像素增加（或减少）一个常数。"},{name:"EOTPGD",description:"Expectation Over Transformation based PGDEOTPGD变换期望PGD是将EOT的思想加入到PGD算法中，即迭代中用损失函数梯度的期望代替符号梯度本身。"},{name:"PGD",description:"Projected Gradient DescentPGD投影梯度下降法是FGSM的迭代版本，与BIM不同的是，它对每次迭代的结果进行裁剪，保证新样本的各个像素都在x的ϵ邻域（L∞）内。"},{name:"PGDL2",description:"L2-bounded PGDPGDL2L2范数投影梯度下降法是PGD算法的另一个版本，其对每次的迭代结果采用L2范数裁剪。"}]],selectedMethods:[],featureMethodInfo:[[{name:"LRP",id:"lrp",description:"LRP算法：通过自定义模型反向传输规则，将模型决策的结果归因到样本像素中，进而可视化其中对分类有利和对分类有害的显著区域。"},{name:"Grad-CAM",id:"gradcam",description:"Grad-CAM算法：通过解构模型特征提取层的特征图（feature map），提取对模型当前决策起到积极影响的区域。"},{name:"Integrated Grad",id:"integrated_grad",description:"Integrated Grad算法：积分梯度方法，通过对基线到输入进行模型梯度的积分，提升归因质量。"}]],selectedFeatureMethod:[],featureMethodDescription:"",featureMethodHoverIndex:-1,dimensionMethodInfo:[[{name:"PCA",id:"pca",description:"PCA算法：主成分分析（Principal Component Analysis，PCA）是一种数学方法，用于将高维数据转化为低维表示，同时保留数据中最重要的信息。它通过寻找数据中的主要变化方向（主成分），将原始特征转化为这些主成分的线性组合，从而实现数据的降维和压缩。"},{name:"t-SNE",id:"tsne",description:"t-SNE算法：非线性降维算法，非常适用于高维数据降维到2维或者3维，并进行可视化。"},{name:"SVN",id:"svm",description:"SVN算法：采用支持向量机来拟合样本分布，并计算样本到SVM边界的距离和样本数据之间的关系，分析样本的分布情况。"}]],selectedDimensionMethod:[],dimensionMethodDescription:"",dimensionMethodHoverIndex:-1,methodDescription:"",methodHoverIndex:-1,resultVisible:!1,postData:{},uploadflag:!1}},watch:{resultVisible:{immediate:!0,handler:function(t){t?this.noScroll():this.canScroll()}}},created:function(){document.title="攻击机理分析"},beforeDestroy:function(){this.clk&&window.clearInterval(this.clk),this.logclk&&window.clearInterval(this.logclk)},methods:{layerExplainChange:function(t){this.layerchecked=!this.layerchecked,t.target.blur()},closeDialog:function(){this.resultVisible=!1},getData:function(){var t=this;t.$axios.get("/output/Resultdata",{params:{Taskid:t.tid}}).then(function(e){console.log("dataget:",e),t.result=e})},getLog:function(){var t=this;t.percent<99&&(t.percent+=1),t.$axios.get("/Task/QueryLog",{params:{Taskid:t.tid}}).then(function(e){if("{}"==r()(t.stidlist))t.logtext=[n()(e.data.Log).slice(-1)[0]];else for(var s in t.logtext=[],t.stidlist)t.logtext.push(e.data.Log[t.stidlist[s]])})},stopTimer:function(){1==this.result.data.stop&&(this.logflag=!1,window.clearInterval(this.clk),window.clearInterval(this.logclk),this.result=this.result.data.result,this.result.tid=this.tid,this.result.stidlist=this.stidlist,this.resultVisible=!0)},update:function(){this.getData();try{this.stopTimer()}catch(t){}},uploadModel:function(t){"uploading"!==t.file.status&&console.log(t.file,t.fileList),"done"===t.file.status?(this.$message.success(t.file.name+" file uploaded successfully"),this.uploadflag=!0):"error"===t.file.status&&this.$message.error(t.file.name+" file upload failed.")},initParam:function(){this.logtext=[],this.percent=0,this.postData={},this.result={},this.tid="",this.stidlist={},""!=this.clk&&(window.clearInterval(this.clk),this.clk=""),""!=this.logclk&&(window.clearInterval(this.logclk),this.logclk="")},dataEvaClick:function(){this.initParam();var t=this.dataSetInfo[this.selectedDataset].name,e=this.modelInfo[this.selectedModel].subset[this.subModel];if(0!=this.selectedMethods.length)if(0!=this.selectedFeatureMethod.length||0!=this.selectedDimensionMethod.length){this.logtext=[],this.logflag=!0;var s=this;s.$axios.post("/Task/CreateTask",{AttackAndDefenseTask:0}).then(function(i){console.log(i),s.tid=i.data.Taskid,s.logclk=window.setInterval(function(){s.getLog()},2e4),s.postData.Taskid=s.tid,s.postData.DatasetParam={name:t},s.postData.ModelParam={name:e,ckpt:null},s.postData.AdvMethods=s.selectedMethods,s.selectedDimensionMethod.length>0&&(s.postData.VisMethods=s.selectedDimensionMethod,s.$axios.post("/Attack/AttackDimReduciton",s.postData).then(function(t){s.stidlist.dimention=t.data.stid,s.clk=window.setInterval(function(){s.update()},6e4)}).catch(function(t){console.log(t),window.clearInterval(s.logclk)})),s.selectedFeatureMethod.length>0&&(s.postData.ExMethods=s.selectedFeatureMethod,s.postData.Use_layer_explain=s.layerchecked,s.$axios.post("/Attack/AttackAttrbutionAnalysis",s.postData).then(function(t){s.stidlist.feature=t.data.stid,""==s.clk&&(s.clk=window.setInterval(function(){s.update()},6e4))}).catch(function(t){console.log(t),window.clearInterval(s.logclk)}))}).catch(function(t){console.log(t)})}else this.$message.warning("请至少选择一项特征归因可视化算法或数据分布降维可视化算法！",3);else this.$message.warning("请至少选择一项攻击算法！",3)},changeDataset:function(t){this.selectedDataset=t},changeModel:function(t,e){this.selectedModel=t,this.subModel=e,console.log(e)},changeMethods:function(t,e){var s=document.getElementById("button"+t+e);""==s.style.color?(this.methodHoverIndex=t,this.methodDescription=this.methodInfo[t][e].description,s.style.color="#0B55F4",s.style.borderColor="#C8DCFB",s.style.background="#F2F4F9",this.selectedMethods.push(this.methodInfo[t][e].name)):(this.methodHoverIndex=-1,this.methodDescription="",s.style.color="",s.style.borderColor="#C8DCFB",s.style.background="#F2F4F9",s.blur(),this.selectedMethods.splice(this.selectedMethods.indexOf(this.methodInfo[t][e].name),1))},methodButtonOver:function(t,e){this.methodHoverIndex=t,this.methodDescription=this.methodInfo[t][e].description},changeFeatureMethods:function(t,e){var s=document.getElementById("feature"+t+e);""==s.style.color?(this.featureMethodHoverIndex=t,this.featureMethodDescription=this.featureMethodInfo[t][e].description,s.style.color="#0B55F4",s.style.borderColor="#C8DCFB",s.style.background="#F2F4F9",this.selectedFeatureMethod.push(this.featureMethodInfo[t][e].id)):(this.featureMethodHoverIndex=-1,this.featureMethodDescription="",s.style.color="",s.style.borderColor="#C8DCFB",s.style.background="#F2F4F9",s.blur(),this.selectedFeatureMethod.splice(this.selectedFeatureMethod.indexOf(this.featureMethodInfo[t][e].id),1),0==this.selectedFeatureMethod.length&&(this.layerchecked=!1))},featureMethodButtonOver:function(t,e){this.featureMethodHoverIndex=t,this.featureMethodDescription=this.featureMethodInfo[t][e].description},changeDimensionMethods:function(t,e){var s=document.getElementById("dimension"+t+e);""==s.style.color?(this.dimensionMethodHoverIndex=t,this.dimensionMethodDescription=this.dimensionMethodInfo[t][e].description,s.style.color="#0B55F4",s.style.borderColor="#C8DCFB",s.style.background="#F2F4F9",this.selectedDimensionMethod.push(this.dimensionMethodInfo[t][e].id)):(this.dimensionMethodHoverIndex=-1,this.dimensionMethodDescription="",s.style.color="",s.style.borderColor="#C8DCFB",s.style.background="#F2F4F9",s.blur(),this.selectedDimensionMethod.splice(this.selectedDimensionMethod.indexOf(this.dimensionMethodInfo[t][e].id),1))},dimensionMethodButtonOver:function(t,e){this.dimensionMethodHoverIndex=t,this.dimensionMethodDescription=this.dimensionMethodInfo[t][e].description}}},R={render:function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",[s("a-layout",[s("a-layout-header",[s("navmodule")],1),t._v(" "),s("a-layout-content",[s("func_introduce",{attrs:{funcDesText:t.funcDesText}}),t._v(" "),s("div",{staticClass:"paramCon"},[s("h2",{staticClass:"subTitle",staticStyle:{"margin-top":"-96px"}},[t._v("参数配置")]),t._v(" "),s("div",{staticClass:"funcParam"},[s("div",{staticClass:"paramTitle"},[s("img",{staticClass:"paramIcom",attrs:{src:t.funcDesText.imgpath,alt:t.funcDesText.name}}),t._v(" "),s("h3",[t._v(t._s(t.funcDesText.name))]),t._v(" "),s("a-button",{staticClass:"DataEva",style:t.buttonBGColor,attrs:{disabled:t.disStatus},on:{click:t.dataEvaClick}},[s("a-icon",{attrs:{type:"security-scan"}}),t._v("\n                       评估\n                   ")],1)],1),t._v(" "),s("a-divider"),t._v(" "),s("div",{staticClass:"inputdiv"},[s("div",{staticClass:"mainParamNameNotop"},[t._v("请选择数据集")]),t._v(" "),t._l(t.dataSetInfo,function(e,i){return s("DataSetCard",t._b({key:"Dataset"+i,staticStyle:{width:"1104px","margin-bottom":"16px"},attrs:{indexInParent:i,checked:i==t.selectedDataset},on:{selectDataset:t.changeDataset}},"DataSetCard",e,!1))}),t._v(" "),s("div",{staticClass:"mainParamName48"},[t._v("请选择模型")]),t._v(" "),s("div",{staticStyle:{width:"1104px","margin-bottom":"16px"}},[s("a-upload",{attrs:{action:"/ex/uploadModel",name:"ex_upload_model"},on:{change:t.uploadModel}},[s("div",{staticClass:"uploadModelStyle"},[t._v("请上传模型")])])],1),t._v(" "),t._l(t.modelInfo,function(e,i){return s("ModelCard",t._b({key:"Model"+i,staticStyle:{width:"1104px","margin-bottom":"16px"},attrs:{indexInParent:i,checked:i==t.selectedModel},on:{selectModel:t.changeModel}},"ModelCard",e,!1))}),t._v(" "),s("div",{staticClass:"mainParamName48"},[t._v("请选择攻击方法（可多选）")]),t._v(" "),t._l(t.methodInfo,function(e,i){return s("a-row",{key:"attack"+i,staticStyle:{"margin-top":"16px"},attrs:{gutter:16,type:"flex"}},[t._l(e,function(a,n){return s("a-col",{key:n,attrs:{flex:24/e.length}},[s("a-button",{staticClass:"methodButton",attrs:{id:"button"+i+n},on:{click:function(e){return t.changeMethods(i,n)},mouseover:function(e){return t.methodButtonOver(i,n)}}},[t._v(t._s(a.name))])],1)}),t._v(" "),t.methodHoverIndex==i?s("div",{staticClass:"attackmethodDes"},[t._v(" "+t._s(t.methodDescription)+" ")]):t._e()],2)}),t._v(" "),s("div",{staticClass:"mainParamName48"},[t._v("请选择特征归因可视化算法（可多选）")]),t._v(" "),t._l(t.featureMethodInfo,function(e,i){return s("a-row",{key:"feature"+i,staticStyle:{"margin-top":"16px"},attrs:{gutter:16,type:"flex"}},[t._l(e,function(a,n){return s("a-col",{key:n,attrs:{flex:24/e.length}},[s("a-button",{staticClass:"methodButton",attrs:{id:"feature"+i+n},on:{click:function(e){return t.changeFeatureMethods(i,n)},mouseover:function(e){return t.featureMethodButtonOver(i,n)}}},[t._v(t._s(a.name))])],1)}),t._v(" "),t.featureMethodHoverIndex==i?s("div",{staticClass:"attackmethodDes"},[t._v(" "+t._s(t.featureMethodDescription)+" ")]):t._e()],2)}),t._v(" "),s("div",{staticClass:"mainParamName48"},[t._v("请选择数据分布降维可视化算法（可多选）")]),t._v(" "),t._l(t.dimensionMethodInfo,function(e,i){return s("a-row",{key:"dimension"+i,staticStyle:{"margin-top":"16px"},attrs:{gutter:16,type:"flex"}},[t._l(e,function(a,n){return s("a-col",{key:n,attrs:{flex:24/e.length}},[s("a-button",{staticClass:"methodButton",attrs:{id:"dimension"+i+n},on:{click:function(e){return t.changeDimensionMethods(i,n)},mouseover:function(e){return t.dimensionMethodButtonOver(i,n)}}},[t._v(t._s(a.name))])],1)}),t._v(" "),t.dimensionMethodHoverIndex==i?s("div",{staticClass:"attackmethodDes"},[t._v(" "+t._s(t.dimensionMethodDescription)+" ")]):t._e()],2)}),t._v(" "),s("div",{staticClass:"mainParamName48"},[t._v("请选择模型内部特征分析可视化算法")]),t._v(" "),s("a-radio",{style:t.radioStyle,attrs:{checked:t.layerchecked,disabled:0==t.selectedFeatureMethod.length},on:{click:t.layerExplainChange}},[t._v("\n                       Guided-backpropagation\n                   ")]),t._v(" "),s("ExMethodEval",{attrs:{isShow:t.resultVisible,result:t.result,postData:t.postData},on:{"on-close":function(){t.resultVisible=!t.resultVisible}}})],2)],1)]),t._v(" "),t.logflag?s("div",[s("showLog",{attrs:{percent:t.percent,logtext:t.logtext}})],1):t._e()],1),t._v(" "),s("a-layout-footer")],1)],1)},staticRenderFns:[]};var G=s("VU/8")(L,R,!1,function(t){s("sLBS")},"data-v-7f7b95c8",null);e.default=G.exports},HGwa:function(t,e,s){t.exports=s.p+"static/img/ImageNet9.6030904.png"},HL68:function(t,e,s){t.exports=s.p+"static/img/ImageNet7.d3606ae.png"},JIup:function(t,e){},S5lj:function(t,e,s){t.exports=s.p+"static/img/ImageNet3.b5ad0fc.png"},Sisf:function(t,e,s){t.exports=s.p+"static/img/ImageNet6.f051da8.png"},Xe0k:function(t,e,s){t.exports=s.p+"static/img/ImageNet4.a12f4e6.png"},nTcO:function(t,e){t.exports="data:image/png;base64,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"},sLBS:function(t,e){},"v/51":function(t,e){t.exports="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAMKADAAQAAAABAAAAMAAAAADbN2wMAAAC7ElEQVRoBe2Zz2sTQRTH35tEFFS8FKF4sUIOHvQiHooeFI8mrfQQ0LPgD/CktklDPBiCbawIxYvgHyA5GVOlCOJJehAvHvQgaDE3UdFDQDDZ50y6b7qJ28ad7CTRbi5v5s3Me5/vvNndsAsQ/aId2No7gLbk37y1dCQmmo9U/KYTO3Mjm3xtI5cVAcXblcPk4HMJPOJCf0FBp3LXJ9+ELSJ0AT7wzGxFRKgCNoG3JiI0AX8Bb0WE4Ki9WB/4GhA80zHX2jW3P6KuD7VGj/fQ6FmAH3zMEScAYVVzyXbLBxC6iJ4EbASfzSY/aHi3oXw2RBgLCALPYmyIMBJgAm9LRGABvcDbEBFIwOLi0+3kiGUJwk/YmjrXfmeeYTey/sdJLKscG63x8wcSUK87+wBo1A1kDM8gf4qg0bUcPKO7DSRAJSSCiwB4L0543GTnO5FUDBVLxUSEC0FjxjsDduvnMxP3u80JOp7JpD7JNVeCrlPzA1XAJIHtNZEA2zvcLX5UgW47ZHv8n6+Avo0WS9XzRDRNgI/zM6lrtnfOJH5hvrqAQBOIWMpNpx6oGLoCEv6u7CfkhKuFUjVhksDmGsWk2BSjy9pKpwXI3i4GkM7d3B4W28HkZR0WRDMObwXMIgx4VSRgwAVYvwsNGsQ0v70jhPCNoeSz5Su3w7b6QRZ2YNFoLFB82x4VVzR+3Qk7PsezJmB2dkrt+mVOZMvaO0K2iDvi/lcCvrM4Ahrj9rBYh+CAh0WzeiqAb3kCAUxxe1is/CPnYVpn1QII6SHDIsG5wlzlEvcHbV2Ws8zRxsrOcrkce7+6Y0W+9znKPlmJl0j0ggR+Zl8/LTq0lxBPyq8wxzivfHf0KrH/53g6nW4qX9sXmsL80kFZqhX59q11/+ZFw2Pxh3wojudnku+YSR8h5WgNxJuH2r6u8MxBW/WVR7J54RVSWwW8jMW5ymkQmCIHxghhp3esX215LdZRwEdwqJrLTD7pV94oz5bagd+/ujwh3cjnYgAAAABJRU5ErkJggg=="},wiub:function(t,e,s){t.exports=s.p+"static/img/ImageNet8.dc475fd.png"}});