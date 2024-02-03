webpackJsonp([25],{Quw4:function(e,s,r){"use strict";Object.defineProperty(s,"__esModule",{value:!0});var o=r("pFYg"),t=r.n(o),a=r("I+x0"),n=r("L/O1"),i={name:"login",components:{loginDialog:a.a},data:function(){return{labelCol:{span:4},wrapperCol:{span:20},userinfo:{username:"",password:""},registerinfo:{newUsername:"",newPassword:""},regrules:{newUsername:[{required:!0,message:"请输入用户名",trigger:"blur"},{min:3,max:5,message:"字符长度3 到8 位",trigger:"blur"}],newPassword:[{required:!0,message:"请输入密码",trigger:"blur"},{validator:function(e,s,r){/^(?![\d]+$)(?![a-zA-Z]+$)(?![_]+$)[\da-zA-Z_]{6,20}$/.test(s)?r():r(new Error("数字、字母、下划线任意两种组合，且不能少于6位大于20位"))},trigger:"blur"}]},showLogin:!0,showRegister:!1}},mounted:function(){this.username=Object(n.a)("username"),""!=this.username&&this.$router.push("/homme_menu")},methods:{login:function(){var e=this;if(""==this.userinfo.username||""==this.userinfo.password)return this.$message.error("请输入用户名或者密码"),-1;var s=new URLSearchParams;s.append("username",this.userinfo.username),s.append("password",this.userinfo.password),console.log(s),this.$axios.post("/login",s).then(function(s){console.log(s.data),console.log(t()(s.data)),-1==s.data.code?e.$message.error("用户名密码错误"):1==s.data.code?(e.$message.success("登录成功"),Object(n.b)("username",e.userinfo.username,6e4),setTimeout(function(){this.$router.push("/homme_menu")}.bind(e),1e3)):e.$message.error("未知错误")})},ToRegister:function(){this.showRegister=!0,this.showLogin=!1},ToLogin:function(){this.showRegister=!1,this.showLogin=!0},register:function(){var e=this;if(""==this.registerinfo.newUsername||""==this.registerinfo.newPassword)return this.$message.error("请输入用户名或者密码"),-1;var s=new URLSearchParams;s.append("username",this.registerinfo.newUsername),s.append("password",this.registerinfo.newPassword),this.$axios.post("/register",s).then(function(s){console.log(s),console.log(s.data),1==s.data.code?(e.$message.success("注册成功"),e.registerinfo.newUsername="",e.registerinfo.newPassword="",setTimeout(function(){this.showRegister=!1,this.showLogin=!0}.bind(e),1e3)):-1==s.data.code&&e.$message.error("用户名已存在")})}}},l={render:function(){var e=this,s=e.$createElement,r=e._self._c||s;return r("div",{staticClass:"dialog"},[r("div",{staticClass:"content"},[r("div",{directives:[{name:"show",rawName:"v-show",value:e.showLogin,expression:"showLogin"}],staticClass:"login-wrap"},[r("h1",[e._v("系统登录")]),e._v(" "),r("a-form-model",{attrs:{model:e.userinfo,"label-col":e.labelCol,"wrapper-col":e.wrapperCol},on:{submit:e.login},nativeOn:{submit:function(e){e.preventDefault()}}},[r("a-form-model-item",{attrs:{label:"用户名"}},[r("a-input",{attrs:{size:"large",placeholder:"请输入用户名"},model:{value:e.userinfo.username,callback:function(s){e.$set(e.userinfo,"username",s)},expression:"userinfo.username"}},[r("a-icon",{staticStyle:{color:"rgba(0,0,0,.25)"},attrs:{slot:"prefix",type:"user"},slot:"prefix"})],1)],1),e._v(" "),r("a-form-model-item",{attrs:{label:"密码"}},[r("a-input",{attrs:{size:"large",type:"password",placeholder:"请输入密码"},model:{value:e.userinfo.password,callback:function(s){e.$set(e.userinfo,"password",s)},expression:"userinfo.password"}},[r("a-icon",{staticStyle:{color:"rgba(0,0,0,.25)"},attrs:{slot:"prefix",type:"lock"},slot:"prefix"})],1)],1),e._v(" "),r("a-form-model-item",{attrs:{wrapperCol:{span:24}}},[r("a-button",{attrs:{type:"primary","html-type":"submit",size:"large",disabled:""===e.userinfo.username||""===e.userinfo.password}},[e._v("\n                        登录\n                    ")])],1)],1),e._v(" "),r("span",{on:{click:e.ToRegister}},[e._v("没有账号?马上注册")])],1),e._v(" "),r("div",{directives:[{name:"show",rawName:"v-show",value:e.showRegister,expression:"showRegister"}],staticClass:"register-wrap"},[r("h1",[e._v("系统注册")]),e._v(" "),r("a-form-model",{attrs:{model:e.registerinfo,rules:e.regrules,"label-col":e.labelCol,"wrapper-col":e.wrapperCol},on:{submit:e.register},nativeOn:{submit:function(e){e.preventDefault()}}},[r("a-form-model-item",{ref:"name",attrs:{label:"用户名",prop:"newUsername"}},[r("a-input",{attrs:{size:"large",placeholder:"请输入用户名"},on:{blur:function(){e.$refs.name.onFieldBlur()}},model:{value:e.registerinfo.newUsername,callback:function(s){e.$set(e.registerinfo,"newUsername",s)},expression:"registerinfo.newUsername"}},[r("a-icon",{staticStyle:{color:"rgba(0,0,0,.25)"},attrs:{slot:"prefix",type:"user"},slot:"prefix"})],1)],1),e._v(" "),r("a-form-model-item",{attrs:{label:"密码",prop:"newPassword"}},[r("a-input",{attrs:{size:"large",type:"password",placeholder:"请输入密码"},model:{value:e.registerinfo.newPassword,callback:function(s){e.$set(e.registerinfo,"newPassword",s)},expression:"registerinfo.newPassword"}},[r("a-icon",{staticStyle:{color:"rgba(0,0,0,.25)"},attrs:{slot:"prefix",type:"lock"},slot:"prefix"})],1)],1),e._v(" "),r("a-form-model-item",{attrs:{wrapperCol:{span:24}}},[r("a-button",{attrs:{type:"primary",size:"large","html-type":"submit",disabled:""===e.registerinfo.newUsername||""===e.registerinfo.newPassword}},[e._v("\n                        注册\n                    ")])],1)],1),e._v(" "),r("span",{on:{click:e.ToLogin}},[e._v("已有账号?马上登录")])],1)])])},staticRenderFns:[]};var u=r("VU/8")(i,l,!1,function(e){r("b9oD")},"data-v-6993c0e8",null);s.default=u.exports},b9oD:function(e,s){}});