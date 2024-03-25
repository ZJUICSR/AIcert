// 轮播图js
window.onload = function() {
    var swiper = new Swiper('.swiper-container',{
		autoplay:3000,
		speed:1000,
        spaceBetween: -550,
		autoplayDisableOnInteraction : false,
		loop:true,
		centeredSlides : true,
		slidesPerView: 2,
        // 底部小圆点
        pagination : '.swiper-pagination',
		paginationClickable:true,
		prevButton:'.swiper-button-prev',
        nextButton:'.swiper-button-next',
		onInit:function(swiper){
			swiper.slides[2].className="swiper-slide swiper-slide-active";//第一次打开不要动画
			},
        breakpoints: { 
                668: {
                    slidesPerView: 1,
                 }
            }
		});
		}

// 轮播图功能按钮
function goto_concolic() {
    // 测试样本自动生成
    window.open('/index_function_introduction#/concolic', '_self');
};

function goto_dataFairnessEva() {
    // 数据集公平性评估
    window.open('/index_function_introduction#/dataFairnessEva', '_self');
};

function goto_dataFairnessDebias() {
    // 数据集公平性提升
    window.open('/index_function_introduction#/dataFairnessDebias', '_self');
};

function goto_dataClean() {
    //  异常数据检测
    window.open('/index_function_introduction#/dataClean', '_self');
};

function goto_modelMeasure() {
    // 跳转AI模型安全度量
    window.open('/index_function_introduction#/modelMeasure', '_self');
};

function goto_adv_attack() {
    // 对抗攻击评估
    window.open('/index_function_introduction#/advAttack', '_self');
};

function goto_bkd_attack() {
    // 后门攻击评估
    window.open('/index_function_introduction#/backdoor', '_self');
};

function goto_autoattack() {
    // 模型对抗性测试
    window.open('/index_function_introduction#/autoAttack', '_self');
};

function goto_modelFairnessEva() {
    // 模型公平性评估
    window.open('/index_function_introduction#/modelFairnessEva', '_self');
};

function goto_formal_verification() {
    // 形式化验证
    window.open('/index_function_introduction#/FormalVerfy', '_self');
};

function goto_advAttackDefense() {
    // 对抗攻击防御
    window.open('/index_function_introduction#/advAttackDefense', '_self');
};

function goto_backdoorDefense() {
    // 后门攻击防御
    window.open('/index_function_introduction#/backdoorDefense', '_self');
};

function goto_robust_advTraining() {
    // 鲁棒性训练
    window.open('/index_function_introduction#/robust_advTraining', '_self');
};

function goto_modelFairnessDebias() {
    // 模型公平性提升
    window.open('/index_function_introduction#/modelFairnessDebias', '_self');
};

function goto_crowdDefense() {
    // 模型群智化防御
    window.open('/index_function_introduction#/crowdDefense', '_self');
};

function goto_modularDevelop() {
    // AI模型模块化开发
    window.open('/index_function_introduction#/modularDevelop', '_self');
};

function goto_coverage_neural() {
    // 标准化单元测试
    window.open('/index_function_introduction#/coverage_neural', '_self');
};

function goto_exMethod() {
    // 攻击机理分析
    window.open('/index_function_introduction#/exMethod', '_self');
};

function goto_frameworkTest() {
    // 开发框架安全结构度量
    window.open('/index_function_introduction#/frameworkTest', '_self');
};

function goto_envTest() {
    // 开发环境分析与框架适配
    window.open('/index_function_introduction#/envTest', '_self');
};

function goto_side() {
    // 侧信道分析
    window.open('/index_function_introduction#/side', '_self');
};

function goto_inject() {
    // 故障注入检测
    window.open('/index_function_introduction#/inject', '_self');
};

function goto_hardOpt() {
    // 基于硬件优化的软硬件一体化验证
    window.open('/index_function_introduction#/hardOpt', '_self');
};


function goto_old_version() {
    window.open('http://10.15.201.88:14581/get_adv_input', '_self')
}