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
function goto_adv_attack() {
    // 预留界面跳转
    window.open('/index_function_introduction#/advAttack', '_self');
};

function goto_robust_enhance() {
    // 预留界面跳转
    window.open('/index_function_introduction#/advAttackDefense', '_self');
};

function goto_fairness() {
    // 预留界面跳转
    window.open('/index_function_introduction#/dataFairnessEva', '_self');
};

function goto_formal_verification() {
    // 预留界面跳转
    window.open('/index_function_introduction#/formalVerfy', '_self');
};

function goto_bkd_attack() {
    // 预留界面跳转
    window.open('/index_function_introduction#/backdoor', '_self');
};

function goto_old_version() {
    window.open('http://10.15.201.88:14581/get_adv_input', '_self')
}