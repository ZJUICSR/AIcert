// 页面跳转js

// 任务介绍页
function open_function_introduction() {
    window.open('/index_function_introduction', '_self');
};

// 任务中心页
function open_task_center() {
    window.open('/index_task_center', '_self');
};

// 新建任务页
function open_params_1() {
    storage.clear();
    window.open('/index_params_1', '_self');
};
function open_params_2() {
window.open('/index_params_2', '_self');
};

// 任务中心-->任务结果
function open_task_results(tid) {
    console.log("open_task_results",tid);
    window.open('/index_results?tid='+tid, '_self');
};

// 轮播图功能按钮
function goto_adv_attack() {
    // 预留界面跳转
    window.open('/index_params_1', '_self');
};

function goto_robust_enhance() {
    // 预留界面跳转
    window.open('/ModelRobust', '_self');
};

function goto_fairness() {
    // 预留界面跳转
    window.open('/Fairness', '_self');
};

function goto_formal_verification() {
    // 预留界面跳转
    window.open('/index_params_1', '_self');
};

function goto_bkd_attack() {
    // 预留界面跳转
    window.open('/index_params_1', '_self');
};

function goto_old_version() {
    window.open('http://10.15.201.88:14581/get_adv_input', '_self')
};