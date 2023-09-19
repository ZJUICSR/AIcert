#ifndef _CIFAR10_NN_EXAMPLE_H
#define _CIFAR10_NN_EXAMPLE_H
#include <stdint.h>
#include "../side_channel_attack_methods.h"

// #include "arm_math.h"

//定义train/test两个函数


int8_t* cifar10_nn_run_cpa_dpa(Parameters*);
int8_t* cifar10_nn_run_hpa(Parameters*);


#endif