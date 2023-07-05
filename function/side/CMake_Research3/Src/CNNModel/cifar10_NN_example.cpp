#include <stdio.h>
#include "../../Inc/CNNModel/cnn.h"
#include "../../Inc/CNNModel/cifar10_NN_example.h"
#include "../../Inc/CNNModel/arm_nnexamples_cifar10_parameter.h"
#include "../../Inc/CNNModel/arm_nnexamples_cifar10_weights.h"
#include "../../Inc/CNNModel/arm_nnexamples_cifar10_inputs.h"
#include "../../Inc/side_channel_attack_methods.h"

// #include "..\Inc\cnn.h"
#include <string.h>
#define IMG_LEN 32 * 32 * 32

void print_img(q7_t* img, int len)
{
    for (int i = 0; i < 32; i++)
    {
        for(int j=0;j<32*32;j++){
            printf("%d ",img[32*32*i+j]);
        }
        printf("\n\n");
    }
    return;
}

void print_img2(q7_t* img, int len)
{
    for (int i = 0; i < len; i++)
    {
      
        printf("%d, ",img[i]);
        
        
    }
    return;
}

// include the input and weights
static q7_t conv1_wt[CONV1_IM_CH * CONV1_KER_DIM * CONV1_KER_DIM * CONV1_OUT_CH] = CONV1_WT;
static q7_t conv1_bias[CONV1_OUT_CH]  = CONV1_BIAS;

static q7_t conv2_wt[CONV2_IM_CH * CONV2_KER_DIM * CONV2_KER_DIM * CONV2_OUT_CH] = CONV2_WT;
static q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;

static q7_t conv3_wt[CONV3_IM_CH * CONV3_KER_DIM * CONV3_KER_DIM * CONV3_OUT_CH] = CONV3_WT;
static q7_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;

static q7_t ip1_wt[IP1_DIM * IP1_OUT] = IP1_WT;
static q7_t ip1_bias[IP1_OUT]         = IP1_BIAS;

/* Here the image_data should be the raw uint8 type RGB image in [RGB, RGB, RGB ... RGB] format */
uint8_t image_data[CONV1_IM_CH * CONV1_IM_DIM * CONV1_IM_DIM] = IMG_DATA;

q7_t    output_data[IP1_OUT];

//vector buffer: max(im2col buffer,average pool buffer, fully connected buffer)
q7_t col_buffer[2 * 5 * 5 * 32 * 2];

q7_t scratch_buffer[32 * 32 * 10 * 4];


int8_t* cifar10_nn_run_cpa(Parameters* param){//uint8_t image_data[],q7_t conv1_wt[]
    q7_t* conv1_wt=NULL;
    uint8_t* image_data=NULL;
    
    if(param->getFunctionParameters()!=NULL){
        if(param->getFmapPoint()!=NULL){
            // q7_t* conv1_wt = param->getFmapPoint();
            conv1_wt = param->getFmapPoint();
        }
        if(param->getImageDataPoint()!=NULL){
            // uint8_t* image_data = param->getImageDataPoint();
            image_data = param->getImageDataPoint();
        }
    }

   
    //printf("start_running\r\n");
    /* start the execution */

    q7_t *img_buffer1 = scratch_buffer;
    q7_t *img_buffer2 = img_buffer1 + 32 * 32 * 32;

    /* input pre-processing */
    int          mean_data[3]  = INPUT_MEAN_SHIFT;
    unsigned int scale_data[3] = INPUT_RIGHT_SHIFT;
    for (int i = 0; i < 32 * 32 * 3; i += 3)
    {
        img_buffer2[i]     = (q7_t)__SSAT(((((int)image_data[i] - mean_data[0]) << 7) + (0x1 << (scale_data[0] - 1))) >> scale_data[0], 8);
        img_buffer2[i + 1] = (q7_t)__SSAT(((((int)image_data[i + 1] - mean_data[1]) << 7) + (0x1 << (scale_data[1] - 1))) >> scale_data[1], 8);
        img_buffer2[i + 2] = (q7_t)__SSAT(((((int)image_data[i + 2] - mean_data[2]) << 7) + (0x1 << (scale_data[2] - 1))) >> scale_data[2], 8);
    }

    

    // conv1 img_buffer2 -> img_buffer1
    // trigger_Pin_Pullup(); 
    arm_convolve_HWC_q7_RGB_mid_cpa_dpa(img_buffer2, CONV1_IM_DIM, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
                            CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, img_buffer1, CONV1_OUT_DIM,
                            (q15_t *)col_buffer, NULL, param);
    // trigger_Pin_Nopull();

    // print_img(img_buffer1, IMG_LEN);

    #if 0

    arm_relu_q7(img_buffer1, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH);

    // pool1 img_buffer1 -> img_buffer2
    arm_maxpool_q7_HWC(img_buffer1, CONV1_OUT_DIM, CONV1_OUT_CH, POOL1_KER_DIM,
                       POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, NULL, img_buffer2);

    // conv2 img_buffer2 -> img_buffer1
    arm_convolve_HWC_q7_fast(img_buffer2, CONV2_IM_DIM, CONV2_IM_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM,
                             CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, img_buffer1,
                             CONV2_OUT_DIM, (q15_t *)col_buffer, NULL);

    arm_relu_q7(img_buffer1, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);

    // pool2 img_buffer1 -> img_buffer2
    arm_maxpool_q7_HWC(img_buffer1, CONV2_OUT_DIM, CONV2_OUT_CH, POOL2_KER_DIM,
                       POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, col_buffer, img_buffer2);

    // conv3 img_buffer2 -> img_buffer1
    arm_convolve_HWC_q7_fast(img_buffer2, CONV3_IM_DIM, CONV3_IM_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM,
                             CONV3_PADDING, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, img_buffer1,
                             CONV3_OUT_DIM, (q15_t *)col_buffer, NULL);

    arm_relu_q7(img_buffer1, CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH);

    // pool3 img_buffer-> img_buffer2
    arm_maxpool_q7_HWC(img_buffer1, CONV3_OUT_DIM, CONV3_OUT_CH, POOL3_KER_DIM,
                       POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM, col_buffer, img_buffer2);

    arm_fully_connected_q7_opt(img_buffer2, ip1_wt, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias,
                               output_data, (q15_t *)img_buffer1);

    arm_softmax_q7(output_data, 10, output_data);

    #endif

    #if 0// 将网络执行结果输出
    for (int i = 0; i < 10; i++)
    {
        printf("%d: %d\n", i, output_data[i]);
    }
    #endif

    return output_data;

   

   
}






