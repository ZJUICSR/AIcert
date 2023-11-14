#include <stdio.h>
#include "../../Inc/CNNModel/cnn.h"



void arm_relu_q7(q7_t * data, uint16_t size)
{

#if 0
// #if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    uint16_t  i = size >> 2;
    q7_t     *pIn = data;
    q7_t     *pOut = data;
    q31_t     in;
    q31_t     buf;
    q31_t     mask;

    while (i)
    {
        in = *__SIMD32(pIn)++;

        /* extract the first bit */
        buf = __ROR(in & 0x80808080, 7);

        /* if MSB=1, mask will be 0xFF, 0x0 otherwise */
        mask = __QSUB8(0x00000000, buf);

        *__SIMD32(pOut)++ = in & (~mask);
        i--;
    }

    i = size & 0x3;
    while (i)
    {
        if (*pIn < 0)
        {
            *pIn = 0;
        }
        pIn++;
        i--;
    }

#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    uint16_t  i;
    // mid=data[0];
    // printf("%d,",data[0]);
    for (i = 0; i < size; i++)
    {
        // printf("%d:%d,",i,data[i]);
        if (data[i] < 0)
            data[i] = 0;
    }

#endif                          /* ARM_MATH_DSP */

}