#include <stdio.h>
#include "../../Inc/CNNModel/cnn.h"

void arm_softmax_q7(const q7_t * vec_in, const uint16_t dim_vec, q7_t * p_out)
{
    q31_t     sum;
    int16_t   i;
    uint8_t   shift;
    q15_t     base;
    base = -257;

    /* We first search for the maximum */
    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }

    /* 
     * So the base is set to max-8, meaning 
     * that we ignore really small values. 
     * anyway, they will be 0 after shrinking to q7_t.
     */
    base = base - 8;

    sum = 0;

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base) 
        {
            shift = (uint8_t)__USAT(vec_in[i] - base, 5);
            sum += 0x1 << shift;
        }
    }

    /* This is effectively (0x1 << 20) / sum */
    int output_base = 0x100000 / sum;

    /* 
     * Final confidence will be output_base >> ( 13 - (vec_in[i] - base) )
     * so 128 (0x1<<7) -> 100% confidence when sum = 0x1 << 8, output_base = 0x1 << 12 
     * and vec_in[i]-base = 8
     */
    for (i = 0; i < dim_vec; i++) 
    {
        if (vec_in[i] > base) 
        {
            /* Here minimum value of 13+base-vec_in[i] will be 5 */
            shift = (uint8_t)__USAT(13+base-vec_in[i], 5);
            p_out[i] = (q7_t) __SSAT((output_base >> shift), 8);
        } else {
            p_out[i] = 0;
        }
    }
}




