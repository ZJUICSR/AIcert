#include <stdio.h>
#include "../../Inc/CNNModel/cnn.h"

arm_status
arm_fully_connected_q7_opt(const q7_t * pV,
                           const q7_t * pM,
                           const uint16_t dim_vec,
                           const uint16_t num_of_rows,
                           const uint16_t bias_shift,
                           const uint16_t out_shift, 
                           const q7_t * bias, 
                           q7_t * pOut, 
                           q15_t * vec_buffer)
{
#if 0
// #if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    const q7_t *pB = pM;
    q7_t     *pO = pOut;
    const q7_t *pBias = bias;
    q15_t    *pA;
    uint16_t  rowCnt = num_of_rows >> 2;

    arm_q7_to_q15_reordered_no_shift(pV, vec_buffer, dim_vec);

    while (rowCnt)
    {

        q31_t     sum =  ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum2 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum3 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum4 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

        uint16_t  colCnt = dim_vec >> 2;

        pA = vec_buffer;

#ifdef USE_INTRINSIC

#ifndef ARM_MATH_BIG_ENDIAN
        while (colCnt)
        {
            q31_t     inM11, inM12, inM13, inM14;
            q31_t     inV;

            inV = *__SIMD32(pA)++;
            inM11 = *__SIMD32(pB)++;
            inM12 = __SXTB16(__ROR(inM11, 8));
            inM11 = __SXTB16(inM11);
            sum = __SMLAD(inM11, inV, sum);
            sum2 = __SMLAD(inM12, inV, sum2);
            inM13 = *__SIMD32(pB)++;
            inM14 = __SXTB16(__ROR(inM13, 8));
            inM13 = __SXTB16(inM13);
            sum3 = __SMLAD(inM13, inV, sum3);
            sum4 = __SMLAD(inM14, inV, sum4);

            inV = *__SIMD32(pA)++;
            inM11 = *__SIMD32(pB)++;
            inM12 = __SXTB16(__ROR(inM11, 8));
            inM11 = __SXTB16(inM11);
            sum = __SMLAD(inM11, inV, sum);
            sum2 = __SMLAD(inM12, inV, sum2);
            inM13 = *__SIMD32(pB)++;
            inM14 = __SXTB16(__ROR(inM13, 8));
            inM13 = __SXTB16(inM13);
            sum3 = __SMLAD(inM13, inV, sum3);
            sum4 = __SMLAD(inM14, inV, sum4);
            colCnt--;
        }
#else
        while (colCnt)
        {
            q31_t     inM11, inM12, inM13, inM14;
            q31_t     inV;

            inV = *__SIMD32(pA)++;
            inM11 = *__SIMD32(pB)++;
            inM12 = __SXTB16(__ROR(inM11, 8));
            inM11 = __SXTB16(inM11);
            sum = __SMLAD(inM12, inV, sum);
            sum2 = __SMLAD(inM11, inV, sum2);
            inM13 = *__SIMD32(pB)++;
            inM14 = __SXTB16(__ROR(inM13, 8));
            inM13 = __SXTB16(inM13);
            sum3 = __SMLAD(inM14, inV, sum3);
            sum4 = __SMLAD(inM13, inV, sum4);

            inV = *__SIMD32(pA)++;
            inM11 = *__SIMD32(pB)++;
            inM12 = __SXTB16(__ROR(inM11, 8));
            inM11 = __SXTB16(inM11);
            sum = __SMLAD(inM12, inV, sum);
            sum2 = __SMLAD(inM11, inV, sum2);
            inM13 = *__SIMD32(pB)++;
            inM14 = __SXTB16(__ROR(inM13, 8));
            inM13 = __SXTB16(inM13);
            sum3 = __SMLAD(inM14, inV, sum3);
            sum4 = __SMLAD(inM13, inV, sum4);
            colCnt--;
        }
#endif                          /* ARM_MATH_BIG_ENDIAN */

#else

        /*
         * register needed:
         * loop counter: colCnt
         * accumulators: sum, sum2, sum3, sum4
         * pointers: pB, pA
         * weight data: inM11, inM12, inM13, inM14
         * activation data: inV
         */

#ifndef ARM_MATH_BIG_ENDIAN
        asm volatile ("COL_LOOP_%=:\n"
                      "ldr.w r4, [%[pA]], #8\n"
                      "ldr.w r1, [%[pB]], #16\n"
                      "mov.w r0, r1, ror #8\n"
                      "sxtb16 r0, r0\n"
                      "sxtb16 r1, r1\n"
                      "smlad %[sum], r4, r1, %[sum]\n"
                      "smlad %[sum2], r4, r0, %[sum2]\n"
                      "ldr.w r3, [%[pB], #-12]\n"
                      "mov.w r2, r3, ror #8\n"
                      "sxtb16 r2, r2\n"
                      "sxtb16 r3, r3\n"
                      "smlad %[sum3], r4, r3, %[sum3]\n"
                      "smlad %[sum4], r4, r2, %[sum4]\n"
                      "ldr.w r4, [%[pA], #-4]\n"
                      "ldr.w r1, [%[pB], #-8]\n"
                      "mov.w r0, r1, ror #8\n"
                      "sxtb16 r0, r0\n"
                      "sxtb16 r1, r1\n"
                      "smlad %[sum], r4, r1, %[sum]\n"
                      "smlad %[sum2], r4, r0, %[sum2]\n"
                      "ldr.w r3, [%[pB], #-4]\n"
                      "mov.w r2, r3, ror #8\n"
                      "sxtb16 r2, r2\n"
                      "sxtb16 r3, r3\n"
                      "smlad %[sum3], r4, r3, %[sum3]\n"
                      "smlad %[sum4], r4, r2, %[sum4]\n"
                      "subs %[colCnt], #1\n"
                      "bne COL_LOOP_%=\n":[sum] "+r"(sum),
                      [sum2] "+r"(sum2),[sum3] "+r"(sum3),
                      [sum4] "+r"(sum4),[pB] "+r"(pB),[pA] "+r"(pA):[colCnt] "r"(colCnt):"r0", "r1", "r2", "r3", "r4");
#else
        asm volatile ("COL_LOOP_%=:\n"
                      "ldr.w r4, [%[pA]], #8\n"
                      "ldr.w r1, [%[pB]], #16\n"
                      "mov.w r0, r1, ror #8\n"
                      "sxtb16 r0, r0\n"
                      "sxtb16 r1, r1\n"
                      "smlad %[sum], r4, r0, %[sum]\n"
                      "smlad %[sum2], r4, r1, %[sum2]\n"
                      "ldr.w r3, [%[pB], #-12]\n"
                      "mov.w r2, r3, ror #8\n"
                      "sxtb16 r2, r2\n"
                      "sxtb16 r3, r3\n"
                      "smlad %[sum3], r4, r2, %[sum3]\n"
                      "smlad %[sum4], r4, r3, %[sum4]\n"
                      "ldr.w r4, [%[pA], #-4]\n"
                      "ldr.w r1, [%[pB], #-8]\n"
                      "mov.w r0, r1, ror #8\n"
                      "sxtb16 r0, r0\n"
                      "sxtb16 r1, r1\n"
                      "smlad %[sum], r4, r0, %[sum]\n"
                      "smlad %[sum2], r4, r1, %[sum2]\n"
                      "ldr.w r3, [%[pB], #-4]\n"
                      "mov.w r2, r3, ror #8\n"
                      "sxtb16 r2, r2\n"
                      "sxtb16 r3, r3\n"
                      "smlad %[sum3], r4, r2, %[sum3]\n"
                      "smlad %[sum4], r4, r3, %[sum4]\n"
                      "subs %[colCnt], #1\n"
                      "bne COL_LOOP_%=\n":[sum] "+r"(sum),
                      [sum2] "+r"(sum2),[sum3] "+r"(sum3),
                      [sum4] "+r"(sum4),[pB] "+r"(pB),[pA] "+r"(pA):[colCnt] "r"(colCnt):"r0", "r1", "r2", "r3", "r4");
#endif                          /* ARM_MATH_BIG_ENDIAN */

#endif                          /* USE_INTRINSIC */

        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q15_t     inV = *pA++;
            q7_t      inM = *pB++;
            q7_t      inM2 = *pB++;
            q7_t      inM3 = *pB++;
            q7_t      inM4 = *pB++;

            sum += inV * inM;
            sum2 += inV * inM2;
            sum3 += inV * inM3;
            sum4 += inV * inM4;
            colCnt--;
        }                       /* while over colCnt */
        *pO++ = (q7_t) (__SSAT((sum >> out_shift), 8));
        *pO++ = (q7_t) (__SSAT((sum2 >> out_shift), 8));
        *pO++ = (q7_t) (__SSAT((sum3 >> out_shift), 8));
        *pO++ = (q7_t) (__SSAT((sum4 >> out_shift), 8));

        /* adjust the pointers and counters */
        rowCnt--;
    }

    /* left-over part of the rows */
    rowCnt = num_of_rows & 0x3;

    while (rowCnt)
    {
        q31_t     sum = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        uint16_t  colCnt = dim_vec >> 2;

        pA = vec_buffer;

        while (colCnt)
        {
            q31_t     inV1, inV2, inM11, inM12;

            pB = (q7_t *) read_and_pad_reordered((void *)pB, &inM11, &inM12);

            inV1 = *__SIMD32(pA)++;
            sum = __SMLAD(inV1, inM11, sum);

            inV2 = *__SIMD32(pA)++;
            sum = __SMLAD(inV2, inM12, sum);

            colCnt--;
        }

        /* left-over of the vector */
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q15_t     inV = *pA++;
            q7_t      inM = *pB++;
            sum += inV * inM;
            colCnt--;
        }

        *pO++ = (q7_t) (__SSAT((sum >> out_shift), 8));

        rowCnt--;
    }

#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    uint16_t  rowCnt = num_of_rows >> 2;
    const q7_t *pB = pM;
    const q7_t *pA;
    q7_t     *pO = pOut;
    const q7_t *pBias = bias;

    while (rowCnt)
    {
        q31_t     sum =  ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum2 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum3 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t     sum4 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

        uint16_t  colCnt = dim_vec >> 2;

        pA = pV;

        while (colCnt)
        {
            q7_t      inA1 = *pA++;
            q7_t      inA3 = *pA++;
            q7_t      inA2 = *pA++;
            q7_t      inA4 = *pA++;

            q7_t      inB1 = *pB++;
            q7_t      inB3 = *pB++;
            q7_t      inB2 = *pB++;
            q7_t      inB4 = *pB++;

            sum += inA1 * inB1 + inA2 * inB2;
            sum2 += inA1 * inB3 + inA2 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum3 += inA1 * inB1 + inA2 * inB2;
            sum4 += inA1 * inB3 + inA2 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum += inA3 * inB1 + inA4 * inB2;
            sum2 += inA3 * inB3 + inA4 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum3 += inA3 * inB1 + inA4 * inB2;
            sum4 += inA3 * inB3 + inA4 * inB4;

            colCnt--;
        }
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q7_t      inA = *pA++;
            q7_t      inB = *pB++;
            sum += inA * inB;
            inB = *pB++;
            sum2 += inA * inB;
            inB = *pB++;
            sum3 += inA * inB;
            inB = *pB++;
            sum4 += inA * inB;

            colCnt--;
        }
        *pO++ = (q7_t) __SSAT((sum >> out_shift), 8);
        *pO++ = (q7_t) __SSAT((sum2 >> out_shift), 8);
        *pO++ = (q7_t) __SSAT((sum3 >> out_shift), 8);
        *pO++ = (q7_t) __SSAT((sum4 >> out_shift), 8);

        rowCnt--;
    }

    rowCnt = num_of_rows & 0x3;

    while (rowCnt)
    {
        int       ip_out = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

        int       j;

        pA = pV;
        for (j = 0; j < dim_vec; j++)
        {
            q7_t      inA = *pA++;
            q7_t      inB = *pB++;
            ip_out += inA * inB;
        }
        *pO++ = (q7_t) __SSAT((ip_out >> out_shift), 8);

        rowCnt--;
    }

#endif                          /* ARM_MATH_DSP */

    /* Return to ARM_MATH_SUCCESS */
    return (ARM_MATH_SUCCESS);

}