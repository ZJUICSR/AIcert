#include"../../Inc/CNNModel/cnn.h"
#include <stdio.h>


uint32_t __SXTB16(uint32_t x)
  {
    return ((uint32_t)(((((q31_t)x << 24) >> 24) & (q31_t)0x0000FFFF) |
                       ((((q31_t)x <<  8) >>  8) & (q31_t)0xFFFF0000)  ));
  }

void *read_and_pad(void *source, q31_t * out1, q31_t * out2)
{
        q31_t     inA = *__SIMD32(source)++;
        q31_t     inAbuf1 = __SXTB16(__ROR(inA, 8));
        q31_t     inAbuf2 = __SXTB16(inA);

#ifndef ARM_MATH_BIG_ENDIAN
        *out2 = __PKHTB(inAbuf1, inAbuf2, 16);
        *out1 = __PKHBT(inAbuf2, inAbuf1, 16);
#else
        *out1 = __PKHTB(inAbuf1, inAbuf2, 16);
        *out2 = __PKHBT(inAbuf2, inAbuf1, 16);
#endif

        return source;
}

void *read_and_pad_reordered(void *source, q31_t * out1, q31_t * out2)
{
        q31_t     inA = *__SIMD32(source)++;
#ifndef ARM_MATH_BIG_ENDIAN
        *out2 = __SXTB16(__ROR(inA, 8));
        *out1 = __SXTB16(inA);
#else
        *out1 = __SXTB16(__ROR(inA, 8));
        *out2 = __SXTB16(inA);
#endif

        return source;
}



uint32_t __ROR(uint32_t op1, uint32_t op2)
{
  op2 %= 32U;
  if (op2 == 0U)
  {
    return op1;
  }
  return (op1 >> op2) | (op1 << (32U - op2));
}


uint32_t __SMLAD(
  uint32_t x,
  uint32_t y ,
  uint32_t sum)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y      ) >> 16)) +
                       ( ((q31_t)sum    )                                  )   ));
  }

uint32_t __QSUB8(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s, t, u;

    r = __SSAT(((((q31_t)x << 24) >> 24) - (((q31_t)y << 24) >> 24)), 8) & (int32_t)0x000000FF;
    s = __SSAT(((((q31_t)x << 16) >> 24) - (((q31_t)y << 16) >> 24)), 8) & (int32_t)0x000000FF;
    t = __SSAT(((((q31_t)x <<  8) >> 24) - (((q31_t)y <<  8) >> 24)), 8) & (int32_t)0x000000FF;
    u = __SSAT(((((q31_t)x      ) >> 24) - (((q31_t)y      ) >> 24)), 8) & (int32_t)0x000000FF;

    return ((uint32_t)((u << 24) | (t << 16) | (s <<  8) | (r      )));
  }

void compare_and_replace_if_larger_q7(q7_t * base,   // base data
                                             q7_t * target, // compare target
                                             const uint16_t length  // data length
                                    )
{
    q7_t     *pIn = base;
    q7_t     *pCom = target;
    union arm_nnword in;
    union arm_nnword com;
    uint16_t  cnt = length >> 2;

    while (cnt > 0u)
    {
        in.word = *__SIMD32(pIn);
        com.word = *__SIMD32(pCom)++;

        // if version
        if (com.bytes[0] > in.bytes[0])
            in.bytes[0] = com.bytes[0];
        if (com.bytes[1] > in.bytes[1])
            in.bytes[1] = com.bytes[1];
        if (com.bytes[2] > in.bytes[2])
            in.bytes[2] = com.bytes[2];
        if (com.bytes[3] > in.bytes[3])
            in.bytes[3] = com.bytes[3];

        *__SIMD32(pIn)++ = in.word;

        cnt--;
    }
}


#ifndef   __ASM
  #define __ASM                                  __asm
#endif
uint32_t __QADD16(uint32_t op1, uint32_t op2)
{

  uint32_t result;

  // int16_t op1_up=op1>>16;
  // int16_t op1_down=op1&0xFFFF;

  // int16_t op2_up=op2>>16;
  // int16_t op2_down=op2&0xFFFF;

  // int16_t up_add=op1_up+op2_up;
  // int16_t down_add=op1_down+op2_down;

  // result=((up_add&0x0000ffff)<<16)|(down_add&0x0000ffff);

  result=((((op1>>16)+(op2>>16))&0x0000ffff)<<16)|(((op1&0xFFFF)+(op2&0xFFFF))&0x0000ffff);

  return result;

  // uint32_t result;

  // __ASM volatile ("qadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  
  // return(result);
}

uint32_t __USAT(int32_t val, uint32_t sat)
{
  if (sat <= 31U)
  {
    const uint32_t max = ((1U << sat) - 1U);
    if (val > (int32_t)max)
    {
      return max;
    }
    else if (val < 0)
    {
      return 0U;
    }
  }
  return (uint32_t)val;
}