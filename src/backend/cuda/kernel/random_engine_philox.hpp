/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

namespace cuda
{
namespace kernel
{
    //Utils
#define m4x32_0 uint(0xD2511F53)
#define m4x32_1 uint(0xCD9E8D57)
#define w32_0 uint(0x9E3779B9)
#define w32_1 uint(0xBB67AE85)

    static inline __device__ void mulhilo(const uint &a, const uint &b, uint &hi, uint &lo)
    {
        hi = __umulhi(a,b);
        lo = a*b;
    }

    static inline __device__ void philoxBump(uint k[2])
    {
        k[0] += w32_0;
        k[1] += w32_1;
    }

    static inline __device__ void philoxRound(const uint k[2], uint c[4])
    {
        uint hi0, lo0, hi1, lo1;
        mulhilo(m4x32_0, c[0], hi0, lo0);
        mulhilo(m4x32_1, c[2], hi1, lo1);
        c[0] = hi1^c[1]^k[0];
        c[1] = lo1;
        c[2] = hi0^c[3]^k[1];
        c[3] = lo0;
    }

    static inline __device__ void philox(uint key[2], uint ctr[4])
    {
        //10 Rounds
                           philoxRound(key, ctr);
        philoxBump(key);   philoxRound(key, ctr);
        philoxBump(key);   philoxRound(key, ctr);
        philoxBump(key);   philoxRound(key, ctr);
        philoxBump(key);   philoxRound(key, ctr);
        philoxBump(key);   philoxRound(key, ctr);
        philoxBump(key);   philoxRound(key, ctr);
        philoxBump(key);   philoxRound(key, ctr);
        philoxBump(key);   philoxRound(key, ctr);
        philoxBump(key);   philoxRound(key, ctr);
    }
}
}
