/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/*******************************************************
 * Modified version of Random123 library:
 * https://www.deshawresearch.com/downloads/download_random123.cgi/
 * The original copyright can be seen here:
 *
 * RANDOM123 LICENSE AGREEMENT
 *
 * Copyright 2010-2011, D. E. Shaw Research. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions, and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions, and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * Neither the name of D. E. Shaw Research nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************/

namespace cuda
{
namespace kernel
{
    //Utils

    static const uint  SKEIN_KS_PARITY32 = 0x1BD11BDA;
    static const uintl SKEIN_KS_PARITY64 = 0x1BD11BDAA9FC1A22;

    static const uint R0_32=13;
    static const uint R1_32=15;
    static const uint R2_32=26;
    static const uint R3_32= 6;
    static const uint R4_32=17;
    static const uint R5_32=29;
    static const uint R6_32=16;
    static const uint R7_32=24;

    static const uint R0_64=16;
    static const uint R1_64=42;
    static const uint R2_64=12;
    static const uint R3_64=31;
    static const uint R4_64=16;
    static const uint R5_64=32;
    static const uint R6_64=24;
    static const uint R7_64=21;

    static inline __device__ void setSkeinParity(uint *ptr)
    {
        *ptr = SKEIN_KS_PARITY32;
    }

    static inline __device__ void setSkeinParity(uintl *ptr)
    {
        *ptr = SKEIN_KS_PARITY64;
    }

    static inline __device__ uintl rotL(uintl x, uint N)
    {
        return (x << (N & 63)) | (x >> ((64-N) & 63));
    }

    static inline __device__ uint rotL(uint x, uint N)
    {
        return (x << (N & 31)) | (x >> ((32-N) & 31));
    }

    template <typename T, uint R0, uint R1, uint R2, uint R3, uint R4, uint R5, uint R6, uint R7>
    static inline __device__ void threefry_kernel(T k[2], T c[2], T X[2])
    {
        T ks[3];

        setSkeinParity(&ks[2]);
        ks[0] = k[0];
        X[0] = c[0];
        ks[2] ^= k[0];
        ks[1] = k[1];
        X[1] = c[1];
        ks[2] ^= k[1];

        X[0] += ks[0]; X[1] += ks[1];

        X[0] += X[1]; X[1] = rotL(X[1],R0); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R1); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R2); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R3); X[1] ^= X[0];

        /* InjectKey(r=1) */
        X[0] += ks[1]; X[1] += ks[2];
        X[1] += 1;     /* X[2-1] += r  */

        X[0] += X[1]; X[1] = rotL(X[1],R4); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R5); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R6); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R7); X[1] ^= X[0];

        /* InjectKey(r=2) */
        X[0] += ks[2]; X[1] += ks[0];
        X[1] += 2;

        X[0] += X[1]; X[1] = rotL(X[1],R0); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R1); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R2); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R3); X[1] ^= X[0];

        /* InjectKey(r=3) */
        X[0] += ks[0]; X[1] += ks[1];
        X[1] += 3;

        X[0] += X[1]; X[1] = rotL(X[1],R4); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R5); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R6); X[1] ^= X[0];
        X[0] += X[1]; X[1] = rotL(X[1],R7); X[1] ^= X[0];

        /* InjectKey(r=4) */
        X[0] += ks[1]; X[1] += ks[2];
        X[1] += 4;
    }

    __device__ void threefry(uint k[2], uint c[2], uint X[2])
    {
        threefry_kernel<uint, R0_32, R1_32, R2_32, R3_32, R4_32, R5_32, R6_32, R7_32>(k, c, X);
    }

    __device__ void threefry(uintl k[2], uintl c[2], uintl X[2])
    {
        threefry_kernel<uintl, R0_64, R1_64, R2_64, R3_64, R4_64, R5_64, R6_64, R7_64>(k, c, X);
    }
}
}
