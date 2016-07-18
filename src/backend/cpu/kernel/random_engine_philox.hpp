/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 *
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

#pragma once
namespace cpu
{
namespace kernel
{

#define m4x32_0 0xD2511F53
#define m4x32_1 0xCD9E8D57
#define w32_0 0x9E3779B9
#define w32_1 0xBB67AE85

static inline void mulhilo(const uint a, const uint b, uint * const hi, uint * const lo)
{
    *hi = (((uintl)a) * ((uintl)b))>>32;
    *lo = a*b;
}

static inline void philoxBump(uint k[2])
{
    k[0] += w32_0;
    k[1] += w32_1;
}

static inline void philoxRound(const uint k[2], uint c[4])
{
    uint hi0, lo0, hi1, lo1;
    mulhilo(m4x32_0, c[0], &hi0, &lo0);
    mulhilo(m4x32_1, c[2], &hi1, &lo1);
    c[0] = hi1^c[1]^k[0];
    c[1] = lo1;
    c[2] = hi0^c[3]^k[1];
    c[3] = lo0;
}

static inline void philox(uint key[2], uint ctr[4])
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

/*
template <> struct Random<AF_RANDOM_PHILOX>
{
    uint hi;
    uint lo;
    uintl counter;
    uint key[2];
    uint ctr[4];
    int reset;

    template <typename T>
    T uniform(void);

    Random(uintl seed, uintl counter);
};

Random<AF_RANDOM_PHILOX>::Random(uintl seed, uintl counterInput) : hi(seed>>32), lo(seed), counter(counterInput), reset(0)
{
    key[0] = counter;
    key[1] = hi;
    ctr[0] = counter;
    ctr[1] = 0;
    ctr[2] = 0;
    ctr[3] = lo;
}

template <typename T>
void Random<AF_RANDOM_PHILOX>::uniform(T* out, size_t elements)
{
    int reset = (4*sizeof(uint))/sizeof(T);
    philox(key, ctr);
    for (int i = 0; i < (int)out.elements(); ++i) {
        if (fresh == reset) {
            philox(key, ctr);
            ctr[0] += 4;
            fresh = 0;
        }
        out[i] = transform<T>(ctr, fresh);
        fresh++;
    }

}
*/

}
}
