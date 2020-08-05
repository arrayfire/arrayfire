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

// Utils
// Source of these constants :
// github.com/DEShawResearch/Random123-Boost/blob/master/boost/random/threefry.hpp

#define SKEIN_KS_PARITY 0x1BD11BDA

#define R0 13
#define R1 15
#define R2 26
#define R3 6
#define R4 17
#define R5 29
#define R6 16
#define R7 24

inline uint rotL(uint x, uint N) {
    return (x << (N & 31)) | (x >> ((32 - N) & 31));
}

inline void threefry(uint k[2], uint c[2], uint X[2]) {
    uint ks[3];

    ks[2] = SKEIN_KS_PARITY;
    ks[0] = k[0];
    X[0]  = c[0];
    ks[2] ^= k[0];
    ks[1] = k[1];
    X[1]  = c[1];
    ks[2] ^= k[1];

    X[0] += ks[0];
    X[1] += ks[1];

    X[0] += X[1];
    X[1] = rotL(X[1], R0);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R1);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R2);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R3);
    X[1] ^= X[0];

    /* InjectKey(r=1) */
    X[0] += ks[1];
    X[1] += ks[2];
    X[1] += 1; /* X[2-1] += r  */

    X[0] += X[1];
    X[1] = rotL(X[1], R4);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R5);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R6);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R7);
    X[1] ^= X[0];

    /* InjectKey(r=2) */
    X[0] += ks[2];
    X[1] += ks[0];
    X[1] += 2;

    X[0] += X[1];
    X[1] = rotL(X[1], R0);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R1);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R2);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R3);
    X[1] ^= X[0];

    /* InjectKey(r=3) */
    X[0] += ks[0];
    X[1] += ks[1];
    X[1] += 3;

    X[0] += X[1];
    X[1] = rotL(X[1], R4);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R5);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R6);
    X[1] ^= X[0];
    X[0] += X[1];
    X[1] = rotL(X[1], R7);
    X[1] ^= X[0];

    /* InjectKey(r=4) */
    X[0] += ks[1];
    X[1] += ks[2];
    X[1] += 4;
}

kernel void threefryGenerator(global T *output, unsigned elements, unsigned hic,
                              unsigned loc, unsigned hi, unsigned lo) {
    unsigned gid   = get_group_id(0);
    unsigned off   = get_local_size(0);
    unsigned index = gid * ELEMENTS_PER_BLOCK + get_local_id(0);

    uint key[2] = {lo, hi};
    uint ctr[2] = {loc, hic};
    uint o[4];

    ctr[0] += index;
    ctr[1] += (ctr[0] < index);

    threefry(key, ctr, o);
    uint step = ELEMENTS_PER_BLOCK / 2;
    ctr[0] += step;
    ctr[1] += (ctr[0] < step);
    threefry(key, ctr, o + 2);

    if (gid != get_num_groups(0) - 1) {
        WRITE(output, index, o[0], o[1], o[2], o[3]);
    } else {
        PARTIAL_WRITE(output, index, o[0], o[1], o[2], o[3], elements);
    }
}
