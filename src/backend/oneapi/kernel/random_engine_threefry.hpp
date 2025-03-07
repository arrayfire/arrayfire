/*******************************************************
 * Copyright (c) 2022, ArrayFire
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

#pragma once
#include <kernel/accessors.hpp>
#include <kernel/random_engine_write.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {
// Utils
// Source of these constants :
// github.com/DEShawResearch/Random123-Boost/blob/master/boost/random/threefry.hpp

static const uint SKEIN_KS_PARITY32 = 0x1BD11BDA;

static const uint R0 = 13;
static const uint R1 = 15;
static const uint R2 = 26;
static const uint R3 = 6;
static const uint R4 = 17;
static const uint R5 = 29;
static const uint R6 = 16;
static const uint R7 = 24;

static inline void setSkeinParity(uint* ptr) { *ptr = SKEIN_KS_PARITY32; }

static inline uint rotL(uint x, uint N) {
    return (x << (N & 31)) | (x >> ((32 - N) & 31));
}

void threefry(uint k[2], uint c[2], uint X[2]) {
    uint ks[3];

    setSkeinParity(&ks[2]);
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

template<typename T>
class uniformThreefry {
   public:
    uniformThreefry(write_accessor<T> out, uint hi, uint lo, uint hic, uint loc,
                    uint elementsPerBlock, uint elements)
        : out_(out)
        , hi_(hi)
        , lo_(lo)
        , hic_(hic)
        , loc_(loc)
        , elementsPerBlock_(elementsPerBlock)
        , elements_(elements) {}

    void operator()(sycl::nd_item<1> it) const {
        sycl::group g = it.get_group();
        uint index = g.get_group_id(0) * elementsPerBlock_ + it.get_local_id(0);

        uint key[2] = {lo_, hi_};
        uint ctr[4] = {loc_, hic_, 0, 0};
        ctr[0] += index;
        ctr[1] += (ctr[0] < loc_);
        uint o[4];

        threefry(key, ctr, o);
        uint step = elementsPerBlock_ / 2;
        ctr[0] += step;
        ctr[1] += (ctr[0] < step);
        threefry(key, ctr, o + 2);

        T* optr = out_.get_pointer();
        if (g.get_group_id(0) != (g.get_group_range(0) - 1)) {
            writeOut128Bytes(optr, index, g.get_local_range(0), o[0], o[1],
                             o[2], o[3]);
        } else {
            partialWriteOut128Bytes(optr, index, g.get_local_range(0), o[0],
                                    o[1], o[2], o[3], elements_);
        }
    }

   protected:
    write_accessor<T> out_;
    uint hi_, lo_, hic_, loc_;
    uint elementsPerBlock_, elements_;
};

template<typename T>
class normalThreefry {
   public:
    normalThreefry(write_accessor<T> out, uint hi, uint lo, uint hic, uint loc,
                   uint elementsPerBlock, uint elements)
        : out_(out)
        , hi_(hi)
        , lo_(lo)
        , hic_(hic)
        , loc_(loc)
        , elementsPerBlock_(elementsPerBlock)
        , elements_(elements) {}

    void operator()(sycl::nd_item<1> it) const {
        sycl::group g = it.get_group();
        uint index = g.get_group_id(0) * elementsPerBlock_ + it.get_local_id(0);

        uint key[2] = {lo_, hi_};
        uint ctr[4] = {loc_, hic_, 0, 0};
        ctr[0] += index;
        ctr[1] += (ctr[0] < loc_);
        uint o[4];

        threefry(key, ctr, o);
        uint step = elementsPerBlock_ / 2;
        ctr[0] += step;
        ctr[1] += (ctr[0] < step);
        threefry(key, ctr, o + 2);

        T* optr = out_.get_pointer();
        if (g.get_group_id(0) != (g.get_group_range(0) - 1)) {
            boxMullerWriteOut128Bytes(optr, index, g.get_local_range(0), o[0],
                                      o[1], o[2], o[3]);
        } else {
            partialBoxMullerWriteOut128Bytes(optr, index, g.get_local_range(0),
                                             o[0], o[1], o[2], o[3], elements_);
        }
    }

   protected:
    write_accessor<T> out_;
    uint hi_, lo_, hic_, loc_;
    uint elementsPerBlock_, elements_;
};

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
