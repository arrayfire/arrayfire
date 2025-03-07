/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/********************************************************
 * Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University.
 * Copyright (c) 2011, 2012 Mutsuo Saito, Makoto Matsumoto, Hiroshima
 * University and University of Tokyo.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of the Hiroshima University, The Uinversity
 *       of Tokyo nor the names of its contributors may be used to
 *       endorse or promote products derived from this software without
 *       specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************/
#pragma once
#include <kernel/accessors.hpp>
#include <kernel/random_engine_write.hpp>

#include <sycl/sycl.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

constexpr int N          = 351;
constexpr int BLOCKS     = 32;
constexpr int STATE_SIZE = (256 * 3);
constexpr int TABLE_SIZE = 16;

// Utils
static inline void read_table(uint *const sharedTable, const uint *const table,
                              size_t groupId, size_t localId) {
    const uint *const t = table + (groupId * TABLE_SIZE);
    if (localId < TABLE_SIZE) { sharedTable[localId] = t[localId]; }
}

static inline void state_read(uint *const state, const uint *const gState,
                              size_t groupRange, size_t groupId,
                              size_t localId) {
    const uint *const g             = gState + (groupId * N);
    state[STATE_SIZE - N + localId] = g[localId];
    if (localId < N - groupRange) {
        state[STATE_SIZE - N + groupRange + localId] = g[groupRange + localId];
    }
}

static inline void state_write(uint *const gState, const uint *const state,
                               size_t groupRange, size_t groupId,
                               size_t localId) {
    uint *const g = gState + (groupId * N);
    g[localId]    = state[STATE_SIZE - N + localId];
    if (localId < N - groupRange) {
        g[groupRange + localId] = state[STATE_SIZE - N + groupRange + localId];
    }
}

static inline uint recursion(const uint *const recursion_table, const uint mask,
                             const uint sh1, const uint sh2, const uint x1,
                             const uint x2, uint y) {
    uint x = (x1 & mask) ^ x2;
    x ^= x << sh1;
    y        = x ^ (y >> sh2);
    uint mat = recursion_table[y & 0x0f];
    return y ^ mat;
}

static inline uint temper(const uint *const temper_table, const uint v,
                          uint t) {
    t ^= t >> 16;
    t ^= t >> 8;
    uint mat = temper_table[t & 0x0f];
    return v ^ mat;
}

// Initialization
class initMersenneKernel {
   public:
    initMersenneKernel(write_accessor<uint> state, read_accessor<uint> tbl,
                       sycl::local_accessor<uint, 1> lstate, uintl seed)
        : state_(state), tbl_(tbl), lstate_(lstate), seed_(seed) {}

    void operator()(sycl::nd_item<1> it) const {
        sycl::group g = it.get_group();

        const uint *ltbl =
            tbl_.get_pointer() + (TABLE_SIZE * g.get_group_id(0));
        uint hidden_seed = ltbl[4] ^ (ltbl[8] << 16);
        uint tmp         = hidden_seed;
        tmp += tmp >> 16;
        tmp += tmp >> 8;
        tmp &= 0xff;
        tmp |= tmp << 8;
        tmp |= tmp << 16;
        lstate_[it.get_local_id(0)] = tmp;
        it.barrier();
        if (it.get_local_id(0) == 0) {
            lstate_[0] = seed_;
            lstate_[1] = hidden_seed;
            for (int i = 1; i < N; ++i) {
                lstate_[i] ^= ((uint)(1812433253) *
                                   (lstate_[i - 1] ^ (lstate_[i - 1] >> 30)) +
                               i);
            }
        }
        it.barrier();
        state_[N * g.get_group_id(0) + it.get_local_id(0)] =
            lstate_[it.get_local_id(0)];
    }

   protected:
    write_accessor<uint> state_;
    read_accessor<uint> tbl_;
    sycl::local_accessor<uint, 1> lstate_;
    uintl seed_;
};

void initMersenneState(Param<uint> state, const Param<uint> tbl, uintl seed) {
    sycl::nd_range<1> ndrange({BLOCKS * N}, {N});
    getQueue().submit([&](sycl::handler &h) {
        write_accessor<uint> state_acc{*state.data, h};
        read_accessor<uint> tbl_acc{*tbl.data, h};
        auto lstate_acc = sycl::local_accessor<uint, 1>(N, h);

        h.parallel_for(
            ndrange, initMersenneKernel(state_acc, tbl_acc, lstate_acc, seed));
    });
    // TODO: do we need to sync before using Mersenne generators?
    // force wait() here?
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T>
class uniformMersenne {
   public:
    uniformMersenne(write_accessor<T> out, sycl::accessor<uint> gState,
                    sycl::accessor<uint> pos_tbl, sycl::accessor<uint> sh1_tbl,
                    sycl::accessor<uint> sh2_tbl, uint mask,
                    sycl::accessor<uint> g_recursion_table,
                    sycl::accessor<uint> g_temper_table,
                    // local memory caches of global state
                    sycl::local_accessor<uint, 1> state,
                    sycl::local_accessor<uint, 1> recursion_table,
                    sycl::local_accessor<uint, 1> temper_table,
                    uint elementsPerBlock, size_t elements)
        : out_(out)
        , gState_(gState)
        , pos_tbl_(pos_tbl)
        , sh1_tbl_(sh1_tbl)
        , sh2_tbl_(sh2_tbl)
        , mask_(mask)
        , g_recursion_table_(g_recursion_table)
        , g_temper_table_(g_temper_table)
        , state_(state)
        , recursion_table_(recursion_table)
        , temper_table_(temper_table)
        , elementsPerBlock_(elementsPerBlock)
        , elements_(elements) {}

    void operator()(sycl::nd_item<1> it) const {
        sycl::group g = it.get_group();
        uint start    = g.get_group_id(0) * elementsPerBlock_;
        uint end      = start + elementsPerBlock_;
        end           = (end > elements_) ? elements_ : end;
        int elementsPerBlockIteration =
            (g.get_local_range(0) * 4 * sizeof(uint)) / sizeof(T);
        int iter = divup((end - start), elementsPerBlockIteration);

        uint pos = pos_tbl_[it.get_group(0)];
        uint sh1 = sh1_tbl_[it.get_group(0)];
        uint sh2 = sh2_tbl_[it.get_group(0)];
        state_read(state_.get_pointer(), gState_.get_pointer(),
                   g.get_local_range(0), g.get_group_id(0), it.get_local_id(0));
        read_table(recursion_table_.get_pointer(),
                   g_recursion_table_.get_pointer(), g.get_group_id(0),
                   it.get_local_id(0));
        read_table(temper_table_.get_pointer(), g_temper_table_.get_pointer(),
                   g.get_group_id(0), it.get_local_id(0));
        it.barrier();

        uint index = start;
        uint o[4];
        int offsetX1 = (STATE_SIZE - N + it.get_local_id(0)) % STATE_SIZE;
        int offsetX2 = (STATE_SIZE - N + it.get_local_id(0) + 1) % STATE_SIZE;
        int offsetY  = (STATE_SIZE - N + it.get_local_id(0) + pos) % STATE_SIZE;
        int offsetT =
            (STATE_SIZE - N + it.get_local_id(0) + pos - 1) % STATE_SIZE;
        int offsetO = it.get_local_id(0);

        for (int i = 0; i < iter; ++i) {
            for (int ii = 0; ii < 4; ++ii) {
                uint r = recursion(recursion_table_.get_pointer(), mask_, sh1,
                                   sh2, state_[offsetX1], state_[offsetX2],
                                   state_[offsetY]);
                state_[offsetO] = r;
                o[ii] = temper(temper_table_.get_pointer(), r, state_[offsetT]);
                offsetX1 = (offsetX1 + g.get_local_range(0)) % STATE_SIZE;
                offsetX2 = (offsetX2 + g.get_local_range(0)) % STATE_SIZE;
                offsetY  = (offsetY + g.get_local_range(0)) % STATE_SIZE;
                offsetT  = (offsetT + g.get_local_range(0)) % STATE_SIZE;
                offsetO  = (offsetO + g.get_local_range(0)) % STATE_SIZE;
                it.barrier();
            }
            if (i == iter - 1) {
                partialWriteOut128Bytes(
                    out_.get_pointer(), index + it.get_local_id(0),
                    g.get_local_range(0), o[0], o[1], o[2], o[3], elements_);
            } else {
                writeOut128Bytes(out_.get_pointer(), index + it.get_local_id(0),
                                 g.get_local_range(0), o[0], o[1], o[2], o[3]);
            }
            index += elementsPerBlockIteration;
        }
        state_write(gState_.get_pointer(), state_.get_pointer(),
                    g.get_local_range(0), g.get_group_id(0),
                    it.get_local_id(0));
    }

   protected:
    write_accessor<T> out_;
    sycl::accessor<uint> gState_;
    sycl::accessor<uint> pos_tbl_, sh1_tbl_, sh2_tbl_;
    uint mask_;
    sycl::accessor<uint> g_recursion_table_, g_temper_table_;
    sycl::local_accessor<uint, 1> state_, recursion_table_, temper_table_;
    uint elementsPerBlock_;
    size_t elements_;
};

template<typename T>
class normalMersenne {
   public:
    normalMersenne(write_accessor<T> out, sycl::accessor<uint> gState,
                   sycl::accessor<uint> pos_tbl, sycl::accessor<uint> sh1_tbl,
                   sycl::accessor<uint> sh2_tbl, uint mask,
                   sycl::accessor<uint> g_recursion_table,
                   sycl::accessor<uint> g_temper_table,
                   // local memory caches of global state
                   sycl::local_accessor<uint, 1> state,
                   sycl::local_accessor<uint, 1> recursion_table,
                   sycl::local_accessor<uint, 1> temper_table,
                   uint elementsPerBlock, size_t elements)
        : out_(out)
        , gState_(gState)
        , pos_tbl_(pos_tbl)
        , sh1_tbl_(sh1_tbl)
        , sh2_tbl_(sh2_tbl)
        , mask_(mask)
        , g_recursion_table_(g_recursion_table)
        , g_temper_table_(g_temper_table)
        , state_(state)
        , recursion_table_(recursion_table)
        , temper_table_(temper_table)
        , elementsPerBlock_(elementsPerBlock)
        , elements_(elements) {}

    void operator()(sycl::nd_item<1> it) const {
        sycl::group g = it.get_group();
        uint start    = g.get_group_id(0) * elementsPerBlock_;
        uint end      = start + elementsPerBlock_;
        end           = (end > elements_) ? elements_ : end;
        int elementsPerBlockIteration =
            (g.get_local_range(0) * 4 * sizeof(uint)) / sizeof(T);
        int iter = divup((end - start), elementsPerBlockIteration);

        uint pos = pos_tbl_[it.get_group(0)];
        uint sh1 = sh1_tbl_[it.get_group(0)];
        uint sh2 = sh2_tbl_[it.get_group(0)];
        state_read(state_.get_pointer(), gState_.get_pointer(),
                   g.get_local_range(0), g.get_group_id(0), it.get_local_id(0));
        read_table(recursion_table_.get_pointer(),
                   g_recursion_table_.get_pointer(), g.get_group_id(0),
                   it.get_local_id(0));
        read_table(temper_table_.get_pointer(), g_temper_table_.get_pointer(),
                   g.get_group_id(0), it.get_local_id(0));
        it.barrier();

        uint index = start;
        uint o[4];
        int offsetX1 = (STATE_SIZE - N + it.get_local_id(0)) % STATE_SIZE;
        int offsetX2 = (STATE_SIZE - N + it.get_local_id(0) + 1) % STATE_SIZE;
        int offsetY  = (STATE_SIZE - N + it.get_local_id(0) + pos) % STATE_SIZE;
        int offsetT =
            (STATE_SIZE - N + it.get_local_id(0) + pos - 1) % STATE_SIZE;
        int offsetO = it.get_local_id(0);

        for (int i = 0; i < iter; ++i) {
            for (int ii = 0; ii < 4; ++ii) {
                uint r = recursion(recursion_table_.get_pointer(), mask_, sh1,
                                   sh2, state_[offsetX1], state_[offsetX2],
                                   state_[offsetY]);
                state_[offsetO] = r;
                o[ii] = temper(temper_table_.get_pointer(), r, state_[offsetT]);
                offsetX1 = (offsetX1 + g.get_local_range(0)) % STATE_SIZE;
                offsetX2 = (offsetX2 + g.get_local_range(0)) % STATE_SIZE;
                offsetY  = (offsetY + g.get_local_range(0)) % STATE_SIZE;
                offsetT  = (offsetT + g.get_local_range(0)) % STATE_SIZE;
                offsetO  = (offsetO + g.get_local_range(0)) % STATE_SIZE;
                it.barrier();
            }
            if (i == iter - 1) {
                partialBoxMullerWriteOut128Bytes(
                    out_.get_pointer(), index + it.get_local_id(0),
                    g.get_local_range(0), o[0], o[1], o[2], o[3], elements_);
            } else {
                boxMullerWriteOut128Bytes(
                    out_.get_pointer(), index + it.get_local_id(0),
                    g.get_local_range(0), o[0], o[1], o[2], o[3]);
            }
            index += elementsPerBlockIteration;
        }
        state_write(gState_.get_pointer(), state_.get_pointer(),
                    g.get_local_range(0), g.get_group_id(0),
                    it.get_local_id(0));
    }

   protected:
    write_accessor<T> out_;
    sycl::accessor<uint> gState_;
    sycl::accessor<uint> pos_tbl_, sh1_tbl_, sh2_tbl_;
    uint mask_;
    sycl::accessor<uint> g_recursion_table_, g_temper_table_;
    sycl::local_accessor<uint, 1> state_, recursion_table_, temper_table_;
    uint elementsPerBlock_;
    size_t elements_;
};

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
