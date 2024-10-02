/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <index.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <kernel/index.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <utility>
#include <vector>

using af::dim4;
using arrayfire::common::half;  // NOLINT(misc-unused-using-decls) bug in
                                // clang-tidy
using std::vector;

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> index(const Array<T>& in, const af_index_t idxrs[]) {
    vector<bool> isSeq(4);
    vector<af_seq> seqs(4, af_span);
    // create seq vector to retrieve output
    // dimensions, offsets & offsets
    for (unsigned x = 0; x < isSeq.size(); ++x) {
        if (idxrs[x].isSeq) {
            af_seq seq = idxrs[x].idx.seq;
            // Handle af_span as a sequence that covers the complete axis
            if (seq.begin == af_span.begin && seq.end == af_span.end &&
                seq.step == af_span.step) {
                seqs[x] = af_seq{0, (double)(in.dims()[x] - 1), 1};
            } else {
                seqs[x] = seq;
            }
        }
        isSeq[x] = idxrs[x].isSeq;
    }

    // retrieve
    dim4 oDims = toDims(seqs, in.dims());

    vector<Array<uint>> idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexs to read af_array indexs
    for (unsigned x = 0; x < isSeq.size(); ++x) {
        if (!isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].idx.arr);
            // set output array ith dimension value
            oDims[x] = idxArrs[x].elements();
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);
    vector<CParam<uint>> idxParams(idxArrs.begin(), idxArrs.end());

    getQueue().enqueue(kernel::index<T>, out, in, in.getDataDims(),
                       std::move(isSeq), std::move(seqs), std::move(idxParams));

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> index<T>(const Array<T>& in, const af_index_t idxrs[]);

INSTANTIATE(cdouble)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(float)
INSTANTIATE(uintl)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(int)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

}  // namespace cpu
}  // namespace arrayfire
