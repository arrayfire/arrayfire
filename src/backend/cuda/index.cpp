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
#include <assign_kernel_param.hpp>
#include <common/half.hpp>
#include <err_cuda.hpp>
#include <handle.hpp>
#include <kernel/index.hpp>
#include <af/dim4.hpp>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace cuda {

template<typename T>
Array<T> index(const Array<T>& in, const af_index_t idxrs[]) {
    IndexKernelParam p;
    std::vector<af_seq> seqs(4, af_span);
    // create seq vector to retrieve output
    // dimensions, offsets & offsets
    for (dim_t x = 0; x < 4; ++x) {
        if (idxrs[x].isSeq) { seqs[x] = idxrs[x].idx.seq; }
    }

    // retrieve dimensions, strides and offsets
    const dim4& iDims = in.dims();
    dim4 dDims        = in.getDataDims();
    dim4 oDims        = toDims(seqs, iDims);
    dim4 iOffs        = toOffset(seqs, dDims);
    dim4 iStrds       = in.strides();

    for (dim_t i = 0; i < 4; ++i) {
        p.isSeq[i] = idxrs[i].isSeq;
        p.offs[i]  = iOffs[i];
        p.strds[i] = iStrds[i];
        p.steps[i] = 0;
        if (idxrs[i].isSeq) {
            af_seq seq = idxrs[i].idx.seq;
            // The step for af_span used in the kernel must be 1
            if (seq.begin == af_span.begin && seq.end == af_span.end &&
                seq.step == af_span.step)
                p.steps[i] = 1;
            else
                p.steps[i] = seq.step;
        }
    }

    std::vector<Array<uint>> idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexs to read af_array indexs
    for (dim_t x = 0; x < 4; ++x) {
        // set idxPtrs to null
        p.ptr[x] = 0;
        // set index pointers were applicable
        if (!p.isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].idx.arr);
            p.ptr[x]   = idxArrs[x].get();
            // set output array ith dimension value
            oDims[x] = idxArrs[x].elements();
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);
    if (oDims.elements() == 0) { return out; }

    kernel::index<T>(out, in, p);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> index<T>(const Array<T>& in, const af_index_t idxrs[]);

INSTANTIATE(cdouble)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(float)
INSTANTIATE(uint)
INSTANTIATE(int)
INSTANTIATE(uintl)
INSTANTIATE(intl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

}  // namespace cuda
}  // namespace arrayfire
