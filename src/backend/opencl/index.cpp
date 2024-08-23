/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <index.hpp>
#include <kernel/index.hpp>

#include <Array.hpp>
#include <err_opencl.hpp>
#include <handle.hpp>
#include <memory.hpp>
#include <af/dim4.hpp>

using arrayfire::common::half;

namespace arrayfire {
namespace opencl {

template<typename T>
Array<T> index(const Array<T>& in, const af_index_t idxrs[]) {
    kernel::IndexKernelParam_t p;
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
        p.isSeq[i] = idxrs[i].isSeq ? 1 : 0;
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

    cl::Buffer* bPtrs[4];

    auto buf = cl::Buffer();
    std::vector<Array<uint>> idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexs to read af_array indexs
    for (dim_t x = 0; x < 4; ++x) {
        // set index pointers were applicable
        if (!p.isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].idx.arr);
            bPtrs[x]   = idxArrs[x].get();
            // set output array ith dimension value
            oDims[x] = idxArrs[x].elements();
        } else {
            // alloc an 1-element buffer to avoid OpenCL from failing
            bPtrs[x] = &buf;
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);
    if (oDims.elements() == 0) { return out; }

    kernel::index<T>(out, in, p, bPtrs);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> index<T>(const Array<T>& in, const af_index_t idxrs[]);

INSTANTIATE(cdouble)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace opencl
}  // namespace arrayfire
