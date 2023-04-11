/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <index.hpp>

#include <Array.hpp>
#include <err_oneapi.hpp>
#include <handle.hpp>
#include <kernel/assign_kernel_param.hpp>
#include <kernel/index.hpp>
#include <memory.hpp>
#include <af/dim4.hpp>

using arrayfire::common::half;
using arrayfire::oneapi::IndexKernelParam;

namespace arrayfire {
namespace oneapi {

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
    }

    std::vector<Array<uint>> idxArrs(4, createEmptyArray<uint>(dim4(1)));
    // look through indexs to read af_array indexs
    for (dim_t x = 0; x < 4; ++x) {
        if (!p.isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].idx.arr);
            oDims[x]   = idxArrs[x].elements();
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);
    if (oDims.elements() == 0) { return out; }
    kernel::index<T>(out, in, p, idxArrs);

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

}  // namespace oneapi
}  // namespace arrayfire
