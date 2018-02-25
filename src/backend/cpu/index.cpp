/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <index.hpp>
#include <handle.hpp>
#include <vector>
#include <platform.hpp>
#include <queue.hpp>
#include <utility>
#include <kernel/index.hpp>

using std::vector;
using af::dim4;

namespace cpu
{

template<typename T>
Array<T> index(const Array<T>& in, const af_index_t idxrs[])
{
    in.eval();

    vector<bool> isSeq(4);
    vector<af_seq> seqs(4, af_span);
    // create seq vector to retrieve output
    // dimensions, offsets & offsets
    for (unsigned x=0; x<isSeq.size(); ++x) {
        if (idxrs[x].isSeq) {
            seqs[x] = idxrs[x].idx.seq;
        }
        isSeq[x] = idxrs[x].isSeq;
    }

    // retrieve
    dim4 oDims = toDims(seqs, in.dims());

    vector< Array<uint> > idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexs to read af_array indexs
    for (unsigned x=0; x<isSeq.size(); ++x) {
        if (!isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].idx.arr);
            idxArrs[x].eval();
            // set output array ith dimension value
            oDims[x] = idxArrs[x].elements();
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);
    std::vector<CParam<uint>> idxParams(idxArrs.begin(), idxArrs.end());

    getQueue().enqueue(kernel::index<T>, out, in, in.getDataDims(),
                       std::move(isSeq), std::move(seqs), std::move(idxParams));

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> index<T>(const Array<T>& in, const af_index_t idxrs[]);

INSTANTIATE(cdouble)
INSTANTIATE(double )
INSTANTIATE(cfloat )
INSTANTIATE(float  )
INSTANTIATE(uintl  )
INSTANTIATE(uint   )
INSTANTIATE(intl   )
INSTANTIATE(int    )
INSTANTIATE(uchar  )
INSTANTIATE(char   )
INSTANTIATE(ushort )
INSTANTIATE(short  )

#undef INSTANTIATE
}
