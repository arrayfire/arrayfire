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
#include <handle.hpp>
#include <kernel/assign.hpp>
#include <assign.hpp>
#include <platform.hpp>
#include <queue.hpp>

namespace cpu
{

using af::dim4;
using std::vector;

template<typename T>
void assign(Array<T>& out, const af_index_t idxrs[], const Array<T>& rhs)
{
    out.eval();
    rhs.eval();

    vector<bool> isSeq(4);
    vector<af_seq> seqs(4, af_span);
    // create seq vector to retrieve output dimensions, offsets & offsets
    for (dim_t x=0; x<4; ++x) {
        if (idxrs[x].isSeq) {
            seqs[x] = idxrs[x].idx.seq;
        }
        isSeq[x] = idxrs[x].isSeq;
    }

    vector< Array<uint> > idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexs to read af_array indexs
    for (dim_t x=0; x<4; ++x) {
        if (!isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].idx.arr);
            idxArrs[x].eval();
        }
    }

    vector<CParam<uint>> idxParams(idxArrs.begin(), idxArrs.end());
    getQueue().enqueue(kernel::assign<T>, out, out.getDataDims(), rhs, std::move(isSeq),
            std::move(seqs), std::move(idxParams));
}

#define INSTANTIATE(T) \
    template void assign<T>(Array<T>& out, const af_index_t idxrs[], const Array<T>& rhs);

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

}
