/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <handle.hpp>
#include <index.hpp>
#include <kernel/index.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T>
Array<T> index(const Array<T>& in, const af_index_t idxrs[])
{
    kernel::IndexKernelParam_t p;
    std::vector<af_seq> seqs(4, af_span);
    // create seq vector to retrieve output
    // dimensions, offsets & offsets
    for (dim_type x=0; x<4; ++x) {
        if (idxrs[x].mIsSeq) {
            seqs[x] = idxrs[x].mIndexer.seq;
        }
    }

    // retrieve dimensions, strides and offsets
    dim4 iDims = in.dims();
    dim4 dDims = in.getDataDims();
    dim4 oDims = af::toDims  (seqs, iDims);
    dim4 iOffs = af::toOffset(seqs, dDims);
    dim4 iStrds= af::toStride(seqs, dDims);

    for (dim_type i=0; i<4; ++i) {
        p.isSeq[i] = idxrs[i].mIsSeq;
        p.offs[i]  = iOffs[i];
        p.strds[i] = iStrds[i];
    }

    std::vector< Array<uint> > idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexers to read af_array indexers
    for (dim_type x=0; x<4; ++x) {
        // set idxPtrs to null
        p.ptr[x] = 0;
        // set index pointers were applicable
        if (!p.isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].mIndexer.arr);
            p.ptr[x] = idxArrs[x].get();
            // set output array ith dimension value
            oDims[x] = idxArrs[x].elements();
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::index<T>(out, in, p);

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

}
