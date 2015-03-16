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
#include <assign.hpp>
#include <kernel/assign.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{

template<typename T>
void assign(Array<T>& out, const af_index_t idxrs[], const Array<T>& rhs)
{
    kernel::AssignKernelParam_t p;
    std::vector<af_seq> seqs(4, af_span);
    // create seq vector to retrieve output
    // dimensions, offsets & offsets
    for (dim_type x=0; x<4; ++x) {
        if (idxrs[x].mIsSeq) {
            seqs[x] = idxrs[x].mIndexer.seq;
        }
    }

    // retrieve dimensions, strides and offsets
    dim4 dDims = out.dims();
    // retrieve dimensions & strides for array
    // to which rhs is being copied to
    dim4 dstOffs = af::toOffset(seqs, dDims);
    dim4 dstStrds= af::toStride(seqs, dDims);

    for (dim_type i=0; i<4; ++i) {
        p.isSeq[i] = idxrs[i].mIsSeq;
        p.offs[i]  = dstOffs[i];
        p.strds[i] = dstStrds[i];
    }

    Buffer* bPtrs[4];

    std::vector< Array<uint> > idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexers to read af_array indexers
    for (dim_type x=0; x<4; ++x) {
        // set idxPtrs to null
        bPtrs[x] = new Buffer();
        // set index pointers were applicable
        if (!p.isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].mIndexer.arr);
            bPtrs[x] = idxArrs[x].get();
        }
    }

    kernel::assign<T>(out, rhs, p, bPtrs);
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

}
