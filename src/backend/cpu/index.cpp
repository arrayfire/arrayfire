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
#include <index.hpp>
#include <handle.hpp>
#include <err_cpu.hpp>
#include <vector>

using af::dim4;

namespace cpu
{

static inline
dim_type trimIndex(dim_type idx, const dim_type &len)
{
    dim_type ret_val = idx;
    dim_type offset  = abs(ret_val)%len;
    if (ret_val<0) {
        ret_val = offset-1;
    } else if (ret_val>=len) {
        ret_val = len-offset-1;
    }
    return ret_val;
}

template<typename T>
Array<T> index(const Array<T>& in, const af_index_t idxrs[])
{
    bool isSeq[4];
    std::vector<af_seq> seqs(4, af_span);
    // create seq vector to retrieve output
    // dimensions, offsets & offsets
    for (dim_type x=0; x<4; ++x) {
        if (idxrs[x].mIsSeq) {
            seqs[x] = idxrs[x].mIndexer.seq;
        }
        isSeq[x] = idxrs[x].mIsSeq;
    }

    // rettrieve
    dim4 iDims = in.dims();
    dim4 dDims = in.getDataDims();
    dim4 oDims = af::toDims  (seqs, iDims);
    dim4 iOffs = af::toOffset(seqs, dDims);
    dim4 iStrds= af::toStride(seqs, dDims);

    std::vector< Array<uint> > idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexers to read af_array indexers
    for (dim_type x=0; x<4; ++x) {
        if (!isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].mIndexer.arr);
            // set output array ith dimension value
            oDims[x] = idxArrs[x].elements();
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);
    dim4 oStrides= out.strides();

    const T *src = in.get();
    T *dst = out.get();

    const uint* ptr0 = idxArrs[0].get();
    const uint* ptr1 = idxArrs[1].get();
    const uint* ptr2 = idxArrs[2].get();
    const uint* ptr3 = idxArrs[3].get();

    for (dim_type l=0; l<oDims[3]; ++l) {

        dim_type lOff   = l*oStrides[3];
        dim_type inIdx3 = trimIndex(isSeq[3] ? l+iOffs[3] : ptr3[l], iDims[3]);
        dim_type inOff3 = inIdx3*iStrds[3];

        for (dim_type k=0; k<oDims[2]; ++k) {

            dim_type kOff   = k*oStrides[2];
            dim_type inIdx2 = trimIndex(isSeq[2] ? k+iOffs[2] : ptr2[k], iDims[2]);
            dim_type inOff2 = inIdx2*iStrds[2];

            for (dim_type j=0; j<oDims[1]; ++j) {

                dim_type jOff   = j*oStrides[1];
                dim_type inIdx1 = trimIndex(isSeq[1] ? j+iOffs[1] : ptr1[j], iDims[1]);
                dim_type inOff1 = inIdx1*iStrds[1];

                for (dim_type i=0; i<oDims[0]; ++i) {

                    dim_type iOff   = i*oStrides[0];
                    dim_type inIdx0 = trimIndex(isSeq[0] ? i+iOffs[0] : ptr0[i], iDims[0]);
                    dim_type inOff0 = inIdx0*iStrds[0];

                    dst[lOff+kOff+jOff+iOff] = src[inOff3+inOff2+inOff1+inOff0];
                }
            }
        }
    }

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
