/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <vector>

namespace kernel
{

using af::dim4;
using cpu::Array;
using std::vector;

inline
dim_t trimIndex(int idx, const dim_t &len)
{
    int ret_val = idx;
    int offset  = abs(ret_val)%len;
    if (ret_val<0) {
        ret_val = offset-1;
    } else if (ret_val>=(int)len) {
        ret_val = len-offset-1;
    }
    return ret_val;
}

template<typename T>
void assign(Array<T> out, const Array<T> rhs, const vector<bool> isSeq,
            const vector<af_seq> seqs, const vector< Array<uint> > idxArrs)
{
    dim4 dDims = out.getDataDims();
    dim4 pDims = out.dims();
    // retrieve dimensions & strides for array to which rhs is being copied to
    dim4 dst_offsets = toOffset(seqs, dDims);
    dim4 dst_strides = toStride(seqs, dDims);
    // retrieve rhs array dimenesions & strides
    dim4 src_dims    = rhs.dims();
    dim4 src_strides = rhs.strides();
    // declare pointers to af_array index data
    const uint* ptr0 = idxArrs[0].get();
    const uint* ptr1 = idxArrs[1].get();
    const uint* ptr2 = idxArrs[2].get();
    const uint* ptr3 = idxArrs[3].get();

    const T * src= rhs.get();
    T * dst      = out.get();

    for(dim_t l=0; l<src_dims[3]; ++l) {

        dim_t src_loff = l*src_strides[3];

        dim_t dst_lIdx = trimIndex(isSeq[3] ? l+dst_offsets[3] : ptr3[l], pDims[3]);
        dim_t dst_loff = dst_lIdx * dst_strides[3];

        for(dim_t k=0; k<src_dims[2]; ++k) {

            dim_t src_koff = k*src_strides[2];

            dim_t dst_kIdx = trimIndex(isSeq[2] ? k+dst_offsets[2] : ptr2[k], pDims[2]);
            dim_t dst_koff = dst_kIdx * dst_strides[2];

            for(dim_t j=0; j<src_dims[1]; ++j) {

                dim_t src_joff = j*src_strides[1];

                dim_t dst_jIdx = trimIndex(isSeq[1] ? j+dst_offsets[1] : ptr1[j], pDims[1]);
                dim_t dst_joff = dst_jIdx * dst_strides[1];

                for(dim_t i=0; i<src_dims[0]; ++i) {

                    dim_t src_ioff = i*src_strides[0];
                    dim_t src_idx  = src_ioff + src_joff + src_koff + src_loff;

                    dim_t dst_iIdx = trimIndex(isSeq[0] ? i+dst_offsets[0] : ptr0[i], pDims[0]);
                    dim_t dst_ioff = dst_iIdx * dst_strides[0];
                    dim_t dst_idx  = dst_ioff + dst_joff + dst_koff + dst_loff;

                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

}
