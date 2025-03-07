/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <common/ArrayInfo.hpp>
#include <types.hpp>
#include <utility.hpp>

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/seq.h>

#include <vector>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void assign(Param<T> out, af::dim4 dDims, CParam<T> rhs,
            std::vector<bool> const isSeq, std::vector<af_seq> const seqs,
            std::vector<CParam<uint>> idxArrs) {
    af::dim4 pDims = out.dims();
    // retrieve dimensions & strides for array to which rhs is being copied to
    af::dim4 dst_offsets = toOffset(seqs, dDims);
    af::dim4 dst_strides = toStride(seqs, dDims);
    // retrieve rhs array dimenesions & strides
    af::dim4 src_dims    = rhs.dims();
    af::dim4 src_strides = rhs.strides();
    // declare pointers to af_array index data
    uint const* const ptr0 = idxArrs[0].get();
    uint const* const ptr1 = idxArrs[1].get();
    uint const* const ptr2 = idxArrs[2].get();
    uint const* const ptr3 = idxArrs[3].get();

    const T* src = rhs.get();
    T* dst       = out.get();

    for (dim_t l = 0; l < src_dims[3]; ++l) {
        dim_t src_loff = l * src_strides[3];

        dim_t dst_lIdx =
            trimIndex(isSeq[3] ? l + dst_offsets[3] : ptr3[l], pDims[3]);
        dim_t dst_loff = dst_lIdx * dst_strides[3];

        for (dim_t k = 0; k < src_dims[2]; ++k) {
            dim_t src_koff = k * src_strides[2];

            dim_t dst_kIdx =
                trimIndex(isSeq[2] ? k + dst_offsets[2] : ptr2[k], pDims[2]);
            dim_t dst_koff = dst_kIdx * dst_strides[2];

            for (dim_t j = 0; j < src_dims[1]; ++j) {
                dim_t src_joff = j * src_strides[1];

                dim_t dst_jIdx = trimIndex(
                    isSeq[1] ? j + dst_offsets[1] : ptr1[j], pDims[1]);
                dim_t dst_joff = dst_jIdx * dst_strides[1];

                for (dim_t i = 0; i < src_dims[0]; ++i) {
                    dim_t src_ioff = i * src_strides[0];
                    dim_t src_idx  = src_ioff + src_joff + src_koff + src_loff;

                    dim_t dst_iIdx = trimIndex(
                        isSeq[0] ? i + dst_offsets[0] : ptr0[i], pDims[0]);
                    dim_t dst_ioff = dst_iIdx * dst_strides[0];
                    dim_t dst_idx  = dst_ioff + dst_joff + dst_koff + dst_loff;

                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
