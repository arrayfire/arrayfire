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
#include <utility.hpp>
#include <vector>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void index(Param<T> out, CParam<T> in, const af::dim4 dDims,
           std::vector<bool> const isSeq, std::vector<af_seq> const seqs,
           std::vector<CParam<uint>> idxArrs) {
    const af::dim4 iDims    = in.dims();
    const af::dim4 iOffs    = toOffset(seqs, dDims);
    const af::dim4 iStrds   = in.strides();
    const af::dim4 oDims    = out.dims();
    const af::dim4 oStrides = out.strides();
    const T* src            = in.get();
    T* dst                  = out.get();
    const uint* ptr0        = idxArrs[0].get();
    const uint* ptr1        = idxArrs[1].get();
    const uint* ptr2        = idxArrs[2].get();
    const uint* ptr3        = idxArrs[3].get();

    for (dim_t l = 0; l < oDims[3]; ++l) {
        dim_t lOff   = l * oStrides[3];
        dim_t inIdx3 = trimIndex(
            isSeq[3] ? l * seqs[3].step + iOffs[3] : ptr3[l], iDims[3]);
        dim_t inOff3 = inIdx3 * iStrds[3];

        for (dim_t k = 0; k < oDims[2]; ++k) {
            dim_t kOff   = k * oStrides[2];
            dim_t inIdx2 = trimIndex(
                isSeq[2] ? k * seqs[2].step + iOffs[2] : ptr2[k], iDims[2]);
            dim_t inOff2 = inIdx2 * iStrds[2];

            for (dim_t j = 0; j < oDims[1]; ++j) {
                dim_t jOff   = j * oStrides[1];
                dim_t inIdx1 = trimIndex(
                    isSeq[1] ? j * seqs[1].step + iOffs[1] : ptr1[j], iDims[1]);
                dim_t inOff1 = inIdx1 * iStrds[1];

                for (dim_t i = 0; i < oDims[0]; ++i) {
                    dim_t iOff   = i * oStrides[0];
                    dim_t inIdx0 = trimIndex(
                        isSeq[0] ? i * seqs[0].step + iOffs[0] : ptr0[i],
                        iDims[0]);
                    dim_t inOff0 = inIdx0 * iStrds[0];

                    dst[lOff + kOff + jOff + iOff] =
                        src[inOff3 + inOff2 + inOff1 + inOff0];
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
