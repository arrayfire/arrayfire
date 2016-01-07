/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <Array.hpp>

namespace cpu
{
namespace kernel
{

template<typename T>
void stridedCopy(T* dst, af::dim4 const & ostrides, T const * src,
                 af::dim4 const & dims, af::dim4 const & strides, unsigned dim)
{
    if(dim == 0) {
        if(strides[dim] == 1) {
            //FIXME: Check for errors / exceptions
            memcpy(dst, src, dims[dim] * sizeof(T));
        } else {
            for(dim_t i = 0; i < dims[dim]; i++) {
                dst[i] = src[strides[dim]*i];
            }
        }
    } else {
        for(dim_t i = dims[dim]; i > 0; i--) {
            stridedCopy<T>(dst, ostrides, src, dims, strides, dim - 1);
            src += strides[dim];
            dst += ostrides[dim];
        }
    }
}

template<typename OutT, typename InT>
void copy(Array<OutT> dst, Array<InT> const src, OutT default_value, double factor)
{
    af::dim4 src_dims       = src.dims();
    af::dim4 dst_dims       = dst.dims();
    af::dim4 src_strides    = src.strides();
    af::dim4 dst_strides    = dst.strides();

    InT const * const src_ptr = src.get();
    OutT * dst_ptr      = dst.get();

    dim_t trgt_l = std::min(dst_dims[3], src_dims[3]);
    dim_t trgt_k = std::min(dst_dims[2], src_dims[2]);
    dim_t trgt_j = std::min(dst_dims[1], src_dims[1]);
    dim_t trgt_i = std::min(dst_dims[0], src_dims[0]);

    for(dim_t l=0; l<dst_dims[3]; ++l) {

        dim_t src_loff = l*src_strides[3];
        dim_t dst_loff = l*dst_strides[3];
        bool isLvalid = l<trgt_l;

        for(dim_t k=0; k<dst_dims[2]; ++k) {

            dim_t src_koff = k*src_strides[2];
            dim_t dst_koff = k*dst_strides[2];
            bool isKvalid = k<trgt_k;

            for(dim_t j=0; j<dst_dims[1]; ++j) {

                dim_t src_joff = j*src_strides[1];
                dim_t dst_joff = j*dst_strides[1];
                bool isJvalid = j<trgt_j;

                for(dim_t i=0; i<dst_dims[0]; ++i) {
                    OutT temp = default_value;
                    if (isLvalid && isKvalid && isJvalid && i<trgt_i) {
                        dim_t src_idx = i*src_strides[0] + src_joff + src_koff + src_loff;
                        temp = OutT(src_ptr[src_idx])*OutT(factor);
                    }
                    dim_t dst_idx = i*dst_strides[0] + dst_joff + dst_koff + dst_loff;
                    dst_ptr[dst_idx] = temp;
                }
            }
        }
    }
}

}
}
