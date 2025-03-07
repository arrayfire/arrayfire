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
#include <math.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>

#include <cstring>  //memcpy

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void stridedCopy(T* dst, af::dim4 const& ostrides, T const* src,
                 af::dim4 const& dims, af::dim4 const& strides, unsigned dim) {
    if (dim == 0) {
        if (strides[dim] == 1) {
            // FIXME: Check for errors / exceptions
            std::memcpy(dst, src, dims[dim] * sizeof(T));
        } else {
            for (dim_t i = 0; i < dims[dim]; i++) {
                dst[i] = src[strides[dim] * i];
            }
        }
    } else {
        for (dim_t i = dims[dim]; i > 0; i--) {
            stridedCopy<T>(dst, ostrides, src, dims, strides, dim - 1);
            src += strides[dim];
            dst += ostrides[dim];
        }
    }
}

template<typename OutT, typename InT>
void copyElemwise(Param<OutT> dst, CParam<InT> src, OutT default_value,
                  double factor) {
    af::dim4 src_dims    = src.dims();
    af::dim4 dst_dims    = dst.dims();
    af::dim4 src_strides = src.strides();
    af::dim4 dst_strides = dst.strides();

    data_t<InT> const* const src_ptr = src.get();
    data_t<OutT>* dst_ptr            = dst.get();

    dim_t trgt_l = std::min(dst_dims[3], src_dims[3]);
    dim_t trgt_k = std::min(dst_dims[2], src_dims[2]);
    dim_t trgt_j = std::min(dst_dims[1], src_dims[1]);
    dim_t trgt_i = std::min(dst_dims[0], src_dims[0]);

    for (dim_t l = 0; l < dst_dims[3]; ++l) {
        dim_t src_loff = l * src_strides[3];
        dim_t dst_loff = l * dst_strides[3];
        bool isLvalid  = l < trgt_l;

        for (dim_t k = 0; k < dst_dims[2]; ++k) {
            dim_t src_koff = k * src_strides[2];
            dim_t dst_koff = k * dst_strides[2];
            bool isKvalid  = k < trgt_k;

            for (dim_t j = 0; j < dst_dims[1]; ++j) {
                dim_t src_joff = j * src_strides[1];
                dim_t dst_joff = j * dst_strides[1];
                bool isJvalid  = j < trgt_j;

                for (dim_t i = 0; i < dst_dims[0]; ++i) {
                    data_t<OutT> temp = default_value;
                    if (isLvalid && isKvalid && isJvalid && i < trgt_i) {
                        dim_t src_idx =
                            i * src_strides[0] + src_joff + src_koff + src_loff;
                        // The conversions here are necessary because the half
                        // type does not convert to complex automatically
                        temp =
                            compute_t<OutT>(compute_t<InT>(src_ptr[src_idx])) *
                            compute_t<OutT>(factor);
                    }
                    dim_t dst_idx =
                        i * dst_strides[0] + dst_joff + dst_koff + dst_loff;
                    dst_ptr[dst_idx] = temp;
                }
            }
        }
    }
}

template<typename OutT, typename InT>
struct CopyImpl {
    static void copy(Param<OutT> dst, CParam<InT> src) {
        copyElemwise(dst, src, scalar<OutT>(0), 1.0);
    }
};

template<typename T>
struct CopyImpl<T, T> {
    static void copy(Param<T> dst, CParam<T> src) {
        af::dim4 src_dims    = src.dims();
        af::dim4 dst_dims    = dst.dims();
        af::dim4 src_strides = src.strides();
        af::dim4 dst_strides = dst.strides();

        T const* src_ptr = src.get();
        T* dst_ptr       = dst.get();

        // find the major-most dimension, which is linear in both arrays
        int linear_end = 0;
        dim_t count    = 1;
        while (linear_end < 4 && count == src_strides[linear_end] &&
               count == dst_strides[linear_end]) {
            count *= src_dims[linear_end];
            ++linear_end;
        }

        // traverse through the array using strides only until neccessary
        copy_go(dst_ptr, dst_strides, dst_dims, src_ptr, src_strides, src_dims,
                3, linear_end);
    }

    static void copy_go(T* dst_ptr, const af::dim4& dst_strides,
                        const af::dim4& dst_dims, T const* src_ptr,
                        const af::dim4& src_strides, const af::dim4& src_dims,
                        int dim, int linear_end) {
        // if we are in a higher dimension, copy the entire stride if possible
        if (linear_end == dim + 1) {
            std::memcpy(dst_ptr, src_ptr,
                        sizeof(T) * src_strides[dim] * src_dims[dim]);
            return;
        }

        // 0th dimension is recursion bottom - copy element by element
        if (dim == 0) {
            for (dim_t i = 0; i < dst_dims[0]; ++i) {
                *dst_ptr = *src_ptr;
                dst_ptr += dst_strides[0];
                src_ptr += src_strides[0];
            }
            return;
        }

        // otherwise recurse to a lower dimenstion
        for (dim_t i = 0; i < dst_dims[dim]; ++i) {
            copy_go(dst_ptr, dst_strides, dst_dims, src_ptr, src_strides,
                    src_dims, dim - 1, linear_end);
            dst_ptr += dst_strides[dim];
            src_ptr += src_strides[dim];
        }
    }
};

template<typename OutT, typename InT>
void copy(Param<OutT> dst, CParam<InT> src) {
    CopyImpl<OutT, InT>::copy(dst, src);
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
