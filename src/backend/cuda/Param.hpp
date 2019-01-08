/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <backend.hpp>
#include <types.hpp>

#ifndef __CUDACC_RTC__
#include <af/defines.h>
#endif
#include <algorithm>  // TODO(umar): remove this

namespace cuda {

template<typename T>
class Param {
   public:
    dim_t dims[4];
    dim_t strides[4];
    T *ptr;
    bool is_linear;

    __DH__ Param() : ptr(nullptr) {}

    __DH__ Param(T *iptr, const dim_t *idims, const dim_t *istrides)
        : dims{idims[0], idims[1], idims[2], idims[3]}
        , strides{istrides[0], istrides[1], istrides[2], istrides[3]}
        , ptr(iptr) {
        // for (int i = 0; i < 4; i++) { dims[i] = idims[i]; }
        // for (int i = 0; i < 4; i++) { strides[i] = istrides[i]; }
    }

    __DH__ size_t elements() const noexcept {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }

    Param<T> &operator=(const Param<T> &other) {
        ptr = other.ptr;
        std::copy(other.dims, other.dims + AF_MAX_DIMS, dims);
        std::copy(other.strides, other.strides + AF_MAX_DIMS, strides);
        return *this;
    }
};

template<typename TL, typename TR>
bool equal_shape(const Param<TL> &lhs, const Param<TR> &rhs) {
    return std::equal(lhs.dims, lhs.dims + 4, rhs.dims) &&
           std::equal(lhs.strides, lhs.strides + 4, rhs.strides);
}

template<typename TL, typename TR>
bool less_stride(const Param<TL> &lhs, const Param<TR> &rhs) {
    return std::lexicographical_compare(lhs.strides, lhs.strides + AF_MAX_DIMS,
                                        rhs.strides, rhs.strides + AF_MAX_DIMS);
}

template<typename T>
Param<T> flat(Param<T> in) {
    in.dims[0] = in.elements();
    in.dims[1] = 1;
    in.dims[2] = 1;
    in.dims[3] = 1;
    return in;
}

template<typename T>
class CParam {
   public:
    const T *ptr;
    dim_t dims[4];
    dim_t strides[4];

    __DH__ CParam(const T *iptr, const dim_t *idims, const dim_t *istrides)
        : ptr(iptr) {
        for (int i = 0; i < 4; i++) {
            dims[i]    = idims[i];
            strides[i] = istrides[i];
        }
    }

    __DH__ CParam(Param<T> &in) : ptr(in.ptr) {
        for (int i = 0; i < 4; i++) {
            dims[i]    = in.dims[i];
            strides[i] = in.strides[i];
        }
    }

    __DH__ size_t elements() const noexcept {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }
};
}  // namespace cuda
