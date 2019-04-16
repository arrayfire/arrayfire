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
#include <af/defines.h>

namespace cuda {

template<typename T>
class Param {
   public:
    T *ptr;
    dim_t dims[4];
    dim_t strides[4];

    __DH__ Param() : ptr(nullptr) {}

    __DH__ Param(T *iptr, const dim_t *idims, const dim_t *istrides)
        : ptr(iptr) {
        for (int i = 0; i < 4; i++) {
            dims[i]    = idims[i];
            strides[i] = istrides[i];
        }
    }
    size_t elements() const noexcept {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }
};

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

    __DH__ ~CParam() {}
};

}  // namespace cuda
