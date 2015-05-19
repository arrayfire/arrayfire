/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <backend.hpp>

namespace cuda
{

template<typename T>
struct Param
{
    T *ptr;
    dim_t dims[4];
    dim_t strides[4];
};

template<typename T>
class CParam
{
public:
    const T *ptr;
    dim_t dims[4];
    dim_t strides[4];

    __DH__ CParam(const T *iptr, const dim_t *idims, const dim_t *istrides) :
        ptr(iptr)
    {
        for (int i = 0; i < 4; i++) {
            dims[i] = idims[i];
            strides[i] = istrides[i];
        }
    }

    __DH__ CParam(Param<T> &in) : ptr(in.ptr)
    {
        for (int i = 0; i < 4; i++) {
            dims[i] = in.dims[i];
            strides[i] = in.strides[i];
        }
    }

    __DH__ ~CParam() {}
};

}
