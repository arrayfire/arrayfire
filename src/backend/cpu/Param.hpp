/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <af/dim4.hpp>
#include <backend.hpp>

namespace cpu
{

using af::dim4;

template<typename T>
class CParam
{
private:
    const T *ptr;

public:
    dim4 dims;
    dim4 strides;

    CParam(const T *iptr, const dim4 &idims, const dim4 &istrides) :
        ptr(iptr)
    {
        for (int i = 0; i < 4; i++) {
            dims[i] = idims[i];
            strides[i] = istrides[i];
        }
    }
    const T *get() const
    {
        return ptr;
    }
};

template<typename T>
class Param
{
private:
    T *ptr;
public:
    dim4 dims;
    dim4 strides;

public:
    Param() : ptr(nullptr)
    {
    }

    Param(T *iptr, const dim4 &idims, const dim4 &istrides) :
        ptr(iptr)
    {
        for (int i = 0; i < 4; i++) {
            dims[i] = idims[i];
            strides[i] = istrides[i];
        }
    }

    T *get()
    {
        return ptr;
    }

    operator CParam<T>() const
    {
        return CParam<T>(const_cast<T *>(ptr), dims, strides);
    }
};

template<typename T> class Array;

template<typename T>
T toParam(const T &val)
{
    return val;
}


template<typename T>
Param<T> toParam(Array<T> &val)
{
    return (Param<T>)(val);
}


template<typename T>
CParam<T> toParam(const Array<T> &val)
{
    return (CParam<T>)(val);
}

}
