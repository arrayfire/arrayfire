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

template<typename T>
class CParam
{
private:
    const T *m_ptr;
    af::dim4 m_dims;
    af::dim4 m_strides;

public:
    CParam(const T *iptr, const af::dim4 &idims, const af::dim4 &istrides) :
        m_ptr(iptr)
    {
        for (int i = 0; i < 4; i++) {
            m_dims[i] = idims[i];
            m_strides[i] = istrides[i];
        }
    }

    const T *get() const
    {
        return m_ptr;
    }

    af::dim4 dims() const
    {
        return m_dims;
    }

    af::dim4 strides() const
    {
        return m_strides;
    }

    dim_t dims(int i) const
    {
        return m_dims[i];
    }

    dim_t strides(int i) const
    {
        return m_strides[i];
    }
};

template<typename T>
class Param
{
private:
    T *m_ptr;
    af::dim4 m_dims;
    af::dim4 m_strides;

public:
    Param() : m_ptr(nullptr)
    {
    }

    Param(T *iptr, const af::dim4 &idims, const af::dim4 &istrides) :
        m_ptr(iptr)
    {
        for (int i = 0; i < 4; i++) {
            m_dims[i] = idims[i];
            m_strides[i] = istrides[i];
        }
    }

    T *get()
    {
        return m_ptr;
    }

    operator CParam<T>() const
    {
        return CParam<T>(const_cast<T *>(m_ptr), m_dims, m_strides);
    }

    af::dim4 dims() const
    {
        return m_dims;
    }

    af::dim4 strides() const
    {
        return m_strides;
    }

    dim_t dims(int i) const
    {
        return m_dims[i];
    }

    dim_t strides(int i) const
    {
        return m_strides[i];
    }
};

template<typename T> class Array;

// These functions are needed to convert Array<T> to Param<T> when queueing up functions.
// This is necessary because the memory used by Array<T> can be put back into the queue faster.
// This is fine becacuse we only have 1 compute queue. This ensures there's no race conditions.
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
