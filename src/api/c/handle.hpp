/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>
#include <Array.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <math.hpp>
#include <copy.hpp>
#include <cast.hpp>
#include <af/dim4.hpp>

const ArrayInfo& getInfo(const af_array arr, bool sparse_check = true, bool device_check = true);

// Implemented in src/api/c/moddims.cpp
template<typename T>
detail::Array<T> modDims(const detail::Array<T>& in, const af::dim4 &newDims);

template<typename T>
static const detail::Array<T> &
getArray(const af_array &arr)
{
    detail::Array<T> *A = reinterpret_cast<detail::Array<T>*>(arr);
    if ((af_dtype)af::dtype_traits<T>::af_type != A->getType())
        AF_ERROR("Invalid type for input array.", AF_ERR_INTERNAL);
    ARG_ASSERT(0, A->isSparse() == false);
    return *A;
}

template<typename To>
detail::Array<To> castArray(const af_array &in)
{
    using detail::cfloat;
    using detail::cdouble;
    using detail::uint;
    using detail::uchar;
    using detail::ushort;

    const ArrayInfo& info = getInfo(in);
    switch (info.getType()) {
    case f32: return detail::cast<To, float  >(getArray<float  >(in));
    case f64: return detail::cast<To, double >(getArray<double >(in));
    case c32: return detail::cast<To, cfloat >(getArray<cfloat >(in));
    case c64: return detail::cast<To, cdouble>(getArray<cdouble>(in));
    case s32: return detail::cast<To, int    >(getArray<int    >(in));
    case u32: return detail::cast<To, uint   >(getArray<uint   >(in));
    case u8 : return detail::cast<To, uchar  >(getArray<uchar  >(in));
    case b8 : return detail::cast<To, char   >(getArray<char   >(in));
    case s64: return detail::cast<To, intl   >(getArray<intl   >(in));
    case u64: return detail::cast<To, uintl  >(getArray<uintl  >(in));
    case s16: return detail::cast<To, short  >(getArray<short  >(in));
    case u16: return detail::cast<To, ushort >(getArray<ushort >(in));
    default: TYPE_ERROR(1, info.getType());
    }
}

template<typename T>
static detail::Array<T> &
getWritableArray(const af_array &arr)
{
    const detail::Array<T> &A = getArray<T>(arr);
    ARG_ASSERT(0, A.isSparse() == false);
    return const_cast<detail::Array<T>&>(A);
}

template<typename T>
static af_array
getHandle(const detail::Array<T> &A)
{
    detail::Array<T> *ret = detail::initArray<T>();
    *ret = A;
    af_array arr = reinterpret_cast<af_array>(ret);
    return arr;
}

template<typename T>
static af_array createHandle(af::dim4 d)
{
    return getHandle(detail::createEmptyArray<T>(d));
}

template<typename T>
static af_array createHandleFromValue(af::dim4 d, double val)
{
    return getHandle(detail::createValueArray<T>(d, detail::scalar<T>(val)));
}

template<typename T>
static af_array createHandleFromData(af::dim4 d, const T * const data)
{
    return getHandle(detail::createHostDataArray<T>(d, data));
}

template<typename T>
static void copyData(T *data, const af_array &arr)
{
    return detail::copyData(data, getArray<T>(arr));
}

template<typename T>
static af_array copyArray(const af_array in)
{
    const detail::Array<T> &inArray = getArray<T>(in);
    return getHandle<T>(detail::copyArray<T>(inArray));
}

template<typename T>
static void releaseHandle(const af_array arr)
{
    detail::destroyArray(reinterpret_cast<detail::Array<T>*>(arr));
}

af_array retain(const af_array in);

af::dim4 verifyDims(const unsigned ndims, const dim_t * const dims);


template<typename T>
static detail::Array<T> &
getCopyOnWriteArray(const af_array &arr)
{
    detail::Array<T> *A = reinterpret_cast<detail::Array<T>*>(arr);

    if ((af_dtype)af::dtype_traits<T>::af_type != A->getType())
        AF_ERROR("Invalid type for input array.", AF_ERR_INTERNAL);

    ARG_ASSERT(0, A->isSparse() == false);

    if (A->useCount() > 1) {
        *A = copyArray(*A);
    }

    return *A;
}
