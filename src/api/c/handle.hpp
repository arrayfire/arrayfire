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
#include <err_common.hpp>
#include <math.hpp>
#include <copy.hpp>

template<typename T>
static const detail::Array<T> &
getArray(const af_array &arr)
{
    detail::Array<T> *A = reinterpret_cast<detail::Array<T>*>(arr);
    return *A;
}

template<typename T>
static detail::Array<T> &
getWritableArray(const af_array &arr)
{
    const detail::Array<T> &A = getArray<T>(arr);
    return const_cast<detail::Array<T>&>(A);
}

template<typename T>
static af_array
getHandle(const detail::Array<T> &A)
{
    af_array arr = reinterpret_cast<af_array>(&A);
    return arr;
}

template<typename T>
static af_array createHandle(af::dim4 d)
{
    return getHandle(*detail::createEmptyArray<T>(d));
}

template<typename T>
static af_array createHandleFromValue(af::dim4 d, double val)
{
    return getHandle(*detail::createValueArray<T>(d, detail::scalar<T>(val)));
}

template<typename T>
static af_array createHandleFromData(af::dim4 d, const T * const data)
{
    return getHandle(*detail::createHostDataArray<T>(d, data));
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
    return getHandle<T>(*detail::copyArray<T>(inArray));
}

template<typename T>
static void destroyHandle(const af_array arr)
{
    detail::destroyArray(getWritableArray<T>(arr));
}

af_array weakCopy(const af_array in);
