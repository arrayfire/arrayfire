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
#include <handle.hpp>
#include <af/dim4.hpp>

#include <common/SparseArray.hpp>

const common::SparseArrayBase& getSparseArrayBase(const af_array arr, bool device_check = true);

template<typename T>
const common::SparseArray<T>& getSparseArray(const af_array &arr)
{
    common::SparseArray<T> *A = reinterpret_cast<common::SparseArray<T>*>(arr);
    ARG_ASSERT(0, A->isSparse() == true);
    return *A;
}

template<typename T>
common::SparseArray<T>& getWritableSparseArray(const af_array &arr)
{
    const common::SparseArray<T> &A = getSparseArray<T>(arr);
    ARG_ASSERT(0, A.isSparse() == true);
    return const_cast<common::SparseArray<T>&>(A);
}

template<typename T>
static af_array
getHandle(const common::SparseArray<T> &A)
{
    common::SparseArray<T> *ret = common::initSparseArray<T>();
    *ret = A;
    af_array arr = reinterpret_cast<af_array>(ret);
    return arr;
}

template<typename T>
static void releaseSparseHandle(const af_array arr)
{
    common::destroySparseArray(reinterpret_cast<common::SparseArray<T>*>(arr));
}

template<typename T>
af_array retainSparseHandle(const af_array in)
{
    common::SparseArray<T> *sparse = reinterpret_cast<common::SparseArray<T> *>(in);
    common::SparseArray<T> *out = common::initSparseArray<T>();
    *out = *sparse;
    return reinterpret_cast<af_array>(out);
}

// based on castArray in handle.hpp
template<typename To>
common::SparseArray<To> castSparse(const af_array &in)
{
    const ArrayInfo& info = getInfo(in, false, true);
    using namespace common;

#define CAST_SPARSE(Ti) do {                                            \
        const SparseArray<Ti> sparse = getSparseArray<Ti>(in);          \
        Array<To> values = detail::cast<To, Ti>(sparse.getValues());    \
        return createArrayDataSparseArray(sparse.dims(), values,        \
                                          sparse.getRowIdx(),           \
                                          sparse.getColIdx(),           \
                                          sparse.getStorage());         \
    } while(0)

    switch(info.getType()) {
    case f32: CAST_SPARSE(float);
    case f64: CAST_SPARSE(double);
    case c32: CAST_SPARSE(cfloat);
    case c64: CAST_SPARSE(cdouble);
    default: TYPE_ERROR(1, info.getType());
    }
}

template<typename T>
static af_array copySparseArray(const af_array in)
{
  const common::SparseArray<T> &inArray = getSparseArray<T>(in);
  return getHandle<T>(common::copySparseArray<T>(inArray));
}
