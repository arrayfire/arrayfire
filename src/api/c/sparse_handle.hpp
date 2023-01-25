/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <math.hpp>
#include <af/array.h>
#include <af/dim4.hpp>

#include <common/SparseArray.hpp>

namespace arrayfire {

const common::SparseArrayBase &getSparseArrayBase(const af_array in,
                                                  bool device_check = true);

template<typename T>
const common::SparseArray<T> &getSparseArray(const af_array &arr) {
    const common::SparseArray<T> *A =
        static_cast<const common::SparseArray<T> *>(arr);
    ARG_ASSERT(0, A->isSparse() == true);
    return *A;
}

template<typename T>
common::SparseArray<T> &getSparseArray(af_array &arr) {
    common::SparseArray<T> *A = static_cast<common::SparseArray<T> *>(arr);
    ARG_ASSERT(0, A->isSparse() == true);
    return *A;
}

template<typename T>
static af_array getHandle(const common::SparseArray<T> &A) {
    common::SparseArray<T> *ret = new common::SparseArray<T>(A);
    return static_cast<af_array>(ret);
}

template<typename T>
static void releaseSparseHandle(const af_array arr) {
    common::destroySparseArray(static_cast<common::SparseArray<T> *>(arr));
}

template<typename T>
af_array retainSparseHandle(const af_array in) {
    const common::SparseArray<T> *sparse =
        static_cast<const common::SparseArray<T> *>(in);
    common::SparseArray<T> *out = new common::SparseArray<T>(*sparse);
    return static_cast<af_array>(out);
}

// based on castArray in handle.hpp
template<typename To>
common::SparseArray<To> castSparse(const af_array &in) {
    const ArrayInfo &info = getInfo(in, false, true);
    using namespace common;

#define CAST_SPARSE(Ti)                                                          \
    do {                                                                         \
        const SparseArray<Ti> sparse = getSparseArray<Ti>(in);                   \
        detail::Array<To> values     = common::cast<To, Ti>(sparse.getValues()); \
        return createArrayDataSparseArray(                                       \
            sparse.dims(), values, sparse.getRowIdx(), sparse.getColIdx(),       \
            sparse.getStorage());                                                \
    } while (0)

    switch (info.getType()) {
        case f32: CAST_SPARSE(float);
        case f64: CAST_SPARSE(double);
        case c32: CAST_SPARSE(detail::cfloat);
        case c64: CAST_SPARSE(detail::cdouble);
        default: TYPE_ERROR(1, info.getType());
    }
}

template<typename T>
static af_array copySparseArray(const af_array in) {
    const common::SparseArray<T> &inArray = getSparseArray<T>(in);
    return getHandle<T>(common::copySparseArray<T>(inArray));
}

}  // namespace arrayfire

using arrayfire::getHandle;
