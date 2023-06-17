/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <handle.hpp>

#include <backend.hpp>
#include <platform.hpp>
#include <sparse_handle.hpp>

#include <af/dim4.hpp>

using af::dim4;
using arrayfire::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::createDeviceDataArray;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

namespace arrayfire {

af_array retain(const af_array in) {
    const ArrayInfo &info = getInfo(in, false);
    af_dtype ty           = info.getType();

    if (info.isSparse()) {
        switch (ty) {
            case f32: return retainSparseHandle<float>(in);
            case f64: return retainSparseHandle<double>(in);
            case c32: return retainSparseHandle<detail::cfloat>(in);
            case c64: return retainSparseHandle<detail::cdouble>(in);
            default: TYPE_ERROR(1, ty);
        }
    } else {
        switch (ty) {
            case f32: return retainHandle<float>(in);
            case f64: return retainHandle<double>(in);
            case s32: return retainHandle<int>(in);
            case u32: return retainHandle<uint>(in);
            case u8: return retainHandle<uchar>(in);
            case c32: return retainHandle<detail::cfloat>(in);
            case c64: return retainHandle<detail::cdouble>(in);
            case b8: return retainHandle<char>(in);
            case s64: return retainHandle<intl>(in);
            case u64: return retainHandle<uintl>(in);
            case s16: return retainHandle<short>(in);
            case u16: return retainHandle<ushort>(in);
            case f16: return retainHandle<half>(in);
            default: TYPE_ERROR(1, ty);
        }
    }
}

af_array createHandle(const dim4 &d, af_dtype dtype) {
    // clang-format off
    switch (dtype) {
        case f32: return createHandle<float  >(d);
        case c32: return createHandle<cfloat >(d);
        case f64: return createHandle<double >(d);
        case c64: return createHandle<cdouble>(d);
        case b8:  return createHandle<char   >(d);
        case s32: return createHandle<int    >(d);
        case u32: return createHandle<uint   >(d);
        case u8:  return createHandle<uchar  >(d);
        case s64: return createHandle<intl   >(d);
        case u64: return createHandle<uintl  >(d);
        case s16: return createHandle<short  >(d);
        case u16: return createHandle<ushort >(d);
        case f16: return createHandle<half   >(d);
        default: TYPE_ERROR(3, dtype);
    }
    // clang-format on
}

af_array createHandleFromValue(const dim4 &d, double val, af_dtype dtype) {
    // clang-format off
    switch (dtype) {
        case f32: return createHandleFromValue<float  >(d, val);
        case c32: return createHandleFromValue<cfloat >(d, val);
        case f64: return createHandleFromValue<double >(d, val);
        case c64: return createHandleFromValue<cdouble>(d, val);
        case b8:  return createHandleFromValue<char   >(d, val);
        case s32: return createHandleFromValue<int    >(d, val);
        case u32: return createHandleFromValue<uint   >(d, val);
        case u8:  return createHandleFromValue<uchar  >(d, val);
        case s64: return createHandleFromValue<intl   >(d, val);
        case u64: return createHandleFromValue<uintl  >(d, val);
        case s16: return createHandleFromValue<short  >(d, val);
        case u16: return createHandleFromValue<ushort >(d, val);
        case f16: return createHandleFromValue<half   >(d, val);
        default: TYPE_ERROR(3, dtype);
    }
    // clang-format on
}

af_array createHandleFromDeviceData(const af::dim4 &d, af_dtype dtype,
                                    void *data) {
    // clang-format off
    switch (dtype) {
        case f32: return getHandle(createDeviceDataArray<float  >(d, data, false));
        case c32: return getHandle(createDeviceDataArray<cfloat >(d, data, false));
        case f64: return getHandle(createDeviceDataArray<double >(d, data, false));
        case c64: return getHandle(createDeviceDataArray<cdouble>(d, data, false));
        case b8:  return getHandle(createDeviceDataArray<char   >(d, data, false));
        case s32: return getHandle(createDeviceDataArray<int    >(d, data, false));
        case u32: return getHandle(createDeviceDataArray<uint   >(d, data, false));
        case u8:  return getHandle(createDeviceDataArray<uchar  >(d, data, false));
        case s64: return getHandle(createDeviceDataArray<intl   >(d, data, false));
        case u64: return getHandle(createDeviceDataArray<uintl  >(d, data, false));
        case s16: return getHandle(createDeviceDataArray<short  >(d, data, false));
        case u16: return getHandle(createDeviceDataArray<ushort >(d, data, false));
        case f16: return getHandle(createDeviceDataArray<half   >(d, data, false));
        default: TYPE_ERROR(2, dtype);
    }
    // clang-format on
}

dim4 verifyDims(const unsigned ndims, const dim_t *const dims) {
    DIM_ASSERT(1, ndims >= 1);

    dim4 d(1, 1, 1, 1);

    for (unsigned i = 0; i < ndims; i++) {
        d[i] = dims[i];
        DIM_ASSERT(2, dims[i] >= 1);
    }

    return d;
}

template<typename T>
void releaseHandle(const af_array arr) {
    auto &info     = getInfo(arr);
    int old_device = detail::getActiveDeviceId();
    int array_id   = info.getDevId();
    if (array_id != old_device) {
        detail::setDevice(array_id);
        detail::destroyArray(static_cast<detail::Array<T> *>(arr));
        detail::setDevice(old_device);
    } else {
        detail::destroyArray(static_cast<detail::Array<T> *>(arr));
    }
}

template<typename T>
detail::Array<T> &getCopyOnWriteArray(const af_array &arr) {
    detail::Array<T> *A = static_cast<detail::Array<T> *>(arr);

    if ((af_dtype)af::dtype_traits<T>::af_type != A->getType())
        AF_ERROR("Invalid type for input array.", AF_ERR_INTERNAL);

    ARG_ASSERT(0, A->isSparse() == false);

    if (A->useCount() > 1) { *A = copyArray(*A); }

    return *A;
}

#define INSTANTIATE(TYPE)                                  \
    template void releaseHandle<TYPE>(const af_array arr); \
    template detail::Array<TYPE> &getCopyOnWriteArray<TYPE>(const af_array &arr)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(cfloat);
INSTANTIATE(cdouble);
INSTANTIATE(int);
INSTANTIATE(uint);
INSTANTIATE(intl);
INSTANTIATE(uintl);
INSTANTIATE(uchar);
INSTANTIATE(char);
INSTANTIATE(short);
INSTANTIATE(ushort);
INSTANTIATE(half);

}  // namespace arrayfire
