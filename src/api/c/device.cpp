/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/device.h>
#include <af/version.h>
#include <af/backend.h>
#include <backend.hpp>
#include <platform.hpp>
#include <Array.hpp>
#include <handle.hpp>
#include <sparse_handle.hpp>
#include <common/err_common.hpp>
#include <cstring>

using namespace detail;

af_err af_set_backend(const af_backend bknd)
{
    try {
        ARG_ASSERT(0, bknd==getBackend());
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_backend_count(unsigned* num_backends)
{
    *num_backends = 1;
    return AF_SUCCESS;
}

af_err af_get_available_backends(int* result)
{
    try {
        *result = getBackend();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_backend_id(af_backend *result, const af_array in)
{
    try {
        ARG_ASSERT(1, in != 0);
        const ArrayInfo& info = getInfo(in, false, false);
        *result = info.getBackendId();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_device_id(int *device, const af_array in)
{
    try {
        ARG_ASSERT(1, in != 0);
        const ArrayInfo& info = getInfo(in, false, false);
        *device = info.getDevId();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_active_backend(af_backend *result)
{
    *result = (af_backend)getBackend();
    return AF_SUCCESS;
}

af_err af_init()
{
    try {
        thread_local std::once_flag flag;
        std::call_once(flag, []() {
                getDeviceInfo();
            });
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_info()
{
    try {
        printf("%s", getDeviceInfo().c_str());
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_info_string(char **str, const bool verbose)
{
    try {
        std::string infoStr = getDeviceInfo();
        af_alloc_host((void**)str, sizeof(char) * (infoStr.size() + 1));

        // Need to do a deep copy
        // str.c_str wont cut it
        infoStr.copy(*str, infoStr.size());
        (*str)[infoStr.size()] = '\0';
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_device_info(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
{
    try {
        devprop(d_name, d_platform, d_toolkit, d_compute);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_dbl_support(bool* available, const int device)
{
    try {
        *available = isDoubleSupported(device);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_device_count(int *nDevices)
{
    try {
        *nDevices = getDeviceCount();
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_device(int *device)
{
    try {
        *device = getActiveDeviceId();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_device(const int device)
{
    try {
        ARG_ASSERT(0, device >= 0);
        ARG_ASSERT(0, setDevice(device) >= 0);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_sync(const int device)
{
    try {
        int dev = device == -1 ? getActiveDeviceId() : device;
        detail::sync(dev);
    } CATCHALL;
    return AF_SUCCESS;
}


template<typename T>
static inline void eval(af_array arr)
{
    getArray<T>(arr).eval();
    return;
}

template<typename T>
static inline void sparseEval(af_array arr)
{
    getSparseArray<T>(arr).eval();
    return;
}

af_err af_eval(af_array arr)
{
    try {
        const ArrayInfo& info = getInfo(arr, false);
        af_dtype type = info.getType();

        if(info.isSparse()) {
            switch(type) {
                case f32: sparseEval<float  >(arr); break;
                case f64: sparseEval<double >(arr); break;
                case c32: sparseEval<cfloat >(arr); break;
                case c64: sparseEval<cdouble>(arr); break;
                default : TYPE_ERROR(0, type);
            }
        } else {
            switch (type) {
                case f32: eval<float  >(arr); break;
                case f64: eval<double >(arr); break;
                case c32: eval<cfloat >(arr); break;
                case c64: eval<cdouble>(arr); break;
                case s32: eval<int    >(arr); break;
                case u32: eval<uint   >(arr); break;
                case u8 : eval<uchar  >(arr); break;
                case b8 : eval<char   >(arr); break;
                case s64: eval<intl   >(arr); break;
                case u64: eval<uintl  >(arr); break;
                case s16: eval<short  >(arr); break;
                case u16: eval<ushort >(arr); break;
                default: TYPE_ERROR(0, type);
            }
        }
    } CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static inline void evalMultiple(int num, af_array *arrayPtrs)
{
    Array<T> empty = createEmptyArray<T>(dim4());
    std::vector<Array<T>*> arrays(num, &empty);

    for (int i = 0; i < num; i++) {
        arrays[i] = reinterpret_cast<Array<T>*>(arrayPtrs[i]);
    }

    evalMultiple<T>(arrays);
    return;
}

af_err af_eval_multiple(int num, af_array *arrays)
{
    try {
        const ArrayInfo& info = getInfo(arrays[0]);
        af_dtype type = info.getType();
        dim4 dims = info.dims();

        for (int i = 1; i < num; i++) {
            const ArrayInfo& currInfo = getInfo(arrays[i]);

            // FIXME: This needs to be removed when new functionality is added
            if (type != currInfo.getType()) {
                AF_ERROR("All arrays must be of same type", AF_ERR_TYPE);
            }

            if (dims != currInfo.dims()) {
                AF_ERROR("All arrays must be of same size", AF_ERR_SIZE);
            }
        }

        switch (type) {
        case f32: evalMultiple<float  >(num, arrays); break;
        case f64: evalMultiple<double >(num, arrays); break;
        case c32: evalMultiple<cfloat >(num, arrays); break;
        case c64: evalMultiple<cdouble>(num, arrays); break;
        case s32: evalMultiple<int    >(num, arrays); break;
        case u32: evalMultiple<uint   >(num, arrays); break;
        case u8 : evalMultiple<uchar  >(num, arrays); break;
        case b8 : evalMultiple<char   >(num, arrays); break;
        case s64: evalMultiple<intl   >(num, arrays); break;
        case u64: evalMultiple<uintl  >(num, arrays); break;
        case s16: evalMultiple<short  >(num, arrays); break;
        case u16: evalMultiple<ushort >(num, arrays); break;
        default:
            TYPE_ERROR(0, type);
        }
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_set_manual_eval_flag(bool flag)
{
    try {
        bool& backendFlag = evalFlag();
        backendFlag = !flag;
    } CATCHALL;
    return AF_SUCCESS;
}


af_err af_get_manual_eval_flag(bool *flag)
{
    try {
        bool backendFlag = evalFlag();
        *flag = !backendFlag;
    } CATCHALL;
    return AF_SUCCESS;
}
