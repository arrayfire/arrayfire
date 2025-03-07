/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <common/util.hpp>
#include <handle.hpp>
#include <platform.hpp>
#include <sparse_handle.hpp>
#include <af/backend.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/version.h>

#if defined(USE_MKL)
#include <mkl_service.h>
#endif

#include <cstring>
#include <string>

using af::dim4;
using arrayfire::getSparseArray;
using arrayfire::common::getCacheDirectory;
using arrayfire::common::getEnvVar;
using arrayfire::common::half;
using arrayfire::common::JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::devprop;
using detail::evalFlag;
using detail::getActiveDeviceId;
using detail::getBackend;
using detail::getDeviceCount;
using detail::getDeviceInfo;
using detail::init;
using detail::intl;
using detail::isDoubleSupported;
using detail::isHalfSupported;
using detail::setDevice;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

af_err af_set_backend(const af_backend bknd) {
    try {
        if (bknd != getBackend() && bknd != AF_BACKEND_DEFAULT) {
            return AF_ERR_ARG;
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_backend_count(unsigned* num_backends) {
    *num_backends = 1;
    return AF_SUCCESS;
}

af_err af_get_available_backends(int* result) {
    try {
        *result = getBackend();
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_backend_id(af_backend* result, const af_array in) {
    try {
        if (in) {
            const ArrayInfo& info = getInfo(in, false);
            *result               = info.getBackendId();
        } else {
            return AF_ERR_ARG;
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_device_id(int* device, const af_array in) {
    try {
        if (in) {
            const ArrayInfo& info = getInfo(in, false);
            *device               = static_cast<int>(info.getDevId());
        } else {
            return AF_ERR_ARG;
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_active_backend(af_backend* result) {
    *result = static_cast<af_backend>(getBackend());
    return AF_SUCCESS;
}

af_err af_init() {
    try {
        thread_local std::once_flag flag;
        std::call_once(flag, []() {
            init();
#if defined(USE_MKL) && !defined(USE_STATIC_MKL)
            int errCode = -1;
            // Have used the AF_MKL_INTERFACE_SIZE as regular if's so that
            // we will know if these are not defined when using MKL when a
            // compilation error is generated.
            if (AF_MKL_INTERFACE_SIZE == 4) {
                errCode = mkl_set_interface_layer(MKL_INTERFACE_LP64);
            } else if (AF_MKL_INTERFACE_SIZE == 8) {
                errCode = mkl_set_interface_layer(MKL_INTERFACE_ILP64);
            }
            if (errCode == -1) {
                AF_ERROR(
                    "Intel MKL Interface layer was not specified prior to the "
                    "call and the input parameter is incorrect.",
                    AF_ERR_RUNTIME);
            }
            switch (AF_MKL_THREAD_LAYER) {
                case 0:
                    errCode = mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
                    break;
                case 1:
                    errCode = mkl_set_threading_layer(MKL_THREADING_GNU);
                    break;
                case 2:
                    errCode = mkl_set_threading_layer(MKL_THREADING_INTEL);
                    break;
                case 3:
                    errCode = mkl_set_threading_layer(MKL_THREADING_TBB);
                    break;
            }
            if (errCode == -1) {
                AF_ERROR(
                    "Intel MKL Thread layer was not specified prior to the "
                    "call and the input parameter is incorrect.",
                    AF_ERR_RUNTIME);
            }
#endif
        });
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_info() {
    try {
        printf("%s", getDeviceInfo().c_str());  // NOLINT
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_info_string(char** str, const bool verbose) {
    UNUSED(verbose);  // TODO(umar): Add something useful
    try {
        std::string infoStr = getDeviceInfo();
        void* halloc_ptr    = nullptr;
        af_alloc_host(&halloc_ptr, sizeof(char) * (infoStr.size() + 1));
        memcpy(str, &halloc_ptr, sizeof(void*));

        // Need to do a deep copy
        // str.c_str wont cut it
        infoStr.copy(*str, infoStr.size());
        (*str)[infoStr.size()] = '\0';
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_device_info(char* d_name, char* d_platform, char* d_toolkit,
                      char* d_compute) {
    try {
        devprop(d_name, d_platform, d_toolkit, d_compute);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_dbl_support(bool* available, const int device) {
    try {
        *available = isDoubleSupported(device);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_half_support(bool* available, const int device) {
    try {
        *available = isHalfSupported(device);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_device_count(int* nDevices) {
    try {
        *nDevices = getDeviceCount();
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_device(int* device) {
    try {
        *device = static_cast<int>(getActiveDeviceId());
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_device(const int device) {
    try {
        ARG_ASSERT(0, device >= 0);
        if (setDevice(device) < 0) {
            int ndevices = getDeviceCount();
            if (ndevices == 0) {
                AF_ERROR(
                    "No devices were found on this system. Ensure "
                    "you have installed the device driver as well as the "
                    "necessary runtime libraries for your platform.",
                    AF_ERR_RUNTIME);
            } else {
                char buf[512];
                char err_msg[] =
                    "The device index of %d is out of range. Use a value "
                    "between 0 and %d.";
                snprintf(buf, 512, err_msg, device, ndevices - 1);  // NOLINT
                AF_ERROR(buf, AF_ERR_ARG);
            }
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_sync(const int device) {
    try {
        int dev = device == -1 ? static_cast<int>(getActiveDeviceId()) : device;
        detail::sync(dev);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<typename T>
static inline void eval(af_array arr) {
    getArray<T>(arr).eval();
}

template<typename T>
static inline void sparseEval(af_array arr) {
    getSparseArray<T>(arr).eval();
}

af_err af_eval(af_array arr) {
    try {
        const ArrayInfo& info = getInfo(arr, false);
        af_dtype type         = info.getType();

        if (info.isSparse()) {
            switch (type) {
                case f32: sparseEval<float>(arr); break;
                case f64: sparseEval<double>(arr); break;
                case c32: sparseEval<cfloat>(arr); break;
                case c64: sparseEval<cdouble>(arr); break;
                default: TYPE_ERROR(0, type);
            }
        } else {
            switch (type) {
                case f32: eval<float>(arr); break;
                case f64: eval<double>(arr); break;
                case c32: eval<cfloat>(arr); break;
                case c64: eval<cdouble>(arr); break;
                case s32: eval<int>(arr); break;
                case u32: eval<uint>(arr); break;
                case u8: eval<uchar>(arr); break;
                case b8: eval<char>(arr); break;
                case s64: eval<intl>(arr); break;
                case u64: eval<uintl>(arr); break;
                case s16: eval<short>(arr); break;
                case u16: eval<ushort>(arr); break;
                case f16: eval<half>(arr); break;
                default: TYPE_ERROR(0, type);
            }
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static inline void evalMultiple(int num, af_array* arrayPtrs) {
    Array<T> empty = createEmptyArray<T>(dim4());
    std::vector<Array<T>*> arrays(num, &empty);

    for (int i = 0; i < num; i++) {
        arrays[i] = reinterpret_cast<Array<T>*>(arrayPtrs[i]);
    }

    evalMultiple<T>(arrays);
}

af_err af_eval_multiple(int num, af_array* arrays) {
    try {
        const ArrayInfo& info = getInfo(arrays[0]);
        af_dtype type         = info.getType();
        const dim4& dims      = info.dims();

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
            case f32: evalMultiple<float>(num, arrays); break;
            case f64: evalMultiple<double>(num, arrays); break;
            case c32: evalMultiple<cfloat>(num, arrays); break;
            case c64: evalMultiple<cdouble>(num, arrays); break;
            case s32: evalMultiple<int>(num, arrays); break;
            case u32: evalMultiple<uint>(num, arrays); break;
            case u8: evalMultiple<uchar>(num, arrays); break;
            case b8: evalMultiple<char>(num, arrays); break;
            case s64: evalMultiple<intl>(num, arrays); break;
            case u64: evalMultiple<uintl>(num, arrays); break;
            case s16: evalMultiple<short>(num, arrays); break;
            case u16: evalMultiple<ushort>(num, arrays); break;
            case f16: evalMultiple<half>(num, arrays); break;
            default: TYPE_ERROR(0, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_set_manual_eval_flag(bool flag) {
    try {
        bool& backendFlag = evalFlag();
        backendFlag       = !flag;
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_manual_eval_flag(bool* flag) {
    try {
        bool backendFlag = evalFlag();
        *flag            = !backendFlag;
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_kernel_cache_directory(size_t* length, char* path) {
    try {
        std::string& cache_path = getCacheDirectory();
        if (path == nullptr) {
            ARG_ASSERT(length != nullptr, 1);
            *length = cache_path.size();
        } else {
            size_t min_len = cache_path.size();
            if (length) {
                if (*length < cache_path.size()) {
                    AF_ERROR("Length not sufficient to store the path",
                             AF_ERR_SIZE);
                }
                min_len = std::min(*length, cache_path.size());
            }
            memcpy(path, cache_path.c_str(), min_len);
        }
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_set_kernel_cache_directory(const char* path, int override_env) {
    try {
        ARG_ASSERT(path != nullptr, 1);
        if (override_env) {
            getCacheDirectory() = std::string(path);
        } else {
            auto env_path = getEnvVar(JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME);
            if (env_path.empty()) { getCacheDirectory() = std::string(path); }
        }
    }
    CATCHALL
    return AF_SUCCESS;
}
