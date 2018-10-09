/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <cudnn.hpp>
#include <platform.hpp>

namespace cuda {
const char *errorString(cudnnStatus_t err) {
    switch (err) {
        case CUDNN_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED:
            return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED";
        case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_INVALID_VALUE: return "CUDNN_STATUS_INVALID_VALUE";
        case CUDNN_STATUS_ARCH_MISMATCH: return "CUDNN_STATUS_ARCH_MISMATCH";
        case CUDNN_STATUS_MAPPING_ERROR: return "CUDNN_STATUS_MAPPING_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED:
            return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_LICENSE_ERROR: return "CUDNN_STATUS_LICENSE_ERROR";
        case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
            return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
        case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
            return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
        case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
            return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
        default: return "UNKNOWN";
    }
}

void cudnnHandle::createHandle(NNHandle *handle) {
    CUDNN_CHECK(cudnnCreate(handle));
}
}
