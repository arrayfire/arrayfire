/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/SparseArray.hpp>
#include <common/defines.hpp>
#include <common/unique_handle.hpp>
#include <cudaDataType.hpp>
#include <cusparse_v2.h>
#include <err_cuda.hpp>

#if defined(AF_USE_NEW_CUSPARSE_API)
namespace arrayfire {
namespace cuda {

template<typename T>
cusparseStatus_t createSpMatDescr(
    cusparseSpMatDescr_t *out, const arrayfire::common::SparseArray<T> &arr) {
    switch (arr.getStorage()) {
        case AF_STORAGE_CSR: {
            return cusparseCreateCsr(
                out, arr.dims()[0], arr.dims()[1], arr.getNNZ(),
                (void *)arr.getRowIdx().get(), (void *)arr.getColIdx().get(),
                (void *)arr.getValues().get(), CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, getType<T>());
        }
#if CUSPARSE_VERSION >= 11300
        case AF_STORAGE_CSC: {
            return cusparseCreateCsc(
                out, arr.dims()[0], arr.dims()[1], arr.getNNZ(),
                (void *)arr.getColIdx().get(), (void *)arr.getRowIdx().get(),
                (void *)arr.getValues().get(), CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, getType<T>());
        }
#else
        case AF_STORAGE_CSC:
            CUDA_NOT_SUPPORTED(
                "Sparse not supported for CSC on this version of the CUDA "
                "Toolkit");
#endif
        case AF_STORAGE_COO: {
            return cusparseCreateCoo(
                out, arr.dims()[0], arr.dims()[1], arr.getNNZ(),
                (void *)arr.getColIdx().get(), (void *)arr.getRowIdx().get(),
                (void *)arr.getValues().get(), CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, getType<T>());
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

}  // namespace cuda
}  // namespace arrayfire
#endif

// clang-format off
DEFINE_HANDLER(cusparseHandle_t, cusparseCreate, cusparseDestroy);
DEFINE_HANDLER(cusparseMatDescr_t, cusparseCreateMatDescr, cusparseDestroyMatDescr);
#if defined(AF_USE_NEW_CUSPARSE_API)
DEFINE_HANDLER(cusparseSpMatDescr_t, arrayfire::cuda::createSpMatDescr, cusparseDestroySpMat);
DEFINE_HANDLER(cusparseDnVecDescr_t, cusparseCreateDnVec, cusparseDestroyDnVec);
DEFINE_HANDLER(cusparseDnMatDescr_t, cusparseCreateDnMat, cusparseDestroyDnMat);
#endif
// clang-format on

namespace arrayfire {
namespace cuda {

const char *errorString(cusparseStatus_t err);

#define CUSPARSE_CHECK(fn)                                                    \
    do {                                                                      \
        cusparseStatus_t _error = fn;                                         \
        if (_error != CUSPARSE_STATUS_SUCCESS) {                              \
            char _err_msg[1024];                                              \
            snprintf(_err_msg, sizeof(_err_msg), "CUSPARSE Error (%d): %s\n", \
                     (int)(_error), arrayfire::cuda::errorString(_error));    \
                                                                              \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);                              \
        }                                                                     \
    } while (0)

}  // namespace cuda
}  // namespace arrayfire
