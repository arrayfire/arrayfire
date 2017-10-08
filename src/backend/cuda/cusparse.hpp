/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <cusparse_v2.h>
#include <common/defines.hpp>
#include <common/MatrixAlgebraHandle.hpp>

namespace cuda
{
typedef cusparseHandle_t SparseHandle;

const char * errorString(cusparseStatus_t err);

#define CUSPARSE_CHECK(fn) do {                         \
        cusparseStatus_t _error = fn;                   \
        if (_error != CUSPARSE_STATUS_SUCCESS) {        \
            char _err_msg[1024];                        \
            snprintf(_err_msg, sizeof(_err_msg),        \
                     "CUSPARSE Error (%d): %s\n",       \
                     (int)(_error),                     \
                     errorString( _error));             \
                                                        \
            AF_ERROR(_err_msg, AF_ERR_INTERNAL);        \
        }                                               \
    } while(0)

class cusparseHandle : public common::MatrixAlgebraHandle<cusparseHandle, SparseHandle>
{
    public:
        void createHandle(SparseHandle* handle);
        void destroyHandle(SparseHandle handle) {
            cusparseDestroy(handle);
        }
};
}
