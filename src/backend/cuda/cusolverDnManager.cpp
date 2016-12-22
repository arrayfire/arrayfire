/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_CUDA_LINEAR_ALGEBRA)

#include <cusolverDnManager.hpp>
#include <platform.hpp>
#include <debug_cuda.hpp>

#include <stdexcept>
#include <string>
#include <iostream>
#include <boost/scoped_ptr.hpp>

namespace cusolver
{

const char *errorString(cusolverStatus_t err)
{
    switch(err) {
    case    CUSOLVER_STATUS_SUCCESS                     :   return "CUSOLVER_STATUS_SUCCESS"                    ;
    case    CUSOLVER_STATUS_NOT_INITIALIZED             :   return "CUSOLVER_STATUS_NOT_INITIALIZED"            ;
    case    CUSOLVER_STATUS_ALLOC_FAILED                :   return "CUSOLVER_STATUS_ALLOC_FAILED"               ;
    case    CUSOLVER_STATUS_INVALID_VALUE               :   return "CUSOLVER_STATUS_INVALID_VALUE"              ;
    case    CUSOLVER_STATUS_ARCH_MISMATCH               :   return "CUSOLVER_STATUS_ARCH_MISMATCH"              ;
    case    CUSOLVER_STATUS_MAPPING_ERROR               :   return "CUSOLVER_STATUS_MAPPING_ERROR"              ;
    case    CUSOLVER_STATUS_EXECUTION_FAILED            :   return "CUSOLVER_STATUS_EXECUTION_FAILED"           ;
    case    CUSOLVER_STATUS_INTERNAL_ERROR              :   return "CUSOLVER_STATUS_INTERNAL_ERROR"             ;
    case    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED   :   return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED"  ;
    case    CUSOLVER_STATUS_NOT_SUPPORTED               :   return "CUSOLVER_STATUS_NOT_SUPPORTED"              ;
    case    CUSOLVER_STATUS_ZERO_PIVOT                  :   return "CUSOLVER_STATUS_ZERO_PIVOT"                 ;
    case    CUSOLVER_STATUS_INVALID_LICENSE             :   return "CUSOLVER_STATUS_INVALID_LICENSE"            ;
    default                                             :   return "UNKNOWN";
    }
}


cusolverDnHandle::cusolverDnHandle()
    : handle(0)
{
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
}

cusolverDnHandle::~cusolverDnHandle()
{
    cusolverDnDestroy(handle);
}

cusolverDnHandle_t cusolverDnHandle::get() const
{
    return handle;
}

}

#endif
