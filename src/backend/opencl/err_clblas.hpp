/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <stdio.h>
#include <err_common.hpp>
#include <clBLAS.h>

static const char * _clblasGetResultString(clblasStatus st)
{
    switch (st)
    {
    case clblasSuccess:              return "Success";
    case clblasInvalidValue:         return "Invalid value";
    case clblasInvalidCommandQueue:  return "Invalid queue";
    case clblasInvalidContext:       return "Invalid context";
    case clblasInvalidMemObject:     return "Invalid memory object";
    case clblasInvalidDevice:        return "Invalid device";
    case clblasInvalidEventWaitList: return "Invalid event list";
    case clblasOutOfResources:       return "Out of resources";
    case clblasOutOfHostMemory:      return "Out of host memory";
    case clblasInvalidOperation:     return "Invalid operation";
    case clblasCompilerNotAvailable: return "Compiler not available";
    case clblasBuildProgramFailure:  return "Build program failure";
    case clblasNotImplemented:       return "Not implemented";
    case clblasNotInitialized:       return "CLBLAS Not initialized";
    case clblasInvalidMatA:          return "Invalid matrix A";
    case clblasInvalidMatB:          return "Invalid matrix B";
    case clblasInvalidMatC:          return "Invalid matrix C";
    case clblasInvalidVecX:          return "Invalid vector X";
    case clblasInvalidVecY:          return "Invalid vector Y";
    case clblasInvalidDim:           return "Invalid dimension";
    case clblasInvalidLeadDimA:      return "Invalid lda";
    case clblasInvalidLeadDimB:      return "Invalid ldb";
    case clblasInvalidLeadDimC:      return "Invalid ldc";
    case clblasInvalidIncX:          return "Invalid incx";
    case clblasInvalidIncY:          return "Invalid incy";
    case clblasInsufficientMemMatA:  return  "Insufficient Memory for Matrix A";
    case clblasInsufficientMemMatB:  return  "Insufficient Memory for Matrix B";
    case clblasInsufficientMemMatC:  return  "Insufficient Memory for Matrix C";
    case clblasInsufficientMemVecX:  return  "Insufficient Memory for Vector X";
    case clblasInsufficientMemVecY:  return  "Insufficient Memory for Vector Y";
    }

    return "Unknown error";
}

#define CLBLAS_CHECK(fn) do {                   \
        clblasStatus _clblas_st = fn;           \
        if (_clblas_st != clblasSuccess) {     \
            char clblas_st_msg[1024];           \
            snprintf(clblas_st_msg,             \
                     sizeof(clblas_st_msg),     \
                     "clblas Error (%d): %s\n", \
                     (int)(_clblas_st),         \
                     _clblasGetResultString(    \
                         _clblas_st));          \
                                                \
            AF_ERROR(clblas_st_msg,             \
                     AF_ERR_INTERNAL);          \
        }                                       \
    } while(0)
