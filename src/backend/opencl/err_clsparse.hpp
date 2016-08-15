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
#include <clSPARSE.h>

static const char * _clsparseGetResultString(clsparseStatus st)
{
    switch (st)
    {
        case clsparseSuccess:                       return "Success";
        case clsparseInvalidValue:                  return "Invalid value";
        case clsparseInvalidCommandQueue:           return "Invalid queue";
        case clsparseInvalidContext:                return "Invalid context";
        case clsparseInvalidMemObject:              return "Invalid memory object";
        case clsparseInvalidDevice:                 return "Invalid device";
        case clsparseInvalidEventWaitList:          return "Invalid event list";
        case clsparseInvalidEvent:                  return "Invalid event";
        case clsparseOutOfResources:                return "Out of resources";
        case clsparseOutOfHostMemory:               return "Out of host memory";
        case clsparseInvalidOperation:              return "Invalid operation";
        case clsparseCompilerNotAvailable:          return "Compiler not available";
        case clsparseBuildProgramFailure:           return "Build program failure";
        case clsparseInvalidKernelArgs:             return "Invalid kernel arguments";

        case clsparseNotImplemented:                return "Not implemented";
        case clsparseNotInitialized:                return "clSPARSE Not initialized";
        case clsparseStructInvalid:                 return "Struct invalid";
        case clsparseInvalidSize:                   return "Invalid size";
        case clsparseInvalidMemObj:                 return "Invalid memory object";
        case clsparseInsufficientMemory:            return "Insufficient Memory";
        case clsparseInvalidControlObject:          return "Invalid control object";
        case clsparseInvalidFile:                   return "Invalid file";
        case clsparseInvalidFileFormat:             return "Invalid file format";
        case clsparseInvalidKernelExecution:        return "Invalid kernel execution";
        case clsparseInvalidType:                   return "Invalid type";

        case clsparseInvalidSolverControlObject:    return "Invalid solver control object";
        case clsparseInvalidSystemSize:             return "Invalid system size";
        case clsparseIterationsExceeded:            return "Iterations exceeded";
        case clsparseToleranceNotReached:           return "Tolerance not reached";
        case clsparseSolverError:                   return "Solver error";
        default:                                    return "Unknown clSPARSE Error";
    }

    return "Unknown error";
}

#define CLSPARSE_CHECK(fn) do {                     \
        clsparseStatus _clsparse_st = fn;           \
        if (_clsparse_st != clsparseSuccess) {      \
            char clsparse_st_msg[1024];             \
            snprintf(clsparse_st_msg,               \
                     sizeof(clsparse_st_msg),       \
                     "clsparse Error (%d): %s\n",   \
                     (int)(_clsparse_st),           \
                     _clsparseGetResultString(      \
                         _clsparse_st));            \
                                                    \
            AF_ERROR(clsparse_st_msg,               \
                     AF_ERR_INTERNAL);              \
        }                                           \
    } while(0)

