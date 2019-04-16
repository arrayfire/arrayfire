/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <clblast.h>
#include <common/err_common.hpp>
#include <stdio.h>
#include <mutex>

static const char* _clblastGetResultString(clblast::StatusCode st) {
    switch (st) {
        // Status codes in common with the OpenCL standard
        case clblast::StatusCode::kSuccess: return "CL_SUCCESS";
        case clblast::StatusCode::kOpenCLCompilerNotAvailable:
            return "CL_COMPILER_NOT_AVAILABLE";
        case clblast::StatusCode::kTempBufferAllocFailure:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case clblast::StatusCode::kOpenCLOutOfResources:
            return "CL_OUT_OF_RESOURCES";
        case clblast::StatusCode::kOpenCLOutOfHostMemory:
            return "CL_OUT_OF_HOST_MEMORY";
        case clblast::StatusCode::kOpenCLBuildProgramFailure:
            return "CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error";
        case clblast::StatusCode::kInvalidValue: return "CL_INVALID_VALUE";
        case clblast::StatusCode::kInvalidCommandQueue:
            return "CL_INVALID_COMMAND_QUEUE";
        case clblast::StatusCode::kInvalidMemObject:
            return "CL_INVALID_MEM_OBJECT";
        case clblast::StatusCode::kInvalidBinary: return "CL_INVALID_BINARY";
        case clblast::StatusCode::kInvalidBuildOptions:
            return "CL_INVALID_BUILD_OPTIONS";
        case clblast::StatusCode::kInvalidProgram: return "CL_INVALID_PROGRAM";
        case clblast::StatusCode::kInvalidProgramExecutable:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case clblast::StatusCode::kInvalidKernelName:
            return "CL_INVALID_KERNEL_NAME";
        case clblast::StatusCode::kInvalidKernelDefinition:
            return "CL_INVALID_KERNEL_DEFINITION";
        case clblast::StatusCode::kInvalidKernel: return "CL_INVALID_KERNEL";
        case clblast::StatusCode::kInvalidArgIndex:
            return "CL_INVALID_ARG_INDEX";
        case clblast::StatusCode::kInvalidArgValue:
            return "CL_INVALID_ARG_VALUE";
        case clblast::StatusCode::kInvalidArgSize: return "CL_INVALID_ARG_SIZE";
        case clblast::StatusCode::kInvalidKernelArgs:
            return "CL_INVALID_KERNEL_ARGS";
        case clblast::StatusCode::kInvalidLocalNumDimensions:
            return "CL_INVALID_WORK_DIMENSION: Too many thread dimensions";
        case clblast::StatusCode::kInvalidLocalThreadsTotal:
            return "CL_INVALID_WORK_GROUP_SIZE: Too many threads in total";
        case clblast::StatusCode::kInvalidLocalThreadsDim:
            return "CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension";
        case clblast::StatusCode::kInvalidGlobalOffset:
            return "CL_INVALID_GLOBAL_OFFSET";
        case clblast::StatusCode::kInvalidEventWaitList:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case clblast::StatusCode::kInvalidEvent: return "CL_INVALID_EVENT";
        case clblast::StatusCode::kInvalidOperation:
            return "CL_INVALID_OPERATION";
        case clblast::StatusCode::kInvalidBufferSize:
            return "CL_INVALID_BUFFER_SIZE";
        case clblast::StatusCode::kInvalidGlobalWorkSize:
            return "CL_INVALID_GLOBAL_WORK_SIZE";

        // Status codes in common with the clBLAS library
        case clblast::StatusCode::kNotImplemented:
            return "Routine or functionality not implemented yet";
        case clblast::StatusCode::kInvalidMatrixA:
            return "Matrix A is not a valid OpenCL buffer";
        case clblast::StatusCode::kInvalidMatrixB:
            return "Matrix B is not a valid OpenCL buffer";
        case clblast::StatusCode::kInvalidMatrixC:
            return "Matrix C is not a valid OpenCL buffer";
        case clblast::StatusCode::kInvalidVectorX:
            return "Vector X is not a valid OpenCL buffer";
        case clblast::StatusCode::kInvalidVectorY:
            return "Vector Y is not a valid OpenCL buffer";
        case clblast::StatusCode::kInvalidDimension:
            return "Dimensions M, N, and K have to be larger than zero";
        case clblast::StatusCode::kInvalidLeadDimA:
            return "LD of A is smaller than the matrix's first dimension";
        case clblast::StatusCode::kInvalidLeadDimB:
            return "LD of B is smaller than the matrix's first dimension";
        case clblast::StatusCode::kInvalidLeadDimC:
            return "LD of C is smaller than the matrix's first dimension";
        case clblast::StatusCode::kInvalidIncrementX:
            return "Increment of vector X cannot be zero";
        case clblast::StatusCode::kInvalidIncrementY:
            return "Increment of vector Y cannot be zero";
        case clblast::StatusCode::kInsufficientMemoryA:
            return "Matrix A's OpenCL buffer is too small";
        case clblast::StatusCode::kInsufficientMemoryB:
            return "Matrix B's OpenCL buffer is too small";
        case clblast::StatusCode::kInsufficientMemoryC:
            return "Matrix C's OpenCL buffer is too small";
        case clblast::StatusCode::kInsufficientMemoryX:
            return "Vector X's OpenCL buffer is too small";
        case clblast::StatusCode::kInsufficientMemoryY:
            return "Vector Y's OpenCL buffer is too small";

        // Custom additional status codes for CLBlast
        case clblast::StatusCode::kInsufficientMemoryTemp:
            return "Temporary buffer provided to GEMM routine is too small";
        case clblast::StatusCode::kInvalidBatchCount:
            return "The batch count needs to be positive";
        case clblast::StatusCode::kInvalidOverrideKernel:
            return "Trying to override parameters for an invalid kernel";
        case clblast::StatusCode::kMissingOverrideParameter:
            return "Missing override parameter(s) for the target kernel";
        case clblast::StatusCode::kInvalidLocalMemUsage:
            return "Not enough local memory available on this device";
        case clblast::StatusCode::kNoHalfPrecision:
            return "Half precision (16-bits) not supported by the device";
        case clblast::StatusCode::kNoDoublePrecision:
            return "Double precision (64-bits) not supported by the device";
        case clblast::StatusCode::kInvalidVectorScalar:
            return "The unit-sized vector is not a valid OpenCL buffer";
        case clblast::StatusCode::kInsufficientMemoryScalar:
            return "The unit-sized vector's OpenCL buffer is too small";
        case clblast::StatusCode::kDatabaseError:
            return "Entry for the device was not found in the database";
        case clblast::StatusCode::kUnknownError:
            return "A catch-all error code representing an unspecified error";
        case clblast::StatusCode::kUnexpectedError:
            return "A catch-all error code representing an unexpected "
                   "exception";
    }

    return "Unknown error";
}

static std::recursive_mutex gCLBlastMutex;

#define CLBLAST_CHECK(fn)                                            \
    do {                                                             \
        gCLBlastMutex.lock();                                        \
        clblast::StatusCode _clblast_st = fn;                        \
        gCLBlastMutex.unlock();                                      \
        if (_clblast_st != clblast::StatusCode::kSuccess) {          \
            char clblast_st_msg[1024];                               \
            snprintf(clblast_st_msg, sizeof(clblast_st_msg),         \
                     "CLBlast Error (%d): %s\n", (int)(_clblast_st), \
                     _clblastGetResultString(_clblast_st));          \
                                                                     \
            AF_ERROR(clblast_st_msg, AF_ERR_INTERNAL);               \
        }                                                            \
    } while (0)
